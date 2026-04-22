"""Shared pipeline helpers for the dual-lens BA scripts: keyframe
decoding, per-lens visual frontend, multi-view triangulation, and
dual-lens consistency check."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import MeiParams, mei_project, mei_unproject
from specialist.loader.parse_insv import InsvData
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.extrinsics import Extrinsics, LensOffset
from specialist.vi_ba.features import (
  Frame,
  chain_tracks,
  dis_flow_track,
  essential_ransac_inliers,
  klt_track,
)
from specialist.vi_ba.init import triangulate_track_multiview

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "FrontendConfig",
  "decode_dual_keyframes",
  "dual_lens_consistency",
  "filter_tracks_by_parallax",
  "keyframe_timestamps",
  "lens_setups",
  "run_lens_frontend",
  "scaled_mei",
  "triangulate_lens",
]


@dataclass(frozen=True)
class FrontendConfig:
  n_features: int = 2000
  ransac_threshold_px: float = 3.0
  max_half_angle_deg: float = 70.0
  klt_window_size: int = 21
  klt_max_pyramid_level: int = 3
  klt_fb_error_threshold_px: float = 1.0
  klt_min_distance_px: float = 8.0
  klt_redetect_ratio: float = 0.5
  use_dis_flow: bool = True


def scaled_mei(offset: LensOffset, decoded_full_width: int, lens_idx: int) -> MeiParams:
  """Rescale raw-sensor MEI intrinsics into decoded per-lens pixel coords.

  ``lens_idx = 0`` selects the front (left half), ``1`` the rear (right
  half). The rear's raw ``cx`` is measured from the stitched-frame left
  edge, so we subtract the raw half-width before scaling.
  """
  scale = decoded_full_width / offset.frame_width
  half_raw = offset.frame_width / 2.0
  return MeiParams(
    fx=offset.fx * scale,
    fy=offset.fy * scale,
    cx=(offset.cx - half_raw * lens_idx) * scale,
    cy=offset.cy * scale,
    k1=offset.k1,
    k2=offset.k2,
    k3=offset.k3,
    p1=offset.p1,
    p2=offset.p2,
    xi=offset.xi,
  )


def lens_setups(data: InsvData, extrinsics: Extrinsics) -> tuple[LensSetup, LensSetup]:
  front = LensSetup(
    extrinsic=extrinsics.front_T_body_lens,
    params=scaled_mei(extrinsics.front, data.frame_width, lens_idx=0),
  )
  rear = LensSetup(
    extrinsic=extrinsics.rear_T_body_lens,
    params=scaled_mei(extrinsics.rear, data.frame_width, lens_idx=1),
  )
  return front, rear


def resolve_keyframe_every(keyframe_every: int, fps: float) -> int:
  """Resolve the keyframe stride: an explicit positive value, else auto — round to
  a ~0.10s interval (6 at 60fps, 3 at 30fps). A frame-count default undersamples
  sub-60fps clips (9 frames is 0.15s at 60fps but 0.30s at 30fps), so auto keeps
  the time interval constant regardless of the recording rate."""
  return keyframe_every if keyframe_every > 0 else max(1, round(fps / 10))


def decode_dual_keyframes(
  video_path: Path,
  start: float,
  duration: float | None,
  fps: float,
  keyframe_every: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
  """Decode side-by-side ``.insv`` frames into ``(front, rear)`` grayscale lists.

  ``duration=None`` decodes until the file ends.
  """
  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    raise RuntimeError(f"OpenCV cannot open {video_path}")
  start_frame = int(round(start * fps))
  if duration is None:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame
  else:
    total_frames = int(round(duration * fps))
  cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  half = width // 2
  front: list[np.ndarray] = []
  rear: list[np.ndarray] = []
  for i in range(total_frames):
    ok, frame = cap.read()
    if not ok:
      break
    if i % keyframe_every == 0:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      front.append(gray[:, :half])
      rear.append(gray[:, half:])
  cap.release()
  return front, rear


def keyframe_timestamps(
  start: float, fps: float, keyframe_every: int, n_frames: int
) -> Float64Array:
  """Timestamp (seconds in the `.insv`'s clock) of each decoded keyframe."""
  return (start + np.arange(n_frames, dtype=np.float64) * keyframe_every / fps).astype(
    np.float64
  )


def run_lens_frontend(
  frames: list[np.ndarray],
  params: MeiParams,
  cfg: FrontendConfig,
  masks: list[np.ndarray] | None = None,
  rotation_predictor: "Callable[[int, Float64Array], Float64Array] | None" = None,
) -> tuple[list[Frame], dict[int, dict[int, Float64Array]]]:
  """KLT → essential-RANSAC → chain for one lens. ``masks[k]`` (optional,
  ``uint8`` 255=keep / 0=ignore) suppresses new detections and filters
  tracked points that drift into masked regions. ``rotation_predictor``
  (optional) seeds KLT with IMU-rotated pixel positions for large-motion
  robustness; passed straight through to ``klt_track``."""
  if masks is not None and len(masks) != len(frames):
    raise ValueError(f"masks length {len(masks)} != frames length {len(frames)}")
  if cfg.use_dis_flow:
    orb_frames, matches_between = dis_flow_track(
      frames,
      n_features=cfg.n_features,
      masks=masks,
      fb_error_threshold_px=cfg.klt_fb_error_threshold_px,
      min_distance_px=cfg.klt_min_distance_px,
      redetect_ratio=cfg.klt_redetect_ratio,
    )
  else:
    orb_frames, matches_between = klt_track(
      frames,
      n_features=cfg.n_features,
      masks=masks,
      window_size=cfg.klt_window_size,
      max_pyramid_level=cfg.klt_max_pyramid_level,
      fb_error_threshold_px=cfg.klt_fb_error_threshold_px,
      min_distance_px=cfg.klt_min_distance_px,
      redetect_ratio=cfg.klt_redetect_ratio,
      rotation_predictor=rotation_predictor,
    )
  max_ha = float(np.deg2rad(cfg.max_half_angle_deg))
  filtered: list[np.ndarray] = []
  for k in range(1, len(orb_frames)):
    m = matches_between[k - 1]
    if m.shape[0] == 0:
      filtered.append(np.empty((0, 2), dtype=np.int64))
      continue
    a, b = orb_frames[k - 1], orb_frames[k]
    pts_a = a.keypoints[m[:, 0]]
    pts_b = b.keypoints[m[:, 1]]
    bears_a = mei_unproject(pts_a, params)
    bears_b = mei_unproject(pts_b, params)
    inliers = essential_ransac_inliers(
      pts_a,
      pts_b,
      bears_a,
      bears_b,
      threshold_px=cfg.ransac_threshold_px,
      fx=params.fx,
      max_half_angle=max_ha,
    )
    filtered.append(m[inliers])
  tracks = chain_tracks(orb_frames, filtered)
  return orb_frames, tracks


def filter_tracks_by_parallax(
  tracks: dict[int, dict[int, Float64Array]],
  params: MeiParams,
  min_parallax_deg: float,
) -> tuple[dict[int, dict[int, Float64Array]], dict[int, dict[int, Float64Array]]]:
  """Split tracks into ``(kept, rejected)`` by max bearing-angle change.
  Body-attached features (gripper, camera mount) are stationary in image
  space and get rejected; real-world features parallax and survive."""
  if min_parallax_deg < 0.0:
    raise ValueError(f"min_parallax_deg must be ≥ 0, got {min_parallax_deg}")
  cos_min = float(np.cos(np.radians(min_parallax_deg)))
  kept: dict[int, dict[int, Float64Array]] = {}
  rejected: dict[int, dict[int, Float64Array]] = {}
  for tid, obs in tracks.items():
    if len(obs) < 2:
      rejected[tid] = obs
      continue
    pixels = np.stack([obs[k] for k in sorted(obs.keys())], axis=0)
    bearings = mei_unproject(pixels, params)
    norms = np.linalg.norm(bearings, axis=1, keepdims=True)
    bearings = bearings / np.where(norms == 0.0, 1.0, norms)
    dots = np.clip(bearings @ bearings.T, -1.0, 1.0)
    if float(dots.min()) <= cos_min:
      kept[tid] = obs
    else:
      rejected[tid] = obs
  return kept, rejected


def triangulate_lens(
  tracks: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  extrinsic: gtsam.Pose3,
  params: MeiParams,
  min_track_length: int,
  min_landmark_depth: float = 0.15,
  max_landmark_depth: float = 5.0,
  rejection_counter: dict[str, int] | None = None,
) -> dict[int, Float64Array]:
  """Triangulate every track via multi-view DLT across all observations.
  ``min_landmark_depth`` / ``max_landmark_depth`` gate the median-view
  depth — callers triangulating against unscaled visual-frame poses
  should widen these bounds."""
  out: dict[int, Float64Array] = {}
  too_short = 0
  for tid, obs in tracks.items():
    if len(obs) < min_track_length:
      too_short += 1
      continue
    fs = sorted(obs.keys())
    pixels = np.stack([obs[k] for k in fs], axis=0)
    bearings = list(mei_unproject(pixels, params))
    poses = [body_poses[k] for k in fs]
    xyz = triangulate_track_multiview(
      bearings,
      poses,
      extrinsic,
      min_landmark_depth=min_landmark_depth,
      max_landmark_depth=max_landmark_depth,
      rejection_counter=rejection_counter,
    )
    if xyz is not None:
      out[tid] = xyz
  if rejection_counter is not None and too_short > 0:
    rejection_counter["too_short"] = rejection_counter.get("too_short", 0) + too_short
  return out


def dual_lens_consistency(
  front_points: dict[int, Float64Array],
  rear_points: dict[int, Float64Array],
  body_poses: list[gtsam.Pose3],
  extrinsics: Extrinsics,
  front_params: MeiParams,
  rear_params: MeiParams,
  max_check: int = 40,
) -> tuple[float, int]:
  """Project shared-world landmarks through both lenses at kf 0; return
  mean pixel disagreement and count of matched pairs."""
  pose0 = body_poses[0]
  front_arr = np.asarray(list(front_points.values()), dtype=np.float64)
  rear_arr = np.asarray(list(rear_points.values()), dtype=np.float64)
  if front_arr.shape[0] == 0 or rear_arr.shape[0] == 0:
    return float("nan"), 0
  diffs: list[float] = []
  checked = 0
  for xyz in front_arr:
    if checked >= max_check:
      break
    d = np.linalg.norm(rear_arr - xyz, axis=1)
    if d.size == 0 or d.min() > 0.05:
      continue
    matched = rear_arr[int(d.argmin())]
    mid = 0.5 * (xyz + matched)
    lens_f = pose0.compose(extrinsics.front_T_body_lens).transformTo(
      gtsam.Point3(float(mid[0]), float(mid[1]), float(mid[2]))
    )
    lens_r = pose0.compose(extrinsics.rear_T_body_lens).transformTo(
      gtsam.Point3(float(mid[0]), float(mid[1]), float(mid[2]))
    )
    pred_f = mei_project(np.asarray(lens_f)[None, :], front_params)[0]
    pred_r = mei_project(np.asarray(lens_r)[None, :], rear_params)[0]
    if np.all(np.isfinite(pred_f)) and np.all(np.isfinite(pred_r)):
      diffs.append(float(np.linalg.norm(pred_f - pred_r)))
      checked += 1
  return (float(np.mean(diffs)) if diffs else float("nan")), checked
