"""Shared pipeline helpers for the dual-lens BA scripts: keyframe
decoding, per-lens visual frontend, and multi-view triangulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader._camera_math import unit_rows
from specialist.loader.mei_model import MeiParams, mei_unproject
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.extrinsics import Extrinsics, LensOffset
from specialist.vi_ba.features import (
  Frame,
  chain_tracks,
  dis_flow_track,
  essential_ransac_inliers,
  parallel_map,
)
from specialist.vi_ba.init import triangulate_track_multiview

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "FrontendConfig",
  "capture_duration",
  "decode_dual_keyframes",
  "filter_tracks_by_parallax",
  "keyframe_timestamps",
  "lens_setups",
  "probe_capture",
  "run_lens_frontend",
  "scaled_mei",
  "triangulate_lens",
]


@dataclass(frozen=True)
class FrontendConfig:
  n_features: int = 2000
  ransac_threshold: float = 3.0
  max_half_angle: float = float(np.deg2rad(70.0))
  fb_error_threshold: float = 1.0
  min_distance: float = 8.0
  redetect_ratio: float = 0.5


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


def lens_setups(
  extrinsics: Extrinsics, decoded_width: int
) -> tuple[LensSetup, LensSetup]:
  """Per-lens MEI setups at the decode resolution: the ``offset_v3``
  intrinsics are authored in the camera's raw stitched frame and rescale by
  the ratio of the decoded stitched width (two square lenses) to the raw
  one (see ``scaled_mei``)."""
  front = LensSetup(
    extrinsic=extrinsics.body_T_front_lens,
    params=scaled_mei(extrinsics.front, 2 * decoded_width, lens_idx=0),
  )
  rear = LensSetup(
    extrinsic=extrinsics.body_T_rear_lens,
    params=scaled_mei(extrinsics.rear, 2 * decoded_width, lens_idx=1),
  )
  return front, rear


def resolve_keyframe_every(keyframe_every: int, fps: float) -> int:
  """Resolve the keyframe stride: an explicit positive value, else auto — round to
  a ~0.10s interval (6 at 60fps, 3 at 30fps). A frame-count default undersamples
  sub-60fps clips (9 frames is 0.15s at 60fps but 0.30s at 30fps), so auto keeps
  the time interval constant regardless of the recording rate."""
  return keyframe_every if keyframe_every > 0 else max(1, round(fps / 10))


# Segment fan-out for the parallel decode; short windows stay on one capture
# (a seek costs more than it saves on a small span).
_DECODE_MAX_SEGMENTS = 8
_DECODE_MIN_KEYFRAMES_PER_SEGMENT = 32


def _video_props(path: Path, *props: int) -> list[float]:
  """Container properties read via one short-lived ``VideoCapture``."""
  cap = cv2.VideoCapture(str(path))
  if not cap.isOpened():
    raise RuntimeError(f"OpenCV cannot open {path}")
  values = [float(cap.get(prop)) for prop in props]
  cap.release()
  return values


def probe_capture(video_path: Path) -> tuple[Path, float, int]:
  """Locate a capture's rear-lens file and read the container frame rate
  and native per-lens width.

  ``video_path`` is the square ``_00_`` master carrying the telemetry; its
  rear sibling swaps ``_00_`` for ``_10_`` in the filename. The frame rate
  comes from the container: the telemetry rounds the NTSC rate (30.0 for a
  30000/1001 stream), which would stretch keyframe timestamps against the
  IMU clock. Raises ``RuntimeError`` unless the video is square and its
  sibling exists."""
  raw_w, raw_h, fps = _video_props(
    video_path, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS
  )
  width, height = int(raw_w), int(raw_h)
  rear_path = video_path.with_name(video_path.name.replace("_00_", "_10_", 1))
  if width != height or rear_path == video_path or not rear_path.is_file():
    raise RuntimeError(
      f"{video_path.name} ({width}x{height}) is not a square per-lens master "
      f"with sibling {rear_path.name}"
    )
  return rear_path, fps, width


def _shared_frame_count(front_path: Path, rear_path: Path) -> int:
  """Frames present in both lens files; the camera may close one file of the
  pair a frame before the other."""
  return min(
    int(_video_props(path, cv2.CAP_PROP_FRAME_COUNT)[0])
    for path in (front_path, rear_path)
  )


def capture_duration(video_path: Path) -> float:
  """Seconds decodable from a capture, bounded by the shorter lens file."""
  rear_path, fps, _ = probe_capture(video_path)
  return _shared_frame_count(video_path, rear_path) / fps


def decode_dual_keyframes(
  video_path: Path,
  rear_path: Path,
  fps: float,
  start: float,
  duration: float | None,
  keyframe_every: int,
  keep_color: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[bytes], list[bytes]]:
  """Decode keyframes from the ``_00_``/``_10_`` pair into ``(front_gray,
  rear_gray)`` grayscale lists, which the frontend and tag detection consume,
  plus ``(front_color, rear_color)`` BGR frames for the Rerun overlay, at the
  native per-lens resolution. ``keep_color=False`` returns empty color lists:
  a full-length dual capture's BGR keyframes alone are tens of GB, so runs
  that never render frames must not hold them.

  The window splits into stride-aligned segments decoded concurrently, one
  seeking capture per segment; seeks land on the exact frame, so the output
  is identical to a single sequential pass. ``duration=None`` decodes until
  the file ends.
  """
  paths = [video_path, rear_path]
  container_frames = _shared_frame_count(video_path, rear_path)
  start_frame = int(round(start * fps))
  available = container_frames - start_frame
  if duration is None:
    total_frames = available
  else:
    total_frames = min(int(round(duration * fps)), available)

  def decode_span(
    span: tuple[Path, int, int],
  ) -> tuple[list[np.ndarray], list[bytes]]:
    path, begin, end = span
    cap = cv2.VideoCapture(str(path))
    if start_frame + begin > 0:
      cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + begin)
    gray_frames: list[np.ndarray] = []
    color_frames: list[bytes] = []
    for i in range(begin, end):
      # Non-keyframes still need decoding (inter-frame references) but not the
      # YUV->BGR convert/copy, so grab() them and only retrieve() the keyframes.
      if i % keyframe_every != 0:
        if not cap.grab():
          break
        continue
      ok, frame = cap.read()
      if not ok:
        break
      gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
      if keep_color:
        color_frames.append(
          cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
        )
    cap.release()
    return gray_frames, color_frames

  n_segments = min(
    _DECODE_MAX_SEGMENTS,
    max(1, total_frames // (_DECODE_MIN_KEYFRAMES_PER_SEGMENT * keyframe_every)),
  )
  # Split points floor onto the keyframe stride so every segment begins on a
  # keyframe.
  bounds = [
    round(total_frames * seg / n_segments) // keyframe_every * keyframe_every
    for seg in range(n_segments)
  ] + [total_frames]
  spans = [
    (path, begin, end)
    for path in paths
    for begin, end in zip(bounds, bounds[1:], strict=False)
  ]
  parts = parallel_map(decode_span, spans)
  per_file = [parts[f * n_segments : (f + 1) * n_segments] for f in range(len(paths))]
  # Hitting the end of the stream is only legitimate in a file's last segment;
  # an early stop anywhere else would silently misalign every later keyframe
  # with its timestamp. Both ends of every checked segment are stride-aligned,
  # so a full segment yields exactly (end - begin) / keyframe_every keyframes.
  for file_parts in per_file:
    for begin, end, part in zip(bounds, bounds[1:], file_parts[:-1], strict=False):
      if len(part[0]) != (end - begin) // keyframe_every:
        raise RuntimeError(f"decode stopped early inside segment [{begin}, {end})")
  gray = [
    [frame for part in file_parts for frame in part[0]] for file_parts in per_file
  ]
  color = [
    [frame for part in file_parts for frame in part[1]] for file_parts in per_file
  ]
  # The trailing keyframe can land past the shorter file's final decoded frame.
  n_keyframes = min(len(gray[0]), len(gray[1]))
  return (
    gray[0][:n_keyframes],
    gray[1][:n_keyframes],
    color[0][:n_keyframes],
    color[1][:n_keyframes],
  )


def keyframe_timestamps(
  start: float, fps: float, keyframe_every: int, n_keyframes: int
) -> Float64Array:
  """Timestamp (seconds in the `.insv`'s clock) of each decoded keyframe."""
  return (
    start + np.arange(n_keyframes, dtype=np.float64) * keyframe_every / fps
  ).astype(np.float64)


def run_lens_frontend(
  frames: list[np.ndarray],
  params: MeiParams,
  cfg: FrontendConfig,
  masks: list[np.ndarray] | None = None,
) -> tuple[list[Frame], dict[int, dict[int, Float64Array]]]:
  """DIS flow → essential-RANSAC → chain for one lens. ``masks[k]`` (optional,
  ``uint8`` 255=keep / 0=ignore) suppresses new detections and filters
  tracked points that drift into masked regions."""
  if masks is not None and len(masks) != len(frames):
    raise ValueError(f"masks length {len(masks)} != frames length {len(frames)}")
  feature_frames, matches_between = dis_flow_track(
    frames,
    n_features=cfg.n_features,
    masks=masks,
    fb_error_threshold=cfg.fb_error_threshold,
    min_distance=cfg.min_distance,
    redetect_ratio=cfg.redetect_ratio,
  )
  filtered: list[np.ndarray] = []
  for k in range(1, len(feature_frames)):
    m = matches_between[k - 1]
    if m.shape[0] == 0:
      filtered.append(np.empty((0, 2), dtype=np.int64))
      continue
    a, b = feature_frames[k - 1], feature_frames[k]
    pts_a = a.keypoints[m[:, 0]]
    pts_b = b.keypoints[m[:, 1]]
    bears_a = mei_unproject(pts_a, params)
    bears_b = mei_unproject(pts_b, params)
    inliers = essential_ransac_inliers(
      pts_a,
      pts_b,
      bears_a,
      bears_b,
      threshold_px=cfg.ransac_threshold,
      fx=params.fx,
      max_half_angle=cfg.max_half_angle,
    )
    filtered.append(m[inliers])
  tracks = chain_tracks(feature_frames, filtered)
  n_kp = sum(int(len(f.keypoints)) for f in feature_frames)
  n_match = sum(int(m.shape[0]) for m in matches_between)
  n_in = sum(int(f.shape[0]) for f in filtered)
  print(
    f"  [frontend funnel] keypoints={n_kp} matched={n_match} "
    f"ransac_inliers={n_in} chained_tracks={len(tracks)}"
  )
  return feature_frames, tracks


def filter_tracks_by_parallax(
  tracks: dict[int, dict[int, Float64Array]],
  params: MeiParams,
  min_parallax: float,
) -> tuple[dict[int, dict[int, Float64Array]], dict[int, dict[int, Float64Array]]]:
  """Split tracks into ``(kept, rejected)`` by max bearing-angle change.
  Body-attached features (gripper, camera mount) are stationary in image
  space and get rejected; real-world features parallax and survive."""
  if min_parallax < 0.0:
    raise ValueError(f"min_parallax must be ≥ 0, got {min_parallax}")
  cos_min = float(np.cos(min_parallax))
  kept: dict[int, dict[int, Float64Array]] = {}
  rejected: dict[int, dict[int, Float64Array]] = {}
  for tid, obs in tracks.items():
    if len(obs) < 2:
      rejected[tid] = obs
      continue
    pixels = np.stack([obs[k] for k in sorted(obs.keys())])
    bearings = unit_rows(mei_unproject(pixels, params))
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
) -> dict[int, Float64Array]:
  """Triangulate every track by geometric closest-point across all observations.
  ``min_landmark_depth`` / ``max_landmark_depth`` gate the median-view
  depth — callers triangulating against unscaled visual-frame poses
  should widen these bounds."""
  out: dict[int, Float64Array] = {}
  for tid, obs in tracks.items():
    if len(obs) < min_track_length:
      continue
    kfs = sorted(obs.keys())
    pixels = np.stack([obs[k] for k in kfs])
    bearings = list(mei_unproject(pixels, params))
    poses = [body_poses[k] for k in kfs]
    xyz = triangulate_track_multiview(
      bearings,
      poses,
      extrinsic,
      min_landmark_depth=min_landmark_depth,
      max_landmark_depth=max_landmark_depth,
    )
    if xyz is not None:
      out[tid] = xyz
  return out
