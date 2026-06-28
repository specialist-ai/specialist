"""Per-tracker prep for VI-BA: decode → frontend → tag detect → anchor PnP →
IMU stationary gravity → VI alignment → anchor-init pose seed → metric
re-triangulation. Returns a ``PreparedTracker`` ready for ``track.py`` to
compose into a joint factor graph.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import MeiParams, mei_project, mei_unproject
from specialist.loader.parse_insv import load_insv
from specialist.vi_ba.anchor_detection import (
  TagDetection,
  anchor_pose_from_tags,
  detect_all_tags,
  estimate_camera_to_tag_pose,
)
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.configs import (
  AnchorCfg,
  FrontendCfg,
  GripperCfg,
  ImuCfg,
)
from specialist.vi_ba.extrinsics import parse_insv_extrinsics
from specialist.vi_ba.features import detect_orb, match_pair
from specialist.vi_ba.imu import (
  ImuSamples,
  factory_imu_bias,
  find_stationary_windows,
  make_preintegration_params,
  preintegrate_between_keyframes,
)
from specialist.vi_ba.init import (
  Alignment,
  apply_alignment_to_poses,
  body_poses_from_camera_poses,
  incremental_sfm,
  interpolate_camera_poses,
  pick_init_pair,
  two_view_essential,
  visual_inertial_align,
)
from specialist.vi_ba.pipeline import (
  FrontendConfig,
  decode_dual_keyframes,
  filter_tracks_by_parallax,
  keyframe_timestamps,
  lens_setups,
  resolve_keyframe_every,
  run_lens_frontend,
  triangulate_lens,
)

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "PreparedTracker",
  "TrackerArgs",
  "prepare_tracker",
]


@dataclass
class TrackerArgs:
  """Per-tracker prep knobs. Composes nested configs from ``configs.py``."""

  insv_path: Path
  start: float = 0.0
  duration: float | None = None
  keyframe_every: int = 0  # 0 = auto (round(fps/10)); see WindowCfg.keyframe_every
  base_tag_id: int = 6
  # 0 disables the corresponding URDF revolute joint angle recovery.
  left_finger_tag_id: int = 0
  right_finger_tag_id: int = 0
  frontend: FrontendCfg = field(default_factory=FrontendCfg)
  imu: ImuCfg = field(default_factory=ImuCfg)
  anchor: AnchorCfg = field(default_factory=AnchorCfg)
  gripper: GripperCfg = field(default_factory=GripperCfg)


@dataclass
class PreparedTracker:
  """Everything one tracker contributes to the joint graph + viz."""

  name: str
  body_poses: list[gtsam.Pose3]
  velocities: list[Float64Array]
  biases: list[gtsam.imuBias.ConstantBias]
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements]
  front_lens: LensSetup
  rear_lens: LensSetup
  front_tracks_tri: dict[int, dict[int, Float64Array]]
  rear_tracks_tri: dict[int, dict[int, Float64Array]]
  front_points: dict[int, Float64Array]
  rear_points: dict[int, Float64Array]
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]]
  anchor_pose_world_local: gtsam.Pose3 | None
  body_T_base_tag: gtsam.Pose3 | None
  # URDF joint name → per-keyframe rotation about the URDF z-axis. NaN where
  # either tag is missing or PnP fails. Empty when no finger ids configured.
  joint_angles: dict[str, list[float]]
  kf_times: Float64Array
  front_frames: list[np.ndarray]
  rear_frames: list[np.ndarray]
  front_keypoints: list[Float64Array]
  rear_keypoints: list[Float64Array]
  front_tags_per_kf: list[list[TagDetection]]
  rear_tags_per_kf: list[list[TagDetection]]
  front_rejected: dict[int, dict[int, Float64Array]]
  rear_rejected: dict[int, dict[int, Float64Array]]
  rear_static_mask_rects: list[tuple[int, int, int, int]]
  body_T_left_finger_per_kf: dict[int, gtsam.Pose3]
  body_T_right_finger_per_kf: dict[int, gtsam.Pose3]
  left_finger_tag_id: int
  right_finger_tag_id: int
  n_frames: int

  def vi_graph_inputs(self) -> dict[str, object]:
    """Bundle this tracker's state + lens data for ``build_vi_graph``."""
    return {
      "body_poses": self.body_poses,
      "velocities": self.velocities,
      "biases": self.biases,
      "preintegrated": self.preintegrated,
      "front": self.front_lens,
      "front_tracks": self.front_tracks_tri,
      "front_points": self.front_points,
      "rear": self.rear_lens,
      "rear_tracks": self.rear_tracks_tri,
      "rear_points": self.rear_points,
    }


def _build_non_anchor_tag_masks(
  frames: list[np.ndarray],
  tags_per_kf: list[list[TagDetection]],
  exclude_tag_id: int,
  pad_px: int,
) -> list[np.ndarray]:
  masks: list[np.ndarray] = []
  for frame, tags in zip(frames, tags_per_kf, strict=True):
    h, w = frame.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for tag in tags:
      if exclude_tag_id >= 0 and int(tag.tag_id) == exclude_tag_id:
        continue
      corners = np.asarray(tag.corners, dtype=np.float64)
      x0 = max(int(np.floor(corners[:, 0].min())) - pad_px, 0)
      y0 = max(int(np.floor(corners[:, 1].min())) - pad_px, 0)
      x1 = min(int(np.ceil(corners[:, 0].max())) + pad_px, w)
      y1 = min(int(np.ceil(corners[:, 1].max())) + pad_px, h)
      if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 0
    masks.append(mask)
  return masks


def _apply_static_mask_rects(
  masks: list[np.ndarray], rects: list[tuple[int, int, int, int]]
) -> list[np.ndarray]:
  if not rects or not masks:
    return masks
  h, w = masks[0].shape[:2]
  for x1, y1, x2, y2 in rects:
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
      continue
    for m in masks:
      m[y1c:y2c, x1c:x2c] = 0
  return masks


def _per_kf_cam_T_tag(
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  tag_id: int,
  tag_size: float,
  n_frames: int,
) -> list[gtsam.Pose3 | None]:
  """Per-keyframe ``cam_T_tag`` for one tag id, with IPPE's reflectional
  ambiguity disambiguated temporally via the previous frame's pose."""
  out: list[gtsam.Pose3 | None] = [None] * n_frames
  prev: gtsam.Pose3 | None = None
  for kf in range(min(n_frames, len(tags_per_kf))):
    corners = next(
      (d.corners for d in tags_per_kf[kf] if int(d.tag_id) == tag_id), None
    )
    if corners is None:
      continue
    p = estimate_camera_to_tag_pose(corners, tag_size, lens.params, prev_cam_T_tag=prev)
    if p is None:
      continue
    out[kf] = p
    prev = p
  return out


def _mean_pose(poses: list[gtsam.Pose3]) -> gtsam.Pose3:
  """Mean of a list of poses: arithmetic mean of translations + log-mean
  rotation about the first sample. Used for a fixed body-rigid tag where
  every observation should agree up to PnP noise."""
  ref = poses[0]
  translations = np.stack(
    [np.asarray(p.translation(), dtype=np.float64) for p in poses], axis=0
  )
  log_sum = np.zeros(3, dtype=np.float64)
  for p in poses:
    log_sum += gtsam.Rot3.Logmap(ref.rotation().between(p.rotation()))
  delta = gtsam.Rot3.Expmap(log_sum / len(poses))
  return gtsam.Pose3(
    ref.rotation().compose(delta), gtsam.Point3(*translations.mean(axis=0))
  )


def _finger_joint_angles_per_kf(
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  base_tag_id: int,
  finger_tag_id: int,
  tag_size: float,
  n_frames: int,
) -> list[float]:
  """``Logmap(base_tag_T_finger)[2]`` per keyframe — the signed rotation about the
  URDF revolute-joint axis (``+z`` by URDF convention). The base tag is
  rigidly mounted to the body so its ``cam_T_tag`` is averaged once over all
  visible keyframes; the finger tag's ``cam_T_tag`` is recovered per kf."""
  cam_T_base_per_kf = _per_kf_cam_T_tag(
    tags_per_kf, lens, base_tag_id, tag_size, n_frames
  )
  cam_T_finger_per_kf = _per_kf_cam_T_tag(
    tags_per_kf, lens, finger_tag_id, tag_size, n_frames
  )
  observed_base = [p for p in cam_T_base_per_kf if p is not None]
  if not observed_base:
    return [float("nan")] * n_frames
  cam_T_base = _mean_pose(observed_base)
  out: list[float] = [float("nan")] * n_frames
  for kf in range(n_frames):
    cam_T_finger = cam_T_finger_per_kf[kf]
    if cam_T_finger is None:
      continue
    base_tag_T_finger = cam_T_base.between(cam_T_finger)
    out[kf] = float(gtsam.Rot3.Logmap(base_tag_T_finger.rotation())[2])
  return out


def _gyro_integrated_rotations(
  imu_t: Float64Array, gyro_body: Float64Array, kf_times: Float64Array
) -> list[Float64Array]:
  """Per-consecutive-kf body rotation matrix from raw gyro integration.
  ``gyro_body`` is the bias-uncorrected angular velocity in the body
  frame (rad/s). Each output is ``R_body_kf-to-body_kf+1`` so a vector
  fixed in the world satisfies ``v_body_k+1 = R.T @ v_body_k``."""
  out: list[Float64Array] = []
  for k in range(len(kf_times) - 1):
    mask = (imu_t > kf_times[k]) & (imu_t <= kf_times[k + 1])
    if not mask.any():
      out.append(np.eye(3, dtype=np.float64))
      continue
    times = imu_t[mask]
    gyros = gyro_body[mask]
    R = np.eye(3, dtype=np.float64)
    if len(times) < 2:
      out.append(R)
      continue
    dts = np.diff(times)
    for i in range(len(dts)):
      omega = gyros[i]
      dt = float(dts[i])
      angle = float(np.linalg.norm(omega) * dt)
      if angle > 1e-9:
        axis = omega * dt / angle
        K = np.array(
          [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
          ],
          dtype=np.float64,
        )
        R_step = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        R = R @ R_step
    out.append(R)
  return out


def _make_imu_klt_predictor(
  lens_params: MeiParams,
  R_body_lens: Float64Array,
  body_rotations_between: list[Float64Array],
) -> Callable[[int, Float64Array], Float64Array]:
  """Predict ``frame_k`` pixel positions from ``frame_{k-1}`` pixels by
  rotating the bearings through the IMU-integrated body rotation between
  the two kfs and reprojecting through the same lens. Used as
  ``rotation_predictor`` for ``klt_track``."""
  R_lens_body = R_body_lens.T

  def predictor(k: int, prev_pixels: Float64Array) -> Float64Array:
    if k - 1 >= len(body_rotations_between):
      return np.full_like(prev_pixels, np.nan)
    R_b = body_rotations_between[k - 1]
    R_lens_change = R_lens_body @ R_b.T @ R_body_lens
    bearings = mei_unproject(prev_pixels.astype(np.float64), lens_params)
    bearings_new = bearings @ R_lens_change.T
    return mei_project(bearings_new, lens_params)

  return predictor


def _bearings_by_track(
  tracks: dict[int, dict[int, Float64Array]], params: MeiParams
) -> dict[int, dict[int, Float64Array]]:
  """``{tid: {kf: unit bearing}}`` for the visual-odometry front-end."""
  out: dict[int, dict[int, Float64Array]] = {}
  for tid, obs in tracks.items():
    kfs = sorted(obs)
    b = mei_unproject(np.asarray([obs[k] for k in kfs], dtype=np.float64), params)
    nrm = np.linalg.norm(b, axis=1, keepdims=True)
    b = b / np.where(nrm == 0.0, 1.0, nrm)
    out[tid] = {k: b[i] for i, k in enumerate(kfs)}
  return out


def _subsample_obs(
  track: dict[int, Float64Array], max_obs: int
) -> dict[int, Float64Array]:
  """Keep at most ``max_obs`` observations (endpoints + evenly-spaced middle),
  preserving the track's baseline. Trims reprojection-factor count for the BA."""
  kfs = sorted(track)
  if max_obs <= 0 or len(kfs) <= max_obs:
    return track
  idx = np.unique(np.linspace(0, len(kfs) - 1, max_obs).round().astype(int))
  return {kfs[i]: track[kfs[i]] for i in idx}


def _cap_landmarks(
  points: dict[int, Float64Array],
  tracks: dict[int, dict[int, Float64Array]],
  cap: int,
) -> dict[int, Float64Array]:
  """Keep at most ``cap`` landmarks, preferring the longest tracks (most
  observations → best-constrained). No-op when already under the cap."""
  if cap <= 0 or len(points) <= cap:
    return points
  keep = set(sorted(points, key=lambda t: -len(tracks.get(t, {})))[:cap])
  return {t: p for t, p in points.items() if t in keep}


def _loop_closure_tracks(
  frames: list[np.ndarray],
  masks: list[np.ndarray],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  n_features: int,
  base_track_id: int,
  max_center_dist: float,
  min_kf_gap: int,
  n_neighbors: int = 4,
  ratio: float = 0.7,
  ransac_px: float = 3.0,
  min_views: int = 3,
  min_inliers: int = 20,
) -> dict[int, dict[int, Float64Array]]:
  """Recover tracks broken by fast lifts via ORB loop closure.

  Fast repositioning breaks the chained optical-flow tracks, so a physical point
  seen before and after a lift becomes two disconnected tracks and nothing ties
  the revisited poses together — the visual odometry drifts across the lift. Here
  ORB features are re-matched between spatially-close, non-adjacent keyframes
  (revisits of the writing workspace), geometrically verified, and union-found
  into multi-view tracks spanning the lifts. Triangulated against the (drifted)
  poses these become wide-baseline landmarks whose reprojection inconsistency
  drives the bundle to pull the revisits back together, cancelling the drift."""
  orb = [detect_orb(f, n_features, 20, m) for f, m in zip(frames, masks, strict=True)]
  centers = np.asarray(
    [
      np.asarray(p.compose(lens.extrinsic).translation(), dtype=np.float64)
      for p in body_poses
    ]
  )
  n = len(frames)
  parent: dict[tuple[int, int], tuple[int, int]] = {}

  def find(x: tuple[int, int]) -> tuple[int, int]:
    parent.setdefault(x, x)
    root = x
    while parent[root] != root:
      root = parent[root]
    while parent[x] != root:
      parent[x], x = root, parent[x]
    return root

  for i in range(n):
    if len(orb[i].keypoints) == 0:
      continue
    dist = np.linalg.norm(centers - centers[i], axis=1)
    cand = [
      int(j)
      for j in np.argsort(dist)
      if j > i and (j - i) > min_kf_gap and dist[j] <= max_center_dist
    ][:n_neighbors]
    for j in cand:
      m = match_pair(orb[i], orb[j], ratio)
      if len(m) < min_inliers:
        continue
      pa = orb[i].keypoints[m[:, 0]]
      pb = orb[j].keypoints[m[:, 1]]
      # Revisits are near-identical small-baseline views, so the essential matrix
      # is degenerate and lets false matches through (which wreck the bundle).
      # A homography is the right model for the small inter-view change and
      # rejects the inconsistent matches.
      _, mask = cv2.findHomography(pa, pb, cv2.RANSAC, ransac_px)
      if mask is None:
        continue
      inl = mask.ravel().astype(bool)
      if int(inl.sum()) < min_inliers:
        continue
      for k in np.where(inl)[0]:
        ra = find((i, int(m[k, 0])))
        rb = find((j, int(m[k, 1])))
        if ra != rb:
          parent[ra] = rb

  comps: dict[tuple[int, int], dict[int, int]] = {}
  for node in list(parent):
    comps.setdefault(find(node), {})[node[0]] = node[1]
  tracks: dict[int, dict[int, Float64Array]] = {}
  tid = base_track_id + 1
  for by_kf in comps.values():
    if len(by_kf) < min_views:
      continue
    # Store pixel coordinates (like the optical-flow tracks): downstream
    # triangulation and reprojection unproject through the MEI model themselves.
    tracks[tid] = {
      kf: np.asarray(orb[kf].keypoints[kp], dtype=np.float64)
      for kf, kp in by_kf.items()
    }
    tid += 1
  return tracks


def _add_loop_landmarks(
  front_points: dict[int, Float64Array],
  front_kept: dict[int, dict[int, Float64Array]],
  loop_tracks: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  min_track_length: int,
  max_landmark_depth: float,
  name: str,
  label: str,
  anchor_kfs: set[int] | None = None,
) -> dict[int, Float64Array]:
  """Triangulate revisit ``loop_tracks`` and merge them into the front-lens
  landmarks (``front_points``) and tracks (``front_kept``), so they enter the
  bundle as wide-baseline reprojection factors that tie the revisited poses
  together and cancel the drift. Mutates ``front_points`` and ``front_kept``.

  When ``anchor_kfs`` is given, each landmark is triangulated from its
  anchor-covisible (well-localized) views ONLY, while the FULL track — including
  the drifted anchor-free views — still enters ``front_kept`` as reprojection
  factors. This pulls the drifted poses onto structure fixed by the reliable
  early trajectory (real drift cancellation); triangulating from all views would
  instead fit the drift and leave no corrective residual."""
  if anchor_kfs:
    tri_tracks = {
      t: {k: v for k, v in obs.items() if k in anchor_kfs}
      for t, obs in loop_tracks.items()
    }
  else:
    tri_tracks = loop_tracks
  loop_pts = triangulate_lens(
    tri_tracks,
    body_poses,
    lens.extrinsic,
    lens.params,
    min_track_length,
    max_landmark_depth=max_landmark_depth,
  )
  front_points.update(loop_pts)
  front_kept.update(loop_tracks)
  print(
    f"  [{name}] {label}: +{len(loop_pts)} landmarks "
    f"from {len(loop_tracks)} revisit tracks"
  )
  return loop_pts


def _reject_vo_pose_outliers(
  cam_poses: dict[int, gtsam.Pose3], reject_factor: float = 8.0
) -> tuple[dict[int, gtsam.Pose3], list[int]]:
  """Drop incremental-SfM keyframes whose camera center is a local spike.

  Pure monocular SfM has no global consistency, so a degenerate PnP (few
  co-visible landmarks during fast rotation) can teleport a single keyframe far
  off the smooth path. Such a spike corrupts the visual displacements that the
  IMU scale alignment consumes, collapsing the recovered scale. Each interior
  registered keyframe is compared against the linear interpolation of its
  immediate registered neighbors; a residual exceeding ``reject_factor`` times
  the median per-keyframe step marks a spike. Spikes are removed iteratively
  (worst first) so ``_fill_camera_poses`` interpolates cleanly across them.

  ``reject_factor`` is deliberately loose (8×): a genuine real-motion burst can
  reach ~7× the slow-motion median step, and rejecting those over-interpolates
  the rotation and wrecks the recovered orientation. Only the far-larger PnP
  teleports should be cut."""
  kept = dict(cam_poses)
  removed: list[int] = []
  for _ in range(len(cam_poses)):
    ks = sorted(kept)
    if len(ks) < 5:
      break
    c = {k: np.asarray(kept[k].translation(), dtype=np.float64) for k in ks}
    steps = [
      float(np.linalg.norm(c[ks[i + 1]] - c[ks[i]])) / (ks[i + 1] - ks[i])
      for i in range(len(ks) - 1)
    ]
    median_step = float(np.median(steps))
    if median_step < 1e-12:
      break
    worst_k: int | None = None
    worst_r = 0.0
    for i in range(1, len(ks) - 1):
      kp, k, kn = ks[i - 1], ks[i], ks[i + 1]
      f = (k - kp) / (kn - kp)
      pred = (1.0 - f) * c[kp] + f * c[kn]
      r = float(np.linalg.norm(c[k] - pred))
      if r > worst_r:
        worst_r, worst_k = r, k
    if worst_k is not None and worst_r > reject_factor * median_step:
      del kept[worst_k]
      removed.append(worst_k)
    else:
      break
  return kept, sorted(removed)


def _fill_camera_poses(
  cam_poses: dict[int, gtsam.Pose3], n_frames: int
) -> list[gtsam.Pose3]:
  """Expand SfM-registered camera poses to a full per-keyframe list, filling
  unregistered keyframes by SLERP+lerp between nearest neighbors (held at the
  ends). The joint BA refines these from IMU + reprojection."""
  reg = sorted(cam_poses)
  out: list[gtsam.Pose3] = []
  for k in range(n_frames):
    if k in cam_poses:
      out.append(cam_poses[k])
      continue
    lo = max((r for r in reg if r < k), default=None)
    hi = min((r for r in reg if r > k), default=None)
    if lo is None:
      out.append(cam_poses[hi])  # type: ignore[index]
    elif hi is None:
      out.append(cam_poses[lo])
    else:
      pa, pb = cam_poses[lo], cam_poses[hi]
      f = (k - lo) / (hi - lo)
      out.append(pa.compose(gtsam.Pose3.Expmap(f * gtsam.Pose3.Logmap(pa.between(pb)))))
  return out


def prepare_tracker(name: str, args: TrackerArgs) -> PreparedTracker:
  """Run every per-tracker stage. Returns a ``PreparedTracker`` ready for
  joint-graph composition. Body poses are in this tracker's own gravity-
  aligned visual world; the caller aligns trackers to a shared frame via
  ``anchor_pose_world_local``."""
  data = load_insv(args.insv_path)
  extrinsics = parse_insv_extrinsics(data)
  front_lens, rear_lens = lens_setups(data, extrinsics)

  keyframe_every = resolve_keyframe_every(args.keyframe_every, data.frame_rate)
  front_frames, rear_frames = decode_dual_keyframes(
    args.insv_path, args.start, args.duration, data.frame_rate, keyframe_every
  )
  n_frames = len(front_frames)
  if n_frames < 3:
    raise RuntimeError(f"only {n_frames} keyframes decoded — window too short")
  kf_times = keyframe_timestamps(args.start, data.frame_rate, keyframe_every, n_frames)

  front_tags_per_kf = detect_all_tags(front_frames)
  rear_tags_per_kf = detect_all_tags(rear_frames)

  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]] = {}
  if args.anchor.tag_id >= 0:
    for side, lens, tags in (
      ("front", front_lens, front_tags_per_kf),
      ("rear", rear_lens, rear_tags_per_kf),
    ):
      for kf, pose in anchor_pose_from_tags(
        tags, args.anchor.tag_id, args.anchor.tag_size, lens.params
      ).items():
        anchor_obs_per_kf.setdefault(kf, {})[side] = pose

  if args.anchor.blackout_start is not None and args.anchor.blackout_end is not None:
    blackout = {
      kf
      for kf in anchor_obs_per_kf
      if args.anchor.blackout_start <= kf_times[kf] <= args.anchor.blackout_end
    }
    for kf in blackout:
      del anchor_obs_per_kf[kf]
    if blackout:
      print(
        f"  [{name}] anchor blackout "
        f"{args.anchor.blackout_start:.1f}-{args.anchor.blackout_end:.1f}s: "
        f"dropped {len(blackout)} anchor kfs"
      )

  if args.anchor.keep_every > 1 and anchor_obs_per_kf:
    keep = set(sorted(anchor_obs_per_kf.keys())[:: args.anchor.keep_every])
    dropped = len(anchor_obs_per_kf) - len(keep)
    anchor_obs_per_kf = {kf: v for kf, v in anchor_obs_per_kf.items() if kf in keep}
    print(
      f"  [{name}] anchor keep_every={args.anchor.keep_every}: "
      f"kept {len(keep)}, dropped {dropped}"
    )

  front_masks = _build_non_anchor_tag_masks(
    front_frames, front_tags_per_kf, args.anchor.tag_id, args.frontend.mask_tag_pad_px
  )
  rear_masks = _build_non_anchor_tag_masks(
    rear_frames, rear_tags_per_kf, args.anchor.tag_id, args.frontend.mask_tag_pad_px
  )
  static_rects = list(args.frontend.mask_rear_rects)
  _apply_static_mask_rects(rear_masks, static_rects)

  cfg = FrontendConfig(
    n_features=args.frontend.orb_n_features,
    ransac_threshold_px=args.frontend.ransac_threshold_px,
    max_half_angle_deg=args.frontend.max_half_angle_deg,
    use_dis_flow=args.frontend.use_dis_flow,
  )
  imu_to_body_pre = np.asarray(args.imu.imu_to_body, dtype=np.float64)
  gyro_body_pre = (
    np.deg2rad(np.asarray(data.gyro_deg, dtype=np.float64)) @ imu_to_body_pre.T
  )
  imu_t_pre = np.asarray(data.imu_timestamps, dtype=np.float64)
  body_rotations_between = _gyro_integrated_rotations(
    imu_t_pre, gyro_body_pre, kf_times
  )
  front_predictor = _make_imu_klt_predictor(
    front_lens.params, front_lens.extrinsic.rotation().matrix(), body_rotations_between
  )
  rear_predictor = _make_imu_klt_predictor(
    rear_lens.params, rear_lens.extrinsic.rotation().matrix(), body_rotations_between
  )
  front_orb, front_tracks = run_lens_frontend(
    front_frames,
    front_lens.params,
    cfg,
    masks=front_masks,
    rotation_predictor=front_predictor,
  )
  rear_orb, rear_tracks = run_lens_frontend(
    rear_frames,
    rear_lens.params,
    cfg,
    masks=rear_masks,
    rotation_predictor=rear_predictor,
  )
  front_kept = {
    t: o for t, o in front_tracks.items() if len(o) >= args.frontend.min_track_length
  }
  rear_kept = {
    t: o for t, o in rear_tracks.items() if len(o) >= args.frontend.min_track_length
  }
  front_rejected: dict[int, dict[int, Float64Array]] = {}
  rear_rejected: dict[int, dict[int, Float64Array]] = {}
  if args.frontend.min_parallax_deg > 0.0:
    front_kept, front_rejected = filter_tracks_by_parallax(
      front_kept, front_lens.params, args.frontend.min_parallax_deg
    )
    rear_kept, rear_rejected = filter_tracks_by_parallax(
      rear_kept, rear_lens.params, args.frontend.min_parallax_deg
    )

  f_first, f_last, f_shared = pick_init_pair(
    front_kept, n_frames, args.frontend.init_min_shared_tracks
  )
  r_first, r_last, r_shared = pick_init_pair(
    rear_kept, n_frames, args.frontend.init_min_shared_tracks
  )
  if f_shared >= r_shared:
    init_tracks, init_params, init_extrinsic = (
      front_kept,
      front_lens.params,
      front_lens.extrinsic,
    )
    init_first, init_last, init_shared = f_first, f_last, f_shared
  else:
    init_tracks, init_params, init_extrinsic = (
      rear_kept,
      rear_lens.params,
      rear_lens.extrinsic,
    )
    init_first, init_last, init_shared = r_first, r_last, r_shared
  if init_shared < 8:
    raise RuntimeError(f"{name}: no lens has ≥8 shared tracks (best={init_shared})")

  pix_a = np.asarray(
    [
      obs[init_first]
      for obs in init_tracks.values()
      if init_first in obs and init_last in obs
    ],
    dtype=np.float64,
  )
  pix_b = np.asarray(
    [
      obs[init_last]
      for obs in init_tracks.values()
      if init_first in obs and init_last in obs
    ],
    dtype=np.float64,
  )
  bears_a = mei_unproject(pix_a, init_params)
  bears_b = mei_unproject(pix_b, init_params)
  r_cw_last, t_cw_last, _ = two_view_essential(bears_a, bears_b, threshold=0.01)
  cam_poses = interpolate_camera_poses(r_cw_last, t_cw_last, n_frames)
  body_poses_unscaled = body_poses_from_camera_poses(cam_poses, init_extrinsic)

  # Visual-odometry front-end: recover the real per-keyframe trajectory shape
  # from front-lens correspondences (front owns rotation/shape) instead of the
  # two-keyframe straight line. Falls back to the straight line above if the
  # incremental SfM cannot register enough keyframes.
  if args.frontend.use_visual_odometry:
    sfm_cams, _ = incremental_sfm(
      _bearings_by_track(front_kept, front_lens.params),
      n_frames,
      min_shared=args.frontend.init_min_shared_tracks,
    )
    sfm_cams, _vo_outliers = _reject_vo_pose_outliers(sfm_cams)
    if _vo_outliers:
      print(
        f"  [{name}] visual-odometry rejected {len(_vo_outliers)} rotation-spike kfs"
      )
    if len(sfm_cams) >= max(8, n_frames // 4):
      body_poses_unscaled = body_poses_from_camera_poses(
        _fill_camera_poses(sfm_cams, n_frames), front_lens.extrinsic
      )
      print(
        f"  [{name}] visual-odometry init: registered "
        f"{len(sfm_cams)}/{n_frames} keyframes"
      )
    else:
      print(
        f"  [{name}] visual-odometry init: only {len(sfm_cams)}/{n_frames} "
        f"keyframes registered — using straight-line init"
      )

  imu_to_body = np.asarray(args.imu.imu_to_body, dtype=np.float64)
  raw_gyro = np.deg2rad(np.asarray(data.gyro_deg, dtype=np.float64)) @ imu_to_body.T
  raw_accel = np.asarray(data.accel, dtype=np.float64) @ imu_to_body.T
  imu = ImuSamples(
    t=np.asarray(data.imu_timestamps, dtype=np.float64), gyro=raw_gyro, accel=raw_accel
  )
  raw_factory_bias = factory_imu_bias(data.gyro_calib_numbers)
  bias0 = gtsam.imuBias.ConstantBias(
    imu_to_body @ np.asarray(raw_factory_bias.accelerometer(), dtype=np.float64),
    imu_to_body @ np.asarray(raw_factory_bias.gyroscope(), dtype=np.float64),
  )
  windows = find_stationary_windows(imu, bias0)
  clip_end = imu.t[-1] if args.duration is None else args.start + args.duration
  windows = [(s, e) for s, e in windows if args.start <= imu.t[s] <= clip_end]
  gravity_world_initial: Float64Array | None = None
  if windows:
    first_start, first_end = windows[0]
    stationary_mean = (imu.accel[first_start:first_end] - bias0.accelerometer()).mean(
      axis=0
    )
    stationary_mag = float(np.linalg.norm(stationary_mean))
    if 8.0 < stationary_mag < 11.0 and abs(stationary_mag - 9.81) > 0.05:
      accel_scale = 9.81 / stationary_mag
      imu = ImuSamples(
        t=imu.t, gyro=imu.gyro, accel=(imu.accel * accel_scale).astype(np.float64)
      )
      stationary_mean = stationary_mean * accel_scale
    gravity_body = -stationary_mean / float(np.linalg.norm(stationary_mean)) * 9.81
    t_stationary_mid = 0.5 * (imu.t[first_start] + imu.t[first_end - 1])
    kf_for_g = int(np.argmin(np.abs(kf_times - t_stationary_mid)))
    gravity_world_initial = (
      body_poses_unscaled[kf_for_g].rotation().matrix() @ gravity_body
    )

  front_unscaled = triangulate_lens(
    front_kept,
    body_poses_unscaled,
    front_lens.extrinsic,
    front_lens.params,
    args.frontend.min_track_length,
    min_landmark_depth=0.01,
    max_landmark_depth=1000.0,
  )
  rear_unscaled = triangulate_lens(
    rear_kept,
    body_poses_unscaled,
    rear_lens.extrinsic,
    rear_lens.params,
    args.frontend.min_track_length,
    min_landmark_depth=0.01,
    max_landmark_depth=1000.0,
  )
  body_poses_visual = list(body_poses_unscaled)

  # Without a stationary window the LSQ-fit gravity is the only signal we
  # have; preintegration uses a placeholder vertical gravity, then the
  # alignment runs against the same data and the LSQ-fit value sticks.
  pim_gravity = (
    (
      float(gravity_world_initial[0]),
      float(gravity_world_initial[1]),
      float(gravity_world_initial[2]),
    )
    if gravity_world_initial is not None
    else (0.0, 0.0, -9.81)
  )
  params_pim = make_preintegration_params(
    gyro_noise_sigma=args.imu.gyro_noise_sigma,
    accel_noise_sigma=args.imu.accel_noise_sigma,
    gyro_walk=args.imu.gyro_random_walk,
    accel_walk=args.imu.accel_random_walk,
    gravity_world=pim_gravity,
  )
  pim_init = preintegrate_between_keyframes(params_pim, imu, kf_times, bias0)
  alignment = visual_inertial_align(
    body_poses_visual, pim_init, fixed_gravity=gravity_world_initial
  )
  if alignment.scale < 0.0:
    origin = np.asarray(body_poses_visual[0].translation(), dtype=np.float64)
    body_poses_visual = [
      gtsam.Pose3(
        p.rotation(),
        gtsam.Point3(*(2.0 * origin - np.asarray(p.translation(), dtype=np.float64))),
      )
      for p in body_poses_visual
    ]
    alignment = visual_inertial_align(
      body_poses_visual, pim_init, fixed_gravity=gravity_world_initial
    )
  # Override LSQ-fit gravity with the IMU stationary-window estimate so the
  # world frame stays gravity-aligned (LSQ-fit gravity can drift up to ~30°
  # from the IMU truth, which then tilts the rerun viewer's z-up convention).
  if gravity_world_initial is not None:
    alignment = Alignment(
      scale=alignment.scale,
      gravity_world=gravity_world_initial,
      velocities=alignment.velocities,
    )
  print(f"  [{name}] VI scale={alignment.scale:.4f}")
  body_poses, world_rotation = apply_alignment_to_poses(body_poses_visual, alignment)
  world_rot_mat = world_rotation.matrix()
  velocities = [world_rot_mat @ v for v in alignment.velocities]
  params_aligned = make_preintegration_params(
    gyro_noise_sigma=args.imu.gyro_noise_sigma,
    accel_noise_sigma=args.imu.accel_noise_sigma,
    gyro_walk=args.imu.gyro_random_walk,
    accel_walk=args.imu.accel_random_walk,
    gravity_world=(0.0, 0.0, -9.81),
  )
  pim = preintegrate_between_keyframes(params_aligned, imu, kf_times, bias0)

  use_anchor = args.anchor.tag_id >= 0 and bool(anchor_obs_per_kf)
  anchor_pose_world_local: gtsam.Pose3 | None = None
  if use_anchor:
    first_kf = min(anchor_obs_per_kf.keys())
    sides = anchor_obs_per_kf[first_kf]
    first_lens = front_lens if "front" in sides else rear_lens
    first_cam_T_tag = sides.get("front", sides.get("rear"))
    assert first_cam_T_tag is not None
    anchor_pose_world_local = (
      body_poses[first_kf].compose(first_lens.extrinsic).compose(first_cam_T_tag)
    )
    # Visual-odometry mode keeps the vision trajectory and lets the anchor
    # enter only as a joint-graph factor; the per-keyframe PnP pose overwrite
    # and IMU gap dead-reckon below are the anchor-owns-the-trajectory path.
    if not args.frontend.use_visual_odometry:
      for kf, kf_sides in anchor_obs_per_kf.items():
        lens = front_lens if "front" in kf_sides else rear_lens
        cam_T_tag = kf_sides.get("front", kf_sides.get("rear"))
        assert cam_T_tag is not None
        body_poses[kf] = anchor_pose_world_local.compose(
          lens.extrinsic.compose(cam_T_tag).inverse()
        )
      sorted_kfs = sorted(anchor_obs_per_kf.keys())
      if len(sorted_kfs) >= 2:
        jumps = [
          float(
            np.linalg.norm(
              np.asarray(body_poses[sorted_kfs[i]].translation())
              - np.asarray(body_poses[sorted_kfs[i - 1]].translation())
            )
          )
          for i in range(1, len(sorted_kfs))
        ]
        i = int(np.argmax(jumps))
        print(
          f"  [{name}] max anchor-init Δp: {jumps[i]:.3f} m at kf {sorted_kfs[i + 1]}"
        )
      # IMU forward-propagation from each anchor kf fills the gap kfs with the
      # IMU-integrated trajectory instead of constant-velocity pose interpolation.
      first_anchor = sorted_kfs[0]
      for k in range(first_anchor):
        body_poses[k] = body_poses[first_anchor]
        velocities[k] = velocities[first_anchor]
      for i in range(len(sorted_kfs)):
        a = sorted_kfs[i]
        b = sorted_kfs[i + 1] if i + 1 < len(sorted_kfs) else n_frames
        if b - a <= 1:
          continue
        # Bidirectional only for interior gaps (b is an anchor kf with a known
        # snapped pose): distribute the forward-prop end drift across the gap.
        bidir = b < n_frames
        nav = gtsam.NavState(body_poses[a], velocities[a])
        fwd_pose: dict[int, gtsam.Pose3] = {}
        fwd_vel: dict[int, Float64Array] = {}
        for k in range(a + 1, b + 1 if bidir else b):
          nav = pim[k - 1].predict(nav, bias0)
          fwd_pose[k] = nav.pose()
          fwd_vel[k] = np.asarray(nav.velocity(), dtype=np.float64)
        dp = (
          np.asarray(body_poses[b].translation(), dtype=np.float64)
          - np.asarray(fwd_pose[b].translation(), dtype=np.float64)
          if bidir
          else np.zeros(3, dtype=np.float64)
        )
        for k in range(a + 1, b):
          if k in anchor_obs_per_kf:
            continue
          p = fwd_pose[k]
          if bidir:
            tk = np.asarray(p.translation(), dtype=np.float64) + (k - a) / (b - a) * dp
            body_poses[k] = gtsam.Pose3(p.rotation(), gtsam.Point3(*tk))
          else:
            body_poses[k] = p
          velocities[k] = fwd_vel[k]

  align_mat = alignment.scale * world_rotation.matrix()
  front_unscaled_metric = {
    tid: align_mat @ np.asarray(front_unscaled[tid], dtype=np.float64)
    for tid in front_unscaled
  }
  rear_unscaled_metric = {
    tid: align_mat @ np.asarray(rear_unscaled[tid], dtype=np.float64)
    for tid in rear_unscaled
  }
  loop_tracks: dict[int, dict[int, Float64Array]] = {}
  if args.frontend.loop_closure and (
    args.frontend.use_visual_odometry
    or (args.frontend.anchored_loop_closure and use_anchor)
  ):
    loop_tracks = _loop_closure_tracks(
      front_frames,
      front_masks,
      body_poses,
      front_lens,
      args.frontend.orb_n_features,
      max(front_kept, default=0),
      args.frontend.loop_closure_max_dist,
      args.frontend.loop_closure_min_kf_gap,
    )
  if use_anchor:
    anchor_kfs = set(anchor_obs_per_kf.keys())
    gap_par = args.frontend.gap_min_parallax_deg

    def _gated(
      kept: dict[int, dict[int, Float64Array]],
    ) -> dict[int, dict[int, Float64Array]]:
      """Default behavior: keep only tracks co-visible with an anchor kf."""
      return {t: kept[t] for t in kept if anchor_kfs & kept[t].keys()}

    def _gap_subset(
      kept: dict[int, dict[int, Float64Array]], params: MeiParams
    ) -> dict[int, dict[int, Float64Array]]:
      """keep_gap: keep full tracks but triangulate from anchor-covisible views
      only (the blackout poses are IMU dead-reckon and would corrupt the DLT),
      then parallax-gate to drop short-baseline gap landmarks. The full track
      (incl. blackout obs) stays in front_kept/rear_kept for the reprojection
      factors, so vision constrains the unanchored poses."""
      sub = {
        t: {k: v for k, v in obs.items() if k in anchor_kfs} for t, obs in kept.items()
      }
      if gap_par > 0.0:
        sub, _ = filter_tracks_by_parallax(sub, params, gap_par)
      return sub

    if args.frontend.use_visual_odometry:
      # Vision owns every keyframe pose, so triangulate the full tracks (no
      # anchor-covisibility gating): the near-field rear structure survives.
      front_for_tri = front_kept
      rear_for_tri = rear_kept
    elif args.frontend.keep_gap_landmarks:
      front_for_tri = _gap_subset(front_kept, front_lens.params)
      if args.frontend.keep_gap_front_only:
        rear_kept = _gated(rear_kept)
        rear_for_tri = rear_kept
      else:
        rear_for_tri = _gap_subset(rear_kept, rear_lens.params)
    else:
      front_kept = _gated(front_kept)
      rear_kept = _gated(rear_kept)
      front_for_tri = front_kept
      rear_for_tri = rear_kept
    front_points = triangulate_lens(
      {t: front_for_tri[t] for t in front_for_tri if t in front_unscaled},
      body_poses,
      front_lens.extrinsic,
      front_lens.params,
      args.frontend.min_track_length,
      max_landmark_depth=args.frontend.max_landmark_depth,
    )
    rear_points = triangulate_lens(
      {t: rear_for_tri[t] for t in rear_for_tri if t in rear_unscaled},
      body_poses,
      rear_lens.extrinsic,
      rear_lens.params,
      args.frontend.min_track_length,
      max_landmark_depth=args.frontend.max_landmark_depth,
    )
    if args.frontend.use_visual_odometry:
      cap = args.frontend.vo_max_landmarks_per_lens
      front_points = _cap_landmarks(front_points, front_for_tri, cap)
      rear_points = _cap_landmarks(rear_points, rear_for_tri, cap)
      if loop_tracks:
        # Triangulated + merged after the cap so the drift-cancelling revisit
        # landmarks are never culled by the per-lens landmark budget. With
        # anchored_loop_closure, triangulate from the anchor-covisible views so
        # the revisit structure corrects (not fits) the drifted anchor-free tail.
        _add_loop_landmarks(
          front_points,
          front_kept,
          loop_tracks,
          body_poses,
          front_lens,
          args.frontend.min_track_length,
          args.frontend.max_landmark_depth,
          name,
          "loop closure",
          anchor_kfs=anchor_kfs if args.frontend.anchored_loop_closure else None,
        )
    elif loop_tracks:
      # Anchor-seeded mode: the gated front_kept dropped the anchor-free tail
      # tracks, so merge the revisit loop landmarks here (after gating) to tie
      # the drifting tail back to the early trajectory in the bundle.
      _add_loop_landmarks(
        front_points,
        front_kept,
        loop_tracks,
        body_poses,
        front_lens,
        args.frontend.min_track_length,
        args.frontend.max_landmark_depth,
        name,
        "anchored loop closure",
        anchor_kfs=anchor_kfs,
      )
  else:
    front_points = front_unscaled_metric
    rear_points = rear_unscaled_metric

  front_tracks_tri = {t: front_kept[t] for t in front_points if t in front_kept}
  rear_tracks_tri = {t: rear_kept[t] for t in rear_points if t in rear_kept}
  if args.frontend.use_visual_odometry:
    m = args.frontend.vo_max_obs_per_track
    front_tracks_tri = {t: _subsample_obs(o, m) for t, o in front_tracks_tri.items()}
    rear_tracks_tri = {t: _subsample_obs(o, m) for t, o in rear_tracks_tri.items()}

  v_max = float(max(np.linalg.norm(v) for v in velocities))
  print(f"  [{name}] VIO LSQ |v|max={v_max:.2f} m/s")
  biases = [bias0 for _ in range(n_frames)]

  base_tag_obs = anchor_pose_from_tags(
    rear_tags_per_kf,
    anchor_tag_id=args.base_tag_id,
    tag_size=args.gripper.tag_size,
    params=rear_lens.params,
  )
  body_T_base_tag: gtsam.Pose3 | None = None
  if base_tag_obs:
    body_T_base_tag = _mean_pose(
      [rear_lens.extrinsic.compose(p) for p in base_tag_obs.values()]
    )

  def _per_kf_body_T_finger(finger_id: int) -> dict[int, gtsam.Pose3]:
    if finger_id <= 0:
      return {}
    out: dict[int, gtsam.Pose3] = {}
    for kf in range(min(n_frames, len(rear_tags_per_kf))):
      corners = next(
        (d.corners for d in rear_tags_per_kf[kf] if int(d.tag_id) == finger_id),
        None,
      )
      if corners is None:
        continue
      cam_T_finger = estimate_camera_to_tag_pose(
        corners, args.gripper.tag_size, rear_lens.params
      )
      if cam_T_finger is None:
        continue
      out[kf] = rear_lens.extrinsic.compose(cam_T_finger)
    return out

  body_T_left_finger_per_kf = _per_kf_body_T_finger(args.left_finger_tag_id)
  body_T_right_finger_per_kf = _per_kf_body_T_finger(args.right_finger_tag_id)

  joint_angles: dict[str, list[float]] = {}
  for joint_name, finger_id in (
    ("left_finger", args.left_finger_tag_id),
    ("right_finger", args.right_finger_tag_id),
  ):
    if finger_id <= 0:
      continue
    joint_angles[joint_name] = _finger_joint_angles_per_kf(
      rear_tags_per_kf,
      rear_lens,
      base_tag_id=args.base_tag_id,
      finger_tag_id=finger_id,
      tag_size=args.gripper.tag_size,
      n_frames=n_frames,
    )

  return PreparedTracker(
    name=name,
    body_poses=body_poses,
    velocities=velocities,
    biases=biases,
    preintegrated=pim,
    front_lens=front_lens,
    rear_lens=rear_lens,
    front_tracks_tri=front_tracks_tri,
    rear_tracks_tri=rear_tracks_tri,
    front_points=front_points,
    rear_points=rear_points,
    anchor_obs_per_kf=anchor_obs_per_kf,
    anchor_pose_world_local=anchor_pose_world_local,
    body_T_base_tag=body_T_base_tag,
    joint_angles=joint_angles,
    kf_times=kf_times,
    front_frames=front_frames,
    rear_frames=rear_frames,
    front_keypoints=[f.keypoints for f in front_orb],
    rear_keypoints=[f.keypoints for f in rear_orb],
    front_tags_per_kf=front_tags_per_kf,
    rear_tags_per_kf=rear_tags_per_kf,
    front_rejected=front_rejected,
    rear_rejected=rear_rejected,
    rear_static_mask_rects=static_rects,
    body_T_left_finger_per_kf=body_T_left_finger_per_kf,
    body_T_right_finger_per_kf=body_T_right_finger_per_kf,
    left_finger_tag_id=args.left_finger_tag_id,
    right_finger_tag_id=args.right_finger_tag_id,
    n_frames=n_frames,
  )
