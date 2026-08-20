"""Per-tracker prep for VI-BA: read one tracker's lenses and IMU out of a base
recording, detect its tags, recover the anchor poses by PnP, chain each lens's
tracks, seed a trajectory from visual odometry (the metric anchor poses are the
starved fallback), align it to the IMU for scale and gravity, and re-triangulate
the landmarks metrically. Returns a ``PreparedTracker`` ready for
``specialist.vi_ba.compose`` to assemble into a joint factor graph.
"""

from __future__ import annotations

import ctypes
from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import cv2
import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import MeiParams, mei_unproject
from specialist.record import schema
from specialist.record.frames import LensVideo, RecordingKeyframes
from specialist.record.session import (
  SessionRecording,
  covered_rows,
  fps_from_times,
  nearest_rows,
)
from specialist.vi_ba.anchor_detection import (
  TagDetection,
  anchor_pose_from_tags,
  configure_detector,
  detect_all_tags,
  tag_half_angle,
)
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.configs import (
  AnchorCfg,
  FrontendCfg,
  GripperCfg,
)
from specialist.vi_ba.features import (
  Frame,
  Rect,
  available_ram,
  detect_orb,
  mask_plane,
  match_pair,
  parallel_map,
)
from specialist.vi_ba.finger_kinematics import (
  FingerReproj,
  build_finger_reproj,
  frame_finger_angles,
)
from specialist.vi_ba.imu import (
  GRAVITY,
  GRAVITY_WORLD,
  ImuSamples,
  find_stationary_windows,
  keyframe_delta_rotations,
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
from specialist.vi_ba.tracks import (
  filter_tracks_by_parallax,
  run_lens_frontend,
  triangulate_lens,
  unproject_tracks,
)
from specialist.vi_ba.urdf_joints import read_revolute_joints

Float64Array = npt.NDArray[np.float64]

_A = TypeVar("_A")

# ORB loop-closure track recovery (``_loop_closure_tracks``).
_LOOP_RANSAC_PX = 4.5  # homography RANSAC reprojection tolerance
_LOOP_MIN_VIEWS = 3  # keyframes a union-found track must span to be kept

# Visual-odometry pose-spike rejection (``_reject_vo_pose_outliers``).
_VO_REJECT_FACTOR = 8.0  # residual/median-step ratio marking a PnP teleport

# Keyframes read at once from a banded lens.
PREP_BAND = 256

# Share of the available memory the concurrent decodes may claim between them.
_PREP_MEMORY_SHARE = 0.35


__all__ = [
  "PreparedTracker",
  "TrackerArgs",
  "captures_that_fit",
  "prepare_tracker",
]


def resolve_keyframe_every(keyframe_every: int, fps: float) -> int:
  """Resolve the keyframe stride: an explicit positive value, else a ~0.10s
  interval (6 at 60fps, 3 at 30fps). The review timeline buckets by the auto
  stride, so an override must be a multiple of it."""
  stride = schema.keyframe_stride(fps)
  if keyframe_every > 0 and keyframe_every % stride:
    raise ValueError(
      f"--keyframe-every {keyframe_every} does not land on the {stride}-frame "
      f"keyframe lattice at {fps:.2f} fps; use a multiple of {stride}"
    )
  return keyframe_every if keyframe_every > 0 else stride


def release_freed_memory() -> None:
  """Hand freed heap back to the OS. glibc raises its mmap threshold once it has
  seen large blocks freed, after which keyframe-sized allocations come from the
  heap and are retained. A no-op where the C library has no ``malloc_trim``."""
  try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
  except (OSError, AttributeError):
    pass


def _per_band(
  fn: Callable[[list[np.ndarray]], list[_A]], frames: Sequence[np.ndarray]
) -> list[_A]:
  """Map over the frames a band at a time so a banded sequence is read in
  order; mapping across the whole lens would reload a band per frame."""
  out: list[_A] = []
  for begin in range(0, len(frames), PREP_BAND):
    end = min(begin + PREP_BAND, len(frames))
    out.extend(fn([frames[i] for i in range(begin, end)]))
  return out


def _enough_registered(
  n_registered: int, n_keyframes: int, ratio: float = 0.25
) -> bool:
  """A seed needs ``ratio`` of the keyframes, and at least eight."""
  return n_registered >= max(8, int(n_keyframes * ratio))


@dataclass
class TrackerArgs:
  """Per-tracker prep knobs. Composes nested configs from ``configs.py``."""

  base_rrd: Path  # the session's base recording; the recording is the window
  tracker: str  # this tracker's entity name inside the base recording
  base_tag_id: int
  anchor: AnchorCfg
  keyframe_every: int  # 0 = auto (round(fps/10))
  urdf_path: Path | None  # gripper URDF for finger joint kinematics
  # 0 disables the corresponding URDF revolute joint angle recovery.
  left_finger_tag_id: int
  right_finger_tag_id: int
  frontend: FrontendCfg
  gripper: GripperCfg
  # True when sibling captures prepare concurrently; gates per-thread AprilTag
  # detectors, which only pay for themselves when tag walks actually overlap.
  parallel_siblings: bool = False


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
  front_tracks: dict[int, dict[int, Float64Array]]
  rear_tracks: dict[int, dict[int, Float64Array]]
  front_points: dict[int, Float64Array]
  rear_points: dict[int, Float64Array]
  # ``{tag_id: {kf: {side: cam_T_tag}}}``; ``anchor_locals_by_tag`` gives each
  # sighted tag's pose in this tracker's local world.
  anchor_obs_by_tag: dict[int, dict[int, dict[str, gtsam.Pose3]]]
  anchor_locals_by_tag: dict[int, gtsam.Pose3]
  body_T_base_tag: gtsam.Pose3 | None
  # URDF joint name → per-keyframe rotation about the URDF z-axis. NaN where
  # either tag is missing or PnP fails. Empty when no finger ids configured.
  joint_angles: dict[str, list[float]]
  keyframe_times: Float64Array
  # Session-clock frame number per keyframe: the ``keyframe`` timeline value
  # is ``schema.keyframe_index(row, stride)``.
  keyframe_rows: npt.NDArray[np.int64]
  front_keypoints: list[Float64Array]
  rear_keypoints: list[Float64Array]
  front_tags_per_kf: list[list[TagDetection]]
  rear_tags_per_kf: list[list[TagDetection]]
  front_rejected: dict[int, dict[int, Float64Array]]
  rear_rejected: dict[int, dict[int, Float64Array]]
  rear_static_mask_rects: list[tuple[int, int, int, int]]
  # URDF joint name -> kinematic inputs for the joint-angle reprojection factors.
  finger_reproj: dict[str, FingerReproj]
  left_finger_tag_id: int
  right_finger_tag_id: int
  # Body-frame, rescaled IMU samples, for IMU-rate propagation between keyframes.
  imu: ImuSamples | None = None
  # Per-joint detection-only finger angles at the frame rate, as (times, angles);
  # raw, pinned to the solved keyframe angles at export time.
  finger_frame_angles: dict[str, tuple[Float64Array, Float64Array]] | None = None

  @property
  def n_keyframes(self) -> int:
    return len(self.body_poses)


def _build_non_anchor_tag_masks(
  shape: tuple[int, int],
  tags_per_kf: list[list[TagDetection]],
  exclude_tag_ids: set[int],
  pad_px: int,
) -> list[list[Rect]]:
  """Per keyframe, the padded boxes of every non-anchor tag. Rectangles rather
  than planes: a full-length dual capture would hold tens of GB of masks. Only
  the frame shape is needed, so this never walks the lens's pixels."""
  h, w = shape
  blocked: list[list[Rect]] = []
  for tags in tags_per_kf:
    rects: list[Rect] = []
    for tag in tags:
      if int(tag.tag_id) in exclude_tag_ids:
        continue
      corners = np.asarray(tag.corners, dtype=np.float64)
      x0 = max(int(np.floor(corners[:, 0].min())) - pad_px, 0)
      y0 = max(int(np.floor(corners[:, 1].min())) - pad_px, 0)
      x1 = min(int(np.ceil(corners[:, 0].max())) + pad_px, w)
      y1 = min(int(np.ceil(corners[:, 1].max())) + pad_px, h)
      if x1 > x0 and y1 > y0:
        rects.append((x0, y0, x1, y1))
    blocked.append(rects)
  return blocked


def _apply_static_mask_rects(
  blocked: list[list[Rect]], rects: list[Rect], shape: tuple[int, int]
) -> None:
  if not rects or not blocked:
    return
  h, w = shape
  for x1, y1, x2, y2 in rects:
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
      raise ValueError(
        f"mask_rear_rects entry {(x1, y1, x2, y2)} is empty after clamping to the "
        f"{w}x{h} frame — check the coordinate order (x1 y1 x2 y2) and bounds"
      )
    for frame_rects in blocked:
      frame_rects.append((x1c, y1c, x2c, y2c))


# One lens side for anchor detection: name, lens, per-kf tag detections, and
# the gyro-derived per-lens rotation increments.
_LensSide = tuple[str, LensSetup, list[list[TagDetection]], list[Float64Array]]


def _detect_anchor_obs(
  tag_id: int,
  tag_size: float,
  lens_sides: tuple[_LensSide, ...],
) -> dict[int, dict[str, gtsam.Pose3]]:
  """``{kf: {side: cam_T_tag}}`` for one anchor id across both lens sides.
  Ungated: the prep grounding and the secondary-anchor seeds need edge
  sightings where nothing better is visible; the corner factors gate."""
  obs: dict[int, dict[str, gtsam.Pose3]] = {}
  for side, lens, tags, rotations in lens_sides:
    for kf, pose in anchor_pose_from_tags(
      tags,
      tag_id,
      tag_size,
      lens.params,
      kf_rotations=rotations,
    ).items():
      obs.setdefault(kf, {})[side] = pose
  return obs


def _choose_prep_anchor(
  tag_ids: tuple[int, ...],
  obs_by_tag: dict[int, dict[int, dict[str, gtsam.Pose3]]],
) -> int:
  """The configured id sighted at the most keyframes; ties break toward the
  earlier position in ``tag_ids``."""
  return max(tag_ids, key=lambda tid: len(obs_by_tag.get(tid, {})))


def _print_anchor_geometry(
  name: str,
  tag_id: int,
  lens_sides: tuple[_LensSide, ...],
  obs: dict[int, dict[str, gtsam.Pose3]],
) -> None:
  """Print per-anchor sighting-geometry stats: edge length (px), bearing
  half-angle, PnP incidence angle. Diagnostics only, never gated on."""
  edges: list[float] = []
  half_angles: list[float] = []
  for _side, lens, tags, _rotations in lens_sides:
    for dets in tags:
      for det in dets:
        if int(det.tag_id) != tag_id:
          continue
        corners = det.corners
        edge = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        edges.append(float(edge.min()))
        half_angle = tag_half_angle(det, lens.params)
        if np.isfinite(half_angle):
          half_angles.append(float(np.degrees(half_angle)))
  if not edges:
    return
  incidences: list[float] = []
  for sides in obs.values():
    for pose in sides.values():
      t = np.asarray(pose.translation(), dtype=np.float64)
      dist = float(np.linalg.norm(t))
      if dist < 1e-9:
        continue
      cos_incidence = -float(pose.rotation().matrix()[:, 2] @ t) / dist
      incidences.append(float(np.degrees(np.arccos(np.clip(cos_incidence, -1.0, 1.0)))))

  def med_p90(values: list[float]) -> str:
    if not values:
      return "-/-"
    return f"{float(np.median(values)):.0f}/{float(np.percentile(values, 90)):.0f}"

  print(
    f"  [{name}] anchor {tag_id} geometry: {len(edges)} sightings, "
    f"edge px min/median {min(edges):.0f}/{float(np.median(edges)):.0f}, "
    f"half-angle median/p90 {med_p90(half_angles)} deg, "
    f"incidence median/p90 {med_p90(incidences)} deg"
  )


def _per_kf_cam_T_tag(
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  tag_id: int,
  tag_size: float,
  n_keyframes: int,
  kf_rotations: list[Float64Array],
) -> list[gtsam.Pose3 | None]:
  """Per-keyframe ``cam_T_tag`` for one tag id as an index-aligned list;
  IPPE disambiguated by the gyro-rotated prior."""
  out: list[gtsam.Pose3 | None] = [None] * n_keyframes
  for kf, p in anchor_pose_from_tags(
    tags_per_kf[:n_keyframes], tag_id, tag_size, lens.params, kf_rotations
  ).items():
    out[kf] = p
  return out


def _first_sighting_local(
  obs: dict[int, dict[str, gtsam.Pose3]],
  front_lens: LensSetup,
  rear_lens: LensSetup,
  body_poses: list[gtsam.Pose3],
) -> gtsam.Pose3:
  """An anchor's pose in the local world from its first sighting (front lens
  preferred)."""
  first_kf = min(obs)
  lens, cam_T_tag = _side_lens_and_pose(obs[first_kf], front_lens, rear_lens)
  return body_poses[first_kf].compose(lens.extrinsic).compose(cam_T_tag)


def _translation(pose: gtsam.Pose3) -> Float64Array:
  return np.asarray(pose.translation(), dtype=np.float64)


def _mean_pose(poses: list[gtsam.Pose3]) -> gtsam.Pose3:
  """Mean of a list of poses: arithmetic mean of translations + log-mean
  rotation about the first sample. Used for a fixed body-rigid tag where
  every observation should agree up to PnP noise."""
  ref = poses[0]
  translations = np.stack([_translation(p) for p in poses])
  log_sum = np.zeros(3)
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
  n_keyframes: int,
  kf_rotations: list[Float64Array],
) -> list[float]:
  """``Logmap(base_tag_T_finger)[2]`` per keyframe — the signed rotation about
  the URDF revolute-joint axis (``+z`` by URDF convention). The base tag is
  rigidly mounted to the body so its ``cam_T_tag`` is averaged once over all
  visible keyframes; the finger tag's ``cam_T_tag`` is recovered per kf."""
  cam_T_base_per_kf = _per_kf_cam_T_tag(
    tags_per_kf, lens, base_tag_id, tag_size, n_keyframes, kf_rotations
  )
  cam_T_finger_per_kf = _per_kf_cam_T_tag(
    tags_per_kf, lens, finger_tag_id, tag_size, n_keyframes, kf_rotations
  )
  observed_base = [p for p in cam_T_base_per_kf if p is not None]
  if not observed_base:
    return [float("nan")] * n_keyframes
  cam_T_base = _mean_pose(observed_base)
  out: list[float] = [float("nan")] * n_keyframes
  for kf in range(n_keyframes):
    cam_T_finger = cam_T_finger_per_kf[kf]
    if cam_T_finger is None:
      continue
    base_tag_T_finger = cam_T_base.between(cam_T_finger)
    out[kf] = float(gtsam.Rot3.Logmap(base_tag_T_finger.rotation())[2])
  return out


def _bearings_by_track(
  tracks: dict[int, dict[int, Float64Array]], params: MeiParams
) -> dict[int, dict[int, Float64Array]]:
  """``{tid: {kf: unit bearing}}`` for the visual-odometry front-end."""
  return {
    tid: dict(zip(kfs, bearings, strict=True))
    for tid, kfs, bearings in unproject_tracks(
      tracks, params, min_obs=1, normalize=True
    )
  }


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
  orb: list[Frame],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  base_track_id: int,
  max_center_dist: float,
  min_kf_gap: int,
  n_neighbors: int,
  match_ratio: float,
  min_inliers: int,
) -> dict[int, dict[int, Float64Array]]:
  """Recover tracks broken by fast lifts via ORB loop closure.

  Fast repositioning breaks the chained optical-flow tracks, so a physical point
  seen before and after a lift becomes two disconnected tracks and nothing ties
  the revisited poses together — the visual odometry drifts across the lift. Here
  ORB features are re-matched between spatially-close, non-adjacent keyframes
  (revisits of the same region), geometrically verified, and union-found
  into multi-view tracks spanning the lifts. Triangulated against the (drifted)
  poses these become wide-baseline landmarks whose reprojection inconsistency
  drives the bundle to pull the revisits back together, cancelling the drift."""
  centers = np.asarray([_translation(p.compose(lens.extrinsic)) for p in body_poses])
  n = len(orb)
  parent: dict[tuple[int, int], tuple[int, int]] = {}

  def find(x: tuple[int, int]) -> tuple[int, int]:
    parent.setdefault(x, x)
    root = x
    while parent[root] != root:
      root = parent[root]
    while parent[x] != root:
      parent[x], x = root, parent[x]
    return root

  pairs: list[tuple[int, int]] = []
  for i in range(n):
    if len(orb[i].keypoints) == 0:
      continue
    dist = np.linalg.norm(centers - centers[i], axis=1)
    cand = [
      int(j)
      for j in np.argsort(dist)
      if j > i and (j - i) > min_kf_gap and dist[j] <= max_center_dist
    ][:n_neighbors]
    pairs.extend((i, j) for j in cand)

  def verify_pair(i: int, j: int) -> npt.NDArray[np.int64] | None:
    m = match_pair(orb[i], orb[j], match_ratio)
    if len(m) < min_inliers:
      return None
    pa = orb[i].keypoints[m[:, 0]]
    pb = orb[j].keypoints[m[:, 1]]
    # Revisits are near-identical small-baseline views, so the essential matrix
    # is degenerate and lets false matches through (which wreck the bundle).
    # A homography is the right model for the small inter-view change and
    # rejects the inconsistent matches. The cv2 RNG is thread-local, so a
    # per-pair seed keeps the RANSAC deterministic on the worker pool.
    cv2.setRNGSeed(i * n + j)
    _, mask = cv2.findHomography(pa, pb, cv2.RANSAC, _LOOP_RANSAC_PX)
    if mask is None:
      return None
    inl = mask.ravel().astype(bool)
    if int(inl.sum()) < min_inliers:
      return None
    return np.asarray(m[inl], dtype=np.int64)

  verified = parallel_map(verify_pair, [i for i, _ in pairs], [j for _, j in pairs])
  for (i, j), matches in zip(pairs, verified, strict=True):
    if matches is None:
      continue
    for a_idx, b_idx in matches:
      ra = find((i, int(a_idx)))
      rb = find((j, int(b_idx)))
      if ra != rb:
        parent[ra] = rb

  comps: dict[tuple[int, int], dict[int, int]] = {}
  for node in list(parent):
    comps.setdefault(find(node), {})[node[0]] = node[1]
  tracks: dict[int, dict[int, Float64Array]] = {}
  tid = base_track_id + 1
  for by_kf in comps.values():
    if len(by_kf) < _LOOP_MIN_VIEWS:
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
  *,
  min_track_length: int,
  max_landmark_depth: float,
  name: str,
  anchor_kfs: set[int] | None = None,
) -> dict[int, Float64Array]:
  """Triangulate revisit ``loop_tracks`` and merge them into the front-lens
  landmarks (``front_points``) and tracks (``front_kept``). Mutates both.

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
    f"  [{name}] loop closure: +{len(loop_pts)} landmarks "
    f"from {len(loop_tracks)} revisit tracks"
  )
  return loop_pts


def _reject_vo_pose_outliers(
  cam_poses: dict[int, gtsam.Pose3],
) -> tuple[dict[int, gtsam.Pose3], list[int]]:
  """Drop incremental-SfM keyframes whose camera center is a local spike.

  Pure monocular SfM has no global consistency, so a degenerate PnP (few
  co-visible landmarks during fast rotation) can teleport a single keyframe far
  off the smooth path. Such a spike corrupts the visual displacements that the
  IMU scale alignment consumes, collapsing the recovered scale. Each interior
  registered keyframe is compared against the linear interpolation of its
  immediate registered neighbors; a residual exceeding ``_VO_REJECT_FACTOR``
  times the median per-keyframe step marks a spike. Spikes are removed
  iteratively (worst first) so ``_fill_camera_poses`` interpolates cleanly
  across them.

  The factor is deliberately loose: a genuine real-motion burst can reach ~7×
  the slow-motion median step and the threshold sits just above that, because
  rejecting such a burst over-interpolates the rotation and wrecks the
  recovered orientation. Only the far-larger PnP teleports should be cut."""
  kept = dict(cam_poses)
  removed: list[int] = []
  for _ in range(len(cam_poses)):
    kfs = sorted(kept)
    if len(kfs) < 5:
      break
    centers = {kf: _translation(kept[kf]) for kf in kfs}
    steps = [
      float(np.linalg.norm(centers[kfs[i + 1]] - centers[kfs[i]]))
      / (kfs[i + 1] - kfs[i])
      for i in range(len(kfs) - 1)
    ]
    median_step = float(np.median(steps))
    if median_step < 1e-12:
      break
    worst_kf: int | None = None
    worst_residual = 0.0
    for i in range(1, len(kfs) - 1):
      prev_kf, kf, next_kf = kfs[i - 1], kfs[i], kfs[i + 1]
      frac = (kf - prev_kf) / (next_kf - prev_kf)
      interpolated = (1.0 - frac) * centers[prev_kf] + frac * centers[next_kf]
      residual = float(np.linalg.norm(centers[kf] - interpolated))
      if residual > worst_residual:
        worst_residual, worst_kf = residual, kf
    if worst_kf is not None and worst_residual > _VO_REJECT_FACTOR * median_step:
      del kept[worst_kf]
      removed.append(worst_kf)
    else:
      break
  return kept, sorted(removed)


def _fill_camera_poses(
  cam_poses: dict[int, gtsam.Pose3], n_keyframes: int
) -> list[gtsam.Pose3]:
  """Expand SfM-registered camera poses to a full per-keyframe list, filling
  unregistered keyframes by SLERP+lerp between nearest neighbors (held at the
  ends). The joint BA refines these from IMU + reprojection."""
  reg = sorted(cam_poses)
  out: list[gtsam.Pose3] = []
  for k in range(n_keyframes):
    if k in cam_poses:
      out.append(cam_poses[k])
      continue
    after = bisect_left(reg, k)
    lo = reg[after - 1] if after else None
    hi = reg[after] if after < len(reg) else None
    if lo is None:
      out.append(cam_poses[hi])  # type: ignore[index]
    elif hi is None:
      out.append(cam_poses[lo])
    else:
      pa, pb = cam_poses[lo], cam_poses[hi]
      f = (k - lo) / (hi - lo)
      out.append(pa.compose(gtsam.Pose3.Expmap(f * gtsam.Pose3.Logmap(pa.between(pb)))))
  return out


def _anchor_pose_seed(
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]],
  front_extrinsic: gtsam.Pose3,
  n_keyframes: int,
) -> list[gtsam.Pose3] | None:
  """Metric body-pose seed from the front-lens anchor PnP: world (= the anchor
  tag frame) ``T_body`` at every keyframe that sees the anchor, with gaps
  geodesic-filled. Returns ``None`` when too few keyframes see the anchor to
  seed a trajectory."""
  cam_poses = {
    kf: sides["front"].inverse()
    for kf, sides in anchor_obs_per_kf.items()
    if "front" in sides
  }
  if not _enough_registered(len(cam_poses), n_keyframes):
    return None
  return body_poses_from_camera_poses(
    _fill_camera_poses(cam_poses, n_keyframes), front_extrinsic
  )


def _anchor_metric_scale(
  body_poses_unscaled: list[gtsam.Pose3],
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]],
  front_lens: LensSetup,
  rear_lens: LensSetup,
) -> float | None:
  """Metric scale from the anchor: the ratio of the anchor-implied body-position
  spread (from each keyframe's ``cam_T_tag``) to the unscaled visual-odometry
  spread, over the anchored keyframes. ``None`` if fewer than three anchored
  keyframes or the visual spread is degenerate."""
  vo: list[Float64Array] = []
  anc: list[Float64Array] = []
  for kf, sides in anchor_obs_per_kf.items():
    lens, cam_T_tag = _side_lens_and_pose(sides, front_lens, rear_lens)
    body_in_tag = lens.extrinsic.compose(cam_T_tag).inverse()
    anc.append(_translation(body_in_tag))
    vo.append(_translation(body_poses_unscaled[kf]))
  if len(vo) < 3:
    return None
  vo_arr = np.asarray(vo, dtype=np.float64)
  anc_arr = np.asarray(anc, dtype=np.float64)
  vo_spread = float(np.sqrt(((vo_arr - vo_arr.mean(axis=0)) ** 2).sum()))
  anc_spread = float(np.sqrt(((anc_arr - anc_arr.mean(axis=0)) ** 2).sum()))
  if vo_spread < 1e-9:
    return None
  return anc_spread / vo_spread


def _side_lens_and_pose(
  sides: dict[str, gtsam.Pose3], front_lens: LensSetup, rear_lens: LensSetup
) -> tuple[LensSetup, gtsam.Pose3]:
  """Pick the anchor observation's lens + ``cam_T_tag``, preferring the front
  lens when both sides saw the tag. ``sides`` always carries at least one."""
  if "front" in sides:
    return front_lens, sides["front"]
  return rear_lens, sides["rear"]


def _imu_and_initial_gravity(
  imu: ImuSamples,
  raw_factory_bias: gtsam.imuBias.ConstantBias,
  imu_to_body: Float64Array,
  keyframe_times: Float64Array,
  body_poses_prealign: list[gtsam.Pose3],
) -> tuple[ImuSamples, gtsam.imuBias.ConstantBias, Float64Array | None]:
  """Rescaled IMU samples, the body-frame factory bias, and an initial world
  gravity.

  Rotates the factory bias into the body frame. Where a stationary window
  exists, rescales the accelerometer so its magnitude there matches
  |g| = 9.81 m/s², then reads gravity off the mean stationary acceleration and
  rotates it into the world frame via the nearest keyframe pose. Returns
  ``gravity_world_initial=None`` when the clip has no stationary window.
  """
  bias0 = gtsam.imuBias.ConstantBias(
    imu_to_body @ np.asarray(raw_factory_bias.accelerometer(), dtype=np.float64),
    imu_to_body @ np.asarray(raw_factory_bias.gyroscope(), dtype=np.float64),
  )
  windows = find_stationary_windows(imu, bias0)
  gravity_world_initial: Float64Array | None = None
  if windows:
    first_start, first_end = windows[0]
    stationary_mean = (imu.accel[first_start:first_end] - bias0.accelerometer()).mean(
      axis=0
    )
    stationary_mag = float(np.linalg.norm(stationary_mean))
    if 8.0 < stationary_mag < 11.0 and abs(stationary_mag - GRAVITY) > 0.05:
      accel_scale = GRAVITY / stationary_mag
      imu = ImuSamples(
        t=imu.t, gyro=imu.gyro, accel=(imu.accel * accel_scale).astype(np.float64)
      )
      stationary_mean = stationary_mean * accel_scale
    gravity_body = -stationary_mean / float(np.linalg.norm(stationary_mean)) * GRAVITY
    t_stationary_mid = 0.5 * (imu.t[first_start] + imu.t[first_end - 1])
    kf_for_g = int(np.argmin(np.abs(keyframe_times - t_stationary_mid)))
    gravity_world_initial = (
      body_poses_prealign[kf_for_g].rotation().matrix() @ gravity_body
    )
  return imu, bias0, gravity_world_initial


def _recover_gripper_tags(
  args: TrackerArgs,
  rear_tags_per_kf: list[list[TagDetection]],
  rear_lens: LensSetup,
  n_keyframes: int,
  kf_rotations: list[Float64Array],
  rear_tags_per_frame: list[list[TagDetection]] | None = None,
  frame_times: Float64Array | None = None,
) -> tuple[
  gtsam.Pose3 | None,
  dict[str, list[float]],
  dict[str, FingerReproj],
  dict[str, tuple[Float64Array, Float64Array]],
]:
  """Base-tag body pose, per-finger joint angles, and finger reprojection inputs.

  ``body_T_base_tag`` is the mean of the base tag's body-frame pose
  (``rear_lens.extrinsic ∘ cam_T_tag``) over the rear keyframes that see it.
  ``joint_angles`` holds each configured finger's per-keyframe URDF
  revolute-joint angle from PnP (the viz seed and the bundle's per-keyframe
  fallback). ``finger_reproj`` carries the corner-reprojection inputs the joint
  bundle refines the angles from, built from the URDF when one is given.
  ``frame_angles`` holds the same joints' per-frame PnP angles on the finger
  detection stride, for the finger transform streams.
  """
  base_tag_obs = anchor_pose_from_tags(
    rear_tags_per_kf,
    tag_id=args.base_tag_id,
    tag_size=args.gripper.tag_size,
    params=rear_lens.params,
    kf_rotations=kf_rotations,
  )
  body_T_base_tag: gtsam.Pose3 | None = None
  if base_tag_obs:
    body_T_base_tag = _mean_pose(
      [rear_lens.extrinsic.compose(p) for p in base_tag_obs.values()]
    )
  if args.base_tag_id > 0 and body_T_base_tag is None:
    raise ValueError(
      f"base tag {args.base_tag_id} was configured but never detected in the rear "
      f"lens across {n_keyframes} keyframes — its base-tag, finger, and cross-tracker "
      f"factors would all be silently disabled. Wrong id, or the tag out of the "
      f"rear-lens field of view?"
    )

  # The URDF is authored in the base (URDF root) frame; the bundle's
  # body_T_base_tag carries each finger joint to the body frame.
  joints = read_revolute_joints(args.urdf_path) if args.urdf_path is not None else {}
  joint_angles: dict[str, list[float]] = {}
  finger_reproj: dict[str, FingerReproj] = {}
  frame_angles: dict[str, tuple[Float64Array, Float64Array]] = {}
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
      n_keyframes=n_keyframes,
      kf_rotations=kf_rotations,
    )
    if args.urdf_path is None:
      continue
    joint = joints.get(joint_name)
    if joint is None or joint.joint_T_tag0 is None:
      raise ValueError(
        f"finger tag {finger_id} was configured for '{joint_name}', but "
        f"{args.urdf_path} has no revolute joint '{joint_name}' carrying an "
        f"AprilTag frame (revolute joints present: {sorted(joints)}). Wrong URDF?"
      )
    if body_T_base_tag is None:
      continue
    reproj = build_finger_reproj(
      joint,
      rear_tags_per_kf,
      rear_lens,
      finger_tag_id=finger_id,
      body_T_base_tag=body_T_base_tag,
      tag_size=args.gripper.tag_size,
      n_keyframes=n_keyframes,
    )
    if reproj is None:
      print(
        f"  [warn] finger tag {finger_id} ({joint_name}) never detected in the "
        f"rear lens; its angle stays PnP-only, unrefined by the bundle"
      )
    else:
      finger_reproj[joint_name] = reproj
    if rear_tags_per_frame is None or frame_times is None:
      continue
    indices, angles = frame_finger_angles(
      joint,
      rear_tags_per_frame,
      rear_lens,
      finger_tag_id=finger_id,
      body_T_base_tag=body_T_base_tag,
      tag_size=args.gripper.tag_size,
    )
    if len(indices):
      frame_angles[joint_name] = (frame_times[indices], angles)
  return body_T_base_tag, joint_angles, finger_reproj, frame_angles


@dataclass
class _LensChannel:
  """One lens's prep state: its per-keyframe tag detections in, the surviving
  tracks and their triangulated landmarks out."""

  setup: LensSetup
  tags: list[list[TagDetection]]
  feature_frames: list[Frame] = field(default_factory=list)
  kept: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  rejected: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  landmarks_prealign: dict[int, Float64Array] = field(default_factory=dict)
  points: dict[int, Float64Array] = field(default_factory=dict)
  tracks: dict[int, dict[int, Float64Array]] = field(default_factory=dict)


def _seed_body_poses(
  name: str,
  front: _LensChannel,
  rear: _LensChannel,
  n_keyframes: int,
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]],
  cfg: FrontendCfg,
) -> tuple[list[gtsam.Pose3], bool]:
  """The pre-alignment body-pose seed, and whether it came from the anchor.

  A two-view essential solve on the widest init pair either lens offers gives a
  straight line through the keyframes. The visual-odometry front-end then
  recovers the real trajectory shape from the front lens (which owns rotation
  and shape) and replaces it, unless too few keyframes register. In that case
  the anchor-PnP poses seed instead: they are metric (known tag size) and
  correctly shaped wherever the tag is visible. The straight line is what is
  left when the anchor is not visible either."""
  sfm_cams, _ = incremental_sfm(
    _bearings_by_track(front.kept, front.setup.params),
    min_shared=cfg.init_min_shared_tracks,
  )
  sfm_cams, vo_outliers = _reject_vo_pose_outliers(sfm_cams)
  if vo_outliers:
    print(f"  [{name}] visual-odometry rejected {len(vo_outliers)} rotation-spike kfs")
  if _enough_registered(len(sfm_cams), n_keyframes, cfg.vo_min_registered_ratio):
    print(
      f"  [{name}] visual-odometry init: registered "
      f"{len(sfm_cams)}/{n_keyframes} keyframes"
    )
    filled = _fill_camera_poses(sfm_cams, n_keyframes)
    return body_poses_from_camera_poses(filled, front.setup.extrinsic), False

  anchor_seed = _anchor_pose_seed(anchor_obs_per_kf, front.setup.extrinsic, n_keyframes)
  if anchor_seed is not None:
    n_anchor_kf = sum("front" in sides for sides in anchor_obs_per_kf.values())
    print(
      f"  [{name}] visual-odometry starved ({len(sfm_cams)}/{n_keyframes}); "
      f"anchor-pose init from {n_anchor_kf} anchor keyframes"
    )
    return anchor_seed, True
  print(
    f"  [{name}] visual-odometry init: only {len(sfm_cams)}/{n_keyframes} "
    f"keyframes registered — using straight-line init"
  )
  init_ch, (init_first, init_last, init_shared) = max(
    (
      (ch, pick_init_pair(ch.kept, n_keyframes, cfg.init_min_shared_tracks))
      for ch in (front, rear)
    ),
    key=lambda candidate: candidate[1][2],
  )
  if init_shared < 8:
    raise RuntimeError(f"{name}: no lens has ≥8 shared tracks (best={init_shared})")
  if init_first:
    print(f"  [{name}] init pair based at keyframe {init_first}", flush=True)
  shared = [
    obs for obs in init_ch.kept.values() if init_first in obs and init_last in obs
  ]
  pix_a = np.asarray([obs[init_first] for obs in shared], dtype=np.float64)
  pix_b = np.asarray([obs[init_last] for obs in shared], dtype=np.float64)
  r_cw_last, t_cw_last, _ = two_view_essential(
    mei_unproject(pix_a, init_ch.setup.params),
    mei_unproject(pix_b, init_ch.setup.params),
  )
  cam_poses = interpolate_camera_poses(
    r_cw_last, t_cw_last, n_keyframes, init_first=init_first, init_last=init_last
  )
  return body_poses_from_camera_poses(cam_poses, init_ch.setup.extrinsic), False


def _align_visual_inertial(
  name: str,
  imu: ImuSamples,
  keyframe_times: Float64Array,
  bias0: gtsam.imuBias.ConstantBias,
  body_poses_prealign: list[gtsam.Pose3],
  gravity_world_initial: Float64Array | None,
  fixed_scale: float | None,
) -> tuple[
  list[gtsam.Pose3],
  list[Float64Array],
  list[gtsam.PreintegratedCombinedMeasurements],
  Float64Array,
]:
  """Metric gravity-aligned body poses, their velocities, the keyframe
  preintegration, and the 3x3 map (scale · rotation) that carries prep-frame
  points into the same world.

  The alignment runs against a preintegration that already carries the IMU
  stationary-window gravity; without such a window a placeholder vertical
  gravity stands in and the alignment's own least-squares fit sticks. A
  negative recovered scale means the two-view init reflected the trajectory,
  so it is reflected back and re-aligned."""

  def pim_params(
    gravity: tuple[float, float, float],
  ) -> gtsam.PreintegrationCombinedParams:
    return make_preintegration_params(gravity_world=gravity)

  # Gravity enters prediction and the factor, never the preintegrated deltas,
  # so one preintegration serves both the alignment and the export.
  pim = preintegrate_between_keyframes(
    pim_params(GRAVITY_WORLD), imu, keyframe_times, bias0
  )

  def align(poses: list[gtsam.Pose3]) -> Alignment:
    return visual_inertial_align(
      poses, pim, fixed_gravity=gravity_world_initial, fixed_scale=fixed_scale
    )

  alignment = align(body_poses_prealign)
  if alignment.scale < 0.0:
    origin = _translation(body_poses_prealign[0])
    body_poses_prealign = [
      gtsam.Pose3(p.rotation(), gtsam.Point3(*(2.0 * origin - _translation(p))))
      for p in body_poses_prealign
    ]
    alignment = align(body_poses_prealign)
  # The least-squares gravity can sit ~30° off the IMU truth.
  if gravity_world_initial is not None:
    alignment = Alignment(
      scale=alignment.scale,
      gravity_world=gravity_world_initial,
      velocities=alignment.velocities,
    )
  print(f"  [{name}] VI scale={alignment.scale:.4f}")
  body_poses, world_rotation = apply_alignment_to_poses(body_poses_prealign, alignment)
  world_rot = world_rotation.matrix()
  velocities = [world_rot @ v for v in alignment.velocities]
  return body_poses, velocities, pim, alignment.scale * world_rot


def captures_that_fit(session: SessionRecording, names: Sequence[str]) -> int:
  """How many captures may prep at once before their decoded frames exceed the
  memory on hand. A capture decodes one band of keyframes per lens at a time;
  the encoded samples those bands decode from are shared by every capture."""
  if len(names) <= 1:
    return len(names)
  height, width = session.decoded_shape(names[0])
  per_capture = PREP_BAND * width * height  # one lens's resident band at a time
  shared = session.path.stat().st_size
  budget = available_ram() * _PREP_MEMORY_SHARE - shared
  affordable = int(budget / per_capture)
  return max(1, min(len(names), affordable))


def keyframe_grid(
  times: Float64Array,
  covered: npt.NDArray[np.int64],
  fps: float,
  keyframe_every: int,
) -> npt.NDArray[np.int64]:
  """Keyframe rows over the covered span, phase snapped to the session-clock
  stride lattice the review timeline buckets by. When no covered row sits on
  the lattice the grid simply starts at the first covered row."""
  stride = schema.keyframe_stride(fps)
  session_rows = np.round(times * fps).astype(np.int64)
  on_lattice = np.flatnonzero(session_rows[covered] % stride == 0)
  phase = int(on_lattice[0]) if len(on_lattice) else 0
  return covered[phase::keyframe_every]


def finger_stride(keyframe_every: int, fps: float) -> int:
  """Frame stride for the ~30 Hz finger series: the largest divisor of the
  keyframe interval at or under ``fps / 30``, so the keyframe subset of the
  dense walk stays exact at any capture rate."""
  return max(
    (
      d
      for d in range(1, keyframe_every + 1)
      if keyframe_every % d == 0 and d <= fps / 30.0
    ),
    default=1,
  )


def prepare_tracker(name: str, args: TrackerArgs) -> PreparedTracker:
  """Run every per-tracker stage. Returns a ``PreparedTracker`` ready for
  joint-graph composition. Body poses are in this tracker's own gravity-
  aligned visual world; the caller aligns trackers to a shared frame via
  ``anchor_locals_by_tag``."""
  configure_detector(args.frontend.parallel_tag_detect and args.parallel_siblings)
  session = SessionRecording(args.base_rrd)
  front_video = session.lens_video(args.tracker, "front")
  rear_video = session.lens_video(args.tracker, "rear")
  fps = fps_from_times(front_video.times)

  keyframe_every = resolve_keyframe_every(args.keyframe_every, fps)
  raw_imu = session.imu(args.tracker)
  covered = covered_rows(front_video.times, rear_video.times, raw_imu.t, fps)
  front_rows = keyframe_grid(front_video.times, covered, fps, keyframe_every)
  grid_times = front_video.times[front_rows]
  rear_kf_rows = nearest_rows(rear_video.times, grid_times)

  anchor_tag_ids = set(args.anchor.tag_ids)

  def tag_masks(
    shape: tuple[int, int], tags_per_kf: list[list[TagDetection]]
  ) -> list[list[Rect]]:
    """Per keyframe, the rectangles blocking every non-anchor tag; empty lists
    when tag masking is off."""
    if not args.frontend.mask_tags:
      return [[] for _ in tags_per_kf]
    return _build_non_anchor_tag_masks(
      shape, tags_per_kf, anchor_tag_ids, args.frontend.mask_tag_pad
    )

  def walk_front() -> tuple[int, list[list[TagDetection]], list[Frame]]:
    """One banded walk detecting tags and loop-closure ORB together. With loop
    closure off the ORB list comes back empty."""
    walk = RecordingKeyframes(front_video, front_rows, band=PREP_BAND)
    width = int(walk[0].shape[1])
    if not args.frontend.loop_closure:
      return width, _per_band(detect_all_tags, walk), []

    def band_fn(
      band: list[np.ndarray],
    ) -> list[tuple[list[TagDetection], Frame]]:
      tags = detect_all_tags(band)
      rects = tag_masks(band[0].shape[:2], tags)
      orb = parallel_map(
        lambda f, m: detect_orb(
          f, args.frontend.orb_n_features, 20, mask_plane(f.shape[:2], m)
        ),
        band,
        rects,
      )
      return list(zip(tags, orb, strict=True))

    fused = _per_band(band_fn, walk)
    return width, [tags for tags, _ in fused], [orb for _, orb in fused]

  decoded_width, front_tags_per_kf, front_orb = walk_front()
  front_lens, rear_lens = session.lens_setups(args.tracker, decoded_width)
  release_freed_memory()
  # Rear: with fingers configured, detect at every frame in one dense walk; the
  # keyframe subset feeds the existing machinery and the full series carries the
  # finger angles. Without fingers the dense detections would be pure waste.
  wants_fingers = args.urdf_path is not None and (
    args.left_finger_tag_id > 0 or args.right_finger_tag_id > 0
  )
  rear_tags_per_frame: list[list[TagDetection]] | None = None
  rear_frame_times: Float64Array | None = None
  if wants_fingers:
    stride = finger_stride(keyframe_every, fps)
    dense_rows = nearest_rows(rear_video.times, front_video.times[covered][::stride])
    rear_tags_per_frame = _per_band(
      detect_all_tags, RecordingKeyframes(rear_video, dense_rows, band=PREP_BAND)
    )
    rear_frame_times = rear_video.times[dense_rows]
    rear_tags_per_kf = rear_tags_per_frame[:: keyframe_every // stride]
  else:
    rear_tags_per_kf = _per_band(
      detect_all_tags, RecordingKeyframes(rear_video, rear_kf_rows, band=PREP_BAND)
    )
  n_keyframes = min(len(front_tags_per_kf), len(rear_tags_per_kf))
  release_freed_memory()
  if n_keyframes < 3:
    raise RuntimeError(f"only {n_keyframes} keyframes decoded — window too short")
  front_tags_per_kf = front_tags_per_kf[:n_keyframes]
  rear_tags_per_kf = rear_tags_per_kf[:n_keyframes]
  keyframe_times = grid_times[:n_keyframes]

  imu_to_body = session.imu_to_body(args.tracker)
  # The recording's raw-axis samples rotated into the body frame; gyro is
  # already rad/s.
  imu = ImuSamples(
    t=raw_imu.t,
    gyro=raw_imu.gyro @ imu_to_body.T,
    accel=raw_imu.accel @ imu_to_body.T,
  )
  body_deltas = keyframe_delta_rotations(imu, keyframe_times)
  del raw_imu

  def lens_kf_rotations(lens: LensSetup) -> list[Float64Array]:
    r_bl = lens.extrinsic.rotation().matrix()
    return [r_bl.T @ d.T @ r_bl for d in body_deltas]

  front_kf_rotations = lens_kf_rotations(front_lens)
  rear_kf_rotations = lens_kf_rotations(rear_lens)

  lens_sides = (
    ("front", front_lens, front_tags_per_kf, front_kf_rotations),
    ("rear", rear_lens, rear_tags_per_kf, rear_kf_rotations),
  )
  obs_by_tag = {
    tag_id: _detect_anchor_obs(tag_id, args.anchor.tag_size, lens_sides)
    for tag_id in args.anchor.enabled_tag_ids
  }
  prep_anchor_tag_id = _choose_prep_anchor(args.anchor.tag_ids, obs_by_tag)
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]] = obs_by_tag.get(
    prep_anchor_tag_id, {}
  )
  if prep_anchor_tag_id != args.anchor.tag_ids[0]:
    n_ref = len(obs_by_tag.get(args.anchor.tag_ids[0], {}))
    print(
      f"  [{name}] prep reference anchor: tag {prep_anchor_tag_id} "
      f"({len(anchor_obs_per_kf)} sighted keyframes vs {n_ref} for "
      f"tag {args.anchor.tag_ids[0]})"
    )
  for tag_id, obs in obs_by_tag.items():
    _print_anchor_geometry(name, tag_id, lens_sides, obs)

  front = _LensChannel(setup=front_lens, tags=front_tags_per_kf)
  rear = _LensChannel(setup=rear_lens, tags=rear_tags_per_kf)
  anchor_tag_ids = set(obs_by_tag)
  # The rects are applied inside the rear lens's frontend, so a run that does
  # not build rear landmarks never masks anything and has none to record.
  static_rects = (
    list(args.frontend.mask_rear_rects) if args.frontend.rear_landmarks else []
  )

  def run_lens(
    ch: _LensChannel,
    video: LensVideo,
    rows: npt.NDArray[np.int64],
    blocked: list[Rect],
  ) -> None:
    """Decode this lens, take everything downstream needs from its pixels, and
    drop them again before the other lens is touched. ``blocked`` are the
    always-masked rectangles of this lens's own frame."""
    frames = RecordingKeyframes(video, rows, band=PREP_BAND)
    shape = frames[0].shape[:2]
    masks = (
      _build_non_anchor_tag_masks(
        shape, ch.tags, anchor_tag_ids, args.frontend.mask_tag_pad
      )
      if args.frontend.mask_tags
      else [[] for _ in ch.tags]
    )
    _apply_static_mask_rects(masks, blocked, shape)
    ch.feature_frames, tracks = run_lens_frontend(
      frames, ch.setup.params, args.frontend, masks=masks
    )
    ch.kept = {
      t: o for t, o in tracks.items() if len(o) >= args.frontend.min_track_length
    }
    if args.frontend.min_parallax > 0.0:
      ch.kept, ch.rejected = filter_tracks_by_parallax(
        ch.kept, ch.setup.params, args.frontend.min_parallax
      )
    release_freed_memory()

  run_lens(front, front_video, front_rows, [])
  if args.frontend.rear_landmarks:
    run_lens(rear, rear_video, rear_kf_rows, static_rects)

  body_poses_prealign, anchor_seed_used = _seed_body_poses(
    name, front, rear, n_keyframes, anchor_obs_per_kf, args.frontend
  )

  imu, bias0, gravity_world_initial = _imu_and_initial_gravity(
    imu,
    session.factory_bias(args.tracker),
    imu_to_body,
    keyframe_times,
    body_poses_prealign,
  )

  for ch in (front, rear):
    ch.landmarks_prealign = triangulate_lens(
      ch.kept,
      body_poses_prealign,
      ch.setup.extrinsic,
      ch.setup.params,
      args.frontend.min_track_length,
      min_landmark_depth=0.01,
      max_landmark_depth=1000.0,
    )

  # An anchor seed is already metric, so its scale is 1.0; otherwise take the
  # scale from a visible anchor, and float it on the IMU only with no anchor.
  anchor_scale = _anchor_metric_scale(
    body_poses_prealign, anchor_obs_per_kf, front_lens, rear_lens
  )
  body_poses, velocities, pim, prealign_to_world = _align_visual_inertial(
    name,
    imu,
    keyframe_times,
    bias0,
    body_poses_prealign,
    gravity_world_initial,
    fixed_scale=1.0 if anchor_seed_used else anchor_scale,
  )

  # Each sighted anchor's pose in this tracker's local world, for the
  # shared-world alignment BFS.
  anchor_locals_by_tag: dict[int, gtsam.Pose3] = {
    tag_id: _first_sighting_local(obs, front_lens, rear_lens, body_poses)
    for tag_id, obs in obs_by_tag.items()
    if obs
  }

  if anchor_obs_per_kf:
    loop_tracks: dict[int, dict[int, Float64Array]] = {}
    if args.frontend.loop_closure:
      loop_tracks = _loop_closure_tracks(
        front_orb[:n_keyframes],
        body_poses,
        front_lens,
        max(front.kept, default=0),
        args.frontend.loop_closure_max_dist,
        args.frontend.loop_closure_min_kf_gap,
        args.frontend.loop_n_neighbors,
        args.frontend.loop_match_ratio,
        args.frontend.loop_min_inliers,
      )
    for ch in (front, rear):
      ch.points = triangulate_lens(
        {t: ch.kept[t] for t in ch.kept if t in ch.landmarks_prealign},
        body_poses,
        ch.setup.extrinsic,
        ch.setup.params,
        args.frontend.min_track_length,
        max_landmark_depth=args.frontend.max_landmark_depth,
      )
      ch.points = _cap_landmarks(
        ch.points, ch.kept, args.frontend.max_landmarks_per_lens
      )
    if loop_tracks:
      # Merged after the cap so the drift-cancelling revisit landmarks are never
      # culled by the per-lens landmark budget.
      _add_loop_landmarks(
        front.points,
        front.kept,
        loop_tracks,
        body_poses,
        front_lens,
        min_track_length=args.frontend.min_track_length,
        max_landmark_depth=args.frontend.max_landmark_depth,
        name=name,
        anchor_kfs=set(anchor_obs_per_kf)
        if args.frontend.anchored_loop_closure
        else None,
      )
  else:
    for ch in (front, rear):
      ch.points = {
        tid: prealign_to_world @ xyz for tid, xyz in ch.landmarks_prealign.items()
      }

  for ch in (front, rear):
    ch.tracks = {
      t: _subsample_obs(ch.kept[t], args.frontend.max_obs_per_track)
      for t in ch.points
      if t in ch.kept
    }

  v_max = float(max(np.linalg.norm(v) for v in velocities))
  print(f"  [{name}] VIO LSQ |v|max={v_max:.2f} m/s")
  biases = [bias0 for _ in range(n_keyframes)]

  body_T_base_tag, joint_angles, finger_reproj, finger_frame_angles = (
    _recover_gripper_tags(
      args,
      rear_tags_per_kf,
      rear_lens,
      n_keyframes,
      rear_kf_rotations,
      rear_tags_per_frame,
      rear_frame_times,
    )
  )

  return PreparedTracker(
    name=name,
    body_poses=body_poses,
    velocities=velocities,
    biases=biases,
    preintegrated=pim,
    front_lens=front_lens,
    rear_lens=rear_lens,
    front_tracks=front.tracks,
    rear_tracks=rear.tracks,
    front_points=front.points,
    rear_points=rear.points,
    anchor_obs_by_tag=obs_by_tag,
    anchor_locals_by_tag=anchor_locals_by_tag,
    body_T_base_tag=body_T_base_tag,
    joint_angles=joint_angles,
    keyframe_times=keyframe_times,
    keyframe_rows=np.round(keyframe_times * fps).astype(np.int64),
    front_keypoints=[f.keypoints for f in front.feature_frames][:n_keyframes],
    rear_keypoints=[f.keypoints for f in rear.feature_frames][:n_keyframes],
    front_tags_per_kf=front_tags_per_kf,
    rear_tags_per_kf=rear_tags_per_kf,
    front_rejected=front.rejected,
    rear_rejected=rear.rejected,
    rear_static_mask_rects=static_rects,
    finger_reproj=finger_reproj,
    left_finger_tag_id=args.left_finger_tag_id,
    right_finger_tag_id=args.right_finger_tag_id,
    imu=imu,
    finger_frame_angles=finger_frame_angles or None,
  )
