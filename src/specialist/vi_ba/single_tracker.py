"""Per-tracker prep for VI-BA: decode → frontend → tag detect → anchor PnP →
IMU stationary gravity → VI alignment → visual-odometry trajectory (anchor-init
pose seed as the starved fallback) → metric re-triangulation. Returns a
``PreparedTracker`` ready for ``specialist.vi_ba.compose`` to assemble into a
joint factor graph.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, TypeVar

import cv2
import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import MeiParams, mei_unproject
from specialist.loader.parse_insv import InsvData, load_insv
from specialist.vi_ba.anchor_detection import (
  TagDetection,
  anchor_pose_from_tags,
  detect_all_tags,
)
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.configs import (
  AnchorCfg,
  FrontendCfg,
  GripperCfg,
  ImuCfg,
)
from specialist.vi_ba.extrinsics import parse_insv_extrinsics
from specialist.vi_ba.features import (
  Frame,
  Rect,
  detect_orb,
  mask_plane,
  match_pair,
  parallel_map,
)
from specialist.vi_ba.finger_kinematics import FingerReproj, build_finger_reproj
from specialist.vi_ba.imu import (
  ImuSamples,
  factory_imu_bias,
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
from specialist.vi_ba.pipeline import (
  BandedKeyframes,
  LensFrontendConfig,
  decode_lens_keyframes,
  filter_tracks_by_parallax,
  keyframe_timestamps,
  lens_setups,
  probe_capture,
  release_freed_memory,
  resolve_keyframe_every,
  run_lens_frontend,
  shared_keyframe_bound,
  triangulate_lens,
  unproject_tracks,
)
from specialist.vi_ba.urdf_joints import read_revolute_joints

Float64Array = npt.NDArray[np.float64]

_A = TypeVar("_A")
_B = TypeVar("_B")

# ORB loop-closure track recovery (``_loop_closure_tracks``).
_LOOP_N_NEIGHBORS = 4  # spatially-closest non-adjacent revisit kfs to re-match
_LOOP_MATCH_RATIO = 0.7  # Lowe ratio-test threshold
_LOOP_RANSAC_PX = 4.5  # homography RANSAC reprojection tolerance
_LOOP_MIN_VIEWS = 3  # keyframes a union-found track must span to be kept
_LOOP_MIN_INLIERS = 20  # geometrically-verified matches required per pair

# Visual-odometry pose-spike rejection (``_reject_vo_pose_outliers``).
_VO_REJECT_FACTOR = 8.0  # residual/median-step ratio marking a PnP teleport

# Keyframes read at once from a banded lens.
_PREP_BAND = 256


__all__ = [
  "PreparedTracker",
  "TrackerArgs",
  "prepare_tracker",
  "run_two_concurrently",
]


def _enough_registered(n_registered: int, n_keyframes: int) -> bool:
  """A seed needs a quarter of the keyframes, and at least eight."""
  return n_registered >= max(8, n_keyframes // 4)


def run_two_concurrently(
  task_a: Callable[[], _A], task_b: Callable[[], _B]
) -> tuple[_A, _B]:
  """Run two independent CPU-bound thunks on separate threads, return ``(a, b)``.
  OpenCV (DIS flow, ORB, essential/homography RANSAC) and the AprilTag detector
  release the GIL during their native calls, so the front and rear lens stages
  run on separate cores. The result is independent of completion order, so it
  matches sequential execution."""
  with ThreadPoolExecutor(max_workers=2) as ex:
    fut_a = ex.submit(task_a)
    fut_b = ex.submit(task_b)
    return fut_a.result(), fut_b.result()


@dataclass
class TrackerArgs:
  """Per-tracker prep knobs. Composes nested configs from ``configs.py``."""

  insv_path: Path
  base_tag_id: int
  anchor: AnchorCfg
  start: float
  duration: float | None
  keyframe_every: int  # 0 = auto (round(fps/10)); see WindowCfg.keyframe_every
  # Retain decoded keyframes (as JPEG) for the Rerun frame overlay.
  keep_frames: bool
  urdf_path: Path | None  # gripper URDF for finger joint kinematics
  # 0 disables the corresponding URDF revolute joint angle recovery.
  left_finger_tag_id: int
  right_finger_tag_id: int
  frontend: FrontendCfg
  imu: ImuCfg
  gripper: GripperCfg


class ViGraphInputs(TypedDict):
  """One tracker's state and lens data, keyed to ``build_vi_graph``'s
  parameters so the call site can unpack it directly."""

  body_poses: list[gtsam.Pose3]
  velocities: list[Float64Array]
  biases: list[gtsam.imuBias.ConstantBias]
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements]
  front: LensSetup
  front_tracks: dict[int, dict[int, Float64Array]]
  front_points: dict[int, Float64Array]
  rear: LensSetup
  rear_tracks: dict[int, dict[int, Float64Array]]
  rear_points: dict[int, Float64Array]


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
  decoded_width: int
  # ``{tag_id: {kf: {side: cam_T_tag}}}``; ``anchor_locals_by_tag`` gives each
  # sighted tag's pose in this tracker's local world.
  anchor_obs_by_tag: dict[int, dict[int, dict[str, gtsam.Pose3]]]
  anchor_locals_by_tag: dict[int, gtsam.Pose3]
  body_T_base_tag: gtsam.Pose3 | None
  # URDF joint name → per-keyframe rotation about the URDF z-axis. NaN where
  # either tag is missing or PnP fails. Empty when no finger ids configured.
  joint_angles: dict[str, list[float]]
  keyframe_times: Float64Array
  # JPEG-encoded keyframes for the Rerun overlay (empty unless retained).
  front_frames: list[bytes]
  rear_frames: list[bytes]
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

  @property
  def n_keyframes(self) -> int:
    return len(self.body_poses)

  def vi_graph_inputs(self) -> ViGraphInputs:
    """Bundle this tracker's state + lens data for ``build_vi_graph``."""
    return ViGraphInputs(
      body_poses=self.body_poses,
      velocities=self.velocities,
      biases=self.biases,
      preintegrated=self.preintegrated,
      front=self.front_lens,
      front_tracks=self.front_tracks_tri,
      front_points=self.front_points,
      rear=self.rear_lens,
      rear_tracks=self.rear_tracks_tri,
      rear_points=self.rear_points,
    )


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
  """``{kf: {side: cam_T_tag}}`` for one anchor id across both lens sides."""
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
        corners = np.asarray(det.corners, dtype=np.float64)
        edge = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        edges.append(float(edge.min()))
        bearing = mei_unproject(corners.mean(axis=0).reshape(1, 2), lens.params)[0]
        norm = float(np.linalg.norm(bearing))
        if np.isfinite(norm) and norm > 1e-9:
          half_angles.append(
            float(np.degrees(np.arccos(np.clip(bearing[2] / norm, -1.0, 1.0))))
          )
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
    ][:_LOOP_N_NEIGHBORS]
    pairs.extend((i, j) for j in cand)

  def verify_pair(i: int, j: int) -> npt.NDArray[np.int64] | None:
    m = match_pair(orb[i], orb[j], _LOOP_MATCH_RATIO)
    if len(m) < _LOOP_MIN_INLIERS:
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
    if int(inl.sum()) < _LOOP_MIN_INLIERS:
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


def _anchor_pose_seed(
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]],
  front_extrinsic: gtsam.Pose3,
  n_keyframes: int,
) -> list[gtsam.Pose3] | None:
  """Metric body-pose seed from the front-lens anchor PnP: world (= the anchor
  tag frame) ``T_body`` at every keyframe that sees the anchor, with gaps
  SLERP+lerp-filled. The anchor poses are metric (known tag size) and correctly
  shaped wherever the tag is visible, so this is the seed of choice when the
  visual odometry is too starved to register; the joint BA then refines it from
  the anchor, IMU, and reprojection factors. Returns ``None`` when too few
  keyframes see the anchor to seed a trajectory."""
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
  front_extrinsic: gtsam.Pose3,
  rear_extrinsic: gtsam.Pose3,
) -> float | None:
  """Metric scale from the anchor: the ratio of the anchor-implied body-position
  spread (from each keyframe's ``cam_T_tag``) to the unscaled visual-odometry
  spread, over the anchored keyframes. ``None`` if fewer than three anchored
  keyframes or the visual spread is degenerate."""
  vo: list[Float64Array] = []
  anc: list[Float64Array] = []
  for kf, sides in anchor_obs_per_kf.items():
    if "front" in sides:
      ext, cam_T_tag = front_extrinsic, sides["front"]
    elif "rear" in sides:
      ext, cam_T_tag = rear_extrinsic, sides["rear"]
    else:
      continue
    body_in_tag = ext.compose(cam_T_tag).inverse()
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


def _body_imu_samples(data: InsvData, imu_to_body: Float64Array) -> ImuSamples:
  """Raw telemetry IMU rotated into the body frame (gyro deg/s → rad/s)."""
  return ImuSamples(
    t=np.asarray(data.imu_timestamps, dtype=np.float64),
    gyro=np.deg2rad(np.asarray(data.gyro_deg, dtype=np.float64)) @ imu_to_body.T,
    accel=np.asarray(data.accel, dtype=np.float64) @ imu_to_body.T,
  )


def _imu_and_initial_gravity(
  args: TrackerArgs,
  data: InsvData,
  imu: ImuSamples,
  keyframe_times: Float64Array,
  body_poses_prealign: list[gtsam.Pose3],
) -> tuple[ImuSamples, gtsam.imuBias.ConstantBias, Float64Array | None]:
  """Body-frame IMU samples, the factory bias, and an initial world gravity.

  Rotates the factory bias into the body frame. When the clip contains a
  stationary window, rescales the accelerometer so its magnitude there
  matches |g| = 9.81 m/s², then reads gravity off the mean stationary
  acceleration and rotates it into the world frame via the nearest keyframe
  pose. Returns ``gravity_world_initial=None`` when no stationary window
  falls inside the clip.
  """
  imu_to_body = np.asarray(args.imu.imu_to_body, dtype=np.float64)
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
) -> tuple[gtsam.Pose3 | None, dict[str, list[float]], dict[str, FingerReproj]]:
  """Base-tag body pose, per-finger joint angles, and finger reprojection inputs.

  ``body_T_base_tag`` is the mean of the base tag's body-frame pose
  (``rear_lens.extrinsic ∘ cam_T_tag``) over the rear keyframes that see it.
  ``joint_angles`` holds each configured finger's per-keyframe URDF
  revolute-joint angle from PnP (the viz seed and the bundle's per-keyframe
  fallback). ``finger_reproj`` carries the corner-reprojection inputs the joint
  bundle refines the angles from, built from the URDF when one is given.
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

  finger_joint_tags = (
    ("left_finger", args.left_finger_tag_id),
    ("right_finger", args.right_finger_tag_id),
  )
  joint_angles: dict[str, list[float]] = {}
  for joint_name, finger_id in finger_joint_tags:
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

  finger_reproj: dict[str, FingerReproj] = {}
  if args.urdf_path is not None:
    # The URDF is authored in the base (URDF root) frame; the bundle's
    # body_T_base_tag carries each finger joint to the body frame.
    joints = read_revolute_joints(args.urdf_path)
    for joint_name, finger_id in finger_joint_tags:
      if finger_id <= 0:
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
  return body_T_base_tag, joint_angles, finger_reproj


@dataclass
class _LensChannel:
  """One lens's prep state: decoded grays and tag detections in, tracks and
  triangulated landmarks out."""

  setup: LensSetup
  gray: Sequence[np.ndarray]
  tags: list[list[TagDetection]]
  masks: list[list[Rect]] = field(default_factory=list)
  orb: list[Frame] = field(default_factory=list)
  feature_frames: list[Frame] = field(default_factory=list)
  tracks: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  kept: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  rejected: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  landmarks_prealign: dict[int, Float64Array] = field(default_factory=dict)
  points: dict[int, Float64Array] = field(default_factory=dict)
  tracks_tri: dict[int, dict[int, Float64Array]] = field(default_factory=dict)


def prepare_tracker(name: str, args: TrackerArgs) -> PreparedTracker:
  """Run every per-tracker stage. Returns a ``PreparedTracker`` ready for
  joint-graph composition. Body poses are in this tracker's own gravity-
  aligned visual world; the caller aligns trackers to a shared frame via
  ``anchor_locals_by_tag``."""
  data = load_insv(args.insv_path)
  extrinsics = parse_insv_extrinsics(data)
  rear_path, fps, decoded_width = probe_capture(args.insv_path)
  front_lens, rear_lens = lens_setups(extrinsics, decoded_width)

  keyframe_every = resolve_keyframe_every(args.keyframe_every, fps)
  # One lens at a time: a lens's keyframes are the largest thing prep holds, and
  # tag detection is the only stage that needs both lenses before the frontend.
  bound = shared_keyframe_bound(args.insv_path, rear_path)

  def lens_frames(path: Path) -> Sequence[np.ndarray]:
    """This lens's keyframes. Banded unless the Rerun overlay wants the colour
    frames too, which only the eager decode produces."""
    if args.keep_frames:
      return decode_lens_keyframes(
        path, bound, fps, args.start, args.duration, keyframe_every, True
      )[0]
    return BandedKeyframes(path, bound, fps, args.start, args.duration, keyframe_every)

  def per_band(
    fn: Callable[[list[np.ndarray]], list[_A]], frames: Sequence[np.ndarray]
  ) -> list[_A]:
    """Map over the frames a band at a time so a banded sequence is read in
    order; mapping across the whole lens would reload a band per frame."""
    out: list[_A] = []
    for begin in range(0, len(frames), _PREP_BAND):
      end = min(begin + _PREP_BAND, len(frames))
      out.extend(fn([frames[i] for i in range(begin, end)]))
    return out

  front_color: list[bytes] = []
  rear_color: list[bytes] = []
  if args.keep_frames:
    _, front_color = decode_lens_keyframes(
      args.insv_path, bound, fps, args.start, args.duration, keyframe_every, True
    )
    _, rear_color = decode_lens_keyframes(
      rear_path, bound, fps, args.start, args.duration, keyframe_every, True
    )
  front_tags_per_kf = per_band(detect_all_tags, lens_frames(args.insv_path))
  release_freed_memory()
  rear_tags_per_kf = per_band(detect_all_tags, lens_frames(rear_path))
  n_keyframes = min(len(front_tags_per_kf), len(rear_tags_per_kf))
  release_freed_memory()
  if n_keyframes < 3:
    raise RuntimeError(f"only {n_keyframes} keyframes decoded — window too short")
  front_tags_per_kf = front_tags_per_kf[:n_keyframes]
  rear_tags_per_kf = rear_tags_per_kf[:n_keyframes]
  front_color = front_color[:n_keyframes]
  rear_color = rear_color[:n_keyframes]
  keyframe_times = keyframe_timestamps(args.start, fps, keyframe_every, n_keyframes)

  imu_to_body = np.asarray(args.imu.imu_to_body, dtype=np.float64)
  imu = _body_imu_samples(data, imu_to_body)
  body_deltas = keyframe_delta_rotations(imu, keyframe_times)

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

  front = _LensChannel(setup=front_lens, gray=[], tags=front_tags_per_kf)
  rear = _LensChannel(setup=rear_lens, gray=[], tags=rear_tags_per_kf)
  anchor_tag_ids = set(obs_by_tag)
  static_rects = list(args.frontend.mask_rear_rects)

  cfg = LensFrontendConfig(
    n_features=args.frontend.orb_n_features,
    ransac_threshold=args.frontend.ransac_threshold,
    max_half_angle=args.frontend.max_half_angle,
    gpu_flow=args.frontend.gpu_flow,
  )

  def run_lens(ch: _LensChannel, path: Path) -> None:
    """Decode this lens, take everything downstream needs from its pixels, and
    drop them again before the other lens is touched."""
    ch.gray = lens_frames(path)
    shape = ch.gray[0].shape[:2]
    ch.masks = _build_non_anchor_tag_masks(
      shape, ch.tags, anchor_tag_ids, args.frontend.mask_tag_pad
    )
    if ch is rear:
      _apply_static_mask_rects(ch.masks, static_rects, shape)
    ch.feature_frames, ch.tracks = run_lens_frontend(
      ch.gray, ch.setup.params, cfg, masks=ch.masks
    )
    if ch is front and args.frontend.loop_closure:
      orb: list[Frame] = []
      for begin in range(0, n_keyframes, _PREP_BAND):
        end = min(begin + _PREP_BAND, n_keyframes)
        orb.extend(
          parallel_map(
            lambda f, m: detect_orb(
              f, args.frontend.orb_n_features, 20, mask_plane(f.shape[:2], m)
            ),
            [ch.gray[i] for i in range(begin, end)],
            ch.masks[begin:end],
          )
        )
      ch.orb = orb
    ch.gray = []
    ch.masks = []
    release_freed_memory()

  run_lens(front, args.insv_path)
  run_lens(rear, rear_path)
  for ch in (front, rear):
    ch.kept = {
      t: o for t, o in ch.tracks.items() if len(o) >= args.frontend.min_track_length
    }
    if args.frontend.min_parallax > 0.0:
      ch.kept, ch.rejected = filter_tracks_by_parallax(
        ch.kept, ch.setup.params, args.frontend.min_parallax
      )

  f_last, f_shared = pick_init_pair(
    front.kept, n_keyframes, args.frontend.init_min_shared_tracks
  )
  r_last, r_shared = pick_init_pair(
    rear.kept, n_keyframes, args.frontend.init_min_shared_tracks
  )
  init_tracks, init_lens, init_last, init_shared = (
    (front.kept, front_lens, f_last, f_shared)
    if f_shared >= r_shared
    else (rear.kept, rear_lens, r_last, r_shared)
  )
  if init_shared < 8:
    raise RuntimeError(f"{name}: no lens has ≥8 shared tracks (best={init_shared})")

  # The init pair is anchored at keyframe 0.
  shared = [obs for obs in init_tracks.values() if 0 in obs and init_last in obs]
  pix_a = np.asarray([obs[0] for obs in shared], dtype=np.float64)
  pix_b = np.asarray([obs[init_last] for obs in shared], dtype=np.float64)
  bears_a = mei_unproject(pix_a, init_lens.params)
  bears_b = mei_unproject(pix_b, init_lens.params)
  r_cw_last, t_cw_last, _ = two_view_essential(bears_a, bears_b)
  cam_poses = interpolate_camera_poses(r_cw_last, t_cw_last, n_keyframes)
  body_poses_prealign = body_poses_from_camera_poses(cam_poses, init_lens.extrinsic)

  # Visual-odometry front-end: recover the real per-keyframe trajectory shape
  # from front-lens correspondences (front owns rotation/shape) instead of the
  # two-keyframe straight line. Falls back to the anchor-PnP seed, or to that
  # straight line, when the incremental SfM cannot register enough keyframes.
  anchor_seed_used = False
  sfm_cams, _ = incremental_sfm(
    _bearings_by_track(front.kept, front_lens.params),
    min_shared=args.frontend.init_min_shared_tracks,
  )
  sfm_cams, vo_outliers = _reject_vo_pose_outliers(sfm_cams)
  if vo_outliers:
    print(f"  [{name}] visual-odometry rejected {len(vo_outliers)} rotation-spike kfs")
  if _enough_registered(len(sfm_cams), n_keyframes):
    body_poses_prealign = body_poses_from_camera_poses(
      _fill_camera_poses(sfm_cams, n_keyframes), front_lens.extrinsic
    )
    print(
      f"  [{name}] visual-odometry init: registered "
      f"{len(sfm_cams)}/{n_keyframes} keyframes"
    )
  else:
    anchor_seed = _anchor_pose_seed(
      anchor_obs_per_kf, front_lens.extrinsic, n_keyframes
    )
    if anchor_seed is not None:
      body_poses_prealign = anchor_seed
      anchor_seed_used = True
      n_anchor_kf = sum("front" in s for s in anchor_obs_per_kf.values())
      print(
        f"  [{name}] visual-odometry starved ({len(sfm_cams)}/{n_keyframes}); "
        f"anchor-pose init from {n_anchor_kf} anchor keyframes"
      )
    else:
      print(
        f"  [{name}] visual-odometry init: only {len(sfm_cams)}/{n_keyframes} "
        f"keyframes registered — using straight-line init"
      )

  imu, bias0, gravity_world_initial = _imu_and_initial_gravity(
    args, data, imu, keyframe_times, body_poses_prealign
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

  # Without a stationary window the LSQ-fit gravity is the only signal we
  # have; preintegration uses a placeholder vertical gravity, then the
  # alignment runs against the same data and the LSQ-fit value sticks.
  pim_gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
  if gravity_world_initial is not None:
    gx, gy, gz = gravity_world_initial
    pim_gravity = (float(gx), float(gy), float(gz))

  def _pim_params(
    gravity: tuple[float, float, float],
  ) -> gtsam.PreintegrationCombinedParams:
    return make_preintegration_params(
      gyro_noise_sigma=args.imu.gyro_noise_sigma,
      accel_noise_sigma=args.imu.accel_noise_sigma,
      gyro_walk=args.imu.gyro_random_walk,
      accel_walk=args.imu.accel_random_walk,
      gravity_world=gravity,
    )

  params_initial = _pim_params(pim_gravity)
  # An anchor seed is already metric, so its scale is 1.0; otherwise take the
  # scale from a visible anchor, and float it on the IMU only with no anchor.
  anchor_scale = _anchor_metric_scale(
    body_poses_prealign, anchor_obs_per_kf, front_lens.extrinsic, rear_lens.extrinsic
  )
  fixed_scale = 1.0 if anchor_seed_used else anchor_scale
  pim_init = preintegrate_between_keyframes(params_initial, imu, keyframe_times, bias0)

  def _align(poses: list[gtsam.Pose3]) -> Alignment:
    return visual_inertial_align(
      poses,
      pim_init,
      fixed_gravity=gravity_world_initial,
      fixed_scale=fixed_scale,
    )

  alignment = _align(body_poses_prealign)
  if alignment.scale < 0.0:
    origin = _translation(body_poses_prealign[0])
    body_poses_prealign = [
      gtsam.Pose3(
        p.rotation(),
        gtsam.Point3(*(2.0 * origin - _translation(p))),
      )
      for p in body_poses_prealign
    ]
    alignment = _align(body_poses_prealign)
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
  body_poses, world_rotation = apply_alignment_to_poses(body_poses_prealign, alignment)
  world_rot_mat = world_rotation.matrix()
  velocities = [world_rot_mat @ v for v in alignment.velocities]
  params_aligned = _pim_params((0.0, 0.0, -9.81))
  pim = preintegrate_between_keyframes(params_aligned, imu, keyframe_times, bias0)

  use_anchor = bool(anchor_obs_per_kf)
  # Each sighted anchor's pose in this tracker's local world, for the
  # shared-world alignment BFS.
  anchor_locals_by_tag: dict[int, gtsam.Pose3] = {
    tag_id: _first_sighting_local(obs, front_lens, rear_lens, body_poses)
    for tag_id, obs in obs_by_tag.items()
    if obs
  }

  if use_anchor:
    loop_tracks: dict[int, dict[int, Float64Array]] = {}
    if args.frontend.loop_closure:
      loop_tracks = _loop_closure_tracks(
        front.orb,
        body_poses,
        front_lens,
        max(front.kept, default=0),
        args.frontend.loop_closure_max_dist,
        args.frontend.loop_closure_min_kf_gap,
      )
    anchor_kfs = set(anchor_obs_per_kf.keys())
    cap = args.frontend.max_landmarks_per_lens
    for ch in (front, rear):
      ch.points = triangulate_lens(
        {t: ch.kept[t] for t in ch.kept if t in ch.landmarks_prealign},
        body_poses,
        ch.setup.extrinsic,
        ch.setup.params,
        args.frontend.min_track_length,
        max_landmark_depth=args.frontend.max_landmark_depth,
      )
      ch.points = _cap_landmarks(ch.points, ch.kept, cap)
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
        anchor_kfs=anchor_kfs if args.frontend.anchored_loop_closure else None,
      )
  else:
    align_mat = alignment.scale * world_rotation.matrix()
    for ch in (front, rear):
      ch.points = {
        tid: align_mat @ np.asarray(ch.landmarks_prealign[tid], dtype=np.float64)
        for tid in ch.landmarks_prealign
      }

  n_obs = args.frontend.max_obs_per_track
  for ch in (front, rear):
    ch.tracks_tri = {
      t: _subsample_obs(ch.kept[t], n_obs) for t in ch.points if t in ch.kept
    }

  v_max = float(max(np.linalg.norm(v) for v in velocities))
  print(f"  [{name}] VIO LSQ |v|max={v_max:.2f} m/s")
  biases = [bias0 for _ in range(n_keyframes)]

  body_T_base_tag, joint_angles, finger_reproj = _recover_gripper_tags(
    args, rear_tags_per_kf, rear_lens, n_keyframes, rear_kf_rotations
  )

  return PreparedTracker(
    name=name,
    body_poses=body_poses,
    velocities=velocities,
    biases=biases,
    preintegrated=pim,
    front_lens=front_lens,
    rear_lens=rear_lens,
    front_tracks_tri=front.tracks_tri,
    rear_tracks_tri=rear.tracks_tri,
    front_points=front.points,
    rear_points=rear.points,
    decoded_width=decoded_width,
    anchor_obs_by_tag=obs_by_tag,
    anchor_locals_by_tag=anchor_locals_by_tag,
    body_T_base_tag=body_T_base_tag,
    joint_angles=joint_angles,
    keyframe_times=keyframe_times,
    front_frames=front_color,
    rear_frames=rear_color,
    front_keypoints=[f.keypoints for f in front.feature_frames],
    rear_keypoints=[f.keypoints for f in rear.feature_frames],
    front_tags_per_kf=front_tags_per_kf,
    rear_tags_per_kf=rear_tags_per_kf,
    front_rejected=front.rejected,
    rear_rejected=rear.rejected,
    rear_static_mask_rects=static_rects,
    finger_reproj=finger_reproj,
    left_finger_tag_id=args.left_finger_tag_id,
    right_finger_tag_id=args.right_finger_tag_id,
  )
