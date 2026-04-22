"""Initialization helpers for dual-lens VI-BA: two-view essential init on
unit bearings, SLERP + linear-t pose interpolation, multi-view closest-point
triangulation, and visual-inertial alignment for scale + gravity."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import gtsam
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as Rscp
from scipy.spatial.transform import Slerp

Float64Array = npt.NDArray[np.float64]

# Frozen incremental-SfM tuning: minimum triangulation parallax, minimum PnP
# inlier points to register a keyframe, maximum median triangulation
# reprojection residual, and the number of sequential registration passes.
SFM_MIN_PARALLAX = float(np.deg2rad(0.6))
# Minimum parallax the two-view init pair (0, K) must span.
SFM_INIT_PAIR_MIN_PARALLAX = float(np.deg2rad(2.0))
SFM_MIN_PNP_POINTS = 6
SFM_MAX_REPROJ = 0.01
SFM_PASSES = 8


@dataclass(frozen=True)
class Alignment:
  """VI alignment result: ``scale`` for visual positions, ``gravity_world``
  in the visual BA's world frame, per-keyframe ``velocities``."""

  scale: float
  gravity_world: Float64Array
  velocities: list[Float64Array]


__all__ = [
  "Alignment",
  "apply_alignment_to_poses",
  "body_poses_from_camera_poses",
  "gravity_alignment_rotation",
  "interpolate_camera_poses",
  "pick_init_pair",
  "triangulate_track_multiview",
  "two_view_essential",
  "visual_inertial_align",
]


def _point3(vec: Float64Array) -> gtsam.Point3:
  return gtsam.Point3(float(vec[0]), float(vec[1]), float(vec[2]))


def two_view_essential(
  bearings_a: Float64Array,
  bearings_b: Float64Array,
  threshold: float = 1e-2,
) -> tuple[Float64Array, Float64Array, int]:
  """Essential-matrix init between two views; returns ``(R_cw_last,
  t_cw_last, n_inliers)`` with unit-norm translation (scale unobservable)."""
  if bearings_a.shape[0] < 8 or bearings_b.shape[0] != bearings_a.shape[0]:
    raise RuntimeError(
      f"two_view_essential needs ≥8 correspondences, got {bearings_a.shape[0]}"
    )
  ok = (bearings_a[:, 2] > 1e-3) & (bearings_b[:, 2] > 1e-3)
  if int(ok.sum()) < 8:
    raise RuntimeError(
      f"only {int(ok.sum())} in-forward-hemisphere correspondences — too few"
    )
  norm_a = (bearings_a[ok][:, :2] / bearings_a[ok][:, 2:3]).astype(np.float64)
  norm_b = (bearings_b[ok][:, :2] / bearings_b[ok][:, 2:3]).astype(np.float64)
  E, _ = cv2.findEssentialMat(
    norm_a,
    norm_b,
    focal=1.0,
    pp=(0.0, 0.0),
    method=cv2.RANSAC,
    prob=0.999,
    threshold=threshold,
  )
  if E is None:
    raise RuntimeError("essential matrix failed on the init pair")
  n_inliers, R, t, _ = cv2.recoverPose(E, norm_a, norm_b)
  return R.astype(np.float64), t.flatten().astype(np.float64), int(n_inliers)


def pick_init_pair(
  tracks: dict[int, dict[int, Float64Array]],
  n_keyframes: int,
  min_shared: int,
) -> tuple[int, int, int]:
  """Widest ``(0, K)`` pair that still has ≥``min_shared`` shared tracks.

  Falls back to the pair with the most shared observations if no ``K``
  meets the threshold.
  """
  first = 0

  def shared_count(last: int) -> int:
    return sum(1 for obs in tracks.values() if first in obs and last in obs)

  for last in range(n_keyframes - 1, 0, -1):
    shared = shared_count(last)
    if shared >= min_shared:
      return first, last, shared
  best_last, best_shared = 1, 0
  for last in range(1, n_keyframes):
    shared = shared_count(last)
    if shared > best_shared:
      best_last, best_shared = last, shared
  return first, best_last, best_shared


def interpolate_camera_poses(
  r_cw_last: Float64Array,
  t_cw_last: Float64Array,
  n_keyframes: int,
  init_first: int = 0,
  init_last: int | None = None,
) -> list[gtsam.Pose3]:
  """SLERP + linear-t interpolation across the init pair, returning
  ``T_world_camera`` for every keyframe. Keyframes outside
  ``[init_first, init_last]`` are extrapolated linearly.
  """
  if init_last is None:
    init_last = n_keyframes - 1
  if init_last <= init_first:
    raise ValueError(f"init_last ({init_last}) must exceed init_first ({init_first})")
  key_rots = Rscp.from_matrix(np.stack([np.eye(3), r_cw_last]))
  slerp = Slerp([0.0, 1.0], key_rots)
  out: list[gtsam.Pose3] = []
  denom = float(init_last - init_first)
  for i in range(n_keyframes):
    alpha = (i - init_first) / denom
    if 0.0 <= alpha <= 1.0:
      r_cw = slerp([alpha])[0].as_matrix()
    else:
      rotvec = Rscp.from_matrix(r_cw_last).as_rotvec() * alpha
      r_cw = Rscp.from_rotvec(rotvec).as_matrix()
    t_cw = alpha * t_cw_last
    r_wc = r_cw.T
    t_wc = -r_wc @ t_cw
    out.append(gtsam.Pose3(gtsam.Rot3(r_wc), _point3(t_wc)))
  return out


def body_poses_from_camera_poses(
  camera_poses: list[gtsam.Pose3],
  extrinsic: gtsam.Pose3,
) -> list[gtsam.Pose3]:
  """``T_world_body = T_world_camera · T_body_lens⁻¹``.

  ``extrinsic`` is body → lens for the lens whose init drove
  ``interpolate_camera_poses``.
  """
  inv = extrinsic.inverse()
  return [p.compose(inv) for p in camera_poses]


def triangulate_track_multiview(
  bearings: list[Float64Array],
  body_poses: list[gtsam.Pose3],
  extrinsic: gtsam.Pose3,
  min_cheirality_z: float = 0.05,
  min_landmark_depth: float = 0.15,
  max_landmark_depth: float = 5.0,
) -> Float64Array | None:
  """Geometric closest point across all observations of one track: the world
  point minimizing the summed squared perpendicular distance to the
  back-projected rays.
  Bearings with z ≤ ``min_cheirality_z`` are dropped; the result is rejected
  if fewer than two surviving views see it in front of the camera or its
  median depth falls outside ``[min_landmark_depth, max_landmark_depth]``."""
  if len(bearings) != len(body_poses):
    raise ValueError(
      f"bearings/poses length mismatch: {len(bearings)} vs {len(body_poses)}"
    )

  valid = [
    (b, p) for b, p in zip(bearings, body_poses, strict=True) if b[2] > min_cheirality_z
  ]
  if len(valid) < 2:
    return None
  # With camera centers Cᵢ and unit ray directions dᵢ the normal equations are
  # (Σ I - dᵢdᵢᵀ) X = Σ (I - dᵢdᵢᵀ) Cᵢ.
  normal_matrix = np.zeros((3, 3))
  rhs = np.zeros(3)
  t_world_lenses: list[gtsam.Pose3] = []
  for bearing, pose in valid:
    t_world_lens = pose.compose(extrinsic)
    t_world_lenses.append(t_world_lens)
    center = np.asarray(t_world_lens.translation(), dtype=np.float64)
    ray = t_world_lens.rotation().matrix() @ np.asarray(bearing, dtype=np.float64)
    ray = ray / float(np.linalg.norm(ray))
    perp = np.eye(3) - np.outer(ray, ray)
    normal_matrix += perp
    rhs += perp @ center
  try:
    xyz = np.linalg.solve(normal_matrix, rhs).astype(np.float64)
  except np.linalg.LinAlgError:
    return None
  if not np.all(np.isfinite(xyz)):
    return None
  depths: list[float] = []
  n_good_cheirality = 0
  for t_world_lens in t_world_lenses:
    p_lens = t_world_lens.transformTo(_point3(xyz))
    z = float(np.asarray(p_lens)[2])
    depths.append(z)
    if z >= min_cheirality_z:
      n_good_cheirality += 1
  # Two views in front of the camera are enough: on a super-fisheye capture a
  # landmark can sit near the 90° horizon of one frame and still be good.
  if n_good_cheirality < 2:
    return None
  median_depth = float(np.median([z for z in depths if z >= min_cheirality_z]))
  if median_depth < min_landmark_depth:
    return None
  if median_depth > max_landmark_depth:
    return None
  return np.asarray(xyz, dtype=np.float64)


def visual_inertial_align(
  body_poses: list[gtsam.Pose3],
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements],
  velocity_regularization: float = 1.0e-2,
) -> Alignment:
  """Linear VINS-Mono-style alignment: stacks the discrete NavState
  constraints (position + velocity) across consecutive keyframes and
  solves ``A·x = b`` for ``x = [s, g, v_0, …, v_{N-1}]`` via lstsq.

  ``velocity_regularization`` is a Tikhonov weight λ on each ``v_i`` unknown,
  capping the velocity blow-up on near-stationary windows, where ``Δp_visual``
  is too small to constrain scale.

  Requires ``N ≥ 4`` keyframes.
  """
  n = len(body_poses)
  if n < 4:
    raise ValueError(f"visual_inertial_align needs ≥4 keyframes, got {n}")
  if len(preintegrated) != n - 1:
    raise ValueError(f"expected {n - 1} pim entries, got {len(preintegrated)}")
  dts = [float(pim.deltaTij()) for pim in preintegrated]

  n_unknowns = 3 * n + 4
  n_eq_data = 6 * (n - 1)
  n_eq_reg = 3 * n if velocity_regularization > 0 else 0
  A = np.zeros((n_eq_data + n_eq_reg, n_unknowns))
  b = np.zeros(n_eq_data + n_eq_reg)
  eye3 = np.eye(3)

  for i in range(n - 1):
    dt = dts[i]
    rot_i = body_poses[i].rotation().matrix()
    p_visual_delta = np.asarray(body_poses[i + 1].translation()) - np.asarray(
      body_poses[i].translation()
    )
    pim = preintegrated[i]
    delta_p = rot_i @ np.asarray(pim.deltaPij(), dtype=np.float64)
    delta_v = rot_i @ np.asarray(pim.deltaVij(), dtype=np.float64)

    row_p = 6 * i
    row_v = 6 * i + 3
    # Position equation: s·Δp_v − v_i·Δt − 0.5·g·Δt² = R_i·Δp_pim
    A[row_p : row_p + 3, 0] = p_visual_delta
    A[row_p : row_p + 3, 1:4] = -0.5 * dt * dt * eye3
    A[row_p : row_p + 3, 4 + 3 * i : 4 + 3 * i + 3] = -dt * eye3
    b[row_p : row_p + 3] = delta_p
    # Velocity equation: v_{i+1} − v_i − g·Δt = R_i·Δv_pim
    A[row_v : row_v + 3, 1:4] = -dt * eye3
    A[row_v : row_v + 3, 4 + 3 * i : 4 + 3 * i + 3] = -eye3
    A[row_v : row_v + 3, 4 + 3 * (i + 1) : 4 + 3 * (i + 1) + 3] = eye3
    b[row_v : row_v + 3] = delta_v

  if n_eq_reg:
    sqrt_lam = float(np.sqrt(velocity_regularization))
    for i in range(n):
      r = n_eq_data + 3 * i
      A[r : r + 3, 4 + 3 * i : 4 + 3 * i + 3] = sqrt_lam * eye3

  x, *_ = np.linalg.lstsq(A, b, rcond=None)
  scale = float(x[0])
  gravity = x[1:4].astype(np.float64)
  velocities = [
    np.asarray(x[4 + 3 * i : 4 + 3 * i + 3], dtype=np.float64) for i in range(n)
  ]
  return Alignment(scale=scale, gravity_world=gravity, velocities=velocities)


def gravity_alignment_rotation(gravity_world: Float64Array) -> gtsam.Rot3:
  """Rotation that maps ``gravity_world`` onto ``(0, 0, −|g|)``.

  Picks the minimum-angle rotation (Rodrigues between unit vectors) — yaw
  around gravity is left unconstrained, so the caller should fix yaw by
  convention if needed.
  """
  g = np.asarray(gravity_world, dtype=np.float64)
  g_norm = float(np.linalg.norm(g))
  if g_norm < 1e-10:
    return gtsam.Rot3()
  target = np.array([0.0, 0.0, -1.0], dtype=np.float64)
  src = g / g_norm
  dot = float(np.dot(src, target))
  if dot > 1.0 - 1e-12:
    return gtsam.Rot3()
  if dot < -1.0 + 1e-12:
    # 180° — pick any perpendicular axis.
    axis = np.cross(src, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(axis) < 1e-6:
      axis = np.cross(src, np.array([0.0, 1.0, 0.0]))
    axis = axis / np.linalg.norm(axis)
    return gtsam.Rot3.AxisAngle(gtsam.Point3(*axis), np.pi)
  axis = np.cross(src, target)
  axis_norm = np.linalg.norm(axis)
  axis = axis / axis_norm
  angle = float(np.arccos(dot))
  return gtsam.Rot3.AxisAngle(gtsam.Point3(*axis), angle)


def apply_alignment_to_poses(
  body_poses: list[gtsam.Pose3],
  alignment: Alignment,
) -> tuple[list[gtsam.Pose3], gtsam.Rot3]:
  """Rescale the trajectory by ``alignment.scale`` and rotate the world so
  gravity points along ``−z``.

  Returns the rescaled and rotated body poses plus the ``R_world_visual``
  rotation used; the caller applies the same rotation to landmarks and
  velocities.
  """
  rot = gravity_alignment_rotation(alignment.gravity_world)
  r_matrix = rot.matrix()
  out: list[gtsam.Pose3] = []
  for pose in body_poses:
    t = alignment.scale * np.asarray(pose.translation(), dtype=np.float64)
    t_new = r_matrix @ t
    r_new = r_matrix @ pose.rotation().matrix()
    out.append(
      gtsam.Pose3(
        gtsam.Rot3(r_new),
        _point3(t_new),
      )
    )
  return out, rot
