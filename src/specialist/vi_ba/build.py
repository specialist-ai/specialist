"""Assemble a dual-lens VI-BA ``NonlinearFactorGraph``: one ``Pose3`` per
keyframe, one ``Point3`` per triangulated landmark (front/rear in disjoint
namespaces), one MEI projection factor per observation. Consecutive
keyframes are tied by IMU preintegration factors (visual-only mode
substitutes weak pose priors), with the gauge pinned by a tight prior on
the first keyframe pose."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt
from gtsam.symbol_shorthand import B, L, V, X

from specialist.loader.mei_model import MeiParams

try:
  # Native C++ MEI reprojection factors (pose+point, base-tag, finger, and
  # cross-tracker base + finger variants).
  from specialist.vi_ba._mei_factor_cpp import (
    make_mei_base_tag_factor_cpp,
    make_mei_cross_finger_factor_cpp,
    make_mei_cross_tag_factor_cpp,
    make_mei_factor_cpp,
    make_mei_finger_factor_cpp,
    make_mei_world_point_factor_cpp,
  )
except ImportError as _exc:  # pragma: no cover - extension is built by uv sync
  raise ImportError(
    "The native MEI factor extension (specialist.vi_ba._mei_factor_cpp) is not "
    "built. It is compiled automatically by `uv sync`; to (re)build it manually "
    "run `uv run build-mei-factor`."
  ) from _exc

Float64Array = npt.NDArray[np.float64]

# Prior on the first bias state, in gtsam's (accel, gyro) tangent order. A
# belief about the starting bias, distinct from ImuCfg's white-noise densities
# and from the per-step init sigma the preintegration params carry.
_BIAS_PRIOR_SIGMA_ACCEL = 1.0e-2
_BIAS_PRIOR_SIGMA_GYRO = 1.0e-3

_REAR_LANDMARK_OFFSET = 1_000_000

__all__ = [
  "LensSetup",
  "add_mei_base_tag_factor",
  "add_mei_cross_finger_factor",
  "add_mei_cross_tag_factor",
  "add_mei_finger_factor",
  "add_mei_world_point_factor",
  "build_vi_graph",
  "front_landmark_key",
  "huber_pixel_noise",
  "rear_landmark_key",
]


def _as_pixel(v: Float64Array) -> Float64Array:
  return np.asarray(v, dtype=np.float64).reshape(2)


def _as_vec3(v: Float64Array) -> Float64Array:
  return np.asarray(v, dtype=np.float64).reshape(3)


def _as_point3(v: Float64Array) -> gtsam.Point3:
  return gtsam.Point3(*np.asarray(v, dtype=np.float64).reshape(3))


def _gauge_noise() -> gtsam.noiseModel.Base:
  """Tight gauge prior (σ=1e-4 on all six pose DoF) that pins the frame."""
  return gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-4, dtype=np.float64))


def _mei_intrinsics(params: MeiParams) -> tuple[float, ...]:
  """The ten MEI intrinsics in the order every ``make_mei_*`` factory expects."""
  return (
    params.fx,
    params.fy,
    params.cx,
    params.cy,
    params.k1,
    params.k2,
    params.k3,
    params.p1,
    params.p2,
    params.xi,
  )


@dataclass(frozen=True)
class LensSetup:
  """Fixed per-lens inputs: body→lens extrinsic + MEI intrinsics."""

  extrinsic: gtsam.Pose3
  params: MeiParams


def front_landmark_key(track_id: int) -> int:
  return int(L(track_id))


def rear_landmark_key(track_id: int) -> int:
  return int(L(_REAR_LANDMARK_OFFSET + track_id))


def huber_pixel_noise(sigma: float) -> gtsam.noiseModel.Base:
  """Robust isotropic pixel noise (Huber, k=1.345) for a reprojection factor."""
  return gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(1.345),
    gtsam.noiseModel.Isotropic.Sigma(2, sigma),
  )


def add_mei_factor(
  graph: gtsam.NonlinearFactorGraph,
  pose_key: int,
  point_key: int,
  pixel: Float64Array,
  noise: gtsam.noiseModel.Base,
  lens: LensSetup,
) -> None:
  """Push one native MEI reprojection factor tying ``pose_key`` to ``point_key``
  through ``lens``'s extrinsic and intrinsics."""
  graph.push_back(
    make_mei_factor_cpp(
      pose_key,
      point_key,
      _as_pixel(pixel),
      noise,
      lens.extrinsic,
      *_mei_intrinsics(lens.params),
      np.zeros(3),
    )
  )


def add_mei_world_point_factor(
  graph: gtsam.NonlinearFactorGraph,
  pose_key: int,
  pixel: Float64Array,
  noise: gtsam.noiseModel.Base,
  lens: LensSetup,
  p_world: Float64Array,
  obs_bearing: Float64Array,
) -> None:
  """Push one native MEI reprojection factor, unary on the observing pose: the
  world point ``p_world`` is a constant (an anchor tag corner), so it
  reprojects as ``pose⁻¹ · p_world`` through ``lens`` onto ``pixel``.
  ``obs_bearing`` is the lens-frame tag direction for the cheirality guard."""
  graph.push_back(
    make_mei_world_point_factor_cpp(
      pose_key,
      _as_pixel(pixel),
      noise,
      lens.extrinsic,
      _as_point3(p_world),
      *_mei_intrinsics(lens.params),
      _as_vec3(obs_bearing),
    )
  )


def add_mei_finger_factor(
  graph: gtsam.NonlinearFactorGraph,
  theta_key: int,
  base_tag_key: int,
  pixel: Float64Array,
  noise: gtsam.noiseModel.Base,
  lens: LensSetup,
  base_T_joint: gtsam.Pose3,
  corner_in_pivot: Float64Array,
  obs_bearing: Float64Array,
) -> None:
  """Push one native MEI finger-angle reprojection factor, binary on the scalar
  joint angle ``theta_key`` and the base-tag pose ``base_tag_key``: the tag corner
  rides a 1-DoF revolute joint (``base_T_joint`` pivot in the base-tag frame, +z
  axis), so it reprojects as ``body_T_base_tag · base_T_joint · Rz(theta) ·
  corner_in_pivot`` through ``lens`` onto ``pixel``. ``obs_bearing`` is the
  lens-frame tag direction for the ξ>1 behind-camera cheirality guard."""
  graph.push_back(
    make_mei_finger_factor_cpp(
      theta_key,
      base_tag_key,
      _as_pixel(pixel),
      noise,
      lens.extrinsic,
      base_T_joint,
      _as_point3(corner_in_pivot),
      *_mei_intrinsics(lens.params),
      _as_vec3(obs_bearing),
    )
  )


def add_mei_base_tag_factor(
  graph: gtsam.NonlinearFactorGraph,
  base_tag_key: int,
  pixel: Float64Array,
  noise: gtsam.noiseModel.Base,
  lens: LensSetup,
  corner: Float64Array,
  obs_bearing: Float64Array,
) -> None:
  """Push one native MEI base-tag reprojection factor, unary on the base-tag pose
  ``base_tag_key``: the tag corner is fixed in the base-tag frame, so it
  reprojects as ``body_T_base_tag · corner`` through ``lens`` onto ``pixel``.
  ``obs_bearing`` is the lens-frame tag direction for the cheirality guard."""
  graph.push_back(
    make_mei_base_tag_factor_cpp(
      base_tag_key,
      _as_pixel(pixel),
      noise,
      lens.extrinsic,
      _as_point3(corner),
      *_mei_intrinsics(lens.params),
      _as_vec3(obs_bearing),
    )
  )


def add_mei_cross_tag_factor(
  graph: gtsam.NonlinearFactorGraph,
  observer_key: int,
  owner_key: int,
  base_tag_key: int,
  pixel: Float64Array,
  noise: gtsam.noiseModel.Base,
  observer_lens: LensSetup,
  corner: Float64Array,
  obs_bearing: Float64Array,
) -> None:
  """Push one native MEI cross-tracker reprojection factor, ternary on the
  observer pose, owner pose, and owner's base-tag pose: owner W's base-tag corner
  reprojects as ``X_O⁻¹ · X_W · body_T_base_tag · corner`` through the observer's
  ``observer_lens`` onto ``pixel``. ``obs_bearing`` is the observer-lens tag
  direction for the cheirality guard."""
  graph.push_back(
    make_mei_cross_tag_factor_cpp(
      observer_key,
      owner_key,
      base_tag_key,
      _as_pixel(pixel),
      noise,
      observer_lens.extrinsic,
      _as_point3(corner),
      *_mei_intrinsics(observer_lens.params),
      _as_vec3(obs_bearing),
    )
  )


def add_mei_cross_finger_factor(
  graph: gtsam.NonlinearFactorGraph,
  observer_key: int,
  owner_key: int,
  base_tag_key: int,
  theta_key: int,
  pixel: Float64Array,
  noise: gtsam.noiseModel.Base,
  observer_lens: LensSetup,
  base_T_joint: gtsam.Pose3,
  corner_in_pivot: Float64Array,
  obs_bearing: Float64Array,
) -> None:
  """Push one native MEI cross-tracker finger reprojection factor, quaternary on
  the observer pose, owner pose, owner's base-tag pose, and owner's joint angle:
  owner W's finger-tag corner reprojects as ``X_O⁻¹ · X_W · body_T_base_tag ·
  base_T_joint · Rz(θ) · corner_in_pivot`` through ``observer_lens`` onto
  ``pixel``. ``obs_bearing`` is the observer-lens tag direction for the
  cheirality guard."""
  graph.push_back(
    make_mei_cross_finger_factor_cpp(
      observer_key,
      owner_key,
      base_tag_key,
      theta_key,
      _as_pixel(pixel),
      noise,
      observer_lens.extrinsic,
      base_T_joint,
      _as_point3(corner_in_pivot),
      *_mei_intrinsics(observer_lens.params),
      _as_vec3(obs_bearing),
    )
  )


def _add_projection_factors(
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
  tracks: Mapping[int, Mapping[int, Float64Array]],
  points: Mapping[int, Float64Array],
  lens: LensSetup,
  landmark_key_fn: Callable[[int], int],
  pixel_noise: gtsam.noiseModel.Base,
  pose_key_fn: Callable[[int], int],
) -> None:
  for tid, xyz in points.items():
    lm_key = landmark_key_fn(tid)
    if not values.exists(lm_key):
      values.insert(lm_key, _as_point3(xyz))
    for kf, pt in tracks[tid].items():
      add_mei_factor(graph, pose_key_fn(kf), lm_key, pt, pixel_noise, lens)


def build_vi_graph(
  body_poses: list[gtsam.Pose3],
  *,
  velocities: list[Float64Array] | None = None,
  biases: list[gtsam.imuBias.ConstantBias] | None = None,
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements] | None = None,
  front: LensSetup,
  front_tracks: Mapping[int, Mapping[int, Float64Array]],
  front_points: Mapping[int, Float64Array],
  rear: LensSetup,
  rear_tracks: Mapping[int, Mapping[int, Float64Array]],
  rear_points: Mapping[int, Float64Array],
  sigma: float,
  velocity_prior_sigma: float = 0.1,
  anchor_last_pose: bool = False,
  pose_key_fn: Callable[[int], int] | None = None,
  velocity_key_fn: Callable[[int], int] | None = None,
  bias_key_fn: Callable[[int], int] | None = None,
  front_landmark_key_fn: Callable[[int], int] | None = None,
  rear_landmark_key_fn: Callable[[int], int] | None = None,
  graph: gtsam.NonlinearFactorGraph | None = None,
  values: gtsam.Values | None = None,
  weak_pose_sigmas: tuple[float, ...] | None = None,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """Dual-lens visual-inertial graph: MEI projection factors plus
  ``preintegrated[i]`` covering ``(t_i, t_{i+1}]`` between consecutive
  keyframes. Gauge is fixed by a tight prior on ``X(0)`` plus Gaussian
  priors on ``V(0)`` / ``B(0)``.

  Pass ``preintegrated=None`` for a visual-only graph: the velocity/bias
  variables, IMU factors, and ``V(0)`` / ``B(0)`` priors are dropped and the
  inter-pose motion model becomes a gauge prior on ``X(0)``, an optional
  tight ``anchor_last_pose`` prior on ``X(N-1)``, and weak
  ``PriorFactorPose3`` priors (σ = ``weak_pose_sigmas``) on the rest."""
  n = len(body_poses)
  if preintegrated is not None:
    assert velocities is not None and biases is not None
    if n < 2:
      raise ValueError("build_vi_graph needs at least two keyframes")
    if len(velocities) != n or len(biases) != n:
      raise ValueError(
        f"velocities/biases count mismatch: {len(velocities)}/{len(biases)} vs {n}"
      )
    if len(preintegrated) != n - 1:
      raise ValueError(
        f"preintegrated count must be N-1={n - 1}, got {len(preintegrated)}"
      )

  if graph is None:
    graph = gtsam.NonlinearFactorGraph()
  if values is None:
    values = gtsam.Values()
  pose_key = pose_key_fn or (lambda i: int(X(i)))
  front_lm = front_landmark_key_fn or front_landmark_key
  rear_lm = rear_landmark_key_fn or rear_landmark_key
  pixel_noise = huber_pixel_noise(sigma)
  gauge = _gauge_noise()

  if preintegrated is None:
    # Visual-only: gauge prior on X(0), an optional tight prior on X(N-1), and
    # weak pose priors on the rest standing in for the IMU chain.
    weak = gtsam.noiseModel.Diagonal.Sigmas(
      np.asarray(weak_pose_sigmas, dtype=np.float64)
    )
    for i, pose in enumerate(body_poses):
      values.insert(pose_key(i), pose)
    graph.push_back(gtsam.PriorFactorPose3(pose_key(0), body_poses[0], gauge))
    if anchor_last_pose and n > 1:
      graph.push_back(gtsam.PriorFactorPose3(pose_key(n - 1), body_poses[-1], gauge))
    weak_end = n - 1 if anchor_last_pose else n
    for i in range(1, weak_end):
      graph.push_back(gtsam.PriorFactorPose3(pose_key(i), body_poses[i], weak))
  else:
    # Visual-inertial: CombinedImuFactor chains the NavState between keyframes,
    # with gauge + initial priors on (X0, V0, B0).
    assert velocities is not None and biases is not None
    vel_key = velocity_key_fn or (lambda i: int(V(i)))
    bias_key = bias_key_fn or (lambda i: int(B(i)))
    vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, velocity_prior_sigma)
    bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
      np.asarray(
        [_BIAS_PRIOR_SIGMA_ACCEL] * 3 + [_BIAS_PRIOR_SIGMA_GYRO] * 3,
        dtype=np.float64,
      )
    )
    for i in range(n):
      values.insert(pose_key(i), body_poses[i])
      values.insert(vel_key(i), np.asarray(velocities[i], dtype=np.float64))
      values.insert(bias_key(i), biases[i])
    graph.push_back(gtsam.PriorFactorPose3(pose_key(0), body_poses[0], gauge))
    graph.push_back(
      gtsam.PriorFactorVector(
        vel_key(0), np.asarray(velocities[0], dtype=np.float64), vel_noise
      )
    )
    graph.push_back(gtsam.PriorFactorConstantBias(bias_key(0), biases[0], bias_noise))
    for i, pim in enumerate(preintegrated):
      graph.push_back(
        gtsam.CombinedImuFactor(
          pose_key(i),
          vel_key(i),
          pose_key(i + 1),
          vel_key(i + 1),
          bias_key(i),
          bias_key(i + 1),
          pim,
        )
      )

  _add_projection_factors(
    graph, values, front_tracks, front_points, front, front_lm, pixel_noise, pose_key
  )
  _add_projection_factors(
    graph, values, rear_tracks, rear_points, rear, rear_lm, pixel_noise, pose_key
  )
  return graph, values
