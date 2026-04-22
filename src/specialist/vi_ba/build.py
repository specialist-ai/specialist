"""Assemble a dual-lens VI-BA ``NonlinearFactorGraph``: one ``Pose3`` per
keyframe, one ``Point3`` per triangulated landmark (front/rear in disjoint
namespaces), one MEI projection factor per observation. Gauge is fixed by
a tight prior on the first and last keyframe poses plus weak priors on
the intermediates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt
from gtsam.symbol_shorthand import B, L, V, X

from specialist.loader.mei_model import MeiParams
from specialist.vi_ba.mei_factor import make_mei_factor

Float64Array = npt.NDArray[np.float64]

_REAR_LANDMARK_OFFSET = 1_000_000

__all__ = [
  "LensSetup",
  "build_graph",
  "build_vi_graph",
  "front_landmark_key",
  "rear_landmark_key",
]


@dataclass(frozen=True)
class LensSetup:
  """Fixed per-lens inputs: body→lens extrinsic + MEI intrinsics."""

  extrinsic: gtsam.Pose3
  params: MeiParams


def front_landmark_key(track_id: int) -> int:
  return int(L(track_id))


def rear_landmark_key(track_id: int) -> int:
  return int(L(_REAR_LANDMARK_OFFSET + track_id))


def _add_projection_factors(
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
  tracks: Mapping[int, Mapping[int, Float64Array]],
  points: Mapping[int, Float64Array],
  lens: LensSetup,
  landmark_key_fn: Callable[[int], int],
  pixel_noise: gtsam.noiseModel.Base,
  pose_key_fn: Callable[[int], int] | None = None,
) -> None:
  pose_key = pose_key_fn if pose_key_fn is not None else (lambda i: int(X(i)))
  for tid, xyz in points.items():
    lm_key = landmark_key_fn(tid)
    if not values.exists(lm_key):
      values.insert(lm_key, gtsam.Point3(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    for fi, pt in tracks[tid].items():
      graph.push_back(
        make_mei_factor(
          pose_key=pose_key(fi),
          landmark_key=lm_key,
          pixel=np.asarray(pt, dtype=np.float64),
          noise=pixel_noise,
          extrinsic=lens.extrinsic,
          params=lens.params,
        )
      )


def build_graph(
  body_poses: list[gtsam.Pose3],
  front: LensSetup,
  front_tracks: Mapping[int, Mapping[int, Float64Array]],
  front_points: Mapping[int, Float64Array],
  rear: LensSetup | None,
  rear_tracks: Mapping[int, Mapping[int, Float64Array]] | None,
  rear_points: Mapping[int, Float64Array] | None,
  pixel_sigma: float,
  weak_pose_sigmas: tuple[float, float, float, float, float, float],
  anchor_last_pose: bool = True,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """Assemble the dual-lens visual-only BA graph + initial values. Pass
  ``rear=None`` (and matching ``None`` for tracks/points) to run a
  front-only graph. ``anchor_last_pose=False`` skips the tight prior on
  ``X(N-1)`` so VI alignment can rescale the trajectory afterwards."""
  graph = gtsam.NonlinearFactorGraph()
  values = gtsam.Values()

  # Noise models.
  pixel_noise = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(1.345),
    gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma),
  )
  gauge = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-4, dtype=np.float64))
  weak = gtsam.noiseModel.Diagonal.Sigmas(
    np.asarray(weak_pose_sigmas, dtype=np.float64)
  )

  # Pose variables + gauge / weak priors.
  n = len(body_poses)
  for i, pose in enumerate(body_poses):
    values.insert(X(i), pose)
  graph.push_back(gtsam.PriorFactorPose3(X(0), body_poses[0], gauge))
  if anchor_last_pose and n > 1:
    graph.push_back(gtsam.PriorFactorPose3(X(n - 1), body_poses[-1], gauge))
  for i in range(1, n):
    if anchor_last_pose and i == n - 1:
      continue
    graph.push_back(gtsam.PriorFactorPose3(X(i), body_poses[i], weak))

  # Per-lens projection factors.
  _add_projection_factors(
    graph, values, front_tracks, front_points, front, front_landmark_key, pixel_noise
  )
  if rear is not None and rear_tracks is not None and rear_points is not None:
    _add_projection_factors(
      graph, values, rear_tracks, rear_points, rear, rear_landmark_key, pixel_noise
    )

  return graph, values


def build_vi_graph(
  body_poses: list[gtsam.Pose3],
  velocities: list[Float64Array],
  biases: list[gtsam.imuBias.ConstantBias],
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements],
  front: LensSetup,
  front_tracks: Mapping[int, Mapping[int, Float64Array]],
  front_points: Mapping[int, Float64Array],
  rear: LensSetup | None,
  rear_tracks: Mapping[int, Mapping[int, Float64Array]] | None,
  rear_points: Mapping[int, Float64Array] | None,
  pixel_sigma: float,
  velocity_prior_sigma: float = 0.1,
  bias_prior_accel_sigma: float = 1.0e-2,
  bias_prior_gyro_sigma: float = 1.0e-3,
  anchor_last_pose: bool = False,
  pose_key_fn: Callable[[int], int] | None = None,
  velocity_key_fn: Callable[[int], int] | None = None,
  bias_key_fn: Callable[[int], int] | None = None,
  front_landmark_key_fn: Callable[[int], int] | None = None,
  rear_landmark_key_fn: Callable[[int], int] | None = None,
  graph: gtsam.NonlinearFactorGraph | None = None,
  values: gtsam.Values | None = None,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """Dual-lens visual-inertial graph: MEI projection factors plus
  ``preintegrated[i]`` covering ``(t_i, t_{i+1}]`` between consecutive
  keyframes. Gauge is fixed by a tight prior on ``X(0)`` plus Gaussian
  priors on ``V(0)`` / ``B(0)``."""
  n = len(body_poses)
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
  pose_key = pose_key_fn if pose_key_fn is not None else (lambda i: int(X(i)))
  vel_key = velocity_key_fn if velocity_key_fn is not None else (lambda i: int(V(i)))
  bias_key = bias_key_fn if bias_key_fn is not None else (lambda i: int(B(i)))
  front_lm = (
    front_landmark_key_fn if front_landmark_key_fn is not None else front_landmark_key
  )
  rear_lm = (
    rear_landmark_key_fn if rear_landmark_key_fn is not None else rear_landmark_key
  )

  pixel_noise = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(1.345),
    gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma),
  )
  gauge = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-4, dtype=np.float64))
  vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, velocity_prior_sigma)
  bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.asarray(
      [bias_prior_accel_sigma] * 3 + [bias_prior_gyro_sigma] * 3,
      dtype=np.float64,
    )
  )

  # Variables + gauge / initial priors on (X0, V0, B0).
  for i in range(n):
    values.insert(pose_key(i), body_poses[i])
    values.insert(vel_key(i), np.asarray(velocities[i], dtype=np.float64))
    values.insert(bias_key(i), biases[i])
  graph.push_back(gtsam.PriorFactorPose3(pose_key(0), body_poses[0], gauge))
  if anchor_last_pose:
    graph.push_back(gtsam.PriorFactorPose3(pose_key(n - 1), body_poses[-1], gauge))
  graph.push_back(
    gtsam.PriorFactorVector(
      vel_key(0), np.asarray(velocities[0], dtype=np.float64), vel_noise
    )
  )
  graph.push_back(gtsam.PriorFactorConstantBias(bias_key(0), biases[0], bias_noise))

  # IMU factors chain the NavState through preintegrated measurements.
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

  # Per-lens projection factors.
  _add_projection_factors(
    graph, values, front_tracks, front_points, front, front_lm, pixel_noise, pose_key
  )
  if rear is not None and rear_tracks is not None and rear_points is not None:
    _add_projection_factors(
      graph, values, rear_tracks, rear_points, rear, rear_lm, pixel_noise, pose_key
    )

  return graph, values
