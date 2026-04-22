"""Dual-lens graph assembly and the LM wrapper over small synthetic scenes."""

from __future__ import annotations

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, V, X

from specialist.loader.mei_model import MeiParams, mei_project
from specialist.vi_ba.build import (
  LensSetup,
  build_vi_graph,
  front_landmark_key,
  rear_landmark_key,
)
from specialist.vi_ba.imu import (
  ImuSamples,
  integrate_samples,
  make_preintegration_params,
)
from specialist.vi_ba.optimize import optimize_graph


def _params() -> MeiParams:
  return MeiParams(
    fx=500.0,
    fy=500.0,
    cx=320.0,
    cy=240.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
    xi=0.0,  # pinhole-like
  )


def _project_point(
  pose_wb: gtsam.Pose3,
  extrinsic: gtsam.Pose3,
  world_xyz: np.ndarray,
  params: MeiParams,
) -> np.ndarray:
  t_wl = pose_wb.compose(extrinsic)
  p_lens = t_wl.transformTo(gtsam.Point3(*world_xyz))
  return mei_project(np.asarray(p_lens)[None, :], params)[0]


def test_build_graph_minimum_two_poses_one_landmark() -> None:
  params = _params()
  extrinsic = gtsam.Pose3()
  poses = [gtsam.Pose3(), gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.3, 0.0, 0.0))]

  xyz = np.array([0.1, 0.05, 2.0])
  pix_a = _project_point(poses[0], extrinsic, xyz, params)
  pix_b = _project_point(poses[1], extrinsic, xyz, params)

  front_setup = LensSetup(extrinsic=extrinsic, params=params)
  front_tracks = {7: {0: pix_a, 1: pix_b}}
  front_points = {7: xyz}

  graph, values = build_vi_graph(
    body_poses=poses,
    preintegrated=None,
    front=front_setup,
    front_tracks=front_tracks,
    front_points=front_points,
    rear=None,
    rear_tracks=None,
    rear_points=None,
    sigma=1.0,
    weak_pose_sigmas=(0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
    anchor_last_pose=True,
  )

  # 2 tight priors (first + last), 0 intermediate priors, 2 projection factors.
  assert graph.size() == 4
  assert values.exists(X(0)) and values.exists(X(1))
  assert values.exists(front_landmark_key(7))


def test_build_graph_dual_lens_uses_disjoint_landmark_keys() -> None:
  params = _params()
  poses = [gtsam.Pose3(), gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.3, 0.0, 0.0))]
  extrinsic = gtsam.Pose3()

  xyz_front = np.array([0.1, 0.0, 2.0])
  xyz_rear = np.array([-0.1, 0.0, 2.0])  # behind-world landmark

  front_tracks = {
    0: {
      0: _project_point(poses[0], extrinsic, xyz_front, params),
      1: _project_point(poses[1], extrinsic, xyz_front, params),
    }
  }
  rear_tracks = {
    0: {
      0: _project_point(poses[0], extrinsic, xyz_rear, params),
      1: _project_point(poses[1], extrinsic, xyz_rear, params),
    }
  }

  lens = LensSetup(extrinsic=extrinsic, params=params)
  graph, values = build_vi_graph(
    body_poses=poses,
    preintegrated=None,
    front=lens,
    front_tracks=front_tracks,
    front_points={0: xyz_front},
    rear=lens,
    rear_tracks=rear_tracks,
    rear_points={0: xyz_rear},
    sigma=1.0,
    weak_pose_sigmas=(0.3,) * 6,
    anchor_last_pose=True,
  )
  # Front landmark key != rear landmark key.
  assert front_landmark_key(0) != rear_landmark_key(0)
  assert values.exists(front_landmark_key(0))
  assert values.exists(rear_landmark_key(0))
  # 2 tight priors + 2 front obs + 2 rear obs = 6.
  assert graph.size() == 6


def test_build_graph_weak_priors_on_intermediate_keyframes() -> None:
  params = _params()
  extrinsic = gtsam.Pose3()
  poses = [
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 0.0)),
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.1, 0.0, 0.0)),
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.2, 0.0, 0.0)),
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.3, 0.0, 0.0)),
  ]
  graph, _ = build_vi_graph(
    body_poses=poses,
    preintegrated=None,
    front=LensSetup(extrinsic=extrinsic, params=params),
    front_tracks={},
    front_points={},
    rear=None,
    rear_tracks=None,
    rear_points=None,
    sigma=1.0,
    weak_pose_sigmas=(0.3,) * 6,
    anchor_last_pose=True,
  )
  # 2 tight priors + 2 weak priors (frames 1, 2) = 4 total.
  assert graph.size() == 4


def test_optimize_graph_converges_from_small_perturbation() -> None:
  params = _params()
  extrinsic = gtsam.Pose3()
  true_poses = [
    gtsam.Pose3(),
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.3, 0.0, 0.0)),
  ]
  rng = np.random.default_rng(4)
  xyzs = rng.uniform(-0.4, 0.4, size=(6, 3))
  xyzs[:, 2] = rng.uniform(1.5, 3.0, size=6)

  front_tracks = {}
  front_points = {}
  for tid, xyz in enumerate(xyzs):
    front_tracks[tid] = {
      0: _project_point(true_poses[0], extrinsic, xyz, params),
      1: _project_point(true_poses[1], extrinsic, xyz, params),
    }
    front_points[tid] = xyz

  init_poses = [
    true_poses[0],
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.25, 0.02, 0.01)),
  ]
  graph, values = build_vi_graph(
    body_poses=init_poses,
    preintegrated=None,
    front=LensSetup(extrinsic=extrinsic, params=params),
    front_tracks=front_tracks,
    front_points=front_points,
    rear=None,
    rear_tracks=None,
    rear_points=None,
    sigma=1.0,
    weak_pose_sigmas=(0.3,) * 6,
    anchor_last_pose=True,
  )
  initial_err = graph.error(values)
  result = optimize_graph(graph, values, max_iterations=50)
  assert result.final_error <= initial_err * 0.5
  assert result.iterations >= 1


def _stationary_pim(
  duration: float, gravity_world: np.ndarray
) -> gtsam.PreintegratedCombinedMeasurements:
  n = int(duration * 1000)
  t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)
  samples = ImuSamples(
    t=t,
    gyro=np.zeros((n, 3), dtype=np.float64),
    accel=np.tile(-gravity_world, (n, 1)),
  )
  p = make_preintegration_params(
    gyro_noise_sigma=1e-4,
    accel_noise_sigma=1e-3,
    gyro_walk=1e-5,
    accel_walk=1e-4,
    gravity_world=(
      float(gravity_world[0]),
      float(gravity_world[1]),
      float(gravity_world[2]),
    ),
  )
  bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
  return integrate_samples(p, samples, bias)


def test_build_vi_graph_adds_velocity_bias_imu_factors() -> None:
  params = _params()
  extrinsic = gtsam.Pose3()
  gravity = np.array([0.0, 0.0, -9.81])
  poses = [gtsam.Pose3(), gtsam.Pose3()]
  vel = [np.zeros(3), np.zeros(3)]
  bias = [
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
  ]
  pim = [_stationary_pim(1.0, gravity)]

  graph, values = build_vi_graph(
    body_poses=poses,
    velocities=vel,
    biases=bias,
    preintegrated=pim,
    front=LensSetup(extrinsic=extrinsic, params=params),
    front_tracks={},
    front_points={},
    rear=None,
    rear_tracks=None,
    rear_points=None,
    sigma=1.0,
  )
  # 3 initial priors (X0, V0, B0) + 1 IMU factor = 4 factors; no projection.
  assert graph.size() == 4
  for i in range(2):
    assert values.exists(X(i))
    assert values.exists(V(i))
    assert values.exists(B(i))


def test_build_vi_graph_wrong_pim_count_raises() -> None:
  params = _params()
  extrinsic = gtsam.Pose3()
  poses = [gtsam.Pose3(), gtsam.Pose3(), gtsam.Pose3()]
  vel = [np.zeros(3)] * 3
  bias = [gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))] * 3
  # 3 poses need 2 PIMs.
  pim = [_stationary_pim(1.0, np.array([0.0, 0.0, -9.81]))]
  try:
    build_vi_graph(
      body_poses=poses,
      velocities=vel,
      biases=bias,
      preintegrated=pim,
      front=LensSetup(extrinsic=extrinsic, params=params),
      front_tracks={},
      front_points={},
      rear=None,
      rear_tracks=None,
      rear_points=None,
      sigma=1.0,
    )
  except ValueError as e:
    assert "N-1" in str(e)
    return
  raise AssertionError("expected ValueError for mismatched pim count")


def test_build_vi_graph_stationary_lm_converges() -> None:
  """2 poses, one IMU factor and no visual: LM keeps the state near-stationary."""
  params = _params()
  extrinsic = gtsam.Pose3()
  gravity = np.array([0.0, 0.0, -9.81])
  poses = [gtsam.Pose3(), gtsam.Pose3()]
  vel = [np.zeros(3), np.array([0.05, 0.0, 0.0])]  # slight init perturbation
  bias = [
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
  ]
  pim = [_stationary_pim(1.0, gravity)]
  graph, values = build_vi_graph(
    body_poses=poses,
    velocities=vel,
    biases=bias,
    preintegrated=pim,
    front=LensSetup(extrinsic=extrinsic, params=params),
    front_tracks={},
    front_points={},
    rear=None,
    rear_tracks=None,
    rear_points=None,
    sigma=1.0,
  )
  initial_err = graph.error(values)
  result = optimize_graph(graph, values, max_iterations=20)
  assert result.final_error <= initial_err
  v1 = np.asarray(result.values.atVector(V(1)))
  assert np.linalg.norm(v1) < 0.03
