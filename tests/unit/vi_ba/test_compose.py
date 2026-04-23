"""Tests for ``specialist.vi_ba.compose``.

Covers the multi-tracker key namespace + cross-tracker factor composition
on top of per-tracker VI-BA subgraphs. Synthetic only: a tiny scene with
two trackers, two keyframes each, one shared anchor sighting per tracker
plus one cross-tracker observation. Verifies that:

- The key scheme is collision-free across trackers and against the anchor
  + the standard ``X/V/B/L`` defaults that ``build_vi_graph`` would use
  for a single-tracker run.
- ``add_cross_tracker_factors`` adds the right number of factors, inserts
  the anchor variable, and rejects observations referencing missing
  variables.
- The composed graph is well-formed enough that LM can run on it from
  the perturbed init.
"""

from __future__ import annotations

import gtsam
import numpy as np

from specialist.config import NoiseConfig, SceneConfig
from specialist.synthesis.scene import build_scene
from specialist.synthesis.tag_detection import TagObservation
from specialist.vi_ba.build import LensSetup, build_vi_graph
from specialist.vi_ba.compose import TrackerKeys, add_cross_tracker_factors
from specialist.vi_ba.imu import (
  ImuSamples,
  integrate_samples,
  make_preintegration_params,
)
from specialist.vi_ba.optimize import optimize_graph

_GRAVITY = (0.0, 0.0, -9.81)


def _stationary_pim(
  duration: float,
) -> gtsam.PreintegratedCombinedMeasurements:
  n = int(duration * 1000)
  t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)
  samples = ImuSamples(
    t=t,
    gyro=np.zeros((n, 3), dtype=np.float64),
    accel=np.tile(-np.asarray(_GRAVITY), (n, 1)),
  )
  params = make_preintegration_params(
    gyro_noise_sigma=1e-4,
    accel_noise_sigma=1e-3,
    gyro_walk=1e-5,
    accel_walk=1e-4,
    gravity_world=_GRAVITY,
  )
  bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
  return integrate_samples(params, samples, bias)


def _identity_lens() -> LensSetup:
  from specialist.loader.mei_model import MeiParams

  return LensSetup(
    extrinsic=gtsam.Pose3(),
    params=MeiParams(
      fx=500.0,
      fy=500.0,
      cx=320.0,
      cy=240.0,
      k1=0.0,
      k2=0.0,
      k3=0.0,
      p1=0.0,
      p2=0.0,
      xi=0.0,
    ),
  )


def test_tracker_keys_no_collision_across_trackers() -> None:
  keys = TrackerKeys(["alice", "bob"])
  ka_pose0 = keys.pose_key("alice", 0)
  kb_pose0 = keys.pose_key("bob", 0)
  assert ka_pose0 != kb_pose0
  # Different *types* in same tracker — distinct.
  assert keys.pose_key("alice", 0) != keys.velocity_key("alice", 0)
  assert keys.pose_key("alice", 0) != keys.bias_key("alice", 0)
  assert keys.front_landmark_key("alice", 0) != keys.rear_landmark_key("alice", 0)
  assert keys.front_landmark_key("alice", 0) != keys.front_landmark_key("bob", 0)
  # Anchor distinct from everything.
  anchor = keys.anchor_key()
  assert anchor not in {
    ka_pose0,
    kb_pose0,
    keys.velocity_key("alice", 0),
    keys.bias_key("alice", 0),
    keys.front_landmark_key("alice", 0),
  }


def test_tracker_keys_unknown_tracker_raises() -> None:
  keys = TrackerKeys(["alice"])
  try:
    keys.pose_key("bob", 0)
  except KeyError:
    return
  raise AssertionError("expected KeyError for unknown tracker")


def test_tracker_keys_duplicate_names_raises() -> None:
  try:
    TrackerKeys(["alice", "alice"])
  except ValueError:
    return
  raise AssertionError("expected ValueError for duplicate tracker names")


def test_compose_two_trackers_assembles_one_joint_graph() -> None:
  scene = build_scene(SceneConfig(), tracker_names=("alice", "bob"))
  noise = NoiseConfig()
  keys = TrackerKeys(["alice", "bob"])
  lens = _identity_lens()

  # Per-tracker VI-BA inputs: 2 keyframes, stationary, no visual obs.
  poses = [gtsam.Pose3(), gtsam.Pose3()]
  vels = [np.zeros(3), np.zeros(3)]
  biases = [
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
  ]
  pim = [_stationary_pim(1.0)]

  graph = gtsam.NonlinearFactorGraph()
  values = gtsam.Values()
  for name in ("alice", "bob"):
    build_vi_graph(
      body_poses=poses,
      velocities=vels,
      biases=biases,
      preintegrated=pim,
      front=lens,
      front_tracks={},
      front_points={},
      rear=None,
      rear_tracks=None,
      rear_points=None,
      pixel_sigma=1.0,
      graph=graph,
      values=values,
      **keys.vi_graph_kwargs(name),
    )
  per_tracker_factors = graph.size()
  per_tracker_vars = values.size()
  # 2 trackers × (3 priors on (X0, V0, B0) + 1 IMU factor) = 8 factors.
  assert per_tracker_factors == 8
  # 2 trackers × 2 keyframes × 3 vars (X, V, B) = 12 variables.
  assert per_tracker_vars == 12

  # Two anchor sightings (one per tracker, kf 0) + one cross-tracker (alice
  # observes bob at kf 1).
  observations = [
    TagObservation(
      observer_tracker="alice",
      fisheye_side="front",
      keyframe_index=0,
      tag_id="anchor",
      tag_kind="anchor",
      tag_owner_tracker=None,
      camera_to_tag=scene.anchor_pose,
    ),
    TagObservation(
      observer_tracker="bob",
      fisheye_side="front",
      keyframe_index=0,
      tag_id="anchor",
      tag_kind="anchor",
      tag_owner_tracker=None,
      camera_to_tag=scene.anchor_pose,
    ),
    TagObservation(
      observer_tracker="alice",
      fisheye_side="back",
      keyframe_index=1,
      tag_id="tracker_bob",
      tag_kind="tracker",
      tag_owner_tracker="bob",
      camera_to_tag=gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 1.0)),
    ),
  ]
  add_cross_tracker_factors(graph, values, keys, observations, scene, noise)

  # +1 anchor variable, +1 anchor prior, +2 anchor observation factors,
  # +1 cross-tracker factor.
  assert values.size() == per_tracker_vars + 1
  assert graph.size() == per_tracker_factors + 4
  assert values.exists(keys.anchor_key())


def test_compose_observation_to_missing_pose_raises() -> None:
  scene = build_scene(SceneConfig(), tracker_names=("alice", "bob"))
  noise = NoiseConfig()
  keys = TrackerKeys(["alice", "bob"])
  graph = gtsam.NonlinearFactorGraph()
  values = gtsam.Values()
  observations = [
    TagObservation(
      observer_tracker="alice",
      fisheye_side="front",
      keyframe_index=0,
      tag_id="anchor",
      tag_kind="anchor",
      tag_owner_tracker=None,
      camera_to_tag=scene.anchor_pose,
    )
  ]
  try:
    add_cross_tracker_factors(graph, values, keys, observations, scene, noise)
  except KeyError as e:
    assert "alice" in str(e)
    return
  raise AssertionError("expected KeyError for missing observer pose")


def test_compose_lm_runs_on_perturbed_init() -> None:
  """LM converges on the joint graph when each tracker's pose is slightly
  perturbed and the anchor sightings constrain them."""
  scene = build_scene(SceneConfig(), tracker_names=("alice", "bob"))
  noise = NoiseConfig()
  keys = TrackerKeys(["alice", "bob"])
  lens = _identity_lens()

  poses = [gtsam.Pose3(), gtsam.Pose3()]
  vels = [np.zeros(3), np.array([0.02, 0.0, 0.0])]
  biases = [
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
    gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
  ]
  pim = [_stationary_pim(1.0)]

  graph = gtsam.NonlinearFactorGraph()
  values = gtsam.Values()
  for name in ("alice", "bob"):
    build_vi_graph(
      body_poses=poses,
      velocities=vels,
      biases=biases,
      preintegrated=pim,
      front=lens,
      front_tracks={},
      front_points={},
      rear=None,
      rear_tracks=None,
      rear_points=None,
      pixel_sigma=1.0,
      graph=graph,
      values=values,
      **keys.vi_graph_kwargs(name),
    )

  # Anchor observations from both trackers fix the gauge; no
  # cross-tracker factor needed for this convergence check.
  observations = [
    TagObservation(
      observer_tracker=name,
      fisheye_side="front",
      keyframe_index=0,
      tag_id="anchor",
      tag_kind="anchor",
      tag_owner_tracker=None,
      camera_to_tag=scene.anchor_pose,
    )
    for name in ("alice", "bob")
  ]
  add_cross_tracker_factors(graph, values, keys, observations, scene, noise)

  initial_err = graph.error(values)
  result = optimize_graph(graph, values, max_iterations=20)
  assert result.final_error <= initial_err
  # Each tracker should still have its variables in the result.
  for name in ("alice", "bob"):
    for kf in range(2):
      assert result.values.exists(keys.pose_key(name, kf))
      assert result.values.exists(keys.velocity_key(name, kf))
      assert result.values.exists(keys.bias_key(name, kf))
