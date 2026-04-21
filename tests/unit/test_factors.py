"""Factor-construction tests: minimal graphs recover ground truth with noiseless measurements."""

from __future__ import annotations

import math

import gtsam
import numpy as np

from specialist.graph.factors import (
  make_anchor_observation_factor,
  make_anchor_prior,
  make_cross_tracker_factor,
  make_odometry_factor,
)
from tests._helpers import pose


def _optimize(graph: gtsam.NonlinearFactorGraph, initial: gtsam.Values) -> gtsam.Values:
  return gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()


_SMALL_ROT = math.radians(0.1)
_PI = math.pi


def test_odometry_factor_recovers_secondpose() -> None:
  key0, key1 = gtsam.symbol("x", 0), gtsam.symbol("x", 1)
  gt0 = pose((0, 0, 0), (0, 0, 0))
  gt1 = pose((1, 0, 0), (0, 0, math.radians(45)))
  gt_relative = gt0.between(gt1)

  graph = gtsam.NonlinearFactorGraph()
  graph.add(make_anchor_prior(key0, gt0, translation_sigma=1e-6, rotation_sigma=1e-6))
  graph.add(make_odometry_factor(key0, key1, gt_relative, 1e-3, _SMALL_ROT))
  initial = gtsam.Values()
  initial.insert(key0, gtsam.Pose3())
  initial.insert(key1, gtsam.Pose3())
  result = _optimize(graph, initial)

  assert np.allclose(result.atPose3(key1).translation(), gt1.translation(), atol=1e-5)
  assert np.allclose(
    result.atPose3(key1).rotation().matrix(), gt1.rotation().matrix(), atol=1e-5
  )


def test_anchor_prior_pins_variable() -> None:
  key = gtsam.symbol("A", 0)
  anchor_gt = pose((0.5, -0.25, 1.0), (0, 0, math.radians(30)))
  graph = gtsam.NonlinearFactorGraph()
  graph.add(
    make_anchor_prior(key, anchor_gt, translation_sigma=1e-6, rotation_sigma=1e-6)
  )
  initial = gtsam.Values()
  initial.insert(key, gtsam.Pose3())
  result = _optimize(graph, initial)
  assert np.allclose(
    result.atPose3(key).translation(), anchor_gt.translation(), atol=1e-5
  )


def test_anchor_observation_recovers_tracker() -> None:
  tracker_key = gtsam.symbol("a", 0)
  anchor_key = gtsam.symbol("A", 0)
  anchor_gt = pose((0, 0, 1.0), (_PI, 0, 0))
  tracker_gt = pose((0.2, 0.1, 0), (0, 0, math.radians(15)))
  camera_extrinsic = pose((0.0, 0.0, 0.02), (0, 0, 0))
  camera_world = tracker_gt.compose(camera_extrinsic)
  cam_to_anchor = camera_world.between(anchor_gt)

  graph = gtsam.NonlinearFactorGraph()
  graph.add(
    make_anchor_prior(
      anchor_key, anchor_gt, translation_sigma=1e-6, rotation_sigma=1e-6
    )
  )
  graph.add(
    make_anchor_observation_factor(
      tracker_key,
      anchor_key,
      cam_to_anchor,
      camera_extrinsic,
      translation_sigma=1e-3,
      rotation_sigma=_SMALL_ROT,
    )
  )
  initial = gtsam.Values()
  initial.insert(tracker_key, gtsam.Pose3())
  initial.insert(anchor_key, gtsam.Pose3())
  result = _optimize(graph, initial)
  assert np.allclose(
    result.atPose3(tracker_key).translation(), tracker_gt.translation(), atol=1e-5
  )
  assert np.allclose(
    result.atPose3(tracker_key).rotation().matrix(),
    tracker_gt.rotation().matrix(),
    atol=1e-5,
  )


def test_cross_tracker_factor_links_two_trackers() -> None:
  observer_key = gtsam.symbol("a", 0)
  owner_key = gtsam.symbol("b", 0)
  observer_gt = pose((0, 0, 0), (0, 0, 0))
  owner_gt = pose((0.3, 0.1, 0), (0, 0, math.radians(-10)))
  observer_cam_ext = pose((0.0, 0.0, 0.02), (0, 0, 0))
  tag_on_owner_ext = pose((0.0, 0.05, 0.0), (_PI, 0, 0))
  observer_cam_world = observer_gt.compose(observer_cam_ext)
  owner_tag_world = owner_gt.compose(tag_on_owner_ext)
  cam_to_tag = observer_cam_world.between(owner_tag_world)

  graph = gtsam.NonlinearFactorGraph()
  # Pin observer tightly so owner is determined by the cross-tracker factor alone.
  graph.add(
    make_anchor_prior(
      observer_key, observer_gt, translation_sigma=1e-6, rotation_sigma=1e-6
    )
  )
  graph.add(
    make_cross_tracker_factor(
      observer_key,
      owner_key,
      cam_to_tag,
      observer_cam_ext,
      tag_on_owner_ext,
      translation_sigma=1e-3,
      rotation_sigma=_SMALL_ROT,
    )
  )
  initial = gtsam.Values()
  initial.insert(observer_key, gtsam.Pose3())
  initial.insert(owner_key, gtsam.Pose3())
  result = _optimize(graph, initial)
  assert np.allclose(
    result.atPose3(owner_key).translation(), owner_gt.translation(), atol=1e-5
  )
  assert np.allclose(
    result.atPose3(owner_key).rotation().matrix(),
    owner_gt.rotation().matrix(),
    atol=1e-5,
  )


def test_odometry_factor_residual_zero_at_ground_truth() -> None:
  key0, key1 = gtsam.symbol("x", 0), gtsam.symbol("x", 1)
  gt0 = pose((0, 0, 0), (0, 0, 0))
  gt1 = pose((1, 0, 0), (0, 0, math.radians(45)))
  factor = make_odometry_factor(key0, key1, gt0.between(gt1), 1e-3, _SMALL_ROT)
  values = gtsam.Values()
  values.insert(key0, gt0)
  values.insert(key1, gt1)
  err = factor.error(values)
  assert err < 1e-12


def test_anchor_observation_residual_nonzero_when_perturbed() -> None:
  tracker_key, anchor_key = gtsam.symbol("a", 0), gtsam.symbol("A", 0)
  anchor_gt = pose((0, 0, 1.0), (_PI, 0, 0))
  tracker_gt = pose((0, 0, 0), (0, 0, 0))
  camera_extrinsic = gtsam.Pose3()
  cam_to_anchor = tracker_gt.compose(camera_extrinsic).between(anchor_gt)
  factor = make_anchor_observation_factor(
    tracker_key,
    anchor_key,
    cam_to_anchor,
    camera_extrinsic,
    translation_sigma=1e-3,
    rotation_sigma=_SMALL_ROT,
  )
  good = gtsam.Values()
  good.insert(tracker_key, tracker_gt)
  good.insert(anchor_key, anchor_gt)
  assert factor.error(good) < 1e-12
  bad = gtsam.Values()
  bad.insert(tracker_key, pose((0.1, 0, 0), (0, 0, 0)))
  bad.insert(anchor_key, anchor_gt)
  assert factor.error(bad) > 1.0
