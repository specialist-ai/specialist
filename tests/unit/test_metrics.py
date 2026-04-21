"""Tests for specialist.eval.metrics."""

from __future__ import annotations

import math

import pytest

from specialist.eval.metrics import absolute_trajectory_error, relative_pose_error
from tests._helpers import pose


def test_ate_zero_for_identical_trajectories() -> None:
  gt = [pose((i * 0.1, 0, 0), (0, 0, math.radians(i))) for i in range(10)]
  err = absolute_trajectory_error(gt, gt)
  assert err.translation_rmse < 1e-12
  assert err.rotation_rmse < 1e-10


def test_ate_translation_only() -> None:
  gt = [pose((0, 0, 0), (0, 0, 0)), pose((1, 0, 0), (0, 0, 0))]
  est = [pose((0.1, 0, 0), (0, 0, 0)), pose((1.1, 0, 0), (0, 0, 0))]
  err = absolute_trajectory_error(est, gt)
  assert math.isclose(err.translation_rmse, 0.1, abs_tol=1e-9)
  assert err.rotation_rmse < 1e-10


def test_ate_rotation_only() -> None:
  gt = [pose((0, 0, 0), (0, 0, 0)), pose((0, 0, 0), (0, 0, math.radians(30)))]
  est = [
    pose((0, 0, 0), (0, 0, math.radians(5))),
    pose((0, 0, 0), (0, 0, math.radians(35))),
  ]
  err = absolute_trajectory_error(est, gt)
  assert err.translation_rmse < 1e-9
  assert math.isclose(err.rotation_rmse, math.radians(5.0), abs_tol=1e-8)


def test_ate_length_mismatch_raises() -> None:
  with pytest.raises(ValueError):
    absolute_trajectory_error([pose((0, 0, 0), (0, 0, 0))], [])


def test_rpe_zero_for_identical_pairs() -> None:
  gt_a = [pose((0, 0, 0), (0, 0, 0)), pose((0.1, 0, 0), (0, 0, 0))]
  gt_b = [pose((0.3, 0, 0), (0, 0, 0)), pose((0.4, 0, 0), (0, 0, 0))]
  err = relative_pose_error(gt_a, gt_b, gt_a, gt_b)
  assert err.translation_rmse < 1e-12
  assert err.rotation_rmse < 1e-10


def test_rpe_invariant_to_common_global_transform() -> None:
  """A rigid transform applied to both est_a and est_b leaves the pair-relative error unchanged."""
  gt_a = [pose((0, 0, 0), (0, 0, 0)), pose((0.1, 0, 0), (0, 0, math.radians(10)))]
  gt_b = [
    pose((0.3, 0, 0), (0, 0, 0)),
    pose((0.4, 0.1, 0), (0, 0, math.radians(5))),
  ]
  shift = pose((1, 2, 3), (0, 0, math.radians(45)))
  est_a = [shift.compose(p) for p in gt_a]
  est_b = [shift.compose(p) for p in gt_b]
  err = relative_pose_error(est_a, est_b, gt_a, gt_b)
  assert err.translation_rmse < 1e-10
  assert err.rotation_rmse < 1e-9
