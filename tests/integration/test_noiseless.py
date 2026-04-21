"""Noiseless integration test: the full pipeline must recover ground truth to 1e-6."""

from __future__ import annotations

import numpy as np

from specialist.eval.metrics import absolute_trajectory_error

from ._scenario import default_visible_scenario, near_zero_noise, run_scenario


def test_noiseless_recovery_to_1e6() -> None:
  scenario = default_visible_scenario(seed=0, noise=near_zero_noise())
  run = run_scenario(scenario)

  # There must be meaningful tag observations or the optimization is trivial.
  assert len(run.anchor_observations) > 0, "scenario produced no anchor observations"
  assert len(run.cross_tracker_observations) > 0, (
    "scenario produced no cross-tracker observations"
  )

  for name in scenario.tracker_names:
    err = absolute_trajectory_error(
      run.estimated_keyframes[name], run.ground_truth_keyframes[name]
    )
    assert err.translation_rmse < 1e-6, (name, err)
    assert err.rotation_rmse < 2e-6, (name, err)


def test_noiseless_optimization_converges_quickly() -> None:
  scenario = default_visible_scenario(seed=0, noise=near_zero_noise())
  run = run_scenario(scenario)
  assert run.optimization.iterations < 20
  assert run.optimization.final_error < run.optimization.initial_error


def test_noiseless_anchor_recovered_at_prior() -> None:
  scenario = default_visible_scenario(seed=0, noise=near_zero_noise())
  run = run_scenario(scenario)
  anchor_est = run.optimization.values.atPose3(run.key_map.anchor_key())
  anchor_gt = run.scene.anchor_pose
  assert np.allclose(anchor_est.translation(), anchor_gt.translation(), atol=1e-6)
