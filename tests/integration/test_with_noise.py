"""With-noise integration test: ATE must stay within 1.5× the Cramér-Rao bound."""

from __future__ import annotations

from specialist.eval.crb import crb_trajectory_bounds
from specialist.eval.metrics import absolute_trajectory_error

from ._scenario import default_visible_scenario, run_scenario

TOLERANCE_MULTIPLIER = 1.5


def test_ate_within_tolerance_of_crb() -> None:
  scenario = default_visible_scenario(seed=0)
  run = run_scenario(scenario)

  keys_per_tracker = {
    name: [run.key_map.tracker_pose_key(name, i) for i in range(run.n_keyframes)]
    for name in scenario.tracker_names
  }
  crb = crb_trajectory_bounds(run.graph, run.ground_truth_values, keys_per_tracker)

  for name in scenario.tracker_names:
    err = absolute_trajectory_error(
      run.estimated_keyframes[name], run.ground_truth_keyframes[name]
    )
    bound = crb[name]
    assert err.translation_rmse < TOLERANCE_MULTIPLIER * bound.translation_rmse, (
      name,
      err,
      bound,
    )
    assert err.rotation_rmse < TOLERANCE_MULTIPLIER * bound.rotation_rmse, (
      name,
      err,
      bound,
    )


def test_optimization_reduces_error() -> None:
  scenario = default_visible_scenario(seed=0)
  run = run_scenario(scenario)
  assert run.optimization.final_error < run.optimization.initial_error
  assert run.optimization.iterations > 0


def test_anchor_stays_near_prior() -> None:
  scenario = default_visible_scenario(seed=0)
  run = run_scenario(scenario)
  anchor_est = run.optimization.values.atPose3(run.key_map.anchor_key())
  gt = run.scene.anchor_pose
  import numpy as np

  # Anchor prior is 1e-6 tight; the optimized anchor should be practically identical to GT.
  assert np.allclose(anchor_est.translation(), gt.translation(), atol=1e-4)
