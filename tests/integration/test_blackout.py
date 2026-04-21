"""Blackout integration test: anchor visible only at the ends of the trajectory.

Mid-trajectory drift must stay bounded and the final per-tracker ATE should
be close to what we'd get with full anchor visibility — the anchor
observations at the endpoints, combined with the dense cross-tracker
observations and odometry, keep the estimate pinned.
"""

from __future__ import annotations

import math

from specialist.eval.metrics import absolute_trajectory_error

from ._scenario import (
  anchor_endpoint_window,
  default_visible_scenario,
  run_scenario,
)

BLACKOUT_WINDOW = 3  # keep anchor observations in [0, W) and [N-W, N)
DRIFT_MULTIPLIER = 5.0  # allow up to 5× the full-anchor ATE before we call it drift


def test_blackout_drift_stays_bounded() -> None:
  scenario = default_visible_scenario(seed=1)
  baseline = run_scenario(scenario)

  n = baseline.n_keyframes
  blackout = run_scenario(
    scenario, observation_filter=anchor_endpoint_window(BLACKOUT_WINDOW, n)
  )

  baseline_anchor_count = sum(
    1 for o in baseline.observations if o.tag_kind == "anchor"
  )
  dropped = baseline_anchor_count - len(blackout.anchor_observations)
  assert dropped > 0, "blackout scenario did not actually drop any anchor observations"

  for name in scenario.tracker_names:
    base_err = absolute_trajectory_error(
      baseline.estimated_keyframes[name], baseline.ground_truth_keyframes[name]
    )
    blackout_err = absolute_trajectory_error(
      blackout.estimated_keyframes[name], blackout.ground_truth_keyframes[name]
    )
    assert (
      blackout_err.translation_rmse
      < DRIFT_MULTIPLIER * base_err.translation_rmse + 0.05
    ), (
      name,
      base_err,
      blackout_err,
    )
    assert (
      blackout_err.rotation_rmse
      < DRIFT_MULTIPLIER * base_err.rotation_rmse + math.radians(1.0)
    ), (
      name,
      base_err,
      blackout_err,
    )


def test_blackout_mid_trajectory_has_bounded_error() -> None:
  scenario = default_visible_scenario(seed=1)
  n = scenario.duration * scenario.rate
  run = run_scenario(
    scenario,
    observation_filter=anchor_endpoint_window(BLACKOUT_WINDOW, int(round(n)) + 1),
  )

  middle = run.n_keyframes // 2
  for name in scenario.tracker_names:
    est_mid = run.estimated_keyframes[name][middle]
    gt_mid = run.ground_truth_keyframes[name][middle]
    err = absolute_trajectory_error([est_mid], [gt_mid])
    # Mid-trajectory absolute error should still be under ~10 cm even with blackout.
    assert err.translation_rmse < 0.1, (name, err)
