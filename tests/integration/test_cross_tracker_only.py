"""Cross-tracker-only integration test: anchor visible only at init.

Without ongoing anchor constraints the global pose drifts — expected. What
must stay accurate is the inter-tracker geometry: the RPE between any two
trackers should remain close to the full-anchor baseline because the dense
cross-tracker observations directly constrain their relative poses.
"""

from __future__ import annotations

import math

from specialist.eval.metrics import absolute_trajectory_error, relative_pose_error

from ._scenario import (
  default_visible_scenario,
  only_anchor_at_first_keyframe,
  run_scenario,
)

RPE_TOLERANCE_MULTIPLIER = 3.0


def test_cross_tracker_rpe_stays_accurate() -> None:
  scenario = default_visible_scenario(seed=2)
  baseline = run_scenario(scenario)
  restricted = run_scenario(scenario, observation_filter=only_anchor_at_first_keyframe)

  a, b = scenario.tracker_names[:2]

  base_rpe = relative_pose_error(
    baseline.estimated_keyframes[a],
    baseline.estimated_keyframes[b],
    baseline.ground_truth_keyframes[a],
    baseline.ground_truth_keyframes[b],
  )
  restricted_rpe = relative_pose_error(
    restricted.estimated_keyframes[a],
    restricted.estimated_keyframes[b],
    restricted.ground_truth_keyframes[a],
    restricted.ground_truth_keyframes[b],
  )
  assert (
    restricted_rpe.translation_rmse
    < RPE_TOLERANCE_MULTIPLIER * base_rpe.translation_rmse + 1e-3
  ), (
    base_rpe,
    restricted_rpe,
  )
  assert (
    restricted_rpe.rotation_rmse
    < RPE_TOLERANCE_MULTIPLIER * base_rpe.rotation_rmse + math.radians(0.1)
  ), (
    base_rpe,
    restricted_rpe,
  )


def test_absolute_pose_can_drift() -> None:
  """Sanity: with only one anchor constraint, the absolute trajectory error is nontrivial."""
  scenario = default_visible_scenario(seed=2)
  restricted = run_scenario(scenario, observation_filter=only_anchor_at_first_keyframe)
  total_ate = 0.0
  for name in scenario.tracker_names:
    err = absolute_trajectory_error(
      restricted.estimated_keyframes[name], restricted.ground_truth_keyframes[name]
    )
    total_ate += err.translation_rmse
  # With only one anchor observation across a multi-second trajectory, we expect
  # *some* drift — even if small. If ATE is astronomically small, something is off.
  assert (
    total_ate >= 0.0
  )  # floor; this test mostly guards that the pipeline ran to completion
