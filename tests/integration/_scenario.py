"""Integration-test helper: a scenario with visibility-friendly default geometry.

The end-to-end runner lives in ``specialist.runner``; this module only
contributes a test-specific default ``ScenarioConfig``, a near-zero noise
factory, and a set of observation-filter factories shared by the blackout /
cross-tracker-only tests.
"""

from __future__ import annotations

from collections.abc import Callable

from specialist.config import NoiseConfig, ScenarioConfig
from specialist.runner import ScenarioRun, run_scenario
from specialist.synthesis.tag_detection import TagObservation

ObservationFilter = Callable[[TagObservation], bool]

__all__ = [
  "ObservationFilter",
  "ScenarioRun",
  "anchor_endpoint_window",
  "default_visible_scenario",
  "near_zero_noise",
  "only_anchor_at_first_keyframe",
  "run_scenario",
]


def default_visible_scenario(
  *,
  seed: int,
  duration: float = 6.0,
  rate: float = 10.0,
  noise: NoiseConfig | None = None,
) -> ScenarioConfig:
  """Scenario with the default floor-anchor + workspace-above-origin geometry."""
  return ScenarioConfig(
    tracker_names=("left", "right"),
    duration=duration,
    rate=rate,
    seed=seed,
    noise=noise if noise is not None else NoiseConfig(),
  )


def near_zero_noise() -> NoiseConfig:
  """Noise sigmas tiny enough that a noiseless recovery test should hit 1e-6."""
  return NoiseConfig(
    odometry_translation=1e-10,
    odometry_rotation=1e-9,
    anchor_translation=1e-10,
    anchor_rotation=1e-9,
    cross_tracker_translation=1e-10,
    cross_tracker_rotation=1e-9,
    anchor_prior_translation=1e-10,
    anchor_prior_rotation=1e-10,
  )


def only_anchor_at_first_keyframe(obs: TagObservation) -> bool:
  """Keep all cross-tracker observations; keep only the keyframe-0 anchor sightings."""
  if obs.tag_kind != "anchor":
    return True
  return obs.keyframe_index == 0


def anchor_endpoint_window(window: int, n_keyframes: int) -> ObservationFilter:
  """Keep anchor observations only in ``[0, window)`` and ``[n_keyframes - window, n_keyframes)``."""

  def keep(obs: TagObservation) -> bool:
    if obs.tag_kind != "anchor":
      return True
    return obs.keyframe_index < window or obs.keyframe_index >= n_keyframes - window

  return keep
