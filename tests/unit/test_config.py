"""Tests for specialist.config dataclasses."""

from __future__ import annotations

import math

import tyro

from specialist.config import NoiseConfig, ScenarioConfig, SceneConfig


def test_defaults_instantiate() -> None:
  noise = NoiseConfig()
  scene = SceneConfig()
  scenario = ScenarioConfig()
  assert noise.odometry_translation > 0
  assert scene.fov > 0
  assert scenario.duration > 0
  assert scenario.rate > 0
  assert len(scenario.tracker_names) >= 1


def test_noise_config_matches_spec_table() -> None:
  n = NoiseConfig()
  assert n.odometry_translation == 0.003
  assert math.isclose(n.odometry_rotation, math.radians(0.1))
  assert n.anchor_translation == 0.01
  assert math.isclose(n.anchor_rotation, math.radians(0.5))
  assert n.cross_tracker_translation == 0.005
  assert math.isclose(n.cross_tracker_rotation, math.radians(1.0))
  assert n.anchor_prior_translation == 1e-6
  assert n.anchor_prior_rotation == 1e-6


def test_tyro_roundtrip_empty_args_gives_defaults() -> None:
  parsed = tyro.cli(ScenarioConfig, args=[])
  assert parsed == ScenarioConfig()


def test_tyro_override_scalar() -> None:
  parsed = tyro.cli(ScenarioConfig, args=["--seed", "42", "--duration", "12.5"])
  assert parsed.seed == 42
  assert parsed.duration == 12.5


def test_tyro_override_nested_noise_sigma() -> None:
  parsed = tyro.cli(
    ScenarioConfig,
    args=["--noise.odometry-translation", "0.01"],
  )
  assert parsed.noise.odometry_translation == 0.01
  assert parsed.noise.odometry_rotation == NoiseConfig().odometry_rotation
