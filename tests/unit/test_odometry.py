"""Tests for specialist.synthesis.odometry."""

from __future__ import annotations

import math

import gtsam
import numpy as np

from specialist.synthesis.odometry import (
  OdometryMeasurement,
  sample_keyframes,
  synthesize_odometry,
)
from specialist.synthesis.trajectory import SmoothTrajectory


def _trajectory(seed: int = 0) -> SmoothTrajectory:
  return SmoothTrajectory(
    duration=5.0, workspace_radius=1.0, rng=np.random.default_rng(seed)
  )


def test_sample_keyframes_count_matches_rate() -> None:
  traj = _trajectory()
  frames = sample_keyframes(traj, rate=10.0)
  assert len(frames) == 51  # 0, 0.1, ..., 5.0 inclusive


def test_sample_keyframes_returns_poses() -> None:
  traj = _trajectory()
  frames = sample_keyframes(traj, rate=10.0)
  assert all(isinstance(p, gtsam.Pose3) for p in frames)


def test_synthesize_odometry_length() -> None:
  traj = _trajectory()
  frames = sample_keyframes(traj, rate=10.0)
  meas = synthesize_odometry(
    frames,
    tracker_name="left",
    translation_sigma=0.003,
    rotation_sigma=math.radians(0.1),
    rng=np.random.default_rng(1),
  )
  assert len(meas) == len(frames) - 1
  assert all(isinstance(m, OdometryMeasurement) for m in meas)


def test_zero_noise_recovers_ground_truth() -> None:
  traj = _trajectory()
  frames = sample_keyframes(traj, rate=10.0)
  meas = synthesize_odometry(
    frames,
    tracker_name="left",
    translation_sigma=0.0,
    rotation_sigma=0.0,
    rng=np.random.default_rng(2),
  )
  for m in meas:
    gt = frames[m.from_index].between(frames[m.to_index])
    assert np.allclose(m.relative_pose.translation(), gt.translation())
    assert np.allclose(m.relative_pose.rotation().matrix(), gt.rotation().matrix())


def test_indices_cover_consecutive_pairs() -> None:
  traj = _trajectory()
  frames = sample_keyframes(traj, rate=10.0)
  meas = synthesize_odometry(
    frames,
    tracker_name="left",
    translation_sigma=0.0,
    rotation_sigma=0.0,
    rng=np.random.default_rng(3),
  )
  for i, m in enumerate(meas):
    assert m.from_index == i
    assert m.to_index == i + 1
    assert m.tracker_name == "left"


def test_noise_statistics() -> None:
  """Mean of tangent-space noise over many samples ≈ 0; std ≈ configured sigma."""
  traj = _trajectory()
  frames = sample_keyframes(traj, rate=10.0)
  t_sigma, r_sigma = 0.01, math.radians(0.5)
  meas = synthesize_odometry(
    frames,
    tracker_name="left",
    translation_sigma=t_sigma,
    rotation_sigma=r_sigma,
    rng=np.random.default_rng(4),
  )
  errors = []
  for m in meas:
    gt = frames[m.from_index].between(frames[m.to_index])
    noise_tangent = np.asarray(gtsam.Pose3.Logmap(gt.between(m.relative_pose)))
    errors.append(noise_tangent)
  errors_arr = np.asarray(errors)
  # [ω_x, ω_y, ω_z, v_x, v_y, v_z]
  assert np.max(np.abs(np.mean(errors_arr, axis=0))) < 0.05
  rot_std = np.std(errors_arr[:, :3])
  trans_std = np.std(errors_arr[:, 3:])
  assert abs(rot_std - r_sigma) / r_sigma < 0.3
  assert abs(trans_std - t_sigma) / t_sigma < 0.3


def test_same_seed_same_measurements() -> None:
  traj = _trajectory()
  frames = sample_keyframes(traj, rate=10.0)
  m1 = synthesize_odometry(
    frames,
    tracker_name="a",
    translation_sigma=0.01,
    rotation_sigma=math.radians(0.5),
    rng=np.random.default_rng(7),
  )
  m2 = synthesize_odometry(
    frames,
    tracker_name="a",
    translation_sigma=0.01,
    rotation_sigma=math.radians(0.5),
    rng=np.random.default_rng(7),
  )
  for a, b in zip(m1, m2, strict=True):
    assert np.allclose(a.relative_pose.translation(), b.relative_pose.translation())
    assert np.allclose(
      a.relative_pose.rotation().matrix(), b.relative_pose.rotation().matrix()
    )
