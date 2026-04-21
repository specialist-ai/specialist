"""Tests for specialist.synthesis.trajectory."""

from __future__ import annotations

import gtsam
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from specialist.synthesis.trajectory import SmoothTrajectory


def test_pose_returns_pose3() -> None:
  rng = np.random.default_rng(0)
  traj = SmoothTrajectory(duration=10.0, workspace_radius=1.0, rng=rng)
  p = traj.pose(5.0)
  assert isinstance(p, gtsam.Pose3)


def test_same_seed_same_trajectory() -> None:
  t1 = SmoothTrajectory(
    duration=10.0, workspace_radius=1.0, rng=np.random.default_rng(42)
  )
  t2 = SmoothTrajectory(
    duration=10.0, workspace_radius=1.0, rng=np.random.default_rng(42)
  )
  for t in (0.0, 2.5, 5.0, 7.5, 10.0):
    p1, p2 = t1.pose(t), t2.pose(t)
    assert np.allclose(p1.translation(), p2.translation())
    assert np.allclose(p1.rotation().matrix(), p2.rotation().matrix())


def test_different_seeds_differ() -> None:
  a = SmoothTrajectory(
    duration=10.0, workspace_radius=1.0, rng=np.random.default_rng(1)
  )
  b = SmoothTrajectory(
    duration=10.0, workspace_radius=1.0, rng=np.random.default_rng(2)
  )
  assert not np.allclose(a.pose(5.0).translation(), b.pose(5.0).translation())


def test_position_continuity() -> None:
  traj = SmoothTrajectory(
    duration=5.0, workspace_radius=1.0, rng=np.random.default_rng(1)
  )
  ts = np.linspace(0.0, 5.0, 501)
  last = np.asarray(traj.pose(ts[0]).translation())
  for t in ts[1:]:
    cur = np.asarray(traj.pose(t).translation())
    assert np.linalg.norm(cur - last) < 0.1
    last = cur


def test_linear_velocity_matches_finite_diff() -> None:
  traj = SmoothTrajectory(
    duration=5.0, workspace_radius=1.0, rng=np.random.default_rng(2)
  )
  t, dt = 2.0, 1e-4
  p_plus = np.asarray(traj.pose(t + dt).translation())
  p_minus = np.asarray(traj.pose(t - dt).translation())
  vel_fd = (p_plus - p_minus) / (2.0 * dt)
  vel_analytic = traj.velocity(t)
  assert np.allclose(vel_fd, vel_analytic, atol=1e-3)


def test_angular_velocity_is_body_frame() -> None:
  """Body-frame convention: R(t+dt) ≈ R(t) · exp([ω_body · dt]×)."""
  traj = SmoothTrajectory(
    duration=5.0, workspace_radius=1.0, rng=np.random.default_rng(3)
  )
  t, dt = 2.0, 1e-4
  r_plus = traj.pose(t + dt).rotation().matrix()
  r_minus = traj.pose(t - dt).rotation().matrix()
  omega_fd = Rotation.from_matrix(r_minus.T @ r_plus).as_rotvec() / (2.0 * dt)
  omega_analytic = traj.angular_velocity(t)
  assert np.allclose(omega_fd, omega_analytic, atol=1e-2)


def test_t_out_of_range_raises() -> None:
  traj = SmoothTrajectory(
    duration=5.0, workspace_radius=1.0, rng=np.random.default_rng(4)
  )
  with pytest.raises(ValueError):
    traj.pose(-0.01)
  with pytest.raises(ValueError):
    traj.pose(5.01)


def test_endpoints_are_evaluable() -> None:
  traj = SmoothTrajectory(
    duration=5.0, workspace_radius=1.0, rng=np.random.default_rng(5)
  )
  traj.pose(0.0)
  traj.pose(5.0)
  traj.velocity(0.0)
  traj.velocity(5.0)
  traj.angular_velocity(0.0)
  traj.angular_velocity(5.0)


def test_invalid_constructor_args() -> None:
  rng = np.random.default_rng(6)
  with pytest.raises(ValueError):
    SmoothTrajectory(duration=0.0, workspace_radius=1.0, rng=rng)
  with pytest.raises(ValueError):
    SmoothTrajectory(duration=1.0, workspace_radius=-0.1, rng=rng)
  with pytest.raises(ValueError):
    SmoothTrajectory(duration=1.0, workspace_radius=1.0, rng=rng, n_control_points=3)
