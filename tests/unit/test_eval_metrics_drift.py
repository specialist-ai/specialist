"""Tests for specialist.vi_ba.eval_metrics.trajectory_position_drift."""

from __future__ import annotations

import math

import numpy as np

from specialist.vi_ba.eval_metrics import trajectory_position_drift


def _traj(points: list[tuple[float, float, float]]) -> list[np.ndarray]:
  return [np.asarray(p, dtype=np.float64) for p in points]


def test_drift_zero_for_identical_trajectories() -> None:
  traj = _traj([(i * 0.1, 0.0, 0.0) for i in range(5)])
  kf_times = np.arange(5, dtype=np.float64)
  d = trajectory_position_drift(traj, traj, kf_times)
  assert all(v < 1e-12 for v in d["per_kf"])
  assert d["overall"]["max"] < 1e-12
  assert int(d["overall"]["n"]) == 5


def test_drift_constant_offset() -> None:
  ref = _traj([(0.0, 0.0, 0.0)] * 4)
  cand = _traj([(0.03, 0.04, 0.0)] * 4)  # 0.05 m offset
  kf_times = np.arange(4, dtype=np.float64)
  d = trajectory_position_drift(cand, ref, kf_times)
  assert math.isclose(d["overall"]["max"], 0.05, abs_tol=1e-9)
  assert math.isclose(d["overall"]["median"], 0.05, abs_tol=1e-9)


def test_window_restricts_to_blackout_kfs() -> None:
  # Drift only inside t in [2, 3]; zero elsewhere.
  ref = _traj([(0.0, 0.0, 0.0)] * 5)
  cand = _traj(
    [
      (0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0),
      (0.1, 0.0, 0.0),
      (0.2, 0.0, 0.0),
      (0.0, 0.0, 0.0),
    ]
  )
  kf_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
  d = trajectory_position_drift(cand, ref, kf_times, window=(2.0, 3.0))
  assert int(d["window"]["n"]) == 2
  assert math.isclose(d["window"]["max"], 0.2, abs_tol=1e-9)
  assert math.isclose(d["window"]["median"], 0.15, abs_tol=1e-9)
  # Overall includes the zero-drift kfs, so its median is below the window's.
  assert d["overall"]["median"] < d["window"]["median"]


def test_handles_unequal_lengths() -> None:
  ref = _traj([(0.0, 0.0, 0.0)] * 6)
  cand = _traj([(0.01, 0.0, 0.0)] * 4)
  kf_times = np.arange(6, dtype=np.float64)
  d = trajectory_position_drift(cand, ref, kf_times)
  assert int(d["overall"]["n"]) == 4  # min of the two lengths
