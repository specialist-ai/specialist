"""End-to-end IMU preintegration on a real stationary capture.

A 30-second IMU fixture clipped from an Insta360 stationary ``.insv`` (under
``tests/fixtures/vi_ba/``) seeds the bias from its first 3 seconds, and the
integrated rotation over the remaining 27 seconds stays inside a 0.3° budget.
The stationary gate of 1° / 5 min allows 0.09° over 27 s.
"""

from __future__ import annotations

import math
from pathlib import Path

import gtsam
import numpy as np

from specialist.vi_ba.imu import (
  ImuSamples,
  integrate_samples,
  make_preintegration_params,
)

_FIXTURE = (
  Path(__file__).resolve().parents[1] / "fixtures" / "vi_ba" / "imu_stationary_30s.npz"
)


def _load_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  data = np.load(_FIXTURE)
  return (
    data["t"].astype(np.float64),
    data["gyro_deg"].astype(np.float64),
    data["accel"].astype(np.float64),
  )


def test_stationary_orientation_drift_within_budget() -> None:
  t, gyro_deg, accel = _load_fixture()
  samples = ImuSamples(t=t, gyro=np.deg2rad(gyro_deg), accel=accel)
  assert len(samples) > 20_000  # 1 kHz × 30 s minus small head/tail

  # The camera sat on a table with body +y up; the world frame is chosen so
  # gravity points along world −y.
  g_magnitude = float(np.linalg.norm(accel.mean(axis=0)))
  assert 9.6 < g_magnitude < 10.0
  gravity_world = np.array([0.0, -g_magnitude, 0.0], dtype=np.float64)
  initial_rotation = gtsam.Rot3()  # body == world at t = 0

  # Over a stationary window the mean gyro reading is the gyro bias and the
  # mean accel reading is gravity in the body frame, so the accel bias is
  # mean(accel) − R^T · g_world.
  seed_mask = samples.t < (samples.t[0] + 3.0)
  gyro_bias = samples.gyro[seed_mask].mean(axis=0)
  accel_mean = samples.accel[seed_mask].mean(axis=0)
  g_body = initial_rotation.matrix().T @ gravity_world
  accel_bias = accel_mean + g_body
  bias = gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)

  tail_mask = samples.t >= (samples.t[0] + 3.0)
  tail = ImuSamples(
    t=samples.t[tail_mask],
    gyro=samples.gyro[tail_mask],
    accel=samples.accel[tail_mask],
  )

  params = make_preintegration_params(
    gyro_noise_sigma=3e-3,  # rad/√s white-noise density
    accel_noise_sigma=1e-2,
    gyro_walk=1e-4,
    accel_walk=1e-3,
    gravity_world=(
      float(gravity_world[0]),
      float(gravity_world[1]),
      float(gravity_world[2]),
    ),
  )
  pim = integrate_samples(params, tail, bias)

  delta_r = pim.deltaRij()
  drift = float(np.linalg.norm(gtsam.Rot3.Logmap(delta_r)))
  drift_deg = math.degrees(drift)
  assert drift_deg < 0.3, f"stationary orientation drift {drift_deg:.3f}° > 0.3°"
