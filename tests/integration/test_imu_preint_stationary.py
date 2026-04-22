"""End-to-end IMU preintegration on a real stationary capture.

Loads a 30-second IMU fixture clipped from an Insta360 X3 stationary
``.insv`` (committed under ``tests/fixtures/vi_ba/``), takes the mean
gyro/accel of the first 3 seconds as the bias estimate, preintegrates
the remaining 27 seconds, and asserts the integrated rotation stays
within an orientation-drift budget proportional to the 1° / 5 min
stationary gate.

Budget arithmetic: 1° per 300 s → ~0.003°/s. For 27 s that's ≤ 0.09°;
we test with a generous 0.3° ceiling to stay robust across minor
firmware / calibration drift between X3 units.
"""

from __future__ import annotations

import math
from pathlib import Path

import gtsam
import numpy as np

from specialist.vi_ba.imu import (
  from_deg_per_s,
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
  samples = from_deg_per_s(t, gyro_deg, accel)
  assert len(samples) > 20_000  # 1 kHz × 30 s minus small head/tail

  # The X3 sat on a table with body +y pointing up (gravity in body +y);
  # world frame is chosen so gravity points along world −y, matching the
  # measured accelerometer direction over the stationary window.
  g_magnitude = float(np.linalg.norm(accel.mean(axis=0)))
  assert 9.6 < g_magnitude < 10.0  # sanity: gravity bucket
  gravity_world = np.array([0.0, -g_magnitude, 0.0], dtype=np.float64)
  initial_rotation = gtsam.Rot3()  # body == world at t = 0

  # On a known-stationary window, the mean gyro reading IS the gyro bias
  # and the mean accel reading is gravity in the body frame, so accel bias
  # = mean(accel) − R^T · g_world. We only do this here because the
  # capture is verified stationary; the runtime pipeline uses the X3's
  # factory `gyro_calib.numbers` instead, which doesn't assume rest.
  seed_mask = samples.t < (samples.t[0] + 3.0)
  gyro_bias = samples.gyro[seed_mask].mean(axis=0)
  accel_mean = samples.accel[seed_mask].mean(axis=0)
  g_body = initial_rotation.matrix().T @ gravity_world
  accel_bias = accel_mean + g_body
  bias = gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)

  # Preintegrate the remaining ~27 s.
  from specialist.vi_ba.imu import ImuSamples

  tail_mask = samples.t >= (samples.t[0] + 3.0)
  tail = ImuSamples(
    t=samples.t[tail_mask],
    gyro=samples.gyro[tail_mask],
    accel=samples.accel[tail_mask],
  )

  # Noise sigmas taken from the imu.py probe's Allan numbers for this unit.
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
  # 0.3° budget across ~27 s is well inside the 1° / 5 min gate.
  assert drift_deg < 0.3, f"stationary orientation drift {drift_deg:.3f}° > 0.3°"
