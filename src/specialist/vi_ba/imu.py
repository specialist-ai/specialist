"""IMU preintegration helpers wrapping GTSAM's
``PreintegratedCombinedMeasurements``: typed ``ImuSamples`` container,
``PreintegrationCombinedParams`` builder, and a decoder for the factory
bias from ``gyro_calib.numbers``. ``gravity_world`` follows the GTSAM
convention (acceleration of gravity, e.g. ``(0, 0, -9.81)`` for z-up)."""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]
Float64Array = npt.NDArray[np.float64]

GRAVITY = 9.81
GRAVITY_WORLD = (0.0, 0.0, -GRAVITY)

_INITIAL_BIAS_SIGMA = 1e-3
# Samples per stationarity window: a quarter second of the capture's IMU.
_STATIONARY_WINDOW = 50

__all__ = [
  "GRAVITY",
  "GRAVITY_WORLD",
  "ImuSamples",
  "find_stationary_windows",
  "integrate_samples",
  "make_preintegration_params",
  "preintegrate_between_keyframes",
  "slice_samples_at_keyframes",
  "stationary_mask",
]


@dataclass(frozen=True)
class ImuSamples:
  """Aligned IMU triple in SI units: timestamps (s), gyro (rad/s), accel (m/s²).

  Row i of ``gyro`` and ``accel`` shares timestamp ``t[i]``.
  """

  t: Float64Array
  gyro: Float64Array
  accel: Float64Array

  def __len__(self) -> int:
    return int(self.t.shape[0])


def stationary_mask(
  samples: ImuSamples,
  bias: gtsam.imuBias.ConstantBias | None = None,
  *,
  window: int = _STATIONARY_WINDOW,
  gyro_threshold: float = 0.1,
  accel_var_threshold: float = 0.05,
) -> BoolArray:
  """Stationarity of each non-overlapping ``window``-sample block: bias-corrected
  gyro magnitude (rad/s) under ``gyro_threshold`` throughout, and accel sample
  variance under ``accel_var_threshold``. Accel magnitude is not checked."""
  blocks = len(samples) // window
  gyro = samples.gyro[: blocks * window]
  accel = samples.accel[: blocks * window]
  if bias is not None:
    gyro, accel = gyro - bias.gyroscope(), accel - bias.accelerometer()
  peak = np.linalg.norm(gyro, axis=1).reshape(blocks, window).max(axis=1)
  variance = accel.reshape(blocks, window, 3).var(axis=1).max(axis=1)
  quiet: BoolArray = (peak <= gyro_threshold) & (variance <= accel_var_threshold)
  return quiet


def find_stationary_windows(
  samples: ImuSamples,
  bias: gtsam.imuBias.ConstantBias,
) -> list[tuple[int, int]]:
  """Non-overlapping ``(start, end)`` index ranges the IMU spends stationary, in
  ascending order. Same block predicate as ``stationary_mask``."""
  quiet = stationary_mask(samples, bias)
  return [
    (int(i) * _STATIONARY_WINDOW, (int(i) + 1) * _STATIONARY_WINDOW)
    for i in np.flatnonzero(quiet)
  ]


# White-noise densities from the measured Allan floor, and the bias random
# walks.
_GYRO_NOISE_SIGMA = 0.0125
_ACCEL_NOISE_SIGMA = 0.01
_GYRO_RANDOM_WALK = 1.0e-7
_ACCEL_RANDOM_WALK = 1.0e-7


def make_preintegration_params(
  gravity_world: tuple[float, float, float],
  *,
  gyro_noise_sigma: float = _GYRO_NOISE_SIGMA,
  accel_noise_sigma: float = _ACCEL_NOISE_SIGMA,
  gyro_walk: float = _GYRO_RANDOM_WALK,
  accel_walk: float = _ACCEL_RANDOM_WALK,
) -> gtsam.PreintegrationCombinedParams:
  """Populate a ``PreintegrationCombinedParams`` with per-axis isotropic
  noise + bias-walk covariances."""
  params = gtsam.PreintegrationCombinedParams(
    np.asarray(gravity_world, dtype=np.float64)
  )
  params.setAccelerometerCovariance(np.eye(3) * accel_noise_sigma**2)
  params.setGyroscopeCovariance(np.eye(3) * gyro_noise_sigma**2)
  params.setIntegrationCovariance(np.eye(3) * 1e-10)
  params.setBiasAccCovariance(np.eye(3) * accel_walk**2)
  params.setBiasOmegaCovariance(np.eye(3) * gyro_walk**2)
  params.setBiasAccOmegaInit(np.eye(6) * _INITIAL_BIAS_SIGMA**2)
  return params


def slice_samples_at_keyframes(
  samples: ImuSamples,
  keyframe_times: Float64Array,
) -> list[ImuSamples]:
  """Partition ``samples`` into ``len(keyframe_times) - 1`` windows.

  Window ``i`` holds samples whose timestamp lies in ``(t_i, t_{i+1}]``. Empty
  windows come back as zero-length ``ImuSamples``.
  """
  t_kf = np.asarray(keyframe_times, dtype=np.float64)
  bounds = np.searchsorted(samples.t, t_kf, side="right")
  out: list[ImuSamples] = []
  for i in range(len(t_kf) - 1):
    window = slice(bounds[i], bounds[i + 1])
    out.append(
      ImuSamples(
        t=samples.t[window],
        gyro=samples.gyro[window],
        accel=samples.accel[window],
      )
    )
  return out


def integrate_samples(
  params: gtsam.PreintegrationCombinedParams,
  samples: ImuSamples,
  bias: gtsam.imuBias.ConstantBias,
) -> gtsam.PreintegratedCombinedMeasurements:
  """Integrate every consecutive-pair Δt from ``samples`` into one PIM.

  The GTSAM convention uses the earlier sample's measurement over the
  ``t[i+1] - t[i]`` interval (zero-order hold). Non-monotone timestamps
  contribute zero.
  """
  pim = gtsam.PreintegratedCombinedMeasurements(params, bias)
  n = len(samples)
  if n < 2:
    return pim
  dts = np.diff(samples.t)
  for i in range(n - 1):
    dt = float(dts[i])
    if dt <= 0.0:
      continue
    pim.integrateMeasurement(
      samples.accel[i].astype(np.float64),
      samples.gyro[i].astype(np.float64),
      dt,
    )
  return pim


def preintegrate_between_keyframes(
  params: gtsam.PreintegrationCombinedParams,
  samples: ImuSamples,
  keyframe_times: Float64Array,
  bias: gtsam.imuBias.ConstantBias,
) -> list[gtsam.PreintegratedCombinedMeasurements]:
  """One ``PreintegratedCombinedMeasurements`` per consecutive keyframe pair."""
  per_window = slice_samples_at_keyframes(samples, keyframe_times)
  return [integrate_samples(params, w, bias) for w in per_window]
