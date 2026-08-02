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

Float64Array = npt.NDArray[np.float64]

GRAVITY = 9.81
GRAVITY_WORLD = (0.0, 0.0, -GRAVITY)

_INITIAL_BIAS_SIGMA = 1e-3

__all__ = [
  "GRAVITY",
  "GRAVITY_WORLD",
  "ImuSamples",
  "factory_imu_bias",
  "find_stationary_windows",
  "from_deg_per_s",
  "integrate_samples",
  "keyframe_delta_rotations",
  "make_preintegration_params",
  "preintegrate_between_keyframes",
  "slice_samples_at_keyframes",
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


def from_deg_per_s(
  t: Float64Array,
  gyro_deg: Float64Array,
  accel: Float64Array,
) -> ImuSamples:
  """Build an ``ImuSamples`` from raw parse_insv output (gyro is in deg/s)."""
  return ImuSamples(
    t=np.asarray(t, dtype=np.float64),
    gyro=np.deg2rad(np.asarray(gyro_deg, dtype=np.float64)),
    accel=np.asarray(accel, dtype=np.float64),
  )


def find_stationary_windows(
  samples: ImuSamples,
  bias: gtsam.imuBias.ConstantBias,
) -> list[tuple[int, int]]:
  """Find non-overlapping ``(start, end)`` index ranges where the IMU
  appears stationary: bias-corrected gyro magnitude below
  ``gyro_threshold`` and accel sample variance below
  ``accel_var_threshold``. The accel-magnitude check is intentionally
  skipped — an IMU accel-scale error would otherwise reject every real
  stationary window; the caller rescales accel using the detected
  windows."""
  window_size = 50
  gyro_threshold = 0.1
  accel_var_threshold = 0.05
  n = len(samples)
  if n < window_size:
    return []
  accel_corr = samples.accel - bias.accelerometer()
  gyro_corr = samples.gyro - bias.gyroscope()
  windows: list[tuple[int, int]] = []
  for start in range(0, n - window_size + 1, window_size):
    end = start + window_size
    if np.max(np.linalg.norm(gyro_corr[start:end], axis=1)) > gyro_threshold:
      continue
    if float(accel_corr[start:end].var(axis=0).max()) > accel_var_threshold:
      continue
    windows.append((start, end))
  return windows


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


def factory_imu_bias(
  gyro_calib_numbers: Float64Array,
) -> gtsam.imuBias.ConstantBias:
  """Factory IMU bias from ``gyro_calib.numbers``: ``[gx, gy, gz, ax, ay,
  az]`` (3 gyro rad/s + 3 accel m/s²). Use this when the capture doesn't
  start stationary and a window-mean seed would absorb real motion."""
  arr = np.asarray(gyro_calib_numbers, dtype=np.float64).reshape(6)
  return gtsam.imuBias.ConstantBias(arr[3:6], arr[:3])


def keyframe_delta_rotations(
  samples: ImuSamples,
  keyframe_times: Float64Array,
) -> list[Float64Array]:
  """Per-keyframe body attitude increments integrated from the gyro.

  Entry ``k`` composes the gyro rotations over ``(t_{k-1}, t_k]`` so that
  ``R_world_body(t_k) ≈ R_world_body(t_{k-1}) @ delta[k]``; entry 0 is the
  identity. Zero-order hold over consecutive-sample intervals, matching
  ``integrate_samples``. Windows without at least two samples come back as
  the identity.
  """
  deltas: list[Float64Array] = [np.eye(3)]
  for window in slice_samples_at_keyframes(samples, keyframe_times):
    delta = np.eye(3)
    for j in range(len(window) - 1):
      dt = float(window.t[j + 1] - window.t[j])
      if dt > 0.0:
        delta = delta @ gtsam.Rot3.Expmap(window.gyro[j] * dt).matrix()
    deltas.append(delta)
  return deltas


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
