"""IMU characterization probe; bias instability needs a still capture of 30 minutes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import allantools
import numpy as np
import numpy.typing as npt
import rerun as rr
import tyro
from _probe import finalize_probe

from specialist.loader.parse_insv import InsvData, load_insv

Float64Array = npt.NDArray[np.float64]


@dataclass
class ImuArgs:
  """IMU characterization of one capture."""

  input_insv: Path
  output_dir: Path = Path("logs")
  stationary_window: float = 1.0
  stationary_gyro_threshold_dps: float = 0.5
  allan_max_tau: float = 100.0


@dataclass
class StationarySegment:
  start: float
  end: float
  gyro_sigma_dps: tuple[float, float, float]
  accel_sigma: tuple[float, float, float]
  n_samples: int


@dataclass
class AllanResult:
  axis: str
  taus: list[float]
  adev: list[float]
  white_noise_density: float
  bias_instability: float


@dataclass
class ImuSummary:
  source: str
  n_samples: int
  duration: float
  median_dt: float
  rate: float
  rate_jitter_pct: float
  gap_count: int
  gap_max: float
  gyro_accel_aligned: bool
  stationary: StationarySegment | None
  allan: list[AllanResult] = field(default_factory=list)
  verdict: str = "unknown"


def _build_stationary(
  ts: Float64Array,
  gyro: Float64Array,
  accel: Float64Array,
  *,
  window: float,
  threshold_dps: float,
) -> StationarySegment | None:
  median_dt = float(np.median(np.diff(ts)))
  win = max(8, int(round(window / max(median_dt, 1e-6))))
  mag = np.linalg.norm(gyro, axis=1)
  if mag.size < win:
    return None
  kernel = np.ones(win) / win
  mag_smooth = np.convolve(mag, kernel, mode="valid")
  quiet = mag_smooth < threshold_dps
  if not quiet.any():
    return None
  best_start, best_len, run_start = -1, 0, -1
  for i, q in enumerate(quiet):
    if q and run_start == -1:
      run_start = i
    elif not q and run_start != -1:
      length = i - run_start
      if length > best_len:
        best_start, best_len = run_start, length
      run_start = -1
  if run_start != -1:
    length = quiet.size - run_start
    if length > best_len:
      best_start, best_len = run_start, length
  if best_start == -1 or best_len < win * 4:
    return None
  i0 = best_start + win // 2
  i1 = min(best_start + best_len + win // 2, ts.size)
  g_seg = gyro[i0:i1]
  a_seg = accel[i0:i1]
  return StationarySegment(
    start=float(ts[i0]),
    end=float(ts[i1 - 1]),
    gyro_sigma_dps=tuple(float(s) for s in g_seg.std(axis=0)),  # type: ignore[arg-type]
    accel_sigma=tuple(float(s) for s in a_seg.std(axis=0)),  # type: ignore[arg-type]
    n_samples=int(i1 - i0),
  )


def _allan_for_axis(
  signal: Float64Array, rate: float, axis: str, max_tau: float
) -> AllanResult:
  taus_target = np.logspace(np.log10(2.0 / rate), np.log10(max_tau), num=50, base=10.0)
  taus, adev, _, _ = allantools.oadev(
    signal, rate=rate, data_type="freq", taus=taus_target
  )
  taus = np.asarray(taus, dtype=np.float64)
  adev = np.asarray(adev, dtype=np.float64)

  # White-noise density N = ADEV * sqrt(tau) on the -1/2 slope, in signal/sqrt(Hz).
  short_mask = (taus > 0) & (taus < 1.0)
  white_noise_density = (
    float(np.median(adev[short_mask] * np.sqrt(taus[short_mask])))
    if short_mask.any()
    else float("nan")
  )

  long_mask = taus > 1.0
  bias_instability = float(np.min(adev[long_mask])) if long_mask.any() else float("nan")

  return AllanResult(
    axis=axis,
    taus=taus.tolist(),
    adev=adev.tolist(),
    white_noise_density=white_noise_density,
    bias_instability=bias_instability,
  )


def _log_to_rerun(data: InsvData, allans: list[AllanResult]) -> None:
  rr.log("imu", rr.SeriesLines(names=["x", "y", "z"]), static=True)
  for i, ts in enumerate(data.imu_timestamps):
    rr.set_time("imu_time", duration=float(ts))
    rr.log("imu/gyro", rr.Scalars(data.gyro_deg[i].tolist()))
    rr.log("imu/accel", rr.Scalars(data.accel[i].tolist()))
  for allan in allans:
    for tau, val in zip(allan.taus, allan.adev, strict=True):
      rr.set_time("allan_tau", duration=float(tau))
      rr.log(f"imu/allan/{allan.axis}", rr.Scalars([float(val)]))


def _verdict(summary: ImuSummary) -> str:
  problems: list[str] = []
  if summary.gap_max > 0.01:
    problems.append(f"gap up to {summary.gap_max * 1e3:.1f} ms")
  if summary.rate_jitter_pct > 5.0:
    problems.append(f"rate jitter {summary.rate_jitter_pct:.1f}%")
  if not summary.gyro_accel_aligned:
    problems.append("gyro/accel timestamps not aligned")
  return "green" if not problems else "yellow: " + ", ".join(problems)


def main(args: ImuArgs) -> None:
  args.output_dir.mkdir(parents=True, exist_ok=True)
  data = load_insv(args.input_insv)

  ts = data.imu_timestamps
  dts = np.diff(ts)
  median_dt = float(np.median(dts))
  rate = 1.0 / median_dt if median_dt > 0 else float("nan")
  rate_jitter_pct = (
    float(np.std(dts) / median_dt) * 100.0 if median_dt > 0 else float("nan")
  )
  gap_threshold = 5.0 * median_dt
  gap_count = int((dts > gap_threshold).sum())
  gap_max = float(dts.max()) if dts.size else 0.0

  # telemetry-parser bundles gyro and accel per sample, so equal counts mean one clock.
  gyro_accel_aligned = (
    data.gyro_deg.shape[0] == ts.shape[0] and data.accel.shape[0] == ts.shape[0]
  )

  stationary = _build_stationary(
    ts,
    data.gyro_deg,
    data.accel,
    window=args.stationary_window,
    threshold_dps=args.stationary_gyro_threshold_dps,
  )

  allans: list[AllanResult] = []
  if stationary is not None:
    i0 = int(np.searchsorted(ts, stationary.start))
    i1 = int(np.searchsorted(ts, stationary.end))
    if (i1 - i0) > int(rate * 4):
      gyro_seg = data.gyro_deg[i0:i1]
      max_tau = min(args.allan_max_tau, (i1 - i0) / rate / 4.0)
      for ax_idx, axis_name in enumerate(("gyro_x", "gyro_y", "gyro_z")):
        allans.append(_allan_for_axis(gyro_seg[:, ax_idx], rate, axis_name, max_tau))

  summary = ImuSummary(
    source=str(args.input_insv),
    n_samples=int(ts.size),
    duration=float(ts[-1] - ts[0]),
    median_dt=median_dt,
    rate=rate,
    rate_jitter_pct=rate_jitter_pct,
    gap_count=gap_count,
    gap_max=gap_max,
    gyro_accel_aligned=gyro_accel_aligned,
    stationary=stationary,
    allan=allans,
  )
  summary.verdict = _verdict(summary)

  finalize_probe(
    "imu",
    "specialist.imu_probe",
    args.output_dir,
    summary,
    lambda: _log_to_rerun(data, allans),
  )


if __name__ == "__main__":
  main(tyro.cli(ImuArgs))
