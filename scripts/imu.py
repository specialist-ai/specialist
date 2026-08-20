"""IMU characterization probe; bias instability needs a still capture of 30 minutes."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import allantools
import numpy as np
import numpy.typing as npt
import rerun as rr
import tyro

if __package__ in (None, ""):  # `python scripts/x.py` puts scripts/ itself first
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._probe import finalize_probe, verdict
from specialist.loader.parse_insv import InsvData, load_insv
from specialist.vi_ba.imu import ImuSamples, stationary_mask

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
  white_noise_density_dps: float
  bias_instability_dps: float


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
  """Per-axis noise over the longest run of stationary windows."""
  median_dt = float(np.median(np.diff(ts)))
  win = max(8, int(round(window / max(median_dt, 1e-6))))
  # Gate on the gyro alone; the accel sigma is reported, never thresholded.
  quiet = stationary_mask(
    ImuSamples(t=ts, gyro=np.radians(gyro), accel=accel),
    window=win,
    gyro_threshold=float(np.deg2rad(threshold_dps)),
    accel_var_threshold=float("inf"),
  )
  if not quiet.any():
    return None
  # The False padding turns a run touching either end into a normal edge pair.
  padded = np.concatenate(([False], quiet, [False])).astype(np.int8)
  edges = np.flatnonzero(np.diff(padded))
  starts, ends = edges[::2], edges[1::2]
  longest = int(np.argmax(ends - starts))
  if int(ends[longest] - starts[longest]) < 4:
    return None
  i0, i1 = int(starts[longest]) * win, int(ends[longest]) * win
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

  # At slope -1/2, ADEV(tau) = N / sqrt(tau), so N is in deg/s/sqrt(Hz).
  short_mask = (taus > 0) & (taus < 1.0)
  white_noise_density_dps = (
    float(np.median(adev[short_mask] * np.sqrt(taus[short_mask])))
    if short_mask.any()
    else float("nan")
  )

  long_mask = taus > 1.0
  bias_instability_dps = (
    float(np.min(adev[long_mask])) if long_mask.any() else float("nan")
  )

  return AllanResult(
    axis=axis,
    taus=taus.tolist(),
    adev=adev.tolist(),
    white_noise_density_dps=white_noise_density_dps,
    bias_instability_dps=bias_instability_dps,
  )


def _log_to_rerun(data: InsvData, allans: list[AllanResult]) -> None:
  rr.log("imu", rr.SeriesLines(names=["x", "y", "z"]), static=True)
  times = rr.TimeColumn("imu_time", duration=data.imu_timestamps)
  triples = np.full(data.imu_timestamps.size, 3)
  for entity, samples in (("gyro", data.gyro_deg), ("accel", data.accel)):
    rr.send_columns(
      f"imu/{entity}",
      indexes=[times],
      columns=rr.Scalars.columns(scalars=samples.reshape(-1)).partition(triples),
    )
  for allan in allans:
    rr.send_columns(
      f"imu/allan/{allan.axis}",
      indexes=[rr.TimeColumn("allan_tau", duration=allan.taus)],
      columns=rr.Scalars.columns(scalars=allan.adev),
    )


def _verdict(summary: ImuSummary) -> str:
  problems: list[str] = []
  if summary.gap_max > 0.01:
    problems.append(f"gap up to {summary.gap_max * 1e3:.1f} ms")
  if summary.rate_jitter_pct > 5.0:
    problems.append(f"rate jitter {summary.rate_jitter_pct:.1f}%")
  if not summary.gyro_accel_aligned:
    problems.append("gyro/accel timestamps not aligned")
  return verdict(problems)


def main(args: ImuArgs) -> None:
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

  finalize_probe("imu", args.output_dir, summary, lambda: _log_to_rerun(data, allans))


if __name__ == "__main__":
  main(tyro.cli(ImuArgs))
