"""Video <-> IMU timing probe.

Two checks against the metadata:

  - Video / IMU clock offset, from cross-correlating LK optical-flow magnitude
    against gyro magnitude. Needs yaw motion; a sharp yaw flick is best.
  - Rolling-shutter readout, from the tilt of near-vertical Hough line segments
    divided by the simultaneous gyro yaw rate. Needs visible vertical edges
    under a smooth yaw.

Outputs ``logs/video_imu_timing_<ts>.rrd``, a JSON summary, and a console
verdict on both checks.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr
import tyro

if __package__ in (None, ""):  # `python scripts/x.py` puts scripts/ itself first
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._probe import UInt8Array, finalize_probe, read_lens_grays, verdict
from specialist.loader.audio_sync import correlation_offset
from specialist.loader.parse_insv import load_insv
from specialist.vi_ba.features import detect_corners

Float64Array = npt.NDArray[np.float64]

# Height the correlation peak must reach over its tallest rival before the
# offset means anything.
_MIN_PEAK_RATIO = 1.3


@dataclass
class TimingArgs:
  """CLI arguments for the timing probe."""

  input_insv: Path
  output_dir: Path = Path("logs")
  lens: Literal["front", "rear"] = "front"  # _00_ file / _10_ sibling
  max_frames: int = 600
  flow_max_corners: int = 200
  flow_quality_level: float = 0.01
  flow_min_distance_px: float = 8.0
  shear_min_yaw_rate_dps: float = 30.0
  hough_threshold: int = 80


@dataclass
class OffsetResult:
  offset: float
  peak_ratio: float


@dataclass
class ShearResult:
  measured_readout: float
  factory_readout: float
  n_frames_used: int
  median_yaw_rate_dps: float


@dataclass
class TimingSummary:
  source: str
  n_video_frames_used: int
  fps: float
  imu_rate: float
  offset: OffsetResult | None
  shear: ShearResult | None
  verdict: str = "unknown"


def _flow_magnitudes(grays: list[UInt8Array], args: TimingArgs) -> Float64Array:
  """Median Lucas-Kanade flow magnitude per frame transition."""

  def corners(img: UInt8Array) -> npt.NDArray[np.float32]:
    points = detect_corners(
      img,
      None,
      max_corners=args.flow_max_corners,
      quality_level=args.flow_quality_level,
      min_distance=args.flow_min_distance_px,
    )
    return points.reshape(-1, 1, 2).astype(np.float32)

  mags: list[float] = []
  prev, prev_points = grays[0], corners(grays[0])
  for curr in grays[1:]:
    # A frame down to a handful of points measures noise, not motion.
    if len(prev_points) < 10:
      mags.append(0.0)
    else:
      next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev, curr, prev_points, np.empty_like(prev_points)
      )
      good = status.flatten() == 1
      delta = (next_points[good] - prev_points[good]).reshape(-1, 2)
      norms = np.linalg.norm(delta, axis=1)
      mags.append(float(np.median(norms)) if norms.size else 0.0)
    prev, prev_points = curr, corners(curr)
  return np.asarray([0.0, *mags], dtype=np.float64)


def _estimate_offset(
  video_t: Float64Array,
  flow_mag: Float64Array,
  imu_t: Float64Array,
  gyro_mag: Float64Array,
) -> OffsetResult:
  """Clock offset to add to the IMU timestamps to land them on the video
  clock."""
  rate = 1.0 / float(np.median(np.diff(imu_t)))
  grid = np.arange(max(video_t[0], imu_t[0]), min(video_t[-1], imu_t[-1]), 1.0 / rate)
  offset, peak_ratio = correlation_offset(
    np.interp(grid, imu_t, gyro_mag),
    np.interp(grid, video_t, flow_mag),
    int(round(rate)),
  )
  return OffsetResult(offset=offset, peak_ratio=peak_ratio)


def _vertical_line_shear(gray: UInt8Array, threshold: int) -> float | None:
  """Median per-pixel horizontal shift along near-vertical Hough line segments."""
  edges = cv2.Canny(gray, 60, 180)
  segments = cv2.HoughLinesP(
    edges, 1, np.pi / 360.0, threshold, minLineLength=80, maxLineGap=10
  )
  if segments is None:
    return None
  shifts: list[float] = []
  for seg in segments.reshape(-1, 4):
    x1, y1, x2, y2 = (float(v) for v in seg)
    dy = y2 - y1
    if abs(dy) < 60:
      continue
    angle = np.degrees(np.arctan2(x2 - x1, dy))
    if abs(angle) > 25.0:
      continue
    shifts.append((x2 - x1) / dy)
  if not shifts:
    return None
  return float(np.median(shifts))


def _estimate_rolling_shutter(
  grays: list[UInt8Array],
  video_t: Float64Array,
  imu_t: Float64Array,
  gyro_dps: Float64Array,
  *,
  args: TimingArgs,
  applied_offset: float,
  factory_readout: float,
) -> ShearResult | None:
  """Per-frame: shear (px/px) / gyro_yaw_rate_pix -> readout time (s)."""
  height, width = grays[0].shape[:2]
  # Degrees per pixel for a ~190° fisheye spanning the half-frame width.
  deg_per_pixel = 190.0 / width

  yaw_imu = gyro_dps[:, 1]  # yaw axis after telemetry-parser normalization
  imu_t_shifted = imu_t + applied_offset

  readouts: list[float] = []
  used_yaw_rates: list[float] = []
  for k, gray in enumerate(grays):
    yaw_rate = float(np.interp(video_t[k], imu_t_shifted, yaw_imu))
    if abs(yaw_rate) < args.shear_min_yaw_rate_dps:
      continue
    shear_px_per_px = _vertical_line_shear(gray, args.hough_threshold)
    if shear_px_per_px is None:
      continue
    yaw_rate_px = yaw_rate / deg_per_pixel
    if abs(yaw_rate_px) < 1.0:
      continue
    # shear (dx/dy) = yaw_rate_px * t_readout / height.
    readout = shear_px_per_px * height / yaw_rate_px
    if 0.0 < readout < 0.05:
      readouts.append(readout)
      used_yaw_rates.append(yaw_rate)

  if not readouts:
    return None
  return ShearResult(
    measured_readout=float(np.median(readouts)),
    factory_readout=factory_readout,
    n_frames_used=len(readouts),
    median_yaw_rate_dps=float(np.median(np.abs(used_yaw_rates))),
  )


def _verdict(summary: TimingSummary) -> str:
  parts: list[str] = []
  if summary.offset is None:
    parts.append("offset N/A")
  elif summary.offset.peak_ratio < _MIN_PEAK_RATIO:
    parts.append(
      f"offset unreliable: peak {summary.offset.peak_ratio:.2f}x its tallest rival"
    )
  elif abs(summary.offset.offset) > 0.05:
    parts.append(f"large video<->IMU offset {summary.offset.offset * 1e3:.1f} ms")
  if summary.shear is None:
    parts.append("RS shear unmeasured (need yaw motion + vertical edges)")
  elif abs(summary.shear.measured_readout - summary.shear.factory_readout) > 5e-3:
    parts.append(
      f"RS readout mismatch: {summary.shear.measured_readout * 1e3:.2f} ms vs "
      f"{summary.shear.factory_readout * 1e3:.2f} ms factory"
    )
  return verdict(parts)


def _log_to_rerun(
  video_t: Float64Array,
  flow_mag: Float64Array,
  imu_t: Float64Array,
  gyro_mag: Float64Array,
  applied_offset: float,
) -> None:
  rr.send_columns(
    "flow/median_magnitude",
    indexes=[rr.TimeColumn("t", duration=video_t)],
    columns=rr.Scalars.columns(scalars=flow_mag),
  )
  rr.send_columns(
    "gyro/magnitude_dps",
    indexes=[rr.TimeColumn("t", duration=imu_t + applied_offset)],
    columns=rr.Scalars.columns(scalars=gyro_mag),
  )


def main(args: TimingArgs) -> None:
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  data = load_insv(args.input_insv)

  grays, fps = read_lens_grays(args.input_insv, args.lens, max_frames=args.max_frames)
  if len(grays) < 5:
    raise RuntimeError(f"only {len(grays)} frames decoded — capture too short?")
  video_t = np.arange(len(grays), dtype=np.float64) / fps

  flow_mag = _flow_magnitudes(grays, args)
  gyro_mag = np.linalg.norm(data.gyro_deg, axis=1)
  imu_rate = 1.0 / float(np.median(np.diff(data.imu_timestamps)))

  offset = _estimate_offset(video_t, flow_mag, data.imu_timestamps, gyro_mag)
  shear = _estimate_rolling_shutter(
    grays,
    video_t,
    data.imu_timestamps,
    data.gyro_deg,
    args=args,
    applied_offset=offset.offset,
    factory_readout=data.rolling_shutter_readout,
  )

  summary = TimingSummary(
    source=str(args.input_insv),
    n_video_frames_used=len(grays),
    fps=fps,
    imu_rate=imu_rate,
    offset=offset,
    shear=shear,
  )
  summary.verdict = _verdict(summary)

  finalize_probe(
    "video_imu_timing",
    args.output_dir,
    summary,
    lambda: _log_to_rerun(
      video_t, flow_mag, data.imu_timestamps, gyro_mag, offset.offset
    ),
  )


if __name__ == "__main__":
  main(tyro.cli(TimingArgs))
