"""Video <-> IMU timing probe.

Two checks against the metadata:

  - Video / IMU clock offset, estimated by cross-correlating LK optical-flow
    magnitude against gyro magnitude across a fine offset grid. Best done on
    a sharp yaw-flick capture where the signal is strong.
  - Rolling-shutter readout, estimated from the tilt of near-vertical Hough
    line segments divided by the simultaneous gyro yaw rate. Best done on a
    smooth-yaw capture with visible vertical edges.

Capture: a capture with both yaw motion and vertical structure works for
both checks; dedicated yaw-flick + smooth-yaw captures maximise SNR.

Outputs: ``logs/video_imu_timing_<ts>.rrd`` (gyro mag, flow mag), JSON
summary, console verdict on both checks.
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
from specialist.loader.parse_insv import load_insv

Float64Array = npt.NDArray[np.float64]

# Offset search grid: the video and IMU clocks come off the same capture, so
# half a second at 5 ms brackets any plausible shift.
_MAX_OFFSET = 0.5
_OFFSET_STEP = 0.005


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
  best_offset: float
  peak_correlation: float
  search_range: float


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

  def corners(img: np.ndarray) -> np.ndarray | None:
    return cv2.goodFeaturesToTrack(
      img,
      maxCorners=args.flow_max_corners,
      qualityLevel=args.flow_quality_level,
      minDistance=args.flow_min_distance_px,
    )

  mags: list[float] = []
  prev = grays[0]
  prev_pts = corners(prev)
  for k in range(1, len(grays)):
    curr = grays[k]
    if prev_pts is None or len(prev_pts) < 10:
      mags.append(0.0)
      prev_pts = corners(curr)
      prev = curr
      continue
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
      prev, curr, prev_pts, np.empty_like(prev_pts)
    )
    if next_pts is None or status is None:
      mags.append(0.0)
    else:
      good = status.flatten() == 1
      if good.any():
        delta = (next_pts[good] - prev_pts[good]).reshape(-1, 2)
        mags.append(float(np.median(np.linalg.norm(delta, axis=1))))
      else:
        mags.append(0.0)
    prev_pts = corners(curr)
    prev = curr
  return np.asarray([0.0, *mags], dtype=np.float64)


def _estimate_offset(
  video_t: Float64Array,
  flow_mag: Float64Array,
  imu_t: Float64Array,
  gyro_mag: Float64Array,
) -> OffsetResult:
  flow = (flow_mag - flow_mag.mean()) / (flow_mag.std() + 1e-9)

  best_off, best_corr = 0.0, -np.inf
  for off in np.arange(-_MAX_OFFSET, _MAX_OFFSET + _OFFSET_STEP, _OFFSET_STEP):
    shifted = np.interp(video_t, imu_t + off, gyro_mag)
    z = (shifted - shifted.mean()) / (shifted.std() + 1e-9)
    corr = float(np.dot(flow, z) / flow.size)
    if corr > best_corr:
      best_corr, best_off = corr, float(off)
  return OffsetResult(
    best_offset=best_off,
    peak_correlation=best_corr,
    search_range=_MAX_OFFSET,
  )


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
  elif abs(summary.offset.best_offset) > 0.05:
    parts.append(f"large video<->IMU offset {summary.offset.best_offset * 1e3:.1f} ms")
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
    applied_offset=offset.best_offset,
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
      video_t, flow_mag, data.imu_timestamps, gyro_mag, offset.best_offset
    ),
  )


if __name__ == "__main__":
  main(tyro.cli(TimingArgs))
