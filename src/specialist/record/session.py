"""The solve's view of a base recording: slots, video, IMU and calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gtsam
import numpy as np
import numpy.typing as npt
from rerun.experimental import RrdReader
from specialist_camera import Side
from specialist_dataset import frames, schema
from specialist_dataset.poses import quats_to_matrices
from specialist_dataset.read import Recording, lens_from_row

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.imu import ImuSamples

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

__all__ = [
  "SessionRecording",
  "covered_rows",
  "fps_from_times",
  "nearest_rows",
]


def fps_from_times(times: Float64Array) -> float:
  """Frame rate from per-frame times, since the telemetry rounds NTSC rates."""
  if len(times) < 2:
    raise ValueError(f"need at least two frame times, got {len(times)}")
  return (len(times) - 1) / float(times[-1] - times[0])


def nearest_rows(times: Float64Array, grid: Float64Array) -> Int64Array:
  """The row of ``times`` nearest each ``grid`` time; raises beyond half a frame."""
  times = np.asarray(times, dtype=np.float64)
  grid = np.asarray(grid, dtype=np.float64)
  if len(times) < 2:
    raise ValueError(f"need at least two frame times, got {len(times)}")
  pos = np.searchsorted(times, grid)
  lo = np.clip(pos - 1, 0, len(times) - 1)
  hi = np.clip(pos, 0, len(times) - 1)
  rows = np.where(np.abs(times[hi] - grid) < np.abs(times[lo] - grid), hi, lo)
  deltas = np.abs(times[rows] - grid)
  half_frame = 0.5 * float(np.median(np.diff(times)))
  if float(deltas.max()) > half_frame:
    raise ValueError(
      f"nearest-PTS match misses the grid by {float(deltas.max()):.4f} s, "
      f"more than half a frame ({half_frame:.4f} s)"
    )
  out: Int64Array = rows.astype(np.int64)
  return out


def covered_rows(
  front: Float64Array, rear: Float64Array, imu: Float64Array, fps: float
) -> Int64Array:
  """The front-lens rows inside the span front, rear and IMU all cover."""
  half_frame = 0.5 / fps
  lo = max(front[0], rear[0], imu[0]) - half_frame
  hi = min(front[-1], rear[-1], imu[-1]) + half_frame
  rows: Int64Array = np.flatnonzero((front >= lo) & (front <= hi)).astype(np.int64)
  return rows


class SessionRecording:
  """One session's base recording: the solve's only input."""

  def __init__(self, base_rrd: Path) -> None:
    self.path = base_rrd
    recordings = RrdReader(str(base_rrd)).recordings()
    if not recordings:
      raise ValueError(f"{base_rrd} holds no recording")
    self.session_id: str = recordings[0].recording_id
    self._rows: dict[str, dict[str, Any]] = {}
    self._recording = Recording(base_rrd)

  @property
  def slots(self) -> list[str]:
    """The camera slots this recording holds video for."""
    paths = self._recording.entity_paths()
    slots = [
      slot
      for slot in schema.slots_in(paths)
      if any(schema.video(slot, side) in paths for side in schema.SIDES)
    ]
    if not slots:
      raise ValueError(f"{self.path} holds no /{schema.CAMERA_ROOT}/<slot> video")
    return slots

  def _entity_row(self, entity: str) -> dict[str, Any]:
    """Component name -> newest value across the entity's chunks."""
    cached = self._rows.get(entity)
    if cached is None:
      cached = self._recording.row(entity)
      self._rows[entity] = cached
    return cached

  def lens_times(self, slot: str, side: str) -> Float64Array:
    """One lens's frame times (s), one per encoded sample."""
    return frames.lens_times(self._recording, schema.video(slot, side))

  @property
  def frame_times(self) -> dict[str, Float64Array]:
    """Per-camera front-lens frame times (s), the grid keyframes must land on."""
    return {slot: self.lens_times(slot, "front") for slot in self.slots}

  def lens_video(self, slot: str, side: str) -> frames.LensVideo:
    """One lens's encoded samples, ready for banded decoding."""
    return frames.LensVideo(self._recording, slot, side)

  def decoded_shape(self, slot: str) -> tuple[int, int]:
    """``(height, width)`` of decoded front-lens frames, from a one-frame decode."""
    frame = frames.LensFrames(
      self.lens_video(slot, "front"), np.zeros(1, dtype=np.int64), band=1
    )[0]
    return int(frame.shape[0]), int(frame.shape[1])

  def lens_setups(self, slot: str, decoded_width: int) -> tuple[LensSetup, LensSetup]:
    """Both lenses at the decode resolution, as the solve holds them."""
    return (
      self._lens_setup(slot, "front", decoded_width),
      self._lens_setup(slot, "rear", decoded_width),
    )

  def _lens_setup(self, slot: str, side: Side, decoded_width: int) -> LensSetup:
    lens = lens_from_row(self._entity_row(schema.lens(slot, side)), decoded_width, side)
    return LensSetup(extrinsic=gtsam.Pose3(lens.body_from_lens), params=lens.params)

  def imu(self, slot: str) -> ImuSamples:
    """Raw-axis IMU samples as stored: t (s), gyro (rad/s), accel (m/s^2)."""
    t_gyro, gyro = self._recording.scalars(schema.imu_signal(slot, "gyro"))
    t_accel, accel = self._recording.scalars(schema.imu_signal(slot, "accel"))
    if not np.array_equal(t_gyro, t_accel):
      raise ValueError(f"{self.path}: gyro and accel rows disagree for {slot}")
    return ImuSamples(t=t_gyro, gyro=gyro, accel=accel)

  def imu_to_body(self, slot: str) -> Float64Array:
    """The IMU mounting rotation, raw IMU axes -> body."""
    row = self._entity_row(schema.imu(slot))
    matrix: Float64Array = quats_to_matrices(row[schema.POSE_QUATERNION])[0]
    return matrix

  def factory_bias(self, slot: str) -> gtsam.imuBias.ConstantBias:
    """The factory IMU bias in raw IMU axes (accel m/s^2, gyro rad/s)."""
    row = self._entity_row(schema.imu(slot))
    accel = np.asarray(row["accel_bias"], dtype=np.float64).reshape(3)
    gyro = np.asarray(row["gyro_bias"], dtype=np.float64).reshape(3)
    return gtsam.imuBias.ConstantBias(accel, gyro)

  def rolling_shutter_readout(self, slot: str) -> float:
    """The rolling-shutter readout time (s) stored with the IMU calibration."""
    row = self._entity_row(schema.imu(slot))
    return float(np.asarray(row["rolling_shutter_readout"]).reshape(-1)[0])
