"""Base-recording assembly: the session's raw facts as one rrd.

Video, IMU and calibration merge into one stream, written under the session's
``recording_id`` so later layers attach to it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import gtsam
import numpy as np
import numpy.typing as npt
import rerun as rr
from rerun.chunk import Chunk, LazyChunkStream, OptimizationProfile
from specialist_corpus import schema

from specialist.loader.parse_insv import InsvData
from specialist.vi_ba.extrinsics import LensOffset, parse_insv_extrinsics
from specialist.vi_ba.geometry import quat_xyzw

Float64Array = npt.NDArray[np.float64]


def imu_chunks(
  slot: str,
  insv: InsvData,
  time_shift: float,
  window: tuple[float, float],
) -> list[Chunk]:
  """Gyro (rad/s) and accel (m/s^2) at their native per-sample timestamps,
  trimmed row-exact to the session ``window``."""
  shifted = insv.imu_timestamps + time_shift
  keep = (shifted >= window[0]) & (shifted <= window[1])
  times = rr.TimeColumn(schema.TIMELINE, duration=shifted[keep])
  gyro = np.radians(insv.gyro_deg[keep])
  return [
    Chunk.from_columns(
      schema.imu_signal(slot, "gyro"),
      indexes=[times],
      columns=rr.Scalars.columns(scalars=gyro),
    ),
    Chunk.from_columns(
      schema.imu_signal(slot, "accel"),
      indexes=[times],
      columns=rr.Scalars.columns(scalars=insv.accel[keep]),
    ),
  ]


def _lens_calibration(entity: str, offset: LensOffset, pose: gtsam.Pose3) -> Chunk:
  return Chunk.from_columns(
    entity,
    indexes=[],
    columns=[
      *rr.Transform3D.columns(
        translation=np.asarray(pose.translation(), dtype=np.float64)[None, :],
        quaternion=quat_xyzw(pose.rotation())[None, :],
      ),
      *rr.AnyValues.columns(
        fx=[offset.fx],
        fy=[offset.fy],
        cx=[offset.cx],
        cy=[offset.cy],
        k1=[offset.k1],
        k2=[offset.k2],
        k3=[offset.k3],
        p1=[offset.p1],
        p2=[offset.p2],
        xi=[offset.xi],
        frame_width=[float(offset.frame_width)],
        frame_height=[float(offset.frame_height)],
      ),
    ],
  )


# Raw IMU axes to the body frame.
IMU_TO_BODY: Float64Array = np.array(
  [(0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)], dtype=np.float64
)


def exposure_chunks(
  slot: str,
  insv: InsvData,
  time_shift: float,
  window: tuple[float, float],
) -> list[Chunk]:
  """Per-frame exposure durations and the camera's auto-exposure reports at
  their native timestamps, trimmed row-exact to the session ``window``."""
  chunks: list[Chunk] = []
  if insv.exposure_timestamps.size:
    shifted = insv.exposure_timestamps + time_shift
    keep = (shifted >= window[0]) & (shifted <= window[1])
    chunks.append(
      Chunk.from_columns(
        schema.exposure(slot),
        indexes=[rr.TimeColumn(schema.TIMELINE, duration=shifted[keep])],
        columns=rr.Scalars.columns(scalars=insv.exposure_durations[keep]),
      )
    )
  if insv.ae_timestamps.size:
    shifted = insv.ae_timestamps + time_shift
    keep = (shifted >= window[0]) & (shifted <= window[1])
    chunks.append(
      Chunk.from_columns(
        schema.autoexposure(slot),
        indexes=[rr.TimeColumn(schema.TIMELINE, duration=shifted[keep])],
        columns=cast("Any", rr.AnyValues).columns(
          **{name: values[keep] for name, values in insv.ae_stats.items()}
        ),
      )
    )
  return chunks


def calibration_chunks(slot: str, insv: InsvData) -> list[Chunk]:
  """Static per-lens MEI intrinsics in raw stitched-sensor units, carrying the
  frame size they are expressed in, the body -> lens extrinsics, and the IMU
  mounting with its factory bias calibration."""
  extrinsics = parse_insv_extrinsics(insv)
  rotation = gtsam.Rot3(IMU_TO_BODY)
  chunks = [
    _lens_calibration(
      schema.lens(slot, "front"),
      extrinsics.front,
      extrinsics.body_T_front_lens,
    ),
    _lens_calibration(
      schema.lens(slot, "rear"),
      extrinsics.rear,
      extrinsics.body_T_rear_lens,
    ),
    Chunk.from_columns(
      schema.imu(slot),
      indexes=[],
      columns=[
        *rr.Transform3D.columns(
          translation=np.zeros((1, 3)), quaternion=quat_xyzw(rotation)[None, :]
        ),
        *rr.AnyValues.columns(
          rolling_shutter_readout=[insv.rolling_shutter_readout],
          gyro_bias=[insv.gyro_calib_numbers[:3]],
          accel_bias=[insv.gyro_calib_numbers[3:]],
        ),
      ],
    ),
  ]
  return chunks


# The IMU calibration already carries this, in seconds.
_SUPERSEDED_CAPTURE = frozenset({"rolling_shutter_time"})


def capture_chunk(
  slot: str,
  insv: InsvData,
  source: str,
  clock_shift: float,
  clock_confidence: float,
  clock_method: str,
) -> Chunk:
  """The camera's own account of itself, plus the shift that put its clock on
  the session's: ``t_session = t_capture + clock_shift``."""
  return Chunk.from_columns(
    schema.camera(slot),
    indexes=[],
    columns=cast("Any", rr.AnyValues).columns(
      source=[source],
      clock_shift=[clock_shift],
      clock_confidence=[clock_confidence],
      clock_method=[clock_method],
      **{
        name: [value]
        for name, value in insv.capture.items()
        if name not in _SUPERSEDED_CAPTURE
      },
    ),
  )


def write_base_recording(
  out_path: Path,
  session_id: str,
  video_streams: list[LazyChunkStream],
  sidecar_chunks: list[Chunk],
  properties: dict[str, dict[str, Any]],
) -> None:
  """Merge all streams, optimize for storage, and write one base rrd."""
  merged = LazyChunkStream.merge(
    *video_streams, LazyChunkStream.from_iter(sidecar_chunks)
  )
  # The GOP-granular trim keeps the reader's one keyframe-marker chunk whole, so
  # the profile re-derives the markers from the samples that survive.
  optimize = dataclasses.replace(OptimizationProfile.OBJECT_STORE, fix_keyframe=True)
  store = merged.collect(optimize=optimize)
  partial = out_path.parent / (out_path.name + ".partial")
  partial.unlink(missing_ok=True)
  # The partial becomes the recording only once whole.
  try:
    with rr.RecordingStream(schema.APPLICATION_ID, recording_id=session_id) as stream:
      stream.save(str(partial))
      rr.send_chunks(store, recording=stream)
      for name, values in properties.items():
        if values:
          stream.send_property(name, rr.AnyValues(**values))
    partial.replace(out_path)
  finally:
    partial.unlink(missing_ok=True)
