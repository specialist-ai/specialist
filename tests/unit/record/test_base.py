"""Base-recording chunk builders round-trip through a written rrd."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rerun.experimental import Chunk, LazyChunkStream, RrdReader

from specialist.loader.parse_insv import InsvData, LensIntrinsics
from specialist.record.base import (
  calibration_chunks,
  imu_chunks,
  write_base_recording,
)
from specialist.vi_ba.configs import ImuCfg
from specialist.vi_ba.extrinsics import parse_offset_v3


def _synthetic_insv() -> InsvData:
  block = [
    1.9,  # xi
    5000.0,
    5001.0,
    2970.0,
    2971.0,  # fx fy cx cy
    0.5,
    -0.3,
    90.0,  # yaw pitch roll deg
    0.01,
    0.02,
    0.03,  # t
    0.1,
    -0.05,
    0.01,
    0.001,
    -0.002,  # k1..k3 p1 p2
    11904.0,
    5952.0,
    0.0,  # frame w h unknown
  ]
  offset_v3 = np.array([2.0, *block, *block, 0.0])
  return InsvData(
    imu_timestamps=np.array([-0.5, 0.0, 0.001, 0.002]),
    gyro_deg=np.array([[90.0, 0.0, 0.0]] * 4),
    accel=np.array([[0.0, 0.0, 9.81]] * 4),
    intrinsics=LensIntrinsics(
      camera_matrix=np.eye(3),
      distortion_coeffs=np.zeros(6),
      calib_width=11904,
      calib_height=5952,
      distortion_model="mei",
    ),
    rolling_shutter_readout=0.0157,
    offset_v3=offset_v3,
    gyro_calib_numbers=np.array([0.001, 0.002, 0.003, 0.01, 0.02, 0.03]),
  )


def _write(tmp_path: Path, chunks: list[Chunk], session: str = "sess") -> RrdReader:
  out = tmp_path / "base.rrd"
  write_base_recording(
    out, session, [LazyChunkStream.from_iter(chunks[:1])], chunks[1:], {}
  )
  return RrdReader(str(out))


def _column(reader: RrdReader, entity: str, component: str) -> tuple[Any, Any]:
  """All (time, value) rows of one component, times None for static."""
  times: list[Any] = []
  values: list[Any] = []
  for chunk in reader.store().stream().to_chunks():
    if chunk.entity_path != entity:
      continue
    batch = chunk.to_record_batch()
    names = batch.schema.names
    if component not in names:
      continue
    col = batch.column(names.index(component))
    if chunk.is_static:
      values += col.to_pylist()
      times += [None] * len(col)
    else:
      timeline = chunk.timeline_names[0]
      times += [v.value for v in batch.column(names.index(timeline))]
      values += col.to_pylist()
  return times, values


def test_imu_chunks_units_and_shift(tmp_path: Path) -> None:
  reader = _write(tmp_path, imu_chunks("tracker_0", _synthetic_insv(), 0.25))
  times, gyro = _column(reader, "/imu/tracker_0/gyro", "Scalars:scalars")
  assert times[0] == round((-0.5 + 0.25) * 1e9)
  np.testing.assert_allclose(gyro[0], [np.pi / 2, 0.0, 0.0])
  _, accel = _column(reader, "/imu/tracker_0/accel", "Scalars:scalars")
  np.testing.assert_allclose(accel[0], [0.0, 0.0, 9.81])


def test_imu_window_trims_rows(tmp_path: Path) -> None:
  reader = _write(
    tmp_path, imu_chunks("tracker_0", _synthetic_insv(), 0.0, window=(0.0, 10.0))
  )
  times, _ = _column(reader, "/imu/tracker_0/gyro", "Scalars:scalars")
  assert times and min(times) >= 0


def test_calibration_chunks_extrinsics(tmp_path: Path) -> None:
  insv = _synthetic_insv()
  reader = _write(
    tmp_path, calibration_chunks("tracker_0", insv, np.asarray(ImuCfg().imu_to_body))
  )
  expected = parse_offset_v3(insv.offset_v3)
  _, front_t = _column(
    reader, "/calibration/tracker_0/front", "Transform3D:translation"
  )
  np.testing.assert_allclose(
    np.asarray(front_t[0]).reshape(3),
    np.asarray(expected.body_T_front_lens.translation()),
    atol=1e-12,
  )
  _, rear_t = _column(reader, "/calibration/tracker_0/rear", "Transform3D:translation")
  np.testing.assert_allclose(
    np.asarray(rear_t[0]).reshape(3),
    np.asarray(expected.body_T_rear_lens.translation()),
    atol=1e-12,
  )
  _, xi = _column(reader, "/calibration/tracker_0/front", "xi")
  assert np.asarray(xi[0]).reshape(-1)[0] == pytest.approx(1.9)
  _, readout = _column(reader, "/calibration/tracker_0/imu", "rolling_shutter_readout")
  assert np.asarray(readout[0]).reshape(-1)[0] == pytest.approx(0.0157)


def test_write_base_recording_identity_and_properties(tmp_path: Path) -> None:
  out = tmp_path / "base.rrd"
  chunks = imu_chunks("tracker_0", _synthetic_insv())
  write_base_recording(
    out,
    "ft005",
    [LazyChunkStream.from_iter(chunks)],
    [],
    {"session": {"session_id": "ft005"}, "audio_offset": {"tracker_0": 0.0}},
  )
  reader = RrdReader(str(out))
  entries = reader.recordings()
  assert [e.recording_id for e in entries] == ["ft005"]
  chunk_paths = {c.entity_path for c in reader.store().stream().to_chunks()}
  assert "/imu/tracker_0/gyro" in chunk_paths
  # entity_paths() hides internal paths; properties live under /__properties/.
  assert "/__properties/session" in chunk_paths
  assert "/__properties/audio_offset" in chunk_paths
