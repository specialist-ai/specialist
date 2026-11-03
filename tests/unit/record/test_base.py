"""Base-recording chunk builders round-trip through a written rrd."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rerun.chunk import Chunk, LazyChunkStream, RrdReader
from specialist_dataset import schema
from specialist_dataset.read import Recording

from specialist.record.base import (
  calibration_chunks,
  imu_chunks,
  write_base_recording,
)
from specialist.vi_ba.extrinsics import parse_offset_v3
from tests.record_fixtures import synthetic_insv

# Wide enough to keep every synthetic IMU sample.
_WHOLE_SESSION = (-10.0, 10.0)


def _write(tmp_path: Path, chunks: list[Chunk], session: str = "sess") -> Recording:
  out = tmp_path / "base.rrd"
  write_base_recording(
    out, session, [LazyChunkStream.from_iter(chunks[:1])], chunks[1:], {}
  )
  return Recording(out)


def test_imu_chunks_units_and_shift(tmp_path: Path) -> None:
  recording = _write(
    tmp_path, imu_chunks("cam00", synthetic_insv(), 0.25, _WHOLE_SESSION)
  )
  times, gyro = recording.scalars("camera/cam00/imu/gyro", 3)
  assert times[0] == -0.5 + 0.25
  np.testing.assert_allclose(gyro[0], [np.pi / 2, 0.0, 0.0])
  _, accel = recording.scalars("camera/cam00/imu/accel", 3)
  np.testing.assert_allclose(accel[0], [0.0, 0.0, 9.81])


def test_imu_window_trims_rows(tmp_path: Path) -> None:
  recording = _write(
    tmp_path, imu_chunks("cam00", synthetic_insv(), 0.0, window=(0.0, 10.0))
  )
  times, _ = recording.scalars("camera/cam00/imu/gyro", 3)
  assert times.size and times.min() >= 0


def test_calibration_chunks_extrinsics(tmp_path: Path) -> None:
  insv = synthetic_insv()
  recording = _write(tmp_path, calibration_chunks("cam00", insv))
  expected = parse_offset_v3(insv.offset_v3)
  front = recording.row("camera/cam00/front")
  np.testing.assert_allclose(
    np.asarray(front["Transform3D:translation"]).reshape(3),
    np.asarray(expected.body_T_front_lens.translation()),
    atol=1e-12,
  )
  rear = recording.row("camera/cam00/rear")
  np.testing.assert_allclose(
    np.asarray(rear["Transform3D:translation"]).reshape(3),
    np.asarray(expected.body_T_rear_lens.translation()),
    atol=1e-12,
  )
  assert np.asarray(front["xi"]).reshape(-1)[0] == pytest.approx(1.9)
  readout = recording.row("camera/cam00/imu")["rolling_shutter_readout"]
  assert np.asarray(readout).reshape(-1)[0] == pytest.approx(0.0157)


def test_write_base_recording_identity_and_properties(tmp_path: Path) -> None:
  out = tmp_path / "base.rrd"
  chunks = imu_chunks("cam00", synthetic_insv(), 0.0, _WHOLE_SESSION)
  write_base_recording(
    out,
    "ft005",
    [LazyChunkStream.from_iter(chunks)],
    [],
    {schema.BASE_PROPERTIES: {"task": "fold_towel", "date": "2026-08-23"}},
  )
  reader = RrdReader(str(out))
  entries = reader.recordings()
  assert [e.recording_id for e in entries] == ["ft005"]
  chunk_paths = {c.entity_path for c in reader.store().stream().to_chunks()}
  assert "/camera/cam00/imu/gyro" in chunk_paths
  # entity_paths() hides internal paths; properties live under /__properties/.
  assert f"/__properties/{schema.BASE_PROPERTIES}" in chunk_paths


def test_write_base_recording_failure_leaves_no_final_file(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  out = tmp_path / "base.rrd"

  def explode(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("mid-write failure")

  monkeypatch.setattr("rerun.send_chunks", explode)
  chunks = imu_chunks("cam00", synthetic_insv(), 0.0, _WHOLE_SESSION)
  with pytest.raises(RuntimeError, match="mid-write failure"):
    write_base_recording(out, "s", [LazyChunkStream.from_iter(chunks)], [], {})
  assert not out.exists()


def test_write_base_recording_replaces_a_stale_partial(tmp_path: Path) -> None:
  out = tmp_path / "base.rrd"
  stale = tmp_path / "base.rrd.partial"
  stale.write_bytes(b"leftover from a crashed run")
  chunks = imu_chunks("cam00", synthetic_insv(), 0.0, _WHOLE_SESSION)
  write_base_recording(out, "s", [LazyChunkStream.from_iter(chunks)], [], {})
  assert not stale.exists()
  assert RrdReader(str(out)).recordings()[0].recording_id == "s"
