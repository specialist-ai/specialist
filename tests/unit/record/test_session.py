"""``SessionRecording`` reads back everything the solve needs from a base rrd:
identity, per-lens PTS, decoded frames, calibration, IMU, and the PTS grid
alignment between the two lenses' GOP-trimmed row grids."""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import pytest
from rerun.chunk import RrdReader
from specialist_corpus import frames, schema
from specialist_corpus.frames import RecordingKeyframes

from specialist.record.base import (
  IMU_TO_BODY,
  calibration_chunks,
  capture_chunk,
  exposure_chunks,
  imu_chunks,
  write_base_recording,
)
from specialist.record.session import (
  SessionRecording,
  covered_rows,
  fps_from_times,
  nearest_rows,
)
from specialist.record.video import normalize_h264, video_chunks
from specialist.vi_ba.extrinsics import parse_offset_v3
from tests.record_fixtures import encode_clip, synthetic_insv

_FPS = 30
# The rear lens starts one frame earlier and runs two frames longer than the
# front: counts differ, times overlap.
_FRONT_FRAMES = 18
_REAR_FRAMES = 21
_REAR_SHIFT = -1.0 / _FPS
# Wide enough to keep every frame and IMU sample.
_WINDOW = (-1.0, 10.0)


@pytest.fixture(scope="module")
def base_rrd(tmp_path_factory: pytest.TempPathFactory) -> Path:
  tmp = tmp_path_factory.mktemp("session")
  insv = synthetic_insv()
  streams = []
  for lens, n_frames, shift in (
    ("front", _FRONT_FRAMES, 0.0),
    ("rear", _REAR_FRAMES, _REAR_SHIFT),
  ):
    clip = encode_clip(tmp / f"{lens}.mp4", n_frames, rate=_FPS, gop=5)
    clean = normalize_h264(clip, tmp / f"{lens}_clean.mp4")
    streams.append(
      video_chunks(clean, schema.video("cam00", lens), time_shift=shift, window=_WINDOW)
    )
  sidecars = [
    *imu_chunks("cam00", insv, time_shift=0.0, window=_WINDOW),
    *exposure_chunks("cam00", insv, time_shift=0.0, window=_WINDOW),
    *calibration_chunks("cam00", insv),
    capture_chunk("cam00", insv, "a_00_.insv", -0.25, 4.5, "audio"),
  ]
  out = tmp / "base.rrd"
  write_base_recording(out, "sess", streams, sidecars, {"base": {"task": "fold"}})
  return out


def test_identity_slots(base_rrd: Path) -> None:
  session = SessionRecording(base_rrd)
  assert session.session_id == "sess"
  assert session.slots == ["cam00"]


def test_fps_is_pts_derived(base_rrd: Path) -> None:
  session = SessionRecording(base_rrd)
  front = session.lens_times("cam00", "front")
  rear = session.lens_times("cam00", "rear")
  assert fps_from_times(front) == pytest.approx(_FPS, rel=1e-6)
  assert len(front) == _FRONT_FRAMES
  assert len(rear) == _REAR_FRAMES
  assert front[0] == pytest.approx(0.0, abs=1e-9)
  assert rear[0] == pytest.approx(_REAR_SHIFT, abs=1e-6)


def test_fps_from_times_undoes_ntsc_rounding() -> None:
  times = np.arange(300) * 1001 / 30000
  assert fps_from_times(times) == pytest.approx(30000 / 1001, rel=1e-9)
  with pytest.raises(ValueError, match="two frame times"):
    fps_from_times(np.array([0.0]))


def test_grid_alignment_matches_rear_rows_by_pts(base_rrd: Path) -> None:
  """The keyframe grid is the front lens's PTS; the rear lens's rows come from
  nearest-PTS matching, absorbing the one-frame edge trim difference."""
  session = SessionRecording(base_rrd)
  front = session.lens_video("cam00", "front")
  rear = session.lens_video("cam00", "rear")
  grid = front.times[::2]
  rows = nearest_rows(rear.times, grid)
  np.testing.assert_array_equal(rows, np.arange(1, _FRONT_FRAMES + 1, 2))
  assert float(np.abs(rear.times[rows] - grid).max()) < 0.5 / _FPS


def test_nearest_rows_raises_past_half_a_frame() -> None:
  times = np.arange(15) / 30.0
  grid = np.arange(0, 20, 2) / 30.0  # tail grid points have no matching row
  with pytest.raises(ValueError, match="half a frame"):
    nearest_rows(times, grid)
  # a sub-half-frame jitter is matched, not rejected
  rows = nearest_rows(times, times[::2] + 0.25 / 30.0)
  np.testing.assert_array_equal(rows, np.arange(0, 15, 2))


def test_covered_rows_spans_then_clips_to_the_common_overlap() -> None:
  """Every front row survives when both other streams span it; rows outside the
  three-way overlap drop, with half a frame of slack at each edge."""
  fps = 30.0
  front = np.arange(10) / fps
  rear = np.arange(-1, 11) / fps
  imu = np.arange(-5, 55) / (5 * fps)
  np.testing.assert_array_equal(covered_rows(front, rear, imu, fps), np.arange(10))
  # The rear lens starts a third of a frame past front row 2 and the IMU stops a
  # third of a frame short of row 6; the half-frame tolerance keeps both rows.
  late_rear = np.arange(10) / fps + 2.3 / fps
  short_imu = np.arange(58) / (10 * fps)
  np.testing.assert_array_equal(
    covered_rows(front, late_rear, short_imu, fps), np.arange(2, 7)
  )


def test_recording_keyframes_rows_select_exact_frames(base_rrd: Path) -> None:
  session = SessionRecording(base_rrd)
  video = session.lens_video("cam00", "front")
  reference = RecordingKeyframes(video, np.arange(_FRONT_FRAMES), band=64)
  chosen = np.array([0, 3, 5, 11, 17])
  banded = RecordingKeyframes(video, chosen, band=2)
  assert len(banded) == len(chosen)
  for i, row in enumerate(chosen):
    assert np.array_equal(banded[i], reference[int(row)]), f"row {row} differs"
  # The rows index the lens's own PTS table.
  np.testing.assert_allclose(video.times[chosen], chosen / _FPS, atol=1e-6)


def test_seam_reach_back_does_not_redecode(base_rrd: Path) -> None:
  """An ascending pair walk across band seams costs one monotone decode
  sweep."""
  session = SessionRecording(base_rrd)
  video = session.lens_video("cam00", "front")
  reference = RecordingKeyframes(video, np.arange(_FRONT_FRAMES), band=64)
  banded = RecordingKeyframes(video, np.arange(_FRONT_FRAMES), band=4)
  decodes: list[int] = []
  original = banded._decode_band

  def counting(first: int, span: int) -> list[np.ndarray]:
    decodes.append(first)
    return original(first, span)

  banded._decode_band = counting  # type: ignore[method-assign]
  for k in range(1, _FRONT_FRAMES):
    cur, prev = banded[k], banded[k - 1]
    assert np.array_equal(cur, reference[k])
    assert np.array_equal(prev, reference[k - 1])
  assert decodes == [0, 4, 8, 12, 16]


def test_decoded_shape_and_lens_setups(base_rrd: Path) -> None:
  session = SessionRecording(base_rrd)
  assert session.decoded_shape("cam00") == (64, 64)
  front, rear = session.lens_setups("cam00", 64)
  expected = parse_offset_v3(synthetic_insv().offset_v3)
  scale = 2 * 64 / expected.front.frame_width
  assert front.params.fx == pytest.approx(expected.front.fx * scale)
  assert front.params.cx == pytest.approx(expected.front.cx * scale)
  assert front.params.xi == pytest.approx(expected.front.xi)
  assert front.params.k1 == pytest.approx(expected.front.k1)
  half_raw = expected.rear.frame_width / 2.0
  assert rear.params.cx == pytest.approx((expected.rear.cx - half_raw) * scale)
  # The calibrated radius is a fraction of the decoded per-lens width.
  assert front.calibrated_radius == pytest.approx(64 * 1000.0 / 2880.0)
  assert rear.calibrated_radius == front.calibrated_radius
  wide, _ = session.lens_setups("cam00", 128)
  assert wide.calibrated_radius == pytest.approx(2 * front.calibrated_radius)
  # Transform3D stores float32, so the round-trip is exact to ~1e-7 only.
  np.testing.assert_allclose(
    front.extrinsic.matrix(), expected.body_T_front_lens.matrix(), atol=1e-6
  )
  np.testing.assert_allclose(
    rear.extrinsic.matrix(), expected.body_T_rear_lens.matrix(), atol=1e-6
  )


def test_imu_comes_back_in_si_without_reconversion(base_rrd: Path) -> None:
  """The base stores gyro already in rad/s; the reader must not deg2rad again."""
  session = SessionRecording(base_rrd)
  imu = session.imu("cam00")
  assert imu.t[0] == pytest.approx(-0.5, abs=1e-9)
  np.testing.assert_allclose(imu.gyro[0], [np.pi / 2, 0.0, 0.0], atol=1e-12)
  np.testing.assert_allclose(imu.accel[0], [0.0, 0.0, 9.81], atol=1e-12)
  assert np.all(np.diff(imu.t) >= 0)


def test_imu_calibration_roundtrips(base_rrd: Path) -> None:
  session = SessionRecording(base_rrd)
  insv = synthetic_insv()
  numbers = insv.gyro_calib_numbers
  expected = gtsam.imuBias.ConstantBias(numbers[3:6], numbers[0:3])
  bias = session.factory_bias("cam00")
  np.testing.assert_allclose(
    np.asarray(bias.accelerometer()), np.asarray(expected.accelerometer()), atol=1e-12
  )
  np.testing.assert_allclose(
    np.asarray(bias.gyroscope()), np.asarray(expected.gyroscope()), atol=1e-12
  )
  np.testing.assert_allclose(session.imu_to_body("cam00"), IMU_TO_BODY, atol=1e-12)


def test_capture_telemetry_reaches_the_base(base_rrd: Path) -> None:
  """Identity, the clock shift that put this capture on the session clock,
  per-frame exposure and the AE reports all ride the base recording."""
  rows: dict[str, int] = {}
  columns: dict[str, set[str]] = {}
  for chunk in RrdReader(str(base_rrd)).store().stream().to_chunks():
    e = chunk.entity_path
    rows[e] = rows.get(e, 0) + chunk.num_rows
    batch_names = chunk.to_record_batch().schema.names
    columns.setdefault(e, set()).update(
      n.split(":")[-1] for n in batch_names if n not in chunk.timeline_names
    )
  cap = f"/{schema.camera('cam00')}"
  assert {"serial_number", "camera_type", "fw_version", "photo_rot"} <= columns[cap]
  assert {"source", "clock_shift", "clock_confidence", "clock_method"} <= columns[cap]
  row = SessionRecording(base_rrd)._recording.row(cap.lstrip("/"))
  # The shift is the one the streams were written with.
  assert np.asarray(row["clock_shift"]).reshape(-1)[0] == pytest.approx(-0.25)
  assert str(np.asarray(row["source"]).reshape(-1)[0]) == "a_00_.insv"
  # The rolling shutter is the IMU calibration's, in seconds, and only there.
  assert "rolling_shutter_time" not in columns[cap]
  assert rows[f"/{schema.exposure('cam00')}"] == 3
  assert rows[f"/{schema.autoexposure('cam00')}"] == 2
  assert "iso_value" in columns[f"/{schema.autoexposure('cam00')}"]


def test_lenses_stage_where_the_session_says(base_rrd: Path, tmp_path: Path) -> None:
  frames.release_videos()
  stage = tmp_path / "stage"
  stage.mkdir()
  session = SessionRecording(base_rrd, stage)
  video = session.lens_video("cam00", "front")
  assert video.path.parent == stage
  session.close()
  assert stage.exists() and list(stage.iterdir()) == []


def test_a_session_removes_the_stage_it_made(base_rrd: Path) -> None:
  frames.release_videos()
  session = SessionRecording(base_rrd)
  own = session.stage
  session.lens_video("cam00", "front")
  session.close()
  assert not own.exists()
