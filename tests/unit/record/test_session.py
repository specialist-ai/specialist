"""``SessionRecording`` reads back everything the solve needs from a base rrd:
identity, per-lens PTS, decoded frames, calibration, IMU, and the PTS grid
alignment between the two lenses' GOP-trimmed row grids."""

from __future__ import annotations

import subprocess
from pathlib import Path

import gtsam
import numpy as np
import pytest

from specialist.loader.audio_sync import resolve_ffmpeg
from specialist.loader.parse_insv import InsvData, LensIntrinsics
from specialist.record.base import (
  calibration_chunks,
  imu_chunks,
  write_base_recording,
)
from specialist.record.frames import RecordingKeyframes
from specialist.record.session import (
  SessionRecording,
  covered_rows,
  fps_from_times,
  nearest_rows,
)
from specialist.record.video import normalize_h264, video_chunks
from specialist.vi_ba.configs import ImuCfg
from specialist.vi_ba.extrinsics import parse_offset_v3

_FPS = 30
# The rear lens starts one frame earlier and runs two frames longer than the
# front: counts differ, times overlap.
_FRONT_FRAMES = 18
_REAR_FRAMES = 21
_REAR_SHIFT = -1.0 / _FPS
# Wide enough to keep every frame and IMU sample.
_WINDOW = (-1.0, 10.0)


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
    source_path=Path("cap.insv"),
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


def _encode_clip(path: Path, n_frames: int) -> Path:
  """A tiny H.264 clip without B-frames, keyframes every 5 frames, so the
  recording holds several GOPs and decode order equals presentation order."""
  subprocess.run(
    [
      resolve_ffmpeg(),
      "-y",
      "-loglevel",
      "error",
      "-f",
      "lavfi",
      "-i",
      f"testsrc=duration={n_frames / _FPS}:size=64x64:rate={_FPS}",
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      "-bf",
      "0",
      "-g",
      "5",
      str(path),
    ],
    check=True,
  )
  return path


@pytest.fixture(scope="module")
def base_rrd(tmp_path_factory: pytest.TempPathFactory) -> Path:
  tmp = tmp_path_factory.mktemp("session")
  insv = _synthetic_insv()
  streams = []
  for lens, n_frames, shift in (
    ("front", _FRONT_FRAMES, 0.0),
    ("rear", _REAR_FRAMES, _REAR_SHIFT),
  ):
    clip = _encode_clip(tmp / f"{lens}.mp4", n_frames)
    clean = normalize_h264(clip, tmp / f"{lens}_clean.mp4")
    streams.append(
      video_chunks(clean, f"image/tracker_0/{lens}", time_shift=shift, window=_WINDOW)
    )
  sidecars = imu_chunks(
    "tracker_0", insv, time_shift=0.0, window=_WINDOW
  ) + calibration_chunks("tracker_0", insv, np.asarray(ImuCfg().imu_to_body))
  out = tmp / "base.rrd"
  write_base_recording(
    out,
    "sess",
    streams,
    sidecars,
    {"session": {"sources": ["a_00_.insv"]}},
  )
  return out


def test_identity_names_and_properties(base_rrd: Path) -> None:
  session = SessionRecording(base_rrd)
  assert session.session_id == "sess"
  assert session.tracker_names == ["tracker_0"]
  assert session.sources == ["a_00_.insv"]


def test_fps_is_pts_derived(base_rrd: Path) -> None:
  session = SessionRecording(base_rrd)
  front = session.lens_times("tracker_0", "front")
  rear = session.lens_times("tracker_0", "rear")
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
  front = session.lens_video("tracker_0", "front")
  rear = session.lens_video("tracker_0", "rear")
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
  video = session.lens_video("tracker_0", "front")
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
  video = session.lens_video("tracker_0", "front")
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
  assert session.decoded_shape("tracker_0") == (64, 64)
  front, rear = session.lens_setups("tracker_0", 64)
  expected = parse_offset_v3(_synthetic_insv().offset_v3)
  scale = 2 * 64 / expected.front.frame_width
  assert front.params.fx == pytest.approx(expected.front.fx * scale)
  assert front.params.cx == pytest.approx(expected.front.cx * scale)
  assert front.params.xi == pytest.approx(expected.front.xi)
  assert front.params.k1 == pytest.approx(expected.front.k1)
  half_raw = expected.rear.frame_width / 2.0
  assert rear.params.cx == pytest.approx((expected.rear.cx - half_raw) * scale)
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
  imu = session.imu("tracker_0")
  assert imu.t[0] == pytest.approx(-0.5, abs=1e-9)
  np.testing.assert_allclose(imu.gyro[0], [np.pi / 2, 0.0, 0.0], atol=1e-12)
  np.testing.assert_allclose(imu.accel[0], [0.0, 0.0, 9.81], atol=1e-12)
  assert np.all(np.diff(imu.t) >= 0)


def test_imu_calibration_roundtrips(base_rrd: Path) -> None:
  session = SessionRecording(base_rrd)
  insv = _synthetic_insv()
  numbers = insv.gyro_calib_numbers
  expected = gtsam.imuBias.ConstantBias(numbers[3:6], numbers[0:3])
  bias = session.factory_bias("tracker_0")
  np.testing.assert_allclose(
    np.asarray(bias.accelerometer()), np.asarray(expected.accelerometer()), atol=1e-12
  )
  np.testing.assert_allclose(
    np.asarray(bias.gyroscope()), np.asarray(expected.gyroscope()), atol=1e-12
  )
  np.testing.assert_allclose(
    session.imu_to_body("tracker_0"), np.asarray(ImuCfg().imu_to_body), atol=1e-12
  )
  assert session.rolling_shutter_readout("tracker_0") == pytest.approx(0.0157)
