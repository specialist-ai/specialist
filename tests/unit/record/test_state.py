"""The state layer round-trips: a synthetic base and estimate through
``state.py``, read back as a trainer sees them."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation
from scripts.state import Args, Cut, main
from specialist_camera import disc_mask, to_image
from specialist_corpus import schema
from specialist_corpus.frames import LensVideo, decode_rows
from specialist_corpus.read import Recording

from specialist.record.base import (
  calibration_chunks,
  imu_chunks,
  write_base_recording,
)
from specialist.record.video import normalize_h264, video_chunks
from tests.record_fixtures import (
  BASE_TAG,
  SPIN,
  TILT,
  camera_estimate,
  encode_clip,
  synthetic_insv,
  urdf_file,
  write_run,
)

_FPS = 30
# Rows 0..11; the estimate fixture's keyframes sit on rows 0, 3, 6, 9.
_FRAMES = 12
_WINDOW = (-1.0, 10.0)
_SCALE = 4
_F_ROW = 1000.0
# The second camera's clock sits a sub-frame offset off the first's, as two
# real captures' do.
_SHIFT = {"cam00": 0.0, "cam01": -0.0043}
# One tracker slower than the lattice and one faster, as two real IMUs are.
_RATE = {"cam00": 997.0, "cam01": 1003.0}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
  """A layer-first root: a two-camera base with rear lenses, and its estimate."""
  root = tmp_path_factory.mktemp("corpus")
  insv = synthetic_insv()
  clip = normalize_h264(
    encode_clip(root / "clip.mp4", _FRAMES, rate=_FPS, gop=4), root / "clean.mp4"
  )
  streams, sidecars = [], []
  for slot, shift in _SHIFT.items():
    streams.append(video_chunks(clip, schema.video(slot, "rear"), shift, _WINDOW))
    sidecars += imu_chunks(slot, insv, 0.0, _WINDOW) + calibration_chunks(slot, insv)
  base = schema.layer_rrd(root, schema.BASE, "sess")
  base.parent.mkdir()
  write_base_recording(base, "sess", streams, sidecars, {})
  cameras = [
    camera_estimate("cam00", rate=_RATE["cam00"]),
    camera_estimate("cam01", offset=1.0, rate=_RATE["cam01"]),
  ]
  write_run(root, "sess", cameras, urdf_path=urdf_file(root), frame_period=1.0 / _FPS)
  return root


def _state(corpus: Path, out_dir: Path, **kwargs: object) -> Path:
  base = schema.layer_rrd(corpus, schema.BASE, "sess")
  args = {"scale": _SCALE, "f_row": _F_ROW, **kwargs}
  main(Args(base=base, output_dir=out_dir, **args))  # type: ignore[arg-type]
  return schema.layer_rrd(out_dir, schema.state_layer(_SCALE, _F_ROW), "sess")


def _frames(
  recording: Recording, hand: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
  entity = schema.state_image(hand)
  table = recording.series(f"/{entity}")
  blobs = table.column(schema.selector(entity, "EncodedImage:blob")).to_pylist()
  images = [
    np.asarray(Image.open(io.BytesIO(bytes(b[0]))).convert("RGB")) for b in blobs
  ]
  rows = table.column(schema.ROW_TIMELINE).to_numpy()
  captured = table.column(schema.selector(entity, schema.CAPTURE_ROW)).to_numpy()
  return schema.times_ns(table.column(schema.TIMELINE)), rows, captured, images


def _expected_state(offset: float, rate: float) -> dict[int, dict[str, np.ndarray]]:
  """The fixture's state at each millisecond the samples at ``rate`` Hz
  bracket, from the closed form: the body slides along x while it turns about
  world z, so the rate is ``SPIN`` about z and the tag rides the turned
  mounting."""
  last = round(0.3 * rate) / rate
  rows = np.arange(0, int(np.floor(last * _F_ROW)) + 1)
  t = rows / _F_ROW
  turn = Rotation.from_rotvec(np.outer(SPIN * t, [0.0, 0.0, 1.0])) * (
    Rotation.from_rotvec([TILT, 0.0, 0.0])
  )
  arm = turn.apply(np.asarray(BASE_TAG.translation()))
  tag = Rotation.from_quat(np.asarray(BASE_TAG.rotation().toQuaternion().coeffs()))
  quats = (turn * tag).as_quat()
  quats[quats[:, 3] < 0.0] *= -1.0
  position = np.stack([offset + t, np.zeros_like(t), np.zeros_like(t)], 1) + arm
  omega = np.tile([0.0, 0.0, SPIN], (len(t), 1))
  lin_vel = np.array([1.0, 0.0, 0.0]) + np.cross(omega, arm)
  return {
    int(k): {
      "t": t[i],
      "position": position[i],
      "quat": quats[i],
      "lin_vel": lin_vel[i],
      "ang_vel": omega[i],
    }
    for i, k in enumerate(rows)
  }


def test_the_layer_holds_each_hand_by_name(corpus: Path) -> None:
  out = _state(corpus, corpus, cut=Cut(0.0, 0.0))
  base = Recording(schema.layer_rrd(corpus, schema.BASE, "sess"))
  with Recording(out) as recording:
    # The left hand is cam00 by the anchor rule; each hand's frames keep their
    # own camera's capture times, and both hands' frames of one session frame
    # carry that frame's row: the label names the sample, the clock the instant,
    # and the capture row the millisecond the frame's own shutter fell on.
    for hand, slot in (("left", "cam00"), ("right", "cam01")):
      assert recording.row(schema.state_hand(hand))["slot"] == [slot]
      times, rows, captured, images = _frames(recording, hand)
      frame_times = np.arange(1, 9) / _FPS + _SHIFT[slot]
      np.testing.assert_allclose(times / 1e9, frame_times, atol=1e-6)
      np.testing.assert_array_equal(rows, [33, 67, 100, 133, 167, 200, 233, 267])
      np.testing.assert_array_equal(captured, np.rint(frame_times * _F_ROW))
      # The hands sit side by side in one plane, so a card shows the pair.
      placed = np.asarray(
        recording.row(schema.state_image(hand))[schema.POSE_TRANSLATION]
      ).reshape(3)
      width = images[0].shape[1]
      np.testing.assert_array_equal(
        placed, [0.0 if hand == "left" else width, 0.0, 0.0]
      )
      video = LensVideo(base, slot, "rear")
      frames = np.arange(1, 9)
      for image, (_, frame) in zip(images, decode_rows(video, frames), strict=True):
        np.testing.assert_array_equal(
          image, to_image(frame.to_ndarray(format="rgb24"), _SCALE)
        )
        assert image[~disc_mask(*image.shape[:2])].max() == 0
    assert np.array_equal(_frames(recording, "left")[1], _frames(recording, "right")[1])
    assert not np.array_equal(
      _frames(recording, "left")[2], _frames(recording, "right")[2]
    )
    account = {
      k: np.asarray(v).reshape(-1)[0]
      for k, v in recording.property(schema.STATE).items()
    }
    assert account["config"] == schema.state_layer(_SCALE, _F_ROW)
    assert account["f_row"] == _F_ROW
    assert int(account["converted_at"]) > 0
    assert (
      account["estimated_at"]
      == np.asarray(
        Recording(schema.layer_rrd(corpus, schema.ESTIMATE, "sess")).property(
          schema.ESTIMATE_PROPERTIES
        )["estimated_at"]
      ).reshape(-1)[0]
    )
    settings = recording.row(schema.STATE)
    assert np.asarray(settings["scale"]).reshape(-1)[0] == _SCALE
    assert np.asarray(settings["side"]).reshape(-1)[0] == "rear"
  with pytest.raises(SystemExit, match="--overwrite"):
    _state(corpus, corpus)


def test_the_state_is_the_base_tag_frame_on_a_dense_row_lattice(
  corpus: Path, tmp_path: Path
) -> None:
  """Both hands cover the same dense row range; each row is the estimate's
  trajectory composed with the tag's mounting at the row's own millisecond,
  between the two samples that bracket it."""
  out = _state(corpus, tmp_path, cut=Cut(0.0, 0.0))
  with Recording(out) as recording:
    for hand, slot, offset in (("left", "cam00", 0.0), ("right", "cam01", 1.0)):
      expected = _expected_state(offset, _RATE[slot])
      entity = schema.state_pose(hand)
      table = recording.series(f"/{entity}")
      rows = table.column(schema.ROW_TIMELINE).to_numpy()
      np.testing.assert_array_equal(rows, np.arange(0, 300))
      t, position = recording.values(entity, schema.POSE_TRANSLATION, 3)
      _, quaternion = recording.values(entity, schema.POSE_QUATERNION, 4)
      _, lin_vel = recording.scalars(schema.state_lin_vel(hand), 3)
      _, ang_vel = recording.scalars(schema.state_ang_vel(hand), 3)
      t_open, q = recording.scalars(schema.state_opening(hand), 1)
      want = [expected[int(k)] for k in rows]
      # A row's stamp is its own millisecond.
      np.testing.assert_allclose(t, rows / _F_ROW, atol=1e-9)
      # The estimate stores float32 rows, and the finite difference divides
      # their quantisation by the sample spacing.
      np.testing.assert_allclose(position, [w["position"] for w in want], atol=1e-6)
      np.testing.assert_allclose(quaternion, [w["quat"] for w in want], atol=1e-6)
      assert (quaternion[:, 3] >= 0.0).all()
      np.testing.assert_allclose(lin_vel, [w["lin_vel"] for w in want], atol=1e-5)
      np.testing.assert_allclose(ang_vel, [w["ang_vel"] for w in want], atol=1e-4)
      # The opening lands on every state row, interpolated from the solved
      # finger angles.
      np.testing.assert_allclose(t_open, t, atol=1e-9)
      np.testing.assert_allclose(q[:, 0], 0.1)


def test_a_frame_whose_capture_row_leaves_the_state_rows_goes_for_both_hands(
  corpus: Path, tmp_path: Path
) -> None:
  out = _state(corpus, tmp_path, cut=Cut(0.0985, 0.0))
  with Recording(out) as recording:
    for hand, slot in (("left", "cam00"), ("right", "cam01")):
      times, rows, captured, _ = _frames(recording, hand)
      np.testing.assert_allclose(
        times / 1e9, np.arange(4, 9) / _FPS + _SHIFT[slot], atol=1e-6
      )
      np.testing.assert_array_equal(rows, [133, 167, 200, 233, 267])
      assert captured.min() >= 99


def test_the_cut_and_the_left_override_apply(corpus: Path, tmp_path: Path) -> None:
  out = _state(corpus, tmp_path, cut=Cut(0.05, 0.1), left="cam01")
  with Recording(out) as recording:
    np.testing.assert_allclose(
      _frames(recording, "right")[0] / 1e9, np.arange(2, 8) / _FPS, atol=1e-6
    )
    assert recording.row(schema.state_hand("left"))["slot"] == ["cam01"]


def test_the_writer_refuses_what_it_cannot_honour(corpus: Path, tmp_path: Path) -> None:
  for message, kwargs in (
    ("does not divide", {"scale": 7}),
    ("--left must name", {"left": "cam99", "cut": Cut(0.0, 0.0)}),
    ("window", {"cut": Cut(0.0, 100.0)}),
  ):
    with pytest.raises(SystemExit, match=message):
      _state(corpus, tmp_path, **kwargs)
