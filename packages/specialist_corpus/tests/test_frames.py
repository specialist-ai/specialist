"""Keyframe decoding out of a base recording's video stream."""

from pathlib import Path

import pytest
import rerun as rr
from rerun.experimental import Chunk, send_chunks
from specialist_corpus import frames, schema
from specialist_corpus.read import Recording


def _base_with_video(tmp_path: Path, *, keyframes: bool) -> Path:
  """A base layer holding one lens's samples, with or without keyframe
  markers."""
  rrd = tmp_path / "base.rrd"
  samples = [b"\x00\x01", b"\x02\x03", b"\x04\x05"]
  chunk = Chunk.from_columns(
    schema.video("cam00", "front"),
    indexes=[rr.TimeColumn(schema.TIMELINE, duration=[0.0, 0.1, 0.2])],
    columns=rr.VideoStream.columns(
      sample=samples, is_keyframe=[keyframes, False, False]
    ),
  )
  with rr.RecordingStream("test", recording_id="s") as stream:
    stream.save(str(rrd))
    send_chunks([chunk], recording=stream)
  return rrd


def test_a_lens_reads_its_sample_times(tmp_path: Path) -> None:
  recording = Recording(_base_with_video(tmp_path, keyframes=True))
  times = frames.lens_times(recording, schema.video("cam00", "front"))
  assert list(times) == pytest.approx([0.0, 0.1, 0.2])


def test_a_lens_anchors_a_decode_on_its_keyframe(tmp_path: Path) -> None:
  recording = Recording(_base_with_video(tmp_path, keyframes=True))
  video = frames.LensVideo(recording, "cam00", "front")
  assert list(video.iframe_rows) == [0]
  assert video.anchor(2) == 0
  assert video.packet(1) == b"\x02\x03"


def test_lens_video_refuses_markerless_streams(tmp_path: Path) -> None:
  rrd = _base_with_video(tmp_path, keyframes=False)
  with pytest.raises(RuntimeError, match="no I-frame markers"):
    frames.LensVideo(Recording(rrd), "cam00", "front")
