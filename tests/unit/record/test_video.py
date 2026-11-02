"""Video ingest: SPS-normalizing remux and VideoStream chunk production."""

from pathlib import Path

import numpy as np
import pytest
from rerun.chunk import Chunk
from specialist_corpus import schema

from specialist.record.video import normalize_h264, video_chunks
from tests.record_fixtures import encode_clip

_FRAMES = 12
_RATE = 30
# Wide enough to keep every frame.
_WHOLE_CLIP = (-10.0, 10.0)


@pytest.fixture(scope="module")
def h264_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
  return encode_clip(
    tmp_path_factory.mktemp("clip") / "clip.mp4", _FRAMES, rate=_RATE, bframes=True
  )


@pytest.fixture(scope="module")
def clean_clip(tmp_path_factory: pytest.TempPathFactory, h264_clip: Path) -> Path:
  return normalize_h264(h264_clip, tmp_path_factory.mktemp("clean") / "clean.mp4")


def _samples(chunk: Chunk) -> bool:
  """Whether the chunk holds frames; keyframe marks come in a chunk of their own."""
  return "VideoStream:sample" in chunk.to_record_batch().schema.names


def _collect(clip: Path, shift: float) -> tuple[list[object], list[int], list[bool]]:
  stream = video_chunks(clip, "image/cam/front", time_shift=shift, window=_WHOLE_CLIP)
  static, times, keyframes = [], [], []
  for chunk in stream.to_chunks():
    assert chunk.entity_path == "/image/cam/front"
    if chunk.is_static:
      static.append(chunk)
      continue
    batch = chunk.to_record_batch()
    names = batch.schema.names
    if "VideoStream:sample" in names:
      times += [v.value for v in batch.column(names.index("video_time"))]
    if "VideoStream:is_keyframe" in names:
      keyframes += [
        bool(v.as_py()) for v in batch.column(names.index("VideoStream:is_keyframe"))
      ]
  return static, times, keyframes


def _keyframe_rows(clip: Path, shift: float) -> list[tuple[int, int]]:
  """Every surviving video row as ``(session time, keyframe index)``, sorted."""
  rows: list[tuple[int, int]] = []
  for chunk in video_chunks(
    clip, "image/t/front", time_shift=shift, window=(0.0, 10.0)
  ).to_chunks():
    if not _samples(chunk):
      continue
    batch = chunk.to_record_batch()
    names = batch.schema.names
    assert schema.KEYFRAME_TIMELINE in names, "keyframe index missing from chunk"
    t = batch.column(names.index(schema.TIMELINE))
    k = batch.column(names.index(schema.KEYFRAME_TIMELINE))
    rows.extend((t[i].value, k[i].as_py()) for i in range(batch.num_rows))
  rows.sort()
  return rows


def _stride(rows: list[tuple[int, int]]) -> tuple[int, float]:
  period = float(np.median(np.diff([t for t, _ in rows])))
  return schema.keyframe_stride(schema.NS_PER_SECOND / period), period


def test_normalize_and_chunks(clean_clip: Path) -> None:
  static, times, keyframes = _collect(clean_clip, shift=0.0)
  assert len(static) == 1
  assert len(times) == _FRAMES
  assert times == sorted(times)
  assert times[0] == 0
  frame_ns = round(1e9 / _RATE)
  assert abs(times[1] - frame_ns) <= 1
  assert keyframes[0] is True


def test_time_shift(clean_clip: Path) -> None:
  _, unshifted, _ = _collect(clean_clip, shift=0.0)
  _, shifted, _ = _collect(clean_clip, shift=0.5)
  assert shifted == [t + 500_000_000 for t in unshifted]


def test_window_keeps_and_drops_chunks(clean_clip: Path) -> None:
  def rows(window: tuple[float, float]) -> int:
    stream = video_chunks(clean_clip, "image/cam/front", time_shift=0.0, window=window)
    return sum(c.num_rows for c in stream.to_chunks() if _samples(c))

  # A chunk overlapping the window survives whole at GOP granularity; one
  # entirely outside is dropped.
  assert rows((0.0, 10.0)) == _FRAMES
  assert rows((0.2, 10.0)) == _FRAMES
  assert rows((100.0, 200.0)) == 0


def test_remux_failure_raises(tmp_path: Path) -> None:
  bogus = tmp_path / "bogus.mp4"
  bogus.write_bytes(b"not a video")
  with pytest.raises(RuntimeError, match="ffmpeg remux"):
    normalize_h264(bogus, tmp_path / "out.mp4")


def test_keyframe_timeline_buckets_frames_to_the_stride(h264_clip: Path) -> None:
  """Every surviving video row carries the ``keyframe`` sequence index with
  ``ceil(row / stride)`` bucketing, so latest-at at keyframe k lands on frame
  ``stride * k``."""
  rows = _keyframe_rows(h264_clip, shift=0.0)
  stride, _period = _stride(rows)
  expected = schema.keyframe_index(np.arange(len(rows)), stride)
  assert [k for _, k in rows] == expected.tolist()
  # The last row of bucket k is frame stride*k.
  buckets: dict[int, int] = {}
  for rank, (_, k) in enumerate(rows):
    buckets[int(k)] = rank
  assert all(rank == min(stride * k, len(rows) - 1) for k, rank in buckets.items())


def test_keyframe_timeline_buckets_on_the_session_clock(h264_clip: Path) -> None:
  """An audio-shifted capture buckets by where its frames fall on the shared
  session clock, not by its own file-relative frame count."""
  # this capture starts two frame periods into the session
  rows = _keyframe_rows(h264_clip, shift=2.0 / _RATE)
  stride, period = _stride(rows)
  session_n = np.asarray([round(t / period) for t, _ in rows], dtype=np.int64)
  assert session_n[0] == 2, "the shift must appear in the session frame number"
  expected = schema.keyframe_index(session_n, stride)
  assert [k for _, k in rows] == expected.tolist()


def test_dropped_keyframe_markers_fail_ingest(tmp_path: Path) -> None:
  """A short-GOP encode with B-frames left on: the container holds one
  keyframe per GOP, but the reader transcodes it and marks only the first."""
  clip = encode_clip(tmp_path / "bframes.mp4", _RATE, rate=_RATE, gop=5, bframes=True)
  clean = normalize_h264(clip, tmp_path / "clean.mp4")
  with pytest.raises(RuntimeError, match="randomly accessible"):
    video_chunks(clean, "image/cam/front", time_shift=0.0, window=_WHOLE_CLIP)
