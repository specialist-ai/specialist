"""Video ingest: SPS-normalizing remux and VideoStream chunk production."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from specialist.loader.audio_sync import resolve_ffmpeg
from specialist.record import schema
from specialist.record.video import normalize_h264, video_chunks

_FRAMES = 12
_RATE = 30
# Wide enough to keep every frame.
_WHOLE_CLIP = (-10.0, 10.0)


@pytest.fixture(scope="module")
def h264_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
  path = tmp_path_factory.mktemp("clip") / "clip.mp4"
  subprocess.run(
    [
      resolve_ffmpeg(),
      "-y",
      "-loglevel",
      "error",
      "-f",
      "lavfi",
      "-i",
      f"testsrc=duration={_FRAMES / _RATE}:size=64x64:rate={_RATE}",
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      str(path),
    ],
    check=True,
  )
  return path


def _collect(clip: Path, shift: float) -> tuple[list[object], list[int], list[bool]]:
  stream = video_chunks(clip, "image/cam/front", time_shift=shift, window=_WHOLE_CLIP)
  static, times, keyframes = [], [], []
  for chunk in stream.to_chunks():
    assert chunk.entity_path == "/image/cam/front"
    if chunk.is_static:
      static.append(chunk)
      continue
    batch = chunk.to_record_batch()
    index = batch.schema.get_field_index("video_time")
    times += [v.value for v in batch.column(index)]
    key_index = batch.schema.get_field_index("VideoStream:is_keyframe")
    keyframes += [bool(v.as_py()) for v in batch.column(key_index)]
  return static, times, keyframes


def test_normalize_and_chunks(tmp_path: Path, h264_clip: Path) -> None:
  clean = normalize_h264(h264_clip, tmp_path / "clean.mp4")
  static, times, keyframes = _collect(clean, shift=0.0)
  assert len(static) == 1
  assert len(times) == _FRAMES
  assert times == sorted(times)
  assert times[0] == 0
  frame_ns = round(1e9 / _RATE)
  assert abs(times[1] - frame_ns) <= 1
  assert keyframes[0] is True


def test_time_shift(tmp_path: Path, h264_clip: Path) -> None:
  clean = normalize_h264(h264_clip, tmp_path / "clean.mp4")
  _, unshifted, _ = _collect(clean, shift=0.0)
  _, shifted, _ = _collect(clean, shift=0.5)
  assert shifted == [t + 500_000_000 for t in unshifted]


def test_window_keeps_and_drops_chunks(tmp_path: Path, h264_clip: Path) -> None:
  clean = normalize_h264(h264_clip, tmp_path / "clean.mp4")

  def rows(window: tuple[float, float]) -> int:
    stream = video_chunks(clean, "image/cam/front", time_shift=0.0, window=window)
    return sum(c.num_rows for c in stream.to_chunks() if not c.is_static)

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
  chunks = video_chunks(
    h264_clip, "image/t/front", time_shift=0.0, window=(0.0, 10.0)
  ).to_chunks()
  rows: list[tuple[int, int]] = []
  for chunk in chunks:
    if chunk.is_static:
      continue
    batch = chunk.to_record_batch()
    names = batch.schema.names
    assert schema.KEYFRAME_TIMELINE in names, "keyframe index missing from chunk"
    t = batch.column(names.index(schema.TIMELINE))
    k = batch.column(names.index(schema.KEYFRAME_TIMELINE))
    rows.extend((t[i].value, k[i].as_py()) for i in range(batch.num_rows))
  rows.sort()
  ranks = np.arange(len(rows))
  stride = schema.keyframe_stride(
    schema.NS_PER_SECOND / float(np.median(np.diff([t for t, _ in rows])))
  )
  expected = schema.keyframe_index(ranks, stride)
  assert [k for _, k in rows] == expected.tolist()
  # The last row of bucket k is frame stride*k.
  buckets: dict[int, int] = {}
  for rank, (_, k) in enumerate(rows):
    buckets[int(k)] = rank
  assert all(rank == min(stride * k, len(rows) - 1) for k, rank in buckets.items())


def test_keyframe_timeline_buckets_on_the_session_clock(h264_clip: Path) -> None:
  """An audio-shifted capture buckets by where its frames fall on the shared
  session clock, not by its own file-relative frame count."""
  shift = 2.0 / _RATE  # this capture starts two frame periods into the session
  chunks = video_chunks(
    h264_clip, "image/t/front", time_shift=shift, window=(0.0, 10.0)
  ).to_chunks()
  rows: list[tuple[int, int]] = []
  for chunk in chunks:
    if chunk.is_static:
      continue
    batch = chunk.to_record_batch()
    names = batch.schema.names
    t = batch.column(names.index(schema.TIMELINE))
    k = batch.column(names.index(schema.KEYFRAME_TIMELINE))
    rows.extend((t[i].value, k[i].as_py()) for i in range(batch.num_rows))
  rows.sort()
  period = float(np.median(np.diff([t for t, _ in rows])))
  stride = schema.keyframe_stride(schema.NS_PER_SECOND / period)
  session_n = np.asarray([round(t / period) for t, _ in rows], dtype=np.int64)
  assert session_n[0] == 2, "the shift must appear in the session frame number"
  expected = schema.keyframe_index(session_n, stride)
  assert [k for _, k in rows] == expected.tolist()


@pytest.fixture(scope="module")
def bframe_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
  """A short-GOP encode with B-frames left on: the container holds one
  keyframe per GOP but the reader's markers collapse."""
  path = tmp_path_factory.mktemp("bframes") / "clip.mp4"
  subprocess.run(
    [
      resolve_ffmpeg(),
      "-y",
      "-loglevel",
      "error",
      "-f",
      "lavfi",
      "-i",
      f"testsrc=duration=1.0:size=64x64:rate={_RATE}",
      "-c:v",
      "libx264",
      "-g",
      "5",
      "-pix_fmt",
      "yuv420p",
      str(path),
    ],
    check=True,
  )
  return path


def test_dropped_keyframe_markers_fail_ingest(
  tmp_path: Path, bframe_clip: Path
) -> None:
  clean = normalize_h264(bframe_clip, tmp_path / "clean.mp4")
  with pytest.raises(RuntimeError, match="randomly accessible"):
    video_chunks(clean, "image/cam/front", time_shift=0.0, window=_WHOLE_CLIP)
