"""Keyframe decoding out of a base recording's H.264 video stream."""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import Any

import datafusion
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from specialist_corpus import schema
from specialist_corpus.read import Recording

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

__all__ = ["LensVideo", "RecordingKeyframes", "decode_rows", "lens_times"]

_DECODE_THREADS = 16

_SAMPLE = "VideoStream:sample"
_KEYFRAME = "VideoStream:is_keyframe"


def _column(entity: str, component: str) -> datafusion.Expr:
  return datafusion.col(f'"{schema.selector(entity, component)}"')


def _index() -> datafusion.Expr:
  return datafusion.col(f'"{schema.TIMELINE}"')


def _samples(recording: Recording, entity: str) -> datafusion.DataFrame:
  """One lens entity's samples, with their keyframe markers."""
  reader = recording.reader(f"/{entity}")
  if schema.selector(entity, _SAMPLE) not in reader.schema().names:
    raise RuntimeError(f"{recording.path} has no video samples at /{entity}")
  samples: datafusion.DataFrame = reader.filter(_column(entity, _SAMPLE).is_not_null())
  return samples


def lens_times(recording: Recording, entity: str) -> Float64Array:
  """One lens entity's frame times (s), ascending, one per encoded sample."""
  table = _samples(recording, entity).select(_index()).to_arrow_table()
  return schema.seconds(table.column(schema.TIMELINE))


class LensVideo:
  """One lens's encoded samples and the timeline they sit on."""

  def __init__(self, recording: Recording, slot: str, side: str) -> None:
    self.entity = schema.video(slot, side)
    samples = _samples(recording, self.entity)
    table = samples.select(
      _index(),
      _column(self.entity, _SAMPLE),
      _column(self.entity, _KEYFRAME),
    ).to_arrow_table()
    times_ns = schema.times_ns(table.column(schema.TIMELINE))
    self.times: Float64Array = times_ns / schema.NS_PER_SECOND
    self.iframe_rows = _iframe_rows(
      table.column(schema.selector(self.entity, _KEYFRAME))
    )
    if not self.iframe_rows.size:
      raise RuntimeError(
        f"{self.entity}: no I-frame markers; every decode would start at row 0"
      )
    # Joining the chunks would copy the lens's whole video onto the heap.
    chunks = table.column(schema.selector(self.entity, _SAMPLE)).chunks
    self._chunk_first: Int64Array = np.cumsum(
      [0, *(len(chunk) for chunk in chunks)], dtype=np.int64
    )
    self._chunks = [_packet_arrays(chunk) for chunk in chunks]

  def packet(self, row: int) -> bytes:
    """The encoded sample at ``row``."""
    # An empty chunk repeats its neighbour's first row, which ``right`` skips.
    chunk = int(np.searchsorted(self._chunk_first, row, side="right")) - 1
    sample_offsets, packet_offsets, data = self._chunks[chunk]
    sample = int(sample_offsets[row - self._chunk_first[chunk]])
    lo = int(packet_offsets[sample])
    hi = int(packet_offsets[sample + 1])
    encoded: bytes = data[lo:hi].tobytes()
    return encoded

  def anchor(self, row: int) -> int:
    """The keyframe row a decode of ``row`` must start from."""
    position = int(np.searchsorted(self.iframe_rows, row, side="right")) - 1
    return int(self.iframe_rows[position]) if position >= 0 else 0


def _packet_arrays(
  chunk: pa.ListArray,
) -> tuple[Int64Array, Int64Array, npt.NDArray[np.uint8]]:
  """One chunk's ``(sample offsets, packet offsets, bytes)``."""
  packets = chunk.values
  return (
    chunk.offsets.to_numpy(),
    packets.offsets.to_numpy(),
    packets.values.to_numpy(zero_copy_only=True),
  )


def _iframe_rows(markers: pa.ChunkedArray) -> Int64Array:
  """Sample rows carrying a true ``is_keyframe`` marker."""
  flags = [
    bool(value[0]) if value else False for value in markers.combine_chunks().to_pylist()
  ]
  out: Int64Array = np.flatnonzero(flags).astype(np.int64)
  return out


def decode_rows(video: LensVideo, rows: Int64Array) -> Iterator[tuple[int, Any]]:
  """``(row, frame)`` for each of ``rows``, ascending, out of one pass over the
  packets from the first row's keyframe to the last row. Frame threading holds
  the tail until the context is drained, so the drain is part of the pass."""
  import av

  wanted = {int(r) for r in rows}
  first, last = video.anchor(int(rows[0])), int(rows[-1])
  context = av.CodecContext.create("h264", "r")
  context.thread_count = _DECODE_THREADS
  context.thread_type = "FRAME"
  packets = itertools.chain(
    (av.Packet(video.packet(source)) for source in range(first, last + 1)), [None]
  )
  emitted, found = first, 0
  packet = None
  try:
    for packet in packets:
      for decoded in context.decode(packet):
        if emitted in wanted:
          found += 1
          yield emitted, decoded
        emitted += 1
  finally:
    # Freeing an undrained threaded decoder blocks forever.
    if packet is not None:
      for _ in context.decode(None):
        pass
  if found != len(wanted):
    raise RuntimeError(
      f"{video.entity}: {len(wanted) - found} of {len(wanted)} rows never decoded, "
      f"from row {emitted}"
    )


def _luma(decoded: Any) -> np.ndarray:
  """The full-resolution luma plane of a 4:2:0 frame, taken without a
  swscale conversion."""
  plane = decoded.planes[0]
  rows = np.frombuffer(plane, dtype=np.uint8).reshape(-1, plane.line_size)
  return np.ascontiguousarray(rows[: int(decoded.height), : int(decoded.width)])


class RecordingKeyframes(Sequence[np.ndarray]):
  """A lens's frames at chosen rows out of a base recording, one band at a time.

  ``rows`` are ascending row indices into the lens's sample table. Indexing
  decodes the band the index falls in and keeps only that band.
  """

  def __init__(self, video: LensVideo, rows: Int64Array, band: int = 256) -> None:
    self._video = video
    self._rows = np.asarray(rows, dtype=np.int64)
    if self._rows.size and np.any(np.diff(self._rows) < 0):
      raise ValueError("rows must be ascending")
    self._band = band
    self._start = -1
    self._frames: list[np.ndarray] = []
    self._carry: tuple[int, np.ndarray] | None = None
    self._len = int(self._rows.size)

  def __len__(self) -> int:
    return self._len

  @property
  def band(self) -> int:
    """Frames resident per decode."""
    return self._band

  def _decode_band(self, first: int, span: int) -> list[np.ndarray]:
    band_rows = self._rows[first : first + span]
    frames = {
      row: _luma(decoded) for row, decoded in decode_rows(self._video, band_rows)
    }
    return [frames[int(r)] for r in band_rows]

  def __getitem__(self, index: int) -> np.ndarray:  # type: ignore[override]
    if index < 0:
      index += self._len
    if not 0 <= index < self._len:
      raise IndexError(index)
    if self._carry is not None and index == self._carry[0]:
      return self._carry[1]
    first = (index // self._band) * self._band
    if first != self._start:
      if self._frames:
        self._carry = (self._start + len(self._frames) - 1, self._frames[-1])
      self._frames = self._decode_band(first, min(self._band, self._len - first))
      self._start = first
    return self._frames[index - first]
