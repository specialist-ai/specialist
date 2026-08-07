"""Audio synchronisation for paired ``.insv`` captures.

Demuxes mono PCM via ``ffmpeg``, builds a rectified and lowpassed envelope of
each clip, and takes the offset from a cross correlation of the two.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfiltfilt

Float64Array = npt.NDArray[np.float64]

__all__ = ["AudioSyncResult", "correlation_offset", "resolve_ffmpeg", "sync_two_clips"]

_DEMUX_RATE = 16_000
_ENVELOPE_CUTOFF = 50.0
# The correlation runs at ``rate / _DECIMATION``.
_DECIMATION = 16
# Largest offset considered, in seconds.
_MAX_OFFSET = 30.0
# Half-width (s) of the correlation peak's own lobe.
_LOBE_HALF_WIDTH = 0.25
# Multiple of that rival the peak must reach before the offset is trusted.
_MIN_PEAK_RATIO = 1.3


@dataclass(frozen=True)
class AudioSyncResult:
  """Audio alignment between two clips: ``t_B = t_A + offset``.

  ``peak_ratio`` is the correlation peak over the tallest rival outside its
  lobe.
  """

  offset: float
  peak_ratio: float


@functools.cache
def resolve_ffmpeg() -> str:
  """Locate an ffmpeg binary. Priority: the ``$FFMPEG`` override, then the
  ``imageio-ffmpeg`` wheel-bundled static binary (a declared dependency, so
  present on every platform after ``uv sync`` with no system install), then
  ``ffmpeg`` on ``PATH``. Raises if none resolve."""
  override = os.environ.get("FFMPEG")
  if override:
    return override
  try:
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())
  except Exception:  # noqa: BLE001 - any import/resolution failure falls through
    pass
  on_path = shutil.which("ffmpeg")
  if on_path is not None:
    return on_path
  raise RuntimeError(
    "ffmpeg not found: it ships via the imageio-ffmpeg dependency "
    "(run `uv sync`); or set $FFMPEG; or put ffmpeg on PATH"
  )


def _decode_mono_pcm(source: str | Path) -> Float64Array:
  """Demux mono float32 PCM at ``_DEMUX_RATE`` from any container ffmpeg reads
  (``.insv``, ``.mp4``, ``.wav``)."""
  src = Path(source)
  if not src.is_file():
    raise FileNotFoundError(f"audio source not found: {src}")
  cmd = [
    resolve_ffmpeg(),
    "-v",
    "error",
    "-i",
    str(src),
    "-ac",
    "1",
    "-ar",
    str(_DEMUX_RATE),
    "-f",
    "f32le",
    "-",
  ]
  result = subprocess.run(cmd, capture_output=True, check=False)
  if result.returncode != 0:
    raise RuntimeError(f"ffmpeg failed for {src}:\n{result.stderr.decode()}")
  pcm: Float64Array = np.frombuffer(result.stdout, dtype=np.float32).astype(np.float64)
  return pcm


def _envelope(samples: Float64Array, rate: int) -> Float64Array:
  """Rectified, lowpassed amplitude envelope."""
  sos = butter(4, _ENVELOPE_CUTOFF, btype="low", fs=rate, output="sos")
  envelope: Float64Array = sosfiltfilt(sos, np.abs(samples))
  return envelope


def correlation_offset(
  env_a: Float64Array, env_b: Float64Array, rate: int
) -> tuple[float, float]:
  """Envelope offset ``(offset, peak_ratio)`` by cross correlation, with
  ``t_B = t_A + offset``.

  A parabolic fit around the winning bin resolves finer than one bin.
  ``peak_ratio`` compares the winner against the tallest rival outside its main
  lobe, so a repeating transient pattern scores near 1."""
  a = np.asarray(env_a[::_DECIMATION], dtype=np.float64)
  b = np.asarray(env_b[::_DECIMATION], dtype=np.float64)
  if a.size < 2 or b.size < 2:
    raise ValueError("envelopes are too short to correlate")
  decimated_rate = rate / _DECIMATION
  a = a - a.mean()
  b = b - b.mean()
  # Zero padding biases the winner toward small offsets; clips run tens of
  # seconds against offsets of a few.
  size = 1 << int(np.ceil(np.log2(a.size + b.size)))
  cross = np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size))
  correlation = np.fft.fftshift(np.fft.irfft(cross, size))
  offsets = (np.arange(size) - size // 2) / decimated_rate
  candidates = np.where(np.abs(offsets) <= _MAX_OFFSET, correlation, -np.inf)
  peak = int(np.argmax(candidates))
  height = float(candidates[peak])
  shift = 0.0
  if 0 < peak < size - 1:
    left, right = float(correlation[peak - 1]), float(correlation[peak + 1])
    denominator = left - 2.0 * height + right
    if denominator != 0.0:
      shift = 0.5 * (left - right) / denominator
  lobe = np.abs(offsets - offsets[peak]) <= _LOBE_HALF_WIDTH
  best_rival = float(np.where(lobe, -np.inf, candidates).max())
  ratio = height / best_rival if best_rival > 0.0 else float("inf")
  return float(offsets[peak] + shift / decimated_rate), ratio


def sync_two_clips(
  source_a: str | Path,
  source_b: str | Path,
) -> AudioSyncResult:
  """The offset that aligns clip A onto clip B's clock: ``t_B = t_A + offset``."""
  env_a = _envelope(_decode_mono_pcm(source_a), _DEMUX_RATE)
  env_b = _envelope(_decode_mono_pcm(source_b), _DEMUX_RATE)
  offset, peak_ratio = correlation_offset(env_a, env_b, _DEMUX_RATE)
  if peak_ratio < _MIN_PEAK_RATIO:
    raise RuntimeError(
      f"audio alignment is ambiguous: the correlation peak is only "
      f"{peak_ratio:.2f} times its tallest rival, so the clips may share no "
      f"audible event; pass the offset explicitly if it is known"
    )
  return AudioSyncResult(offset=offset, peak_ratio=peak_ratio)
