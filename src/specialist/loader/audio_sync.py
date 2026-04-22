"""Multi-clap audio synchronisation for paired ``.insv`` captures.

Demuxes mono PCM via ``ffmpeg``, builds a Butterworth-lowpassed rectified
envelope (the same recipe as mocap_prep), runs ``scipy.signal.find_peaks``
with a ``mean + 5σ`` height threshold + 0.3 s minimum spacing, and returns
the time offset that best aligns the two peak trains. The coarse lag
comes from a binary-impulse FFT cross-correlation; it's then refined to
sample resolution by the median pairwise difference of matched peaks, so
a slow envelope rise can't bias the offset.
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
from scipy.signal import butter, find_peaks, sosfiltfilt

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "ClapSyncResult",
  "decode_mono_pcm",
  "detect_clap_times",
  "sync_two_claps",
]


@dataclass(frozen=True)
class ClapSyncResult:
  """Audio-clap sync between two clips on a common ``t = 0`` clock.

  ``offset`` is added to clip-A's local timestamps to align them
  with clip-B's local clock (i.e. ``t_B = t_A + offset``).
  ``clap_times_a`` / ``clap_times_b`` are detected onset times in each
  clip's local clock for inspection / plotting.
  """

  offset: float
  clap_times_a: Float64Array
  clap_times_b: Float64Array
  matched_pairs: int


@functools.cache
def _resolve_ffmpeg() -> str:
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


def decode_mono_pcm(
  source: str | Path,
  target_rate: int = 16_000,
) -> tuple[Float64Array, int]:
  """Demux mono float32 PCM at ``target_rate`` from any container ffmpeg
  reads (``.insv``, ``.mp4``, ``.wav``). Returns ``(samples, rate)``."""
  ffmpeg = _resolve_ffmpeg()
  src = Path(source)
  if not src.is_file():
    raise FileNotFoundError(f"audio source not found: {src}")
  cmd = [
    ffmpeg,
    "-v",
    "error",
    "-i",
    str(src),
    "-ac",
    "1",
    "-ar",
    str(int(target_rate)),
    "-f",
    "f32le",
    "-",
  ]
  result = subprocess.run(cmd, capture_output=True, check=False)
  if result.returncode != 0:
    raise RuntimeError(f"ffmpeg failed for {src}:\n{result.stderr.decode()}")
  pcm = np.frombuffer(result.stdout, dtype=np.float32).astype(np.float64)
  return pcm, int(target_rate)


def detect_clap_times(
  samples: Float64Array,
  rate: int,
  envelope_cutoff: float = 50.0,
  threshold_sigmas: float = 4.0,
  min_separation: float = 0.3,
) -> Float64Array:
  """Find clap onset times (seconds) as peaks in a rectified-and-lowpassed
  envelope. Threshold is ``mean + threshold_sigmas × std``, peaks closer
  than ``min_separation`` collapse to the highest. Tuned for X3 captures
  (claps ~50 ms wide); mocap_prep uses 10 Hz / 5σ for a longer-attack
  whole-body clap impulse, but that smears X3 transients."""
  if samples.size == 0:
    return np.zeros(0, dtype=np.float64)
  sos = butter(4, envelope_cutoff, btype="low", fs=rate, output="sos")
  envelope = sosfiltfilt(sos, np.abs(samples))
  threshold = float(envelope.mean()) + float(threshold_sigmas) * float(envelope.std())
  min_distance = max(1, int(round(min_separation * rate)))
  peak_indices, _ = find_peaks(envelope, height=threshold, distance=min_distance)
  return np.asarray(peak_indices / float(rate), dtype=np.float64)


def _best_lag(
  times_a: Float64Array,
  times_b: Float64Array,
  max_search: float = 30.0,
  match_tolerance: float = 0.05,
) -> tuple[float, int]:
  """Lag (B − A) that the most clap pairs agree on. Returns
  ``(lag, agreeing_pair_count)``.

  Every ``(a, b)`` pair proposes an offset ``b - a``; the returned lag is the
  median of the largest cluster of proposals (those within
  ``match_tolerance``). Maximising the *count* of agreeing claps — rather than a
  binary-impulse cross-correlation — is robust to spurious claps and, critically,
  to periodic clap patterns: a cross-correlation's side-lobe one inter-clap
  spacing away can tie or beat the true peak and shift the lag by a whole clap,
  whereas the true offset always has the most pairs agreeing.
  """
  if times_a.size == 0 or times_b.size == 0:
    raise ValueError("at least one clap on each side is required")
  proposals = [
    float(b - a) for a in times_a for b in times_b if abs(float(b - a)) <= max_search
  ]
  if not proposals:
    raise ValueError("no clap pair within max_search")
  best_lag, best_count = 0.0, 0
  for center in proposals:
    cluster = [p for p in proposals if abs(p - center) <= match_tolerance]
    if len(cluster) > best_count:
      best_count, best_lag = len(cluster), float(np.median(cluster))
  return best_lag, best_count


def sync_two_claps(
  source_a: str | Path,
  source_b: str | Path,
  rate: int = 16_000,
  envelope_cutoff: float = 50.0,
  threshold_sigmas: float = 4.0,
  min_separation: float = 0.3,
  max_search: float = 30.0,
  match_tolerance: float = 0.1,
) -> ClapSyncResult:
  """End-to-end: demux both files, detect claps, return the lag that
  aligns clip A onto clip B's clock (``t_B = t_A + offset``)."""
  pcm_a, rate = decode_mono_pcm(source_a, rate)
  pcm_b, _ = decode_mono_pcm(source_b, rate)
  claps_a = detect_clap_times(
    pcm_a,
    rate,
    envelope_cutoff=envelope_cutoff,
    threshold_sigmas=threshold_sigmas,
    min_separation=min_separation,
  )
  claps_b = detect_clap_times(
    pcm_b,
    rate,
    envelope_cutoff=envelope_cutoff,
    threshold_sigmas=threshold_sigmas,
    min_separation=min_separation,
  )
  if claps_a.size == 0 or claps_b.size == 0:
    raise RuntimeError(
      f"no claps detected (a={claps_a.size}, b={claps_b.size}); "
      f"lower threshold_sigmas or check audio is present"
    )
  offset, matched = _best_lag(
    claps_a,
    claps_b,
    max_search=max_search,
    match_tolerance=match_tolerance,
  )
  return ClapSyncResult(
    offset=offset,
    clap_times_a=claps_a,
    clap_times_b=claps_b,
    matched_pairs=matched,
  )
