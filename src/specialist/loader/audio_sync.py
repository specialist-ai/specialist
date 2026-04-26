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


def decode_mono_pcm(
  source: str | Path,
  target_rate: int = 16_000,
) -> tuple[Float64Array, int]:
  """Demux mono float32 PCM at ``target_rate`` from any container ffmpeg
  reads (``.insv``, ``.mp4``, ``.wav``). Returns ``(samples, rate)``."""
  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    raise RuntimeError("ffmpeg not found on PATH")
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
  bin_size: float = 0.01,
  max_search: float = 30.0,
  match_tolerance: float = 0.05,
) -> tuple[float, int]:
  """Lag (B − A) that maximises a binary-impulse cross-correlation, then
  refined to sample resolution by the median pairwise difference of
  matched peaks. Returns ``(lag, matched_pair_count)``.

  The coarse FFT step picks the right ballpark; the median-of-differences
  step removes envelope-rise jitter from each clap so the final lag is
  driven by a robust statistic over every matched pair.
  """
  if times_a.size == 0 or times_b.size == 0:
    raise ValueError("at least one clap on each side is required")
  duration = float(max(times_a.max(), times_b.max())) + max_search + 1.0
  n = max(2, int(np.ceil(duration / bin_size)))
  size = 1 << (int(np.ceil(np.log2(2 * n))))
  imp_a = np.zeros(size, dtype=np.float64)
  imp_b = np.zeros(size, dtype=np.float64)
  imp_a[np.clip((times_a / bin_size).astype(int), 0, n - 1)] = 1.0
  imp_b[np.clip((times_b / bin_size).astype(int), 0, n - 1)] = 1.0
  spec = np.fft.rfft(imp_b) * np.conj(np.fft.rfft(imp_a))
  corr = np.fft.irfft(spec, size)
  max_lag_bins = int(round(max_search / bin_size))
  pos = corr[: max_lag_bins + 1]
  neg = corr[-max_lag_bins:]
  best = int(np.argmax(np.concatenate([neg, pos])))
  coarse_lag = float(best - max_lag_bins) * bin_size

  # Refine: each A peak picks its closest B peak under the coarse lag, take
  # the median of (b - a) across those pairs. Robust to missed claps.
  diffs: list[float] = []
  for ta in times_a:
    candidates = times_b - (ta + coarse_lag)
    abs_min = np.argmin(np.abs(candidates))
    if abs(candidates[abs_min]) <= match_tolerance:
      diffs.append(float(times_b[abs_min] - ta))
  if not diffs:
    return coarse_lag, 0
  refined_lag = float(np.median(diffs))
  return refined_lag, len(diffs)


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
