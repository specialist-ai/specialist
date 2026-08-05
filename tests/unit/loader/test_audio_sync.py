"""Tests for ffmpeg resolution in the audio-sync loader."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from specialist.loader.audio_sync import _best_lag, resolve_ffmpeg


def testresolve_ffmpeg_returns_runnable_binary(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The resolver returns an existing, executable binary with no system ffmpeg
  or PATH setup."""
  monkeypatch.delenv("FFMPEG", raising=False)
  resolve_ffmpeg.cache_clear()
  path = resolve_ffmpeg()
  assert Path(path).exists()
  assert os.access(path, os.X_OK)
  resolve_ffmpeg.cache_clear()


def testresolve_ffmpeg_honors_env_override(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """``$FFMPEG`` takes priority so a specific binary can be forced."""
  resolve_ffmpeg.cache_clear()
  monkeypatch.setenv("FFMPEG", "/custom/path/ffmpeg")
  assert resolve_ffmpeg() == "/custom/path/ffmpeg"
  resolve_ffmpeg.cache_clear()


def test_best_lag_recovers_known_offset() -> None:
  """A clean, non-periodic clap train shifted by a known offset returns it."""
  a = np.array([1.0, 4.0, 9.5])
  lag, n = _best_lag(a, a + 0.75, match_tolerance=0.05)
  assert abs(lag - 0.75) < 1e-6
  assert n == 3


def test_best_lag_periodic_claps_picks_consensus_not_sidelobe() -> None:
  """Periodic claps (a triple ~0.32 s apart) plus a spurious clap: the lag is
  the offset the most pairs agree on, not a cross-correlation side-lobe one
  inter-clap spacing away. These two triples coincide (true offset ~0), so the
  off-by-one match at -0.32 (only two agreeing pairs) must not win over the three
  pairs at ~0. This is the real failure that misaligned the two-gripper clips."""
  a = np.array([5.636, 15.807, 16.134, 16.468])
  b = np.array([4.502, 15.815, 16.146, 16.451])
  lag, n = _best_lag(a, b, match_tolerance=0.1)
  assert abs(lag) < 0.05, f"expected ~0 offset, got {lag}"
  assert n >= 3
