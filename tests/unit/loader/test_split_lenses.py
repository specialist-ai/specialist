"""Smoke test for ``split_dual_fisheye``.

Needs both an ``.insv`` fixture and an installed ``ffmpeg`` binary. Skips
cleanly if either is missing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from specialist.loader.split_lenses import split_dual_fisheye

_FIXTURE_DIR = Path.home() / "Downloads" / "probes"


def _smallest_insv_or_skip() -> Path:
  if shutil.which("ffmpeg") is None:
    pytest.skip("ffmpeg not on PATH")
  if not _FIXTURE_DIR.is_dir():
    pytest.skip(f"no fixtures at {_FIXTURE_DIR}")
  candidates = sorted(_FIXTURE_DIR.glob("*.insv"))
  if not candidates:
    pytest.skip(f"no .insv files in {_FIXTURE_DIR}")
  return min(candidates, key=lambda p: p.stat().st_size)


def test_split_dual_fisheye_writes_two_files(tmp_path: Path) -> None:
  src = _smallest_insv_or_skip()
  # Even the smallest capture is ~140 MB; this re-encodes only via ffmpeg's
  # input stream, so keep it as a real but minimal smoke test.
  front, rear = split_dual_fisheye(src, tmp_path, preset="ultrafast", crf=28)
  assert front.is_file() and front.stat().st_size > 0
  assert rear.is_file() and rear.stat().st_size > 0
  assert front != rear


def test_split_dual_fisheye_missing_input_raises(tmp_path: Path) -> None:
  with pytest.raises(FileNotFoundError):
    split_dual_fisheye("/tmp/nope.insv", tmp_path)
