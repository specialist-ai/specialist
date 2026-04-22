"""Smoke tests for ``parse_insv`` that skip without a real ``.insv`` capture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from specialist.loader.parse_insv import InsvData, load_insv

_FIXTURE_DIR = Path.home() / "Downloads" / "probes"


def _fixture_or_skip() -> Path:
  if not _FIXTURE_DIR.is_dir():
    pytest.skip(f"no .insv fixtures at {_FIXTURE_DIR}")
  candidates = sorted(_FIXTURE_DIR.glob("*yaw_flick*.insv")) or sorted(
    _FIXTURE_DIR.glob("*.insv")
  )
  if not candidates:
    pytest.skip(f"no .insv files in {_FIXTURE_DIR}")
  # Prefer the smallest file so the smoke test stays fast.
  return min(candidates, key=lambda p: p.stat().st_size)


def test_load_insv_returns_imu_arrays() -> None:
  data = load_insv(_fixture_or_skip())
  assert isinstance(data, InsvData)
  n = data.imu_timestamps.shape[0]
  assert n > 100
  assert data.gyro_deg.shape == (n, 3)
  assert data.accel.shape == (n, 3)
  assert np.all(np.diff(data.imu_timestamps) >= 0)


def test_load_insv_intrinsics_are_x3_shape() -> None:
  data = load_insv(_fixture_or_skip())
  assert data.intrinsics.camera_matrix.shape == (3, 3)
  assert data.intrinsics.distortion_coeffs.shape == (6,)
  # Fisheye lenses give xi just under 2 (super-fisheye regime).
  assert 1.5 < data.intrinsics.xi < 2.5
  assert data.intrinsics.distortion_model == "insta360"


def test_load_insv_rolling_shutter_within_factory_spec() -> None:
  data = load_insv(_fixture_or_skip())
  # Factory value is 15.727 ms; allow ±0.5 ms slack across firmware versions.
  assert abs(data.rolling_shutter_readout - 15.727e-3) < 0.5e-3


def test_load_insv_missing_file_raises() -> None:
  with pytest.raises(FileNotFoundError):
    load_insv("/tmp/this/does/not/exist.insv")
