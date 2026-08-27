"""Import coverage for the CLI drivers, which live outside the package and no
other test imports."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

_DRIVERS = ("estimate", "convert", "state", "register", "imu", "video_imu_timing")


@pytest.mark.parametrize("name", _DRIVERS)
def test_every_driver_imports(name: str) -> None:
  module = importlib.import_module(f"scripts.{name}")
  assert callable(module.main)


@pytest.mark.parametrize("name", _DRIVERS)
def test_every_driver_runs_by_path(name: str) -> None:
  """The README's ``python scripts/x.py`` form runs, which puts scripts/ itself
  first on the path."""
  result = subprocess.run(
    [sys.executable, f"scripts/{name}.py", "--help"],
    capture_output=True,
    cwd=Path(__file__).resolve().parents[2],
  )
  assert result.returncode == 0, result.stderr.decode()[-500:]
