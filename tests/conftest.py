"""Tests solve on the host: the cuDSS factorization is replaced by scipy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from specialist.vi_ba import gpu_lm, optimize
from tests.record_fixtures import camera_estimate, urdf_file, write_run
from tests.solver_fixtures import ScipyFactorization


@pytest.fixture(autouse=True)
def _host_solver(monkeypatch: pytest.MonkeyPatch) -> None:
  def lm_solve(*args: Any, **kwargs: Any) -> Any:
    return gpu_lm.lm_solve(*args, factorization=ScipyFactorization(), **kwargs)

  monkeypatch.setattr(optimize, "lm_solve", lm_solve)


@pytest.fixture(scope="session")
def estimated_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
  """The estimate and viz layers of one solved two-camera run."""
  tmp = tmp_path_factory.mktemp("solved")
  return write_run(
    tmp,
    "sess_route",
    [camera_estimate("cam00"), camera_estimate("cam01", offset=1.0)],
    urdf_path=urdf_file(tmp),
    frame_period=1.0 / 30.0,
  )
