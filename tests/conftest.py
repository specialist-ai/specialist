"""Tests solve on the host, with scipy standing in for the cuDSS factorization."""

from __future__ import annotations

from typing import Any

import pytest

from specialist.vi_ba import gpu_lm, optimize
from tests.solver_fixtures import ScipyFactorization


@pytest.fixture(autouse=True)
def _host_solver(monkeypatch: pytest.MonkeyPatch) -> None:
  def lm_solve(*args: Any, **kwargs: Any) -> Any:
    return gpu_lm.lm_solve(*args, factorization=ScipyFactorization(), **kwargs)

  monkeypatch.setattr(optimize, "lm_solve", lm_solve)
