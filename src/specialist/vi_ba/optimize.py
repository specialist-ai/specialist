"""Levenberg-Marquardt wrapper that returns both values and run metadata."""

from __future__ import annotations

import time
from dataclasses import dataclass

import gtsam
import numpy as np

from specialist.vi_ba.gpu_lm import lm_solve

__all__ = ["OptimizeResult", "optimize_graph"]

# Stop once an accepted step improves the error by less than this fraction.
_RELATIVE_ERROR_TOL = 1e-4


@dataclass(frozen=True)
class OptimizeResult:
  values: gtsam.Values
  iterations: int
  initial_error: float
  final_error: float
  # The damping of the last accepted step.
  final_lambda: float | None = None


def optimize_graph(
  graph: gtsam.NonlinearFactorGraph,
  initial_values: gtsam.Values,
  max_iterations: int,
  *,
  initial_lambda: float | None = None,
) -> OptimizeResult:
  """Schur LM solve of ``graph``; raise when the initial error is not finite."""
  initial_error = float(graph.error(initial_values))
  if not np.isfinite(initial_error):
    raise RuntimeError(
      "graph initial error is not finite: a factor evaluates to inf/NaN at "
      "the initial values, so the optimizer would silently return them"
    )
  begin = time.perf_counter()
  values, iterations, _, final_error, final_lambda = lm_solve(
    graph,
    initial_values,
    max_iterations,
    _RELATIVE_ERROR_TOL,
    initial_error=initial_error,
    initial_lambda=initial_lambda,
  )
  print(
    f"  [solve] {iterations} iters in {time.perf_counter() - begin:.1f}s, "
    f"error {initial_error:.0f} -> {final_error:.0f}"
  )
  return OptimizeResult(values, iterations, initial_error, final_error, final_lambda)
