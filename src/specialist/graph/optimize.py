"""Levenberg-Marquardt optimizer wrapper returning values plus run metadata."""

from __future__ import annotations

from dataclasses import dataclass

import gtsam


@dataclass(eq=False)
class OptimizationResult:
  """Optimized ``Values`` plus per-run diagnostics from the LM optimizer."""

  values: gtsam.Values
  iterations: int
  initial_error: float
  final_error: float


def optimize_graph(
  graph: gtsam.NonlinearFactorGraph,
  initial: gtsam.Values,
  params: gtsam.LevenbergMarquardtParams | None = None,
) -> OptimizationResult:
  """Run Levenberg-Marquardt to convergence and return the optimized values."""
  if params is None:
    params = gtsam.LevenbergMarquardtParams()
  optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
  initial_error = float(graph.error(initial))
  values = optimizer.optimize()
  return OptimizationResult(
    values=values,
    iterations=int(optimizer.iterations()),
    initial_error=initial_error,
    final_error=float(graph.error(values)),
  )
