"""Cramér-Rao lower bounds from the linearized information matrix at ground truth.

GTSAM's ``Marginals`` already does the hard work: given a graph and values,
it returns the marginal covariance at each variable by inverting the
relevant block of the information matrix. Evaluating at the **ground-truth**
values (rather than the optimized estimate) gives the covariance a
maximum-likelihood estimator would approach in the large-sample limit —
that is, the CRB.

Each pose marginal is a 6×6 matrix in GTSAM's ``[ω, v]`` tangent order. We
aggregate it into a scalar expected-RMSE per trajectory by taking the
square root of the mean trace over keyframes — the rotation and translation
blocks are handled separately since their units differ. Integration tests
then compare the observed ATE against the CRB with a tolerance factor
(≤1.5× by spec) that accounts for noise realization and finite-sample
effects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import gtsam
import numpy as np

from specialist.eval.metrics import TrajectoryError


def crb_trajectory_bounds(
  graph: gtsam.NonlinearFactorGraph,
  values_at_ground_truth: gtsam.Values,
  keys_per_tracker: Mapping[str, Sequence[int]],
) -> dict[str, TrajectoryError]:
  """CRB on translation and rotation RMSE per tracker, linearized at ground truth."""
  marginals = gtsam.Marginals(graph, values_at_ground_truth)
  result: dict[str, TrajectoryError] = {}
  for name, keys in keys_per_tracker.items():
    if not keys:
      raise ValueError(f"tracker {name!r} has no keys")
    rot_traces: list[float] = []
    trans_traces: list[float] = []
    for k in keys:
      cov = np.asarray(marginals.marginalCovariance(k), dtype=np.float64)
      rot_traces.append(float(np.trace(cov[:3, :3])))
      trans_traces.append(float(np.trace(cov[3:, 3:])))
    trans_rmse = float(np.sqrt(np.mean(trans_traces)))
    rot_rmse = float(np.sqrt(np.mean(rot_traces)))
    result[name] = TrajectoryError(
      translation_rmse=trans_rmse,
      rotation_rmse=rot_rmse,
    )
  return result
