"""Trajectory metrics: ATE and RPE on the SE(3) tangent space.

All errors are computed on the SE(3) tangent space: the translation part of
``Logmap(gt.between(est))`` gives the translation error (metres) and the
rotation part gives the rotation error (radians). This matches GTSAM's
internal error convention and avoids conflating gimbal-lock-prone
Euler-angle residuals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import gtsam
import numpy as np


@dataclass(eq=False)
class TrajectoryError:
  """Aggregated translation and rotation RMSEs for a sequence of pose errors."""

  translation_rmse: float
  rotation_rmse: float


def _pose_error_tangent(
  estimated: gtsam.Pose3, ground_truth: gtsam.Pose3
) -> np.ndarray:
  """Tangent-space error [ω, v] mapping ``estimated`` to ``ground_truth``."""
  delta = ground_truth.between(estimated)
  return np.asarray(gtsam.Pose3.Logmap(delta), dtype=np.float64)


def _rmse_from_tangents(tangents: Sequence[np.ndarray]) -> TrajectoryError:
  if not tangents:
    return TrajectoryError(translation_rmse=0.0, rotation_rmse=0.0)
  tangent_array = np.asarray(tangents)
  rot_norms = np.linalg.norm(tangent_array[:, :3], axis=1)
  trans_norms = np.linalg.norm(tangent_array[:, 3:], axis=1)
  return TrajectoryError(
    translation_rmse=float(np.sqrt(np.mean(trans_norms**2))),
    rotation_rmse=float(np.sqrt(np.mean(rot_norms**2))),
  )


def absolute_trajectory_error(
  estimated: Sequence[gtsam.Pose3], ground_truth: Sequence[gtsam.Pose3]
) -> TrajectoryError:
  """Translation and rotation RMSE of an estimated trajectory against ground truth."""
  if len(estimated) != len(ground_truth):
    raise ValueError(
      f"trajectory length mismatch: {len(estimated)} vs {len(ground_truth)}"
    )
  tangents = [
    _pose_error_tangent(e, g) for e, g in zip(estimated, ground_truth, strict=True)
  ]
  return _rmse_from_tangents(tangents)


def relative_pose_error(
  estimated_a: Sequence[gtsam.Pose3],
  estimated_b: Sequence[gtsam.Pose3],
  ground_truth_a: Sequence[gtsam.Pose3],
  ground_truth_b: Sequence[gtsam.Pose3],
) -> TrajectoryError:
  """RMSE of the estimated A→B relative pose against the GT A→B relative pose.

  Invariant to any common rigid transform applied to both estimates — this is
  the metric that matters when the anchor gauge fix is unreliable but
  inter-tracker geometry is still accurately constrained.
  """
  if not (
    len(estimated_a) == len(estimated_b) == len(ground_truth_a) == len(ground_truth_b)
  ):
    raise ValueError("all four trajectories must have the same length")
  tangents = []
  for ea, eb, ga, gb in zip(
    estimated_a, estimated_b, ground_truth_a, ground_truth_b, strict=True
  ):
    est_rel = ea.between(eb)
    gt_rel = ga.between(gb)
    tangents.append(
      np.asarray(gtsam.Pose3.Logmap(gt_rel.between(est_rel)), dtype=np.float64)
    )
  return _rmse_from_tangents(tangents)
