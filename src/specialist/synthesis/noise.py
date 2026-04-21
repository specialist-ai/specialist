"""SE(3) tangent-space Gaussian noise for synthetic pose measurements.

Noise is sampled in GTSAM's ``[ω, v]`` (rotation-first) tangent order and
applied via right-composition with ``Pose3.Expmap``, matching the
convention used by ``BetweenFactorPose3``.
"""

from __future__ import annotations

import gtsam
import numpy as np


def add_pose_noise(
  pose: gtsam.Pose3,
  translation_sigma: float,
  rotation_sigma: float,
  rng: np.random.Generator,
) -> gtsam.Pose3:
  """Right-compose ``pose`` with a tangent-space Gaussian perturbation."""
  tangent = np.concatenate(
    [
      rng.normal(0.0, rotation_sigma, size=3),
      rng.normal(0.0, translation_sigma, size=3),
    ]
  )
  return pose.compose(gtsam.Pose3.Expmap(tangent))
