"""Synthetic mock-VIO odometry: noisy relative SE(3) measurements per tracker.

Keyframes are sampled uniformly from a smooth trajectory at ``rate``; for
each consecutive pair we compute the ground-truth relative pose and add white
noise in the SE(3) tangent space using GTSAM's ``[ω, v]`` (rotation-first)
convention. This is a stand-in for real VIO output; the real-data loader
replaces these measurements with values from an actual VIO system.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import gtsam
import numpy as np

from specialist.synthesis.noise import add_pose_noise
from specialist.synthesis.trajectory import SmoothTrajectory


@dataclass(eq=False)
class OdometryMeasurement:
  """Noisy relative SE(3) pose between two consecutive keyframes of one tracker."""

  tracker_name: str
  from_index: int
  to_index: int
  relative_pose: gtsam.Pose3


def sample_keyframes(trajectory: SmoothTrajectory, rate: float) -> list[gtsam.Pose3]:
  """Sample ``trajectory`` at ``rate``, inclusive of both endpoints."""
  if rate <= 0.0:
    raise ValueError(f"rate must be positive, got {rate}")
  n_intervals = int(round(trajectory.duration * rate))
  if n_intervals < 1:
    raise ValueError(
      f"trajectory too short for rate={rate} (duration={trajectory.duration})"
    )
  ts = np.linspace(0.0, trajectory.duration, n_intervals + 1)
  return [trajectory.pose(float(t)) for t in ts]


def synthesize_odometry(
  keyframe_poses: Sequence[gtsam.Pose3],
  tracker_name: str,
  translation_sigma: float,
  rotation_sigma: float,
  rng: np.random.Generator,
) -> list[OdometryMeasurement]:
  """Emit one ``OdometryMeasurement`` per consecutive keyframe pair with white noise."""
  if translation_sigma < 0 or rotation_sigma < 0:
    raise ValueError("noise sigmas must be non-negative")
  if len(keyframe_poses) < 2:
    raise ValueError("need at least two keyframes to synthesize odometry")

  measurements: list[OdometryMeasurement] = []
  for i in range(len(keyframe_poses) - 1):
    ground_truth = keyframe_poses[i].between(keyframe_poses[i + 1])
    noisy = add_pose_noise(ground_truth, translation_sigma, rotation_sigma, rng)
    measurements.append(
      OdometryMeasurement(
        tracker_name=tracker_name,
        from_index=i,
        to_index=i + 1,
        relative_pose=noisy,
      )
    )
  return measurements
