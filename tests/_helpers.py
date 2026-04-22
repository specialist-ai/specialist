"""Shared helpers for unit tests."""

from __future__ import annotations

import gtsam


def pose(
  xyz: tuple[float, float, float], rpy: tuple[float, float, float]
) -> gtsam.Pose3:
  """Build a ``gtsam.Pose3`` from ``(x, y, z)`` and ``(roll, pitch, yaw)`` in radians."""
  return gtsam.Pose3(gtsam.Rot3.RzRyRx(*rpy), gtsam.Point3(*xyz))
