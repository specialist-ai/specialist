"""Tests for the joint graph's single gauge datum."""

from __future__ import annotations

import gtsam
import numpy as np

from specialist.vi_ba.compose import anchor_datum_factor


def _seed() -> gtsam.Pose3:
  """A wall tag: its own z points horizontally, into the room."""
  return gtsam.Pose3(
    gtsam.Rot3.Rodrigues(np.pi / 2, 0.0, 0.0), gtsam.Point3(0.4, 1.2, 0.9)
  )


def _turned(pose: gtsam.Pose3, axis: np.ndarray, angle: float) -> gtsam.Pose3:
  """``pose`` tilted in place about a world axis."""
  world = gtsam.Rot3.Rodrigues(*(axis * angle))
  return gtsam.Pose3(world.compose(pose.rotation()), pose.translation())


def _cost(pose: gtsam.Pose3) -> float:
  seed = _seed()
  values = gtsam.Values()
  values.insert(0, pose)
  return float(anchor_datum_factor(0, seed).error(values))


def test_datum_is_free_about_the_horizontal_axes() -> None:
  """The datum leaves roll and pitch free."""
  seed = _seed()
  angle = np.deg2rad(2.0)
  for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
    assert _cost(_turned(seed, axis, angle)) < 1e-6


def test_datum_pins_yaw_and_translation() -> None:
  seed = _seed()
  yawed = _turned(seed, np.array([0.0, 0.0, 1.0]), np.deg2rad(0.1))
  shifted = gtsam.Pose3(
    seed.rotation(), seed.translation() + np.array([1e-3, 0.0, 0.0])
  )
  assert _cost(seed) < 1e-9
  assert _cost(yawed) > 1.0
  assert _cost(shifted) > 1.0


def test_datum_leaves_four_directions_constrained() -> None:
  """The datum constrains four of the six tangent directions."""
  values = gtsam.Values()
  values.insert(0, _seed())
  jacobian = anchor_datum_factor(0, _seed()).linearize(values).jacobian()[0]
  singular = np.linalg.svd(jacobian, compute_uv=False)
  assert int(np.sum(singular > 1e-3 * singular.max())) == 4
