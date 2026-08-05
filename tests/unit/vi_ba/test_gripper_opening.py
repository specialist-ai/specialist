"""Gripper opening from finger angles via URDF forward kinematics."""

import gtsam
import numpy as np
import pytest

from specialist.vi_ba.gripper_opening import opening_from_angles
from specialist.vi_ba.urdf_joints import RevoluteJoint

_D = 0.03
_L = 0.05


def _finger(side: float) -> RevoluteJoint:
  return RevoluteJoint(
    base_T_joint=gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, side * _D, 0.0)),
    lower=-0.5,
    upper=0.5,
    joint_T_tag0=gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(_L, 0.0, 0.0)),
  )


def test_closed_opening_is_base_separation() -> None:
  opening = opening_from_angles(
    _finger(+1.0), _finger(-1.0), np.array([0.0]), np.array([0.0])
  )
  np.testing.assert_allclose(opening, [2 * _D])


def test_equal_and_opposite_angles_open_monotonically() -> None:
  thetas = np.array([0.0, 0.1, 0.2, 0.3])
  opening = opening_from_angles(_finger(+1.0), _finger(-1.0), thetas, -thetas)
  expected = 2 * (_D + _L * np.sin(thetas))
  np.testing.assert_allclose(opening, expected, atol=1e-12)
  assert np.all(np.diff(opening) > 0)


def test_nan_angles_propagate() -> None:
  opening = opening_from_angles(
    _finger(+1.0), _finger(-1.0), np.array([0.0, np.nan]), np.array([0.0, 0.1])
  )
  assert np.isfinite(opening[0]) and np.isnan(opening[1])


def test_missing_tag_frame_rejected() -> None:
  bad = RevoluteJoint(
    base_T_joint=gtsam.Pose3(), lower=0.0, upper=1.0, joint_T_tag0=None
  )
  with pytest.raises(ValueError, match="tag"):
    opening_from_angles(bad, _finger(-1.0), np.array([0.0]), np.array([0.0]))
