"""Gripper opening from finger joint angles via URDF forward kinematics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from specialist.vi_ba.urdf_joints import RevoluteJoint

Float64Array = npt.NDArray[np.float64]

__all__ = ["opening_from_angles"]


def _tag_points(joint: RevoluteJoint, angles: Float64Array) -> Float64Array:
  """The finger-tag origin in the base frame at each angle, ``(n, 3)``."""
  if joint.joint_T_tag0 is None:
    raise ValueError("finger joint has no tag frame (joint_T_tag0 is None)")
  p = np.asarray(joint.joint_T_tag0.translation(), dtype=np.float64)
  c, s = np.cos(angles), np.sin(angles)
  rotated = np.stack(
    [c * p[0] - s * p[1], s * p[0] + c * p[1], np.full_like(angles, p[2])], axis=1
  )
  rotation = np.asarray(joint.base_T_joint.rotation().matrix(), dtype=np.float64)
  translation = np.asarray(joint.base_T_joint.translation(), dtype=np.float64)
  out: Float64Array = rotated @ rotation.T + translation
  return out


def opening_from_angles(
  joint_a: RevoluteJoint,
  joint_b: RevoluteJoint,
  angles_a: Float64Array,
  angles_b: Float64Array,
) -> Float64Array:
  """Finger-tag origin distance (m) per sample, NaN where either angle is NaN."""
  points_a = _tag_points(joint_a, np.asarray(angles_a, dtype=np.float64))
  points_b = _tag_points(joint_b, np.asarray(angles_b, dtype=np.float64))
  out: Float64Array = np.linalg.norm(points_a - points_b, axis=1)
  return out
