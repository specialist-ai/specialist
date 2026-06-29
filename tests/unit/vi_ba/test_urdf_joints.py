"""The URDF reader recovers the gripper's revolute finger joints (origin, limits)
and their AprilTag frames (joint_T_tag0) from the URDF tag links."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from specialist.vi_ba.urdf_joints import read_revolute_joints

_URDF = Path.home() / (
  "Projects/specialist_description/robots/specialist_gripper/urdf/gripper.urdf"
)


@pytest.mark.skipif(not _URDF.exists(), reason="gripper URDF not present")
def test_reads_finger_joints_and_tag_frames() -> None:
  joints = read_revolute_joints(_URDF)
  assert set(joints) == {"left_finger", "right_finger"}

  left = joints["left_finger"]
  np.testing.assert_allclose(
    left.base_T_joint.translation(), [0, 0.012, -0.008], atol=1e-3
  )
  assert left.lower < left.upper
  # The finger tag pose comes straight from the URDF tag link.
  assert left.joint_T_tag0 is not None
  np.testing.assert_allclose(
    left.joint_T_tag0.translation(), [0.04, 0.023, 0.0], atol=1e-3
  )

  right = joints["right_finger"]
  assert right.joint_T_tag0 is not None
  np.testing.assert_allclose(
    right.joint_T_tag0.translation(), [0.04, -0.023, 0.0], atol=1e-3
  )
