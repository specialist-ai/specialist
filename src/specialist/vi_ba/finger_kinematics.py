"""Kinematic inputs for the finger-tag corner-reprojection factors.

Each finger AprilTag rides a 1-DoF revolute joint off the base tag (URDF
``left_finger`` / ``right_finger``, axis +z). Its tag pose at joint angle
``theta`` is

    body_T_tag(theta) = body_T_base_tag . base_T_joint . Rz(theta) . joint_T_tag0

with ``base_T_joint`` and ``joint_T_tag0`` read from the URDF and
``body_T_base_tag`` a variable the bundle adjustment solves. The BA recovers
``theta`` from the four reprojected tag corners; the initial ``theta`` comes
from a 1-D search over the joint's range that needs no camera pose, the chain
being body-fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import mei_project
from specialist.vi_ba.anchor_detection import (
  TagDetection,
  tag_center_bearing,
  tag_corners_in_frame,
)
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.urdf_joints import RevoluteJoint

Float64Array = npt.NDArray[np.float64]

__all__ = ["FingerKfObs", "FingerReproj", "build_finger_reproj"]

_THETA_SCAN_SAMPLES = 48


@dataclass(frozen=True)
class FingerKfObs:
  """One keyframe's finger-tag corner detection, ready for a reprojection factor."""

  theta_seed: float
  pixels: Float64Array  # (4, 2) detected corners, order [BL, BR, TR, TL]
  obs_bearing: Float64Array  # (3,) tag direction in the lens frame (cheirality)


@dataclass(frozen=True)
class FingerReproj:
  """Everything the joint graph needs to constrain one finger's joint angle."""

  base_T_joint: gtsam.Pose3  # URDF joint origin in the base frame; its z is the axis
  corners_in_pivot: Float64Array  # (4, 3) tag corners in the joint frame at theta=0
  observations: dict[int, FingerKfObs]


def _seed_theta(
  pixels: Float64Array,
  corners_in_pivot: Float64Array,
  body_T_joint: gtsam.Pose3,
  lens: LensSetup,
  lower: float,
  upper: float,
) -> float:
  """Joint angle that best reprojects the four corners, by a 1-D scan over
  ``[lower, upper]``."""
  lens_T_body = lens.extrinsic.inverse()
  r_lb = lens_T_body.rotation().matrix()
  t_lb = np.asarray(lens_T_body.translation(), dtype=np.float64)
  r_bj = body_T_joint.rotation().matrix()
  t_bj = np.asarray(body_T_joint.translation(), dtype=np.float64)
  cx, cy, cz = corners_in_pivot.T
  thetas = np.linspace(lower, upper, _THETA_SCAN_SAMPLES)
  c = np.cos(thetas)[:, None]
  s = np.sin(thetas)[:, None]
  rz = np.stack(  # Rz(theta) . c
    [c * cx - s * cy, s * cx + c * cy, np.broadcast_to(cz, c.shape[:1] + cz.shape)],
    axis=-1,
  )
  p_lens = (rz @ r_bj.T + t_bj) @ r_lb.T + t_lb
  projected = mei_project(p_lens.reshape(-1, 3), lens.params).reshape(
    p_lens.shape[:2] + (2,)
  )
  errors = np.sum((projected - pixels) ** 2, axis=(1, 2))
  # A behind-the-mirror corner projects to NaN; an all-NaN scan falls back to
  # the lower bound.
  return float(thetas[int(np.argmin(np.where(np.isnan(errors), np.inf, errors)))])


def build_finger_reproj(
  joint: RevoluteJoint,
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  finger_tag_id: int,
  body_T_base_tag: gtsam.Pose3,
  tag_size: float,
  n_keyframes: int,
) -> FingerReproj | None:
  """Assemble one finger's reprojection inputs from the URDF.
  ``body_T_base_tag`` seeds each keyframe's joint angle. Returns ``None`` if the
  finger tag is never detected; ``joint`` must carry an AprilTag frame
  (``joint_T_tag0``)."""
  assert joint.joint_T_tag0 is not None
  base_T_joint = joint.base_T_joint
  body_T_joint_seed = body_T_base_tag.compose(base_T_joint)
  corners_in_pivot = tag_corners_in_frame(joint.joint_T_tag0, tag_size)
  observations: dict[int, FingerKfObs] = {}
  for kf in range(min(n_keyframes, len(tags_per_kf))):
    det = next((d for d in tags_per_kf[kf] if d.tag_id == finger_tag_id), None)
    if det is None:
      continue
    pixels = np.asarray(det.corners, dtype=np.float64)
    bearing = tag_center_bearing(det, lens.params)
    theta = _seed_theta(
      pixels, corners_in_pivot, body_T_joint_seed, lens, joint.lower, joint.upper
    )
    observations[kf] = FingerKfObs(theta_seed=theta, pixels=pixels, obs_bearing=bearing)
  if not observations:
    return None
  return FingerReproj(
    base_T_joint=base_T_joint,
    corners_in_pivot=corners_in_pivot,
    observations=observations,
  )
