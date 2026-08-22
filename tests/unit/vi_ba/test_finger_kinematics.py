"""Synthetic finger motion round-trip: ``build_finger_reproj``'s PnP-free 1-D
scan recovers theta, and the corners reproject through the C++ finger factor."""

from __future__ import annotations

import gtsam
import numpy as np
from specialist.vi_ba._mei_factor_cpp import make_mei_finger_factor_cpp

from specialist.vi_ba.anchor_detection import TagDetection
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.finger_kinematics import build_finger_reproj
from specialist.vi_ba.urdf_joints import RevoluteJoint
from tests.mei_fixtures import (
  WIDE_CALIBRATED_RADIUS,
  synthetic_tag_detection,
  wide_omni_params,
)

_PARAMS = wide_omni_params()
_TAG = 0.05
_FING_ID = 2004
# Joint 0.25 m in front; tag tilted off-frontal. joint_T_tag0 is the known CAD pose.
_JOINT = gtsam.Pose3(gtsam.Rot3.Rx(2.7), gtsam.Point3(0, 0, 0.25))
_JOINT_T_TAG0 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.02, 0.03, 0.0))


def _detect(pose_cam_T_tag: gtsam.Pose3, tag_id: int) -> TagDetection:
  det = synthetic_tag_detection(pose_cam_T_tag, tag_id, _TAG, _PARAMS)
  assert det is not None
  return det


def test_seeds_theta_and_reprojects() -> None:
  lens = LensSetup(
    extrinsic=gtsam.Pose3(), params=_PARAMS, calibrated_radius=WIDE_CALIBRATED_RADIUS
  )  # lens == body
  joint = RevoluteJoint(
    base_T_joint=_JOINT, lower=-0.7, upper=0.7, joint_T_tag0=_JOINT_T_TAG0
  )
  thetas = [-0.1, 0.0, 0.2, 0.4]
  tags_per_kf: list[list[TagDetection]] = []
  for th in thetas:
    body_T_tag = _JOINT.compose(
      gtsam.Pose3(gtsam.Rot3.Rz(th), gtsam.Point3(0, 0, 0))
    ).compose(_JOINT_T_TAG0)
    tags_per_kf.append([_detect(body_T_tag, _FING_ID)])

  fr = build_finger_reproj(
    joint, tags_per_kf, lens, _FING_ID, gtsam.Pose3(), _TAG, len(thetas)
  )
  assert fr is not None
  assert set(fr.observations) == set(range(len(thetas)))

  for kf, th in enumerate(thetas):
    assert abs(fr.observations[kf].theta_seed - th) < 0.05, (
      kf,
      fr.observations[kf].theta_seed,
      th,
    )

  # The synthetic tags were built with lens == body, so body_T_base_tag == I.
  noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
  btag_key = gtsam.symbol("G", 0)
  for kf, th in enumerate(thetas):
    vals = gtsam.Values()
    tkey = gtsam.symbol("a", kf)
    vals.insert(tkey, float(th))
    vals.insert(btag_key, gtsam.Pose3())
    for j in range(4):
      f = make_mei_finger_factor_cpp(
        tkey,
        btag_key,
        fr.observations[kf].pixels[j],
        noise,
        lens.extrinsic,
        fr.base_T_joint,
        gtsam.Point3(*fr.corners_in_pivot[j]),
        _PARAMS.fx,
        _PARAMS.fy,
        _PARAMS.cx,
        _PARAMS.cy,
        _PARAMS.k1,
        _PARAMS.k2,
        _PARAMS.k3,
        _PARAMS.p1,
        _PARAMS.p2,
        _PARAMS.xi,
        fr.observations[kf].obs_bearing,
      )
      assert np.linalg.norm(f.unwhitenedError(vals)) < 0.5, (kf, j)
