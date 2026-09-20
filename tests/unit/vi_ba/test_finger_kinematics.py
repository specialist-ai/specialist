"""Synthetic finger angles round-trip through the 1-D scan and the C++ finger factor."""

from __future__ import annotations

import gtsam
import numpy as np
from specialist.vi_ba._mei_factor_cpp import make_mei_seated_finger_factor_cpp

from specialist.vi_ba.anchor_detection import TagDetection
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.finger_kinematics import build_finger_reproj
from specialist.vi_ba.urdf_joints import RevoluteJoint
from tests.mei_fixtures import (
  WIDE_CALIBRATED_RADIUS,
  synthetic_tag_detection,
  wide_mei_params,
)

_PARAMS = wide_mei_params()
_TAG_SIZE = 0.05
_FINGER_TAG_ID = 2004
# Joint 0.25 m in front, tag tilted off-frontal; joint_T_tag0 is the known CAD pose.
_JOINT = gtsam.Pose3(gtsam.Rot3.Rx(2.7), gtsam.Point3(0, 0, 0.25))
_JOINT_T_TAG0 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.02, 0.03, 0.0))


def _detect(pose_cam_T_tag: gtsam.Pose3, tag_id: int) -> TagDetection:
  det = synthetic_tag_detection(pose_cam_T_tag, tag_id, _TAG_SIZE, _PARAMS)
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
    tags_per_kf.append([_detect(body_T_tag, _FINGER_TAG_ID)])

  fr = build_finger_reproj(
    joint, tags_per_kf, lens, _FINGER_TAG_ID, gtsam.Pose3(), _TAG_SIZE, len(thetas)
  )
  assert fr is not None
  assert set(fr.observations) == set(range(len(thetas)))

  for kf, th in enumerate(thetas):
    assert abs(fr.observations[kf].theta_seed - th) < 0.05, (
      kf,
      fr.observations[kf].theta_seed,
      th,
    )

  # The synthetic tags use lens == body, so body_T_base_tag == I.
  noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
  btag_key = gtsam.symbol("G", 0)
  seat_key = gtsam.symbol("E", 0)
  for kf, th in enumerate(thetas):
    vals = gtsam.Values()
    tkey = gtsam.symbol("a", kf)
    vals.insert(tkey, float(th))
    vals.insert(btag_key, gtsam.Pose3())
    vals.insert(seat_key, lens.extrinsic.rotation())
    for j in range(4):
      f = make_mei_seated_finger_factor_cpp(
        tkey,
        btag_key,
        seat_key,
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


def test_batched_theta_seeds_are_the_per_sighting_scans() -> None:
  from specialist.vi_ba.finger_kinematics import _seed_thetas, _theta_scan

  rng = np.random.default_rng(23)
  lens = LensSetup(
    extrinsic=gtsam.Pose3(
      gtsam.Rot3.Expmap(np.array([0.1, -0.2, 0.05])), gtsam.Point3(0.02, 0.0, -0.01)
    ),
    params=wide_mei_params(),
    calibrated_radius=WIDE_CALIBRATED_RADIUS,
  )
  body_T_joint = gtsam.Pose3(
    gtsam.Rot3.Expmap(np.array([0.0, 0.3, 0.0])), gtsam.Point3(0.0, 0.05, 0.3)
  )
  corners_in_pivot = rng.uniform(-0.02, 0.02, (4, 3))
  thetas, projected = _theta_scan(corners_in_pivot, body_T_joint, lens, -0.5, 1.2)
  pixels = projected[rng.integers(0, len(thetas), 30)] + rng.normal(
    0.0, 2.0, (30, 4, 2)
  )
  pixels[7] = np.nan
  got = _seed_thetas(pixels, corners_in_pivot, body_T_joint, lens, -0.5, 1.2)
  for i in range(30):
    errors = np.sum((projected - pixels[i]) ** 2, axis=(1, 2))
    want = thetas[int(np.argmin(np.where(np.isnan(errors), np.inf, errors)))]
    assert got[i] == want
