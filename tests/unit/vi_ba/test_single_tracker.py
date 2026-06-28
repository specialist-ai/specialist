"""Tests for the anchored loop-closure merge (``_add_loop_landmarks``)."""

from __future__ import annotations

import gtsam
import numpy as np

from specialist.loader.mei_model import MeiParams, mei_project
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.single_tracker import _add_loop_landmarks


def _params() -> MeiParams:
  return MeiParams(
    fx=160.0,
    fy=160.0,
    cx=120.0,
    cy=120.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
    xi=0.8,
  )


def test_add_loop_landmarks_merges_revisit_into_bundle() -> None:
  """A revisit track spanning two baselined keyframes triangulates to its true
  world point and lands in ``front_kept``, without dropping pre-existing
  tracks."""
  params = _params()
  ext = gtsam.Pose3()
  lens = LensSetup(extrinsic=ext, params=params)
  n = 6
  body_poses = [gtsam.Pose3() for _ in range(n)]
  # two baselined keyframes (parallax) that both see the point -> a revisit pair
  body_poses[1] = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 0.0))
  body_poses[4] = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.3, 0.0, 0.0))
  p_world = np.array([0.2, 0.1, 1.5])
  obs = {}
  for kf in (1, 4):
    pt_cam = np.asarray(
      body_poses[kf].compose(ext).transformTo(gtsam.Point3(*p_world)), dtype=np.float64
    )
    obs[kf] = mei_project(pt_cam[None, :], params)[0]
  loop_tracks = {99: obs}
  front_points: dict[int, np.ndarray] = {}
  front_kept = {5: {0: np.array([1.0, 1.0])}}

  loop_pts = _add_loop_landmarks(
    front_points,
    front_kept,
    loop_tracks,
    body_poses,
    lens,
    2,
    5.0,
    "t",
    "anchored loop closure",
  )

  assert 99 in front_kept  # loop track reaches the BA reprojection-factor input
  assert 5 in front_kept  # pre-existing tracks retained
  assert 99 in front_points and 99 in loop_pts  # triangulated landmark merged
  assert np.allclose(front_points[99], p_world, atol=1e-2)


def test_add_loop_landmarks_empty_is_noop() -> None:
  """No revisit tracks leaves ``front_points`` and ``front_kept`` unchanged."""
  params = _params()
  lens = LensSetup(extrinsic=gtsam.Pose3(), params=params)
  front_points: dict[int, np.ndarray] = {}
  front_kept = {5: {0: np.array([1.0, 1.0])}}
  loop_pts = _add_loop_landmarks(
    front_points,
    front_kept,
    {},
    [gtsam.Pose3()],
    lens,
    2,
    5.0,
    "t",
    "anchored loop closure",
  )
  assert loop_pts == {}
  assert front_points == {}
  assert list(front_kept) == [5]


def test_add_loop_landmarks_reliable_views_resist_drift() -> None:
  """With ``anchor_kfs`` the revisit landmark triangulates from the
  well-localized views alone, so a drifted anchor-free view cannot pull it off
  truth."""
  params = _params()
  ext = gtsam.Pose3()
  lens = LensSetup(extrinsic=ext, params=params)
  n = 6
  body_poses = [gtsam.Pose3() for _ in range(n)]
  body_poses[1] = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 0.0))
  body_poses[4] = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.3, 0.0, 0.0))
  # tail kf 5: its true pose has a parallax baseline, but the ESTIMATE is drifted.
  true_pose5 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.6, 0.0, 0.0))
  body_poses[5] = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.6, 0.2, 0.0))
  p_world = np.array([0.2, 0.1, 1.5])

  def _proj(pose: gtsam.Pose3) -> np.ndarray:
    pc = np.asarray(
      pose.compose(ext).transformTo(gtsam.Point3(*p_world)), dtype=np.float64
    )
    return mei_project(pc[None, :], params)[0]

  # tail observation is taken from the TRUE pose (what the camera actually saw)
  obs = {1: _proj(body_poses[1]), 4: _proj(body_poses[4]), 5: _proj(true_pose5)}

  fp_rel: dict[int, np.ndarray] = {}
  fk_rel: dict[int, dict[int, np.ndarray]] = {}
  _add_loop_landmarks(
    fp_rel,
    fk_rel,
    {99: dict(obs)},
    body_poses,
    lens,
    2,
    5.0,
    "t",
    "lc",
    anchor_kfs={1, 4},
  )
  fp_all: dict[int, np.ndarray] = {}
  fk_all: dict[int, dict[int, np.ndarray]] = {}
  _add_loop_landmarks(
    fp_all, fk_all, {99: dict(obs)}, body_poses, lens, 2, 5.0, "t", "lc"
  )

  assert 99 in fp_rel and 99 in fp_all
  err_rel = float(np.linalg.norm(fp_rel[99] - p_world))
  err_all = float(np.linalg.norm(fp_all[99] - p_world))
  assert err_rel < 1e-2  # reliable views -> truth
  assert err_rel < err_all  # the drifted view pulls the naive landmark off truth
  # The full track, drifted tail included, still feeds the BA reprojection
  # factors, so the drifted pose is pulled onto the reliable structure.
  assert 5 in fk_rel[99]
