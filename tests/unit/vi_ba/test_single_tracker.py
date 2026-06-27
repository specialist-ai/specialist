"""Tests for ``single_tracker.run_two_concurrently`` — the per-lens parallelism
helper must return results in order and produce output identical to running the
two lens frontends sequentially."""

from __future__ import annotations

import cv2
import gtsam
import numpy as np
import pytest

from specialist.loader.mei_model import MeiParams, mei_project
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.pipeline import FrontendConfig, run_lens_frontend
from specialist.vi_ba.single_tracker import (
  _add_loop_landmarks,
  _anchor_metric_scale,
  _anchor_pose_seed,
  run_two_concurrently,
)


def test_returns_results_in_order() -> None:
  a, b = run_two_concurrently(lambda: "a", lambda: 2)
  assert a == "a"
  assert b == 2


def test_propagates_exceptions() -> None:
  def boom() -> int:
    raise ValueError("boom")

  with pytest.raises(ValueError, match="boom"):
    run_two_concurrently(lambda: 1, boom)


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


def _translating_frames(n: int, shift: int, seed: int) -> list[np.ndarray]:
  """A textured 240x240 frame translated by ``shift`` px per frame — pure
  translation so DIS flow and essential RANSAC are unambiguous (every match an
  inlier), making the frontend output seed-independent and reproducible."""
  rng = np.random.default_rng(seed)
  base = rng.integers(0, 255, size=(240, 320), dtype=np.uint8)
  base = cv2.GaussianBlur(base, (0, 0), sigmaX=2.0)
  return [np.roll(base, shift * k, axis=1)[:, 40:280].copy() for k in range(n)]


def _tracks_equal(
  a: dict[int, dict[int, np.ndarray]], b: dict[int, dict[int, np.ndarray]]
) -> bool:
  if a.keys() != b.keys():
    return False
  for tid in a:
    if a[tid].keys() != b[tid].keys():
      return False
    for kf in a[tid]:
      if not np.array_equal(a[tid][kf], b[tid][kf]):
        return False
  return True


def test_frontend_parallel_matches_serial() -> None:
  """The headline win: running the two lens frontends concurrently must yield
  the exact same tracks as running them one after another."""
  params = _params()
  cfg = FrontendConfig(n_features=300)
  front = _translating_frames(6, 2, seed=1)
  rear = _translating_frames(6, 3, seed=2)

  (pf_orb, pf_tracks), (pr_orb, pr_tracks) = run_two_concurrently(
    lambda: run_lens_frontend(front, params, cfg),
    lambda: run_lens_frontend(rear, params, cfg),
  )
  sf_orb, sf_tracks = run_lens_frontend(front, params, cfg)
  sr_orb, sr_tracks = run_lens_frontend(rear, params, cfg)

  assert _tracks_equal(pf_tracks, sf_tracks)
  assert _tracks_equal(pr_tracks, sr_tracks)
  assert len(sf_tracks) > 0 and len(sr_tracks) > 0


def test_anchor_pose_seed_reconstructs_metric_body_poses() -> None:
  """The anchor-PnP seed recovers the true metric body poses at keyframes that
  see the anchor (world = anchor-tag frame), through the body->lens extrinsic."""
  ext = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.1, 0.0, 0.0), gtsam.Point3(0.05, 0.0, 0.0))
  n = 10
  body = [
    gtsam.Pose3(
      gtsam.Rot3.Rodrigues(0.0, 0.02 * k, 0.0), gtsam.Point3(0.1 * k, 0.0, 0.5)
    )
    for k in range(n)
  ]
  # cam_T_tag = (world_T_lens)^-1 with world = tag and world_T_lens = body o ext.
  anchor_obs = {k: {"front": body[k].compose(ext).inverse()} for k in range(n)}
  seed = _anchor_pose_seed(anchor_obs, ext, n)
  assert seed is not None and len(seed) == n
  for k in range(n):
    assert np.allclose(seed[k].matrix(), body[k].matrix(), atol=1e-9)


def test_anchor_pose_seed_none_when_too_few_anchor_keyframes() -> None:
  anchor_obs = {0: {"front": gtsam.Pose3()}, 1: {"front": gtsam.Pose3()}}
  assert _anchor_pose_seed(anchor_obs, gtsam.Pose3(), 100) is None


def test_anchor_metric_scale_recovers_scale() -> None:
  """The anchor-metric scale maps an unscaled visual trajectory back to the
  metric anchor frame: with VO positions the metric ones scaled by ``c``, the
  estimator returns ``1/c``."""
  ext = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.1, 0.0, 0.0), gtsam.Point3(0.05, 0.0, 0.0))
  n = 10
  metric_body = [
    gtsam.Pose3(
      gtsam.Rot3.Rodrigues(0.0, 0.02 * k, 0.0),
      gtsam.Point3(0.1 * k, 0.03 * k, 0.5),
    )
    for k in range(n)
  ]
  anchor_obs = {k: {"front": metric_body[k].compose(ext).inverse()} for k in range(n)}
  c = 0.5
  vo_body = [
    gtsam.Pose3(
      metric_body[k].rotation(),
      gtsam.Point3(*(c * np.asarray(metric_body[k].translation()))),
    )
    for k in range(n)
  ]
  scale = _anchor_metric_scale(vo_body, anchor_obs, ext, ext)
  assert scale is not None
  assert abs(scale - 1.0 / c) < 1e-9


def test_anchor_metric_scale_none_when_degenerate() -> None:
  """Too few anchored keyframes (<3) or a zero-spread visual trajectory yields
  None."""
  ext = gtsam.Pose3()
  two = {k: {"front": gtsam.Pose3()} for k in range(2)}
  assert _anchor_metric_scale([gtsam.Pose3()] * 2, two, ext, ext) is None
  # Three anchored views but the VO is stationary → zero spread → None.
  stationary = [gtsam.Pose3()] * 3
  obs = {
    k: {"front": gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.1 * k, 0.0, 0.5)).inverse()}
    for k in range(3)
  }
  assert _anchor_metric_scale(stationary, obs, ext, ext) is None


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
