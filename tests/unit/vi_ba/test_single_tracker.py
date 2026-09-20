"""Tests for ``specialist.vi_ba.single_tracker``'s anchor-PnP pose and
metric-scale seeds, and for the loop-closure landmark merge in
``specialist.vi_ba.tracks``."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import gtsam
import numpy as np
import pytest
from specialist_camera import mei_project
from specialist_corpus import schema

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.configs import FrontendCfg
from specialist.vi_ba.single_tracker import (
  _anchor_pose_seed,
  bound_malloc_arenas,
  finger_stride,
  keyframe_grid,
)
from specialist.vi_ba.tracks import add_loop_landmarks, run_lens_frontend
from tests.frame_fixtures import match_pair, translating_frames
from tests.mei_fixtures import WIDE_CALIBRATED_RADIUS, narrow_omni_params


def _cpu_frontend_cfg() -> FrontendCfg:
  """The frontend knobs these tests pin: a small feature budget, and the cv2
  flow at full resolution so the primitives stay GPU-free."""
  return FrontendCfg(
    orb_n_features=300, gpu_flow=False, flow_scale=1.0, ransac_threshold=3.0
  )


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
  """Two frontends run on separate threads yield exactly the tracks they yield
  one after another."""
  params = narrow_omni_params()
  cfg = _cpu_frontend_cfg()
  front = translating_frames(6, 2, seed=1)
  rear = translating_frames(6, 3, seed=2)

  with ThreadPoolExecutor(max_workers=2) as pool:
    fut_front = pool.submit(run_lens_frontend, front, params, cfg)
    fut_rear = pool.submit(run_lens_frontend, rear, params, cfg)
    (pf_orb, pf_tracks) = fut_front.result()
    (pr_orb, pr_tracks) = fut_rear.result()
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
  anchor_obs = {k: body[k].compose(ext).inverse() for k in range(n)}
  seed = _anchor_pose_seed(anchor_obs, ext, n)
  assert seed is not None and len(seed) == n
  for k in range(n):
    assert np.allclose(seed[k].matrix(), body[k].matrix(), atol=1e-9)


def test_anchor_pose_seed_none_when_too_few_anchor_keyframes() -> None:
  anchor_obs = {0: gtsam.Pose3(), 1: gtsam.Pose3()}
  assert _anchor_pose_seed(anchor_obs, gtsam.Pose3(), 100) is None


def test_add_loop_landmarks_merges_revisit_into_bundle() -> None:
  """A revisit track spanning two baselined keyframes triangulates to its true
  world point and lands in ``front_kept``, without dropping pre-existing
  tracks."""
  params = narrow_omni_params()
  ext = gtsam.Pose3()
  lens = LensSetup(
    extrinsic=ext, params=params, calibrated_radius=WIDE_CALIBRATED_RADIUS
  )
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

  loop_pts = add_loop_landmarks(
    front_points,
    front_kept,
    loop_tracks,
    body_poses,
    lens,
    FrontendCfg(min_track_length=2, max_landmark_depth=5.0),
    name="t",
  )

  assert 99 in front_kept  # loop track reaches the BA reprojection-factor input
  assert 5 in front_kept  # pre-existing tracks retained
  assert 99 in front_points and 99 in loop_pts  # triangulated landmark merged
  assert np.allclose(front_points[99], p_world, atol=1e-2)


def test_add_loop_landmarks_empty_is_noop() -> None:
  """No revisit tracks leaves ``front_points`` and ``front_kept`` unchanged."""
  params = narrow_omni_params()
  lens = LensSetup(
    extrinsic=gtsam.Pose3(), params=params, calibrated_radius=WIDE_CALIBRATED_RADIUS
  )
  front_points: dict[int, np.ndarray] = {}
  front_kept = {5: {0: np.array([1.0, 1.0])}}
  loop_pts = add_loop_landmarks(
    front_points,
    front_kept,
    {},
    [gtsam.Pose3()],
    lens,
    FrontendCfg(min_track_length=2, max_landmark_depth=5.0),
    name="t",
  )
  assert loop_pts == {}
  assert front_points == {}
  assert list(front_kept) == [5]


def test_add_loop_landmarks_reliable_views_resist_drift() -> None:
  """With ``anchor_kfs`` the revisit landmark triangulates from the
  well-localized views alone, so a drifted anchor-free view cannot pull it off
  truth."""
  params = narrow_omni_params()
  ext = gtsam.Pose3()
  lens = LensSetup(
    extrinsic=ext, params=params, calibrated_radius=WIDE_CALIBRATED_RADIUS
  )
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
  add_loop_landmarks(
    fp_rel,
    fk_rel,
    {99: dict(obs)},
    body_poses,
    lens,
    FrontendCfg(min_track_length=2, max_landmark_depth=5.0),
    name="t",
    anchor_kfs={1, 4},
  )
  fp_all: dict[int, np.ndarray] = {}
  fk_all: dict[int, dict[int, np.ndarray]] = {}
  add_loop_landmarks(
    fp_all,
    fk_all,
    {99: dict(obs)},
    body_poses,
    lens,
    FrontendCfg(min_track_length=2, max_landmark_depth=5.0),
    name="t",
  )

  assert 99 in fp_rel and 99 in fp_all
  err_rel = float(np.linalg.norm(fp_rel[99] - p_world))
  err_all = float(np.linalg.norm(fp_all[99] - p_world))
  assert err_rel < 1e-2  # reliable views -> truth
  assert err_rel < err_all  # the drifted view pulls the naive landmark off truth
  # The full track, drifted tail included, still feeds the BA reprojection
  # factors, so the drifted pose is pulled onto the reliable structure.
  assert 5 in fk_rel[99]


def test_the_finger_stride_divides_the_keyframe_interval() -> None:
  """The dense walk's keyframe subset is only exact when the stride divides the
  interval, so a rate that allows a wider non-divisor still takes the divisor."""
  assert finger_stride(3, 30.0) == 1
  assert finger_stride(6, 60.0) == 2
  # fps / 30 allows 2, but 2 does not divide 5.
  assert finger_stride(5, 60.0) == 1


def test_the_dense_finger_walk_keeps_the_keyframe_grid() -> None:
  """The dense rear walk's keyframe subset must land on the front grid, at
  every phase a covered span can start on."""
  for fps, keyframe_every in ((30.0, 3), (60.0, 6)):
    times = np.arange(0, 200) / fps
    stride = finger_stride(keyframe_every, fps)
    for start in range(7):
      covered = np.arange(start, 150)
      front_rows, phase = keyframe_grid(times, covered, fps, keyframe_every)
      # The grid snaps to the session-clock lattice, not to the covered span.
      session_stride = schema.keyframe_stride(fps)
      lattice = np.round(times[front_rows] * fps).astype(np.int64) % session_stride
      assert not lattice.any()
      rear_rows = covered[phase::stride][:: keyframe_every // stride]
      n = min(len(front_rows), len(rear_rows))
      np.testing.assert_array_equal(rear_rows[:n], front_rows[:n])


def test_frame_jobs_collect_tags_and_orb_per_frame() -> None:
  """Each submitted frame gets its tag list and, with a budget, its ORB frame,
  in frame order; without a budget no ORB is kept."""
  from specialist.vi_ba.single_tracker import _FrameJobs

  frames = translating_frames(5, 2, seed=4)
  with _FrameJobs(len(frames), 100) as jobs:
    for k, frame in enumerate(frames):
      jobs.submit(k, frame)
  assert len(jobs.tags) == 5 and all(tags == [] for tags in jobs.tags)
  assert len(jobs.orb) == 5 and all(len(f.keypoints) > 0 for f in jobs.orb)

  with _FrameJobs(len(frames), 0) as jobs:
    for k, frame in enumerate(frames):
      jobs.submit(k, frame)
  assert jobs.orb == []


def test_loop_closure_window_bounds_the_revisit_gap(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """With a window, no revisit track spans more keyframes than it allows."""
  from specialist.vi_ba import tracks

  monkeypatch.setattr(
    tracks,
    "match_pairs",
    lambda orb, pairs, ratio: [match_pair(orb[i], orb[j], ratio) for i, j in pairs],
  )

  frames = translating_frames(2, 0, seed=5, height=200, width=200)
  orb = [
    __import__("specialist.vi_ba.features", fromlist=["detect_orb"]).detect_orb(
      frames[0], 300, 20
    )
  ] * 60
  poses = [gtsam.Pose3() for _ in range(60)]
  lens = LensSetup(
    extrinsic=gtsam.Pose3(),
    params=narrow_omni_params(),
    calibrated_radius=WIDE_CALIBRATED_RADIUS,
  )
  unbounded = tracks.loop_closure_tracks(orb, poses, lens, 0)
  bounded = tracks.loop_closure_tracks(orb, poses, lens, 0, window=20)
  assert unbounded
  assert all(max(obs) - min(obs) <= 20 for obs in bounded.values())
  assert any(max(obs) - min(obs) > 20 for obs in unbounded.values())


def test_renumber_landmarks_packs_ids_and_keeps_pairing() -> None:
  """Points and tracks come back keyed 0..n-1 in id order, each track still
  attached to its point; a track with no point is dropped."""
  from specialist.vi_ba.single_tracker import renumber_landmarks

  points = {1_000_783: np.array([1.0, 0, 0]), 7: np.array([2.0, 0, 0])}
  tracks = {1_000_783: {3: np.zeros(2)}, 7: {5: np.ones(2)}, 9: {1: np.ones(2)}}
  new_points, new_tracks = renumber_landmarks(points, tracks)
  assert sorted(new_points) == [0, 1] and sorted(new_tracks) == [0, 1]
  assert new_points[0][0] == 2.0 and list(new_tracks[0]) == [5]
  assert new_points[1][0] == 1.0 and list(new_tracks[1]) == [3]


def test_bounding_arenas_reports_whether_the_allocator_took_it() -> None:
  assert bound_malloc_arenas(8) is (sys.platform == "linux")
