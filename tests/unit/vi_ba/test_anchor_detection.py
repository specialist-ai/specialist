"""Tag poses, the shared detector and window following in ``anchor_detection``."""

from __future__ import annotations

import math

import cv2
import gtsam
import numpy as np
import pytest
from specialist_camera import MeiParams, mei_project, mei_unproject
from specialist_dataset.fiducials import tag_corners_in_frame

from specialist.vi_ba.anchor_detection import (
  TagDetection,
  anchor_pose_from_tags,
  estimate_camera_to_tag_pose,
  tag_corners_local,
)


def _params() -> MeiParams:
  """A super-fisheye MEI close to a real rear lens (downsampled)."""
  return MeiParams(
    fx=400.0,
    fy=400.0,
    cx=480.0,
    cy=480.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
    xi=1.9,
  )


def _project_tag(
  cam_T_tag: gtsam.Pose3, tag_size: float, params: MeiParams
) -> np.ndarray:
  """Project the four tag corners in pupil_apriltags's order (BL, BR, TR, TL)."""
  return mei_project(tag_corners_in_frame(cam_T_tag.matrix(), tag_size), params)


def _ippe_solutions(
  corners_px: np.ndarray, tag_size: float, params: MeiParams
) -> list[gtsam.Pose3]:
  """Direct cv2 enumeration of both IPPE solutions, for test setup."""
  bearings = mei_unproject(corners_px.astype(np.float64), params)
  norm = (bearings[:, :2] / bearings[:, 2:3]).reshape(-1, 1, 2)
  obj = tag_corners_local(tag_size).reshape(-1, 1, 3)
  n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
    obj,
    norm,
    np.eye(3, dtype=np.float64),
    np.zeros(4, dtype=np.float64),
    flags=cv2.SOLVEPNP_IPPE,
  )
  out: list[gtsam.Pose3] = []
  for i in range(n):
    rmat, _ = cv2.Rodrigues(rvecs[i])
    t = np.asarray(tvecs[i], dtype=np.float64).reshape(3)
    out.append(gtsam.Pose3(gtsam.Rot3(rmat), gtsam.Point3(*t)))
  return out


def _rot_geodesic(a: gtsam.Pose3, b: gtsam.Pose3) -> float:
  return float(np.linalg.norm(gtsam.Rot3.Logmap(a.rotation().between(b.rotation()))))


def _camera_facing_tag_pose(
  tilt_axis: tuple[float, float, float], tilt_angle: float, distance: float
) -> gtsam.Pose3:
  """A tag ``distance`` m ahead facing the camera, tilted so PnP is non-degenerate."""
  axis = np.asarray(tilt_axis, dtype=np.float64)
  axis = axis / np.linalg.norm(axis)
  rot = gtsam.Rot3.Rx(math.pi).compose(
    gtsam.Rot3.Rodrigues(*(axis * tilt_angle).tolist())
  )
  return gtsam.Pose3(rot, gtsam.Point3(0.0, 0.0, distance))


def test_recovers_known_pose_without_prior() -> None:
  params = _params()
  tag_size = 0.10
  cam_T_tag = _camera_facing_tag_pose((1.0, 0.5, 0.0), 0.25, 0.30)
  corners = _project_tag(cam_T_tag, tag_size, params)
  est = estimate_camera_to_tag_pose(corners, tag_size, params)
  assert est is not None
  assert np.allclose(est.translation(), cam_T_tag.translation(), atol=1e-3)
  assert np.allclose(est.rotation().matrix(), cam_T_tag.rotation().matrix(), atol=1e-3)


def test_recovers_metric_distance_at_typical_handheld_range() -> None:
  """A 0.15 m tag at 0.40 m recovers its distance to within 1 cm."""
  params = _params()
  tag_size = 0.15
  cam_T_tag = _camera_facing_tag_pose((1.0, 0.0, 0.0), 0.20, 0.40)
  corners = _project_tag(cam_T_tag, tag_size, params)
  est = estimate_camera_to_tag_pose(corners, tag_size, params)
  assert est is not None
  recovered = float(np.linalg.norm(np.asarray(est.translation())))
  assert abs(recovered - 0.40) < 0.01, (
    f"recovered camera→tag distance {recovered:.4f} m, expected 0.40 m"
  )


def test_well_conditioned_views_recover_truth() -> None:
  """For a well-tilted view the lowest-reprojection solution matches truth."""
  params = _params()
  tag_size = 0.10
  cam_T_tag = _camera_facing_tag_pose((1.0, 0.5, 0.2), 0.5, 0.30)
  corners = _project_tag(cam_T_tag, tag_size, params)
  est = estimate_camera_to_tag_pose(corners, tag_size, params)
  assert est is not None
  assert _rot_geodesic(est, cam_T_tag) < 0.01


def _flip_scenario() -> tuple[
  MeiParams, float, gtsam.Pose3, gtsam.Pose3, np.ndarray, np.ndarray
]:
  """Two frames where a fast pitch puts the stale prior nearer the wrong IPPE twin."""
  params = _params()
  tag_size = 0.10
  cam1_T_tag = _camera_facing_tag_pose((1.0, 0.0, 0.0), 0.25, 0.60)
  delta = gtsam.Rot3.Rx(0.90).matrix()
  cam2_T_tag = gtsam.Pose3(gtsam.Rot3(delta), gtsam.Point3(0, 0, 0)).compose(cam1_T_tag)
  corners1 = _project_tag(cam1_T_tag, tag_size, params)
  corners2 = _project_tag(cam2_T_tag, tag_size, params)
  return params, tag_size, cam1_T_tag, cam2_T_tag, corners1, corners2


def test_reprojection_ranking_picks_the_true_twin() -> None:
  """Full-MEI corner reprojection RMS picks the true IPPE solution from one frame."""
  params, tag_size, _cam1_T_tag, cam2_T_tag, _, corners2 = _flip_scenario()
  if len(_ippe_solutions(corners2, tag_size, params)) < 2:
    pytest.skip("IPPE returned a single solution for this geometry")
  est = estimate_camera_to_tag_pose(corners2, tag_size, params)
  assert est is not None
  assert _rot_geodesic(est, cam2_T_tag) < 0.05


def test_anchor_chain_resolves_each_frame_on_its_own() -> None:
  """``anchor_pose_from_tags`` survives a fast turn and an unsighted keyframe."""
  params, tag_size, cam1_T_tag, cam2_T_tag, corners1, corners2 = _flip_scenario()
  tags_per_kf = [
    [TagDetection(tag_id=1, center=corners1.mean(0), corners=corners1)],
    [],
    [TagDetection(tag_id=1, center=corners2.mean(0), corners=corners2)],
  ]
  out = anchor_pose_from_tags(tags_per_kf, 1, tag_size, params)
  assert set(out) == {0, 2}
  assert _rot_geodesic(out[0], cam1_T_tag) < 0.05
  assert _rot_geodesic(out[2], cam2_T_tag) < 0.05


def test_tag_detection_never_runs_concurrently() -> None:
  """Two threads inside the native detect at once segfault."""
  import threading
  import time

  from specialist.vi_ba import anchor_detection

  peak = 0
  live = 0
  guard = threading.Lock()

  def hold() -> None:
    nonlocal peak, live
    with anchor_detection._detect_slot():
      with guard:
        live += 1
        peak = max(peak, live)
      time.sleep(0.005)
      with guard:
        live -= 1

  threads = [threading.Thread(target=hold) for _ in range(16)]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()
  assert peak == 1


def test_the_detector_is_built_once_and_never_dropped() -> None:
  """Native teardown corrupts the heap beside another detector."""
  from specialist.vi_ba import anchor_detection

  assert anchor_detection._shared_detector() is anchor_detection._shared_detector()


def test_detector_teardown_reclaims_its_memory() -> None:
  import gc
  import resource

  from specialist.vi_ba import anchor_detection

  def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if peak > 10**7 else peak / 1024

  anchor_detection._shared_detector()  # warm any one-off allocation
  gc.collect()
  before = rss_mb()
  for _ in range(3):
    detector = anchor_detection._OrderedTeardownDetector(families="tagStandard41h12")
    del detector
    gc.collect()
  assert rss_mb() - before < 200.0


def test_detector_pool_never_exceeds_the_cores_the_process_has(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The pool is the smaller of ``_DETECT_THREADS`` and the process's core budget."""
  from specialist.vi_ba import anchor_detection

  monkeypatch.setattr(anchor_detection, "cpu_budget", lambda: 4)
  assert anchor_detection._detect_threads() == 4
  monkeypatch.setattr(anchor_detection, "cpu_budget", lambda: 384)
  assert anchor_detection._detect_threads() == anchor_detection._DETECT_THREADS


def test_pnp_never_returns_a_non_finite_pose() -> None:
  """The estimator declines the non-finite pose IPPE gives a near-degenerate quad."""
  params = _params()
  world_T_tag = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 1.2))
  for k in range(24):
    body = gtsam.Pose3(
      gtsam.Rot3.Rodrigues(0.02 * k, 0.01 * k, 0.0),
      gtsam.Point3(0.01 * k, 0.0, 0.0),
    )
    pixels = mei_project(
      tag_corners_in_frame(body.inverse().compose(world_T_tag).matrix(), 0.1),
      params,
    )
    if not np.all(np.isfinite(pixels)):
      continue
    pose = estimate_camera_to_tag_pose(pixels, 0.1, params)
    assert pose is None or np.all(np.isfinite(pose.matrix())), k


class _SquareDetector:
  """Reports each 20x20 white square as a tag, in the given image's pixels."""

  def detect(self, img: np.ndarray) -> list[object]:
    ys, xs = np.nonzero(img > 200)
    if xs.size == 0:
      return []
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    corners = np.array([[x0, y1], [x1, y1], [x1, y0], [x0, y0]], dtype=np.float64)

    class Hit:
      tag_id = 7
      center = corners.mean(axis=0)

    hit = Hit()
    hit.corners = corners  # type: ignore[attr-defined]
    return [hit]


def _frame_with_square(
  x: int, y: int, shape: tuple[int, int] = (600, 800)
) -> np.ndarray:
  frame = np.zeros(shape, dtype=np.uint8)
  frame[y : y + 20, x : x + 20] = 255
  return frame


def test_following_reports_windowed_hits_in_frame_coordinates(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from specialist.vi_ba import anchor_detection

  monkeypatch.setattr(anchor_detection, "_shared_detector", lambda: _SquareDetector())
  frames = [_frame_with_square(400 + 10 * k, 300 + 5 * k) for k in range(6)]
  out = anchor_detection.detect_tags_following(frames, full_every=3, follow={7})
  assert [len(dets) for dets in out] == [1] * 6
  for k, dets in enumerate(out):
    assert dets[0].tag_id == 7
    np.testing.assert_allclose(dets[0].corners[3], [400 + 10 * k, 300 + 5 * k])


def test_following_recovers_a_lost_tag_on_the_next_whole_frame(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from specialist.vi_ba import anchor_detection

  monkeypatch.setattr(anchor_detection, "_shared_detector", lambda: _SquareDetector())
  frames = [
    _frame_with_square(100, 100),  # whole
    _frame_with_square(700, 500),  # windowed near (100, 100): missed
    _frame_with_square(700, 500),  # nothing followed: whole
    _frame_with_square(705, 505),  # whole again on the stride
    _frame_with_square(710, 510),  # windowed near (705, 505)
  ]
  out = anchor_detection.detect_tags_following(frames, full_every=3, follow={7})
  assert [len(dets) for dets in out] == [1, 0, 1, 1, 1]
  np.testing.assert_allclose(out[4][0].corners[3], [710, 510])


def test_following_ignores_tags_it_is_not_asked_to_follow(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from specialist.vi_ba import anchor_detection

  monkeypatch.setattr(anchor_detection, "_shared_detector", lambda: _SquareDetector())
  frames = [_frame_with_square(100, 100), _frame_with_square(700, 500)]
  out = anchor_detection.detect_tags_following(frames, full_every=5, follow={8})
  assert [len(dets) for dets in out] == [1, 1]


def _random_sightings(
  rng: np.random.Generator, params: MeiParams, tag_size: float, n: int
) -> tuple[np.ndarray, list[gtsam.Pose3]]:
  """Jittered ``(n, 4, 2)`` tag corners, every eleventh off-image, and their poses."""
  corners = np.empty((n, 4, 3))
  poses = []
  for i in range(n):
    rot = gtsam.Rot3.Expmap(rng.normal(0.0, 0.6, 3))
    t = np.array(
      [rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(0.2, 1.0)]
    )
    corners[i] = (rot.matrix() @ tag_corners_local(tag_size).T).T + t
    poses.append(gtsam.Pose3(rot, gtsam.Point3(*t)))
  pixels = mei_project(corners.reshape(-1, 3), params).reshape(n, 4, 2)
  pixels += rng.normal(0.0, 0.02, pixels.shape)
  pixels[::11] += 5000.0
  return pixels, poses


def test_batched_tag_poses_recover_the_sighted_poses() -> None:
  from specialist.vi_ba.anchor_detection import estimate_camera_to_tag_poses

  rng = np.random.default_rng(17)
  params = _params()
  size = 0.05
  sightings, truth = _random_sightings(rng, params, size, 90)
  got = estimate_camera_to_tag_poses(sightings, size, params)
  posed = [i for i, pose in enumerate(got) if pose is not None]
  assert posed and not set(posed) & set(range(0, 90, 11))
  for i in posed:
    error = np.linalg.norm(got[i].translation() - truth[i].translation())
    assert error < 0.02 * np.linalg.norm(truth[i].translation())


def test_anchor_poses_from_tags_take_each_keyframes_first_resolved_sighting() -> None:
  from specialist.vi_ba.anchor_detection import estimate_camera_to_tag_poses

  rng = np.random.default_rng(18)
  params = _params()
  size = 0.05
  sightings, _ = _random_sightings(rng, params, size, 40)
  tags_per_kf = [
    [
      TagDetection(tag_id=7 if k % 3 else 2, center=s.mean(axis=0), corners=s)
      for s in sightings[2 * k : 2 * k + 2]
    ]
    for k in range(20)
  ]
  got = anchor_pose_from_tags(tags_per_kf, 7, size, params)
  want: dict[int, gtsam.Pose3] = {}
  for kf, tags in enumerate(tags_per_kf):
    for det in tags:
      if det.tag_id == 7:
        (pose,) = estimate_camera_to_tag_poses(det.corners[None], size, params)
        if pose is not None:
          want[kf] = pose
          break
  assert list(got) == list(want)
  for kf, pose in want.items():
    assert np.array_equal(got[kf].matrix(), pose.matrix())


def test_following_walked_in_segments_reports_what_one_walk_reports(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from specialist.vi_ba import anchor_detection
  from specialist.vi_ba.anchor_detection import (
    detect_tags_following,
    detect_tags_following_split,
    following_segments,
  )

  monkeypatch.setattr(anchor_detection, "_shared_detector", lambda: _SquareDetector())
  frames = [_frame_with_square(40 + 9 * k, 60 + 5 * k) for k in range(37)]
  frames[12] = np.zeros_like(frames[12])
  whole = detect_tags_following(frames, 3, {7})
  split = detect_tags_following_split(lambda b, e: frames[b:e], len(frames), 3, {7}, 4)
  assert len(split) == len(whole) == len(frames)
  for got, want in zip(split, whole, strict=True):
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
      assert g.tag_id == w.tag_id
      assert np.array_equal(g.corners, w.corners)
  runs = following_segments(37, 3, 4)
  assert runs[0][0] == 0 and runs[-1][1] == 37
  assert all(begin % 3 == 0 for begin, _ in runs)
  assert all(end == nxt for (_, end), (nxt, _) in zip(runs, runs[1:], strict=False))
