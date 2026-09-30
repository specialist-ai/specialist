"""Synthetic end-to-end integration test for the dual-lens VI-BA pipeline.

Known 3D landmarks projected through ``mei_project`` at known body poses, with
deterministic matches, drive ``chain_tracks``, ``fill_camera_poses``,
``body_poses_from_camera_poses``, ``triangulate_lens`` and ``optimize_graph``.
LM converges to the true landmarks and to the true pose of the keyframe the
anchor missed.
"""

from __future__ import annotations

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.features import chain_tracks
from specialist.vi_ba.init import body_poses_from_camera_poses, fill_camera_poses
from specialist.vi_ba.optimize import optimize_graph
from specialist.vi_ba.tracks import triangulate_lens
from tests.mei_fixtures import (
  WIDE_CALIBRATED_RADIUS,
  front_landmark_key,
  pinhole_params,
  pinned_visual_graph,
  project_point,
)


def test_pipeline_recovers_ground_truth_from_anchor_seed() -> None:
  """20 landmarks, 3 body poses, front lens only, pinhole MEI: the anchor sights
  the end keyframes, the geodesic fill seeds the middle one off its true pose,
  and the pipeline converges to the true landmarks and middle pose."""
  rng = np.random.default_rng(11)
  params = pinhole_params()
  extrinsic = gtsam.Pose3(gtsam.Rot3.Ypr(0.1, 0.0, 0.0), gtsam.Point3(0.02, 0.0, 0.05))

  # Ground-truth body poses: +x translation, the middle one off the straight line.
  true_poses = [
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 0.0)),
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.15, 0.03, 0.0)),
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.30, 0.0, 0.0)),
  ]
  n_keyframes = len(true_poses)

  # 20 landmarks uniformly in a box in front of the camera.
  n_lm = 20
  xyzs = rng.uniform(-0.4, 0.4, size=(n_lm, 3))
  xyzs[:, 2] = rng.uniform(1.5, 3.0, size=n_lm)

  # Keypoints stay in landmark-index order.
  keypoints = [
    np.array(
      [project_point(body_pose, extrinsic, xyzs[i], params) for i in range(n_lm)],
      dtype=np.float64,
    )
    for body_pose in true_poses
  ]

  # Matches are identity pairs (kp i in frame k ↔ kp i in frame k+1).
  matches_between = [
    np.stack([np.arange(n_lm), np.arange(n_lm)], axis=1).astype(np.int64)
    for _ in range(n_keyframes - 1)
  ]
  tracks = chain_tracks(keypoints, matches_between)
  assert len(tracks) == n_lm

  # The anchor fixes the camera at the end keyframes only.
  anchor_cam_poses = {
    kf: true_poses[kf].compose(extrinsic) for kf in (0, n_keyframes - 1)
  }
  body_poses_init = body_poses_from_camera_poses(
    fill_camera_poses(anchor_cam_poses, n_keyframes), extrinsic
  )
  assert not body_poses_init[1].equals(true_poses[1], 1e-2)

  points_init = triangulate_lens(
    tracks, body_poses_init, extrinsic, params, min_track_length=2
  )
  assert len(points_init) >= int(0.9 * n_lm)

  lens = LensSetup(
    extrinsic=extrinsic, params=params, calibrated_radius=WIDE_CALIBRATED_RADIUS
  )
  graph, values = pinned_visual_graph(
    body_poses_init, lens, {tid: tracks[tid] for tid in points_init}, points_init
  )
  initial_err = graph.error(values)
  result = optimize_graph(graph, values, max_iterations=50)

  assert result.final_error < initial_err
  recovered = np.array(
    [np.asarray(result.values.atPoint3(front_landmark_key(tid))) for tid in points_init]
  )
  truth = np.array([xyzs[tid] for tid in points_init])
  np.testing.assert_allclose(recovered, truth, atol=1e-3)
  assert result.values.atPose3(X(1)).equals(true_poses[1], 1e-3)
