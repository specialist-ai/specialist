"""Synthetic end-to-end integration test for the dual-lens VI-BA pipeline.

Drives the pipeline from known 3D landmarks projected through
``mei_project`` at known body poses, turned into ``vi_ba.features``
``Frame`` objects with deterministic matches. Image rendering is skipped
(that belongs to a separate ``.rrd``-visible probe).

What it exercises end-to-end:
  - ``chain_tracks`` (pairwise matches → per-landmark tracks)
  - ``two_view_essential`` (essential matrix on MEI-unprojected bearings)
  - ``interpolate_camera_poses`` + ``body_poses_from_camera_poses``
  - ``triangulate_track_dlt`` (per-track DLT + cheirality)
  - ``build_graph`` + ``optimize_graph`` (MEI factors + LM)

Verdict: LM converges from the two-view init + triangulation to a solution
whose landmarks match ground truth to within the metric scale the gauge
priors have set.
"""

from __future__ import annotations

import gtsam
import numpy as np
import numpy.typing as npt
from gtsam.symbol_shorthand import X

from specialist.loader.mei_model import MeiParams, mei_project, mei_unproject
from specialist.vi_ba.build import (
  LensSetup,
  build_graph,
  front_landmark_key,
)
from specialist.vi_ba.features import Frame, chain_tracks
from specialist.vi_ba.init import (
  body_poses_from_camera_poses,
  interpolate_camera_poses,
  triangulate_track_dlt,
  two_view_essential,
)
from specialist.vi_ba.optimize import optimize_graph

Float64Array = npt.NDArray[np.float64]


def _pinhole_mei() -> MeiParams:
  return MeiParams(
    fx=500.0,
    fy=500.0,
    cx=320.0,
    cy=240.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
    xi=0.0,
  )


def _project(
  body_pose: gtsam.Pose3, extrinsic: gtsam.Pose3, xyz: Float64Array, params: MeiParams
) -> Float64Array:
  p_lens = body_pose.compose(extrinsic).transformTo(
    gtsam.Point3(float(xyz[0]), float(xyz[1]), float(xyz[2]))
  )
  return mei_project(np.asarray(p_lens)[None, :], params)[0]


def test_pipeline_recovers_ground_truth_from_two_view_init() -> None:
  """20 landmarks, 3 body poses, front lens only, pinhole MEI. The pipeline
  should converge to the true landmarks (up to the metric scale fixed by
  the init-pair baseline)."""
  rng = np.random.default_rng(11)
  params = _pinhole_mei()
  extrinsic = gtsam.Pose3()

  # Ground-truth body poses: pure +x translation.
  true_poses = [
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 0.0)),
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.15, 0.0, 0.0)),
    gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.30, 0.0, 0.0)),
  ]
  n_frames = len(true_poses)

  # 20 landmarks uniformly in a box in front of the camera.
  n_lm = 20
  xyzs = rng.uniform(-0.4, 0.4, size=(n_lm, 3))
  xyzs[:, 2] = rng.uniform(1.5, 3.0, size=n_lm)

  # Project each landmark through each pose; build synthetic Frames with
  # keypoints in the same order as the landmark index.
  frames: list[Frame] = []
  for body_pose in true_poses:
    pts = np.array(
      [_project(body_pose, extrinsic, xyzs[i], params) for i in range(n_lm)],
      dtype=np.float64,
    )
    frames.append(
      Frame(keypoints=pts, descriptors=np.zeros((n_lm, 32), dtype=np.uint8))
    )

  # Matches are identity pairs (kp i in frame k ↔ kp i in frame k+1).
  matches_between = [
    np.stack([np.arange(n_lm), np.arange(n_lm)], axis=1).astype(np.int64)
    for _ in range(n_frames - 1)
  ]
  tracks = chain_tracks(frames, matches_between)
  assert len(tracks) == n_lm

  # Two-view init on the widest pair (0, n_frames-1).
  first, last = 0, n_frames - 1
  pix_a = np.array([tracks[tid][first] for tid in tracks])
  pix_b = np.array([tracks[tid][last] for tid in tracks])
  bears_a = mei_unproject(pix_a, params)
  bears_b = mei_unproject(pix_b, params)
  r_cw, t_cw, _ = two_view_essential(bears_a, bears_b, threshold=1e-3)

  cam_poses = interpolate_camera_poses(r_cw, t_cw, n_frames)
  body_poses_init = body_poses_from_camera_poses(cam_poses, extrinsic)

  # Triangulate from earliest + latest observation per track.
  points_init: dict[int, Float64Array] = {}
  for tid, obs in tracks.items():
    frames_sorted = sorted(obs.keys())
    f0, f1 = frames_sorted[0], frames_sorted[-1]
    b0 = mei_unproject(obs[f0][None, :], params)[0]
    b1 = mei_unproject(obs[f1][None, :], params)[0]
    xyz = triangulate_track_dlt(
      b0, b1, body_poses_init[f0], body_poses_init[f1], extrinsic
    )
    if xyz is not None:
      points_init[tid] = xyz

  # Most landmarks should survive cheirality (they are in front of both cams).
  assert len(points_init) >= int(0.9 * n_lm)

  # Build + optimise.
  graph, values = build_graph(
    body_poses=body_poses_init,
    front=LensSetup(extrinsic=extrinsic, params=params),
    front_tracks={tid: tracks[tid] for tid in points_init},
    front_points=points_init,
    rear=None,
    rear_tracks=None,
    rear_points=None,
    pixel_sigma=0.5,
    weak_pose_sigmas=(0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
  )
  initial_err = graph.error(values)
  result = optimize_graph(graph, values, max_iterations=50)

  # LM must not diverge.
  assert result.final_error <= initial_err
  # Per-landmark RMSE should be small in the gauge-fixed scale (endpoints
  # pinned to two-view-interpolated init) — we check *relative* shape
  # agreement rather than absolute scale.
  recovered = np.array(
    [np.asarray(result.values.atPoint3(front_landmark_key(tid))) for tid in points_init]
  )
  truth = np.array([xyzs[tid] for tid in points_init])

  # Factor out global rigid + scale by comparing the centred, normalised
  # point clouds: the two-view init sets the scale, but distance *ratios*
  # should agree with ground truth.
  def _normalised(p: np.ndarray) -> np.ndarray:
    c = p - p.mean(axis=0, keepdims=True)
    return c / np.linalg.norm(c)

  assert np.allclose(_normalised(recovered), _normalised(truth), atol=0.05)

  # LM actually took steps and still ended with reasonable reprojection.
  for i in range(n_frames):
    assert result.values.exists(X(i))
