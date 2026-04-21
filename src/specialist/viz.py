"""Rerun logging helpers for scenario runs.

Uses a blueprint with a large ``Spatial3DView`` in the center and a column of
``TimeSeriesView`` panels on the right so the viewer opens focused on the
scenario rather than on a tree of every entity.

3D content:

- ``world/anchor`` — static anchor-tag pose with a flat box showing the tag plane.
- ``world/trackers/<name>/gt/trajectory`` and ``.../est/trajectory`` — static
  polylines for the full ground-truth and estimated paths of each tracker.
- ``world/trackers/<name>/{gt,est}/body`` — per-keyframe tracker pose, drawn
  as a small colored box. Child entities attach the two fisheye cameras as
  ``Pinhole`` frustums indicating where each camera is pointing.
- ``world/observations/{anchor,cross_tracker}/<observer>/<side>`` — one line
  strip per observation, logged at its keyframe so it appears when the
  scrubber reaches that keyframe.

Right-panel time series (per keyframe, one series per tracker / observation
kind):

- ``metrics/ate/translation/<name>`` — per-tracker translation error in metres.
- ``metrics/ate/rotation/<name>`` — per-tracker rotation error in radians.
- ``metrics/observations/<kind>`` — count of used observations per keyframe.

The Pinhole frustum uses the same ``scene.fov`` that the synthesis
visibility predicate uses, so what the viewer draws matches what the
detector can see. Pinhole can't represent ``fov >= π``, so ``SceneConfig``
defaults to 120° — still a wide fisheye-ish cone, but finite.
"""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from specialist.graph.optimize import OptimizationResult
from specialist.runner import ScenarioRun
from specialist.synthesis.scene import Scene
from specialist.synthesis.tag_detection import TagObservation

_GT_COLOR = (80, 200, 120)
_EST_COLOR = (80, 130, 230)
_ANCHOR_COLOR = (230, 200, 80)
_CROSS_COLOR = (230, 80, 200)
_CAMERA_GT_COLOR = (120, 220, 160)
_CAMERA_EST_COLOR = (120, 170, 240)

_TIMELINE = "keyframe"
_FOV_VIZ_DISTANCE_M = 0.15


def _pose_to_transform(pose: gtsam.Pose3) -> rr.Transform3D:
  q = pose.rotation().toQuaternion()
  return rr.Transform3D(
    translation=np.asarray(pose.translation(), dtype=np.float64),
    quaternion=rr.Quaternion(xyzw=[q.x(), q.y(), q.z(), q.w()]),
  )


def _pinhole_frustum(fov: float, color: tuple[int, int, int]) -> rr.Pinhole:
  return rr.Pinhole(
    fov_y=fov,
    aspect_ratio=1.0,
    resolution=[200, 200],
    image_plane_distance=_FOV_VIZ_DISTANCE_M,
    color=color,
  )


def _build_blueprint() -> rrb.Blueprint:
  return rrb.Blueprint(
    rrb.Horizontal(
      rrb.Spatial3DView(name="Scene", origin="/world"),
      rrb.Vertical(
        rrb.TimeSeriesView(
          name="Translation error (m)",
          origin="/metrics/ate/translation",
        ),
        rrb.TimeSeriesView(
          name="Rotation error (rad)",
          origin="/metrics/ate/rotation",
        ),
        rrb.TimeSeriesView(
          name="Observations per keyframe",
          origin="/metrics/observations",
        ),
      ),
      column_shares=[3, 1],
    ),
    collapse_panels=True,
  )


def log_scenario_run(
  run: ScenarioRun,
  output_path: str | Path | None = None,
  spawn: bool = False,
  application_id: str = "specialist",
) -> None:
  """Log a ``ScenarioRun`` to Rerun — viewer (``spawn=True``) or .rrd file."""
  rr.init(application_id, spawn=spawn)
  rr.send_blueprint(_build_blueprint())
  if output_path is not None:
    rr.save(str(output_path))

  rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

  _log_anchor_static(run.scene)
  _log_trajectory_paths_static(run)
  _log_per_keyframe(run)
  _log_observations(run)
  _log_lm_summary(run.optimization)


def _log_anchor_static(scene: Scene) -> None:
  rr.log("world/anchor", _pose_to_transform(scene.anchor_pose), static=True)
  rr.log(
    "world/anchor/tag_plane",
    rr.Boxes3D(half_sizes=[[0.15, 0.15, 0.003]], colors=[_ANCHOR_COLOR]),
    static=True,
  )


def _log_trajectory_paths_static(run: ScenarioRun) -> None:
  for name in run.scenario.tracker_names:
    gt_points = [
      np.asarray(p.translation(), dtype=np.float64).tolist()
      for p in run.ground_truth_keyframes[name]
    ]
    est_points = [
      np.asarray(p.translation(), dtype=np.float64).tolist()
      for p in run.estimated_keyframes[name]
    ]
    rr.log(
      f"world/trackers/{name}/gt/trajectory",
      rr.LineStrips3D([gt_points], colors=[_GT_COLOR], radii=[0.004]),
      static=True,
    )
    rr.log(
      f"world/trackers/{name}/est/trajectory",
      rr.LineStrips3D([est_points], colors=[_EST_COLOR], radii=[0.004]),
      static=True,
    )


def _log_per_keyframe(run: ScenarioRun) -> None:
  for k in range(run.n_keyframes):
    rr.set_time(_TIMELINE, sequence=k)
    for name in run.scenario.tracker_names:
      _log_tracker_at_k(run.scene, run.ground_truth_keyframes[name][k], name, "gt")
      _log_tracker_at_k(run.scene, run.estimated_keyframes[name][k], name, "est")
    _log_metrics_at_k(run, k)


def _log_tracker_at_k(scene: Scene, pose: gtsam.Pose3, name: str, kind: str) -> None:
  body_color = _GT_COLOR if kind == "gt" else _EST_COLOR
  cam_color = _CAMERA_GT_COLOR if kind == "gt" else _CAMERA_EST_COLOR
  body_root = f"world/trackers/{name}/{kind}/body"

  rr.log(body_root, _pose_to_transform(pose))
  rr.log(
    f"{body_root}/marker",
    rr.Boxes3D(half_sizes=[[0.04, 0.04, 0.02]], colors=[body_color]),
  )
  for side, extrinsic in (
    ("front", scene.camera_front_on_tracker),
    ("back", scene.camera_back_on_tracker),
  ):
    cam_path = f"{body_root}/cam_{side}"
    rr.log(cam_path, _pose_to_transform(extrinsic))
    rr.log(cam_path, _pinhole_frustum(scene.fov, cam_color))


def _log_metrics_at_k(run: ScenarioRun, k: int) -> None:
  for name in run.scenario.tracker_names:
    gt = run.ground_truth_keyframes[name][k]
    est = run.estimated_keyframes[name][k]
    t_err = float(
      np.linalg.norm(
        np.asarray(gt.translation(), dtype=np.float64)
        - np.asarray(est.translation(), dtype=np.float64)
      )
    )
    tangent = np.asarray(gtsam.Pose3.Logmap(gt.between(est)), dtype=np.float64)
    r_err = float(np.linalg.norm(tangent[:3]))
    rr.log(f"metrics/ate/translation/{name}", rr.Scalars([t_err]))
    rr.log(f"metrics/ate/rotation/{name}", rr.Scalars([r_err]))

  n_anchor = sum(1 for o in run.anchor_observations if o.keyframe_index == k)
  n_cross = sum(1 for o in run.cross_tracker_observations if o.keyframe_index == k)
  rr.log("metrics/observations/anchor", rr.Scalars([float(n_anchor)]))
  rr.log("metrics/observations/cross_tracker", rr.Scalars([float(n_cross)]))


def _log_observations(run: ScenarioRun) -> None:
  # Group observations by keyframe so each frame's rays can be logged after
  # clearing the previous frame — otherwise an old ray persists at later
  # keyframes where the same (observer, side) no longer sees any tag.
  observations_by_keyframe: dict[int, list[TagObservation]] = {}
  for obs in run.used_observations:
    observations_by_keyframe.setdefault(obs.keyframe_index, []).append(obs)

  for k in range(run.n_keyframes):
    rr.set_time(_TIMELINE, sequence=k)
    rr.log("world/observations", rr.Clear(recursive=True))

    if k not in observations_by_keyframe:
      continue

    tracker_poses = {
      n: run.ground_truth_keyframes[n][k] for n in run.ground_truth_keyframes
    }
    for obs in observations_by_keyframe[k]:
      camera_world = run.scene.camera_world(
        obs.observer_tracker, obs.fisheye_side, tracker_poses
      )
      tag_world = run.scene.tag_world(obs.tag_id, tracker_poses)
      color = _ANCHOR_COLOR if obs.tag_kind == "anchor" else _CROSS_COLOR
      kind = "anchor" if obs.tag_kind == "anchor" else "cross_tracker"
      # Include tag_id in the path so two observations from the same
      # (observer, side) but different targets at the same keyframe don't
      # overwrite each other.
      path = (
        f"world/observations/{kind}/{obs.observer_tracker}/"
        f"{obs.fisheye_side}/{obs.tag_id}"
      )
      rr.log(
        path,
        rr.LineStrips3D(
          [
            [
              np.asarray(camera_world.translation(), dtype=np.float64).tolist(),
              np.asarray(tag_world.translation(), dtype=np.float64).tolist(),
            ]
          ],
          colors=[color],
          radii=[0.0015],
        ),
      )


def _log_lm_summary(opt: OptimizationResult) -> None:
  rr.log("metrics/lm/initial_error", rr.Scalars([opt.initial_error]), static=True)
  rr.log("metrics/lm/final_error", rr.Scalars([opt.final_error]), static=True)
  rr.log("metrics/lm/iterations", rr.Scalars([float(opt.iterations)]), static=True)
