"""Turn one solved run into the payloads that leave the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import mei_project
from specialist.vi_ba.anchor_detection import (
  TagDetection,
  estimate_camera_to_tag_pose,
  tag_corners_in_frame,
)
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.compose import (
  CrossSighting,
  CulledPoints,
  LandmarkKeep,
  LandmarkPoints,
  LandmarkTracks,
  TrackerKeys,
  anchor_sightings,
  anchored_keyframes,
  points_from_values,
)
from specialist.vi_ba.configs import Args
from specialist.vi_ba.export import TrackerSummary
from specialist.vi_ba.geometry import bounding_box_diagonal, path_length
from specialist.vi_ba.landmark_cull import cull_and_reproject
from specialist.vi_ba.optimize import OptimizeResult
from specialist.vi_ba.single_tracker import PreparedTracker
from specialist.vi_ba.viz import TrackerVizData

Float64Array = npt.NDArray[np.float64]

__all__ = ["RunOutputs", "build_run_outputs"]


def _factor_observation_lines(
  observer: PreparedTracker,
  observer_poses: list[gtsam.Pose3],
  cross_obs_for_observer: list[CrossSighting],
  anchor_corners_by_tag: dict[int, Float64Array],
  base_tag_corners_world: dict[str, list[Float64Array]],
) -> tuple[list[Float64Array], list[Float64Array]]:
  """Per-keyframe ``(n, 2, 3)`` lens-to-corner lines to anchors and base tags."""
  lenses_by_kf: dict[int, list[tuple[LensSetup, Float64Array]]] = {}
  for tag_id, corners_world in anchor_corners_by_tag.items():
    for lens, kf, _det in anchor_sightings(observer, tag_id):
      lenses_by_kf.setdefault(kf, []).append((lens, corners_world))
  anchor_lines: list[Float64Array] = []
  cross_lines: list[Float64Array] = []
  for kf in range(observer.n_keyframes):
    body_pose = observer_poses[kf]
    anchor_strips: list[Float64Array] = []
    for lens, corners_world in lenses_by_kf.get(kf, ()):
      lens_xyz = np.asarray(
        body_pose.compose(lens.extrinsic).translation(), dtype=np.float64
      )
      for corner_xyz in corners_world:
        anchor_strips.append(np.stack([lens_xyz, corner_xyz]))
    anchor_lines.append(
      np.stack(anchor_strips) if anchor_strips else np.empty((0, 2, 3))
    )
    cross_strips: list[Float64Array] = []
    for sighting in cross_obs_for_observer:
      if sighting.finger is not None or sighting.keyframe != kf:
        continue  # base-tag rays only
      owner_corners = base_tag_corners_world.get(sighting.owner)
      if owner_corners is None or kf >= len(owner_corners):
        continue
      lens_xyz = np.asarray(
        body_pose.compose(sighting.lens.extrinsic).translation(), dtype=np.float64
      )
      for corner_xyz in owner_corners[kf]:
        cross_strips.append(np.stack([lens_xyz, corner_xyz]))
    cross_lines.append(np.stack(cross_strips) if cross_strips else np.empty((0, 2, 3)))
  return anchor_lines, cross_lines


def _per_kf_rejected_keypoints(
  rejected_tracks: LandmarkTracks, n_keyframes: int
) -> list[Float64Array]:
  buckets: list[list[Float64Array]] = [[] for _ in range(n_keyframes)]
  for obs in rejected_tracks.values():
    for kf, pix in obs.items():
      if 0 <= kf < n_keyframes:
        buckets[kf].append(np.asarray(pix, dtype=np.float64))
  return [np.stack(b) if b else np.empty((0, 2)) for b in buckets]


def _rms(values: list[float]) -> float:
  """Root-mean-square of ``values``; NaN when empty."""
  return float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")


def _aggregate_rmse_for_print(per_kf: list[float]) -> float:
  return _rms([v for v in per_kf if np.isfinite(v)])


def _project_tag_axes_per_kf(
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  tag_size_by_id: dict[int, float],
) -> list[Float64Array]:
  out: list[Float64Array] = []
  for tags in tags_per_kf:
    strips: list[Float64Array] = []
    for det in tags:
      size = tag_size_by_id.get(int(det.tag_id))
      if size is None:
        continue
      pose = estimate_camera_to_tag_pose(det.corners, size, lens.params)
      if pose is None:
        continue
      tips = np.array(
        [
          [0.0, 0.0, 0.0],
          [size, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, size, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, size],
        ],
        dtype=np.float64,
      )
      rmat = pose.rotation().matrix()
      tvec = np.asarray(pose.translation(), dtype=np.float64)
      pts_cam = (rmat @ tips.T).T + tvec
      if np.any(pts_cam[:, 2] <= 1e-3):
        continue
      pix = mei_project(pts_cam, lens.params)
      if not np.all(np.isfinite(pix)):
        continue
      strips.append(pix.reshape(3, 2, 2))
    out.append(np.concatenate(strips) if strips else np.empty((0, 2, 2)))
  return out


def _build_tracker_outputs(
  tracker: PreparedTracker,
  poses: list[gtsam.Pose3],
  result: OptimizeResult,
  keys: TrackerKeys,
  landmark_keep: tuple[LandmarkTracks, LandmarkTracks],
  args: Args,
  tag_size_by_id: dict[int, float],
  cross_obs: list[CrossSighting],
  anchor_corners_by_tag: dict[int, Float64Array],
  base_tag_corners_world: dict[str, list[Float64Array]],
  audio_offset: float,
) -> tuple[
  TrackerVizData,
  TrackerSummary,
  tuple[LandmarkPoints, LandmarkPoints],
]:
  """``TrackerVizData``, ``TrackerSummary`` and culled ``(front, rear)`` landmarks."""
  front_points = points_from_values(
    result.values, keys.front_landmark_key, tracker.name, landmark_keep[0]
  )
  rear_points = points_from_values(
    result.values, keys.rear_landmark_key, tracker.name, landmark_keep[1]
  )
  # Post-bundle map cull: it cleans the map, RMSE and overlay, never the poses.
  front_points, front_reproj_per_kf, front_rmse_per_kf = cull_and_reproject(
    front_points, tracker.front_tracks_tri, poses, tracker.front_lens
  )
  rear_points, rear_reproj_per_kf, rear_rmse_per_kf = cull_and_reproject(
    rear_points, tracker.rear_tracks_tri, poses, tracker.rear_lens
  )
  front_rmse = _aggregate_rmse_for_print(front_rmse_per_kf)
  rear_rmse = _aggregate_rmse_for_print(rear_rmse_per_kf)
  bbox_diagonal = bounding_box_diagonal(poses)
  summary = TrackerSummary(
    name=tracker.name,
    n_keyframes=tracker.n_keyframes,
    anchor_obs=len(anchored_keyframes(tracker)),
    bounding_box_diagonal=bbox_diagonal,
    path_length=path_length(poses),
  )
  rear_note = f" rear_rmse={rear_rmse:.2f}px" if np.isfinite(rear_rmse) else ""
  print(
    f"  {tracker.name}: kf={tracker.n_keyframes} bbox={bbox_diagonal:.3f} m "
    f"front_rmse={front_rmse:.2f}px{rear_note}"
  )
  if tracker.body_T_base_tag is not None:
    base_tag_poses = [p.compose(tracker.body_T_base_tag) for p in poses]
  else:
    base_tag_poses = poses
  front_tag_viz = [
    [(d.tag_id, d.center, d.corners) for d in tags]
    for tags in tracker.front_tags_per_kf
  ]
  rear_tag_viz = [
    [(d.tag_id, d.center, d.corners) for d in tags] for tags in tracker.rear_tags_per_kf
  ]
  biases_final = [
    result.values.atConstantBias(keys.bias_key(tracker.name, i))
    for i in range(tracker.n_keyframes)
  ]
  # Tag axes cost one PnP solve per detection; only frame-logging runs show them.
  front_tag_axes = (
    _project_tag_axes_per_kf(
      tracker.front_tags_per_kf, tracker.front_lens, tag_size_by_id
    )
    if args.log_frames_in_rrd
    else None
  )
  rear_tag_axes = (
    _project_tag_axes_per_kf(
      tracker.rear_tags_per_kf, tracker.rear_lens, tag_size_by_id
    )
    if args.log_frames_in_rrd
    else None
  )
  front_reproj = front_reproj_per_kf if args.log_frames_in_rrd else None
  rear_reproj = rear_reproj_per_kf if args.log_frames_in_rrd else None
  front_rejected_kp = (
    _per_kf_rejected_keypoints(tracker.front_rejected, tracker.n_keyframes)
    if args.log_frames_in_rrd
    else None
  )
  rear_rejected_kp = (
    _per_kf_rejected_keypoints(tracker.rear_rejected, tracker.n_keyframes)
    if args.log_frames_in_rrd
    else None
  )
  anchor_lines, cross_lines = _factor_observation_lines(
    tracker,
    poses,
    cross_obs,
    anchor_corners_by_tag,
    base_tag_corners_world,
  )
  viz_data = TrackerVizData(
    name=tracker.name,
    body_poses=poses,
    keyframe_times=np.asarray(tracker.keyframe_times, dtype=np.float64) - audio_offset,
    front_points=front_points,
    rear_points=rear_points,
    front_frames=tracker.front_frames if args.log_frames_in_rrd else None,
    rear_frames=tracker.rear_frames if args.log_frames_in_rrd else None,
    front_keypoints=tracker.front_keypoints if args.log_frames_in_rrd else None,
    rear_keypoints=tracker.rear_keypoints if args.log_frames_in_rrd else None,
    front_tag_detections=front_tag_viz,
    rear_tag_detections=rear_tag_viz,
    velocities=[
      np.asarray(result.values.atVector(keys.velocity_key(tracker.name, i)))
      for i in range(tracker.n_keyframes)
    ],
    gyro_biases=[np.asarray(b.gyroscope(), dtype=np.float64) for b in biases_final],
    accel_biases=[
      np.asarray(b.accelerometer(), dtype=np.float64) for b in biases_final
    ],
    front_rmse_per_kf=front_rmse_per_kf,
    rear_rmse_per_kf=rear_rmse_per_kf,
    front_tag_axes=front_tag_axes,
    rear_tag_axes=rear_tag_axes,
    front_landmark_reprojections=front_reproj,
    rear_landmark_reprojections=rear_reproj,
    front_rejected_keypoints=front_rejected_kp,
    rear_rejected_keypoints=rear_rejected_kp,
    rear_static_mask_rects=tracker.rear_static_mask_rects,
    anchor_obs_lines=anchor_lines,
    cross_tracker_lines=cross_lines,
    urdf_root_poses=base_tag_poses,
    joint_angles=tracker.joint_angles or None,
    trajectory_offset_in_gripper=args.gripper.keypoint_offset,
    front_lens_extrinsic=tracker.front_lens.extrinsic,
    rear_lens_extrinsic=tracker.rear_lens.extrinsic,
  )
  return viz_data, summary, (front_points, rear_points)


def _tag_size_by_id(args: Args) -> dict[int, float]:
  """Metric side length per tag id, across anchors, base tags, and fingers."""
  finger_ids = (*args.left_finger_tag_ids, *args.right_finger_tag_ids)
  return {
    **{tid: float(args.anchor.tag_size) for tid in args.anchor.enabled_tag_ids},
    **{int(tid): float(args.gripper.tag_size) for tid in args.base_tag_ids},
    **{int(tid): float(args.gripper.tag_size) for tid in finger_ids if int(tid) > 0},
  }


def _base_tag_corners_world(
  trackers: list[PreparedTracker],
  optimized_body_poses: dict[str, list[gtsam.Pose3]],
  tag_size: float,
) -> dict[str, list[Float64Array]]:
  """Each tracker's base-tag corners in the world, per keyframe."""
  return {
    t.name: [
      tag_corners_in_frame(pose.compose(t.body_T_base_tag), tag_size)
      for pose in optimized_body_poses[t.name]
    ]
    for t in trackers
    if t.body_T_base_tag is not None
  }


@dataclass(frozen=True)
class RunOutputs:
  """A solved run's per-tracker payloads and summaries, and its JSON export geometry."""

  viz_trackers: list[TrackerVizData]
  per_tracker: list[TrackerSummary]
  culled_points: CulledPoints
  anchor_poses_world: dict[int, gtsam.Pose3]
  audio_offset_by_tracker: dict[str, float]


def build_run_outputs(
  trackers: list[PreparedTracker],
  optimized_body_poses: dict[str, list[gtsam.Pose3]],
  result: OptimizeResult,
  keys: TrackerKeys,
  landmark_keep: LandmarkKeep,
  args: Args,
  cross_obs: list[CrossSighting],
  audio_offsets: list[float],
  reference_anchor: gtsam.Pose3,
  reference_tag_id: int,
  secondary_anchor_world: dict[int, gtsam.Pose3],
) -> RunOutputs:
  """Each tracker's payload from one solved joint graph."""
  anchor_poses_world = {reference_tag_id: reference_anchor, **secondary_anchor_world}
  anchor_corners_by_tag = {
    tag_id: tag_corners_in_frame(pose, args.anchor.tag_size)
    for tag_id, pose in anchor_poses_world.items()
  }
  base_tag_corners = _base_tag_corners_world(
    trackers, optimized_body_poses, args.gripper.tag_size
  )
  tag_sizes = _tag_size_by_id(args)
  audio_offset_by_tracker = {t.name: audio_offsets[i] for i, t in enumerate(trackers)}
  by_observer: dict[str, list[CrossSighting]] = {t.name: [] for t in trackers}
  for sighting in cross_obs:
    by_observer[sighting.observer].append(sighting)

  viz_trackers: list[TrackerVizData] = []
  per_tracker: list[TrackerSummary] = []
  culled_points: CulledPoints = {}
  for t in trackers:
    viz_data, summary, culled_pair = _build_tracker_outputs(
      t,
      poses=optimized_body_poses[t.name],
      result=result,
      keys=keys,
      landmark_keep=landmark_keep[t.name],
      args=args,
      tag_size_by_id=tag_sizes,
      cross_obs=by_observer[t.name],
      anchor_corners_by_tag=anchor_corners_by_tag,
      base_tag_corners_world=base_tag_corners,
      audio_offset=audio_offset_by_tracker[t.name],
    )
    viz_trackers.append(viz_data)
    per_tracker.append(summary)
    culled_points[t.name] = culled_pair
  return RunOutputs(
    viz_trackers,
    per_tracker,
    culled_points,
    anchor_poses_world,
    audio_offset_by_tracker,
  )
