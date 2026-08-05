"""Turn one solved run into the payloads that leave the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

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
  LandmarkKeep,
  LandmarkTracks,
  TrackerKeys,
  anchor_sightings,
  anchored_keyframes,
  points_from_values,
)
from specialist.vi_ba.configs import Args
from specialist.vi_ba.finger_kinematics import pin_angles_to_keyframes
from specialist.vi_ba.geometry import bounding_box_diagonal, path_length
from specialist.vi_ba.gripper_opening import opening_from_angles
from specialist.vi_ba.highrate import propagate_highrate
from specialist.vi_ba.landmark_cull import cull_and_reproject
from specialist.vi_ba.optimize import OptimizeResult
from specialist.vi_ba.single_tracker import PreparedTracker
from specialist.vi_ba.summary import TrackerSummary
from specialist.vi_ba.urdf_joints import RevoluteJoint, read_revolute_joints
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


def _gripper_opening(
  joint_frame_angles: dict[str, tuple[Float64Array, Float64Array]],
  urdf_joints: dict[str, RevoluteJoint] | None,
) -> tuple[Float64Array, Float64Array] | None:
  """Opening of the two tagged finger joints, or ``None`` unless their counts match."""
  if not urdf_joints or not joint_frame_angles:
    return None
  tagged = [
    name
    for name in joint_frame_angles
    if name in urdf_joints and urdf_joints[name].joint_T_tag0 is not None
  ]
  if len(tagged) != 2:
    return None
  times_a, angles_a = joint_frame_angles[tagged[0]]
  _, angles_b = joint_frame_angles[tagged[1]]
  if len(angles_a) != len(angles_b):
    return None
  opening = opening_from_angles(
    urdf_joints[tagged[0]], urdf_joints[tagged[1]], angles_a, angles_b
  )
  return times_a, opening


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
  urdf_joints: dict[str, RevoluteJoint] | None,
) -> tuple[TrackerVizData, TrackerSummary]:
  """One tracker's Rerun payload from the solved values."""
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
  front_tag_axes = _project_tag_axes_per_kf(
    tracker.front_tags_per_kf, tracker.front_lens, tag_size_by_id
  )
  rear_tag_axes = _project_tag_axes_per_kf(
    tracker.rear_tags_per_kf, tracker.rear_lens, tag_size_by_id
  )
  front_rejected_kp = _per_kf_rejected_keypoints(
    tracker.front_rejected, tracker.n_keyframes
  )
  rear_rejected_kp = _per_kf_rejected_keypoints(
    tracker.rear_rejected, tracker.n_keyframes
  )
  velocities_final = [
    np.asarray(result.values.atVector(keys.velocity_key(tracker.name, i)))
    for i in range(tracker.n_keyframes)
  ]
  highrate_times = highrate_positions = highrate_quats = highrate_vels = None
  if tracker.imu is not None and biases_final:
    hr_t, hr_p, hr_q, hr_v = propagate_highrate(
      poses, tracker.keyframe_times, velocities_final, biases_final, tracker.imu
    )
    highrate_times = hr_t - audio_offset
    highrate_positions = hr_p
    highrate_quats = hr_q
    highrate_vels = hr_v
  joint_frame_angles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
  if tracker.finger_frame_angles:
    kf_t = np.asarray(tracker.keyframe_times, dtype=np.float64)
    for joint, (f_times, f_angles) in tracker.finger_frame_angles.items():
      solved = np.asarray(
        tracker.joint_angles.get(joint, [float("nan")] * len(kf_t)),
        dtype=np.float64,
      )
      n_j = min(len(kf_t), len(solved))
      pinned = pin_angles_to_keyframes(f_times, f_angles, kf_t[:n_j], solved[:n_j])
      joint_frame_angles[joint] = (f_times - audio_offset, pinned)
  gripper_opening = _gripper_opening(joint_frame_angles, urdf_joints)
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
    front_keypoints=tracker.front_keypoints,
    rear_keypoints=tracker.rear_keypoints,
    front_tag_detections=front_tag_viz,
    rear_tag_detections=rear_tag_viz,
    velocities=velocities_final,
    gyro_biases=[np.asarray(b.gyroscope(), dtype=np.float64) for b in biases_final],
    accel_biases=[
      np.asarray(b.accelerometer(), dtype=np.float64) for b in biases_final
    ],
    front_rmse_per_kf=front_rmse_per_kf,
    rear_rmse_per_kf=rear_rmse_per_kf,
    front_tag_axes=front_tag_axes,
    rear_tag_axes=rear_tag_axes,
    front_landmark_reprojections=front_reproj_per_kf,
    rear_landmark_reprojections=rear_reproj_per_kf,
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
    highrate_times=highrate_times,
    highrate_positions=highrate_positions,
    highrate_quats=highrate_quats,
    highrate_vels=highrate_vels,
    joint_frame_angles=joint_frame_angles or None,
    body_T_base_tag=tracker.body_T_base_tag,
    gripper_opening=gripper_opening,
    front_lens_mei={k: float(v) for k, v in asdict(tracker.front_lens.params).items()},
    rear_lens_mei={k: float(v) for k, v in asdict(tracker.rear_lens.params).items()},
  )
  return viz_data, summary


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
  """A solved run's per-tracker payloads and summaries, anchors and audio offsets."""

  viz_trackers: list[TrackerVizData]
  per_tracker: list[TrackerSummary]
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
  urdf_path: Path | None = None,
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
  urdf_joints = read_revolute_joints(urdf_path) if urdf_path is not None else None

  viz_trackers: list[TrackerVizData] = []
  per_tracker: list[TrackerSummary] = []
  for t in trackers:
    viz_data, summary = _build_tracker_outputs(
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
      urdf_joints=urdf_joints,
    )
    viz_trackers.append(viz_data)
    per_tracker.append(summary)
  return RunOutputs(
    viz_trackers,
    per_tracker,
    anchor_poses_world,
    audio_offset_by_tracker,
  )
