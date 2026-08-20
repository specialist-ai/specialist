"""Turn one solved run into the payloads that leave the pipeline.

``viz`` owns how a ``TrackerVizData`` is logged; this owns how one is built —
the post-bundle landmark cull, the per-keyframe reprojections and their RMSE,
the tag-axis overlay, and the factor-observation lines.
"""

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
from specialist.vi_ba.viz import LensVizData, TrackerVizData

Float64Array = npt.NDArray[np.float64]

__all__ = ["RunOutputs", "build_run_outputs"]


def _factor_observation_lines(
  observer: PreparedTracker,
  observer_poses: list[gtsam.Pose3],
  cross_obs_for_observer: list[CrossSighting],
  anchor_corners_by_tag: dict[int, Float64Array],
  base_tag_corners_world: dict[str, list[Float64Array]],
  max_half_angle: float,
) -> tuple[list[Float64Array], list[Float64Array]]:
  """Per-keyframe ``(n, 2, 3)`` line endpoints for every factor that fires at
  the observer's keyframe: the observing lens centre to each sighted anchor's
  four metric corners, and to each of the four base-tag corners on the owner
  tracker."""
  lenses_by_kf: dict[int, list[tuple[LensSetup, Float64Array]]] = {}
  for tag_id, corners_world in anchor_corners_by_tag.items():
    for ch, kf, _det in anchor_sightings(observer, tag_id, max_half_angle):
      lenses_by_kf.setdefault(kf, []).append((ch.setup, corners_world))
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
  """Root-mean-square over the finite entries; NaN when there are none."""
  finite = [v for v in values if np.isfinite(v)]
  return float(np.sqrt(np.mean(np.square(finite)))) if finite else float("nan")


def _gripper_opening(
  joint_frame_angles: dict[str, tuple[Float64Array, Float64Array]],
  urdf_joints: dict[str, RevoluteJoint] | None,
) -> tuple[Float64Array, Float64Array] | None:
  """The opening series from the two tagged finger joints, when both carry the
  same number of samples."""
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
  base_tag_id: int,
  poses: list[gtsam.Pose3],
  result: OptimizeResult,
  keys: TrackerKeys,
  lens_keeps: tuple[LandmarkTracks, LandmarkTracks],
  args: Args,
  tag_size_by_id: dict[int, float],
  cross_obs: list[CrossSighting],
  anchor_corners_by_tag: dict[int, Float64Array],
  base_tag_corners_world: dict[str, list[Float64Array]],
  urdf_joints: dict[str, RevoluteJoint] | None,
) -> tuple[TrackerVizData, TrackerSummary]:
  """Assemble one tracker's Rerun viz payload and summary line from the solved
  values: post-bundle landmark cull, per-keyframe reprojection RMSE, tag-axis /
  landmark-reprojection overlays, and factor-observation lines."""
  lenses: dict[str, LensVizData] = {}
  lens_rmse: dict[str, list[float]] = {}
  for channel, landmark_key, keep in (
    (tracker.front, keys.front_landmark_key, lens_keeps[0]),
    (tracker.rear, keys.rear_landmark_key, lens_keeps[1]),
  ):
    points, reprojections, rmse_per_kf = cull_and_reproject(
      points_from_values(result.values, landmark_key, tracker.name, keep),
      channel.tracks,
      poses,
      channel.setup,
    )
    lens_rmse[channel.side] = rmse_per_kf
    lenses[channel.side] = LensVizData(
      points=points,
      keypoints=channel.keypoints,
      rejected_keypoints=_per_kf_rejected_keypoints(
        channel.rejected, tracker.n_keyframes
      ),
      landmark_reprojections=reprojections,
      tag_detections=[
        [(d.tag_id, d.center, d.corners) for d in tags] for tags in channel.tags
      ],
      tag_axes=_project_tag_axes_per_kf(channel.tags, channel.setup, tag_size_by_id),
      rmse_per_kf=rmse_per_kf,
      extrinsic=channel.setup.extrinsic,
      mei={k: float(v) for k, v in asdict(channel.setup.params).items()},
      static_mask_rects=channel.static_mask_rects,
    )
  front_rmse = _rms(lens_rmse["front"])
  rear_rmse = _rms(lens_rmse["rear"])
  bbox_diagonal = bounding_box_diagonal(poses)
  summary = TrackerSummary(
    name=tracker.name,
    n_keyframes=tracker.n_keyframes,
    anchor_obs=len(anchored_keyframes(tracker)),
    bounding_box_diagonal=bbox_diagonal,
    path_length=path_length(poses),
    front_rmse=front_rmse,
    rear_rmse=rear_rmse,
    base_tag_id=base_tag_id,
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
  biases_final = [
    result.values.atConstantBias(keys.bias_key(tracker.name, i))
    for i in range(tracker.n_keyframes)
  ]
  velocities_final = [
    np.asarray(result.values.atVector(keys.velocity_key(tracker.name, i)))
    for i in range(tracker.n_keyframes)
  ]
  highrate_times = highrate_positions = highrate_quats = highrate_vels = None
  if tracker.imu is not None and biases_final:
    # Stored at the IMU's own sample times, never resampled to a uniform rate.
    hr_t, hr_p, hr_q, hr_v = propagate_highrate(
      poses, tracker.keyframe_times, velocities_final, biases_final, tracker.imu
    )
    highrate_times = hr_t
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
      joint_frame_angles[joint] = (f_times, pinned)
  gripper_opening = _gripper_opening(joint_frame_angles, urdf_joints)
  anchor_lines, cross_lines = _factor_observation_lines(
    tracker,
    poses,
    cross_obs,
    anchor_corners_by_tag,
    base_tag_corners_world,
    args.anchor.max_half_angle,
  )
  viz_data = TrackerVizData(
    name=tracker.name,
    body_poses=poses,
    keyframe_times=np.asarray(tracker.keyframe_times, dtype=np.float64),
    keyframe_rows=np.asarray(tracker.keyframe_rows, dtype=np.int64),
    lenses=lenses,
    velocities=velocities_final,
    gyro_biases=[np.asarray(b.gyroscope(), dtype=np.float64) for b in biases_final],
    accel_biases=[
      np.asarray(b.accelerometer(), dtype=np.float64) for b in biases_final
    ],
    anchor_obs_lines=anchor_lines,
    cross_tracker_lines=cross_lines,
    urdf_root_poses=base_tag_poses,
    joint_angles=tracker.joint_angles or None,
    trajectory_offset_in_gripper=args.gripper.keypoint_offset,
    highrate_times=highrate_times,
    highrate_positions=highrate_positions,
    highrate_quats=highrate_quats,
    highrate_vels=highrate_vels,
    joint_frame_angles=joint_frame_angles or None,
    body_T_base_tag=tracker.body_T_base_tag,
    gripper_opening=gripper_opening,
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
  """Everything one solved run hands to its consumers: the Rerun payload per
  tracker, the per-tracker summary lines, and the solved anchor poses."""

  viz_trackers: list[TrackerVizData]
  per_tracker: list[TrackerSummary]
  anchor_poses_world: dict[int, gtsam.Pose3]


def build_run_outputs(
  trackers: list[PreparedTracker],
  optimized_body_poses: dict[str, list[gtsam.Pose3]],
  result: OptimizeResult,
  keys: TrackerKeys,
  landmark_keep: LandmarkKeep,
  args: Args,
  cross_obs: list[CrossSighting],
  world_T_ref_anchor: gtsam.Pose3,
  reference_tag_id: int,
  secondary_anchor_world: dict[int, gtsam.Pose3],
  urdf_path: Path | None = None,
) -> RunOutputs:
  """Build every per-tracker payload from one solved joint graph. The
  reference anchor pins the world gauge; each secondary carries the re-estimate
  that pinned the solve. Every timestamp stays on the session clock the base
  recording defines."""
  anchor_poses_world = {reference_tag_id: world_T_ref_anchor, **secondary_anchor_world}
  anchor_corners_by_tag = {
    tag_id: tag_corners_in_frame(pose, args.anchor.tag_size)
    for tag_id, pose in anchor_poses_world.items()
  }
  base_tag_corners = _base_tag_corners_world(
    trackers, optimized_body_poses, args.gripper.tag_size
  )
  tag_sizes = _tag_size_by_id(args)
  by_observer: dict[str, list[CrossSighting]] = {t.name: [] for t in trackers}
  for sighting in cross_obs:
    by_observer[sighting.observer].append(sighting)
  urdf_joints = read_revolute_joints(urdf_path) if urdf_path is not None else None

  viz_trackers: list[TrackerVizData] = []
  per_tracker: list[TrackerSummary] = []
  for t, base_tag_id in zip(trackers, args.base_tag_ids, strict=True):
    viz_data, summary = _build_tracker_outputs(
      t,
      base_tag_id=base_tag_id,
      poses=optimized_body_poses[t.name],
      result=result,
      keys=keys,
      lens_keeps=landmark_keep[t.name],
      args=args,
      tag_size_by_id=tag_sizes,
      cross_obs=by_observer[t.name],
      anchor_corners_by_tag=anchor_corners_by_tag,
      base_tag_corners_world=base_tag_corners,
      urdf_joints=urdf_joints,
    )
    viz_trackers.append(viz_data)
    per_tracker.append(summary)
  return RunOutputs(viz_trackers, per_tracker, anchor_poses_world)
