"""Turn one solved run into the payloads that leave the pipeline.

``viz`` owns how a ``CameraEstimate`` is logged; this owns how one is built —
the post-bundle landmark cull, the per-keyframe reprojections and their RMSE,
the tag-axis overlay, and the factor-observation lines.
"""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt
from specialist_camera import mei_project
from specialist_corpus.fiducials import tag_corners_in_frame

from specialist.vi_ba.anchor_detection import (
  TagDetection,
  estimate_camera_to_tag_pose,
)
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.compose import (
  CrossSighting,
  LandmarkTracks,
  TrackerKeys,
  anchor_sightings,
  anchored_keyframes,
  points_from_values,
)
from specialist.vi_ba.configs import Args
from specialist.vi_ba.finger_kinematics import pin_angles_to_keyframes
from specialist.vi_ba.geometry import bounding_box_diagonal
from specialist.vi_ba.gripper_opening import opening_from_angles
from specialist.vi_ba.highrate import propagate_highrate
from specialist.vi_ba.landmark_cull import cull_and_reproject
from specialist.vi_ba.optimize import OptimizeResult
from specialist.vi_ba.reference_solve import ReferenceSolve
from specialist.vi_ba.single_tracker import PreparedTracker
from specialist.vi_ba.summary import CameraSummary
from specialist.vi_ba.urdf_joints import FINGER_JOINTS, RevoluteJoint
from specialist.vi_ba.viz import CameraEstimate, LensEstimate

Float64Array = npt.NDArray[np.float64]

__all__ = ["RunEstimate", "build_run_estimate"]


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
  urdf_joints: dict[str, RevoluteJoint],
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


def _build_camera_estimate(
  tracker: PreparedTracker,
  poses: list[gtsam.Pose3],
  result: OptimizeResult,
  keys: TrackerKeys,
  lens_keeps: tuple[LandmarkTracks, LandmarkTracks],
  args: Args,
  tag_size_by_id: dict[int, float],
  cross_obs: list[CrossSighting],
  anchor_corners_by_tag: dict[int, Float64Array],
  base_tag_corners_world: dict[str, list[Float64Array]],
  urdf_joints: dict[str, RevoluteJoint],
) -> tuple[CameraEstimate, CameraSummary]:
  """Assemble one camera's output and summary line from the solved
  values: post-bundle landmark cull, per-keyframe reprojection RMSE, tag-axis /
  landmark-reprojection overlays, and factor-observation lines."""
  lenses: dict[str, LensEstimate] = {}
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
    lenses[channel.side] = LensEstimate(
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
      static_mask_rects=channel.static_mask_rects,
    )
  front_rmse = _rms(lens_rmse["front"])
  rear_rmse = _rms(lens_rmse["rear"])
  bbox_diagonal = bounding_box_diagonal(poses)
  summary = CameraSummary(
    slot=tracker.name,
    n_keyframes=tracker.n_keyframes,
    anchor_obs=len(anchored_keyframes(tracker)),
    front_rmse=front_rmse,
    rear_rmse=rear_rmse,
    base_tag_id=tracker.tags.base,
  )
  rear_note = f" rear_rmse={rear_rmse:.2f}px" if np.isfinite(rear_rmse) else ""
  print(
    f"  [{tracker.name}] kf={tracker.n_keyframes} bbox={bbox_diagonal:.3f} m "
    f"front_rmse={front_rmse:.2f}px{rear_note}"
  )
  biases_final = [
    result.values.atConstantBias(keys.bias_key(tracker.name, i))
    for i in range(tracker.n_keyframes)
  ]
  velocities_final = [
    np.asarray(result.values.atVector(keys.velocity_key(tracker.name, i)))
    for i in range(tracker.n_keyframes)
  ]
  # Stored at the IMU's own sample times, never resampled to a uniform rate.
  highrate_times, highrate_positions, highrate_quats, highrate_vels = (
    propagate_highrate(
      poses, tracker.keyframe_times, velocities_final, biases_final, tracker.imu
    )
  )
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
  camera = CameraEstimate(
    slot=tracker.name,
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
    cross_camera_lines=cross_lines,
    joint_angles=tracker.joint_angles,
    trajectory_offset_in_gripper=args.gripper.keypoint_offset,
    highrate_times=highrate_times,
    highrate_positions=highrate_positions,
    highrate_quats=highrate_quats,
    highrate_vels=highrate_vels,
    joint_frame_angles=joint_frame_angles,
    body_T_base_tag=tracker.body_T_base_tag,
    base_tag_id=tracker.tags.base,
    joint_tags={
      joint: tag_id
      for joint, tag_id in zip(FINGER_JOINTS, tracker.tags.fingers, strict=True)
      if tag_id > 0
    },
    gripper_opening=gripper_opening,
  )
  return camera, summary


def _tag_size_by_id(args: Args, trackers: list[PreparedTracker]) -> dict[int, float]:
  """Metric side length per tag id, across anchors and every tracker's
  discovered gripper tags."""
  gripper_ids = {
    tid for t in trackers for tid in (t.tags.base, *t.tags.fingers) if tid > 0
  }
  return {
    **{tid: float(args.anchor.tag_size) for tid in args.anchor.enabled_tag_ids},
    **{tid: float(args.gripper.tag_size) for tid in gripper_ids},
  }


def _base_tag_corners_world(
  trackers: list[PreparedTracker],
  optimized_body_poses: dict[str, list[gtsam.Pose3]],
  tag_size: float,
) -> dict[str, list[Float64Array]]:
  """Each tracker's base-tag corners in the world, per keyframe."""
  return {
    t.name: [
      tag_corners_in_frame(pose.compose(t.body_T_base_tag).matrix(), tag_size)
      for pose in optimized_body_poses[t.name]
    ]
    for t in trackers
  }


@dataclass(frozen=True)
class RunEstimate:
  """A whole run's estimate: each camera's, the per-camera summary lines, and
  the anchor poses that hold the world gauge."""

  cameras: list[CameraEstimate]
  per_camera: list[CameraSummary]
  anchor_poses_world: dict[int, gtsam.Pose3]
  # The anchor whose pose pins the world gauge, one of anchor_poses_world.
  reference_tag_id: int


def build_run_estimate(
  trackers: list[PreparedTracker],
  solve: ReferenceSolve,
  keys: TrackerKeys,
  args: Args,
  cross_obs: list[CrossSighting],
  urdf_joints: dict[str, RevoluteJoint],
) -> RunEstimate:
  """Build every per-camera payload from one solved joint graph. Every timestamp
  is on the session clock the base recording defines."""
  anchor_poses_world = solve.anchor_world
  anchor_corners_by_tag = {
    tag_id: tag_corners_in_frame(pose.matrix(), args.anchor.tag_size)
    for tag_id, pose in anchor_poses_world.items()
  }
  base_tag_corners = _base_tag_corners_world(
    trackers, solve.body_poses, args.gripper.tag_size
  )
  tag_sizes = _tag_size_by_id(args, trackers)
  by_observer: dict[str, list[CrossSighting]] = {t.name: [] for t in trackers}
  for sighting in cross_obs:
    by_observer[sighting.observer].append(sighting)

  cameras: list[CameraEstimate] = []
  per_camera_summaries: list[CameraSummary] = []
  for t in trackers:
    camera, summary = _build_camera_estimate(
      t,
      poses=solve.body_poses[t.name],
      result=solve.result,
      keys=keys,
      lens_keeps=solve.landmark_keep[t.name],
      args=args,
      tag_size_by_id=tag_sizes,
      cross_obs=by_observer[t.name],
      anchor_corners_by_tag=anchor_corners_by_tag,
      base_tag_corners_world=base_tag_corners,
      urdf_joints=urdf_joints,
    )
    cameras.append(camera)
    per_camera_summaries.append(summary)
  return RunEstimate(
    cameras, per_camera_summaries, anchor_poses_world, solve.ref_tag_id
  )
