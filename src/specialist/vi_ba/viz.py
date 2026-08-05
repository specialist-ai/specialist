"""Columnar Rerun logging for dual-lens VI-BA. ``log_multi_ba_entities`` takes
one ``TrackerVizData`` per tracker and logs every track-layer entity and overlay
(poses, RMSE, biases, keypoints, tags, reprojections, URDF, ...) under that
tracker's entity-path roots into the caller's active recording; the layer
writer in ``specialist.record.layer`` owns the stream, blueprint, and sinks."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import gtsam
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb

from specialist.record import schema as record_schema
from specialist.vi_ba.geometry import matrices_to_quats, quat_xyzw, quats_to_matrices

Float64Array = npt.NDArray[np.float64]

_FRONT_COLOR = (80, 200, 120)
_REAR_COLOR = (230, 140, 80)
_RED = (230, 60, 60)
_YELLOW = (250, 250, 50)
_MAGENTA = (255, 0, 255)
_CYAN = (90, 220, 220)
_TRAJECTORY_TRAIL_DURATION = 1.0
_BODY_AXIS_LENGTH = 0.05
_TIMELINE = record_schema.TIMELINE

__all__ = ["TrackerVizData", "log_multi_ba_entities", "run_blueprint"]


def _vec3_column(values: Float64Array) -> pa.FixedSizeListArray:
  """An ``(n, 3)`` float array as the arrow column AnyValues can carry."""
  flat = pa.array(np.ascontiguousarray(values, dtype=np.float64).reshape(-1))
  return pa.FixedSizeListArray.from_arrays(flat, 3)


def _translations(poses: list[gtsam.Pose3]) -> Float64Array:
  return np.asarray([np.asarray(p.translation(), dtype=np.float64) for p in poses])


def _pose_transform3d(pose: gtsam.Pose3) -> rr.Transform3D:
  return rr.Transform3D(
    translation=np.asarray(pose.translation(), dtype=np.float64),
    quaternion=rr.datatypes.Quaternion(xyzw=quat_xyzw(pose.rotation())),
  )


def _trail_strips(
  traj: Float64Array, times: Float64Array, duration: float
) -> list[Float64Array]:
  """Per-keyframe strips of the trajectory positions inside the trailing
  ``[t - duration, t]`` window (``times`` ascending, seconds)."""
  starts = np.searchsorted(times, times - duration, side="left")
  return [traj[start : i + 1] for i, start in enumerate(starts)]


def _log_static_points(
  entity_path: str, points: dict[int, Float64Array], color: tuple[int, int, int]
) -> None:
  if not points:
    return
  xyz = np.asarray(list(points.values()), dtype=np.float64)
  rr.log(entity_path, rr.Points3D(xyz, radii=0.002, colors=color), static=True)


def _solid_colors(color: tuple[int, int, int], n: int) -> np.ndarray:
  """An ``(n, 3)`` uint8 array of one repeated RGB color."""
  return np.tile(np.asarray(color, dtype=np.uint8), (n, 1))


@dataclass(frozen=True)
class _OverlayIndex:
  """The timeline index for per-keyframe overlay batches, or a sequence fallback."""

  times: Float64Array | None  # None = sequence fallback
  n: int

  def resolve(self, partition: list[int]) -> tuple[rr.TimeColumn, list[int]]:
    if self.times is None:
      index = rr.TimeColumn(_TIMELINE, sequence=np.arange(self.n, dtype=np.int64))
      return index, partition
    return rr.TimeColumn(_TIMELINE, duration=self.times), partition


def _send_lens_tag_columns(
  entity_prefix: str,
  tags_per_kf: list[list[tuple[int, Float64Array, Float64Array]]],
  index: _OverlayIndex,
  anchor_tag_ids: set[int],
  mask_pad_px: int,
) -> None:
  """Per-keyframe AprilTag overlay: 5-point closed-quad ``LineStrips2D``
  through detected corners + id label. Anchor tag is yellow, other tags
  are red; a negative ``mask_pad_px`` means the frontend masks no tag, so no
  mask bbox is drawn alongside."""
  yellow = np.asarray(_YELLOW, dtype=np.uint8)
  red = np.asarray(_RED, dtype=np.uint8)
  strips: list[np.ndarray] = []
  centers: list[np.ndarray] = []
  labels: list[str] = []
  colors: list[np.ndarray] = []
  partition: list[int] = []
  mask_centers: list[np.ndarray] = []
  mask_half_sizes: list[np.ndarray] = []
  mask_partition: list[int] = []
  for tags in tags_per_kf:
    partition.append(len(tags))
    masked_in_kf = 0
    for tid, center, corners in tags:
      closed = np.vstack([corners, corners[0:1]]).astype(np.float32)
      strips.append(closed)
      centers.append(np.asarray(center, dtype=np.float32))
      labels.append(str(tid))
      is_anchor = int(tid) in anchor_tag_ids
      colors.append(yellow if is_anchor else red)
      if not is_anchor and mask_pad_px >= 0:
        mn = corners.min(axis=0) - float(mask_pad_px)
        mx = corners.max(axis=0) + float(mask_pad_px)
        mask_centers.append(((mn + mx) * 0.5).astype(np.float32))
        mask_half_sizes.append(((mx - mn) * 0.5).astype(np.float32))
        masked_in_kf += 1
    mask_partition.append(masked_in_kf)
  total = sum(partition)
  if total == 0:
    return
  color_arr = np.stack(colors)
  time_col, cleared = index.resolve(partition)
  rr.send_columns(
    f"{entity_prefix}/tags",
    indexes=[time_col],
    columns=rr.LineStrips2D.columns(
      strips=strips,
      colors=color_arr,
      radii=np.full(total, 5.0, dtype=np.float32),
    ).partition(cleared),
  )
  rr.send_columns(
    f"{entity_prefix}/tags/labels",
    indexes=[time_col],
    columns=rr.Points2D.columns(
      positions=np.stack(centers),
      labels=labels,
      show_labels=[True] * total,
      colors=color_arr,
      radii=np.full(total, 10.0, dtype=np.float32),
    ).partition(cleared),
  )
  total_masked = sum(mask_partition)
  if total_masked > 0:
    mask_col, mask_cleared = index.resolve(mask_partition)
    rr.send_columns(
      f"{entity_prefix}/tags/masked",
      indexes=[mask_col],
      columns=rr.Boxes2D.columns(
        centers=np.stack(mask_centers),
        half_sizes=np.stack(mask_half_sizes),
        colors=_solid_colors(_RED, total_masked),
        labels=["MASKED"] * total_masked,
        show_labels=[True] * total_masked,
      ).partition(mask_cleared),
    )


def _log_static_mask_rects(
  entity_prefix: str, rects: list[tuple[int, int, int, int]]
) -> None:
  if not rects:
    return
  centers: list[np.ndarray] = []
  half_sizes: list[np.ndarray] = []
  for x1, y1, x2, y2 in rects:
    centers.append(np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32))
    half_sizes.append(np.array([(x2 - x1) * 0.5, (y2 - y1) * 0.5], dtype=np.float32))
  rr.log(
    f"{entity_prefix}/static_mask",
    rr.Boxes2D(
      centers=np.stack(centers),
      half_sizes=np.stack(half_sizes),
      colors=_solid_colors(_RED, len(rects)),
      labels=["STATIC"] * len(rects),
      show_labels=True,
    ),
    static=True,
  )


def _send_point_columns(
  entity_path: str,
  points_per_kf: list[Float64Array],
  index: _OverlayIndex,
  radius: float,
  color: tuple[int, int, int],
) -> None:
  partition = [int(kp.shape[0]) for kp in points_per_kf]
  total = sum(partition)
  if total == 0:
    return
  positions = np.concatenate([kp for kp in points_per_kf if kp.shape[0] > 0])
  time_col, cleared = index.resolve(partition)
  rr.send_columns(
    entity_path,
    indexes=[time_col],
    columns=rr.Points2D.columns(
      positions=positions,
      radii=np.full(total, radius, dtype=np.float32),
      colors=_solid_colors(color, total),
    ).partition(cleared),
  )


def _send_tag_axes_columns(
  entity_prefix: str,
  axes_per_kf: list[Float64Array],
  index: _OverlayIndex,
) -> None:
  rgb = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255)], dtype=np.uint8)
  strips: list[np.ndarray] = []
  colors: list[np.ndarray] = []
  partition: list[int] = [int(a.shape[0]) for a in axes_per_kf]
  for axes in axes_per_kf:
    for i in range(int(axes.shape[0])):
      strips.append(axes[i].astype(np.float32))
      colors.append(rgb[i % 3])
  total = sum(partition)
  if total == 0:
    return
  time_col, cleared = index.resolve(partition)
  rr.send_columns(
    f"{entity_prefix}/tags/axes",
    indexes=[time_col],
    columns=rr.LineStrips2D.columns(
      strips=strips,
      colors=np.stack(colors),
      radii=np.full(total, 4.0, dtype=np.float32),
    ).partition(cleared),
  )


def _send_factor_line_columns(
  entity_path: str,
  lines_per_kf: list[Float64Array],
  index: _OverlayIndex,
  color: tuple[int, int, int],
) -> None:
  """Per-keyframe 3D ``LineStrips3D`` overlay. ``lines_per_kf[k]`` is an
  ``(n_lines, 2, 3)`` array of (start, end) endpoints; empty arrays yield
  zero strips at that keyframe."""
  partition = [int(arr.shape[0]) for arr in lines_per_kf]
  total = sum(partition)
  if total == 0:
    return
  strips = [line.astype(np.float32) for arr in lines_per_kf for line in arr]
  time_col, cleared = index.resolve(partition)
  rr.send_columns(
    entity_path,
    indexes=[time_col],
    columns=rr.LineStrips3D.columns(
      strips=strips,
      colors=_solid_colors(color, total),
      radii=np.full(total, 0.0008, dtype=np.float32),
    ).partition(cleared),
  )


def _send_lens_columns(
  entity_path: str,
  metrics_path: str,
  keypoints: list[Float64Array] | None,
  index: _OverlayIndex,
  times: rr.TimeColumn,
  color: tuple[int, int, int],
) -> None:
  if not keypoints:
    return
  _send_point_columns(f"{entity_path}/keypoints", keypoints, index, 1.5, color)
  counts = np.asarray([kp.shape[0] for kp in keypoints], dtype=np.float64)
  rr.send_columns(
    metrics_path,
    indexes=[times],
    columns=rr.Scalars.columns(scalars=counts),
  )


@dataclass
class TrackerVizData:
  """One tracker's Rerun payload, built by ``viz_build.build_run_outputs`` and
  logged by ``log_multi_ba_run``."""

  name: str
  body_poses: list[gtsam.Pose3] = field(default_factory=list)
  keyframe_times: Float64Array | None = None
  front_points: dict[int, Float64Array] = field(default_factory=dict)
  rear_points: dict[int, Float64Array] = field(default_factory=dict)
  front_keypoints: list[Float64Array] | None = None
  rear_keypoints: list[Float64Array] | None = None
  velocities: list[Float64Array] | None = None
  gyro_biases: list[Float64Array] | None = None
  accel_biases: list[Float64Array] | None = None
  front_rmse_per_kf: list[float] | None = None
  rear_rmse_per_kf: list[float] | None = None
  front_tag_detections: list[list[tuple[int, Float64Array, Float64Array]]] | None = None
  rear_tag_detections: list[list[tuple[int, Float64Array, Float64Array]]] | None = None
  front_rejected_keypoints: list[Float64Array] | None = None
  rear_rejected_keypoints: list[Float64Array] | None = None
  rear_static_mask_rects: list[tuple[int, int, int, int]] | None = None
  front_landmark_reprojections: list[Float64Array] | None = None
  rear_landmark_reprojections: list[Float64Array] | None = None
  front_tag_axes: list[Float64Array] | None = None
  rear_tag_axes: list[Float64Array] | None = None
  urdf_root_poses: list[gtsam.Pose3] = field(default_factory=list)
  # URDF joint name → per-keyframe angle, NaN where it could not be recovered.
  joint_angles: dict[str, list[float]] | None = None
  trajectory_offset_in_gripper: tuple[float, float, float] | None = None
  front_lens_extrinsic: gtsam.Pose3 | None = None
  rear_lens_extrinsic: gtsam.Pose3 | None = None
  # Per-keyframe ``(n, 2, 3)`` lens-to-corner ray endpoints, one strip per firing
  # reprojection factor: anchor corners, then another tracker's base-tag corners.
  anchor_obs_lines: list[Float64Array] | None = None
  cross_tracker_lines: list[Float64Array] | None = None
  # IMU-propagated body series at the IMU sample times; times on the shared clock.
  highrate_times: Float64Array | None = None
  highrate_positions: Float64Array | None = None
  highrate_quats: Float64Array | None = None
  highrate_vels: Float64Array | None = None
  # Per-joint pinned finger angles at the frame rate, as (times, angles).
  joint_frame_angles: dict[str, tuple[Float64Array, Float64Array]] | None = None
  # The solved body -> base-tag transform (a BA variable), logged statically.
  body_T_base_tag: gtsam.Pose3 | None = None
  # Gripper opening at the pinned finger cadence, as (times, meters).
  gripper_opening: tuple[Float64Array, Float64Array] | None = None
  # MEI intrinsics at decode resolution per lens, logged as lens statics.
  front_lens_mei: dict[str, float] | None = None
  rear_lens_mei: dict[str, float] | None = None


def _gripper_points(data: TrackerVizData) -> Float64Array:
  """Per-keyframe trajectory points: the URDF root translated by the gripper
  keypoint offset when an offset is given, else the raw body translations."""
  if data.trajectory_offset_in_gripper is not None:
    offset = np.asarray(data.trajectory_offset_in_gripper, dtype=np.float64)
    return np.stack(
      [
        np.asarray(p.translation(), dtype=np.float64)
        + np.asarray(p.rotation().matrix(), dtype=np.float64) @ offset
        for p in data.urdf_root_poses
      ]
    )
  return _translations(data.body_poses)


def _log_anchor(entity: str, world_T_anchor: gtsam.Pose3, tag_size: float) -> None:
  half = tag_size / 2.0
  square = np.array(
    [
      [+half, -half, 0.0],
      [+half, +half, 0.0],
      [-half, +half, 0.0],
      [-half, -half, 0.0],
      [+half, -half, 0.0],
    ],
    dtype=np.float64,
  )
  rr.log(
    f"world/{entity}",
    _pose_transform3d(world_T_anchor),
    rr.LineStrips3D([square], colors=_YELLOW, radii=0.003),
    static=True,
  )
  rr.log(
    f"world/{entity}/frame",
    rr.TransformAxes3D(axis_length=half),
    static=True,
  )


def _log_tracker(
  data: TrackerVizData,
  spatial_root: str,
  image_root: str,
  metrics_suffix: str,
  frame_prefix: str | None,
  urdf_path: Path | None,
  anchor_tag_ids: set[int],
  mask_pad_px: int,
  trajectory_color: tuple[int, int, int],
) -> None:
  """Log every per-tracker entity (poses, landmarks, URDF, image overlays,
  scalar metrics). Shared statics (anchor tag, view coordinates, world TF)
  must already be logged by the caller. Scalar paths are
  ``metrics/<scalar>{metrics_suffix}`` so multi-tracker views with origin
  ``/metrics/<scalar>`` aggregate per-tracker series automatically."""
  pose_translations = _translations(data.body_poses)
  traj = _gripper_points(data)

  _log_static_points(f"{spatial_root}/landmarks/front", data.front_points, _FRONT_COLOR)
  _log_static_points(f"{spatial_root}/landmarks/rear", data.rear_points, _REAR_COLOR)
  rr.log(
    f"{spatial_root}/body",
    rr.TransformAxes3D(axis_length=_BODY_AXIS_LENGTH),
    static=True,
  )
  rr.log(
    f"{spatial_root}/body/marker",
    rr.Points3D(np.zeros((1, 3)), radii=0.002, colors=(240, 80, 80)),
    static=True,
  )
  for lens_name, lens_pose, lens_mei in (
    ("front", data.front_lens_extrinsic, data.front_lens_mei),
    ("rear", data.rear_lens_extrinsic, data.rear_lens_mei),
  ):
    if lens_pose is None:
      continue
    rr.log(
      f"{spatial_root}/body/{lens_name}_lens",
      _pose_transform3d(lens_pose),
      static=True,
    )
    rr.log(
      f"{spatial_root}/body/{lens_name}_lens",
      rr.TransformAxes3D(axis_length=_BODY_AXIS_LENGTH),
      static=True,
    )
    if lens_mei is not None:
      rr.log(
        f"{spatial_root}/body/{lens_name}_lens",
        rr.AnyValues(**cast("dict[str, Any]", lens_mei)),
        static=True,
      )
  if data.body_T_base_tag is not None:
    rr.log(
      record_schema.base_tag_entity(data.name),
      _pose_transform3d(data.body_T_base_tag),
      rr.TransformAxes3D(axis_length=0.02),
      static=True,
    )

  urdf_root_frame: str | None = None
  tree: rr.urdf.UrdfTree | None = None
  if urdf_path is not None:
    try:
      tree = rr.urdf.UrdfTree.from_file_path(
        str(urdf_path), entity_path_prefix=spatial_root, frame_prefix=frame_prefix
      )
      tree.log_urdf_to_recording()
      for link in [tree.root_link(), *map(tree.get_joint_child, tree.joints())]:
        for p in tree.get_collision_geometry_paths(link):
          rr.log(p, rr.Clear(recursive=True), static=True)
      urdf_root_frame = (
        f"{frame_prefix}{tree.root_link().name}"
        if frame_prefix
        else tree.root_link().name
      )
      rr.log(
        f"{spatial_root}/gripper",
        rr.Transform3D(parent_frame="world", child_frame=urdf_root_frame),
        static=True,
      )
      rr.log(
        f"{spatial_root}/gripper",
        rr.TransformAxes3D(axis_length=_BODY_AXIS_LENGTH),
        static=True,
      )
    except Exception as exc:  # noqa: BLE001 - the URDF model is optional; keep the rest
      traceback.print_exc()
      print(f"  [viz] URDF model skipped ({exc}); logging the trajectory without it")
      tree = None
      urdf_root_frame = None

  n = len(data.body_poses)
  if data.keyframe_times is not None:
    trail_times = np.asarray(data.keyframe_times, dtype=np.float64)
    times = rr.TimeColumn(_TIMELINE, duration=trail_times)
    overlay_index = _OverlayIndex(times=trail_times, n=n)
  else:
    trail_times = np.arange(n, dtype=np.float64)
    times = rr.TimeColumn(_TIMELINE, sequence=np.arange(n, dtype=np.int64))
    overlay_index = _OverlayIndex(times=None, n=n)

  hr: tuple[Float64Array, Float64Array, Float64Array] | None = None
  if (
    data.highrate_times is not None
    and data.highrate_positions is not None
    and data.highrate_quats is not None
  ):
    hr = (data.highrate_times, data.highrate_positions, data.highrate_quats)
  if hr is not None:
    hr_times, hr_positions, hr_quats = hr
    rr.send_columns(
      f"{spatial_root}/body",
      indexes=[rr.TimeColumn(_TIMELINE, duration=hr_times)],
      columns=rr.Transform3D.columns(translation=hr_positions, quaternion=hr_quats),
    )
  else:
    quats = np.stack([quat_xyzw(p.rotation()) for p in data.body_poses])
    rr.send_columns(
      f"{spatial_root}/body",
      indexes=[times],
      columns=rr.Transform3D.columns(translation=pose_translations, quaternion=quats),
    )
  # The exact BA keyframe poses, kept beside the propagated stream: they are
  # what the scorers align and what regression diffs compare.
  kf_quats = np.stack([quat_xyzw(p.rotation()) for p in data.body_poses])
  rr.send_columns(
    record_schema.body_keyframes_entity(data.name),
    indexes=[times],
    columns=rr.Transform3D.columns(translation=pose_translations, quaternion=kf_quats),
  )
  if hr is not None and data.highrate_vels is not None:
    rr.send_columns(
      f"{spatial_root}/body",
      indexes=[rr.TimeColumn(_TIMELINE, duration=hr[0])],
      columns=rr.AnyValues.columns(velocity=_vec3_column(data.highrate_vels)),
    )
  elif data.velocities is not None:
    rr.send_columns(
      f"{spatial_root}/body",
      indexes=[times],
      columns=rr.AnyValues.columns(velocity=_vec3_column(np.stack(data.velocities))),
    )
  gripper_hr: tuple[Float64Array, Float64Array] | None = None
  if urdf_root_frame is not None:
    poses = data.urdf_root_poses
    if hr is not None and data.body_poses and poses:
      hr_times, hr_positions, hr_quats = hr
      rel = data.body_poses[0].inverse().compose(poses[0])
      rel_R = np.asarray(rel.rotation().matrix())
      rel_t = np.asarray(rel.translation())
      body_R = quats_to_matrices(hr_quats)
      gripper_t = hr_positions + body_R @ rel_t
      gripper_R = body_R @ rel_R
      gripper_hr = (gripper_t, gripper_R)
      count = len(gripper_t)
      rr.send_columns(
        f"{spatial_root}/gripper",
        indexes=[rr.TimeColumn(_TIMELINE, duration=hr_times)],
        columns=rr.Transform3D.columns(
          translation=gripper_t,
          quaternion=matrices_to_quats(gripper_R),
          parent_frame=["world"] * count,
          child_frame=[urdf_root_frame] * count,
        ),
      )
    else:
      urdf_traj = _translations(poses)
      urdf_quats = np.stack([quat_xyzw(p.rotation()) for p in poses])
      rr.send_columns(
        f"{spatial_root}/gripper",
        indexes=[times],
        columns=rr.Transform3D.columns(
          translation=urdf_traj,
          quaternion=urdf_quats,
          parent_frame=["world"] * len(poses),
          child_frame=[urdf_root_frame] * len(poses),
        ),
      )
    if data.joint_angles is not None:
      assert tree is not None  # reached only on the path that set urdf_root_frame
      for joint_name, raw_values in data.joint_angles.items():
        joint = tree.get_joint_by_name(joint_name)
        if joint is None:
          continue
        frame_series = (data.joint_frame_angles or {}).get(joint_name)
        if frame_series is not None:
          # The pinned frame-rate series passes through the solved keyframe
          # angles, so it carries the keyframe cadence at frame rate.
          f_times, f_angles = frame_series
          rr.send_columns(
            f"{spatial_root}/{joint_name}",
            indexes=[rr.TimeColumn(_TIMELINE, duration=f_times)],
            columns=joint.compute_transform_columns([float(a) for a in f_angles]),
          )
          rr.send_columns(
            f"{spatial_root}/{joint_name}",
            indexes=[rr.TimeColumn(_TIMELINE, duration=f_times)],
            columns=rr.AnyValues.columns(angle=np.asarray(f_angles, dtype=np.float64)),
          )
          continue
        first_finite = next((float(v) for v in raw_values if np.isfinite(v)), None)
        if first_finite is None:
          continue
        # Hold each finite angle through dropouts; seed from the first
        # finite value so leading-NaN keyframes don't snap the URDF to zero.
        clean: list[float] = []
        last = first_finite
        for v in raw_values:
          if np.isfinite(v):
            last = float(v)
          clean.append(last)
        rr.send_columns(
          f"{spatial_root}/{joint_name}",
          indexes=[times],
          columns=joint.compute_transform_columns(clean),
        )
        rr.send_columns(
          f"{spatial_root}/{joint_name}",
          indexes=[times],
          columns=rr.AnyValues.columns(angle=np.asarray(raw_values, dtype=np.float64)),
        )
  if data.gripper_opening is not None:
    opening_times, opening_values = data.gripper_opening
    rr.send_columns(
      f"metrics/gripper_opening{metrics_suffix}",
      indexes=[rr.TimeColumn(_TIMELINE, duration=opening_times)],
      columns=rr.Scalars.columns(scalars=opening_values),
    )
  # The trajectory follows the propagated series at the gripper keypoint; the
  # keyframe polyline stands in when a run has no IMU to propagate.
  if hr is not None:
    hr_times, hr_positions, _ = hr
    if gripper_hr is not None and data.trajectory_offset_in_gripper is not None:
      offset = np.asarray(data.trajectory_offset_in_gripper, dtype=np.float64)
      gripper_t, gripper_R = gripper_hr
      traj_points = gripper_t + gripper_R @ offset
    else:
      traj_points = hr_positions
    traj_times = rr.TimeColumn(_TIMELINE, duration=hr_times)
    traj_trail_times = hr_times
  else:
    traj_points = traj
    traj_times = times
    traj_trail_times = trail_times
  rr.send_columns(
    f"{spatial_root}/trajectory",
    indexes=[traj_times],
    columns=rr.LineStrips3D.columns(
      strips=_trail_strips(traj_points, traj_trail_times, _TRAJECTORY_TRAIL_DURATION),
      radii=np.full(len(traj_points), 0.002, dtype=np.float32),
      colors=_solid_colors(trajectory_color, len(traj_points)),
    ),
  )
  rr.send_columns(
    f"metrics/distance_from_start{metrics_suffix}",
    indexes=[traj_times],
    columns=rr.Scalars.columns(
      scalars=np.linalg.norm(traj_points - traj_points[0:1], axis=1)
    ),
  )

  if data.anchor_obs_lines is not None:
    _send_factor_line_columns(
      f"{spatial_root}/factor_lines/anchor",
      data.anchor_obs_lines,
      overlay_index,
      _YELLOW,
    )
  if data.cross_tracker_lines is not None:
    _send_factor_line_columns(
      f"{spatial_root}/factor_lines/cross_tracker",
      data.cross_tracker_lines,
      overlay_index,
      _CYAN,
    )

  # Landmark colors match the 3D scene; the 2D keypoint measurements are
  # magenta, so the reprojection-to-measurement gap reads at a glance.
  for lens_name, kps, color in (
    ("front", data.front_keypoints, _MAGENTA),
    ("rear", data.rear_keypoints, _MAGENTA),
  ):
    _send_lens_columns(
      f"{image_root}/{lens_name}",
      f"metrics/n_keypoints/{lens_name}{metrics_suffix}",
      kps,
      overlay_index,
      times,
      color,
    )
  for lens_name, tags in (
    ("front", data.front_tag_detections),
    ("rear", data.rear_tag_detections),
  ):
    if tags is None:
      continue
    _send_lens_tag_columns(
      f"{image_root}/{lens_name}", tags, overlay_index, anchor_tag_ids, mask_pad_px
    )
  for lens_name, rejected in (
    ("front", data.front_rejected_keypoints),
    ("rear", data.rear_rejected_keypoints),
  ):
    if rejected is None:
      continue
    _send_point_columns(
      f"{image_root}/{lens_name}/keypoints_rejected", rejected, overlay_index, 1.5, _RED
    )
  for lens_name, reproj, color in (
    ("front", data.front_landmark_reprojections, _FRONT_COLOR),
    ("rear", data.rear_landmark_reprojections, _REAR_COLOR),
  ):
    if reproj is None:
      continue
    _send_point_columns(
      f"{image_root}/{lens_name}/landmark_reprojection",
      reproj,
      overlay_index,
      3.0,
      color,
    )
  for lens_name, axes in (
    ("front", data.front_tag_axes),
    ("rear", data.rear_tag_axes),
  ):
    if axes is None:
      continue
    _send_tag_axes_columns(f"{image_root}/{lens_name}", axes, overlay_index)
  if data.rear_static_mask_rects:
    _log_static_mask_rects(f"{image_root}/rear", data.rear_static_mask_rects)
  velocity: tuple[rr.TimeColumn, Float64Array] | None = None
  if hr is not None and data.highrate_vels is not None:
    velocity = (rr.TimeColumn(_TIMELINE, duration=hr[0]), data.highrate_vels)
  elif data.velocities is not None:
    velocity = (times, np.stack(data.velocities))
  if velocity is not None:
    velocity_times, velocity_values = velocity
    rr.send_columns(
      f"metrics/speed/body{metrics_suffix}",
      indexes=[velocity_times],
      columns=rr.Scalars.columns(scalars=np.linalg.norm(velocity_values, axis=1)),
    )
  for metric_name, biases in (
    ("gyro_bias", data.gyro_biases),
    ("accel_bias", data.accel_biases),
  ):
    if biases is None:
      continue
    for axis, axis_name in enumerate("xyz"):
      rr.send_columns(
        f"metrics/{metric_name}/{axis_name}{metrics_suffix}",
        indexes=[times],
        columns=rr.Scalars.columns(
          scalars=np.asarray([float(b[axis]) for b in biases], dtype=np.float64)
        ),
      )
  for lens_name, rmse in (
    ("front", data.front_rmse_per_kf),
    ("rear", data.rear_rmse_per_kf),
  ):
    if rmse is None:
      continue
    rr.send_columns(
      f"metrics/rmse/{lens_name}{metrics_suffix}",
      indexes=[times],
      columns=rr.Scalars.columns(scalars=np.asarray(rmse, dtype=np.float64)),
    )


_TRAJECTORY_PALETTE = (
  _FRONT_COLOR,
  _REAR_COLOR,
  (130, 140, 240),
  (220, 90, 200),
  _CYAN,
)


def blueprint_for_names(
  tracker_names: list[str], floor_z: float | None = None
) -> rrb.Blueprint:
  """Multi-tracker blueprint: one image-pair column per tracker, scalar
  plots aggregated across trackers under ``/metrics/<name>/...``."""
  line_grid = (
    rrb.LineGrid3D(
      plane=rr.datatypes.Plane3D(normal=(0.0, 0.0, 1.0), distance=float(floor_z))
    )
    if floor_z is not None
    else None
  )
  image_columns = [
    rrb.Vertical(
      rrb.Spatial2DView(name=f"{n}/front", origin=f"/image/{n}/front"),
      rrb.Spatial2DView(name=f"{n}/rear", origin=f"/image/{n}/rear"),
      name=n,
    )
    for n in tracker_names
  ]
  return rrb.Blueprint(
    rrb.Horizontal(
      rrb.Tabs(
        rrb.Spatial3DView(
          name="3D",
          origin="/",
          contents=["+ /**", "- /image/**"],
          eye_controls=rrb.archetypes.EyeControls3D(
            kind=rrb.components.Eye3DKind.Orbital,
            tracking_entity=f"/{tracker_names[0]}/body/marker",
            position=(0.8, 0.8, 0.8),
            look_target=(0.0, 0.0, 0.0),
          ),
          line_grid=line_grid,
        ),
        rrb.Vertical(
          rrb.Horizontal(
            rrb.TimeSeriesView(
              name="distance_from_start", origin="/metrics/distance_from_start"
            ),
            rrb.TimeSeriesView(name="speed", origin="/metrics/speed"),
          ),
          rrb.Horizontal(
            rrb.TimeSeriesView(name="n_keypoints", origin="/metrics/n_keypoints"),
            rrb.TimeSeriesView(name="rmse", origin="/metrics/rmse"),
          ),
          name="plots",
        ),
      ),
      rrb.Horizontal(*image_columns) if len(image_columns) > 1 else image_columns[0],
      column_shares=[2, max(1, len(tracker_names))],
    ),
    rrb.BlueprintPanel(state=rrb.PanelState.Collapsed),
    rrb.SelectionPanel(state=rrb.PanelState.Collapsed),
    rrb.TimePanel(state=rrb.PanelState.Collapsed),
  )


def run_blueprint(trackers: list[TrackerVizData]) -> rrb.Blueprint:
  """The standard dashboard for a solved run, floor grid at the lowest
  trajectory point."""
  gripper_points = [_gripper_points(t) for t in trackers]
  floor_zs = [float(pts[:, 2].min()) for pts in gripper_points if len(pts)]
  floor_z = min(floor_zs) if floor_zs else None
  return blueprint_for_names([t.name for t in trackers], floor_z)


def log_multi_ba_entities(
  trackers: list[TrackerVizData],
  anchor_poses_world: dict[int, gtsam.Pose3],
  reference_tag_id: int,
  anchor_tag_size: float,
  mask_pad_px: int = 0,
  urdf_path: Path | None = None,
) -> None:
  """Log every entity of a solved run into the active recording. Each
  tracker's entities live under ``/<name>/...``, image overlays under
  ``/image/<name>/<lens>``, scalars under ``/metrics/<name>/...``. The
  reference anchor tag is logged once at ``/world/anchor_tag``; every other
  anchor at its solved pose under ``/world/anchor_tag_<id>``. The caller
  owns the stream, blueprint, and sinks."""
  rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
  rr.log("world", rr.Transform3D(child_frame="world"), static=True)
  for tag_id, pose in anchor_poses_world.items():
    entity = "anchor_tag" if tag_id == reference_tag_id else f"anchor_tag_{tag_id}"
    _log_anchor(entity, pose, anchor_tag_size)
  for i, t in enumerate(trackers):
    _log_tracker(
      t,
      spatial_root=t.name,
      image_root=record_schema.image_root(t.name),
      metrics_suffix=f"/{t.name}",
      frame_prefix=f"{t.name}_",
      urdf_path=urdf_path,
      anchor_tag_ids=set(anchor_poses_world),
      mask_pad_px=mask_pad_px,
      trajectory_color=_TRAJECTORY_PALETTE[i % len(_TRAJECTORY_PALETTE)],
    )
