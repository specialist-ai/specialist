"""Rerun chunks of a solved run, each handed to the sink of the layer it belongs in."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import gtsam
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from rerun.experimental import Chunk

from specialist.record import schema
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
_TIMELINE = schema.TIMELINE

__all__ = [
  "TrackerVizData",
  "blueprint_for_names",
  "solution_chunks",
  "run_blueprint",
]


Sink = Callable[[Chunk], None]


def _send(emit: Sink, entity_path: str, *, indexes: Any, columns: Any) -> None:
  """One temporal chunk, shaped exactly as the columnar log call was."""
  emit(Chunk.from_columns(entity_path, indexes=indexes, columns=columns))


def _static(emit: Sink, entity_path: str, *columns: Any) -> None:
  """One static chunk carrying every archetype logged at that entity."""
  emit(
    Chunk.from_columns(
      entity_path, indexes=[], columns=[c for group in columns for c in group]
    )
  )


def _pose_columns(pose: gtsam.Pose3) -> Any:
  """A single-row Transform3D column list for one pose."""
  return rr.Transform3D.columns(
    translation=np.asarray(pose.translation(), dtype=np.float64)[None, :],
    quaternion=quat_xyzw(pose.rotation())[None, :],
  )


def _vec3_column(values: Float64Array) -> pa.FixedSizeListArray:
  """An ``(n, 3)`` float array as the arrow column AnyValues can carry."""
  flat = pa.array(np.ascontiguousarray(values, dtype=np.float64).reshape(-1))
  return pa.FixedSizeListArray.from_arrays(flat, 3)


def _translations(poses: list[gtsam.Pose3]) -> Float64Array:
  return np.asarray([np.asarray(p.translation(), dtype=np.float64) for p in poses])


def _pose_quats(poses: list[gtsam.Pose3]) -> Float64Array:
  return np.stack([quat_xyzw(p.rotation()) for p in poses])


def _trail_strips(
  traj: Float64Array, times: Float64Array, duration: float
) -> list[Float64Array]:
  """Per-keyframe position strips in ``[t - duration, t]``; ``times`` must ascend."""
  starts = np.searchsorted(times, times - duration, side="left")
  return [traj[start : i + 1] for i, start in enumerate(starts)]


def _send_static_points(
  emit: Sink,
  entity_path: str,
  points: dict[int, Float64Array],
  color: tuple[int, int, int],
) -> None:
  if not points:
    return
  xyz = np.asarray(list(points.values()), dtype=np.float64)
  _static(
    emit,
    entity_path,
    rr.Points3D.columns(positions=xyz).partition([len(xyz)]),
    rr.Points3D.columns(radii=[0.002], colors=[color]),
  )


def _solid_colors(color: tuple[int, int, int], n: int) -> np.ndarray:
  """An ``(n, 3)`` uint8 array of one repeated RGB color."""
  return np.tile(np.asarray(color, dtype=np.uint8), (n, 1))


# Overlay data-row and clearing-row indexes; clearing rows never carry ``keyframe``.
_OverlayIndex = tuple[list[rr.TimeColumn], list[rr.TimeColumn] | None]


def _overlay_index(
  trail_times: Float64Array | None,
  frame_period: float | None,
  fallback: list[rr.TimeColumn],
  keyframe_column: rr.TimeColumn | None,
) -> _OverlayIndex:
  """Indexes bounding each overlay to its video frame; ``fallback`` without a period."""
  if trail_times is None or frame_period is None:
    return fallback, None
  data = [rr.TimeColumn(_TIMELINE, duration=trail_times)]
  if keyframe_column is not None:
    data.append(keyframe_column)
  clear = [rr.TimeColumn(_TIMELINE, duration=trail_times + frame_period)]
  return data, clear


def _send_lens_tag_columns(
  emit: Sink,
  entity_prefix: str,
  tags_per_kf: list[list[tuple[int, Float64Array, Float64Array]]],
  index: _OverlayIndex,
  anchor_tag_ids: set[int],
  mask_pad_px: int,
) -> None:
  """AprilTag quads and ids per keyframe; negative ``mask_pad_px`` draws no mask box."""
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
  data_indexes, clear_indexes = index
  n = len(partition)
  color_arr = np.stack(colors)
  _send(
    emit,
    f"{entity_prefix}/tags",
    indexes=data_indexes,
    columns=rr.LineStrips2D.columns(
      strips=strips,
      colors=color_arr,
      radii=np.full(total, 5.0, dtype=np.float32),
    ).partition(partition),
  )
  _send(
    emit,
    f"{entity_prefix}/tags/labels",
    indexes=data_indexes,
    columns=rr.Points2D.columns(
      positions=np.stack(centers),
      labels=labels,
      show_labels=[True] * total,
      colors=color_arr,
      radii=np.full(total, 10.0, dtype=np.float32),
    ).partition(partition),
  )
  if clear_indexes is not None:
    _send(
      emit,
      f"{entity_prefix}/tags",
      indexes=clear_indexes,
      columns=rr.LineStrips2D.columns(
        strips=[],
        colors=np.empty((0, 3), dtype=np.uint8),
        radii=np.empty(0, dtype=np.float32),
      ).partition([0] * n),
    )
    _send(
      emit,
      f"{entity_prefix}/tags/labels",
      indexes=clear_indexes,
      columns=rr.Points2D.columns(
        positions=np.empty((0, 2), dtype=np.float64),
        labels=[],
        show_labels=[],
        colors=np.empty((0, 3), dtype=np.uint8),
        radii=np.empty(0, dtype=np.float32),
      ).partition([0] * n),
    )
  total_masked = sum(mask_partition)
  if total_masked > 0:
    _send(
      emit,
      f"{entity_prefix}/tags/masked",
      indexes=data_indexes,
      columns=rr.Boxes2D.columns(
        centers=np.stack(mask_centers),
        half_sizes=np.stack(mask_half_sizes),
        colors=_solid_colors(_RED, total_masked),
        labels=["MASKED"] * total_masked,
        show_labels=[True] * total_masked,
      ).partition(mask_partition),
    )
    if clear_indexes is not None:
      _send(
        emit,
        f"{entity_prefix}/tags/masked",
        indexes=clear_indexes,
        columns=rr.Boxes2D.columns(
          centers=np.empty((0, 2), dtype=np.float32),
          half_sizes=np.empty((0, 2), dtype=np.float32),
          colors=np.empty((0, 3), dtype=np.uint8),
          labels=[],
          show_labels=[],
        ).partition([0] * n),
      )


def _log_static_mask_rects(
  emit: Sink, entity_prefix: str, rects: list[tuple[int, int, int, int]]
) -> None:
  if not rects:
    return
  centers: list[np.ndarray] = []
  half_sizes: list[np.ndarray] = []
  for x1, y1, x2, y2 in rects:
    centers.append(np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32))
    half_sizes.append(np.array([(x2 - x1) * 0.5, (y2 - y1) * 0.5], dtype=np.float32))
  _static(
    emit,
    f"{entity_prefix}/static_mask",
    rr.Boxes2D.columns(
      centers=np.stack(centers),
      half_sizes=np.stack(half_sizes),
      colors=_solid_colors(_RED, len(rects)),
      labels=["STATIC"] * len(rects),
    ).partition([len(rects)]),
    rr.Boxes2D.columns(show_labels=[True]),
  )


def _send_point_columns(
  emit: Sink,
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
  data_indexes, clear_indexes = index
  positions = np.concatenate([kp for kp in points_per_kf if kp.shape[0] > 0])
  _send(
    emit,
    entity_path,
    indexes=data_indexes,
    columns=rr.Points2D.columns(
      positions=positions,
      radii=np.full(total, radius, dtype=np.float32),
      colors=_solid_colors(color, total),
    ).partition(partition),
  )
  if clear_indexes is not None:
    _send(
      emit,
      entity_path,
      indexes=clear_indexes,
      columns=rr.Points2D.columns(
        positions=np.empty((0, 2), dtype=np.float64),
        radii=np.empty(0, dtype=np.float32),
        colors=np.empty((0, 3), dtype=np.uint8),
      ).partition([0] * len(partition)),
    )


def _send_tag_axes_columns(
  emit: Sink,
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
  data_indexes, clear_indexes = index
  _send(
    emit,
    f"{entity_prefix}/tags/axes",
    indexes=data_indexes,
    columns=rr.LineStrips2D.columns(
      strips=strips,
      colors=np.stack(colors),
      radii=np.full(total, 4.0, dtype=np.float32),
    ).partition(partition),
  )
  if clear_indexes is not None:
    _send(
      emit,
      f"{entity_prefix}/tags/axes",
      indexes=clear_indexes,
      columns=rr.LineStrips2D.columns(
        strips=[],
        colors=np.empty((0, 3), dtype=np.uint8),
        radii=np.empty(0, dtype=np.float32),
      ).partition([0] * len(partition)),
    )


def _send_factor_line_columns(
  emit: Sink,
  entity_path: str,
  lines_per_kf: list[Float64Array],
  indexes: list[rr.TimeColumn],
  color: tuple[int, int, int],
) -> None:
  """Per-keyframe ``(n_lines, 2, 3)`` endpoint arrays as LineStrips3D rows."""
  partition = [int(arr.shape[0]) for arr in lines_per_kf]
  total = sum(partition)
  if total == 0:
    return
  strips = [line.astype(np.float32) for arr in lines_per_kf for line in arr]
  _send(
    emit,
    entity_path,
    indexes=indexes,
    columns=rr.LineStrips3D.columns(
      strips=strips,
      colors=_solid_colors(color, total),
      radii=np.full(total, 0.0008, dtype=np.float32),
    ).partition(partition),
  )


def _send_lens_columns(
  emit: Sink,
  entity_path: str,
  metrics_path: str,
  keypoints: list[Float64Array] | None,
  index: _OverlayIndex,
  metric_indexes: list[rr.TimeColumn],
  color: tuple[int, int, int],
) -> None:
  if not keypoints:
    return
  _send_point_columns(emit, f"{entity_path}/keypoints", keypoints, index, 1.5, color)
  counts = np.asarray([kp.shape[0] for kp in keypoints], dtype=np.float64)
  _send(
    emit,
    metrics_path,
    indexes=metric_indexes,
    columns=rr.Scalars.columns(scalars=counts),
  )


@dataclass
class TrackerVizData:
  """One tracker's Rerun payload from ``build_run_outputs`` for ``solution_chunks``."""

  name: str
  body_poses: list[gtsam.Pose3] = field(default_factory=list)
  keyframe_times: Float64Array | None = None
  keyframe_rows: npt.NDArray[np.int64] | None = None
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
  # Per-keyframe ``(n, 2, 3)`` lens-to-corner rays, one per firing reprojection factor.
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
  """Per-keyframe gripper keypoints given a keypoint offset, else raw body positions."""
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


def _send_anchor(
  emit: Sink, entity: str, world_T_anchor: gtsam.Pose3, tag_size: float
) -> None:
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
  _static(
    emit,
    f"world/{entity}",
    _pose_columns(world_T_anchor),
    rr.LineStrips3D.columns(strips=[square], colors=[_YELLOW], radii=[0.003]),
  )
  _static(
    emit,
    f"world/{entity}/frame",
    rr.TransformAxes3D.columns(axis_length=[half]),
  )


def _dense_indexes(
  times: Float64Array, frame_period: float | None
) -> list[rr.TimeColumn]:
  """Session time, plus each sample's keyframe bucket when ``frame_period`` is given."""
  columns = [rr.TimeColumn(_TIMELINE, duration=times)]
  if frame_period is not None:
    stride = schema.keyframe_stride(1.0 / frame_period)
    rows = np.round(times / frame_period).astype(np.int64)
    columns.append(
      rr.TimeColumn(
        schema.KEYFRAME_TIMELINE,
        sequence=schema.keyframe_index(rows, stride),
      )
    )
  return columns


def _log_tracker(
  track: Sink,
  viz: Sink,
  data: TrackerVizData,
  spatial_root: str,
  image_root: str,
  metrics_suffix: str,
  frame_prefix: str | None,
  urdf_path: Path | None,
  anchor_tag_ids: set[int],
  mask_pad_px: int,
  trajectory_color: tuple[int, int, int],
  timeline_start: float | None,
  frame_period: float | None,
) -> None:
  """Log every per-tracker entity; the caller logs the shared statics."""
  pose_translations = _translations(data.body_poses)
  traj = _gripper_points(data)

  _send_static_points(
    viz, f"{spatial_root}/landmarks/front", data.front_points, _FRONT_COLOR
  )
  _send_static_points(
    viz, f"{spatial_root}/landmarks/rear", data.rear_points, _REAR_COLOR
  )
  _static(
    track,
    f"{spatial_root}/body",
    rr.TransformAxes3D.columns(axis_length=[_BODY_AXIS_LENGTH]),
  )
  _static(
    viz,
    f"{spatial_root}/body/marker",
    rr.Points3D.columns(
      positions=np.zeros((1, 3)), radii=[0.002], colors=[(240, 80, 80)]
    ),
  )
  for lens_name, lens_pose, lens_mei in (
    ("front", data.front_lens_extrinsic, data.front_lens_mei),
    ("rear", data.rear_lens_extrinsic, data.rear_lens_mei),
  ):
    if lens_pose is None:
      continue
    lens_columns: list[Any] = [
      _pose_columns(lens_pose),
      rr.TransformAxes3D.columns(axis_length=[_BODY_AXIS_LENGTH]),
    ]
    if lens_mei is not None:
      mei = cast("dict[str, Any]", lens_mei)
      # The MEI field names are data, so the call is keyed dynamically.
      any_values = cast("Any", rr.AnyValues)
      lens_columns.append(any_values.columns(**{k: [v] for k, v in mei.items()}))
    _static(track, f"{spatial_root}/body/{lens_name}_lens", *lens_columns)
  if data.body_T_base_tag is not None:
    _static(
      track,
      schema.base_tag_entity(data.name),
      _pose_columns(data.body_T_base_tag),
      rr.TransformAxes3D.columns(axis_length=[0.02]),
    )

  urdf_root_frame: str | None = None
  tree: rr.urdf.UrdfTree | None = None
  if urdf_path is not None:
    try:
      tree = rr.urdf.UrdfTree.from_file_path(
        str(urdf_path),
        entity_path_prefix=spatial_root,
        frame_prefix=frame_prefix,
        static_transform_entity_path=f"{spatial_root}/tf_static",
      )
      links = [tree.root_link(), *map(tree.get_joint_child, tree.joints())]
      collision = [p for link in links for p in tree.get_collision_geometry_paths(link)]
      drawn = {p for link in links for p in tree.get_visual_geometry_paths(link)}
      drawn.add(f"/{spatial_root.lstrip('/')}/tf_static")
      model_chunks = tree.stream().drop(content=collision).to_chunks()
      root_frame = (
        f"{frame_prefix}{tree.root_link().name}"
        if frame_prefix
        else tree.root_link().name
      )
    except Exception as exc:  # noqa: BLE001 - the URDF model is optional; keep the rest
      traceback.print_exc()
      print(f"  [viz] URDF model skipped ({exc}); logging the trajectory without it")
      tree = None
    else:
      for chunk in model_chunks:
        # The tree names its own paths, so its meshes are recognised by entity path.
        (viz if chunk.entity_path in drawn else track)(chunk)
      urdf_root_frame = root_frame
      _static(
        track,
        f"{spatial_root}/gripper",
        rr.Transform3D.columns(parent_frame=["world"], child_frame=[urdf_root_frame]),
        rr.TransformAxes3D.columns(axis_length=[_BODY_AXIS_LENGTH]),
      )

  n = len(data.body_poses)
  if data.keyframe_times is not None:
    trail_times = np.asarray(data.keyframe_times, dtype=np.float64)
    times = rr.TimeColumn(_TIMELINE, duration=trail_times)
  else:
    trail_times = np.arange(n, dtype=np.float64)
    times = rr.TimeColumn(_TIMELINE, sequence=np.arange(n, dtype=np.int64))

  keyframe_column: rr.TimeColumn | None = None
  if data.keyframe_rows is not None and frame_period is not None:
    stride = schema.keyframe_stride(1.0 / frame_period)
    keyframe_column = rr.TimeColumn(
      schema.KEYFRAME_TIMELINE,
      sequence=schema.keyframe_index(data.keyframe_rows, stride),
    )
  kf_indexes = [times] if keyframe_column is None else [times, keyframe_column]
  overlay_index = _overlay_index(
    trail_times if data.keyframe_times is not None else None,
    frame_period,
    kf_indexes,
    keyframe_column,
  )
  hr: tuple[Float64Array, Float64Array, Float64Array] | None = None
  if (
    data.highrate_times is not None
    and data.highrate_positions is not None
    and data.highrate_quats is not None
  ):
    hr = (data.highrate_times, data.highrate_positions, data.highrate_quats)
  kf_quats = _pose_quats(data.body_poses)
  if hr is not None:
    hr_times, hr_positions, hr_quats = hr
    _send(
      track,
      f"{spatial_root}/body",
      indexes=[rr.TimeColumn(_TIMELINE, duration=hr_times)],
      columns=rr.Transform3D.columns(translation=hr_positions, quaternion=hr_quats),
    )
    if keyframe_column is not None:
      # Only the exact keyframe states go on the review timeline.
      _send(
        track,
        f"{spatial_root}/body",
        indexes=[keyframe_column],
        columns=rr.Transform3D.columns(
          translation=pose_translations, quaternion=kf_quats
        ),
      )
  else:
    _send(
      track,
      f"{spatial_root}/body",
      indexes=kf_indexes,
      columns=rr.Transform3D.columns(
        translation=pose_translations, quaternion=kf_quats
      ),
    )
  # The exact BA keyframe poses, kept beside the propagated stream.
  _send(
    track,
    schema.body_keyframes_entity(data.name),
    indexes=kf_indexes,
    columns=rr.Transform3D.columns(translation=pose_translations, quaternion=kf_quats),
  )
  velocity: tuple[rr.TimeColumn, Float64Array] | None = None
  if hr is not None and data.highrate_vels is not None:
    velocity = (rr.TimeColumn(_TIMELINE, duration=hr[0]), data.highrate_vels)
  elif data.velocities is not None:
    velocity = (times, np.stack(data.velocities))
  if velocity is not None:
    velocity_times, velocity_values = velocity
    _send(
      track,
      f"{spatial_root}/body",
      indexes=[velocity_times],
      columns=rr.AnyValues.columns(velocity=_vec3_column(velocity_values)),
    )
    _send(
      viz,
      f"metrics/speed/body{metrics_suffix}",
      indexes=[velocity_times],
      columns=rr.Scalars.columns(scalars=np.linalg.norm(velocity_values, axis=1)),
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
      _send(
        track,
        f"{spatial_root}/gripper",
        indexes=[rr.TimeColumn(_TIMELINE, duration=hr_times)],
        columns=rr.Transform3D.columns(
          translation=gripper_t,
          quaternion=matrices_to_quats(gripper_R),
          parent_frame=["world"] * count,
          child_frame=[urdf_root_frame] * count,
        ),
      )
      if keyframe_column is not None:
        _send(
          track,
          f"{spatial_root}/gripper",
          indexes=[keyframe_column],
          columns=rr.Transform3D.columns(
            translation=_translations(poses),
            quaternion=_pose_quats(poses),
            parent_frame=["world"] * len(poses),
            child_frame=[urdf_root_frame] * len(poses),
          ),
        )
    else:
      _send(
        track,
        f"{spatial_root}/gripper",
        indexes=kf_indexes,
        columns=rr.Transform3D.columns(
          translation=_translations(poses),
          quaternion=_pose_quats(poses),
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
          # The pinned frame-rate series passes through the solved keyframe angles.
          f_times, f_angles = frame_series
          # Pad transforms, not angles, back to the merged timeline's start.
          t_times, t_angles = f_times, f_angles
          if (
            timeline_start is not None and len(f_times) and timeline_start < f_times[0]
          ):
            t_times = np.concatenate(([timeline_start], f_times))
            t_angles = np.concatenate(([f_angles[0]], f_angles))
          _send(
            track,
            f"{spatial_root}/{joint_name}",
            indexes=[rr.TimeColumn(_TIMELINE, duration=t_times)],
            columns=joint.compute_transform_columns([float(a) for a in t_angles]),
          )
          if keyframe_column is not None:
            first = next((float(v) for v in raw_values if np.isfinite(v)), 0.0)
            held: list[float] = []
            last_angle = first
            for v in raw_values:
              if np.isfinite(v):
                last_angle = float(v)
              held.append(last_angle)
            _send(
              track,
              f"{spatial_root}/{joint_name}",
              indexes=[keyframe_column],
              columns=joint.compute_transform_columns(held),
            )
          _send(
            track,
            f"{spatial_root}/{joint_name}",
            indexes=[rr.TimeColumn(_TIMELINE, duration=f_times)],
            columns=rr.AnyValues.columns(angle=np.asarray(f_angles, dtype=np.float64)),
          )
          continue
        first_finite = next((float(v) for v in raw_values if np.isfinite(v)), None)
        if first_finite is None:
          continue
        # Hold each finite angle through dropouts, seeded from the first finite value.
        clean: list[float] = []
        last = first_finite
        for v in raw_values:
          if np.isfinite(v):
            last = float(v)
          clean.append(last)
        if (
          timeline_start is not None
          and data.keyframe_times is not None
          and len(trail_times)
          and timeline_start < trail_times[0]
        ):
          joint_index = rr.TimeColumn(
            _TIMELINE, duration=np.concatenate(([timeline_start], trail_times))
          )
          clean = [clean[0], *clean]
        else:
          joint_index = times
        _send(
          track,
          f"{spatial_root}/{joint_name}",
          indexes=[joint_index],
          columns=joint.compute_transform_columns(clean),
        )
        _send(
          track,
          f"{spatial_root}/{joint_name}",
          indexes=kf_indexes,
          columns=rr.AnyValues.columns(angle=np.asarray(raw_values, dtype=np.float64)),
        )
  if data.gripper_opening is not None:
    opening_times, opening_values = data.gripper_opening
    _send(
      track,
      f"{spatial_root}/gripper_opening",
      indexes=_dense_indexes(opening_times, frame_period),
      columns=rr.Scalars.columns(scalars=opening_values),
    )
  # Trajectory: the propagated series, or the keyframe polyline without an IMU.
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
  _send(
    viz,
    f"{spatial_root}/trajectory",
    indexes=[traj_times],
    columns=rr.LineStrips3D.columns(
      strips=_trail_strips(traj_points, traj_trail_times, _TRAJECTORY_TRAIL_DURATION),
      radii=np.full(len(traj_points), 0.002, dtype=np.float32),
      colors=_solid_colors(trajectory_color, len(traj_points)),
    ),
  )
  _static(
    viz,
    f"{spatial_root}/trajectory",
    rr.LineStrips3D.columns(radii=[0.002], colors=[trajectory_color]),
  )
  _send(
    viz,
    f"metrics/distance_from_start{metrics_suffix}",
    indexes=[traj_times],
    columns=rr.Scalars.columns(
      scalars=np.linalg.norm(traj_points - traj_points[0:1], axis=1)
    ),
  )

  if data.anchor_obs_lines is not None:
    _send_factor_line_columns(
      viz,
      f"{spatial_root}/factor_lines/anchor",
      data.anchor_obs_lines,
      kf_indexes,
      _YELLOW,
    )
  if data.cross_tracker_lines is not None:
    _send_factor_line_columns(
      viz,
      f"{spatial_root}/factor_lines/cross_tracker",
      data.cross_tracker_lines,
      kf_indexes,
      _CYAN,
    )

  # Landmarks keep their 3D-scene colors; the measured 2D keypoints are magenta.
  for lens_name, kps, color in (
    ("front", data.front_keypoints, _MAGENTA),
    ("rear", data.rear_keypoints, _MAGENTA),
  ):
    _send_lens_columns(
      viz,
      f"{image_root}/{lens_name}",
      f"metrics/n_keypoints/{lens_name}{metrics_suffix}",
      kps,
      overlay_index,
      kf_indexes,
      color,
    )
  for lens_name, tags in (
    ("front", data.front_tag_detections),
    ("rear", data.rear_tag_detections),
  ):
    if tags is None:
      continue
    _send_lens_tag_columns(
      viz,
      f"{image_root}/{lens_name}",
      tags,
      overlay_index,
      anchor_tag_ids,
      mask_pad_px,
    )
  for lens_name, rejected in (
    ("front", data.front_rejected_keypoints),
    ("rear", data.rear_rejected_keypoints),
  ):
    if rejected is None:
      continue
    _send_point_columns(
      viz,
      f"{image_root}/{lens_name}/keypoints_rejected",
      rejected,
      overlay_index,
      1.5,
      _RED,
    )
  for lens_name, reproj, color in (
    ("front", data.front_landmark_reprojections, _FRONT_COLOR),
    ("rear", data.rear_landmark_reprojections, _REAR_COLOR),
  ):
    if reproj is None:
      continue
    _send_point_columns(
      viz,
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
    _send_tag_axes_columns(viz, f"{image_root}/{lens_name}", axes, overlay_index)
  if data.rear_static_mask_rects:
    _log_static_mask_rects(viz, f"{image_root}/rear", data.rear_static_mask_rects)
  for metric_name, biases in (
    ("gyro_bias", data.gyro_biases),
    ("accel_bias", data.accel_biases),
  ):
    if biases is None:
      continue
    for axis, axis_name in enumerate("xyz"):
      _send(
        viz,
        f"metrics/{metric_name}/{axis_name}{metrics_suffix}",
        indexes=kf_indexes,
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
    _send(
      viz,
      f"metrics/rmse/{lens_name}{metrics_suffix}",
      indexes=kf_indexes,
      columns=rr.Scalars.columns(scalars=np.asarray(rmse, dtype=np.float64)),
    )


_TRAJECTORY_PALETTE = (
  _FRONT_COLOR,
  _REAR_COLOR,
  (130, 140, 240),
  (220, 90, 200),
  _CYAN,
)


def _hidden_overlay_overrides(
  tracker: str, lens: str
) -> dict[rr.datatypes.EntityPath | str, rrb.EntityBehavior]:
  """Visibility overrides hiding image overlays; keys must be exact leaf entities."""
  leaves = (
    "keypoints",
    "keypoints_rejected",
    "landmark_reprojection",
    "tags",
    "tags/labels",
    "tags/masked",
    "tags/axes",
    "static_mask",
  )
  return {
    f"/image/{tracker}/{lens}/{leaf}": rrb.EntityBehavior(visible=False)
    for leaf in leaves
  }


def blueprint_for_names(
  tracker_names: list[str], floor_z: float | None = None
) -> rrb.Blueprint:
  """Multi-tracker blueprint from names alone: one image-pair column per tracker."""
  line_grid = (
    rrb.LineGrid3D(
      plane=rr.datatypes.Plane3D(normal=(0.0, 0.0, 1.0), distance=float(floor_z))
    )
    if floor_z is not None
    else None
  )
  image_columns = [
    rrb.Vertical(
      rrb.Spatial2DView(
        name=f"{n}/front",
        origin=f"/image/{n}/front",
        overrides=_hidden_overlay_overrides(n, "front"),
      ),
      rrb.Spatial2DView(
        name=f"{n}/rear",
        origin=f"/image/{n}/rear",
        overrides=_hidden_overlay_overrides(n, "rear"),
      ),
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
          overrides={
            f"/{n}/landmarks/{lens}": rrb.EntityBehavior(visible=False)
            for n in tracker_names
            for lens in ("front", "rear")
          },
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
    rrb.TimePanel(state=rrb.PanelState.Collapsed, fps=10),
  )


def segment_card_blueprint(tracker_names: list[str]) -> rrb.Blueprint:
  """Segment-table card whose 3D view drops images, landmark clouds and factor lines."""
  contents = ["+ /**", "- /image/**"]
  for name in tracker_names:
    contents.append(f"- /{name}/landmarks/**")
    contents.append(f"- /{name}/factor_lines/**")
  return rrb.Blueprint(
    rrb.Spatial3DView(name="3D", origin="/", contents=contents),
    rrb.BlueprintPanel(state=rrb.PanelState.Collapsed),
    rrb.SelectionPanel(state=rrb.PanelState.Collapsed),
    rrb.TimePanel(state=rrb.PanelState.Collapsed, fps=10),
  )


def run_blueprint(trackers: list[TrackerVizData]) -> rrb.Blueprint:
  """A solved run's dashboard, its floor grid at the lowest trajectory point."""
  gripper_points = [_gripper_points(t) for t in trackers]
  floor_zs = [float(pts[:, 2].min()) for pts in gripper_points if len(pts)]
  floor_z = min(floor_zs) if floor_zs else None
  return blueprint_for_names([t.name for t in trackers], floor_z)


def solution_chunks(
  track: Sink,
  viz: Sink,
  trackers: list[TrackerVizData],
  anchor_poses_world: dict[int, gtsam.Pose3],
  reference_tag_id: int,
  anchor_tag_size: float,
  mask_pad_px: int,
  urdf_path: Path | None,
  timeline_start: float | None = None,
  frame_period: float | None = None,
) -> None:
  """Every entity of a solved run as chunks, sent to the ``track`` or ``viz`` sink."""
  _static(
    track,
    "world",
    rr.ViewCoordinates.columns(xyz=[rr.components.ViewCoordinates.RIGHT_HAND_Z_UP]),
    rr.Transform3D.columns(child_frame=["world"]),
  )
  for tag_id, pose in anchor_poses_world.items():
    entity = "anchor_tag" if tag_id == reference_tag_id else f"anchor_tag_{tag_id}"
    _send_anchor(track, entity, pose, anchor_tag_size)
  for i, t in enumerate(trackers):
    _log_tracker(
      track,
      viz,
      t,
      spatial_root=t.name,
      image_root=schema.image_root(t.name),
      metrics_suffix=f"/{t.name}",
      frame_prefix=f"{t.name}_",
      urdf_path=urdf_path,
      anchor_tag_ids=set(anchor_poses_world),
      mask_pad_px=mask_pad_px,
      trajectory_color=_TRAJECTORY_PALETTE[i % len(_TRAJECTORY_PALETTE)],
      timeline_start=timeline_start,
      frame_period=frame_period,
    )
