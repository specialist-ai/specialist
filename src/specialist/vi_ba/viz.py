"""Columnar Rerun chunks for dual-lens VI-BA. ``solution_chunks`` takes one
``TrackerVizData`` per tracker and builds every entity a solved run has (poses,
RMSE, biases, keypoints, tags, reprojections, URDF, ...) under that tracker's
entity-path roots, handing each chunk to a sink as it is made; the layer writer
in ``specialist.record.layer`` decides which file each one belongs in."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gtsam
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from rerun.experimental import Chunk

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
# The color each lens's map and reprojection overlay are drawn in.
_LENS_COLORS = {"front": _FRONT_COLOR, "rear": _REAR_COLOR}

__all__ = [
  "LensVizData",
  "TrackerVizData",
  "blueprint_for_names",
  "solution_chunks",
  "run_blueprint",
]


Sink = Callable[[Chunk], None]


def _clock(times: Float64Array) -> rr.TimeColumn:
  """The session clock at the given sample times."""
  return rr.TimeColumn(_TIMELINE, duration=times)


def _send(emit: Sink, entity_path: str, indexes: Any, columns: Any) -> None:
  """One temporal chunk, shaped exactly as the columnar log call was."""
  emit(Chunk.from_columns(entity_path, indexes=indexes, columns=columns))


def _static(emit: Sink, entity_path: str, *columns: Any) -> None:
  """One static chunk carrying every archetype logged at that entity."""
  emit(
    Chunk.from_columns(
      entity_path, indexes=[], columns=[c for group in columns for c in group]
    )
  )


def _send_scalars(
  emit: Sink, entity_path: str, indexes: list[rr.TimeColumn], values: Any
) -> None:
  """One unnamed scalar series at the entity that reads it."""
  _send(emit, entity_path, indexes, rr.Scalars.columns(scalars=values))


def _send_series(
  emit: Sink,
  entity_path: str,
  indexes: list[rr.TimeColumn],
  values: Any,
  names: list[str],
) -> None:
  """A metric's series at the entity its plot view reads, one column of values
  per name. The names go in as a single row: an unpartitioned column would make
  each name a timestamp of its own and leave only the first standing."""
  _send_scalars(emit, entity_path, indexes, values)
  _static(
    emit, entity_path, rr.SeriesLines.columns(names=names).partition([len(names)])
  )


def _send_angle(emit: Sink, entity_path: str, indexes: Any, angles: Any) -> None:
  """A joint's angle series, beside the transform that angle drives."""
  values = np.asarray(angles, dtype=np.float64)
  _send(emit, entity_path, indexes, rr.AnyValues.columns(angle=values))


def _pose_columns(pose: gtsam.Pose3) -> Any:
  """A single-row Transform3D column list for one pose."""
  return rr.Transform3D.columns(
    translation=np.asarray(pose.translation(), dtype=np.float64)[None, :],
    quaternion=quat_xyzw(pose.rotation())[None, :],
  )


def _frame_columns(
  translations: Float64Array, quats: Float64Array, child_frame: str
) -> Any:
  """A world-parented Transform3D column list, one row per sample."""
  return rr.Transform3D.columns(
    translation=translations,
    quaternion=quats,
    parent_frame=["world"] * len(translations),
    child_frame=[child_frame] * len(translations),
  )


def _vec3_column(values: Float64Array) -> pa.FixedSizeListArray:
  """An ``(n, 3)`` float array as the arrow column AnyValues can carry."""
  flat = pa.array(np.ascontiguousarray(values, dtype=np.float64).reshape(-1))
  return pa.FixedSizeListArray.from_arrays(flat, 3)


def _translations(poses: list[gtsam.Pose3]) -> Float64Array:
  return np.asarray([np.asarray(p.translation(), dtype=np.float64) for p in poses])


def _pose_quats(poses: list[gtsam.Pose3]) -> Float64Array:
  return np.stack([quat_xyzw(p.rotation()) for p in poses])


def _trail_segments(traj: Float64Array, times: Float64Array, duration: float) -> Any:
  """One segment per sample, from the previous position to this one; the
  view's trailing time range is what accumulates them into a path. A step
  that outruns the window starts where it ends, so a break in the samples
  leaves a break in the path."""
  previous = np.concatenate([traj[:1], traj[:-1]])
  outran = np.concatenate(([0.0], np.diff(times))) > duration
  previous[outran] = traj[outran]
  return np.stack([previous, traj], axis=1)


def _log_static_points(
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


# Image-overlay indexes: the data rows' time columns, and the clearing rows'
# (``None`` when unbounded). Data rows may carry the ``keyframe`` review
# timeline besides ``video_time``; clearing rows never do, so the review
# timeline sees only real overlays.
_OverlayIndex = tuple[list[rr.TimeColumn], list[rr.TimeColumn] | None]


def _overlay_index(
  trail_times: Float64Array | None,
  frame_period: float | None,
  fallback: list[rr.TimeColumn],
  keyframe_column: rr.TimeColumn | None,
) -> _OverlayIndex:
  """Indexes that bound each image overlay to its own video frame. An overlay
  is computed from one keyframe's frame, but the video runs at the full frame
  rate, so latest-at would otherwise hold it over the following frames where
  the camera has moved on. An empty row one frame period after each keyframe
  clears it; without a known frame period the plain keyframe index is used."""
  if trail_times is None or frame_period is None:
    return fallback, None
  data = [_clock(trail_times)]
  if keyframe_column is not None:
    data.append(keyframe_column)
  return data, [_clock(trail_times + frame_period)]


def _rows(archetype: Any, columns: dict[str, Any], partition: list[int]) -> Any:
  """One image overlay's columns, split into a row per keyframe."""
  return archetype.columns(**columns).partition(partition)


def _cleared(archetype: Any, columns: dict[str, Any], n: int) -> Any:
  """The rows that clear an overlay: its archetype with every column it carries
  emptied, one blank row per keyframe."""
  return _rows(archetype, dict.fromkeys(columns, []), [0] * n)


def _send_overlay(
  emit: Sink,
  entity_path: str,
  index: _OverlayIndex,
  archetype: Any,
  partition: list[int],
  **columns: Any,
) -> None:
  """One image overlay: the rows each keyframe produced, then a row of empty
  columns one frame period later, which clears the overlay off the frames that
  keyframe does not cover."""
  data_indexes, clear_indexes = index
  _send(emit, entity_path, data_indexes, _rows(archetype, columns, partition))
  if clear_indexes is not None:
    n = len(partition)
    _send(emit, entity_path, clear_indexes, _cleared(archetype, columns, n))


def _send_lens_tag_columns(
  emit: Sink,
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
  data_indexes, clear_indexes = index
  n = len(partition)
  color_arr = np.stack(colors)
  tags_path = f"{entity_prefix}/tags"
  labels_path = f"{tags_path}/labels"
  tag_columns = {
    "strips": strips,
    "colors": color_arr,
    "radii": np.full(total, 5.0, dtype=np.float32),
  }
  label_columns = {
    "positions": np.stack(centers),
    "labels": labels,
    "show_labels": [True] * total,
    "colors": color_arr,
    "radii": np.full(total, 10.0, dtype=np.float32),
  }
  _send(emit, tags_path, data_indexes, _rows(rr.LineStrips2D, tag_columns, partition))
  _send(emit, labels_path, data_indexes, _rows(rr.Points2D, label_columns, partition))
  if clear_indexes is not None:
    _send(emit, tags_path, clear_indexes, _cleared(rr.LineStrips2D, tag_columns, n))
    _send(emit, labels_path, clear_indexes, _cleared(rr.Points2D, label_columns, n))
  total_masked = sum(mask_partition)
  if total_masked == 0:
    return
  _send_overlay(
    emit,
    f"{tags_path}/masked",
    index,
    rr.Boxes2D,
    mask_partition,
    centers=np.stack(mask_centers),
    half_sizes=np.stack(mask_half_sizes),
    colors=_solid_colors(_RED, total_masked),
    labels=["MASKED"] * total_masked,
    show_labels=[True] * total_masked,
  )


def _log_static_mask_rects(
  emit: Sink, entity_prefix: str, rects: list[tuple[int, int, int, int]]
) -> None:
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


def _send_pixel_columns(
  emit: Sink,
  entity_path: str,
  pixels_per_kf: list[Float64Array],
  index: _OverlayIndex,
  radius: float,
  color: tuple[int, int, int],
) -> None:
  partition = [int(kp.shape[0]) for kp in pixels_per_kf]
  total = sum(partition)
  if total == 0:
    return
  _send_overlay(
    emit,
    entity_path,
    index,
    rr.Points2D,
    partition,
    positions=np.concatenate([kp for kp in pixels_per_kf if kp.shape[0] > 0]),
    radii=np.full(total, radius, dtype=np.float32),
    colors=_solid_colors(color, total),
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
  _send_overlay(
    emit,
    f"{entity_prefix}/tags/axes",
    index,
    rr.LineStrips2D,
    partition,
    strips=strips,
    colors=np.stack(colors),
    radii=np.full(total, 4.0, dtype=np.float32),
  )


def _send_factor_line_columns(
  emit: Sink,
  entity_path: str,
  lines_per_kf: list[Float64Array],
  indexes: list[rr.TimeColumn],
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
  _send(
    emit,
    entity_path,
    indexes,
    rr.LineStrips3D.columns(
      strips=strips,
      colors=_solid_colors(color, total),
      radii=np.full(total, 0.0008, dtype=np.float32),
    ).partition(partition),
  )


def _send_lens_columns(
  emit: Sink,
  entity_path: str,
  keypoints: list[Float64Array] | None,
  index: _OverlayIndex,
  color: tuple[int, int, int],
) -> None:
  if not keypoints:
    return
  _send_pixel_columns(emit, f"{entity_path}/keypoints", keypoints, index, 1.5, color)


@dataclass
class LensVizData:
  """One lens's share of a tracker's payload: the map it contributed, its
  per-keyframe image overlays, and the calibration it was solved with."""

  points: dict[int, Float64Array] = field(default_factory=dict)
  keypoints: list[Float64Array] | None = None
  rejected_keypoints: list[Float64Array] | None = None
  landmark_reprojections: list[Float64Array] | None = None
  tag_detections: list[list[tuple[int, Float64Array, Float64Array]]] | None = None
  tag_axes: list[Float64Array] | None = None
  rmse_per_kf: list[float] | None = None
  extrinsic: gtsam.Pose3 | None = None
  # MEI intrinsics at decode resolution, logged as a lens static.
  mei: dict[str, float] | None = None
  static_mask_rects: list[tuple[int, int, int, int]] | None = None


@dataclass
class TrackerVizData:
  """One tracker's Rerun payload, built by ``viz_build.build_run_outputs`` and
  turned into chunks by ``solution_chunks``."""

  name: str
  # The session clock every stream here is indexed on.
  keyframe_times: Float64Array
  body_poses: list[gtsam.Pose3] = field(default_factory=list)
  keyframe_rows: npt.NDArray[np.int64] | None = None
  # Per-lens payloads, keyed by the lens name their entity paths carry.
  lenses: dict[str, LensVizData] = field(default_factory=dict)
  velocities: list[Float64Array] | None = None
  gyro_biases: list[Float64Array] | None = None
  accel_biases: list[Float64Array] | None = None
  urdf_root_poses: list[gtsam.Pose3] = field(default_factory=list)
  # URDF joint name → per-keyframe angle, NaN where it could not be recovered.
  joint_angles: dict[str, list[float]] | None = None
  trajectory_offset_in_gripper: tuple[float, float, float] | None = None
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


def _gripper_points(data: TrackerVizData) -> Float64Array:
  """Per-keyframe trajectory points: the URDF root translated by the gripper
  keypoint offset when an offset is given and root poses exist, else the raw
  body translations."""
  if data.trajectory_offset_in_gripper is not None and data.urdf_root_poses:
    offset = np.asarray(data.trajectory_offset_in_gripper, dtype=np.float64)
    return np.stack(
      [
        np.asarray(p.translation(), dtype=np.float64)
        + np.asarray(p.rotation().matrix(), dtype=np.float64) @ offset
        for p in data.urdf_root_poses
      ]
    )
  return _translations(data.body_poses)


def _held(values: list[float]) -> list[float] | None:
  """Each finite angle held through the dropouts that follow it, seeded from
  the first finite value so leading-NaN keyframes don't snap the URDF to zero;
  ``None`` when no value is finite at all."""
  last = next((float(v) for v in values if np.isfinite(v)), None)
  if last is None:
    return None
  held: list[float] = []
  for v in values:
    if np.isfinite(v):
      last = float(v)
    held.append(last)
  return held


def _send_joint_transform(
  emit: Sink,
  entity_path: str,
  joint: Any,
  times: Float64Array,
  angles: Any,
  timeline_start: float | None,
) -> None:
  """One joint's transform stream, padded back to the merged timeline's start
  so the joint frames resolve before the first sample the solve covers: a base
  layer's video can begin earlier than the solve."""
  if timeline_start is not None and len(times) and timeline_start < times[0]:
    times = np.concatenate(([timeline_start], times))
    angles = np.concatenate((angles[:1], angles))
  _send(
    emit,
    entity_path,
    [_clock(times)],
    joint.compute_transform_columns([float(a) for a in angles]),
  )


def _log_anchor(
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
  """A dense series' clocks: the session time always, and each sample's
  review-timeline bucket beside it when the frame clock is known, so a
  keyframe-indexed join reaches the dense fields too."""
  columns = [rr.TimeColumn(_TIMELINE, duration=times)]
  if frame_period is not None:
    stride = record_schema.keyframe_stride(1.0 / frame_period)
    rows = np.round(times / frame_period).astype(np.int64)
    columns.append(
      rr.TimeColumn(
        record_schema.KEYFRAME_TIMELINE,
        sequence=record_schema.keyframe_index(rows, stride),
      )
    )
  return columns


@dataclass(frozen=True)
class _Indexes:
  """The timeline vocabulary one tracker's streams are indexed on: the session
  clock at its keyframes, the review timeline when the keyframe rows are known,
  the pair of indexes each image overlay is bounded by, and the IMU-propagated
  body series a run with an IMU carries."""

  trail_times: Float64Array
  times: rr.TimeColumn
  keyframe_column: rr.TimeColumn | None
  keyframes: list[rr.TimeColumn]
  overlay: _OverlayIndex
  highrate: tuple[Float64Array, Float64Array, Float64Array] | None


def _tracker_indexes(data: TrackerVizData, frame_period: float | None) -> _Indexes:
  """Every index a tracker's streams share, derived once from its keyframes."""
  trail_times = np.asarray(data.keyframe_times, dtype=np.float64)
  times = _clock(trail_times)
  keyframe_column: rr.TimeColumn | None = None
  if data.keyframe_rows is not None and frame_period is not None:
    stride = record_schema.keyframe_stride(1.0 / frame_period)
    keyframe_column = rr.TimeColumn(
      record_schema.KEYFRAME_TIMELINE,
      sequence=record_schema.keyframe_index(data.keyframe_rows, stride),
    )
  keyframes = [times] if keyframe_column is None else [times, keyframe_column]
  highrate = None
  if (
    data.highrate_times is not None
    and data.highrate_positions is not None
    and data.highrate_quats is not None
  ):
    highrate = (data.highrate_times, data.highrate_positions, data.highrate_quats)
  return _Indexes(
    trail_times=trail_times,
    times=times,
    keyframe_column=keyframe_column,
    keyframes=keyframes,
    overlay=_overlay_index(trail_times, frame_period, keyframes, keyframe_column),
    highrate=highrate,
  )


def _tracker_statics(track: Sink, viz: Sink, data: TrackerVizData) -> None:
  """The tracker's time-independent geometry: the map each lens contributed,
  the body frame and its marker, the lenses on it, and the base tag."""
  spatial_root = data.name
  body_path = f"{spatial_root}/body"
  for lens_name, lens in data.lenses.items():
    _log_static_points(
      viz, f"{spatial_root}/landmarks/{lens_name}", lens.points, _LENS_COLORS[lens_name]
    )
  _static(track, body_path, rr.TransformAxes3D.columns(axis_length=[_BODY_AXIS_LENGTH]))
  _static(
    viz,
    f"{body_path}/marker",
    rr.Points3D.columns(
      positions=np.zeros((1, 3)), radii=[0.002], colors=[(240, 80, 80)]
    ),
  )
  for lens_name, lens in data.lenses.items():
    if lens.extrinsic is None:
      continue
    lens_columns: list[Any] = [
      _pose_columns(lens.extrinsic),
      rr.TransformAxes3D.columns(axis_length=[_BODY_AXIS_LENGTH]),
    ]
    if lens.mei is not None:
      # The MEI field names are data, so the call is keyed dynamically.
      mei: dict[str, Any] = {k: [v] for k, v in lens.mei.items()}
      lens_columns.append(rr.AnyValues.columns(**mei))
    _static(track, f"{body_path}/{lens_name}_lens", *lens_columns)
  if data.body_T_base_tag is not None:
    _static(
      track,
      record_schema.base_tag_entity(data.name),
      _pose_columns(data.body_T_base_tag),
      rr.TransformAxes3D.columns(axis_length=[0.02]),
    )


def _urdf_model(
  track: Sink, viz: Sink, data: TrackerVizData, urdf_path: Path | None
) -> tuple[rr.urdf.UrdfTree, str] | None:
  """The gripper model's static geometry and the frame naming its root, or
  ``None`` when no URDF was given or the file could not be read: the model is
  decoration, so a bad one still leaves the trajectory logged."""
  if urdf_path is None:
    return None
  spatial_root = data.name
  try:
    tree = rr.urdf.UrdfTree.from_file_path(
      str(urdf_path),
      entity_path_prefix=spatial_root,
      frame_prefix=f"{data.name}_",
      static_transform_entity_path=f"{spatial_root}/tf_static",
    )
    links = [tree.root_link(), *map(tree.get_joint_child, tree.joints())]
    collision = [p for link in links for p in tree.get_collision_geometry_paths(link)]
    drawn = {p for link in links for p in tree.get_visual_geometry_paths(link)}
    drawn.add(f"/{spatial_root.lstrip('/')}/tf_static")
    model_chunks = tree.stream().drop(content=collision).to_chunks()
    root_frame = f"{data.name}_{tree.root_link().name}"
  except Exception as exc:  # noqa: BLE001 - the URDF model is optional; keep the rest
    traceback.print_exc()
    print(
      f"  [warn] [viz] URDF model skipped ({exc}); logging the trajectory without it"
    )
    return None
  for chunk in model_chunks:
    # The tree names its own paths, so the meshes are recognised rather than
    # guessed at from the robot's name.
    (viz if chunk.entity_path in drawn else track)(chunk)
  _static(
    track,
    f"{spatial_root}/gripper",
    rr.Transform3D.columns(parent_frame=["world"], child_frame=[root_frame]),
    rr.TransformAxes3D.columns(axis_length=[_BODY_AXIS_LENGTH]),
  )
  return tree, root_frame


def _body_streams(
  track: Sink, viz: Sink, data: TrackerVizData, indexes: _Indexes
) -> None:
  """The body pose over time, the exact keyframe poses beside it, and the speed
  the velocity states imply."""
  body_path = f"{data.name}/body"
  kf_columns = rr.Transform3D.columns(
    translation=_translations(data.body_poses), quaternion=_pose_quats(data.body_poses)
  )
  if indexes.highrate is not None:
    hr_times, hr_positions, hr_quats = indexes.highrate
    _send(
      track,
      body_path,
      [_clock(hr_times)],
      rr.Transform3D.columns(translation=hr_positions, quaternion=hr_quats),
    )
    if indexes.keyframe_column is not None:
      # The review timeline shows the solver's exact per-keyframe state; the
      # smooth propagated stream stays on the video clock alone.
      _send(track, body_path, [indexes.keyframe_column], kf_columns)
  else:
    _send(track, body_path, indexes.keyframes, kf_columns)
  # The exact BA keyframe poses, kept beside the propagated stream.
  _send(
    track, record_schema.body_keyframes_entity(data.name), indexes.keyframes, kf_columns
  )
  velocity: tuple[rr.TimeColumn, Float64Array] | None = None
  if indexes.highrate is not None and data.highrate_vels is not None:
    velocity = (_clock(indexes.highrate[0]), data.highrate_vels)
  elif data.velocities is not None:
    velocity = (indexes.times, np.stack(data.velocities))
  if velocity is None:
    return
  velocity_times, velocity_values = velocity
  _send(
    track,
    body_path,
    [velocity_times],
    rr.AnyValues.columns(velocity=_vec3_column(velocity_values)),
  )
  _send_series(
    viz,
    f"metrics/speed/{data.name}",
    [velocity_times],
    np.linalg.norm(velocity_values, axis=1),
    [data.name],
  )


def _gripper_streams(
  track: Sink, data: TrackerVizData, indexes: _Indexes, root_frame: str
) -> tuple[Float64Array, Float64Array] | None:
  """The URDF root's pose over time, propagated from the body when the run has
  an IMU; the rotations and translations of that propagated stream, which the
  trajectory is offset from, or ``None`` when only keyframes were placed."""
  gripper_path = f"{data.name}/gripper"
  poses = data.urdf_root_poses
  if indexes.highrate is None or not data.body_poses or not poses:
    _send(
      track,
      gripper_path,
      indexes.keyframes,
      _frame_columns(_translations(poses), _pose_quats(poses), root_frame),
    )
    return None
  hr_times, hr_positions, hr_quats = indexes.highrate
  rel = data.body_poses[0].inverse().compose(poses[0])
  body_R = quats_to_matrices(hr_quats)
  gripper_t = hr_positions + body_R @ np.asarray(rel.translation())
  gripper_R = body_R @ np.asarray(rel.rotation().matrix())
  _send(
    track,
    gripper_path,
    [_clock(hr_times)],
    _frame_columns(gripper_t, matrices_to_quats(gripper_R), root_frame),
  )
  if indexes.keyframe_column is not None:
    _send(
      track,
      gripper_path,
      [indexes.keyframe_column],
      _frame_columns(_translations(poses), _pose_quats(poses), root_frame),
    )
  return gripper_t, gripper_R


def _joint_streams(
  track: Sink,
  data: TrackerVizData,
  indexes: _Indexes,
  tree: rr.urdf.UrdfTree,
  timeline_start: float | None,
) -> None:
  """Each solved joint's transform and angle, at the pinned frame cadence where
  one was recovered and at the keyframes otherwise."""
  if data.joint_angles is None:
    return
  for joint_name, raw_values in data.joint_angles.items():
    joint = tree.get_joint_by_name(joint_name)
    if joint is None:
      continue
    joint_path = f"{data.name}/{joint_name}"
    frame_series = (data.joint_frame_angles or {}).get(joint_name)
    if frame_series is None:
      held = _held(raw_values)
      if held is None:
        continue
      _send_joint_transform(
        track, joint_path, joint, indexes.trail_times, held, timeline_start
      )
      _send_angle(track, joint_path, indexes.keyframes, raw_values)
      continue
    # The pinned frame-rate series passes through the solved keyframe angles.
    f_times, f_angles = frame_series
    _send_joint_transform(track, joint_path, joint, f_times, f_angles, timeline_start)
    if indexes.keyframe_column is not None:
      kf_angles = _held(raw_values) or [0.0] * len(raw_values)
      _send(
        track,
        joint_path,
        [indexes.keyframe_column],
        joint.compute_transform_columns(kf_angles),
      )
    _send_angle(track, joint_path, [_clock(f_times)], f_angles)


def _trajectory_streams(
  viz: Sink,
  data: TrackerVizData,
  indexes: _Indexes,
  gripper_hr: tuple[Float64Array, Float64Array] | None,
  trajectory_color: tuple[int, int, int],
) -> None:
  """The fading trail the gripper keypoint drew, and the distance it covered.
  The trail follows the propagated series; the keyframe polyline stands in when
  a run has no IMU to propagate."""
  if indexes.highrate is not None:
    hr_times, hr_positions, _ = indexes.highrate
    if gripper_hr is not None and data.trajectory_offset_in_gripper is not None:
      offset = np.asarray(data.trajectory_offset_in_gripper, dtype=np.float64)
      gripper_t, gripper_R = gripper_hr
      points = gripper_t + gripper_R @ offset
    else:
      points = hr_positions
    times = _clock(hr_times)
    trail_times = hr_times
  else:
    points = _gripper_points(data)
    times = indexes.times
    trail_times = indexes.trail_times
  entity_path = f"{data.name}/trajectory"
  _send(
    viz,
    entity_path,
    [times],
    rr.LineStrips3D.columns(
      strips=_trail_segments(points, trail_times, _TRAJECTORY_TRAIL_DURATION)
    ),
  )
  _static(
    viz,
    entity_path,
    rr.LineStrips3D.columns(radii=[0.002], colors=[trajectory_color]),
  )
  _send_series(
    viz,
    f"metrics/distance_from_start/{data.name}",
    [times],
    np.linalg.norm(points - points[0:1], axis=1),
    [data.name],
  )


def _factor_line_streams(viz: Sink, data: TrackerVizData, indexes: _Indexes) -> None:
  """The rays of the reprojection factors that fired at each keyframe."""
  for kind, lines, color in (
    ("anchor", data.anchor_obs_lines, _YELLOW),
    ("cross_tracker", data.cross_tracker_lines, _CYAN),
  ):
    if lines is None:
      continue
    _send_factor_line_columns(
      viz, f"{data.name}/factor_lines/{kind}", lines, indexes.keyframes, color
    )


def _lens_streams(
  viz: Sink,
  data: TrackerVizData,
  indexes: _Indexes,
  anchor_tag_ids: set[int],
  mask_pad_px: int,
) -> None:
  """Every image overlay, a kind at a time across the lenses, so one kind's
  chunks land together. Landmark colors match the 3D scene; the 2D keypoint
  measurements are magenta."""
  image_root = record_schema.image_root(data.name)
  for lens_name, lens in data.lenses.items():
    _send_lens_columns(
      viz, f"{image_root}/{lens_name}", lens.keypoints, indexes.overlay, _MAGENTA
    )
  _send_lens_series(
    viz,
    f"metrics/n_keypoints/{data.name}",
    indexes.keyframes,
    data,
    lambda lens: [kp.shape[0] for kp in lens.keypoints] if lens.keypoints else None,
  )
  for lens_name, lens in data.lenses.items():
    if lens.tag_detections is not None:
      _send_lens_tag_columns(
        viz,
        f"{image_root}/{lens_name}",
        lens.tag_detections,
        indexes.overlay,
        anchor_tag_ids,
        mask_pad_px,
      )
  for lens_name, lens in data.lenses.items():
    if lens.rejected_keypoints is not None:
      _send_pixel_columns(
        viz,
        f"{image_root}/{lens_name}/keypoints_rejected",
        lens.rejected_keypoints,
        indexes.overlay,
        1.5,
        _RED,
      )
  for lens_name, lens in data.lenses.items():
    if lens.landmark_reprojections is not None:
      _send_pixel_columns(
        viz,
        f"{image_root}/{lens_name}/landmark_reprojection",
        lens.landmark_reprojections,
        indexes.overlay,
        3.0,
        _LENS_COLORS[lens_name],
      )
  for lens_name, lens in data.lenses.items():
    if lens.tag_axes is not None:
      _send_tag_axes_columns(
        viz, f"{image_root}/{lens_name}", lens.tag_axes, indexes.overlay
      )
  for lens_name, lens in data.lenses.items():
    if lens.static_mask_rects:
      _log_static_mask_rects(viz, f"{image_root}/{lens_name}", lens.static_mask_rects)


def _bias_streams(viz: Sink, data: TrackerVizData, indexes: _Indexes) -> None:
  """The IMU bias states and the per-lens reprojection error, as plots."""
  for metric_name, biases in (
    ("gyro_bias", data.gyro_biases),
    ("accel_bias", data.accel_biases),
  ):
    if biases is None:
      continue
    _send_series(
      viz,
      f"metrics/{metric_name}/{data.name}",
      indexes.keyframes,
      np.asarray(biases, dtype=np.float64),
      [f"{data.name}/{axis}" for axis in "xyz"],
    )
  _send_lens_series(
    viz,
    f"metrics/rmse/{data.name}",
    indexes.keyframes,
    data,
    lambda lens: lens.rmse_per_kf,
  )


def _log_tracker(
  track: Sink,
  viz: Sink,
  data: TrackerVizData,
  urdf_path: Path | None,
  anchor_tag_ids: set[int],
  mask_pad_px: int,
  trajectory_color: tuple[int, int, int],
  timeline_start: float | None,
  frame_period: float | None,
) -> None:
  """Log every per-tracker entity (poses, landmarks, URDF, image overlays,
  scalar metrics). Shared statics (anchor tag, view coordinates, world TF)
  must already be logged by the caller. Scalar paths are
  ``metrics/<scalar>/<tracker>`` so multi-tracker views with origin
  ``/metrics/<scalar>`` aggregate per-tracker series automatically."""
  _tracker_statics(track, viz, data)
  model = _urdf_model(track, viz, data, urdf_path)
  indexes = _tracker_indexes(data, frame_period)
  _body_streams(track, viz, data, indexes)
  gripper_hr = None
  if model is not None:
    tree, root_frame = model
    gripper_hr = _gripper_streams(track, data, indexes, root_frame)
    _joint_streams(track, data, indexes, tree, timeline_start)
  if data.gripper_opening is not None:
    opening_times, opening_values = data.gripper_opening
    _send_scalars(
      track,
      f"{data.name}/gripper_opening",
      _dense_indexes(opening_times, frame_period),
      opening_values,
    )
  _trajectory_streams(viz, data, indexes, gripper_hr, trajectory_color)
  _factor_line_streams(viz, data, indexes)
  _lens_streams(viz, data, indexes, anchor_tag_ids, mask_pad_px)
  _bias_streams(viz, data, indexes)


def _send_lens_series(
  emit: Sink,
  entity_path: str,
  indexes: list[rr.TimeColumn],
  data: TrackerVizData,
  values: Callable[[LensVizData], list[float] | list[int] | None],
) -> None:
  """One plot entity for a per-lens metric, a series per lens that has it."""
  per_lens = {name: values(lens) for name, lens in data.lenses.items()}
  present = {name: v for name, v in per_lens.items() if v}
  if not present:
    return
  _send_series(
    emit,
    entity_path,
    indexes,
    np.stack([np.asarray(v, dtype=np.float64) for v in present.values()], axis=1),
    [f"{data.name}/{name}" for name in present],
  )


_TRAJECTORY_PALETTE = (
  _FRONT_COLOR,
  _REAR_COLOR,
  (130, 140, 240),
  (220, 90, 200),
  _CYAN,
)


_PLOTTED_METRICS = (
  "distance_from_start",
  "speed",
  "n_keypoints",
  "rmse",
  "gyro_bias",
  "accel_bias",
)


def _panels() -> tuple[rrb.BlueprintPanel, rrb.SelectionPanel, rrb.TimePanel]:
  """The chrome both blueprints hide, so the two cannot drift apart."""
  return (
    rrb.BlueprintPanel(state=rrb.PanelState.Collapsed),
    rrb.SelectionPanel(state=rrb.PanelState.Collapsed),
    rrb.TimePanel(state=rrb.PanelState.Collapsed),
  )


def _trail_overrides(
  tracker_names: list[str],
) -> dict[rr.datatypes.EntityPath | str, rrb.VisibleTimeRanges]:
  """The window every view that draws a trajectory ribbon has to set."""
  return {f"/{n}/trajectory": _trail_window() for n in tracker_names}


def _hidden_overlay_overrides(
  tracker: str, lens: str
) -> dict[rr.datatypes.EntityPath | str, rrb.EntityBehavior]:
  """Image overlays start hidden, so the default view is the raw frame beside
  the 3D scene; each overlay is one visibility toggle away. A key covers the
  subtree under it, so the tag overlays need only their root."""
  roots = (
    "keypoints",
    "keypoints_rejected",
    "landmark_reprojection",
    "tags",
    "static_mask",
  )
  return {
    f"/image/{tracker}/{lens}/{root}": rrb.EntityBehavior(visible=False)
    for root in roots
  }


def blueprint_for_names(
  tracker_names: list[str], floor_z: float | None = None
) -> rrb.Blueprint:
  """Multi-tracker blueprint: one image-pair column per tracker, scalar
  plots aggregated across trackers under ``/metrics/<name>/...``. Names alone
  are enough, for consumers (like catalog registration) that have no
  ``TrackerVizData``."""
  line_grid = (
    rrb.LineGrid3D(
      plane=rr.datatypes.Plane3D(normal=(0.0, 0.0, 1.0), distance=float(floor_z))
    )
    if floor_z is not None
    else None
  )
  hidden = rrb.EntityBehavior(visible=False)
  overrides: dict[
    rr.datatypes.EntityPath | str, rrb.EntityBehavior | rrb.VisibleTimeRanges
  ] = {**_trail_overrides(tracker_names)}
  overrides.update({f"/{n}/landmarks": hidden for n in tracker_names})
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
          overrides=overrides,
          line_grid=line_grid,
        ),
        rrb.Grid(
          *(
            rrb.TimeSeriesView(name=metric, origin=f"/metrics/{metric}")
            for metric in _PLOTTED_METRICS
          ),
          name="plots",
        ),
      ),
      rrb.Horizontal(*image_columns) if len(image_columns) > 1 else image_columns[0],
      column_shares=[2, max(1, len(tracker_names))],
    ),
    *_panels(),
  )


def _trail_window() -> rrb.VisibleTimeRanges:
  """The stretch of time a trajectory ribbon is drawn over. Its samples carry
  one step each, so a view that does not set this draws a single step."""
  return rrb.VisibleTimeRanges(
    [
      rrb.VisibleTimeRange(
        _TIMELINE,
        start=rrb.TimeRangeBoundary.cursor_relative(
          seconds=-_TRAJECTORY_TRAIL_DURATION
        ),
        end=rrb.TimeRangeBoundary.cursor_relative(),
      )
    ]
  )


def segment_card_blueprint(tracker_names: list[str]) -> rrb.Blueprint:
  """Segment-table card: the solved bodies and anchors alone. The card grid
  renders every segment at once, so the video streams stay out, and so do the
  landmark clouds and factor lines, which otherwise fill the frame and shrink
  the grippers to a speck."""
  contents = ["+ /**", "- /image/**"]
  for name in tracker_names:
    contents.append(f"- /{name}/landmarks/**")
    contents.append(f"- /{name}/factor_lines/**")
  return rrb.Blueprint(
    rrb.Spatial3DView(
      name="3D",
      origin="/",
      contents=contents,
      overrides=_trail_overrides(tracker_names),
    ),
    *_panels(),
  )


def run_blueprint(trackers: list[TrackerVizData]) -> rrb.Blueprint:
  """The standard dashboard for a solved run, floor grid at the lowest
  trajectory point."""
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
  """Build every entity of a solved run as chunks, each handed to the sink for
  the layer it belongs to: ``track`` for what a consumer reads, ``viz`` for what
  only a viewer draws.

  A tracker's entities live under ``/<name>/...``, image overlays under
  ``/image/<name>/<lens>``, scalars under ``/metrics/<name>/...``. The reference
  anchor tag is logged once at ``/world/anchor_tag``; every other anchor at its
  solved pose under ``/world/anchor_tag_<id>``.
  """
  _static(
    track,
    "world",
    rr.ViewCoordinates.columns(xyz=[rr.components.ViewCoordinates.RIGHT_HAND_Z_UP]),
    rr.Transform3D.columns(child_frame=["world"]),
  )
  for tag_id, pose in anchor_poses_world.items():
    entity = "anchor_tag" if tag_id == reference_tag_id else f"anchor_tag_{tag_id}"
    _log_anchor(track, entity, pose, anchor_tag_size)
  for i, t in enumerate(trackers):
    _log_tracker(
      track,
      viz,
      t,
      urdf_path=urdf_path,
      anchor_tag_ids=set(anchor_poses_world),
      mask_pad_px=mask_pad_px,
      trajectory_color=_TRAJECTORY_PALETTE[i % len(_TRAJECTORY_PALETTE)],
      timeline_start=timeline_start,
      frame_period=frame_period,
    )
