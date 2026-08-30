"""Columnar Rerun chunks for dual-lens VI-BA. ``estimate_chunks`` takes one
``CameraEstimate`` per camera and builds every entity a solved run has (poses,
biases, gripper, landmarks, tags, reprojections, URDF, ...) under that camera's
slot, handing each chunk to a sink as it is made; the layer writer in
``specialist.record.layer`` decides which file each one belongs in.

Every path is minted by ``specialist_dataset.schema``, which is the only place
that decides what an entity is called."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gtsam
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
from rerun.experimental import Chunk
from specialist_dataset import schema as record_schema
from specialist_dataset.poses import quats_to_matrices

from specialist.vi_ba.blueprint import (
  BODY_AXIS,
  TAG_AXIS,
  TRAJECTORY_TRAIL_DURATION,
)
from specialist.vi_ba.geometry import matrices_to_quats, quat_xyzw

Float64Array = npt.NDArray[np.float64]

_FRONT_COLOR = (80, 200, 120)
_REAR_COLOR = (230, 140, 80)
_RED = (230, 60, 60)
_YELLOW = (250, 250, 50)
_MAGENTA = (255, 0, 255)
_CYAN = (90, 220, 220)
_TIMELINE = record_schema.TIMELINE
# The color each lens's map and reprojection overlay are drawn in.
_LENS_COLORS = {"front": _FRONT_COLOR, "rear": _REAR_COLOR}

__all__ = [
  "LensEstimate",
  "CameraEstimate",
  "Layers",
  "estimate_chunks",
]


Sink = Callable[[Chunk], None]


@dataclass(frozen=True)
class Layers:
  """Where a chunk goes, by what it is: a quantity someone computes with to
  ``estimate``, a way of drawing one to ``viz``."""

  estimate: Sink
  viz: Sink


def _clock(times: Float64Array) -> rr.TimeColumn:
  """The session clock at the given sample times."""
  return rr.TimeColumn(_TIMELINE, duration=times)


def _send(emit: Sink, entity_path: str, indexes: Any, columns: Any) -> None:
  """One temporal chunk, shaped exactly as the columnar log call was."""
  emit(Chunk.from_columns(entity_path, indexes=indexes, columns=columns))


def _in_world(emit: Sink, entity_path: str) -> None:
  """Declare an entity's data to be in the world frame. Goes on the entity that
  holds the data, never on a parent -- see ``schema.WORLD_FRAME``."""
  _static(
    emit, entity_path, rr.CoordinateFrame.columns(frame=[record_schema.WORLD_FRAME])
  )


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
  layers: Layers,
  entity_path: str,
  indexes: list[rr.TimeColumn],
  values: Any,
  names: list[str],
) -> None:
  """A metric's series, one column of values per name.

  The names go to the estimate layer beside the numbers and to viz as the plot
  labels, both as a single row: an unpartitioned column would make each name a
  timestamp of its own and leave only the first standing."""
  _send_scalars(layers.estimate, entity_path, indexes, values)
  _static(
    layers.estimate,
    entity_path,
    rr.AnyValues.columns(series=names).partition([len(names)]),
  )
  _static(
    layers.viz, entity_path, rr.SeriesLines.columns(names=names).partition([len(names)])
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
  """A world-parented Transform3D column list, one row per sample. Every row
  names the edge it defines."""
  return rr.Transform3D.columns(
    translation=translations,
    quaternion=quats,
    parent_frame=[record_schema.WORLD_FRAME] * len(translations),
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
  layers: Layers,
  entity_path: str,
  points: dict[int, Float64Array],
  color: tuple[int, int, int],
) -> None:
  if not points:
    return
  xyz = np.asarray(list(points.values()), dtype=np.float64)
  _static(
    layers.estimate,
    entity_path,
    rr.Points3D.columns(positions=xyz).partition([len(xyz)]),
  )
  _static(layers.viz, entity_path, rr.Points3D.columns(radii=[0.002], colors=[color]))


def _solid_colors(color: tuple[int, int, int], n: int) -> np.ndarray:
  """An ``(n, 3)`` uint8 array of one repeated RGB color."""
  return np.tile(np.asarray(color, dtype=np.uint8), (n, 1))


def _rows(archetype: Any, columns: dict[str, Any], partition: list[int]) -> Any:
  """One image overlay's columns, split into a row per keyframe."""
  return archetype.columns(**columns).partition(partition)


def _send_overlay(
  layers: Layers,
  entity_path: str,
  index: list[rr.TimeColumn],
  archetype: Any,
  partition: list[int],
  measured: dict[str, Any],
  drawn: dict[str, Any],
) -> None:
  """One image overlay: what the lens measured to the estimate layer, and the
  way of drawing it to viz. Written on the keyframe clock alone, so a
  video-rate playback never wears an overlay computed from a frame it has
  passed."""
  if measured:
    _send(layers.estimate, entity_path, index, _rows(archetype, measured, partition))
  if drawn:
    _send(layers.viz, entity_path, index, _rows(archetype, drawn, partition))


def _send_lens_tag_columns(
  layers: Layers,
  tags_path: str,
  tags_per_kf: list[list[tuple[int, Float64Array, Float64Array]]],
  index: list[rr.TimeColumn],
  anchor_tag_ids: set[int],
) -> None:
  """Per-keyframe AprilTag detections: the corner quad, the detected centre and
  the tag id to the estimate layer, the id as a label string in viz. Anchor tags
  draw yellow and the rest red."""
  yellow = np.asarray(_YELLOW, dtype=np.uint8)
  red = np.asarray(_RED, dtype=np.uint8)
  strips: list[np.ndarray] = []
  centers: list[np.ndarray] = []
  ids: list[int] = []
  labels: list[str] = []
  colors: list[np.ndarray] = []
  partition: list[int] = []
  for tags in tags_per_kf:
    partition.append(len(tags))
    for tid, center, corners in tags:
      closed = np.vstack([corners, corners[0:1]]).astype(np.float32)
      strips.append(closed)
      centers.append(np.asarray(center, dtype=np.float32))
      ids.append(int(tid))
      labels.append(str(tid))
      is_anchor = int(tid) in anchor_tag_ids
      colors.append(yellow if is_anchor else red)
  total = sum(partition)
  if total == 0:
    return
  color_arr = np.stack(colors)
  _send_overlay(
    layers,
    tags_path,
    index,
    rr.LineStrips2D,
    partition,
    measured={"strips": strips},
    drawn={"colors": color_arr, "radii": np.full(total, 5.0, dtype=np.float32)},
  )
  _send_overlay(
    layers,
    tags_path,
    index,
    rr.AnyValues,
    partition,
    measured={"tag_id": np.asarray(ids, dtype=np.int32)},
    drawn={},
  )
  _send_overlay(
    layers,
    f"{tags_path}/center",
    index,
    rr.Points2D,
    partition,
    measured={"positions": np.stack(centers)},
    drawn={
      "labels": labels,
      "show_labels": [True] * total,
      "colors": color_arr,
      "radii": np.full(total, 10.0, dtype=np.float32),
    },
  )


def _send_pixel_columns(
  layers: Layers,
  entity_path: str,
  pixels_per_kf: list[Float64Array],
  index: list[rr.TimeColumn],
  radius: float,
  color: tuple[int, int, int],
  measured: bool,
) -> None:
  """Per-keyframe image points. ``measured`` says whether the pixels are what
  the lens saw or what the solve predicts."""
  partition = [int(kp.shape[0]) for kp in pixels_per_kf]
  total = sum(partition)
  if total == 0:
    return
  positions = {
    "positions": np.concatenate([kp for kp in pixels_per_kf if kp.shape[0] > 0])
  }
  drawn = {
    "radii": np.full(total, radius, dtype=np.float32),
    "colors": _solid_colors(color, total),
  }
  _send_overlay(
    layers,
    entity_path,
    index,
    rr.Points2D,
    partition,
    measured=positions if measured else {},
    drawn=drawn if measured else {**positions, **drawn},
  )


def _send_tag_axes_columns(
  layers: Layers,
  axes_path: str,
  axes_per_kf: list[Float64Array],
  index: list[rr.TimeColumn],
) -> None:
  """The tag's pose drawn as three coloured strokes, viz only."""
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
    layers,
    axes_path,
    index,
    rr.LineStrips2D,
    partition,
    measured={},
    drawn={
      "strips": strips,
      "colors": np.stack(colors),
      "radii": np.full(total, 4.0, dtype=np.float32),
    },
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


@dataclass
class LensEstimate:
  """One lens's share of a camera's estimate: the map it contributed, and what
  it measured in its own image per keyframe."""

  points: dict[int, Float64Array] = field(default_factory=dict)
  keypoints: list[Float64Array] = field(default_factory=list)
  rejected_keypoints: list[Float64Array] = field(default_factory=list)
  landmark_reprojections: list[Float64Array] = field(default_factory=list)
  tag_detections: list[list[tuple[int, Float64Array, Float64Array]]] = field(
    default_factory=list
  )
  tag_axes: list[Float64Array] = field(default_factory=list)
  rmse_per_kf: list[float] = field(default_factory=list)


@dataclass
class CameraEstimate:
  """One camera's estimated state over a run: where its body went, what its
  biases and joint angles came out at, and what each lens saw."""

  slot: str
  # The session clock every stream here is indexed on.
  keyframe_times: Float64Array
  keyframe_rows: npt.NDArray[np.int64]
  # The solved body -> base-tag transform (a BA variable), logged statically.
  body_T_base_tag: gtsam.Pose3
  base_tag_id: int
  # The solved body -> rear-lens pose.
  body_T_rear_lens: gtsam.Pose3
  # The series propagated between the keyframes, at the IMU's own sample times.
  highrate_times: Float64Array
  highrate_positions: Float64Array
  highrate_quats: Float64Array
  highrate_vels: Float64Array
  # Where the body went at each keyframe, the pose everything else composes on.
  body_poses: list[gtsam.Pose3]
  # Per-lens payloads, keyed by the lens name their entity paths carry.
  lenses: dict[str, LensEstimate] = field(default_factory=dict)
  velocities: list[Float64Array] = field(default_factory=list)
  gyro_biases: list[Float64Array] = field(default_factory=list)
  accel_biases: list[Float64Array] = field(default_factory=list)
  # URDF joint name → per-keyframe angle, NaN where it could not be recovered.
  joint_angles: dict[str, list[float]] = field(default_factory=dict)
  trajectory_offset_in_gripper: tuple[float, float, float] = (0.0, 0.0, 0.0)
  # Per-keyframe ``(n, 2, 3)`` lens-to-corner ray endpoints, one strip per firing
  # reprojection factor: anchor corners, then another camera's base-tag corners.
  anchor_obs_lines: list[Float64Array] = field(default_factory=list)
  cross_camera_lines: list[Float64Array] = field(default_factory=list)
  # Per-joint pinned finger angles at the frame rate, as (times, angles).
  joint_frame_angles: dict[str, tuple[Float64Array, Float64Array]] = field(
    default_factory=dict
  )
  # The tag riding each revolute joint, by id.
  joint_tags: dict[str, int] = field(default_factory=dict)
  # Gripper opening at the pinned finger cadence, as (times, meters).
  gripper_opening: tuple[Float64Array, Float64Array] | None = None


def _send_joint_transform(
  emit: Sink,
  entity_path: str,
  joint: Any,
  times: Float64Array,
  angles: Any,
  indexes: _Indexes,
) -> tuple[Float64Array, Any]:
  """One joint's transform stream on both clocks, padded back to the merged
  timeline's start so the joint frames resolve before the solve's first sample.
  Returns the padded series."""
  if len(times) and indexes.timeline_start < times[0]:
    times = np.concatenate(([indexes.timeline_start], times))
    angles = np.concatenate((angles[:1], angles))
  _send(
    emit,
    entity_path,
    indexes.dense(times),
    joint.compute_transform_columns([float(a) for a in angles]),
  )
  return times, angles


def _log_tag(
  layers: Layers,
  entity_path: str,
  placement: Any,
  tag_id: int,
  axis_length: float = TAG_AXIS,
  is_reference: bool | None = None,
  tag_size: float | None = None,
) -> None:
  """One physical tag: where it sits, how wide it is, which tag it is, and the
  triad drawing it. ``placement`` is a pose of its own or the name of a frame
  the model already places."""
  values: dict[str, Any] = {"tag_id": [tag_id]}
  if is_reference is not None:
    values["is_reference"] = [is_reference]
  if tag_size is not None:
    values["tag_size"] = [tag_size]
  _static(layers.estimate, entity_path, placement, rr.AnyValues.columns(**values))
  _static(
    layers.viz, entity_path, rr.TransformAxes3D.columns(axis_length=[axis_length])
  )


def _log_anchor(
  layers: Layers,
  tag_id: int,
  world_T_anchor: gtsam.Pose3,
  tag_size: float,
  is_reference: bool,
) -> None:
  _log_tag(
    layers,
    record_schema.anchor(tag_id),
    _pose_columns(world_T_anchor),
    tag_id,
    axis_length=tag_size / 2.0,
    is_reference=is_reference,
    tag_size=tag_size,
  )


def _dense_indexes(times: Float64Array, frame_period: float) -> list[rr.TimeColumn]:
  """A dense series' clocks: the session time, and each sample's review-timeline
  bucket beside it, so a keyframe-indexed join reaches the dense fields too."""
  stride = record_schema.keyframe_stride(1.0 / frame_period)
  rows = np.round(times / frame_period).astype(np.int64)
  return [
    rr.TimeColumn(_TIMELINE, duration=times),
    rr.TimeColumn(
      record_schema.KEYFRAME_TIMELINE,
      sequence=record_schema.keyframe_index(rows, stride),
    ),
  ]


@dataclass(frozen=True)
class _Indexes:
  """The timeline vocabulary one camera's streams are indexed on: the session
  clock at its keyframes, the review timeline when the keyframe rows are known,
  the pair of indexes each image overlay is bounded by, and where the merged
  timeline starts."""

  keyframes: list[rr.TimeColumn]
  overlay: list[rr.TimeColumn]
  frame_period: float
  timeline_start: float

  def dense(self, times: Float64Array) -> list[rr.TimeColumn]:
    """The clocks a sample-rate series is indexed on."""
    return _dense_indexes(times, self.frame_period)


def _camera_indexes(
  data: CameraEstimate, frame_period: float, timeline_start: float
) -> _Indexes:
  """Every index a camera's streams share, derived once from its keyframes."""
  trail_times = np.asarray(data.keyframe_times, dtype=np.float64)
  times = _clock(trail_times)
  stride = record_schema.keyframe_stride(1.0 / frame_period)
  keyframe_column = rr.TimeColumn(
    record_schema.KEYFRAME_TIMELINE,
    sequence=record_schema.keyframe_index(data.keyframe_rows, stride),
  )
  return _Indexes(
    keyframes=[times, keyframe_column],
    overlay=[keyframe_column],
    frame_period=frame_period,
    timeline_start=timeline_start,
  )


def _camera_statics(layers: Layers, data: CameraEstimate) -> None:
  """The camera's time-independent geometry: each lens's map in the world frame,
  the body triad and its keypoint marker, the base tag bolted to the body, and
  the rear lens seat the bundle solved."""
  for lens_name, lens in data.lenses.items():
    _in_world(layers.estimate, record_schema.landmark(data.slot, lens_name))
    _log_static_points(
      layers,
      record_schema.landmark(data.slot, lens_name),
      lens.points,
      _LENS_COLORS[lens_name],
    )
  body = record_schema.camera(data.slot)
  _static(layers.viz, body, rr.TransformAxes3D.columns(axis_length=[BODY_AXIS]))
  _static(
    layers.viz,
    f"{body}/marker",
    rr.Points3D.columns(
      positions=np.zeros((1, 3)), radii=[0.002], colors=[(240, 80, 80)]
    ),
  )
  # A child of the body: entity nesting composes the body -> tag transform.
  _log_tag(
    layers,
    record_schema.base_tag(data.slot),
    _pose_columns(data.body_T_base_tag),
    data.base_tag_id,
  )
  _static(
    layers.estimate,
    record_schema.lens_seat(data.slot, "rear"),
    _pose_columns(data.body_T_rear_lens),
  )


def _urdf_model(
  layers: Layers, data: CameraEstimate, urdf_path: Path
) -> tuple[rr.urdf.UrdfTree, str]:
  """The gripper model and the frame naming its root, its geometry written and
  that root bridged to the world. A mesh that cannot be read costs pictures
  alone: the joint angles, the joint definitions and the model's provenance are
  written either way."""
  gripper_root = "/" + record_schema.gripper(data.slot)
  # The loader escapes slashes in the prefix, so the tree is built at a
  # single-segment root and every chunk is moved under the gripper afterwards.
  tree = rr.urdf.UrdfTree.from_file_path(
    str(urdf_path),
    entity_path_prefix=data.slot,
    frame_prefix=f"{data.slot}_",
    static_transform_entity_path=record_schema.gripper_part(data.slot, "tf_static"),
  )
  root_frame = f"{data.slot}_{tree.root_link().name}"
  model_root = f"/{data.slot}/{tree.name}"
  try:
    links = [tree.root_link(), *map(tree.get_joint_child, tree.joints())]
    collision = [p for link in links for p in tree.get_collision_geometry_paths(link)]
    drawn = {p for link in links for p in tree.get_visual_geometry_paths(link)}
    model_chunks = tree.stream().drop(content=collision).to_chunks()
  except Exception as exc:  # noqa: BLE001 - a mesh failure costs pictures alone
    print(f"  [warn] [model] geometry skipped ({exc})")
    drawn = set()
    model_chunks = [
      chunk
      for chunk in tree.stream().to_chunks()
      if "geometries" not in chunk.entity_path
    ]
  # Schema paths carry no leading slash; a chunk's entity path does.
  static_path = f"/{record_schema.gripper_part(data.slot, 'tf_static')}"
  for chunk in model_chunks:
    # The static transforms are already at their final path.
    tail = chunk.entity_path.removeprefix(model_root)
    sink = layers.viz if chunk.entity_path in drawn else layers.estimate
    if chunk.entity_path == static_path:
      for fixed in _fixed_joint_transforms(chunk, tree, data.slot):
        sink(fixed)
      continue
    sink(
      chunk
      if tail == chunk.entity_path
      else chunk.with_entity_path(gripper_root + tail)
    )
  # CoordinateFrame places the entity; a two-frame transform only defines an edge.
  _static(
    layers.estimate,
    gripper_root,
    rr.CoordinateFrame.columns(frame=[root_frame]),
    rr.AnyValues.columns(
      robot=[tree.name],
      urdf_sha=[hashlib.sha256(urdf_path.read_bytes()).hexdigest()],
    ),
  )
  return tree, root_frame


def _fixed_joint_transforms(chunk: Any, tree: Any, slot: str) -> list[Any]:
  """The model's static transforms, less the joints the solve moves: a frame
  pair is one relation, and the solve writes those per keyframe."""
  fixed = {
    f"{slot}_{tree.get_joint_child(joint).name}"
    for joint in tree.joints()
    if joint.joint_type == "fixed"
  }
  batch = chunk.to_record_batch()
  child = batch.column(batch.schema.get_field_index("Transform3D:child_frame"))
  keep = pc.is_in(pc.list_flatten(child), value_set=pa.array(sorted(fixed)))
  out = Chunk.from_record_batch(batch.filter(keep))
  return out if isinstance(out, list) else [out]


def _pose_streams(layers: Layers, data: CameraEstimate, indexes: _Indexes) -> None:
  """The body pose and linear velocity, each as two separate estimates on
  separate entities: the solver's exact state at the keyframes, and the
  IMU-propagated series between them. The propagated pose is the camera entity
  itself, which is what makes the lenses, the IMU and the base tag ride it."""
  # Naming both ends detaches this from path composition; the frames it would
  # infer sit below the camera's own pose and compose with it.
  kf_columns = _frame_columns(
    _translations(data.body_poses),
    _pose_quats(data.body_poses),
    record_schema.state_frame(data.slot),
  )
  _send(
    layers.estimate,
    record_schema.state(data.slot, "pose"),
    indexes.keyframes,
    kf_columns,
  )
  _send(
    layers.estimate,
    record_schema.camera(data.slot),
    indexes.dense(data.highrate_times),
    rr.Transform3D.columns(
      translation=data.highrate_positions, quaternion=data.highrate_quats
    ),
  )
  if data.velocities:
    _send(
      layers.estimate,
      record_schema.state(data.slot, "lin_vel"),
      indexes.keyframes,
      rr.AnyValues.columns(lin_vel=_vec3_column(np.stack(data.velocities))),
    )
  _send(
    layers.estimate,
    record_schema.lin_vel(data.slot),
    indexes.dense(data.highrate_times),
    rr.AnyValues.columns(lin_vel=_vec3_column(data.highrate_vels)),
  )


def _gripper_streams(
  layers: Layers, data: CameraEstimate, indexes: _Indexes, root_frame: str
) -> tuple[Float64Array, Float64Array]:
  """The URDF root's pose over time, propagated from the body; the rotations
  and translations of that stream, which the trajectory is offset from. One
  estimate only: the solved root at a keyframe is that keyframe's state composed
  with the base tag, both already in the layer."""
  gripper_path = record_schema.gripper_part(data.slot, "transforms")
  rel = data.body_T_base_tag
  body_R = quats_to_matrices(data.highrate_quats)
  gripper_t = data.highrate_positions + body_R @ np.asarray(rel.translation())
  gripper_R = body_R @ np.asarray(rel.rotation().matrix())
  _send(
    layers.estimate,
    gripper_path,
    indexes.dense(data.highrate_times),
    _frame_columns(gripper_t, matrices_to_quats(gripper_R), root_frame),
  )
  return gripper_t, gripper_R


def _joint_streams(
  layers: Layers,
  data: CameraEstimate,
  indexes: _Indexes,
  tree: rr.urdf.UrdfTree,
) -> None:
  """Each solved joint's transform and angle, at the pinned frame cadence where
  one was recovered and at the keyframes otherwise."""
  for joint_name in data.joint_angles:
    joint = tree.get_joint_by_name(joint_name)
    if joint is None:
      continue
    joint_path = record_schema.gripper_part(data.slot, joint_name)
    _static(
      layers.estimate,
      record_schema.joint_model(data.slot, joint_name),
      rr.AnyValues.columns(
        joint_type=[joint.joint_type],
        axis=[np.asarray(joint.axis, dtype=np.float64)],
        limit_lower=[joint.limit_lower],
        limit_upper=[joint.limit_upper],
      ),
    )
    tag_id = data.joint_tags.get(joint_name)
    if tag_id is not None:
      _log_tag(
        layers,
        record_schema.joint_tag(data.slot, joint_name),
        rr.CoordinateFrame.columns(
          frame=[f"{data.slot}_{tree.get_joint_child(joint).name}_tag"]
        ),
        tag_id,
      )
    frame_series = data.joint_frame_angles.get(joint_name)
    if frame_series is None:
      # A joint absent from the frame series was detected in no frame at all.
      continue
    # The pinned frame-rate series passes through the solved keyframe angles.
    times, angles = _send_joint_transform(
      layers.estimate,
      record_schema.gripper_part(data.slot, "transforms"),
      joint,
      *frame_series,
      indexes,
    )
    _static(
      layers.estimate,
      joint_path,
      rr.CoordinateFrame.columns(
        frame=[f"{data.slot}_{tree.get_joint_child(joint).name}"]
      ),
    )
    _send_angle(layers.estimate, joint_path, indexes.dense(times), angles)


def _trajectory_streams(
  layers: Layers,
  data: CameraEstimate,
  gripper_hr: tuple[Float64Array, Float64Array],
  trajectory_color: tuple[int, int, int],
) -> None:
  """The fading trail the gripper keypoint drew, following the propagated
  series."""
  hr_times = data.highrate_times
  offset = np.asarray(data.trajectory_offset_in_gripper, dtype=np.float64)
  gripper_t, gripper_R = gripper_hr
  points = gripper_t + gripper_R @ offset
  times = _clock(hr_times)
  trail_times = hr_times
  entity_path = record_schema.trajectory(data.slot)
  _in_world(layers.viz, entity_path)
  _send(
    layers.viz,
    entity_path,
    [times],
    rr.LineStrips3D.columns(
      strips=_trail_segments(points, trail_times, TRAJECTORY_TRAIL_DURATION)
    ),
  )
  _static(
    layers.viz,
    entity_path,
    rr.LineStrips3D.columns(radii=[0.002], colors=[trajectory_color]),
  )


def _factor_line_streams(
  layers: Layers, data: CameraEstimate, indexes: _Indexes
) -> None:
  """The rays of the reprojection factors that fired at each keyframe."""
  for kind, lines, color in (
    ("anchor", data.anchor_obs_lines, _YELLOW),
    ("cross_camera", data.cross_camera_lines, _CYAN),
  ):
    _in_world(layers.viz, record_schema.factor(data.slot, kind))
    _send_factor_line_columns(
      layers.viz, record_schema.factor(data.slot, kind), lines, indexes.keyframes, color
    )


def _lens_streams(
  layers: Layers,
  data: CameraEstimate,
  indexes: _Indexes,
  anchor_tag_ids: set[int],
) -> None:
  """Every image overlay, a kind at a time across the lenses, so one kind's
  chunks land together. Landmark colors match the 3D scene; the 2D keypoint
  measurements are magenta."""
  detection = record_schema.detection
  overlay = record_schema.overlay
  for lens_name, lens in data.lenses.items():
    if lens.keypoints:
      _send_pixel_columns(
        layers,
        detection(data.slot, lens_name, record_schema.KEYPOINTS_DETECTION),
        lens.keypoints,
        indexes.overlay,
        1.5,
        _MAGENTA,
        measured=True,
      )
  for lens_name, lens in data.lenses.items():
    if lens.tag_detections:
      _send_lens_tag_columns(
        layers,
        detection(data.slot, lens_name, record_schema.TAGS_DETECTION),
        lens.tag_detections,
        indexes.overlay,
        anchor_tag_ids,
      )
  for lens_name, lens in data.lenses.items():
    if lens.rejected_keypoints:
      _send_pixel_columns(
        layers,
        detection(data.slot, lens_name, record_schema.REJECTED_KEYPOINTS_DETECTION),
        lens.rejected_keypoints,
        indexes.overlay,
        1.5,
        _RED,
        measured=True,
      )
  for lens_name, lens in data.lenses.items():
    if lens.landmark_reprojections:
      _send_pixel_columns(
        layers,
        overlay(data.slot, lens_name, record_schema.LANDMARK_REPROJECTION_OVERLAY),
        lens.landmark_reprojections,
        indexes.overlay,
        3.0,
        _LENS_COLORS[lens_name],
        measured=False,
      )
  for lens_name, lens in data.lenses.items():
    if lens.tag_axes:
      _send_tag_axes_columns(
        layers,
        overlay(data.slot, lens_name, record_schema.TAG_AXES_OVERLAY),
        lens.tag_axes,
        indexes.overlay,
      )


def _bias_streams(layers: Layers, data: CameraEstimate, indexes: _Indexes) -> None:
  """The solved IMU biases, beside the raw streams they correct."""
  for kind, biases in (("gyro", data.gyro_biases), ("accel", data.accel_biases)):
    if not biases:
      continue
    _send_series(
      layers,
      record_schema.imu_bias(data.slot, kind),
      indexes.keyframes,
      np.asarray(biases, dtype=np.float64),
      [f"{data.slot}/{kind}_bias/{axis}" for axis in "xyz"],
    )


def _fit_streams(layers: Layers, data: CameraEstimate, indexes: _Indexes) -> None:
  """What each keyframe cost the frontend: how many keypoints survived, and how
  well they reprojected."""
  _send_lens_series(
    layers,
    record_schema.fit(data.slot, "n_keypoints"),
    indexes.keyframes,
    data,
    lambda lens: [kp.shape[0] for kp in lens.keypoints] if lens.keypoints else None,
  )
  _send_lens_series(
    layers,
    record_schema.fit(data.slot, "rmse"),
    indexes.keyframes,
    data,
    lambda lens: lens.rmse_per_kf,
  )


def _log_camera(
  layers: Layers,
  data: CameraEstimate,
  urdf_path: Path,
  anchor_tag_ids: set[int],
  trajectory_color: tuple[int, int, int],
  timeline_start: float,
  frame_period: float,
) -> None:
  """Log every entity under one camera: its poses, gripper, landmarks, URDF,
  image overlays and diagnostics. The world statics the cameras share must
  already be logged by the caller."""
  _camera_statics(layers, data)
  tree, root_frame = _urdf_model(layers, data, urdf_path)
  indexes = _camera_indexes(data, frame_period, timeline_start)
  _pose_streams(layers, data, indexes)
  gripper_hr = _gripper_streams(layers, data, indexes, root_frame)
  _joint_streams(layers, data, indexes, tree)
  if data.gripper_opening is not None:
    opening_times, opening_values = data.gripper_opening
    _send_scalars(
      layers.estimate,
      record_schema.gripper_part(data.slot, "opening"),
      _dense_indexes(opening_times, frame_period),
      opening_values,
    )
  _trajectory_streams(layers, data, gripper_hr, trajectory_color)
  _factor_line_streams(layers, data, indexes)
  _lens_streams(layers, data, indexes, anchor_tag_ids)
  _bias_streams(layers, data, indexes)
  _fit_streams(layers, data, indexes)


def _send_lens_series(
  layers: Layers,
  entity_path: str,
  indexes: list[rr.TimeColumn],
  data: CameraEstimate,
  values: Callable[[LensEstimate], list[float] | list[int] | None],
) -> None:
  """One plot entity for a per-lens metric, a series per lens that has it."""
  per_lens = {name: values(lens) for name, lens in data.lenses.items()}
  present = {name: v for name, v in per_lens.items() if v}
  if not present:
    return
  _send_series(
    layers,
    entity_path,
    indexes,
    np.stack([np.asarray(v, dtype=np.float64) for v in present.values()], axis=1),
    [f"{data.slot}/{name}" for name in present],
  )


_TRAJECTORY_PALETTE = (
  _FRONT_COLOR,
  _REAR_COLOR,
  (130, 140, 240),
  (220, 90, 200),
  _CYAN,
)


def estimate_chunks(
  layers: Layers,
  cameras: list[CameraEstimate],
  anchor_poses_world: dict[int, gtsam.Pose3],
  reference_tag_id: int,
  anchor_tag_size: float,
  urdf_path: Path,
  timeline_start: float,
  frame_period: float,
) -> None:
  """Build every entity of a solved run as chunks, each handed to the sink for
  the layer it belongs to.

  Every entity a camera owns hangs under ``/camera/<slot>/...``; each anchor
  sits at ``/world/anchor/<tag_id>``, carrying whether it is the reference the
  gauge was pinned to.
  """
  _static(
    layers.estimate,
    record_schema.WORLD,
    rr.ViewCoordinates.columns(xyz=[rr.components.ViewCoordinates.RIGHT_HAND_Z_UP]),
    rr.Transform3D.columns(child_frame=[record_schema.WORLD_FRAME]),
  )
  for tag_id, pose in anchor_poses_world.items():
    _log_anchor(
      layers,
      tag_id,
      pose,
      anchor_tag_size,
      tag_id == reference_tag_id,
    )
  for i, camera in enumerate(cameras):
    _log_camera(
      layers,
      camera,
      urdf_path=urdf_path,
      anchor_tag_ids=set(anchor_poses_world),
      trajectory_color=_TRAJECTORY_PALETTE[i % len(_TRAJECTORY_PALETTE)],
      timeline_start=timeline_start,
      frame_period=frame_period,
    )
