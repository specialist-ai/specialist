"""Columnar Rerun logging of dual-lens VI-BA runs, one or many trackers alike."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gtsam
import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

Float64Array = npt.NDArray[np.float64]

_FRONT_COLOR = (80, 200, 120)
_REAR_COLOR = (230, 140, 80)
_RED = (230, 60, 60)
_YELLOW = (250, 250, 50)
_MAGENTA = (255, 0, 255)
_CYAN = (90, 220, 220)
_TRAJECTORY_TRAIL_DURATION = 1.0
_BODY_AXIS_LENGTH = 0.05
_TIMELINE = "video_time"

__all__ = ["TrackerVizData", "log_multi_ba_run"]


def _pose_quaternion_xyzw(pose: gtsam.Pose3) -> Float64Array:
  q = pose.rotation().toQuaternion()
  return np.array([q.x(), q.y(), q.z(), q.w()], dtype=np.float64)


def _translations(poses: list[gtsam.Pose3]) -> Float64Array:
  return np.asarray([np.asarray(p.translation(), dtype=np.float64) for p in poses])


def _pose_transform3d(pose: gtsam.Pose3) -> rr.Transform3D:
  return rr.Transform3D(
    translation=np.asarray(pose.translation(), dtype=np.float64),
    quaternion=rr.datatypes.Quaternion(xyzw=_pose_quaternion_xyzw(pose)),
  )


def _trail_strips(
  traj: Float64Array, times: Float64Array, duration: float
) -> list[Float64Array]:
  """Per-keyframe position strips in ``[t - duration, t]``; ``times`` must ascend."""
  starts = np.searchsorted(times, times - duration, side="left")
  return [traj[start : i + 1] for i, start in enumerate(starts)]


def _log_static_points(
  path: str, points: dict[int, Float64Array], color: tuple[int, int, int]
) -> None:
  if not points:
    return
  xyz = np.asarray(list(points.values()), dtype=np.float64)
  rr.log(path, rr.Points3D(xyz, radii=0.002, colors=color), static=True)


def _solid_colors(color: tuple[int, int, int], n: int) -> np.ndarray:
  """An ``(n, 3)`` uint8 array of one repeated RGB color."""
  return np.tile(np.asarray(color, dtype=np.uint8), (n, 1))


def _send_lens_tag_columns(
  entity_prefix: str,
  tags_per_kf: list[list[tuple[int, Float64Array, Float64Array]]],
  times: rr.TimeColumn,
  anchor_tag_ids: set[int],
  mask_pad_px: int,
) -> None:
  """Per-keyframe AprilTag quads and ids; anchor yellow, others red with mask boxes."""
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
      if not is_anchor:
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
  rr.send_columns(
    f"{entity_prefix}/tags",
    indexes=[times],
    columns=rr.LineStrips2D.columns(
      strips=strips,
      colors=color_arr,
      radii=np.full(total, 5.0, dtype=np.float32),
    ).partition(partition),
  )
  rr.send_columns(
    f"{entity_prefix}/tags/labels",
    indexes=[times],
    columns=rr.Points2D.columns(
      positions=np.stack(centers),
      labels=labels,
      show_labels=[True] * total,
      colors=color_arr,
      radii=np.full(total, 10.0, dtype=np.float32),
    ).partition(partition),
  )
  total_masked = sum(mask_partition)
  if total_masked > 0:
    rr.send_columns(
      f"{entity_prefix}/tags/masked",
      indexes=[times],
      columns=rr.Boxes2D.columns(
        centers=np.stack(mask_centers),
        half_sizes=np.stack(mask_half_sizes),
        colors=_solid_colors(_RED, total_masked),
        labels=["MASKED"] * total_masked,
        show_labels=[True] * total_masked,
      ).partition(mask_partition),
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
  times: rr.TimeColumn,
  radius: float,
  color: tuple[int, int, int],
) -> None:
  partition = [int(kp.shape[0]) for kp in points_per_kf]
  total = sum(partition)
  if total == 0:
    return
  positions = np.concatenate([kp for kp in points_per_kf if kp.shape[0] > 0])
  rr.send_columns(
    entity_path,
    indexes=[times],
    columns=rr.Points2D.columns(
      positions=positions,
      radii=np.full(total, radius, dtype=np.float32),
      colors=_solid_colors(color, total),
    ).partition(partition),
  )


def _send_tag_axes_columns(
  entity_prefix: str,
  axes_per_kf: list[Float64Array],
  times: rr.TimeColumn,
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
  rr.send_columns(
    f"{entity_prefix}/tag_axes",
    indexes=[times],
    columns=rr.LineStrips2D.columns(
      strips=strips,
      colors=np.stack(colors),
      radii=np.full(total, 4.0, dtype=np.float32),
    ).partition(partition),
  )


def _send_factor_line_columns(
  entity_path: str,
  lines_per_kf: list[Float64Array],
  color: tuple[int, int, int],
  times: rr.TimeColumn,
) -> None:
  """Per-keyframe ``(n_lines, 2, 3)`` endpoint arrays as LineStrips3D rows."""
  partition = [int(arr.shape[0]) for arr in lines_per_kf]
  total = sum(partition)
  if total == 0:
    return
  strips = [line.astype(np.float32) for arr in lines_per_kf for line in arr]
  rr.send_columns(
    entity_path,
    indexes=[times],
    columns=rr.LineStrips3D.columns(
      strips=strips,
      colors=_solid_colors(color, total),
      radii=np.full(total, 0.0008, dtype=np.float32),
    ).partition(partition),
  )


def _send_lens_columns(
  entity_prefix: str,
  metrics_prefix: str,
  frames: list[bytes],
  keypoints: list[Float64Array] | None,
  color: tuple[int, int, int],
  times: rr.TimeColumn,
) -> None:
  rr.send_columns(
    entity_prefix,
    indexes=[times],
    columns=rr.EncodedImage.columns(
      blob=frames, media_type=["image/jpeg"] * len(frames)
    ),
  )
  if keypoints is None:
    return
  partition = [int(kp.shape[0]) for kp in keypoints]
  positions = (
    np.concatenate(keypoints) if any(p > 0 for p in partition) else np.empty((0, 2))
  )
  total = positions.shape[0]
  rr.send_columns(
    f"{entity_prefix}/keypoints",
    indexes=[times],
    columns=rr.Points2D.columns(
      positions=positions,
      radii=np.full(total, 10.0, dtype=np.float32),
      colors=_solid_colors(color, total),
    ).partition(partition),
  )
  rr.send_columns(
    metrics_prefix,
    indexes=[times],
    columns=rr.Scalars.columns(scalars=np.asarray(partition, dtype=np.float64)),
  )


@dataclass
class TrackerVizData:
  """One tracker's payload, built by ``log_multi_ba_run`` for ``_log_tracker``."""

  name: str = "default"
  body_poses: list[gtsam.Pose3] = field(default_factory=list)
  keyframe_times: Float64Array | None = None
  front_points: dict[int, Float64Array] = field(default_factory=dict)
  rear_points: dict[int, Float64Array] = field(default_factory=dict)
  # JPEG-encoded keyframes; logged as EncodedImage blobs.
  front_frames: list[bytes] | None = None
  rear_frames: list[bytes] | None = None
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
  urdf_root_poses: list[gtsam.Pose3] | None = None
  # NaN entries hold the last finite angle so dropouts don't snap to zero.
  joint_angles: dict[str, list[float]] | None = None
  trajectory_offset_in_gripper: tuple[float, float, float] | None = None
  front_lens_extrinsic: gtsam.Pose3 | None = None
  rear_lens_extrinsic: gtsam.Pose3 | None = None
  # Per-keyframe ``(n, 2, 3)`` lens lines to anchor corners and to other trackers' tags.
  anchor_obs_lines: list[Float64Array] | None = None
  cross_tracker_lines: list[Float64Array] | None = None
  # ``None`` lets ``log_multi_ba_run`` assign a per-tracker palette color.
  trajectory_color: tuple[int, int, int] | None = None


def _gripper_points(data: TrackerVizData) -> Float64Array:
  """Per-keyframe gripper keypoints with a URDF, else the raw body positions."""
  if data.urdf_root_poses is not None and data.trajectory_offset_in_gripper is not None:
    offset = np.asarray(data.trajectory_offset_in_gripper, dtype=np.float64)
    return np.stack(
      [
        np.asarray(p.translation(), dtype=np.float64)
        + np.asarray(p.rotation().matrix(), dtype=np.float64) @ offset
        for p in data.urdf_root_poses
      ]
    )
  return _translations(data.body_poses)


def _floor_z_for(data: TrackerVizData) -> float | None:
  pts = _gripper_points(data)
  return float(pts[:, 2].min()) if len(pts) else None


def _log_anchor(
  spatial_root: str,
  world_T_anchor: gtsam.Pose3 | None,
  anchor_tag_size: float | None,
  entity: str,
) -> None:
  if anchor_tag_size is None:
    return
  half = float(anchor_tag_size) / 2.0
  if world_T_anchor is not None:
    rr.log(
      f"{spatial_root}/{entity}",
      _pose_transform3d(world_T_anchor),
      static=True,
    )
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
    f"{spatial_root}/{entity}",
    rr.LineStrips3D([square], colors=_YELLOW, radii=0.003),
    static=True,
  )
  rr.log(
    f"{spatial_root}/{entity}/frame",
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
) -> None:
  """Log every per-tracker entity; the caller logs the shared statics."""
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
  for lens_name, lens_pose in (
    ("front", data.front_lens_extrinsic),
    ("rear", data.rear_lens_extrinsic),
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
      print(f"  [viz] URDF model skipped ({exc}); logging the trajectory without it")
      tree = None
      urdf_root_frame = None

  n = len(data.body_poses)
  if data.keyframe_times is not None:
    trail_times = np.asarray(data.keyframe_times, dtype=np.float64)
    times = rr.TimeColumn(_TIMELINE, duration=trail_times)
  else:
    trail_times = np.arange(n, dtype=np.float64)
    times = rr.TimeColumn(_TIMELINE, sequence=np.arange(n, dtype=np.int64))

  quats = np.stack([_pose_quaternion_xyzw(p) for p in data.body_poses])
  rr.send_columns(
    f"{spatial_root}/body",
    indexes=[times],
    columns=rr.Transform3D.columns(translation=pose_translations, quaternion=quats),
  )
  if urdf_root_frame is not None:
    poses = (
      data.urdf_root_poses if data.urdf_root_poses is not None else data.body_poses
    )
    urdf_traj = _translations(poses)
    urdf_quats = np.stack([_pose_quaternion_xyzw(p) for p in poses])
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
      assert tree is not None  # non-None whenever urdf_path is set
      for joint_name, raw_values in data.joint_angles.items():
        joint = tree.get_joint_by_name(joint_name)
        if joint is None:
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
        rr.send_columns(
          f"{spatial_root}/{joint_name}",
          indexes=[times],
          columns=joint.compute_transform_columns(clean),
        )
  rr.send_columns(
    f"{spatial_root}/trajectory",
    indexes=[times],
    columns=rr.LineStrips3D.columns(
      strips=_trail_strips(traj, trail_times, _TRAJECTORY_TRAIL_DURATION),
      radii=np.full(n, 0.002, dtype=np.float32),
      colors=np.tile(np.asarray(data.trajectory_color, dtype=np.uint8), (n, 1)),
    ),
  )
  rr.send_columns(
    f"metrics/distance_from_start{metrics_suffix}",
    indexes=[times],
    columns=rr.Scalars.columns(scalars=np.linalg.norm(traj - traj[0:1], axis=1)),
  )

  if data.anchor_obs_lines is not None:
    _send_factor_line_columns(
      f"{spatial_root}/factor_lines/anchor",
      data.anchor_obs_lines,
      _YELLOW,
      times,
    )
  if data.cross_tracker_lines is not None:
    _send_factor_line_columns(
      f"{spatial_root}/factor_lines/cross_tracker",
      data.cross_tracker_lines,
      _CYAN,
      times,
    )

  for lens_name, frames, kps, color in (
    ("front", data.front_frames, data.front_keypoints, _FRONT_COLOR),
    ("rear", data.rear_frames, data.rear_keypoints, _REAR_COLOR),
  ):
    if frames is None:
      continue
    _send_lens_columns(
      f"{image_root}/{lens_name}",
      f"metrics/n_keypoints/{lens_name}{metrics_suffix}",
      frames,
      kps,
      color,
      times,
    )
  for lens_name, tags in (
    ("front", data.front_tag_detections),
    ("rear", data.rear_tag_detections),
  ):
    if tags is None:
      continue
    _send_lens_tag_columns(
      f"{image_root}/{lens_name}", tags, times, anchor_tag_ids, mask_pad_px
    )
  for lens_name, rejected in (
    ("front", data.front_rejected_keypoints),
    ("rear", data.rear_rejected_keypoints),
  ):
    if rejected is None:
      continue
    _send_point_columns(
      f"{image_root}/{lens_name}/keypoints_rejected", rejected, times, 8.0, _RED
    )
  for lens_name, reproj in (
    ("front", data.front_landmark_reprojections),
    ("rear", data.rear_landmark_reprojections),
  ):
    if reproj is None:
      continue
    _send_point_columns(
      f"{image_root}/{lens_name}/landmark_reprojection", reproj, times, 16.0, _MAGENTA
    )
  for lens_name, axes in (
    ("front", data.front_tag_axes),
    ("rear", data.rear_tag_axes),
  ):
    if axes is None:
      continue
    _send_tag_axes_columns(f"{image_root}/{lens_name}", axes, times)
  if data.rear_static_mask_rects:
    _log_static_mask_rects(f"{image_root}/rear", data.rear_static_mask_rects)

  if data.velocities is not None:
    speeds = np.asarray(
      [float(np.linalg.norm(v)) for v in data.velocities], dtype=np.float64
    )
    rr.send_columns(
      f"metrics/speed/body{metrics_suffix}",
      indexes=[times],
      columns=rr.Scalars.columns(scalars=speeds),
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


def _palette(i: int) -> tuple[int, int, int]:
  table = (
    _FRONT_COLOR,
    _REAR_COLOR,
    (130, 140, 240),
    (220, 90, 200),
    _CYAN,
  )
  return table[i % len(table)]


def _multi_blueprint(tracker_names: list[str], floor_z: float | None) -> rrb.Blueprint:
  """Multi-tracker blueprint: one image-pair column per tracker, scalar plots merged."""
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


def log_multi_ba_run(
  rrd_path: Path,
  trackers: list[TrackerVizData],
  anchor_poses_world: dict[int, gtsam.Pose3],
  reference_tag_id: int,
  anchor_tag_size: float | None = None,
  mask_pad_px: int = 0,
  urdf_path: Path | None = None,
  open_viewer: bool = True,
) -> None:
  """Multi-tracker entry point: each tracker's entities live under ``/<name>/...``."""
  rr.init("specialist.vi_ba", spawn=False)
  for i, t in enumerate(trackers):
    if not t.name:
      t.name = f"tracker_{i}"
    if t.trajectory_color is None:
      t.trajectory_color = _palette(i)
  floor_zs = [z for z in (_floor_z_for(t) for t in trackers) if z is not None]
  floor_z = float(min(floor_zs)) if floor_zs else None
  rr.send_blueprint(
    _multi_blueprint([t.name for t in trackers], floor_z),
    make_active=True,
    make_default=True,
  )
  if open_viewer:
    rr.spawn(connect=False)
    rr.set_sinks(rr.FileSink(str(rrd_path)), rr.GrpcSink())
  else:
    # --no-open-viewer writes the rrd directly, for headless servers.
    rr.save(str(rrd_path))
  rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
  rr.log("world", rr.Transform3D(child_frame="world"), static=True)
  for tag_id, pose in anchor_poses_world.items():
    entity = "anchor_tag" if tag_id == reference_tag_id else f"anchor_tag_{tag_id}"
    _log_anchor("world", pose, anchor_tag_size, entity=entity)
  for t in trackers:
    _log_tracker(
      t,
      spatial_root=t.name,
      image_root=f"image/{t.name}",
      metrics_suffix=f"/{t.name}",
      frame_prefix=f"{t.name}_",
      urdf_path=urdf_path,
      anchor_tag_ids=set(anchor_poses_world),
      mask_pad_px=mask_pad_px,
    )
  rr.disconnect()
