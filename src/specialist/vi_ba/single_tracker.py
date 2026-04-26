"""Per-tracker prep for VI-BA: decode → frontend → tag detect → anchor PnP →
IMU stationary gravity → VI alignment → anchor-init pose seed → metric
re-triangulation. Returns a ``PreparedTracker`` ready for ``track.py`` to
compose into a joint factor graph.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import MeiParams, mei_project, mei_unproject
from specialist.loader.parse_insv import load_insv
from specialist.vi_ba.anchor_detection import (
  TagDetection,
  anchor_pose_from_tags,
  detect_all_tags,
  estimate_camera_to_tag_pose,
)
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.configs import (
  AnchorCfg,
  FrontendCfg,
  GripperCfg,
  ImuCfg,
)
from specialist.vi_ba.extrinsics import parse_insv_extrinsics
from specialist.vi_ba.imu import (
  ImuSamples,
  factory_imu_bias,
  find_stationary_windows,
  make_preintegration_params,
  preintegrate_between_keyframes,
)
from specialist.vi_ba.init import (
  Alignment,
  apply_alignment_to_poses,
  body_poses_from_camera_poses,
  interpolate_camera_poses,
  pick_init_pair,
  two_view_essential,
  visual_inertial_align,
)
from specialist.vi_ba.pipeline import (
  FrontendConfig,
  decode_dual_keyframes,
  filter_tracks_by_parallax,
  keyframe_timestamps,
  lens_setups,
  run_lens_frontend,
  triangulate_lens,
)

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "PreparedTracker",
  "TrackerArgs",
  "prepare_tracker",
]


@dataclass
class TrackerArgs:
  """Per-tracker prep knobs. Composes nested configs from ``configs.py``."""

  insv_path: Path
  start: float = 0.0
  duration: float | None = None
  keyframe_every: int = 9
  gripper_tag_id: int = 6
  # 0 disables the corresponding URDF revolute joint angle recovery.
  left_finger_tag_id: int = 0
  right_finger_tag_id: int = 0
  frontend: FrontendCfg = field(default_factory=FrontendCfg)
  imu: ImuCfg = field(default_factory=ImuCfg)
  anchor: AnchorCfg = field(default_factory=AnchorCfg)
  gripper: GripperCfg = field(default_factory=GripperCfg)


@dataclass
class PreparedTracker:
  """Everything one tracker contributes to the joint graph + viz."""

  name: str
  body_poses: list[gtsam.Pose3]
  velocities: list[Float64Array]
  biases: list[gtsam.imuBias.ConstantBias]
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements]
  front_lens: LensSetup
  rear_lens: LensSetup
  front_tracks_tri: dict[int, dict[int, Float64Array]]
  rear_tracks_tri: dict[int, dict[int, Float64Array]]
  front_points: dict[int, Float64Array]
  rear_points: dict[int, Float64Array]
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]]
  anchor_pose_world_local: gtsam.Pose3 | None
  body_T_gripper_tag: gtsam.Pose3 | None
  # URDF joint name → per-keyframe rotation about the URDF z-axis. NaN where
  # either tag is missing or PnP fails. Empty when no finger ids configured.
  joint_angles: dict[str, list[float]]
  kf_times: Float64Array
  front_frames: list[np.ndarray]
  rear_frames: list[np.ndarray]
  front_keypoints: list[Float64Array]
  rear_keypoints: list[Float64Array]
  front_tags_per_kf: list[list[TagDetection]]
  rear_tags_per_kf: list[list[TagDetection]]
  front_rejected: dict[int, dict[int, Float64Array]]
  rear_rejected: dict[int, dict[int, Float64Array]]
  rear_static_mask_rects: list[tuple[int, int, int, int]]
  body_T_left_finger_per_kf: dict[int, gtsam.Pose3]
  body_T_right_finger_per_kf: dict[int, gtsam.Pose3]
  left_finger_tag_id: int
  right_finger_tag_id: int
  n_frames: int

  def vi_graph_inputs(self) -> dict[str, object]:
    """Bundle this tracker's state + lens data for ``build_vi_graph``."""
    return {
      "body_poses": self.body_poses,
      "velocities": self.velocities,
      "biases": self.biases,
      "preintegrated": self.preintegrated,
      "front": self.front_lens,
      "front_tracks": self.front_tracks_tri,
      "front_points": self.front_points,
      "rear": self.rear_lens,
      "rear_tracks": self.rear_tracks_tri,
      "rear_points": self.rear_points,
    }


def _build_non_anchor_tag_masks(
  frames: list[np.ndarray],
  tags_per_kf: list[list[TagDetection]],
  exclude_tag_id: int,
  pad_px: int,
) -> list[np.ndarray]:
  masks: list[np.ndarray] = []
  for frame, tags in zip(frames, tags_per_kf, strict=True):
    h, w = frame.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for tag in tags:
      if exclude_tag_id >= 0 and int(tag.tag_id) == exclude_tag_id:
        continue
      corners = np.asarray(tag.corners, dtype=np.float64)
      x0 = max(int(np.floor(corners[:, 0].min())) - pad_px, 0)
      y0 = max(int(np.floor(corners[:, 1].min())) - pad_px, 0)
      x1 = min(int(np.ceil(corners[:, 0].max())) + pad_px, w)
      y1 = min(int(np.ceil(corners[:, 1].max())) + pad_px, h)
      if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 0
    masks.append(mask)
  return masks


def _apply_static_mask_rects(
  masks: list[np.ndarray], rects: list[tuple[int, int, int, int]]
) -> list[np.ndarray]:
  if not rects or not masks:
    return masks
  h, w = masks[0].shape[:2]
  for x1, y1, x2, y2 in rects:
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
      continue
    for m in masks:
      m[y1c:y2c, x1c:x2c] = 0
  return masks


def _per_kf_cam_T_tag(
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  tag_id: int,
  tag_size: float,
  n_frames: int,
) -> list[gtsam.Pose3 | None]:
  """Per-keyframe ``cam_T_tag`` for one tag id, with IPPE's reflectional
  ambiguity disambiguated temporally via the previous frame's pose."""
  out: list[gtsam.Pose3 | None] = [None] * n_frames
  prev: gtsam.Pose3 | None = None
  for kf in range(min(n_frames, len(tags_per_kf))):
    corners = next(
      (d.corners for d in tags_per_kf[kf] if int(d.tag_id) == tag_id), None
    )
    if corners is None:
      continue
    p = estimate_camera_to_tag_pose(corners, tag_size, lens.params, prev_cam_T_tag=prev)
    if p is None:
      continue
    out[kf] = p
    prev = p
  return out


def _mean_pose(poses: list[gtsam.Pose3]) -> gtsam.Pose3:
  """Mean of a list of poses: arithmetic mean of translations + log-mean
  rotation about the first sample. Used for a fixed body-rigid tag where
  every observation should agree up to PnP noise."""
  ref = poses[0]
  translations = np.stack(
    [np.asarray(p.translation(), dtype=np.float64) for p in poses], axis=0
  )
  log_sum = np.zeros(3, dtype=np.float64)
  for p in poses:
    log_sum += gtsam.Rot3.Logmap(ref.rotation().between(p.rotation()))
  delta = gtsam.Rot3.Expmap(log_sum / len(poses))
  return gtsam.Pose3(
    ref.rotation().compose(delta), gtsam.Point3(*translations.mean(axis=0))
  )


def _finger_joint_angles_per_kf(
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  gripper_tag_id: int,
  finger_tag_id: int,
  tag_size: float,
  n_frames: int,
) -> list[float]:
  """``Logmap(grip_T_fing)[2]`` per keyframe — the signed rotation about the
  URDF revolute-joint axis (``+z`` by URDF convention). The gripper tag is
  rigidly mounted to the body so its ``cam_T_tag`` is averaged once over all
  visible keyframes; the finger tag's ``cam_T_tag`` is recovered per kf."""
  cam_T_grip_per_kf = _per_kf_cam_T_tag(
    tags_per_kf, lens, gripper_tag_id, tag_size, n_frames
  )
  cam_T_fing_per_kf = _per_kf_cam_T_tag(
    tags_per_kf, lens, finger_tag_id, tag_size, n_frames
  )
  observed_grip = [p for p in cam_T_grip_per_kf if p is not None]
  if not observed_grip:
    return [float("nan")] * n_frames
  cam_T_grip = _mean_pose(observed_grip)
  out: list[float] = [float("nan")] * n_frames
  for kf in range(n_frames):
    cam_T_fing = cam_T_fing_per_kf[kf]
    if cam_T_fing is None:
      continue
    grip_T_fing = cam_T_grip.between(cam_T_fing)
    out[kf] = float(gtsam.Rot3.Logmap(grip_T_fing.rotation())[2])
  return out


def _gyro_integrated_rotations(
  imu_t: Float64Array, gyro_body: Float64Array, kf_times: Float64Array
) -> list[Float64Array]:
  """Per-consecutive-kf body rotation matrix from raw gyro integration.
  ``gyro_body`` is the bias-uncorrected angular velocity in the body
  frame (rad/s). Each output is ``R_body_kf-to-body_kf+1`` so a vector
  fixed in the world satisfies ``v_body_k+1 = R.T @ v_body_k``."""
  out: list[Float64Array] = []
  for k in range(len(kf_times) - 1):
    mask = (imu_t > kf_times[k]) & (imu_t <= kf_times[k + 1])
    if not mask.any():
      out.append(np.eye(3, dtype=np.float64))
      continue
    times = imu_t[mask]
    gyros = gyro_body[mask]
    R = np.eye(3, dtype=np.float64)
    if len(times) < 2:
      out.append(R)
      continue
    dts = np.diff(times)
    for i in range(len(dts)):
      omega = gyros[i]
      dt = float(dts[i])
      angle = float(np.linalg.norm(omega) * dt)
      if angle > 1e-9:
        axis = omega * dt / angle
        K = np.array(
          [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
          ],
          dtype=np.float64,
        )
        R_step = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        R = R @ R_step
    out.append(R)
  return out


def _make_imu_klt_predictor(
  lens_params: MeiParams,
  R_body_lens: Float64Array,
  body_rotations_between: list[Float64Array],
) -> Callable[[int, Float64Array], Float64Array]:
  """Predict ``frame_k`` pixel positions from ``frame_{k-1}`` pixels by
  rotating the bearings through the IMU-integrated body rotation between
  the two kfs and reprojecting through the same lens. Used as
  ``rotation_predictor`` for ``klt_track``."""
  R_lens_body = R_body_lens.T

  def predictor(k: int, prev_pixels: Float64Array) -> Float64Array:
    if k - 1 >= len(body_rotations_between):
      return np.full_like(prev_pixels, np.nan)
    R_b = body_rotations_between[k - 1]
    R_lens_change = R_lens_body @ R_b.T @ R_body_lens
    bearings = mei_unproject(prev_pixels.astype(np.float64), lens_params)
    bearings_new = bearings @ R_lens_change.T
    return mei_project(bearings_new, lens_params)

  return predictor


def prepare_tracker(name: str, args: TrackerArgs) -> PreparedTracker:
  """Run every per-tracker stage. Returns a ``PreparedTracker`` ready for
  joint-graph composition. Body poses are in this tracker's own gravity-
  aligned visual world; the caller aligns trackers to a shared frame via
  ``anchor_pose_world_local``."""
  data = load_insv(args.insv_path)
  extrinsics = parse_insv_extrinsics(data)
  front_lens, rear_lens = lens_setups(data, extrinsics)

  front_frames, rear_frames = decode_dual_keyframes(
    args.insv_path, args.start, args.duration, data.frame_rate, args.keyframe_every
  )
  n_frames = len(front_frames)
  if n_frames < 3:
    raise RuntimeError(f"only {n_frames} keyframes decoded — window too short")
  kf_times = keyframe_timestamps(
    args.start, data.frame_rate, args.keyframe_every, n_frames
  )

  front_tags_per_kf = detect_all_tags(front_frames)
  rear_tags_per_kf = detect_all_tags(rear_frames)

  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]] = {}
  if args.anchor.tag_id >= 0:
    for side, lens, tags in (
      ("front", front_lens, front_tags_per_kf),
      ("rear", rear_lens, rear_tags_per_kf),
    ):
      for kf, pose in anchor_pose_from_tags(
        tags, args.anchor.tag_id, args.anchor.tag_size, lens.params
      ).items():
        anchor_obs_per_kf.setdefault(kf, {})[side] = pose

  front_masks = _build_non_anchor_tag_masks(
    front_frames, front_tags_per_kf, args.anchor.tag_id, args.frontend.mask_tag_pad_px
  )
  rear_masks = _build_non_anchor_tag_masks(
    rear_frames, rear_tags_per_kf, args.anchor.tag_id, args.frontend.mask_tag_pad_px
  )
  static_rects = list(args.frontend.mask_rear_rects)
  _apply_static_mask_rects(rear_masks, static_rects)

  cfg = FrontendConfig(
    n_features=args.frontend.orb_n_features,
    ransac_threshold_px=args.frontend.ransac_threshold_px,
    max_half_angle_deg=args.frontend.max_half_angle_deg,
  )
  imu_to_body_pre = np.asarray(args.imu.imu_to_body, dtype=np.float64)
  gyro_body_pre = (
    np.deg2rad(np.asarray(data.gyro_deg, dtype=np.float64)) @ imu_to_body_pre.T
  )
  imu_t_pre = np.asarray(data.imu_timestamps, dtype=np.float64)
  body_rotations_between = _gyro_integrated_rotations(
    imu_t_pre, gyro_body_pre, kf_times
  )
  front_predictor = _make_imu_klt_predictor(
    front_lens.params, front_lens.extrinsic.rotation().matrix(), body_rotations_between
  )
  rear_predictor = _make_imu_klt_predictor(
    rear_lens.params, rear_lens.extrinsic.rotation().matrix(), body_rotations_between
  )
  front_orb, front_tracks = run_lens_frontend(
    front_frames,
    front_lens.params,
    cfg,
    masks=front_masks,
    rotation_predictor=front_predictor,
  )
  rear_orb, rear_tracks = run_lens_frontend(
    rear_frames,
    rear_lens.params,
    cfg,
    masks=rear_masks,
    rotation_predictor=rear_predictor,
  )
  front_kept = {
    t: o for t, o in front_tracks.items() if len(o) >= args.frontend.min_track_length
  }
  rear_kept = {
    t: o for t, o in rear_tracks.items() if len(o) >= args.frontend.min_track_length
  }
  front_rejected: dict[int, dict[int, Float64Array]] = {}
  rear_rejected: dict[int, dict[int, Float64Array]] = {}
  if args.frontend.min_parallax_deg > 0.0:
    front_kept, front_rejected = filter_tracks_by_parallax(
      front_kept, front_lens.params, args.frontend.min_parallax_deg
    )
    rear_kept, rear_rejected = filter_tracks_by_parallax(
      rear_kept, rear_lens.params, args.frontend.min_parallax_deg
    )

  f_first, f_last, f_shared = pick_init_pair(
    front_kept, n_frames, args.frontend.init_min_shared_tracks
  )
  r_first, r_last, r_shared = pick_init_pair(
    rear_kept, n_frames, args.frontend.init_min_shared_tracks
  )
  if f_shared >= r_shared:
    init_tracks, init_params, init_extrinsic = (
      front_kept,
      front_lens.params,
      front_lens.extrinsic,
    )
    init_first, init_last, init_shared = f_first, f_last, f_shared
  else:
    init_tracks, init_params, init_extrinsic = (
      rear_kept,
      rear_lens.params,
      rear_lens.extrinsic,
    )
    init_first, init_last, init_shared = r_first, r_last, r_shared
  if init_shared < 8:
    raise RuntimeError(f"{name}: no lens has ≥8 shared tracks (best={init_shared})")

  pix_a = np.asarray(
    [
      obs[init_first]
      for obs in init_tracks.values()
      if init_first in obs and init_last in obs
    ],
    dtype=np.float64,
  )
  pix_b = np.asarray(
    [
      obs[init_last]
      for obs in init_tracks.values()
      if init_first in obs and init_last in obs
    ],
    dtype=np.float64,
  )
  bears_a = mei_unproject(pix_a, init_params)
  bears_b = mei_unproject(pix_b, init_params)
  r_cw_last, t_cw_last, _ = two_view_essential(bears_a, bears_b, threshold=0.01)
  cam_poses = interpolate_camera_poses(r_cw_last, t_cw_last, n_frames)
  body_poses_unscaled = body_poses_from_camera_poses(cam_poses, init_extrinsic)

  imu_to_body = np.asarray(args.imu.imu_to_body, dtype=np.float64)
  raw_gyro = np.deg2rad(np.asarray(data.gyro_deg, dtype=np.float64)) @ imu_to_body.T
  raw_accel = np.asarray(data.accel, dtype=np.float64) @ imu_to_body.T
  imu = ImuSamples(
    t=np.asarray(data.imu_timestamps, dtype=np.float64), gyro=raw_gyro, accel=raw_accel
  )
  raw_factory_bias = factory_imu_bias(data.gyro_calib_numbers)
  bias0 = gtsam.imuBias.ConstantBias(
    imu_to_body @ np.asarray(raw_factory_bias.accelerometer(), dtype=np.float64),
    imu_to_body @ np.asarray(raw_factory_bias.gyroscope(), dtype=np.float64),
  )
  windows = find_stationary_windows(imu, bias0)
  clip_end = imu.t[-1] if args.duration is None else args.start + args.duration
  windows = [(s, e) for s, e in windows if args.start <= imu.t[s] <= clip_end]
  gravity_world_initial: Float64Array | None = None
  if windows:
    first_start, first_end = windows[0]
    stationary_mean = (imu.accel[first_start:first_end] - bias0.accelerometer()).mean(
      axis=0
    )
    stationary_mag = float(np.linalg.norm(stationary_mean))
    if 8.0 < stationary_mag < 11.0 and abs(stationary_mag - 9.81) > 0.05:
      accel_scale = 9.81 / stationary_mag
      imu = ImuSamples(
        t=imu.t, gyro=imu.gyro, accel=(imu.accel * accel_scale).astype(np.float64)
      )
      stationary_mean = stationary_mean * accel_scale
    gravity_body = -stationary_mean / float(np.linalg.norm(stationary_mean)) * 9.81
    t_stationary_mid = 0.5 * (imu.t[first_start] + imu.t[first_end - 1])
    kf_for_g = int(np.argmin(np.abs(kf_times - t_stationary_mid)))
    gravity_world_initial = (
      body_poses_unscaled[kf_for_g].rotation().matrix() @ gravity_body
    )

  front_unscaled = triangulate_lens(
    front_kept,
    body_poses_unscaled,
    front_lens.extrinsic,
    front_lens.params,
    args.frontend.min_track_length,
    min_landmark_depth=0.01,
    max_landmark_depth=1000.0,
  )
  rear_unscaled = triangulate_lens(
    rear_kept,
    body_poses_unscaled,
    rear_lens.extrinsic,
    rear_lens.params,
    args.frontend.min_track_length,
    min_landmark_depth=0.01,
    max_landmark_depth=1000.0,
  )
  body_poses_visual = list(body_poses_unscaled)

  # Without a stationary window the LSQ-fit gravity is the only signal we
  # have; preintegration uses a placeholder vertical gravity, then the
  # alignment runs against the same data and the LSQ-fit value sticks.
  pim_gravity = (
    (
      float(gravity_world_initial[0]),
      float(gravity_world_initial[1]),
      float(gravity_world_initial[2]),
    )
    if gravity_world_initial is not None
    else (0.0, 0.0, -9.81)
  )
  params_pim = make_preintegration_params(
    gyro_noise_sigma=args.imu.gyro_noise_sigma,
    accel_noise_sigma=args.imu.accel_noise_sigma,
    gyro_walk=args.imu.gyro_random_walk,
    accel_walk=args.imu.accel_random_walk,
    gravity_world=pim_gravity,
  )
  pim_init = preintegrate_between_keyframes(params_pim, imu, kf_times, bias0)
  alignment = visual_inertial_align(body_poses_visual, pim_init)
  if alignment.scale < 0.0:
    origin = np.asarray(body_poses_visual[0].translation(), dtype=np.float64)
    body_poses_visual = [
      gtsam.Pose3(
        p.rotation(),
        gtsam.Point3(*(2.0 * origin - np.asarray(p.translation(), dtype=np.float64))),
      )
      for p in body_poses_visual
    ]
    alignment = visual_inertial_align(body_poses_visual, pim_init)
  # Override LSQ-fit gravity with the IMU stationary-window estimate so the
  # world frame stays gravity-aligned (LSQ-fit gravity can drift up to ~30°
  # from the IMU truth, which then tilts the rerun viewer's z-up convention).
  if gravity_world_initial is not None:
    alignment = Alignment(
      scale=alignment.scale,
      gravity_world=gravity_world_initial,
      velocities=alignment.velocities,
    )
  body_poses, world_rotation = apply_alignment_to_poses(body_poses_visual, alignment)
  world_rot_mat = world_rotation.matrix()
  velocities = [world_rot_mat @ v for v in alignment.velocities]
  params_aligned = make_preintegration_params(
    gyro_noise_sigma=args.imu.gyro_noise_sigma,
    accel_noise_sigma=args.imu.accel_noise_sigma,
    gyro_walk=args.imu.gyro_random_walk,
    accel_walk=args.imu.accel_random_walk,
    gravity_world=(0.0, 0.0, -9.81),
  )
  pim = preintegrate_between_keyframes(params_aligned, imu, kf_times, bias0)

  use_anchor = args.anchor.tag_id >= 0 and bool(anchor_obs_per_kf)
  anchor_pose_world_local: gtsam.Pose3 | None = None
  if use_anchor:
    first_kf = min(anchor_obs_per_kf.keys())
    sides = anchor_obs_per_kf[first_kf]
    first_lens = front_lens if "front" in sides else rear_lens
    first_cam_T_tag = sides.get("front", sides.get("rear"))
    assert first_cam_T_tag is not None
    anchor_pose_world_local = (
      body_poses[first_kf].compose(first_lens.extrinsic).compose(first_cam_T_tag)
    )
    for kf, kf_sides in anchor_obs_per_kf.items():
      lens = front_lens if "front" in kf_sides else rear_lens
      cam_T_tag = kf_sides.get("front", kf_sides.get("rear"))
      assert cam_T_tag is not None
      body_poses[kf] = anchor_pose_world_local.compose(
        lens.extrinsic.compose(cam_T_tag).inverse()
      )
    sorted_kfs = sorted(anchor_obs_per_kf.keys())
    if len(sorted_kfs) >= 2:
      jumps = [
        float(
          np.linalg.norm(
            np.asarray(body_poses[sorted_kfs[i]].translation())
            - np.asarray(body_poses[sorted_kfs[i - 1]].translation())
          )
        )
        for i in range(1, len(sorted_kfs))
      ]
      i = int(np.argmax(jumps))
      print(
        f"  [{name}] max anchor-init Δp: {jumps[i]:.3f} m at kf {sorted_kfs[i + 1]}"
      )
    # IMU forward-propagation from each anchor kf fills the gap kfs with the
    # IMU-integrated trajectory instead of constant-velocity pose interpolation.
    first_anchor = sorted_kfs[0]
    for k in range(first_anchor):
      body_poses[k] = body_poses[first_anchor]
      velocities[k] = velocities[first_anchor]
    for i in range(len(sorted_kfs)):
      a = sorted_kfs[i]
      b = sorted_kfs[i + 1] if i + 1 < len(sorted_kfs) else n_frames
      if b - a <= 1:
        continue
      nav = gtsam.NavState(body_poses[a], velocities[a])
      for k in range(a + 1, b):
        nav = pim[k - 1].predict(nav, bias0)
        if k in anchor_obs_per_kf:
          continue
        body_poses[k] = nav.pose()
        velocities[k] = np.asarray(nav.velocity(), dtype=np.float64)

  align_mat = alignment.scale * world_rotation.matrix()
  front_unscaled_metric = {
    tid: align_mat @ np.asarray(front_unscaled[tid], dtype=np.float64)
    for tid in front_unscaled
  }
  rear_unscaled_metric = {
    tid: align_mat @ np.asarray(rear_unscaled[tid], dtype=np.float64)
    for tid in rear_unscaled
  }
  if use_anchor:
    anchor_kfs = set(anchor_obs_per_kf.keys())
    front_kept = {
      t: front_kept[t] for t in front_kept if anchor_kfs & front_kept[t].keys()
    }
    rear_kept = {t: rear_kept[t] for t in rear_kept if anchor_kfs & rear_kept[t].keys()}
    front_points = triangulate_lens(
      {t: front_kept[t] for t in front_kept if t in front_unscaled},
      body_poses,
      front_lens.extrinsic,
      front_lens.params,
      args.frontend.min_track_length,
    )
    rear_points = triangulate_lens(
      {t: rear_kept[t] for t in rear_kept if t in rear_unscaled},
      body_poses,
      rear_lens.extrinsic,
      rear_lens.params,
      args.frontend.min_track_length,
    )
  else:
    front_points = front_unscaled_metric
    rear_points = rear_unscaled_metric

  front_tracks_tri = {t: front_kept[t] for t in front_points if t in front_kept}
  rear_tracks_tri = {t: rear_kept[t] for t in rear_points if t in rear_kept}

  v_max = float(max(np.linalg.norm(v) for v in velocities))
  print(f"  [{name}] VIO LSQ |v|max={v_max:.2f} m/s")
  biases = [bias0 for _ in range(n_frames)]

  gripper_obs = anchor_pose_from_tags(
    rear_tags_per_kf,
    anchor_tag_id=args.gripper_tag_id,
    tag_size=args.gripper.tag_size,
    params=rear_lens.params,
  )
  body_T_gripper_tag: gtsam.Pose3 | None = None
  if gripper_obs:
    body_T_gripper_tag = _mean_pose(
      [rear_lens.extrinsic.compose(p) for p in gripper_obs.values()]
    )

  def _per_kf_body_T_finger(finger_id: int) -> dict[int, gtsam.Pose3]:
    if finger_id <= 0:
      return {}
    out: dict[int, gtsam.Pose3] = {}
    for kf in range(min(n_frames, len(rear_tags_per_kf))):
      corners = next(
        (d.corners for d in rear_tags_per_kf[kf] if int(d.tag_id) == finger_id),
        None,
      )
      if corners is None:
        continue
      cam_T_finger = estimate_camera_to_tag_pose(
        corners, args.gripper.tag_size, rear_lens.params
      )
      if cam_T_finger is None:
        continue
      out[kf] = rear_lens.extrinsic.compose(cam_T_finger)
    return out

  body_T_left_finger_per_kf = _per_kf_body_T_finger(args.left_finger_tag_id)
  body_T_right_finger_per_kf = _per_kf_body_T_finger(args.right_finger_tag_id)

  joint_angles: dict[str, list[float]] = {}
  for joint_name, finger_id in (
    ("left_gripper", args.left_finger_tag_id),
    ("right_gripper", args.right_finger_tag_id),
  ):
    if finger_id <= 0:
      continue
    joint_angles[joint_name] = _finger_joint_angles_per_kf(
      rear_tags_per_kf,
      rear_lens,
      gripper_tag_id=args.gripper_tag_id,
      finger_tag_id=finger_id,
      tag_size=args.gripper.tag_size,
      n_frames=n_frames,
    )

  return PreparedTracker(
    name=name,
    body_poses=body_poses,
    velocities=velocities,
    biases=biases,
    preintegrated=pim,
    front_lens=front_lens,
    rear_lens=rear_lens,
    front_tracks_tri=front_tracks_tri,
    rear_tracks_tri=rear_tracks_tri,
    front_points=front_points,
    rear_points=rear_points,
    anchor_obs_per_kf=anchor_obs_per_kf,
    anchor_pose_world_local=anchor_pose_world_local,
    body_T_gripper_tag=body_T_gripper_tag,
    joint_angles=joint_angles,
    kf_times=kf_times,
    front_frames=front_frames,
    rear_frames=rear_frames,
    front_keypoints=[f.keypoints for f in front_orb],
    rear_keypoints=[f.keypoints for f in rear_orb],
    front_tags_per_kf=front_tags_per_kf,
    rear_tags_per_kf=rear_tags_per_kf,
    front_rejected=front_rejected,
    rear_rejected=rear_rejected,
    rear_static_mask_rects=static_rects,
    body_T_left_finger_per_kf=body_T_left_finger_per_kf,
    body_T_right_finger_per_kf=body_T_right_finger_per_kf,
    left_finger_tag_id=args.left_finger_tag_id,
    right_finger_tag_id=args.right_finger_tag_id,
    n_frames=n_frames,
  )
