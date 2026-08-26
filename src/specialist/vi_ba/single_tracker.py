"""One tracker's lenses and IMU prepared for the joint graph."""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, TypeVar

import gtsam
import numpy as np
import numpy.typing as npt
from specialist_camera import mei_unproject
from specialist_dataset import schema
from specialist_dataset.frames import LensFrames, LensVideo

from specialist.record.session import (
  SessionRecording,
  covered_rows,
  fps_from_times,
  nearest_rows,
)
from specialist.vi_ba.anchor_detection import (
  TagDetection,
  anchor_pose_from_tags,
  apply_static_mask_rects,
  configure_detector,
  detect_all_tags,
  non_anchor_tag_masks,
)
from specialist.vi_ba.build import LensChannel, LensSetup
from specialist.vi_ba.configs import (
  FrontendCfg,
  TrackerArgs,
)
from specialist.vi_ba.features import (
  Frame,
  Rect,
  available_ram,
  detect_orb,
  mask_plane,
  parallel_map,
)
from specialist.vi_ba.finger_kinematics import FingerReproj, recover_gripper_tags
from specialist.vi_ba.geometry import translation
from specialist.vi_ba.imu import (
  GRAVITY,
  GRAVITY_WORLD,
  ImuSamples,
  find_stationary_windows,
  keyframe_delta_rotations,
  make_preintegration_params,
  preintegrate_between_keyframes,
)
from specialist.vi_ba.init import (
  Alignment,
  apply_alignment_to_poses,
  body_poses_from_camera_poses,
  fill_camera_poses,
  incremental_sfm,
  interpolate_camera_poses,
  pick_init_pair,
  reject_vo_pose_outliers,
  two_view_essential,
  visual_inertial_align,
)
from specialist.vi_ba.tag_discovery import (
  GripperTags,
  TagDiscoveryError,
  discover_gripper_tags,
  uniform_sample_rows,
)
from specialist.vi_ba.tracks import (
  add_loop_landmarks,
  bearings_by_track,
  cap_landmarks,
  filter_tracks_by_parallax,
  loop_closure_tracks,
  run_lens_frontend,
  subsample_obs,
  triangulate_lens,
)

Float64Array = npt.NDArray[np.float64]

_A = TypeVar("_A")

# Keyframes read at once from a banded lens.
_PREP_BAND = 256

# Share of the available memory the concurrent decodes may claim between them.
_PREP_MEMORY_SHARE = 0.35


__all__ = [
  "PreparedTracker",
  "captures_that_fit",
  "discover_tracker_tags",
  "prepare_tracker",
]


def resolve_keyframe_every(keyframe_every: int, fps: float) -> int:
  """The keyframe stride: ``keyframe_every`` when positive, else the ~0.1 s lattice
  stride; raise when an override is off that lattice."""
  stride = schema.keyframe_stride(fps)
  if keyframe_every > 0 and keyframe_every % stride:
    raise ValueError(
      f"--keyframe-every {keyframe_every} does not land on the {stride}-frame "
      f"keyframe lattice at {fps:.2f} fps; use a multiple of {stride}"
    )
  return keyframe_every if keyframe_every > 0 else stride


def release_freed_memory() -> None:
  """Hand freed heap back to the OS; a no-op without glibc's ``malloc_trim``."""
  try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
  except (OSError, AttributeError):
    pass


def _per_band(
  fn: Callable[[list[np.ndarray]], list[_A]], frames: Sequence[np.ndarray]
) -> list[_A]:
  """Map ``fn`` band by band, so a banded sequence decodes each band once, in order."""
  out: list[_A] = []
  for begin in range(0, len(frames), _PREP_BAND):
    end = min(begin + _PREP_BAND, len(frames))
    out.extend(fn([frames[i] for i in range(begin, end)]))
  return out


def _enough_registered(
  n_registered: int, n_keyframes: int, ratio: float = 0.25
) -> bool:
  """A seed needs ``ratio`` of the keyframes, and at least eight."""
  return n_registered >= max(8, int(n_keyframes * ratio))


@dataclass
class PreparedTracker:
  """Everything one tracker contributes to the joint graph + viz."""

  name: str
  body_poses: list[gtsam.Pose3]
  velocities: list[Float64Array]
  biases: list[gtsam.imuBias.ConstantBias]
  preintegrated: list[gtsam.PreintegratedCombinedMeasurements]
  front: LensChannel
  rear: LensChannel
  # ``{tag_id: {kf: {side: cam_T_tag}}}``.
  anchor_obs_by_tag: dict[int, dict[int, dict[str, gtsam.Pose3]]]
  anchor_locals_by_tag: dict[int, gtsam.Pose3]
  body_T_base_tag: gtsam.Pose3
  # URDF joint name -> per-keyframe angle about the joint z-axis, NaN where unseen.
  joint_angles: dict[str, list[float]]
  keyframe_times: Float64Array
  # Session-clock frame per keyframe; ``schema.keyframe_index`` maps it to the timeline.
  keyframe_rows: npt.NDArray[np.int64]
  # URDF joint name -> kinematic inputs for the joint-angle reprojection factors.
  finger_reproj: dict[str, FingerReproj]
  # This tracker's own tag ids, discovered from its rear lens.
  tags: GripperTags
  # Body-frame, rescaled IMU samples, for IMU-rate propagation between keyframes.
  imu: ImuSamples
  # Per-joint raw ``(times, angles)`` at the finger rate; export pins them to the solve.
  finger_frame_angles: dict[str, tuple[Float64Array, Float64Array]] = field(
    default_factory=dict
  )

  @property
  def n_keyframes(self) -> int:
    return len(self.body_poses)

  @property
  def lenses(self) -> tuple[LensChannel, LensChannel]:
    """Both lens channels, front first."""
    return (self.front, self.rear)


@dataclass
class _LensPrep:
  """One lens's prep state: tag detections in, surviving tracks and landmarks out."""

  side: schema.Side
  setup: LensSetup
  tags: list[list[TagDetection]]
  static_mask_rects: list[Rect] = field(default_factory=list)
  keypoints: list[Float64Array] = field(default_factory=list)
  kept: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  rejected: dict[int, dict[int, Float64Array]] = field(default_factory=dict)
  landmarks_prealign: dict[int, Float64Array] = field(default_factory=dict)
  points: dict[int, Float64Array] = field(default_factory=dict)
  tracks: dict[int, dict[int, Float64Array]] = field(default_factory=dict)

  def channel(self, n_keyframes: int) -> LensChannel:
    """This lens's onward share, leaving ``kept`` and ``landmarks_prealign`` behind."""
    return LensChannel(
      side=self.side,
      setup=self.setup,
      tags=self.tags,
      keypoints=self.keypoints[:n_keyframes],
      tracks=self.tracks,
      points=self.points,
      rejected=self.rejected,
      static_mask_rects=self.static_mask_rects,
    )


class _LensSide(NamedTuple):
  """One lens's prep state and its gyro-derived rotation increments for anchor PnP."""

  lens: _LensPrep
  rotations: list[Float64Array]


def _detect_anchor_obs(
  tag_id: int,
  tag_size: float,
  lens_sides: tuple[_LensSide, ...],
) -> dict[int, dict[str, gtsam.Pose3]]:
  """``{kf: {side: cam_T_tag}}`` for one anchor id over the given lenses, ungated."""
  obs: dict[int, dict[str, gtsam.Pose3]] = {}
  for lens, rotations in lens_sides:
    for kf, pose in anchor_pose_from_tags(
      lens.tags,
      tag_id,
      tag_size,
      lens.setup.params,
      kf_rotations=rotations,
    ).items():
      obs.setdefault(kf, {})[lens.side] = pose
  return obs


def _choose_prep_anchor(
  tag_ids: tuple[int, ...],
  obs_by_tag: dict[int, dict[int, dict[str, gtsam.Pose3]]],
) -> int:
  """The id sighted at most keyframes; a tie goes to the earlier ``tag_ids`` entry."""
  if not tag_ids:
    raise ValueError("no anchor tag id is enabled; prep has nothing to ground on")
  return max(tag_ids, key=lambda tid: len(obs_by_tag[tid]))


def _first_sighting_local(
  obs: dict[int, dict[str, gtsam.Pose3]],
  front_lens: LensSetup,
  rear_lens: LensSetup,
  body_poses: list[gtsam.Pose3],
) -> gtsam.Pose3:
  """An anchor's local-world pose from its first sighting, front lens preferred."""
  first_kf = min(obs)
  lens, cam_T_tag = _side_lens_and_pose(obs[first_kf], front_lens, rear_lens)
  return body_poses[first_kf].compose(lens.extrinsic).compose(cam_T_tag)


def _anchor_pose_seed(
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]],
  front_extrinsic: gtsam.Pose3,
  n_keyframes: int,
) -> list[gtsam.Pose3] | None:
  """Anchor-PnP body poses, gaps filled; ``None`` when too few keyframes see it."""
  cam_poses = {
    kf: sides["front"].inverse()
    for kf, sides in anchor_obs_per_kf.items()
    if "front" in sides
  }
  if not _enough_registered(len(cam_poses), n_keyframes):
    return None
  return body_poses_from_camera_poses(
    fill_camera_poses(cam_poses, n_keyframes), front_extrinsic
  )


def _anchor_metric_scale(
  body_poses_unscaled: list[gtsam.Pose3],
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]],
  front_lens: LensSetup,
  rear_lens: LensSetup,
) -> float | None:
  """Metric scale: anchor-implied over visual body-position spread, or ``None``."""
  vo: list[Float64Array] = []
  anc: list[Float64Array] = []
  for kf, sides in anchor_obs_per_kf.items():
    lens, cam_T_tag = _side_lens_and_pose(sides, front_lens, rear_lens)
    body_in_tag = lens.extrinsic.compose(cam_T_tag).inverse()
    anc.append(translation(body_in_tag))
    vo.append(translation(body_poses_unscaled[kf]))
  if len(vo) < 3:
    return None
  vo_arr = np.asarray(vo, dtype=np.float64)
  anc_arr = np.asarray(anc, dtype=np.float64)
  vo_spread = float(np.sqrt(((vo_arr - vo_arr.mean(axis=0)) ** 2).sum()))
  anc_spread = float(np.sqrt(((anc_arr - anc_arr.mean(axis=0)) ** 2).sum()))
  if vo_spread < 1e-9:
    return None
  return anc_spread / vo_spread


def _side_lens_and_pose(
  sides: dict[str, gtsam.Pose3], front_lens: LensSetup, rear_lens: LensSetup
) -> tuple[LensSetup, gtsam.Pose3]:
  """The anchor observation's lens and ``cam_T_tag``, front lens preferred."""
  if "front" in sides:
    return front_lens, sides["front"]
  return rear_lens, sides["rear"]


def _imu_and_initial_gravity(
  imu: ImuSamples,
  raw_factory_bias: gtsam.imuBias.ConstantBias,
  imu_to_body: Float64Array,
  keyframe_times: Float64Array,
  body_poses_prealign: list[gtsam.Pose3],
) -> tuple[ImuSamples, gtsam.imuBias.ConstantBias, Float64Array | None]:
  """IMU samples, body-frame bias, world gravity or ``None`` if no stationary window."""
  bias0 = gtsam.imuBias.ConstantBias(
    imu_to_body @ np.asarray(raw_factory_bias.accelerometer(), dtype=np.float64),
    imu_to_body @ np.asarray(raw_factory_bias.gyroscope(), dtype=np.float64),
  )
  windows = find_stationary_windows(imu, bias0)
  gravity_world_initial: Float64Array | None = None
  if windows:
    first_start, first_end = windows[0]
    stationary_mean = (imu.accel[first_start:first_end] - bias0.accelerometer()).mean(
      axis=0
    )
    stationary_mag = float(np.linalg.norm(stationary_mean))
    if 8.0 < stationary_mag < 11.0 and abs(stationary_mag - GRAVITY) > 0.05:
      accel_scale = GRAVITY / stationary_mag
      imu = ImuSamples(
        t=imu.t, gyro=imu.gyro, accel=(imu.accel * accel_scale).astype(np.float64)
      )
      stationary_mean = stationary_mean * accel_scale
    gravity_body = -stationary_mean / float(np.linalg.norm(stationary_mean)) * GRAVITY
    t_stationary_mid = 0.5 * (imu.t[first_start] + imu.t[first_end - 1])
    kf_for_g = int(np.argmin(np.abs(keyframe_times - t_stationary_mid)))
    gravity_world_initial = (
      body_poses_prealign[kf_for_g].rotation().matrix() @ gravity_body
    )
  return imu, bias0, gravity_world_initial


def _seed_body_poses(
  name: str,
  front: _LensPrep,
  rear: _LensPrep,
  n_keyframes: int,
  anchor_obs_per_kf: dict[int, dict[str, gtsam.Pose3]],
  cfg: FrontendCfg,
) -> tuple[list[gtsam.Pose3], bool]:
  """The pre-alignment body-pose seed and whether it came from the anchor."""
  sfm_cams, _ = incremental_sfm(
    bearings_by_track(front.kept, front.setup.params),
    min_shared=cfg.init_min_shared_tracks,
  )
  sfm_cams, vo_outliers = reject_vo_pose_outliers(sfm_cams)
  if vo_outliers:
    print(f"  [{name}/vo] rejected {len(vo_outliers)} rotation-spike kfs")
  if _enough_registered(len(sfm_cams), n_keyframes, cfg.vo_min_registered_ratio):
    print(f"  [{name}/vo] registered {len(sfm_cams)}/{n_keyframes} keyframes")
    filled = fill_camera_poses(sfm_cams, n_keyframes)
    return body_poses_from_camera_poses(filled, front.setup.extrinsic), False

  anchor_seed = _anchor_pose_seed(anchor_obs_per_kf, front.setup.extrinsic, n_keyframes)
  if anchor_seed is not None:
    n_anchor_kf = sum("front" in sides for sides in anchor_obs_per_kf.values())
    print(
      f"  [{name}/vo] starved ({len(sfm_cams)}/{n_keyframes}); "
      f"anchor-pose init from {n_anchor_kf} anchor keyframes"
    )
    return anchor_seed, True
  print(
    f"  [{name}/vo] only {len(sfm_cams)}/{n_keyframes} "
    f"keyframes registered; using straight-line init"
  )
  init_ch, (init_first, init_last, init_shared) = max(
    (
      (ch, pick_init_pair(ch.kept, n_keyframes, cfg.init_min_shared_tracks))
      for ch in (front, rear)
    ),
    key=lambda candidate: candidate[1][2],
  )
  if init_shared < 8:
    raise RuntimeError(f"{name}: no lens has ≥8 shared tracks (best={init_shared})")
  if init_first:
    print(f"  [{name}/vo] init pair based at keyframe {init_first}")
  shared = [
    obs for obs in init_ch.kept.values() if init_first in obs and init_last in obs
  ]
  pix_a = np.asarray([obs[init_first] for obs in shared], dtype=np.float64)
  pix_b = np.asarray([obs[init_last] for obs in shared], dtype=np.float64)
  r_cw_last, t_cw_last, _ = two_view_essential(
    mei_unproject(pix_a, init_ch.setup.params),
    mei_unproject(pix_b, init_ch.setup.params),
  )
  cam_poses = interpolate_camera_poses(
    r_cw_last, t_cw_last, n_keyframes, init_first=init_first, init_last=init_last
  )
  return body_poses_from_camera_poses(cam_poses, init_ch.setup.extrinsic), False


def _align_visual_inertial(
  name: str,
  imu: ImuSamples,
  keyframe_times: Float64Array,
  bias0: gtsam.imuBias.ConstantBias,
  body_poses_prealign: list[gtsam.Pose3],
  gravity_world_initial: Float64Array | None,
  fixed_scale: float | None,
) -> tuple[
  list[gtsam.Pose3],
  list[Float64Array],
  list[gtsam.PreintegratedCombinedMeasurements],
  Float64Array,
]:
  """Gravity-aligned poses, velocities, preintegration and the prep-frame 3x3 map."""

  def pim_params(
    gravity: tuple[float, float, float],
  ) -> gtsam.PreintegrationCombinedParams:
    return make_preintegration_params(gravity_world=gravity)

  # Gravity never enters the deltas, so one PIM serves alignment and export.
  pim = preintegrate_between_keyframes(
    pim_params(GRAVITY_WORLD), imu, keyframe_times, bias0
  )

  def align(poses: list[gtsam.Pose3]) -> Alignment:
    return visual_inertial_align(
      poses, pim, fixed_gravity=gravity_world_initial, fixed_scale=fixed_scale
    )

  alignment = align(body_poses_prealign)
  if alignment.scale < 0.0:
    origin = translation(body_poses_prealign[0])
    body_poses_prealign = [
      gtsam.Pose3(p.rotation(), gtsam.Point3(*(2.0 * origin - translation(p))))
      for p in body_poses_prealign
    ]
    alignment = align(body_poses_prealign)
  # The least-squares gravity can sit ~30° off the IMU truth.
  if gravity_world_initial is not None:
    alignment = Alignment(
      scale=alignment.scale,
      gravity_world=gravity_world_initial,
      velocities=alignment.velocities,
    )
  print(f"  [{name}/vi] scale={alignment.scale:.4f}")
  body_poses, world_rotation = apply_alignment_to_poses(body_poses_prealign, alignment)
  world_rot = world_rotation.matrix()
  velocities = [world_rot @ v for v in alignment.velocities]
  return body_poses, velocities, pim, alignment.scale * world_rot


def captures_that_fit(session: SessionRecording, names: Sequence[str]) -> int:
  """How many captures may prep at once within the memory on hand."""
  if len(names) <= 1:
    return len(names)
  height, width = session.decoded_shape(names[0])
  per_capture = _PREP_BAND * width * height  # one lens's resident band at a time
  shared = session.path.stat().st_size
  budget = available_ram() * _PREP_MEMORY_SHARE - shared
  affordable = int(budget / per_capture)
  return max(1, min(len(names), affordable))


def keyframe_grid(
  times: Float64Array,
  covered: npt.NDArray[np.int64],
  fps: float,
  keyframe_every: int,
) -> npt.NDArray[np.int64]:
  """Keyframe rows of the covered span, phased onto the stride lattice when possible."""
  stride = schema.keyframe_stride(fps)
  session_rows = np.round(times * fps).astype(np.int64)
  on_lattice = np.flatnonzero(session_rows[covered] % stride == 0)
  phase = int(on_lattice[0]) if len(on_lattice) else 0
  return covered[phase::keyframe_every]


def finger_stride(keyframe_every: int, fps: float) -> int:
  """The finger stride: the largest divisor of ``keyframe_every`` up to ``fps / 30``."""
  return max(
    (
      d
      for d in range(1, keyframe_every + 1)
      if keyframe_every % d == 0 and d <= fps / 30.0
    ),
    default=1,
  )


def discover_tracker_tags(args: TrackerArgs) -> GripperTags:
  """Identify the tracker's gripper tags from a sparse rear-lens read before prep."""
  session = SessionRecording(args.base_rrd)
  rear = session.lens_video(args.tracker, "rear")
  walk = LensFrames(rear, uniform_sample_rows(len(rear.times)), band=1)
  detections = _per_band(detect_all_tags, walk)
  _, rear_lens = session.lens_setups(args.tracker, int(walk[0].shape[1]))
  try:
    return discover_gripper_tags(
      detections,
      rear_lens.params,
      args.joints,
      tag_size=args.gripper.tag_size,
      anchor_tag_ids=set(args.anchor.enabled_tag_ids),
    )
  except TagDiscoveryError as exc:
    raise TagDiscoveryError(f"{args.tracker}: {exc}") from exc


def prepare_tracker(args: TrackerArgs, tags: GripperTags) -> PreparedTracker:
  """Every per-tracker stage; body poses land in the tracker's gravity-aligned world."""
  name = args.tracker
  configure_detector(args.frontend.parallel_tag_detect and args.parallel_siblings)
  session = SessionRecording(args.base_rrd)
  front_video = session.lens_video(args.tracker, "front")
  rear_video = session.lens_video(args.tracker, "rear")
  fps = fps_from_times(front_video.times)

  keyframe_every = resolve_keyframe_every(args.keyframe_every, fps)
  raw_imu = session.imu(args.tracker)
  covered = covered_rows(front_video.times, rear_video.times, raw_imu.t, fps)
  front_rows = keyframe_grid(front_video.times, covered, fps, keyframe_every)
  grid_times = front_video.times[front_rows]
  rear_kf_rows = nearest_rows(rear_video.times, grid_times)

  anchor_tag_ids = set(args.anchor.enabled_tag_ids)

  def tag_masks(
    shape: tuple[int, int], tags_per_kf: list[list[TagDetection]]
  ) -> list[list[Rect]]:
    """Per-keyframe rectangles over every non-anchor tag, empty with masking off."""
    if not args.frontend.mask_tags:
      return [[] for _ in tags_per_kf]
    return non_anchor_tag_masks(
      shape, tags_per_kf, anchor_tag_ids, args.frontend.mask_tag_pad
    )

  def walk_front() -> tuple[int, list[list[TagDetection]], list[Frame]]:
    """One banded walk for tags and loop-closure ORB, the ORB empty with closure off."""
    walk = LensFrames(front_video, front_rows, band=_PREP_BAND)
    width = int(walk[0].shape[1])
    if not args.frontend.loop_closure:
      return width, _per_band(detect_all_tags, walk), []

    def band_fn(
      band: list[np.ndarray],
    ) -> list[tuple[list[TagDetection], Frame]]:
      tags = detect_all_tags(band)
      rects = tag_masks(band[0].shape[:2], tags)
      orb = parallel_map(
        lambda f, m: detect_orb(
          f, args.frontend.orb_n_features, 20, mask_plane(f.shape[:2], m)
        ),
        band,
        rects,
      )
      return list(zip(tags, orb, strict=True))

    fused = _per_band(band_fn, walk)
    return width, [tags for tags, _ in fused], [orb for _, orb in fused]

  decoded_width, front_tags_per_kf, front_orb = walk_front()
  front_lens, rear_lens = session.lens_setups(args.tracker, decoded_width)
  release_freed_memory()
  # Rear: one dense walk, of which the keyframe set is a stride.
  stride = finger_stride(keyframe_every, fps)
  dense_rows = nearest_rows(rear_video.times, front_video.times[covered][::stride])
  rear_tags_per_frame = _per_band(
    detect_all_tags, LensFrames(rear_video, dense_rows, band=_PREP_BAND)
  )
  rear_frame_times = rear_video.times[dense_rows]
  rear_tags_per_kf = rear_tags_per_frame[:: keyframe_every // stride]
  n_keyframes = min(len(front_tags_per_kf), len(rear_tags_per_kf))
  release_freed_memory()
  if n_keyframes < 3:
    raise RuntimeError(f"only {n_keyframes} keyframes decoded — window too short")
  front_tags_per_kf = front_tags_per_kf[:n_keyframes]
  rear_tags_per_kf = rear_tags_per_kf[:n_keyframes]
  keyframe_times = grid_times[:n_keyframes]

  imu_to_body = session.imu_to_body(args.tracker)
  # Raw-axis samples rotated into the body frame; gyro is already rad/s.
  imu = ImuSamples(
    t=raw_imu.t,
    gyro=raw_imu.gyro @ imu_to_body.T,
    accel=raw_imu.accel @ imu_to_body.T,
  )
  body_deltas = keyframe_delta_rotations(imu, keyframe_times)
  del raw_imu

  def lens_kf_rotations(lens: LensSetup) -> list[Float64Array]:
    r_bl = lens.extrinsic.rotation().matrix()
    return [r_bl.T @ d.T @ r_bl for d in body_deltas]

  front = _LensPrep(side="front", setup=front_lens, tags=front_tags_per_kf)
  rear = _LensPrep(
    side="rear",
    setup=rear_lens,
    tags=rear_tags_per_kf,
    # The rear frontend applies the rects, so a run without rear landmarks records none.
    static_mask_rects=(
      list(args.frontend.mask_rear_rects) if args.frontend.rear_landmarks else []
    ),
  )
  rear_kf_rotations = lens_kf_rotations(rear_lens)
  lens_sides = (
    _LensSide(front, lens_kf_rotations(front_lens)),
    _LensSide(rear, rear_kf_rotations),
  )
  obs_by_tag = {
    tag_id: _detect_anchor_obs(tag_id, args.anchor.tag_size, lens_sides)
    for tag_id in args.anchor.enabled_tag_ids
  }
  prep_anchor_tag_id = _choose_prep_anchor(args.anchor.enabled_tag_ids, obs_by_tag)
  anchor_obs_per_kf = obs_by_tag[prep_anchor_tag_id]
  reference_tag_id = args.anchor.enabled_tag_ids[0]
  if prep_anchor_tag_id != reference_tag_id:
    n_ref = len(obs_by_tag[reference_tag_id])
    print(
      f"  [{name}/anchor] prep reference: tag {prep_anchor_tag_id} "
      f"({len(anchor_obs_per_kf)} sighted keyframes vs {n_ref} for "
      f"tag {reference_tag_id})"
    )

  def run_lens(ch: _LensPrep, video: LensVideo, rows: npt.NDArray[np.int64]) -> None:
    """Decode this lens, take what downstream needs from its pixels, then drop them."""
    frames = LensFrames(video, rows, band=_PREP_BAND)
    shape = frames[0].shape[:2]
    masks = tag_masks(shape, ch.tags)
    apply_static_mask_rects(masks, ch.static_mask_rects, shape)
    ch.keypoints, tracks = run_lens_frontend(
      frames,
      ch.setup.params,
      args.frontend,
      masks=masks,
      label=f"{name}/{ch.side}",
      band=frames.band,
    )
    ch.kept = {
      t: o for t, o in tracks.items() if len(o) >= args.frontend.min_track_length
    }
    if args.frontend.min_parallax > 0.0:
      ch.kept, ch.rejected = filter_tracks_by_parallax(
        ch.kept, ch.setup.params, args.frontend.min_parallax
      )
    release_freed_memory()

  run_lens(front, front_video, front_rows)
  if args.frontend.rear_landmarks:
    run_lens(rear, rear_video, rear_kf_rows)

  body_poses_prealign, anchor_seed_used = _seed_body_poses(
    name, front, rear, n_keyframes, anchor_obs_per_kf, args.frontend
  )

  imu, bias0, gravity_world_initial = _imu_and_initial_gravity(
    imu,
    session.factory_bias(args.tracker),
    imu_to_body,
    keyframe_times,
    body_poses_prealign,
  )

  for ch in (front, rear):
    ch.landmarks_prealign = triangulate_lens(
      ch.kept,
      body_poses_prealign,
      ch.setup.extrinsic,
      ch.setup.params,
      args.frontend.min_track_length,
      min_landmark_depth=0.01,
      max_landmark_depth=1000.0,
    )

  # Scale: 1.0 for an anchor seed, else from a visible anchor, else floated on the IMU.
  anchor_scale = _anchor_metric_scale(
    body_poses_prealign, anchor_obs_per_kf, front_lens, rear_lens
  )
  body_poses, velocities, pim, prealign_to_world = _align_visual_inertial(
    name,
    imu,
    keyframe_times,
    bias0,
    body_poses_prealign,
    gravity_world_initial,
    fixed_scale=1.0 if anchor_seed_used else anchor_scale,
  )

  # Each sighted anchor's pose in this tracker's local world, for the shared-world BFS.
  anchor_locals_by_tag: dict[int, gtsam.Pose3] = {
    tag_id: _first_sighting_local(obs, front_lens, rear_lens, body_poses)
    for tag_id, obs in obs_by_tag.items()
    if obs
  }

  loop_tracks: dict[int, dict[int, Float64Array]] = {}
  if anchor_obs_per_kf and args.frontend.loop_closure:
    loop_tracks = loop_closure_tracks(
      front_orb[:n_keyframes],
      body_poses,
      front_lens,
      max(front.kept, default=0),
      args.frontend,
    )
  # Nothing past loop closure reads the ORB frames, the descriptors' last holder.
  front_orb.clear()
  release_freed_memory()

  if anchor_obs_per_kf:
    for ch in (front, rear):
      ch.points = triangulate_lens(
        {t: ch.kept[t] for t in ch.kept if t in ch.landmarks_prealign},
        body_poses,
        ch.setup.extrinsic,
        ch.setup.params,
        args.frontend.min_track_length,
        max_landmark_depth=args.frontend.max_landmark_depth,
      )
      ch.points = cap_landmarks(
        ch.points, ch.kept, args.frontend.max_landmarks_per_lens
      )
    if loop_tracks:
      # Merge after the cap so the per-lens budget never culls the revisit landmarks.
      add_loop_landmarks(
        front.points,
        front.kept,
        loop_tracks,
        body_poses,
        front_lens,
        args.frontend,
        name=name,
        anchor_kfs=set(anchor_obs_per_kf)
        if args.frontend.anchored_loop_closure
        else None,
      )
  else:
    for ch in (front, rear):
      ch.points = {
        tid: prealign_to_world @ xyz for tid, xyz in ch.landmarks_prealign.items()
      }

  for ch in (front, rear):
    ch.tracks = {
      t: subsample_obs(ch.kept[t], args.frontend.max_obs_per_track)
      for t in ch.points
      if t in ch.kept
    }

  v_max = float(max(np.linalg.norm(v) for v in velocities))
  print(f"  [{name}/vi] LSQ |v|max={v_max:.2f} m/s")
  biases = [bias0 for _ in range(n_keyframes)]

  body_T_base_tag, joint_angles, finger_reproj, finger_frame_angles = (
    recover_gripper_tags(
      tags,
      args.joints,
      rear_tags_per_kf,
      rear_lens,
      args.gripper.tag_size,
      n_keyframes,
      rear_kf_rotations,
      rear_tags_per_frame,
      rear_frame_times,
    )
  )

  return PreparedTracker(
    name=name,
    body_poses=body_poses,
    velocities=velocities,
    biases=biases,
    preintegrated=pim,
    front=front.channel(n_keyframes),
    rear=rear.channel(n_keyframes),
    anchor_obs_by_tag=obs_by_tag,
    anchor_locals_by_tag=anchor_locals_by_tag,
    body_T_base_tag=body_T_base_tag,
    joint_angles=joint_angles,
    keyframe_times=keyframe_times,
    keyframe_rows=np.round(keyframe_times * fps).astype(np.int64),
    finger_reproj=finger_reproj,
    tags=tags,
    imu=imu,
    finger_frame_angles=finger_frame_angles,
  )
