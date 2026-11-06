"""One tracker's lenses and IMU prepared for the joint graph."""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import gtsam
import numpy as np
import numpy.typing as npt
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
  configure_detector,
  detect_all_tags,
  detect_tags_following_split,
)
from specialist.vi_ba.build import LensChannel, LensSetup
from specialist.vi_ba.configs import (
  TrackerArgs,
)
from specialist.vi_ba.features import (
  Frame,
  available_ram,
  detect_orb,
)
from specialist.vi_ba.finger_kinematics import FingerReproj, recover_gripper_tags
from specialist.vi_ba.geometry import average_pose, consensus_pose, translation
from specialist.vi_ba.imu import (
  GRAVITY,
  GRAVITY_WORLD,
  ImuSamples,
  find_stationary_windows,
  make_preintegration_params,
  preintegrate_between_keyframes,
)
from specialist.vi_ba.init import (
  Alignment,
  apply_alignment_to_poses,
  body_poses_from_camera_poses,
  fill_camera_poses,
  visual_inertial_align,
)
from specialist.vi_ba.tag_discovery import (
  GripperTags,
  TagDiscoveryError,
  discover_gripper_tags,
  discovery_rows,
)
from specialist.vi_ba.tracks import (
  add_loop_landmarks,
  cap_landmarks,
  filter_tracks_by_parallax,
  loop_closure_tracks,
  run_lens_frontend,
  split_tracks_to_window,
  subsample_obs,
  triangulate_lens,
)

Float64Array = npt.NDArray[np.float64]

_A = TypeVar("_A")

# Keyframes decoded at once from the front and the rear lens.
_PREP_BAND = 128
_REAR_BAND = 64

# Share of the available memory the concurrent decodes may claim between them.
_PREP_MEMORY_SHARE = 0.35

# Frames that may wait on the per-frame job workers, and the workers.
_FRAME_JOBS_PENDING = 32
_FRAME_JOB_WORKERS = 8
# Runs the rear lens walk splits into, each with its own decoder.
_REAR_WALK_SEGMENTS = 2


__all__ = [
  "PreparedTracker",
  "captures_that_fit",
  "discover_tracker_tags",
  "prepare_tracker",
  "renumber_landmarks",
  "bound_malloc_arenas",
  "stage_timer",
]


def _resident_gb() -> float | None:
  """The process's anonymous resident memory in GB, where the platform reports it."""
  try:
    for line in Path("/proc/self/status").read_text().splitlines():
      if line.startswith("RssAnon:"):
        return int(line.split()[1]) / 2**20
  except OSError, ValueError, IndexError:
    pass
  return None


def stage_timer(scope: str | None = None) -> Callable[[str], None]:
  """A ``[scope/timing] <stage> +<seconds> rss=<GB>`` line per finished stage."""
  tag = f"{scope}/timing" if scope else "timing"
  last = time.perf_counter()

  def mark(stage: str) -> None:
    nonlocal last
    now = time.perf_counter()
    resident = _resident_gb()
    memory = "" if resident is None else f" rss={resident:.1f}GB"
    print(f"  [{tag}] {stage} +{now - last:.1f}s{memory}")
    last = now

  return mark


class _FrameJobs:
  """Tag detection and loop-closure ORB per frame, on workers beside the walk."""

  def __init__(self, n_frames: int, orb_features: int) -> None:
    self._orb_features = orb_features
    self.tags: list[list[TagDetection]] = [[] for _ in range(n_frames)]
    self._orb: list[Frame | None] = [None] * n_frames
    self._pool = ThreadPoolExecutor(max_workers=_FRAME_JOB_WORKERS)
    self._slots = threading.BoundedSemaphore(_FRAME_JOBS_PENDING)
    self._futures: list[Future[None]] = []

  @property
  def orb(self) -> list[Frame]:
    """Every frame's ORB keypoints and descriptors; empty with no budget."""
    return [] if self._orb_features <= 0 else [f for f in self._orb if f is not None]

  def submit(self, k: int, frame: np.ndarray) -> None:
    # The previous band's freed frames go back to the OS as a band starts.
    if k % _PREP_BAND == 0:
      release_freed_memory()
    self._slots.acquire()
    self._futures.append(self._pool.submit(self._run, k, frame))

  def _run(self, k: int, frame: np.ndarray) -> None:
    try:
      self.tags[k] = detect_all_tags([frame])[0]
      if self._orb_features > 0:
        self._orb[k] = detect_orb(frame, self._orb_features, 20)
    finally:
      self._slots.release()

  def __enter__(self) -> _FrameJobs:
    return self

  def __exit__(self, *exc: object) -> None:
    self._pool.shutdown(wait=True)
    for future in self._futures:
      future.result()


def renumber_landmarks(
  points: dict[int, Float64Array], tracks: dict[int, dict[int, Float64Array]]
) -> tuple[dict[int, Float64Array], dict[int, dict[int, Float64Array]]]:
  """``points`` and ``tracks`` re-keyed ``0..n-1`` in ascending order of the
  point ids; a track without a point is dropped."""
  dense = {old: new for new, old in enumerate(sorted(points))}
  return (
    {dense[t]: p for t, p in points.items()},
    {dense[t]: obs for t, obs in tracks.items() if t in dense},
  )


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


def bound_malloc_arenas(limit: int) -> bool:
  """Cap glibc's malloc arenas at ``limit``; whether the allocator took it."""
  try:
    libc = ctypes.CDLL("libc.so.6")
    m_arena_max = -8
    return bool(libc.mallopt(m_arena_max, limit))
  except OSError, AttributeError:
    return False


def release_freed_memory() -> None:
  """Hand freed heap back to the OS; a no-op without glibc's ``malloc_trim``."""
  try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
  except OSError, AttributeError:
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
  # Front-lens anchor PnP, ``{tag_id: {kf: cam_T_tag}}``.
  anchor_obs_by_tag: dict[int, dict[int, gtsam.Pose3]]
  # Each sighted anchor's pose in this tracker's local world.
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
    )


def _choose_prep_anchor(
  tag_ids: tuple[int, ...],
  obs_by_tag: dict[int, dict[int, gtsam.Pose3]],
) -> int:
  """The id sighted at most keyframes; a tie goes to the earlier ``tag_ids`` entry."""
  if not tag_ids:
    raise ValueError("no anchor tag id is enabled; prep has nothing to ground on")
  return max(tag_ids, key=lambda tid: len(obs_by_tag[tid]))


def _anchor_pose_seed(
  anchor_obs_per_kf: dict[int, gtsam.Pose3],
  front_extrinsic: gtsam.Pose3,
  n_keyframes: int,
) -> list[gtsam.Pose3] | None:
  """Body poses at every keyframe from the anchor sightings, gaps geodesic-filled,
  or ``None`` when too few keyframes see the anchor."""
  cam_poses = {kf: pose.inverse() for kf, pose in anchor_obs_per_kf.items()}
  if not _enough_registered(len(cam_poses), n_keyframes):
    return None
  return body_poses_from_camera_poses(
    fill_camera_poses(cam_poses, n_keyframes), front_extrinsic
  )


def _imu_and_initial_gravity(
  imu: ImuSamples,
  raw_factory_bias: gtsam.imuBias.ConstantBias,
  imu_to_body: Float64Array,
  keyframe_times: Float64Array,
  body_poses_prealign: list[gtsam.Pose3],
) -> tuple[ImuSamples, gtsam.imuBias.ConstantBias, Float64Array | None]:
  """IMU samples with accel rescaled to |g| on the first stationary window, the
  body-frame factory bias, and that window's world gravity or ``None``."""
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
  n_keyframes: int,
  anchor_obs_per_kf: dict[int, gtsam.Pose3],
) -> list[gtsam.Pose3]:
  """The pre-alignment body poses, seeded from the anchor sightings."""
  seed = _anchor_pose_seed(anchor_obs_per_kf, front.setup.extrinsic, n_keyframes)
  if seed is None:
    raise RuntimeError(
      f"{name}: only {len(anchor_obs_per_kf)}/{n_keyframes} keyframes see the "
      "anchor; the trajectory seed needs a quarter of them"
    )
  print(
    f"  [{name}/vo] anchor-pose init from {len(anchor_obs_per_kf)}/{n_keyframes} "
    f"anchor keyframes"
  )
  return seed


def _align_visual_inertial(
  name: str,
  imu: ImuSamples,
  keyframe_times: Float64Array,
  bias0: gtsam.imuBias.ConstantBias,
  body_poses_prealign: list[gtsam.Pose3],
  gravity_world_initial: Float64Array | None,
) -> tuple[
  list[gtsam.Pose3],
  list[Float64Array],
  list[gtsam.PreintegratedCombinedMeasurements],
  Float64Array,
]:
  """Gravity-aligned metric body poses, velocities, keyframe preintegration and the
  3x3 map carrying prep-frame points into the same world."""

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
      poses, pim, fixed_gravity=gravity_world_initial, fixed_scale=1.0
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
  bands = 2 * _PREP_BAND + 2 * _REAR_WALK_SEGMENTS * _REAR_BAND
  per_capture = bands * width * height
  budget = available_ram() * _PREP_MEMORY_SHARE
  affordable = int(budget / per_capture)
  return max(1, min(len(names), affordable))


def keyframe_grid(
  times: Float64Array,
  covered: npt.NDArray[np.int64],
  fps: float,
  keyframe_every: int,
) -> tuple[npt.NDArray[np.int64], int]:
  """Keyframe rows over the covered span, snapped to the session-clock stride
  lattice, and the phase they start on."""
  stride = schema.keyframe_stride(fps)
  session_rows = np.round(times * fps).astype(np.int64)
  on_lattice = np.flatnonzero(session_rows[covered] % stride == 0)
  phase = int(on_lattice[0]) if len(on_lattice) else 0
  return covered[phase::keyframe_every], phase


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
  session = SessionRecording(args.base_rrd, args.stage)
  rear = session.lens_video(args.tracker, "rear")
  walk = LensFrames(rear, discovery_rows(rear.iframe_rows), band=1)
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
  lap = stage_timer(name)
  configure_detector(args.frontend.parallel_tag_detect)
  session = SessionRecording(args.base_rrd, args.stage)
  front_video = session.lens_video(args.tracker, "front")
  rear_video = session.lens_video(args.tracker, "rear")
  fps = fps_from_times(front_video.times)

  keyframe_every = resolve_keyframe_every(args.keyframe_every, fps)
  raw_imu = session.imu(args.tracker)
  covered = covered_rows(front_video.times, rear_video.times, raw_imu.t, fps)
  front_rows, grid_phase = keyframe_grid(
    front_video.times, covered, fps, keyframe_every
  )
  grid_times = front_video.times[front_rows]

  # Rear: a dense walk beside the front walk; the keyframes are a stride of it.
  stride = finger_stride(keyframe_every, fps)
  dense_rows = nearest_rows(
    rear_video.times, front_video.times[covered][grid_phase::stride]
  )
  kf_stride = keyframe_every // stride
  rear_frame_times = rear_video.times[dense_rows]
  rear_walk = ThreadPoolExecutor(max_workers=1)

  def rear_frames(begin: int, end: int, video: LensVideo = rear_video) -> LensFrames:
    return LensFrames(video, dense_rows[begin:end], band=_REAR_BAND)

  rear_tags_future = rear_walk.submit(
    detect_tags_following_split,
    rear_frames,
    len(dense_rows),
    kf_stride,
    {tag_id for tag_id in tags.fingers if tag_id > 0},
    _REAR_WALK_SEGMENTS,
  )

  # Front: one walk serves the flow tracks, the tag detections and the ORB.
  lap("inputs loaded")
  front_frames = LensFrames(front_video, front_rows, band=_PREP_BAND)
  decoded_width = int(front_frames[0].shape[1])
  front_lens, rear_lens = session.lens_setups(args.tracker, decoded_width)
  front = _LensPrep(side="front", setup=front_lens, tags=[])
  jobs = _FrameJobs(
    len(front_frames),
    args.frontend.orb_n_features if args.frontend.loop_closure else 0,
  )
  try:
    with jobs:
      front.keypoints, tracks = run_lens_frontend(
        front_frames,
        front_lens.params,
        args.frontend,
        label=f"{name}/front",
        on_frame=jobs.submit,
        band=front_frames.band,
      )
  finally:
    # A front walk that fails must not leave the rear walk decoding.
    rear_walk.shutdown(wait=True, cancel_futures=True)
  front.tags, front_orb = jobs.tags, jobs.orb
  del front_frames, jobs
  lap("front lens walk")
  front.kept = {
    t: o for t, o in tracks.items() if len(o) >= args.frontend.min_track_length
  }
  if args.frontend.min_parallax > 0.0:
    front.kept, front.rejected = filter_tracks_by_parallax(
      front.kept, front.setup.params, args.frontend.min_parallax
    )
  del tracks
  lap("front tracks")

  rear_tags_per_frame = rear_tags_future.result()
  # The encoded samples are read; nothing past the walks touches them.
  del front_video, rear_video
  rear_tags_per_kf = rear_tags_per_frame[::kf_stride]
  n_keyframes = min(len(front.tags), len(rear_tags_per_kf))
  release_freed_memory()
  lap("rear lens walk, past the front")
  if n_keyframes < 3:
    raise RuntimeError(f"only {n_keyframes} keyframes decoded — window too short")
  front.tags = front.tags[:n_keyframes]
  rear = _LensPrep(side="rear", setup=rear_lens, tags=rear_tags_per_kf[:n_keyframes])
  keyframe_times = grid_times[:n_keyframes]

  imu_to_body = session.imu_to_body(args.tracker)
  # Raw-axis samples rotated into the body frame; gyro is already rad/s.
  imu = ImuSamples(
    t=raw_imu.t,
    gyro=raw_imu.gyro @ imu_to_body.T,
    accel=raw_imu.accel @ imu_to_body.T,
  )
  del raw_imu

  # Front lens only: the rear seat is a solved variable, so prep never reads it.
  obs_by_tag = {
    tag_id: anchor_pose_from_tags(
      front.tags, tag_id, args.anchor.tag_size, front.setup.params
    )
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
  lap("anchor pnp")

  body_poses_prealign = _seed_body_poses(name, front, n_keyframes, anchor_obs_per_kf)
  lap("trajectory seed")

  imu, bias0, gravity_world_initial = _imu_and_initial_gravity(
    imu,
    session.factory_bias(args.tracker),
    imu_to_body,
    keyframe_times,
    body_poses_prealign,
  )

  front.landmarks_prealign = triangulate_lens(
    front.kept,
    body_poses_prealign,
    front.setup.extrinsic,
    front.setup.params,
    args.frontend.min_track_length,
    min_landmark_depth=0.01,
    max_landmark_depth=1000.0,
  )
  lap("prealign triangulation")

  body_poses, velocities, pim, prealign_to_world = _align_visual_inertial(
    name,
    imu,
    keyframe_times,
    bias0,
    body_poses_prealign,
    gravity_world_initial,
  )
  lap("vi alignment")

  # Each sighted anchor's pose in this tracker's local world.
  anchor_locals_by_tag: dict[int, gtsam.Pose3] = {}
  for tag_id, obs in obs_by_tag.items():
    if not obs:
      continue
    candidates = [
      body_poses[kf].compose(front_lens.extrinsic).compose(cam_T_tag)
      for kf, cam_T_tag in obs.items()
    ]
    agreed = consensus_pose(candidates)
    if agreed is None:
      print(
        f"  [warn] [{name}/anchor] tag {tag_id} sightings split; seeding on the median"
      )
      agreed = average_pose(candidates)
    anchor_locals_by_tag[tag_id] = agreed

  loop_tracks: dict[int, dict[int, Float64Array]] = {}
  if anchor_obs_per_kf and args.frontend.loop_closure:
    loop_tracks = loop_closure_tracks(
      front_orb[:n_keyframes],
      body_poses,
      front_lens,
      max(front.kept, default=0),
      window=args.frontend.loop_closure_window,
    )
  # Nothing past loop closure reads the ORB frames, the descriptors' last holder.
  front_orb.clear()
  release_freed_memory()
  lap("loop closure")

  if anchor_obs_per_kf:
    front.points = triangulate_lens(
      {t: front.kept[t] for t in front.kept if t in front.landmarks_prealign},
      body_poses,
      front.setup.extrinsic,
      front.setup.params,
      args.frontend.min_track_length,
      max_landmark_depth=args.frontend.max_landmark_depth,
    )
    front.points = cap_landmarks(front.points, front.kept, args.frontend.max_landmarks)
    if loop_tracks:
      # After the cap, so the revisit landmarks survive the landmark budget.
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
    front.points = {
      tid: prealign_to_world @ xyz for tid, xyz in front.landmarks_prealign.items()
    }

  front.points, windowed = split_tracks_to_window(
    front.points,
    {t: front.kept[t] for t in front.points if t in front.kept},
    args.frontend.loop_closure_window,
  )
  front.tracks = {
    t: subsample_obs(obs, args.frontend.max_obs_per_track)
    for t, obs in windowed.items()
  }
  # The landmarks that reach the bundle take dense ids.
  front.points, front.tracks = renumber_landmarks(front.points, front.tracks)

  lap("landmarks")

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
      rear_tags_per_frame,
      rear_frame_times,
    )
  )
  lap("gripper tags")

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
