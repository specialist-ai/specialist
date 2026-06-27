"""VI-BA driver for any number of trackers (N ≥ 1).

Audio-syncs N ``.insv`` captures via clap detection (no-op for N=1),
runs the per-tracker prep (decode → frontend → anchor PnP → VI
alignment), aligns every tracker into the first one's anchor world,
allocates collision-free GTSAM keys via ``TrackerKeys``, builds
one VI subgraph per tracker, then adds:

- Anchor obs: every tracker sees the wall AprilTag and pulls its body
  pose toward the shared anchor variable.
- Cross-tracker obs (N ≥ 2): when tracker O sees tracker W's base tag, that
  ties O's pose at that keyframe to W's via the PnP + body→tag offset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gtsam
import numpy as np
import numpy.typing as npt
import tyro

from specialist.graph.factors import make_cross_tracker_factor
from specialist.loader.audio_sync import sync_two_claps
from specialist.loader.mei_model import MeiParams, mei_project
from specialist.synthesis.tag_detection import TagObservation
from specialist.vi_ba.anchor_detection import (
  TagDetection,
  estimate_camera_to_tag_pose,
  tag_corners_local,
)
from specialist.vi_ba.build import (
  LensSetup,
  add_mei_factor,
  build_vi_graph,
  huber_pixel_noise,
)
from specialist.vi_ba.compose import TrackerKeys
from specialist.vi_ba.configs import (
  AnchorCfg,
  CrossTrackerCfg,
  FrontendCfg,
  GripperCfg,
  ImuCfg,
  SolverCfg,
  WindowCfg,
)
from specialist.vi_ba.optimize import optimize_graph
from specialist.vi_ba.pipeline import capture_duration
from specialist.vi_ba.single_tracker import (
  PreparedTracker,
  TrackerArgs,
  prepare_tracker,
)
from specialist.vi_ba.viz import TrackerVizData, log_multi_ba_run

Float64Array = npt.NDArray[np.float64]


@dataclass
class Args:
  input_insvs: tuple[Path, ...]
  window: WindowCfg
  anchor: AnchorCfg
  output_dir: Path = Path("logs")
  tracker_names: tuple[str, ...] = ()  # auto-derive ("tracker_0", "tracker_1", ...)
  urdf_path: Path | None = None
  log_frames_in_rrd: bool = False
  open_viewer: bool = True
  # Cache the expensive per-tracker prep (decode + frontend + triangulation +
  # anchor PnP) keyed by the prep-affecting args, so re-runs that only change
  # BA-level params (σ, anchor noise, lm iters) skip the ~20-min prep. Not for
  # runs that need full video frames (--log-frames-in-rrd).
  prep_cache: bool = False
  # One id per tracker; cross-tracker disambiguation needs distinct ids.
  base_tag_ids: tuple[int, ...] = ()
  # Per-tracker finger tag ids (or 0 to disable for that tracker). Used to
  # recover the URDF revolute-joint angles each keyframe.
  left_finger_tag_ids: tuple[int, ...] = ()
  right_finger_tag_ids: tuple[int, ...] = ()
  # Lighter frontend feature budget so the joint graph stays cheap; override per run.
  frontend: FrontendCfg = field(
    default_factory=lambda: FrontendCfg(orb_n_features=500, init_min_shared_tracks=30)
  )
  imu: ImuCfg = field(default_factory=ImuCfg)
  gripper: GripperCfg = field(default_factory=GripperCfg)
  cross: CrossTrackerCfg = field(default_factory=CrossTrackerCfg)
  solver: SolverCfg = field(default_factory=SolverCfg)


@dataclass
class Summary:
  sources: tuple[str, ...]
  tracker_names: tuple[str, ...]
  audio_offsets: tuple[float, ...]
  per_tracker_keyframes: tuple[int, ...]
  anchor_obs_per_tracker: tuple[int, ...]
  cross_tracker_obs: int
  initial_graph_error: float
  final_graph_error: float
  solver_iterations: int
  bounding_box_diagonal: tuple[float, ...]
  trajectory_path_length: tuple[float, ...]


def _derive_names(
  input_insvs: tuple[Path, ...], names: tuple[str, ...]
) -> tuple[str, ...]:
  if names:
    if len(names) != len(input_insvs):
      raise ValueError(
        f"tracker_names ({len(names)}) must match input_insvs ({len(input_insvs)})"
      )
    return names
  return tuple(f"tracker_{i}" for i in range(len(input_insvs)))


def _audio_align_starts(
  input_insvs: tuple[Path, ...], master_start: float
) -> tuple[list[float], list[float]]:
  """Per-clip ``start`` (in its own local clock) so every clip aligns on a
  shared timeline. The clip whose clap leads begins at ``master_start`` and
  clips that started later are trimmed to it.

  Shifting by the earliest offset keeps every start ``>= master_start`` so none
  goes negative; a negative start is clamped to 0 by the decoder, which would
  misalign that clip's keyframes from the rest.

  Returns ``(per_clip_starts, offsets_relative_to_clip0)``.
  """
  offsets = [0.0]
  for path in input_insvs[1:]:
    sync = sync_two_claps(input_insvs[0], path)
    offsets.append(sync.offset)
    print(
      f"audio sync: {input_insvs[0].name} ↔ {path.name}  "
      f"offset={sync.offset:+.3f}s  "
      f"matched_pairs={sync.matched_pairs} "
      f"(claps_a={sync.clap_times_a.size}, claps_b={sync.clap_times_b.size})"
    )
  lead = min(offsets)
  starts = [master_start + off - lead for off in offsets]
  return starts, offsets


def _align_to_shared_world(trackers: list[PreparedTracker]) -> gtsam.Pose3:
  """Transform every tracker's body poses, velocities, and landmarks so all
  trackers' anchor obs reproduce the first anchor-observing tracker's
  anchor world. Returns that shared anchor pose."""
  ref = next(
    (t.anchor_local for t in trackers if t.anchor_local is not None),
    None,
  )
  if ref is None:
    raise RuntimeError("no tracker observes the anchor; cannot share a world")
  for t in trackers:
    if t.anchor_local is None:
      continue
    align = ref.compose(t.anchor_local.inverse())
    align_R = align.rotation().matrix()
    align_t = np.asarray(align.translation(), dtype=np.float64)
    t.body_poses = [align.compose(p) for p in t.body_poses]
    t.velocities = [align_R @ v for v in t.velocities]
    t.front_points = {tid: align_R @ p + align_t for tid, p in t.front_points.items()}
    t.rear_points = {tid: align_R @ p + align_t for tid, p in t.rear_points.items()}
  return ref


def _cross_tracker_observations(
  trackers: list[PreparedTracker],
  base_tag_ids: list[int],
  base_tag_size: float,
) -> list[TagObservation]:
  """Per ordered pair (observer, owner), PnP each detection of owner's
  base tag in observer's frames into a ``TagObservation``. Captures are
  audio-synced, so observer and owner share the same keyframe index."""
  obs: list[TagObservation] = []
  ids = {t.name: base_tag_ids[i] for i, t in enumerate(trackers)}
  for owner_idx, owner in enumerate(trackers):
    owner_tag_id = ids[owner.name]
    for obs_idx, observer in enumerate(trackers):
      if obs_idx == owner_idx:
        continue
      n = min(observer.n_keyframes, owner.n_keyframes)
      for side, lens, tags_per_kf in (
        ("front", observer.front_lens, observer.front_tags_per_kf),
        ("rear", observer.rear_lens, observer.rear_tags_per_kf),
      ):
        for kf in range(n):
          for det in tags_per_kf[kf]:
            if int(det.tag_id) != owner_tag_id:
              continue
            cam_T_tag = estimate_camera_to_tag_pose(
              det.corners, base_tag_size, lens.params
            )
            if cam_T_tag is None:
              continue
            obs.append(
              TagObservation(
                observer_tracker=observer.name,
                fisheye_side="front" if side == "front" else "back",
                keyframe_index=kf,
                tag_id=f"tracker_{owner.name}",
                tag_kind="tracker",
                tag_owner_tracker=owner.name,
                camera_to_tag=cam_T_tag,
              )
            )
            break  # one detection per (kf, side) — owner's tag is unique
  return obs


def _anchor_corners_world(ref_anchor: gtsam.Pose3, tag_size: float) -> Float64Array:
  """The anchor tag's four metric corners in the world frame, ``(4, 3)``."""
  return np.array(
    [
      np.asarray(ref_anchor.transformFrom(gtsam.Point3(*corner)), dtype=np.float64)
      for corner in tag_corners_local(tag_size)
    ]
  )


def _anchor_sightings(
  tracker: PreparedTracker, anchor_tag_id: int
) -> list[tuple[LensSetup, int, TagDetection]]:
  """Per ``(lens, keyframe)`` anchor-tag detection. The corner reprojection factors consume
  exactly these sightings."""
  out: list[tuple[LensSetup, int, TagDetection]] = []
  for lens, tags_per_kf in (
    (tracker.front_lens, tracker.front_tags_per_kf),
    (tracker.rear_lens, tracker.rear_tags_per_kf),
  ):
    sightings = [
      (kf, det)
      for kf, dets in enumerate(tags_per_kf)
      for det in dets
      if int(det.tag_id) == anchor_tag_id
    ]
    out.extend((lens, kf, det) for kf, det in sightings)
  return out


def _factor_observation_lines(
  observer: PreparedTracker,
  observer_poses: list[gtsam.Pose3],
  cross_obs_for_observer: list[TagObservation],
) -> tuple[list[Float64Array], list[Float64Array]]:
  """Per-keyframe ``(n, 2, 3)`` line endpoints for every factor that fires
  at the observer's keyframe. Endpoint is the observer's PnP-recovered tag
  centre composed through its optimized lens pose, so the line literally
  ends at where the observer detected the tag — not at the optimized tag
  pose on the other side of the factor (those agree post-LM, but the
  observer-side endpoint is the one that visually anchors the detection)."""
  anchor_lines: list[Float64Array] = []
  cross_lines: list[Float64Array] = []
  for kf in range(observer.n_keyframes):
    body_pose = observer_poses[kf]
    anchor_strips: list[Float64Array] = []
    for side, cam_T_tag in observer.anchor_obs_per_kf.get(kf, {}).items():
      lens = observer.front_lens if side == "front" else observer.rear_lens
      lens_world = body_pose.compose(lens.extrinsic)
      lens_xyz = np.asarray(lens_world.translation(), dtype=np.float64)
      tag_xyz = np.asarray(
        lens_world.compose(cam_T_tag).translation(), dtype=np.float64
      )
      anchor_strips.append(np.stack([lens_xyz, tag_xyz], axis=0))
    anchor_lines.append(
      np.stack(anchor_strips) if anchor_strips else np.empty((0, 2, 3))
    )
    cross_strips: list[Float64Array] = []
    for o in cross_obs_for_observer:
      if o.keyframe_index != kf:
        continue
      lens = observer.front_lens if o.fisheye_side == "front" else observer.rear_lens
      lens_world = body_pose.compose(lens.extrinsic)
      lens_xyz = np.asarray(lens_world.translation(), dtype=np.float64)
      tag_xyz = np.asarray(
        lens_world.compose(o.camera_to_tag).translation(), dtype=np.float64
      )
      cross_strips.append(np.stack([lens_xyz, tag_xyz], axis=0))
    cross_lines.append(
      np.stack(cross_strips, axis=0)
      if cross_strips
      else np.empty((0, 2, 3), dtype=np.float64)
    )
  return anchor_lines, cross_lines


def _per_kf_rejected_keypoints(
  rejected_tracks: dict[int, dict[int, Float64Array]], n_keyframes: int
) -> list[Float64Array]:
  buckets: list[list[Float64Array]] = [[] for _ in range(n_keyframes)]
  for obs in rejected_tracks.values():
    for kf, pix in obs.items():
      if 0 <= kf < n_keyframes:
        buckets[kf].append(np.asarray(pix, dtype=np.float64))
  return [np.stack(b) if b else np.empty((0, 2)) for b in buckets]


def _reproject_landmarks_per_kf(
  tracks: dict[int, dict[int, Float64Array]],
  points: dict[int, Float64Array],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  n_keyframes: int,
) -> list[Float64Array]:
  buckets: list[list[Float64Array]] = [[] for _ in range(n_keyframes)]
  for tid, obs in tracks.items():
    if tid not in points:
      continue
    lm_world = gtsam.Point3(*points[tid])
    for kf in obs:
      if not (0 <= kf < n_keyframes):
        continue
      world_T_lens = body_poses[kf].compose(lens.extrinsic)
      p_lens = np.asarray(world_T_lens.transformTo(lm_world), dtype=np.float64)
      if p_lens[2] <= 1e-3:
        continue
      pix = mei_project(p_lens[None, :], lens.params)[0]
      if np.all(np.isfinite(pix)):
        buckets[kf].append(pix)
  return [np.stack(b) if b else np.empty((0, 2)) for b in buckets]


def _project_tag_axes_per_lens(
  tags_per_kf: list,
  lens: LensSetup,
  tag_size_by_id: dict[int, float],
  n_keyframes: int,
) -> list[Float64Array]:
  out: list[Float64Array] = []
  for kf in range(n_keyframes):
    tags = tags_per_kf[kf] if kf < len(tags_per_kf) else []
    strips: list[np.ndarray] = []
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


def _per_keyframe_reproj_rmse_for_viz(
  values: gtsam.Values,
  tracks: dict[int, dict[int, Float64Array]],
  lens: LensSetup,
  landmark_key_fn: Callable[[int], int],
  pose_key_fn: Callable[[int], int],
  n_keyframes: int,
) -> list[float]:
  """Per-kf reprojection RMSE — visualizer-side helper. Downstream
  evaluation reads ``per_kf_landmark_count_by_tracker`` from the track JSON
  rather than this; this is purely for the rerun overlay and the
  per-tracker stdout sanity print."""
  buckets: list[list[float]] = [[] for _ in range(n_keyframes)]
  for tid, obs in tracks.items():
    lm_key = landmark_key_fn(tid)
    if not values.exists(lm_key):
      continue
    xyz = values.atPoint3(lm_key)
    for kf, pt in obs.items():
      pk = pose_key_fn(kf)
      if not (0 <= kf < n_keyframes) or not values.exists(pk):
        continue
      body_pose = values.atPose3(pk)
      p_lens = body_pose.compose(lens.extrinsic).transformTo(
        gtsam.Point3(float(xyz[0]), float(xyz[1]), float(xyz[2]))
      )
      pred = mei_project(np.asarray(p_lens)[None, :], lens.params)[0]
      if np.all(np.isfinite(pred)):
        buckets[kf].append(float(np.linalg.norm(pred - pt)))
  return [
    float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    for errs in buckets
  ]


def _aggregate_rmse_for_print(per_kf: list[float]) -> float:
  finite = [v for v in per_kf if np.isfinite(v)]
  return float(np.sqrt(np.mean(np.square(finite)))) if finite else float("nan")


def _pose_to_dict(p: gtsam.Pose3) -> dict:
  return {
    "R": [list(map(float, row)) for row in p.rotation().matrix()],
    "t": [float(x) for x in p.translation()],
  }


def _mei_to_dict(p: MeiParams) -> dict:
  return {
    "fx": float(p.fx),
    "fy": float(p.fy),
    "cx": float(p.cx),
    "cy": float(p.cy),
    "k1": float(p.k1),
    "k2": float(p.k2),
    "k3": float(p.k3),
    "p1": float(p.p1),
    "p2": float(p.p2),
    "xi": float(p.xi),
  }


def _prepare_cached(
  name: str, targs: "TrackerArgs", enabled: bool
) -> "PreparedTracker":
  """prepare_tracker with an on-disk cache keyed by the prep-affecting args +
  the input video's mtime. Heavy decoded frames are stripped before pickling
  (only the first frame is kept, for shape), so cached runs must not use
  --log-frames-in-rrd."""
  if not enabled:
    return prepare_tracker(name, targs)
  cache_dir = Path("/tmp/specialist_prep_cache")
  try:
    mt = targs.insv_path.stat().st_mtime
  except OSError:
    mt = 0.0
  key = hashlib.sha256(f"{targs!r}|{mt}".encode()).hexdigest()[:16]
  cpath = cache_dir / f"{name}_{key}.pkl"
  if cpath.exists():
    print(f"  [prep cache HIT] {cpath.name} — skipping decode+frontend")
    return pickle.loads(cpath.read_bytes())
  tracker = prepare_tracker(name, targs)
  cache_dir.mkdir(parents=True, exist_ok=True)
  slim = copy.copy(tracker)
  slim.front_frames = tracker.front_frames[:1]
  slim.rear_frames = tracker.rear_frames[:1]
  try:
    cpath.write_bytes(pickle.dumps(slim))
    print(f"  [prep cache saved] {cpath.name}")
  except Exception as exc:  # noqa: BLE001 - caching is best-effort
    print(f"  [prep cache save failed: {exc}]")
  return tracker


def _bounding_box_diagonal(poses: list[gtsam.Pose3]) -> float:
  if len(poses) < 2:
    return 0.0
  pts = np.array([np.asarray(p.translation(), dtype=np.float64) for p in poses])
  return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def _path_length(poses: list[gtsam.Pose3]) -> float:
  pts = [np.asarray(p.translation(), dtype=np.float64) for p in poses]
  return float(sum(np.linalg.norm(pts[i] - pts[i - 1]) for i in range(1, len(pts))))


def _stage_timer() -> Callable[[str], None]:
  last = time.perf_counter()

  def mark(stage: str) -> None:
    nonlocal last
    now = time.perf_counter()
    print(f"  [{stage}] +{now - last:.1f}s")
    last = now

  return mark


def _lens_side_dict(lens: LensSetup, decoded_width: int) -> dict[str, object]:
  """One lens's extrinsic, intrinsics, and decoded frame shape for the export."""
  return {
    "extrinsic": _pose_to_dict(lens.extrinsic),
    "params": _mei_to_dict(lens.params),
    "frame_shape": [decoded_width, decoded_width],
  }


def _tags_side_list(
  tags_per_kf: list[list[TagDetection]],
) -> list[list[dict[str, object]]]:
  """One lens's per-keyframe AprilTag detections (id + corners) for the export."""
  return [
    [
      {
        "tag_id": int(d.tag_id),
        "corners": [[float(p[0]), float(p[1])] for p in d.corners],
      }
      for d in tags
    ]
    for tags in tags_per_kf
  ]


def _tracks_side_dict(
  tracks: dict[int, dict[int, Float64Array]],
) -> dict[str, dict[str, list[float]]]:
  """One lens's ``{track_id: {keyframe: pixel}}`` observations for the export."""
  return {
    str(tid): {str(kf): [float(p[0]), float(p[1])] for kf, p in obs.items()}
    for tid, obs in tracks.items()
  }


def main(args: Args) -> None:
  if not args.input_insvs:
    raise ValueError("at least one .insv input required")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  names = _derive_names(args.input_insvs, args.tracker_names)
  if not args.base_tag_ids:
    raise ValueError(
      f"--base-tag-ids must be supplied (one per input, got {len(args.input_insvs)} inputs)"
    )
  for field_name, ids in (
    ("base_tag_ids", args.base_tag_ids),
    ("left_finger_tag_ids", args.left_finger_tag_ids),
    ("right_finger_tag_ids", args.right_finger_tag_ids),
  ):
    if ids and len(ids) != len(args.input_insvs):
      raise ValueError(
        f"{field_name} length ({len(ids)}) must match "
        f"input_insvs ({len(args.input_insvs)})"
      )
  mark = _stage_timer()

  starts, offsets = _audio_align_starts(args.input_insvs, args.window.master_start)
  mark(f"audio sync ({len(args.input_insvs)} clips)")

  # The recordings stop at different times; keyframes past the earliest stop
  # would track a gripper that is already off, so the window is capped at the
  # overlap every capture covers.
  overlap = min(
    capture_duration(path) - start
    for path, start in zip(args.input_insvs, starts, strict=True)
  )
  duration = (
    overlap if args.window.duration is None else min(args.window.duration, overlap)
  )
  print(f"window: {duration:.2f}s (capture overlap {overlap:.2f}s)")

  trackers: list[PreparedTracker] = []
  left_finger_ids = args.left_finger_tag_ids or (0,) * len(args.input_insvs)
  right_finger_ids = args.right_finger_tag_ids or (0,) * len(args.input_insvs)
  for name, path, start, base_tag_id, left_id, right_id in zip(
    names,
    args.input_insvs,
    starts,
    args.base_tag_ids,
    left_finger_ids,
    right_finger_ids,
    strict=True,
  ):
    print(f"\n=== prep tracker {name!r} ({path.name}, start={start:.3f}s) ===")
    tracker = _prepare_cached(
      name,
      TrackerArgs(
        insv_path=path,
        start=start,
        duration=duration,
        keyframe_every=args.window.keyframe_every,
        keep_frames=args.log_frames_in_rrd,
        base_tag_id=base_tag_id,
        left_finger_tag_id=int(left_id),
        right_finger_tag_id=int(right_id),
        frontend=args.frontend,
        imu=args.imu,
        anchor=args.anchor,
        gripper=args.gripper,
      ),
      args.prep_cache,
    )
    print(
      f"  kf={tracker.n_keyframes}  anchor_obs={len(tracker.anchor_obs_per_kf)}  "
      f"front_pts={len(tracker.front_points)}  rear_pts={len(tracker.rear_points)}  "
      f"body_T_base_tag={'yes' if tracker.body_T_base_tag else 'no'}"
    )
    trackers.append(tracker)
    mark(f"tracker {name!r} prepared")

  ref_anchor = _align_to_shared_world(trackers)
  print(f"\nshared world: anchor t={np.asarray(ref_anchor.translation())}")

  cross_obs = _cross_tracker_observations(
    trackers, list(args.base_tag_ids), args.gripper.tag_size
  )
  print(f"cross-tracker observations: {len(cross_obs)}")
  if cross_obs:
    by_pair: dict[tuple[str, str], int] = {}
    for o in cross_obs:
      key = (o.observer_tracker, o.tag_owner_tracker or "?")
      by_pair[key] = by_pair.get(key, 0) + 1
    for (obs_t, own_t), n in sorted(by_pair.items()):
      print(f"  {obs_t} → {own_t}: {n}")
  mark("cross-tracker obs assembled")

  keys = TrackerKeys(list(names))
  graph = gtsam.NonlinearFactorGraph()
  values = gtsam.Values()

  for t in trackers:
    build_vi_graph(
      **t.vi_graph_inputs(),
      sigma=args.frontend.reproj_sigma,
      velocity_prior_sigma=args.solver.velocity_prior_sigma,
      graph=graph,
      values=values,
      **keys.vi_graph_kwargs(t.name),
    )

  # Reproject the anchor tag's four metric corners through the MEI distortion
  # model, which stays valid even at the lens edge.
  anchor_tag_id = int(args.anchor.tag_id)
  anchor_corners_world = _anchor_corners_world(ref_anchor, args.anchor.tag_size)
  corner_keys = [keys.anchor_corner_key(j) for j in range(4)]
  corner_prior = gtsam.noiseModel.Isotropic.Sigma(3, 1.0e-4)
  corner_noise = huber_pixel_noise(args.anchor.corner_reproj_sigma)
  for corner_key, corner_xyz in zip(corner_keys, anchor_corners_world, strict=True):
    point = gtsam.Point3(*corner_xyz)
    values.insert(corner_key, point)
    graph.add(gtsam.PriorFactorPoint3(corner_key, point, corner_prior))

  n_anchor_sightings = 0
  for t in trackers:
    for lens, kf, anchor_det in _anchor_sightings(t, anchor_tag_id):
      n_anchor_sightings += 1
      pose_key = keys.pose_key(t.name, kf)
      # The detected tag direction in the lens frame disambiguates the omni
      # depth-sign: the corner reprojection deactivates for poses that flip the
      # frontally-observed tag to the opposite side of the camera.
      side = "front" if lens is t.front_lens else "rear"
      cam_T_tag = t.anchor_obs_per_kf.get(kf, {}).get(side)
      obs_bearing = (
        None
        if cam_T_tag is None
        else np.asarray(cam_T_tag.translation(), dtype=np.float64)
      )
      for corner_key, pixel in zip(corner_keys, anchor_det.corners, strict=True):
        add_mei_factor(
          graph, pose_key, corner_key, pixel, corner_noise, lens, obs_bearing
        )
  print(f"anchor corner reprojection: {n_anchor_sightings} anchor sightings")

  # Cross-tracker tag observation factors. ``tag_on_owner`` is the *owner's*
  # body→base-tag rigid offset, recovered from the owner's own rear-lens
  # PnP at its first stationary observation of its own tag.
  body_T_tag_by_name = {
    t.name: t.body_T_base_tag for t in trackers if t.body_T_base_tag is not None
  }
  for o in cross_obs:
    owner_offset = body_T_tag_by_name.get(o.tag_owner_tracker or "")
    if owner_offset is None:
      continue
    observer_lens = (
      next(t for t in trackers if t.name == o.observer_tracker).front_lens
      if o.fisheye_side == "front"
      else next(t for t in trackers if t.name == o.observer_tracker).rear_lens
    )
    graph.add(
      make_cross_tracker_factor(
        keys.pose_key(o.observer_tracker, o.keyframe_index),
        keys.pose_key(o.tag_owner_tracker or "", o.keyframe_index),
        o.camera_to_tag,
        observer_lens.extrinsic,
        owner_offset,
        args.cross.translation_sigma,
        args.cross.rotation_sigma,
      )
    )

  # Finger-tag cross-tracker constraints. Owner's body_T_finger varies per kf
  # (hinged joint). Use owner's per-kf rear-lens-derived body_T_finger as the
  # offset. Looser σ than base-tag cross because finger PnP is noisier (smaller
  # tag, hinged motion). OUTSIDE original ruleset.
  finger_cross_count = 0
  finger_cross_t_sigma = 0.040  # 40mm
  finger_cross_r_sigma = 0.10  # ~6°
  for owner in trackers:
    for finger_id, btf_per_kf in (
      (owner.left_finger_tag_id, owner.body_T_left_finger_per_kf),
      (owner.right_finger_tag_id, owner.body_T_right_finger_per_kf),
    ):
      if finger_id <= 0 or not btf_per_kf:
        continue
      for observer in trackers:
        if observer.name == owner.name:
          continue
        n = min(observer.n_keyframes, owner.n_keyframes)
        for lens, tags_per_kf in (
          (observer.front_lens, observer.front_tags_per_kf),
          (observer.rear_lens, observer.rear_tags_per_kf),
        ):
          for kf in range(n):
            if kf not in btf_per_kf:
              continue
            for det in tags_per_kf[kf]:
              if int(det.tag_id) != finger_id:
                continue
              cam_T_tag = estimate_camera_to_tag_pose(
                det.corners, args.gripper.tag_size, lens.params
              )
              if cam_T_tag is None:
                continue
              factor = make_cross_tracker_factor(
                keys.pose_key(observer.name, kf),
                keys.pose_key(owner.name, kf),
                cam_T_tag,
                lens.extrinsic,
                btf_per_kf[kf],
                finger_cross_t_sigma,
                finger_cross_r_sigma,
                huber_k=1.0,
              )
              graph.add(factor)
              finger_cross_count += 1
              break
  if finger_cross_count:
    print(f"finger-tag cross-tracker factors: {finger_cross_count}")

  # Pose-smoothness priors at trailing keyframes (after each tracker's
  # last anchor obs): a soft "consecutive body poses are close" Pose3
  # BetweenFactor with measurement = identity. Sigma matches the typical
  # handheld kf-to-kf motion (30 cm / 5°) so the prior is dominated by
  # the data when there's real motion but caps unbounded IMU drift over
  # the unanchored tail.
  smooth_sigmas = np.array([np.deg2rad(2.0)] * 3 + [0.10] * 3, dtype=np.float64)
  smooth_noise = gtsam.noiseModel.Diagonal.Sigmas(smooth_sigmas)
  vel_smooth_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.3, 0.3, 0.3], dtype=np.float64)
  )
  smooth_count = 0
  vel_smooth_count = 0
  zero_t = gtsam.Point3(0.0, 0.0, 0.0)
  for t in trackers:
    for k in range(1, t.n_keyframes):
      # Both endpoints must be unanchored — boundary cases (k-1 anchored,
      # k unanchored) over-constrain the LM by pulling k's pose toward
      # the anchor-pinned k-1, which fights visual factors at k.
      if (k - 1) in t.anchor_obs_per_kf or k in t.anchor_obs_per_kf:
        continue
      # Identity rotation mean would penalise actual handheld rotation
      # between consecutive un-anchored kfs; use the prep-stage relative
      # rotation (IMU-propagated through VI-aligned prep poses) instead.
      prep_rel = t.body_poses[k - 1].inverse().compose(t.body_poses[k])
      smooth_mean = gtsam.Pose3(prep_rel.rotation(), zero_t)
      graph.add(
        gtsam.BetweenFactorPose3(
          keys.pose_key(t.name, k - 1),
          keys.pose_key(t.name, k),
          smooth_mean,
          smooth_noise,
        )
      )
      smooth_count += 1
      vel_delta = np.asarray(t.velocities[k] - t.velocities[k - 1], dtype=np.float64)
      graph.add(
        gtsam.BetweenFactorVector(
          keys.velocity_key(t.name, k - 1),
          keys.velocity_key(t.name, k),
          vel_delta,
          vel_smooth_noise,
        )
      )
      vel_smooth_count += 1
  if smooth_count:
    print(f"unanchored-stretch pose smoothness priors: {smooth_count}")
  if vel_smooth_count:
    print(f"unanchored-stretch velocity smoothness priors: {vel_smooth_count}")

  initial_error = float(graph.error(values))
  print(
    f"\njoint graph: {graph.size()} factors, {values.size()} variables, "
    f"initial error={initial_error:.1f}"
  )
  result = optimize_graph(graph, values, max_iterations=args.solver.max_iterations)
  mark(f"joint LM ({result.iterations} iters)")
  print(
    f"  error: {result.initial_error:.1f} → {result.final_error:.1f} "
    f"({result.iterations} iters)"
  )

  stamp = time.strftime("%Y%m%d_%H%M%S")
  rrd_path = args.output_dir / f"track_{stamp}.rrd"

  urdf_path = args.urdf_path
  if urdf_path is None:
    try:
      from robot_descriptions import load_asset

      urdf_path = Path(
        load_asset(
          "robots/specialist_gripper/urdf/gripper.urdf",
          repo_url="https://github.com/T-K-233/specialist_description",
          cache_dir=str(Path.home() / ".cache" / "specialist_description"),
        )
      )
    except Exception as exc:  # headless env without the gripper-URDF package
      print(f"  [viz] gripper URDF unavailable ({exc}); skipping URDF overlay")
      urdf_path = None

  tag_size_by_id: dict[int, float] = {
    int(args.anchor.tag_id): float(args.anchor.tag_size),
    **{int(tid): float(args.gripper.tag_size) for tid in args.base_tag_ids},
    **{
      int(tid): float(args.gripper.tag_size)
      for tid in (*left_finger_ids, *right_finger_ids)
      if int(tid) > 0
    },
  }

  optimized_body_poses: dict[str, list[gtsam.Pose3]] = {
    t.name: [
      result.values.atPose3(keys.pose_key(t.name, i)) for i in range(t.n_keyframes)
    ]
    for t in trackers
  }

  # Per-tracker audio offset to convert local keyframe_times → master clock.
  master_offset_by_name: dict[str, float] = {
    t.name: offsets[i] for i, t in enumerate(trackers)
  }
  cross_obs_by_observer: dict[str, list[TagObservation]] = {
    t.name: [] for t in trackers
  }
  for o in cross_obs:
    cross_obs_by_observer[o.observer_tracker].append(o)

  per_tracker_results: list[dict[str, object]] = []
  viz_trackers: list[TrackerVizData] = []
  for t in trackers:
    poses = optimized_body_poses[t.name]
    f_pts = {
      tid: np.asarray(
        result.values.atPoint3(keys.front_landmark_key(t.name, tid)), dtype=np.float64
      )
      for tid in t.front_points
    }
    r_pts = {
      tid: np.asarray(
        result.values.atPoint3(keys.rear_landmark_key(t.name, tid)), dtype=np.float64
      )
      for tid in t.rear_points
    }
    front_rmse_per_kf = _per_keyframe_reproj_rmse_for_viz(
      result.values,
      t.front_tracks_tri,
      t.front_lens,
      lambda tid, n=t.name: keys.front_landmark_key(n, tid),
      lambda i, n=t.name: keys.pose_key(n, i),
      t.n_keyframes,
    )
    rear_rmse_per_kf = _per_keyframe_reproj_rmse_for_viz(
      result.values,
      t.rear_tracks_tri,
      t.rear_lens,
      lambda tid, n=t.name: keys.rear_landmark_key(n, tid),
      lambda i, n=t.name: keys.pose_key(n, i),
      t.n_keyframes,
    )
    f_rmse = _aggregate_rmse_for_print(front_rmse_per_kf)
    r_rmse = _aggregate_rmse_for_print(rear_rmse_per_kf)
    per_tracker_results.append(
      {
        "name": t.name,
        "n_keyframes": t.n_keyframes,
        "anchor_obs": len(t.anchor_obs_per_kf),
        "bounding_box_diagonal": _bounding_box_diagonal(poses),
        "path_length": _path_length(poses),
      }
    )
    print(
      f"  {t.name}: kf={t.n_keyframes} bbox={_bounding_box_diagonal(poses):.3f} m "
      f"front_rmse={f_rmse:.2f}px rear_rmse={r_rmse:.2f}px"
    )
    if t.body_T_base_tag is not None:
      base_tag_poses = [p.compose(t.body_T_base_tag) for p in poses]
    else:
      base_tag_poses = poses
    front_tag_viz = [
      [(d.tag_id, d.center, d.corners) for d in tags] for tags in t.front_tags_per_kf
    ]
    rear_tag_viz = [
      [(d.tag_id, d.center, d.corners) for d in tags] for tags in t.rear_tags_per_kf
    ]
    biases_final = [
      result.values.atConstantBias(keys.bias_key(t.name, i))
      for i in range(t.n_keyframes)
    ]
    front_tag_axes = _project_tag_axes_per_lens(
      t.front_tags_per_kf,
      t.front_lens,
      tag_size_by_id,
      t.n_keyframes,
    )
    rear_tag_axes = _project_tag_axes_per_lens(
      t.rear_tags_per_kf,
      t.rear_lens,
      tag_size_by_id,
      t.n_keyframes,
    )
    front_reproj = (
      _reproject_landmarks_per_kf(
        t.front_tracks_tri, f_pts, poses, t.front_lens, t.n_keyframes
      )
      if args.log_frames_in_rrd
      else None
    )
    rear_reproj = (
      _reproject_landmarks_per_kf(
        t.rear_tracks_tri, r_pts, poses, t.rear_lens, t.n_keyframes
      )
      if args.log_frames_in_rrd
      else None
    )
    front_rejected_kp = (
      _per_kf_rejected_keypoints(t.front_rejected, t.n_keyframes)
      if args.log_frames_in_rrd
      else None
    )
    rear_rejected_kp = (
      _per_kf_rejected_keypoints(t.rear_rejected, t.n_keyframes)
      if args.log_frames_in_rrd
      else None
    )
    anchor_lines, cross_lines = _factor_observation_lines(
      t, poses, cross_obs_by_observer[t.name]
    )
    viz_trackers.append(
      TrackerVizData(
        name=t.name,
        body_poses=poses,
        keyframe_times=np.asarray(t.keyframe_times, dtype=np.float64)
        - master_offset_by_name[t.name],
        front_points=f_pts,
        rear_points=r_pts,
        front_frames=t.front_frames if args.log_frames_in_rrd else None,
        rear_frames=t.rear_frames if args.log_frames_in_rrd else None,
        front_keypoints=t.front_keypoints if args.log_frames_in_rrd else None,
        rear_keypoints=t.rear_keypoints if args.log_frames_in_rrd else None,
        front_tag_detections=front_tag_viz,
        rear_tag_detections=rear_tag_viz,
        velocities=[
          np.asarray(result.values.atVector(keys.velocity_key(t.name, i)))
          for i in range(t.n_keyframes)
        ],
        gyro_biases=[np.asarray(b.gyroscope(), dtype=np.float64) for b in biases_final],
        accel_biases=[
          np.asarray(b.accelerometer(), dtype=np.float64) for b in biases_final
        ],
        front_rmse_per_kf=front_rmse_per_kf,
        rear_rmse_per_kf=rear_rmse_per_kf,
        front_tag_axes=front_tag_axes,
        rear_tag_axes=rear_tag_axes,
        front_landmark_reprojections=front_reproj,
        rear_landmark_reprojections=rear_reproj,
        front_rejected_keypoints=front_rejected_kp,
        rear_rejected_keypoints=rear_rejected_kp,
        rear_static_mask_rects=t.rear_static_mask_rects,
        anchor_obs_lines=anchor_lines,
        cross_tracker_lines=cross_lines,
        urdf_root_poses=base_tag_poses,
        joint_angles=t.joint_angles or None,
        trajectory_offset_in_gripper=args.gripper.keypoint_offset,
        front_lens_extrinsic=t.front_lens.extrinsic,
        rear_lens_extrinsic=t.rear_lens.extrinsic,
      )
    )

  bbox_diag = tuple(
    _bounding_box_diagonal(optimized_body_poses[t.name]) for t in trackers
  )
  path_len = tuple(_path_length(optimized_body_poses[t.name]) for t in trackers)
  summary = Summary(
    sources=tuple(str(p) for p in args.input_insvs),
    tracker_names=names,
    audio_offsets=tuple(offsets),
    per_tracker_keyframes=tuple(t.n_keyframes for t in trackers),
    anchor_obs_per_tracker=tuple(len(t.anchor_obs_per_kf) for t in trackers),
    cross_tracker_obs=len(cross_obs),
    initial_graph_error=result.initial_error,
    final_graph_error=result.final_error,
    solver_iterations=result.iterations,
    bounding_box_diagonal=bbox_diag,
    trajectory_path_length=path_len,
  )
  print("\n=== max kf-to-kf speed (real motion if IMU agrees) ===")
  for name, poses in optimized_body_poses.items():
    pts = np.array([np.asarray(p.translation(), dtype=np.float64) for p in poses])
    if pts.shape[0] < 2:
      continue
    t_obj = next(t for t in trackers if t.name == name)
    kf_t = np.asarray(t_obj.keyframe_times, dtype=np.float64)
    dt = np.diff(kf_t)
    speed = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.where(dt > 0, dt, 1.0)
    i = int(np.argmax(speed))
    print(f"  [{name}] max={speed[i]:.2f} m/s at kf {i}→{i + 1} (t={kf_t[i]:.2f}s)")

  # Raw observations for downstream evaluation. The pipeline does
  # not compute quality metrics — downstream tooling reads this JSON,
  # reconstructs the geometry as numpy arrays, and writes a separate
  # ``track_*.eval.json``. Keep this file free of derived metrics so
  # the eval module can be deleted without breaking the pipeline.
  raw: dict[str, object] = {}
  raw["body_poses_by_tracker"] = {
    t.name: [_pose_to_dict(p) for p in optimized_body_poses[t.name]] for t in trackers
  }
  raw["ref_anchor_pose_world"] = _pose_to_dict(ref_anchor)
  raw["lens_by_tracker"] = {
    t.name: {
      "front": _lens_side_dict(t.front_lens, t.decoded_width),
      "rear": _lens_side_dict(t.rear_lens, t.decoded_width),
    }
    for t in trackers
  }
  raw["anchor_obs_by_tracker"] = {
    t.name: [
      {"kf": int(kf), "side": side, "cam_T_tag": _pose_to_dict(cam_T_tag)}
      for kf, sides in t.anchor_obs_per_kf.items()
      for side, cam_T_tag in sides.items()
    ]
    for t in trackers
  }
  raw["tag_detections_by_tracker"] = {
    t.name: {
      "front": _tags_side_list(t.front_tags_per_kf),
      "rear": _tags_side_list(t.rear_tags_per_kf),
    }
    for t in trackers
  }
  raw["tracks_by_tracker"] = {
    t.name: {
      "front": _tracks_side_dict(t.front_tracks_tri),
      "rear": _tracks_side_dict(t.rear_tracks_tri),
    }
    for t in trackers
  }
  raw["landmarks_by_tracker"] = {
    t.name: {
      "front": {
        str(tid): [
          float(x) for x in result.values.atPoint3(keys.front_landmark_key(t.name, tid))
        ]
        for tid in t.front_points
      },
      "rear": {
        str(tid): [
          float(x) for x in result.values.atPoint3(keys.rear_landmark_key(t.name, tid))
        ]
        for tid in t.rear_points
      },
    }
    for t in trackers
  }
  raw["kf_times_by_tracker"] = {
    t.name: [float(x) for x in t.keyframe_times] for t in trackers
  }
  raw["audio_offset_by_tracker"] = master_offset_by_name
  raw["body_T_base_tag_by_tracker"] = {
    t.name: (_pose_to_dict(t.body_T_base_tag) if t.body_T_base_tag else None)
    for t in trackers
  }
  raw["anchor_tag_id"] = int(args.anchor.tag_id)
  raw["anchor_tag_size"] = float(args.anchor.tag_size)
  raw["gripper_keypoint_offset"] = [float(x) for x in args.gripper.keypoint_offset]

  summary_dict = asdict(summary)
  summary_dict["per_tracker"] = per_tracker_results
  summary_dict.update(raw)
  json_path = args.output_dir / f"track_{stamp}.json"
  json_path.write_text(json.dumps(summary_dict, indent=2, default=str))
  print(f"\nWrote {json_path}")

  # Viz runs after the result JSON is on disk and is best-effort: a viz failure
  # must not mask a solve. On a headless host the spawned viewer can't open and
  # the rerun SDK buffers the (possibly dense) scene in memory, which can OOM.
  try:
    log_multi_ba_run(
      rrd_path=rrd_path,
      trackers=viz_trackers,
      world_T_anchor=ref_anchor,
      anchor_tag_size=args.anchor.tag_size,
      anchor_tag_id=args.anchor.tag_id,
      mask_pad_px=args.frontend.mask_tag_pad,
      urdf_path=urdf_path,
      open_viewer=args.open_viewer,
    )
    print(f"Wrote {rrd_path}")
  except Exception as exc:  # noqa: BLE001 - viz is best-effort; JSON already saved
    print(f"  [viz] skipped ({exc})")


if __name__ == "__main__":
  main(tyro.cli(Args))
