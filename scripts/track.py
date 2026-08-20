"""VI-BA driver for any number of trackers (N ≥ 1).

Reads one session's base recording (already audio-synced and window-trimmed
by ``scripts/convert.py``), runs the per-tracker prep (decode → frontend →
anchor PnP → VI alignment), aligns every tracker into the reference anchor's
world (through any co-sighted anchor), allocates collision-free GTSAM keys
via ``TrackerKeys``, builds one VI subgraph per tracker, then adds:

- Anchor obs: reprojecting the reference anchor tag's four metric corners
  pins the trajectory's absolute scale; each additional static anchor is
  corner-reprojected from every sighting at a world pose seeded from the
  prep and re-estimated after the solve as a consistency check.
- Base + finger tags: each tracker's base and finger tags reproject through one
  shared base-tag pose variable, recovering the finger joint angles.
- Cross-tracker obs (N ≥ 2): when tracker O sees tracker W's base tag,
  reprojecting its corners through O's pose, W's pose, and W's base-tag pose
  ties the two trajectories.
"""

from __future__ import annotations

import dataclasses
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gtsam
import numpy as np
import tyro

from specialist.record.layer import write_solution_layers
from specialist.record.session import SessionRecording
from specialist.vi_ba.compose import (
  TrackerKeys,
  anchored_keyframes,
  cross_tracker_observations,
)
from specialist.vi_ba.configs import Args, validate
from specialist.vi_ba.features import set_precompute_concurrency
from specialist.vi_ba.reference_solve import (
  adopt_ba_refined_estimates,
  solve_joint,
)
from specialist.vi_ba.single_tracker import (
  PreparedTracker,
  TrackerArgs,
  captures_that_fit,
  prepare_tracker,
)
from specialist.vi_ba.summary import Summary
from specialist.vi_ba.viz_build import build_run_outputs


def _resolve_urdf_path(explicit: Path | None) -> Path | None:
  """The gripper URDF: an explicit path, else the cached/downloaded asset, else
  None (a headless env without the gripper-URDF package, which disables the
  finger joint-angle factors and the URDF viz overlay)."""
  if explicit is not None:
    return explicit
  try:
    from robot_descriptions import load_asset

    return Path(
      load_asset(
        "robots/specialist_gripper/urdf/gripper.urdf",
        repo_url="https://github.com/T-K-233/specialist_description",
        cache_dir=str(Path.home() / ".cache" / "specialist_description"),
      )
    )
  except Exception as exc:  # headless env without the gripper-URDF package
    print(f"  gripper URDF unavailable ({exc}); finger joint factors disabled")
    return None


def _stage_timer() -> Callable[[str], None]:
  last = time.perf_counter()

  def mark(stage: str) -> None:
    nonlocal last
    now = time.perf_counter()
    print(f"  [{stage}] +{now - last:.1f}s")
    last = now

  return mark


def _non_default_flags(args: object, prefix: str = "") -> list[str]:
  """Every defaulted field whose value differs from its declaration, dotted
  through the nested configs. Required fields carry no default and are the
  run's identity rather than its tuning, so they are not listed."""
  out: list[str] = []
  for spec in dataclasses.fields(args):  # type: ignore[arg-type]
    value = getattr(args, spec.name)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
      out += _non_default_flags(value, f"{prefix}{spec.name}.")
      continue
    if spec.default is not dataclasses.MISSING:
      default = spec.default
    elif spec.default_factory is not dataclasses.MISSING:
      default = spec.default_factory()
    else:
      continue
    if value != default:
      out.append(f"{prefix}{spec.name}={value}")
  return out


def _print_capture_diagnostics(
  trackers: list[PreparedTracker],
  optimized_body_poses: dict[str, list[gtsam.Pose3]],
) -> None:
  """Print the max kf-to-kf speed (a sanity check that any large jump is real
  IMU-agreeing motion)."""
  print("\n=== max kf-to-kf speed (real motion if IMU agrees) ===")
  for t in trackers:
    poses = optimized_body_poses[t.name]
    pts = np.array([np.asarray(p.translation(), dtype=np.float64) for p in poses])
    if pts.shape[0] < 2:
      continue
    kf_t = np.asarray(t.keyframe_times, dtype=np.float64)
    dt = np.diff(kf_t)
    speed = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.where(dt > 0, dt, 1.0)
    i = int(np.argmax(speed))
    print(f"  [{t.name}] max={speed[i]:.2f} m/s at kf {i}→{i + 1} (t={kf_t[i]:.2f}s)")


def main(args: Args) -> None:
  session = SessionRecording(args.base)
  names = tuple(session.tracker_names)
  validate(args, len(names))
  rrd_path = args.output_dir / "track" / f"{session.session_id}.rrd"
  viz_path = args.output_dir / "viz" / f"{session.session_id}.rrd"
  existing = [p for p in (rrd_path, viz_path) if p.exists()]
  if existing and not args.overwrite:
    raise SystemExit(
      f"{', '.join(str(p) for p in existing)} exists; pass --overwrite to "
      f"replace it (a registered layer must also be re-registered, or its "
      f"byte-range index goes stale)"
    )
  rrd_path.parent.mkdir(parents=True, exist_ok=True)
  viz_path.parent.mkdir(parents=True, exist_ok=True)
  mark = _stage_timer()
  print(f"session {session.session_id}: trackers {list(names)}")

  left_finger_ids = args.left_finger_tag_ids or (0,) * len(names)
  right_finger_ids = args.right_finger_tag_ids or (0,) * len(names)
  specs = list(
    zip(names, args.base_tag_ids, left_finger_ids, right_finger_ids, strict=True)
  )

  # The gripper URDF drives both the finger joint-angle factors (prep) and the
  # viz overlay, so resolve it once up front.
  urdf_path = _resolve_urdf_path(args.urdf_path)

  # The flow precompute needs the capture count to shrink its per-lens worker
  # pool, so it does not oversubscribe the shared cv2 thread pool. It also
  # decides whether the tag walks ever overlap, which is what per-thread
  # detectors cost their memory for.
  concurrent = captures_that_fit(session, names)
  set_precompute_concurrency(concurrent)

  def _prep_one(spec: tuple[str, int, int, int]) -> PreparedTracker:
    name, base_tag_id, left_id, right_id = spec
    print(f"\n=== prep tracker {name!r} ===")
    try:
      tracker = prepare_tracker(
        name,
        TrackerArgs(
          base_rrd=args.base,
          tracker=name,
          keyframe_every=args.keyframe_every,
          base_tag_id=base_tag_id,
          left_finger_tag_id=left_id,
          right_finger_tag_id=right_id,
          frontend=args.frontend,
          imu=args.imu,
          anchor=args.anchor,
          parallel_siblings=concurrent > 1,
          gripper=args.gripper,
          urdf_path=urdf_path,
        ),
      )
    except Exception as exc:
      # A sibling capture already decoding cannot be interrupted, so the
      # traceback only lands once it finishes; name the failure when it happens
      # rather than leave the wait looking like a hang.
      if concurrent > 1:
        print(f"\n=== prep tracker {name!r} FAILED: {exc!r} ===", flush=True)
      raise
    print(
      f"  kf={tracker.n_keyframes}  anchor_obs={len(anchored_keyframes(tracker))}  "
      f"front_pts={len(tracker.front_points)}  rear_pts={len(tracker.rear_points)}  "
      f"body_T_base_tag={'yes' if tracker.body_T_base_tag else 'no'}"
    )
    return tracker

  from specialist.vi_ba import gpu_flow

  try:
    if concurrent == 1:
      trackers: list[PreparedTracker] = [_prep_one(spec) for spec in specs]
    else:
      with ThreadPoolExecutor(max_workers=concurrent) as prep_pool:
        trackers = list(prep_pool.map(_prep_one, specs))
  finally:
    # Flow is finished; release the per-stream GPU workers so the linear solver
    # has device memory for the joint bundle adjustment. A no-op on the cv2 path
    # (no stream ever primed).
    gpu_flow.shutdown()
  mark("trackers prepared")

  cross_obs = cross_tracker_observations(
    trackers, list(args.base_tag_ids), args.gripper.tag_size
  )
  print(f"cross-tracker observations: {len(cross_obs)}")
  if cross_obs:
    by_pair = Counter((sighting.observer, sighting.owner) for sighting in cross_obs)
    for (obs_t, own_t), n in sorted(by_pair.items()):
      print(f"  {obs_t} → {own_t}: {n}")
  mark("cross-tracker obs assembled")

  keys = TrackerKeys(list(names))
  solve = solve_joint(trackers, keys, args, cross_obs)
  print(f"\nshared world: anchor t={np.asarray(solve.ref_anchor.translation())}")
  mark("joint LM (cull-reoptimize)")

  adopt_ba_refined_estimates(trackers, solve.result, keys)

  outputs = build_run_outputs(
    trackers,
    solve.body_poses,
    solve.result,
    keys,
    solve.landmark_keep,
    args,
    cross_obs,
    solve.ref_anchor,
    solve.ref_tag_id,
    solve.secondary_world,
    urdf_path=urdf_path,
  )

  worst_move_translation, worst_move_rotation = solve.worst_secondary_move
  summary = Summary(
    non_default_flags=tuple(_non_default_flags(args)),
    sources=tuple(session.sources),
    tracker_names=names,
    cross_tracker_obs=len(cross_obs),
    initial_graph_error=solve.result.initial_error,
    final_graph_error=solve.result.final_error,
    solver_iterations=solve.result.iterations,
    secondary_anchor_move_translation=worst_move_translation,
    secondary_anchor_move_rotation=worst_move_rotation,
  )
  _print_capture_diagnostics(trackers, solve.body_poses)

  timeline_start = min(
    float(session.lens_times(n, lens)[0]) for n in names for lens in ("front", "rear")
  )
  # The frame period bounds each image overlay to the one video frame it was
  # computed from.
  frame_period = float(np.median(np.diff(session.lens_times(names[0], "front"))))
  write_solution_layers(
    track_path=rrd_path,
    viz_path=viz_path,
    timeline_start=timeline_start,
    frame_period=frame_period,
    outputs=outputs,
    summary=summary,
    reference_tag_id=solve.ref_tag_id,
    anchor_tag_size=args.anchor.tag_size,
    mask_pad_px=args.frontend.mask_tag_pad if args.frontend.mask_tags else -1,
    urdf_path=urdf_path,
    session_id=session.session_id,
    open_viewer=args.open_viewer,
  )
  print(f"Wrote {rrd_path} and {viz_path}")


if __name__ == "__main__":
  main(tyro.cli(Args))
