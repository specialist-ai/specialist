"""VI-BA driver for any number of trackers (N ≥ 1).

Reads one session's base recording, runs the per-tracker prep (decode →
frontend → anchor PnP → VI alignment), aligns every tracker into the reference
anchor's world, and solves one joint bundle over:

- Anchor observations: every static anchor is a pose variable seeded from the
  prep; the known tag size pins absolute scale and the reference carries the
  datum.
- Base + finger tags: each tracker's tags reproject through one shared
  base-tag pose variable, recovering the finger joint angles.
- Cross-tracker observations (N ≥ 2): tracker O sighting tracker W's base tag
  ties the two trajectories.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gtsam
import numpy as np
import tyro
from specialist_dataset import schema

from specialist.record.layer import write_estimate_layers
from specialist.record.session import SessionRecording
from specialist.vi_ba.anchor_detection import configure_detector
from specialist.vi_ba.blueprint import blueprint_for_slots
from specialist.vi_ba.compose import (
  TrackerKeys,
  anchored_keyframes,
  cross_sightings,
)
from specialist.vi_ba.configs import Args, TrackerArgs, validate
from specialist.vi_ba.features import set_precompute_concurrency
from specialist.vi_ba.geometry import translation
from specialist.vi_ba.gpu_lm import assert_backend
from specialist.vi_ba.reference_solve import (
  adopt_ba_refined_estimates,
  solve_joint,
)
from specialist.vi_ba.single_tracker import (
  PreparedTracker,
  captures_that_fit,
  discover_tracker_tags,
  prepare_tracker,
)
from specialist.vi_ba.summary import Summary
from specialist.vi_ba.tag_discovery import GripperTags, validate_discovered
from specialist.vi_ba.urdf_joints import read_revolute_joints
from specialist.vi_ba.viz_build import build_run_estimate


def _resolve_urdf_path(explicit: Path | None) -> Path:
  """The gripper URDF: an explicit path, else the cached/downloaded asset."""
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
  except Exception as exc:
    raise SystemExit(
      f"the gripper URDF is unavailable ({exc}); pass --urdf-path"
    ) from exc


def _stage_timer() -> Callable[[str], None]:
  last = time.perf_counter()

  def mark(stage: str) -> None:
    nonlocal last
    now = time.perf_counter()
    print(f"  [timing] {stage} +{now - last:.1f}s")
    last = now

  return mark


def _print_capture_diagnostics(
  trackers: list[PreparedTracker],
  optimized_body_poses: dict[str, list[gtsam.Pose3]],
) -> None:
  print("\n=== max kf-to-kf speed (real motion if IMU agrees) ===")
  for t in trackers:
    poses = optimized_body_poses[t.name]
    pts = np.array([translation(p) for p in poses])
    if pts.shape[0] < 2:
      continue
    kf_t = np.asarray(t.keyframe_times, dtype=np.float64)
    dt = np.diff(kf_t)
    speed = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.where(dt > 0, dt, 1.0)
    i = int(np.argmax(speed))
    print(f"  [{t.name}] max={speed[i]:.2f} m/s at kf {i}→{i + 1} (t={kf_t[i]:.2f}s)")


def main(args: Args) -> None:
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  session = SessionRecording(args.base)
  names = tuple(session.slots)
  validate(args)
  if args.solver.gpu == "require":
    assert_backend()
  rrd_path = schema.layer_rrd(args.output_dir, schema.ESTIMATE, session.session_id)
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

  urdf_path = _resolve_urdf_path(args.urdf_path)

  concurrent = captures_that_fit(session, names)
  set_precompute_concurrency(concurrent)

  joints = read_revolute_joints(urdf_path)
  specs = {
    name: TrackerArgs(
      base_rrd=args.base,
      tracker=name,
      keyframe_every=args.keyframe_every,
      frontend=args.frontend,
      anchor=args.anchor,
      parallel_siblings=concurrent > 1,
      gripper=args.gripper,
      joints=joints,
    )
    for name in names
  }

  def _discover_one(name: str) -> GripperTags:
    try:
      return discover_tracker_tags(specs[name])
    except Exception as exc:
      if concurrent > 1:
        print(f"\n=== discover tracker {name!r} FAILED ===\n{exc}")
      raise

  if concurrent == 1:
    tags = {name: _discover_one(name) for name in names}
  else:
    configure_detector(args.frontend.parallel_tag_detect)
    with ThreadPoolExecutor(max_workers=concurrent) as discovery_pool:
      tags = dict(zip(names, discovery_pool.map(_discover_one, names), strict=True))
  validate_discovered(tags)
  for name, found in tags.items():
    print(f"  {name}: base tag {found.base}, finger tags {found.fingers}")
  mark("tags discovered")

  def _prep_one(name: str) -> PreparedTracker:
    print(f"\n=== prep tracker {name!r} ===")
    try:
      tracker = prepare_tracker(specs[name], tags[name])
    except Exception as exc:
      if concurrent > 1:
        print(f"\n=== prep tracker {name!r} FAILED: {exc!r} ===")
      raise
    print(
      f"  [{name}/prep] kf={tracker.n_keyframes}  "
      f"anchor_obs={len(anchored_keyframes(tracker))}  "
      f"front_pts={len(tracker.front.points)}  rear_pts={len(tracker.rear.points)}  "
      f"tags={tracker.tags.base}/{tracker.tags.fingers[0]}/{tracker.tags.fingers[1]}"
    )
    return tracker

  from specialist.vi_ba import gpu_flow

  try:
    if concurrent == 1:
      trackers: list[PreparedTracker] = [_prep_one(name) for name in names]
    else:
      with ThreadPoolExecutor(max_workers=concurrent) as prep_pool:
        trackers = list(prep_pool.map(_prep_one, names))
  finally:
    # Release the per-stream GPU workers so the linear solver has device memory.
    gpu_flow.shutdown()
  mark("trackers prepared")

  cross_obs = cross_sightings(trackers, args.gripper.tag_size)
  print(f"[graph] cross-tracker observations: {len(cross_obs)}")
  if cross_obs:
    by_pair = Counter((sighting.observer, sighting.owner) for sighting in cross_obs)
    for (obs_t, own_t), n in sorted(by_pair.items()):
      print(f"  [graph] {obs_t} → {own_t}: {n}")
  mark("cross-tracker obs assembled")

  keys = TrackerKeys(list(names))
  solve = solve_joint(trackers, keys, args, cross_obs)
  mark("joint LM (cull-reoptimize)")

  adopt_ba_refined_estimates(trackers, solve.result, keys)

  estimate = build_run_estimate(trackers, solve, keys, args, cross_obs, joints)

  worst_move_translation, worst_move_rotation = solve.worst_anchor_move
  summary = Summary(
    cross_camera_obs=len(cross_obs),
    initial_graph_error=solve.result.initial_error,
    final_graph_error=solve.result.final_error,
    solver_iterations=solve.result.iterations,
    anchor_move_translation=worst_move_translation,
    anchor_move_rotation=worst_move_rotation,
  )
  _print_capture_diagnostics(trackers, solve.body_poses)

  timeline_start = min(
    float(session.lens_times(n, lens)[0]) for n in names for lens in ("front", "rear")
  )
  frame_period = float(np.median(np.diff(session.lens_times(names[0], "front"))))
  write_estimate_layers(
    estimate_path=rrd_path,
    viz_path=viz_path,
    timeline_start=timeline_start,
    frame_period=frame_period,
    estimate=estimate,
    summary=summary,
    anchor_tag_size=args.anchor.tag_size,
    urdf_path=urdf_path,
    session_id=session.session_id,
  )
  # The dashboard for sessions opened straight from their files, refreshed by
  # every solve so it always matches the layers beside it.
  blueprints_dir = args.output_dir / "blueprints"
  blueprints_dir.mkdir(parents=True, exist_ok=True)
  rbl = blueprints_dir / "specialist.rbl"
  blueprint_for_slots(
    [c.slot for c in estimate.cameras],
    anchor_tag_ids=sorted(estimate.anchor_poses_world),
    anchor_tag_size=args.anchor.tag_size,
  ).save(schema.APPLICATION_ID, str(rbl))
  print(f"Wrote {rrd_path}, {viz_path} and {rbl}")


if __name__ == "__main__":
  main(tyro.cli(Args))
