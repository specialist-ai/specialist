"""Solve the joint VI-BA graph and pick the reference anchor.

``specialist.vi_ba.compose`` builds the graph for one fixed reference anchor;
this module owns what sits above that: solving it with a landmark
cull-reoptimize pass, re-estimating each secondary anchor from the solved
poses as a consistency check, and — when more than one anchor is configured —
retrying that whole solve once per candidate reference so the exported
geometry does not depend on the configured tag-id order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import gtsam
import numpy as np

from specialist.vi_ba.anchor_detection import tag_corners_in_frame
from specialist.vi_ba.compose import (
  CrossSighting,
  LandmarkKeep,
  TrackerKeys,
  align_to_shared_world,
  assemble_joint_graph,
  points_from_values,
)
from specialist.vi_ba.configs import Args, GpuMode
from specialist.vi_ba.geometry import average_pose
from specialist.vi_ba.landmark_cull import cull_outlier_landmarks
from specialist.vi_ba.optimize import OptimizeResult, optimize_graph
from specialist.vi_ba.single_tracker import PreparedTracker
from specialist.vi_ba.urdf_joints import FINGER_JOINTS

__all__ = [
  "ReferenceSolve",
  "adopt_ba_refined_estimates",
  "solve_joint",
]

# Within-take consistency thresholds: the post-solve anchor re-estimate should
# agree with the pinned seed; larger motion flags weak sighting geometry.
_ANCHOR_CONSISTENCY_TRANS = 5e-3
_ANCHOR_CONSISTENCY_ROT = float(np.radians(0.2))

# The first solve only ranks residuals for the cull.
_PRECULL_MAX_ITERATIONS = 8


def _prune_factor_graph(
  graph: gtsam.NonlinearFactorGraph, drop_keys: set[int]
) -> gtsam.NonlinearFactorGraph:
  """A copy of ``graph`` with every factor touching a dropped variable key
  removed."""
  pruned = gtsam.NonlinearFactorGraph()
  for i in range(graph.size()):
    factor = graph.at(i)
    if not any(k in drop_keys for k in factor.keys()):
      pruned.push_back(factor)
  return pruned


def _outlier_landmark_keys(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  landmark_keep: LandmarkKeep,
) -> set[int]:
  """The landmark keys the cull dropped, across both lenses of every tracker."""
  drop: set[int] = set()
  for t in trackers:
    front_keep, rear_keep = landmark_keep[t.name]
    drop.update(
      keys.front_landmark_key(t.name, tid)
      for tid in t.front.points
      if tid not in front_keep
    )
    drop.update(
      keys.rear_landmark_key(t.name, tid)
      for tid in t.rear.points
      if tid not in rear_keep
    )
  return drop


def _solve_with_cull_reoptimize(
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  max_iterations: int,
  gpu: GpuMode,
) -> tuple[OptimizeResult, LandmarkKeep]:
  """Solve the joint graph, then drop the landmarks the first solve exposes as
  outliers (mismatched / degenerate) and re-solve warm-started, so they bias
  neither the exported map nor the poses. The prep is untouched, so this costs
  one extra seconds-long solve. Returns the re-solved result and, per tracker,
  the kept ``(front, rear)`` landmark tracks."""
  print(f"\njoint graph: {graph.size()} factors, {values.size()} variables")
  result = optimize_graph(
    graph,
    values,
    max_iterations=min(max_iterations, _PRECULL_MAX_ITERATIONS),
    gpu=gpu,
  )
  print(
    f"  solve 1: error {result.initial_error:.1f} → {result.final_error:.1f} "
    f"({result.iterations} iters)"
  )

  poses_by_tracker = _solved_body_poses(trackers, keys, result.values)
  landmark_keep: LandmarkKeep = {}
  for t in trackers:
    poses_1 = poses_by_tracker[t.name]
    f_pts_1 = points_from_values(
      result.values, keys.front_landmark_key, t.name, t.front.points
    )
    r_pts_1 = points_from_values(
      result.values, keys.rear_landmark_key, t.name, t.rear.points
    )
    front_keep = cull_outlier_landmarks(f_pts_1, t.front.tracks, poses_1, t.front.setup)
    rear_keep = cull_outlier_landmarks(r_pts_1, t.rear.tracks, poses_1, t.rear.setup)
    landmark_keep[t.name] = (front_keep, rear_keep)
  first = result
  outlier_keys = _outlier_landmark_keys(trackers, keys, landmark_keep)
  if not outlier_keys and result.iterations >= _PRECULL_MAX_ITERATIONS:
    # Nothing to cull, but the ranking solve stopped at its own short budget
    # rather than converging; finish on the full budget so the export never
    # carries a deliberately under-iterated result.
    result = optimize_graph(
      graph, result.values, max_iterations=max_iterations, gpu=gpu
    )
  if outlier_keys:
    pruned = _prune_factor_graph(graph, outlier_keys)
    kept_values = gtsam.Values(result.values)
    for key in outlier_keys:
      kept_values.erase(key)
    result = optimize_graph(pruned, kept_values, max_iterations=max_iterations, gpu=gpu)
    result = replace(
      result,
      initial_error=first.initial_error,
      iterations=first.iterations + result.iterations,
    )
    print(
      f"  cull-reoptimize: dropped {len(outlier_keys)} outlier landmarks, "
      f"error → {result.final_error:.1f} ({result.iterations} iters)"
    )
  return result, landmark_keep


def _secondary_anchor_seed(
  trackers: list[PreparedTracker],
  tag_id: int,
  body_poses_by_tracker: dict[str, list[gtsam.Pose3]],
) -> gtsam.Pose3 | None:
  """Median of ``world_T_body(kf) ∘ body_T_lens ∘ cam_T_tag`` over every PnP
  sighting, both lenses, against the poses in ``body_poses_by_tracker``.
  Seeding from sightings keeps the pose in front of its observers, so the
  behind-camera guard stays inactive. ``None`` when unsighted."""
  candidates: list[gtsam.Pose3] = []
  for t in trackers:
    poses = body_poses_by_tracker[t.name]
    obs = t.anchor_obs_by_tag.get(tag_id, {})
    for kf in sorted(obs):
      sides = obs[kf]
      for ch in t.lenses:
        cam_T_tag = sides.get(ch.side)
        if cam_T_tag is None:
          continue
        candidates.append(poses[kf].compose(ch.setup.extrinsic).compose(cam_T_tag))
  if not candidates:
    return None
  return average_pose(candidates, "median")


def _seed_secondary_anchors(
  trackers: list[PreparedTracker], args: Args
) -> dict[int, gtsam.Pose3]:
  """Initial world pose per sighted non-global anchor, from the BFS-aligned prep
  poses (the reference pins the gauge separately). Unsighted ids are skipped."""
  prep_poses = {t.name: t.body_poses for t in trackers}
  world: dict[int, gtsam.Pose3] = {}
  for tag_id in args.anchor.enabled_tag_ids[1:]:
    seed = _secondary_anchor_seed(trackers, tag_id, prep_poses)
    if seed is None:
      print(f"anchor {tag_id}: no PnP sighting; skipped")
      continue
    world[tag_id] = seed
  return world


def _solved_body_poses(
  trackers: list[PreparedTracker], keys: TrackerKeys, values: gtsam.Values
) -> dict[str, list[gtsam.Pose3]]:
  """Each tracker's optimized body poses pulled out of ``values``."""
  return {
    t.name: [values.atPose3(keys.pose_key(t.name, i)) for i in range(t.n_keyframes)]
    for t in trackers
  }


def _secondary_pose_move(
  pinned: gtsam.Pose3, estimate: gtsam.Pose3
) -> tuple[float, float]:
  """Translation (m) and rotation (rad) separating ``pinned`` from ``estimate``."""
  dt = float(
    np.linalg.norm(
      np.asarray(estimate.translation()) - np.asarray(pinned.translation())
    )
  )
  dr = float(
    np.linalg.norm(gtsam.Rot3.Logmap(pinned.rotation().between(estimate.rotation())))
  )
  return dt, dr


@dataclass
class ReferenceSolve:
  """Everything the export needs from one reference's joint solve, kept so the
  selected candidate is reused without re-solving."""

  ref_tag_id: int
  ref_anchor: gtsam.Pose3
  result: OptimizeResult
  landmark_keep: LandmarkKeep
  secondary_world: dict[int, gtsam.Pose3]
  body_poses: dict[str, list[gtsam.Pose3]]
  # Per secondary anchor, how far its post-solve re-estimate moved off its seed
  # (metres, radians): the sighting-geometry consistency check.
  secondary_moves: dict[int, tuple[float, float]]

  @property
  def worst_secondary_move(self) -> tuple[float, float]:
    """The largest translation and rotation any secondary anchor moved, NaN
    when the take has no secondary."""
    if not self.secondary_moves:
      return float("nan"), float("nan")
    moves = self.secondary_moves.values()
    return max(dt for dt, _ in moves), max(dr for _, dr in moves)


def _solve_joint_graph(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  args: Args,
  cross_obs: list[CrossSighting],
  ref_anchor: gtsam.Pose3,
) -> ReferenceSolve:
  """Solve the joint graph with the reference anchor pinned at ``ref_anchor``
  and resolve every secondary anchor's world pose.

  Solves once with the secondaries pinned at their prep-composed seeds, then
  re-estimates each from the solved poses. The solve carries the re-estimates,
  which are what gets exported, and how far each moved off its seed — the
  consistency signal the run summary records."""
  seed = _seed_secondary_anchors(trackers, args)
  graph, values = assemble_joint_graph(
    trackers,
    keys,
    args,
    cross_obs,
    tag_corners_in_frame(ref_anchor, args.anchor.tag_size),
    seed,
  )
  result, landmark_keep = _solve_with_cull_reoptimize(
    graph, values, trackers, keys, args.solver.max_iterations, args.solver.gpu
  )

  solved_poses = _solved_body_poses(trackers, keys, result.values)
  resolved: dict[int, gtsam.Pose3] = {}
  moves: dict[int, tuple[float, float]] = {}
  for tag_id in seed:
    estimate = _secondary_anchor_seed(trackers, tag_id, solved_poses)
    assert estimate is not None  # seeding required a sighting; obs never shrink
    dt, dr = _secondary_pose_move(seed[tag_id], estimate)
    weak = dt > _ANCHOR_CONSISTENCY_TRANS or dr > _ANCHOR_CONSISTENCY_ROT
    note = "  WARNING: weak sighting geometry this take" if weak else ""
    print(
      f"  secondary anchor {tag_id}: post-solve re-estimate moved "
      f"{dt * 1000:.1f}mm / {np.degrees(dr):.2f}deg{note}"
    )
    resolved[tag_id] = estimate
    moves[tag_id] = (dt, dr)
  return ReferenceSolve(
    ref_tag_id=int(args.anchor.enabled_tag_ids[0]),
    ref_anchor=ref_anchor,
    result=result,
    landmark_keep=landmark_keep,
    secondary_world=resolved,
    body_poses=solved_poses,
    secondary_moves=moves,
  )


def solve_joint(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  args: Args,
  cross_obs: list[CrossSighting],
) -> ReferenceSolve:
  """Solve the joint graph, with the first enabled tag pinning the world, and
  return the solution to export."""
  ref_anchor = align_to_shared_world(trackers, args.anchor.enabled_tag_ids)
  return _solve_joint_graph(trackers, keys, args, cross_obs, ref_anchor)


def adopt_ba_refined_estimates(
  trackers: list[PreparedTracker], result: OptimizeResult, keys: TrackerKeys
) -> None:
  """Overwrite each tracker's PnP joint-angle seeds and base-tag pose with the
  BA-optimized values. A finger carries a θ variable only at the keyframes where
  it was seen or inferred from the symmetry prior."""
  for t in trackers:
    if t.finger_reproj:
      for finger_idx, joint_name in enumerate(FINGER_JOINTS):
        angles = list(t.joint_angles.get(joint_name, [float("nan")] * t.n_keyframes))
        for kf in range(t.n_keyframes):
          tkey = keys.theta_key(t.name, finger_idx, kf)
          if result.values.exists(tkey):
            angles[kf] = float(result.values.atDouble(tkey))
        t.joint_angles[joint_name] = angles
    bt_key = keys.base_tag_key(t.name)
    if result.values.exists(bt_key):
      t.body_T_base_tag = result.values.atPose3(bt_key)
