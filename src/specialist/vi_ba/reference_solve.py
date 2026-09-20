"""The joint solve with a landmark cull-reoptimize pass, and the solved anchors
and body poses read back for export."""

from __future__ import annotations

from dataclasses import dataclass, replace

import gtsam
import numpy as np

from specialist.vi_ba._mei_factor_cpp import prune_factors
from specialist.vi_ba.compose import (
  CrossSighting,
  LandmarkKeep,
  TrackerKeys,
  align_to_shared_world,
  assemble_joint_graph,
  points_from_values,
)
from specialist.vi_ba.configs import Args
from specialist.vi_ba.landmark_cull import cull_outlier_landmarks
from specialist.vi_ba.optimize import OptimizeResult, optimize_graph
from specialist.vi_ba.single_tracker import PreparedTracker
from specialist.vi_ba.urdf_joints import FINGER_JOINTS

__all__ = [
  "ReferenceSolve",
  "adopt_ba_refined_estimates",
  "solve_joint",
]

# The first solve only ranks residuals for the cull.
_PRECULL_MAX_ITERATIONS = 8


def _prune_factor_graph(
  graph: gtsam.NonlinearFactorGraph, drop_keys: set[int]
) -> gtsam.NonlinearFactorGraph:
  """A copy of ``graph`` with every factor touching a dropped variable key
  removed."""
  pruned: gtsam.NonlinearFactorGraph = prune_factors(
    graph, np.fromiter(drop_keys, dtype=np.int64, count=len(drop_keys))
  )
  return pruned


def _outlier_landmark_keys(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  landmark_keep: LandmarkKeep,
) -> set[int]:
  """The landmark keys the cull dropped, over every tracker's front lens."""
  drop: set[int] = set()
  for t in trackers:
    keep = landmark_keep[t.name]
    drop.update(
      keys.front_landmark_key(t.name, tid) for tid in t.front.points if tid not in keep
    )
  return drop


def _solve_with_cull_reoptimize(
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  max_iterations: int,
) -> tuple[OptimizeResult, LandmarkKeep]:
  """Solve the joint graph, then drop the landmarks the first solve exposes as
  outliers and re-solve warm-started. Returns the re-solved result and, per
  tracker, the landmark tracks it kept."""
  print(f"\n[graph] joint graph: {graph.size()} factors, {values.size()} variables")
  result = optimize_graph(
    graph,
    values,
    max_iterations=min(max_iterations, _PRECULL_MAX_ITERATIONS),
  )
  poses_by_tracker = _solved_body_poses(trackers, keys, result.values)
  landmark_keep: LandmarkKeep = {}
  for t in trackers:
    points = points_from_values(
      result.values, keys.front_landmark_key, t.name, t.front.points
    )
    landmark_keep[t.name] = cull_outlier_landmarks(
      points, t.front.tracks, poses_by_tracker[t.name], t.front.setup
    )
  first = result
  outlier_keys = _outlier_landmark_keys(trackers, keys, landmark_keep)
  if outlier_keys:
    pruned = _prune_factor_graph(graph, outlier_keys)
    kept_values = gtsam.Values(result.values)
    for key in outlier_keys:
      kept_values.erase(key)
    result = optimize_graph(
      pruned,
      kept_values,
      max_iterations=max_iterations,
      initial_lambda=first.final_lambda,
    )
    print(f"  [solve] cull-reoptimize dropped {len(outlier_keys)} outlier landmarks")
  elif result.iterations >= _PRECULL_MAX_ITERATIONS:
    result = optimize_graph(
      graph,
      result.values,
      max_iterations=max_iterations,
      initial_lambda=first.final_lambda,
    )
  else:
    return result, landmark_keep
  return (
    replace(
      result,
      initial_error=first.initial_error,
      iterations=first.iterations + result.iterations,
    ),
    landmark_keep,
  )


def _solved_body_poses(
  trackers: list[PreparedTracker], keys: TrackerKeys, values: gtsam.Values
) -> dict[str, list[gtsam.Pose3]]:
  """Each tracker's optimized body poses pulled out of ``values``."""
  return {
    t.name: [values.atPose3(keys.pose_key(t.name, i)) for i in range(t.n_keyframes)]
    for t in trackers
  }


def _anchor_pose_move(seed: gtsam.Pose3, estimate: gtsam.Pose3) -> tuple[float, float]:
  """Translation (m) and rotation (rad) separating ``seed`` from ``estimate``."""
  dt = float(
    np.linalg.norm(np.asarray(estimate.translation()) - np.asarray(seed.translation()))
  )
  dr = float(
    np.linalg.norm(gtsam.Rot3.Logmap(seed.rotation().between(estimate.rotation())))
  )
  return dt, dr


@dataclass
class ReferenceSolve:
  """Everything the export needs from the joint solve."""

  ref_tag_id: int
  result: OptimizeResult
  landmark_keep: LandmarkKeep
  anchor_world: dict[int, gtsam.Pose3]
  body_poses: dict[str, list[gtsam.Pose3]]
  # Per anchor, how far the solve moved it off its seed (metres, radians).
  anchor_moves: dict[int, tuple[float, float]]

  @property
  def worst_anchor_move(self) -> tuple[float, float]:
    """The largest translation and rotation any anchor moved off its seed,
    over the anchors the datum does not hold."""
    moves = [m for tag, m in self.anchor_moves.items() if tag != self.ref_tag_id]
    if not moves:
      return float("nan"), float("nan")
    return max(dt for dt, _ in moves), max(dr for _, dr in moves)


def _solve_joint_graph(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  args: Args,
  cross_obs: list[CrossSighting],
  anchor_seed: dict[int, gtsam.Pose3],
) -> ReferenceSolve:
  """Solve the joint graph and read every anchor's world pose back out of it."""
  graph, values = assemble_joint_graph(trackers, keys, args, cross_obs, anchor_seed)
  result, landmark_keep = _solve_with_cull_reoptimize(
    graph, values, trackers, keys, args.solver.max_iterations
  )

  solved: dict[int, gtsam.Pose3] = {}
  moves: dict[int, tuple[float, float]] = {}
  for tag_id, seed in anchor_seed.items():
    estimate = result.values.atPose3(keys.anchor_key(tag_id))
    dt, dr = _anchor_pose_move(seed, estimate)
    print(f"  [anchor] tag {tag_id} moved {dt * 1000:.1f}mm / {np.degrees(dr):.2f}deg")
    solved[tag_id] = estimate
    moves[tag_id] = (dt, dr)
  return ReferenceSolve(
    ref_tag_id=int(args.anchor.enabled_tag_ids[0]),
    result=result,
    landmark_keep=landmark_keep,
    anchor_world=solved,
    body_poses=_solved_body_poses(trackers, keys, result.values),
    anchor_moves=moves,
  )


def solve_joint(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  args: Args,
  cross_obs: list[CrossSighting],
) -> ReferenceSolve:
  """Ground every tracker in the shared world the anchors define, then solve the
  joint graph from that seed."""
  anchor_seed = align_to_shared_world(trackers, args.anchor.enabled_tag_ids)
  return _solve_joint_graph(trackers, keys, args, cross_obs, anchor_seed)


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
