"""End-to-end scenario runner: synthesize, build, optimize.

Ties together ``synthesis``, ``graph``, and ``eval`` so example scripts and
integration tests share the same driver. Takes a ``ScenarioConfig`` and
returns a ``ScenarioRun`` that bundles ground-truth and estimated
trajectories alongside the graph, initial values, optimization metadata,
and the raw measurement streams for inspection or visualization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import gtsam
import numpy as np

from specialist.config import ScenarioConfig
from specialist.graph.build import build_graph, dead_reckon
from specialist.graph.keys import KeyMap
from specialist.graph.optimize import OptimizationResult, optimize_graph
from specialist.synthesis.odometry import (
  OdometryMeasurement,
  sample_keyframes,
  synthesize_odometry,
)
from specialist.synthesis.scene import Scene, build_scene
from specialist.synthesis.tag_detection import TagObservation, synthesize_tag_detections
from specialist.synthesis.trajectory import SmoothTrajectory


@dataclass(eq=False)
class ScenarioRun:
  """All state produced by one scenario run — enough for eval and visualization."""

  scenario: ScenarioConfig
  scene: Scene
  key_map: KeyMap
  n_keyframes: int
  ground_truth_keyframes: dict[str, list[gtsam.Pose3]]
  estimated_keyframes: dict[str, list[gtsam.Pose3]]
  graph: gtsam.NonlinearFactorGraph
  initial_values: gtsam.Values
  ground_truth_values: gtsam.Values
  optimization: OptimizationResult
  odometry: list[OdometryMeasurement]
  observations: list[TagObservation]
  used_observations: list[TagObservation] = field(default_factory=list)

  @property
  def anchor_observations(self) -> list[TagObservation]:
    return [o for o in self.used_observations if o.tag_kind == "anchor"]

  @property
  def cross_tracker_observations(self) -> list[TagObservation]:
    return [o for o in self.used_observations if o.tag_kind == "tracker"]


def run_scenario(
  scenario: ScenarioConfig,
  *,
  observation_filter: Callable[[TagObservation], bool] | None = None,
) -> ScenarioRun:
  """Run the full pipeline once and return the resulting ``ScenarioRun``.

  ``observation_filter`` lets callers drop specific tag observations without
  reaching into the scene config — useful for blackout and cross-tracker-
  only experiments where only the anchor visibility changes.
  """
  scene = build_scene(scenario.scene, scenario.tracker_names)
  rng = np.random.default_rng(scenario.seed)

  trajectories: dict[str, SmoothTrajectory] = {
    name: SmoothTrajectory(
      duration=scenario.duration,
      workspace_radius=scenario.workspace_radius,
      rng=rng,
      workspace_center=scenario.workspace_center_xyz,
    )
    for name in scenario.tracker_names
  }
  gt_keyframes: dict[str, list[gtsam.Pose3]] = {
    name: sample_keyframes(trajectories[name], scenario.rate)
    for name in scenario.tracker_names
  }
  n_keyframes = len(next(iter(gt_keyframes.values())))

  odometry: list[OdometryMeasurement] = []
  for name in scenario.tracker_names:
    odometry.extend(
      synthesize_odometry(
        gt_keyframes[name],
        tracker_name=name,
        translation_sigma=scenario.noise.odometry_translation,
        rotation_sigma=scenario.noise.odometry_rotation,
        rng=rng,
      )
    )

  observations = synthesize_tag_detections(
    gt_keyframes, scene, noise=scenario.noise, rng=rng
  )
  used_observations = (
    list(observations)
    if observation_filter is None
    else [o for o in observations if observation_filter(o)]
  )

  key_map = KeyMap(scenario.tracker_names)
  initial_keyframes = {
    name: dead_reckon(odometry, name, n_keyframes, initial_pose=gt_keyframes[name][0])
    for name in scenario.tracker_names
  }
  graph, initial_values = build_graph(
    initial_keyframes,
    odometry,
    used_observations,
    scene,
    scenario.noise,
    key_map,
  )

  ground_truth_values = gtsam.Values()
  for name in scenario.tracker_names:
    for i, pose in enumerate(gt_keyframes[name]):
      ground_truth_values.insert(key_map.tracker_pose_key(name, i), pose)
  ground_truth_values.insert(key_map.anchor_key(), scene.anchor_pose)

  optimization = optimize_graph(graph, initial_values)
  estimated_keyframes: dict[str, list[gtsam.Pose3]] = {
    name: [
      optimization.values.atPose3(key_map.tracker_pose_key(name, i))
      for i in range(n_keyframes)
    ]
    for name in scenario.tracker_names
  }

  return ScenarioRun(
    scenario=scenario,
    scene=scene,
    key_map=key_map,
    n_keyframes=n_keyframes,
    ground_truth_keyframes=gt_keyframes,
    estimated_keyframes=estimated_keyframes,
    graph=graph,
    initial_values=initial_values,
    ground_truth_values=ground_truth_values,
    optimization=optimization,
    odometry=odometry,
    observations=list(observations),
    used_observations=used_observations,
  )
