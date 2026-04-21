"""Assemble the full NonlinearFactorGraph and initial Values from measurements.

Given per-tracker keyframe initial poses, odometry measurements, tag
observations, the scene geometry and the noise config, ``build_graph``
produces a graph containing:

- One ``PriorFactorPose3`` on the anchor tag (gauge fix).
- One ``BetweenFactorPose3`` per odometry measurement.
- One ``BetweenFactorPose3`` per tag observation: ``make_anchor_observation_factor``
  for anchor sightings, ``make_cross_tracker_factor`` for observations of
  another tracker's tag.

``dead_reckon`` is a helper for producing initialization poses by composing
odometry from identity; useful when ground-truth initialization is not
available (the realistic real-data case).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import gtsam

from specialist.config import NoiseConfig
from specialist.graph.factors import (
  make_anchor_observation_factor,
  make_anchor_prior,
  make_cross_tracker_factor,
  make_odometry_factor,
)
from specialist.graph.keys import KeyMap
from specialist.synthesis.odometry import OdometryMeasurement
from specialist.synthesis.scene import Scene
from specialist.synthesis.tag_detection import TagObservation


def dead_reckon(
  odometry: Sequence[OdometryMeasurement],
  tracker_name: str,
  n_keyframes: int,
  initial_pose: gtsam.Pose3 | None = None,
) -> list[gtsam.Pose3]:
  """Integrate odometry from ``initial_pose`` (default identity) for one tracker."""
  tracker_meas = [m for m in odometry if m.tracker_name == tracker_name]
  tracker_meas.sort(key=lambda m: m.from_index)
  poses: list[gtsam.Pose3] = [gtsam.Pose3() if initial_pose is None else initial_pose]
  for m in tracker_meas:
    if m.from_index != len(poses) - 1:
      raise ValueError(
        f"odometry for {tracker_name!r} not contiguous at index {m.from_index}"
      )
    poses.append(poses[-1].compose(m.relative_pose))
  if len(poses) != n_keyframes:
    raise ValueError(
      f"dead-reckoned {len(poses)} poses for {tracker_name!r}, expected {n_keyframes}"
    )
  return poses


def build_graph(
  initial_keyframes_per_tracker: Mapping[str, Sequence[gtsam.Pose3]],
  odometry: Sequence[OdometryMeasurement],
  tag_observations: Sequence[TagObservation],
  scene: Scene,
  noise: NoiseConfig,
  key_map: KeyMap,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """Assemble the full factor graph plus an initial ``Values`` ready for LM."""
  graph = gtsam.NonlinearFactorGraph()
  values = gtsam.Values()

  for tracker, poses in initial_keyframes_per_tracker.items():
    for i, pose in enumerate(poses):
      values.insert(key_map.tracker_pose_key(tracker, i), pose)

  values.insert(key_map.anchor_key(), scene.anchor_pose)
  graph.add(
    make_anchor_prior(
      key_map.anchor_key(),
      scene.anchor_pose,
      noise.anchor_prior_translation,
      noise.anchor_prior_rotation,
    )
  )

  for m in odometry:
    graph.add(
      make_odometry_factor(
        key_map.tracker_pose_key(m.tracker_name, m.from_index),
        key_map.tracker_pose_key(m.tracker_name, m.to_index),
        m.relative_pose,
        noise.odometry_translation,
        noise.odometry_rotation,
      )
    )

  for obs in tag_observations:
    observer_key = key_map.tracker_pose_key(obs.observer_tracker, obs.keyframe_index)
    camera_extrinsic = (
      scene.camera_front_on_tracker
      if obs.fisheye_side == "front"
      else scene.camera_back_on_tracker
    )
    if obs.tag_kind == "anchor":
      graph.add(
        make_anchor_observation_factor(
          observer_key,
          key_map.anchor_key(),
          obs.camera_to_tag,
          camera_extrinsic,
          noise.anchor_translation,
          noise.anchor_rotation,
        )
      )
      continue

    if obs.tag_owner_tracker is None:
      raise ValueError(f"tracker-kind tag observation missing owner: {obs.tag_id!r}")
    owner_key = key_map.tracker_pose_key(obs.tag_owner_tracker, obs.keyframe_index)
    graph.add(
      make_cross_tracker_factor(
        observer_key,
        owner_key,
        obs.camera_to_tag,
        camera_extrinsic,
        scene.tag_on_tracker,
        noise.cross_tracker_translation,
        noise.cross_tracker_rotation,
      )
    )

  return graph, values
