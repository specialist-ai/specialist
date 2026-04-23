"""Compose per-tracker VI-BA subgraphs with cross-tracker tag observations.

``TrackerKeys`` allocates collision-free symbol namespaces per
tracker. ``add_cross_tracker_factors`` adds the anchor prior + tag-
observation between-factors that couple per-tracker subgraphs into one
joint smoother. Tag observations reference ``(tracker_name, keyframe_index)``
tuples that must map to existing VI-BA keyframes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import gtsam

from specialist.config import NoiseConfig
from specialist.graph.factors import (
  make_anchor_observation_factor,
  make_anchor_prior,
  make_cross_tracker_factor,
)
from specialist.synthesis.scene import Scene
from specialist.synthesis.tag_detection import TagObservation

__all__ = ["TrackerKeys", "add_cross_tracker_factors"]

_ANCHOR_CHAR = "A"
_POSE_CHAR = "p"
_VEL_CHAR = "v"
_BIAS_CHAR = "b"
_LANDMARK_CHAR = "l"


class TrackerKeys:
  """Collision-free GTSAM symbol-key allocator for multi-tracker VI-BA.
  Each tracker gets a unique slice of the integer-key space, indexed by
  ``max_keyframes_per_tracker`` for poses/velocities/biases and
  ``max_landmarks_per_tracker`` for each lens's landmarks. Anchor lives
  at ``symbol('A', 0)``."""

  def __init__(
    self,
    tracker_names: Sequence[str],
    max_keyframes_per_tracker: int = 100_000,
    max_landmarks_per_tracker: int = 1_000_000,
  ) -> None:
    if len(set(tracker_names)) != len(tracker_names):
      raise ValueError(f"tracker names must be unique: {list(tracker_names)}")
    self._names: list[str] = list(tracker_names)
    self._kf_step = int(max_keyframes_per_tracker)
    self._lm_step = int(max_landmarks_per_tracker)

  def tracker_index(self, name: str) -> int:
    if name not in self._names:
      raise KeyError(f"unknown tracker {name!r}")
    return self._names.index(name)

  def pose_key(self, name: str, keyframe_idx: int) -> int:
    return int(
      gtsam.symbol(_POSE_CHAR, self.tracker_index(name) * self._kf_step + keyframe_idx)
    )

  def velocity_key(self, name: str, keyframe_idx: int) -> int:
    return int(
      gtsam.symbol(_VEL_CHAR, self.tracker_index(name) * self._kf_step + keyframe_idx)
    )

  def bias_key(self, name: str, keyframe_idx: int) -> int:
    return int(
      gtsam.symbol(_BIAS_CHAR, self.tracker_index(name) * self._kf_step + keyframe_idx)
    )

  def front_landmark_key(self, name: str, landmark_idx: int) -> int:
    return int(
      gtsam.symbol(
        _LANDMARK_CHAR, (self.tracker_index(name) * 2) * self._lm_step + landmark_idx
      )
    )

  def rear_landmark_key(self, name: str, landmark_idx: int) -> int:
    return int(
      gtsam.symbol(
        _LANDMARK_CHAR,
        (self.tracker_index(name) * 2 + 1) * self._lm_step + landmark_idx,
      )
    )

  def anchor_key(self) -> int:
    return int(gtsam.symbol(_ANCHOR_CHAR, 0))

  def vi_graph_kwargs(self, name: str) -> dict[str, Callable[[int], int]]:
    """Per-tracker key callables to pass to ``build_vi_graph``."""
    return {
      "pose_key_fn": lambda i: self.pose_key(name, i),
      "velocity_key_fn": lambda i: self.velocity_key(name, i),
      "bias_key_fn": lambda i: self.bias_key(name, i),
      "front_landmark_key_fn": lambda i: self.front_landmark_key(name, i),
      "rear_landmark_key_fn": lambda i: self.rear_landmark_key(name, i),
    }


def add_cross_tracker_factors(
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
  keys: TrackerKeys,
  observations: Sequence[TagObservation],
  scene: Scene,
  noise: NoiseConfig,
) -> None:
  """Add anchor prior + cross-tracker tag observation factors in place.
  Each observation references a ``(tracker, keyframe)`` pose that must
  already exist in ``values``."""
  anchor_key = keys.anchor_key()
  if not values.exists(anchor_key):
    values.insert(anchor_key, scene.anchor_pose)
    graph.add(
      make_anchor_prior(
        anchor_key,
        scene.anchor_pose,
        noise.anchor_prior_translation,
        noise.anchor_prior_rotation,
      )
    )

  for obs in observations:
    observer = keys.pose_key(obs.observer_tracker, obs.keyframe_index)
    if not values.exists(observer):
      raise KeyError(
        f"observation references missing pose: "
        f"{obs.observer_tracker!r}@{obs.keyframe_index} "
        f"(call build_vi_graph for that tracker first)"
      )
    camera_extrinsic = (
      scene.camera_front_on_tracker
      if obs.fisheye_side == "front"
      else scene.camera_back_on_tracker
    )
    if obs.tag_kind == "anchor":
      graph.add(
        make_anchor_observation_factor(
          observer,
          anchor_key,
          obs.camera_to_tag,
          camera_extrinsic,
          noise.anchor_translation,
          noise.anchor_rotation,
        )
      )
      continue

    if obs.tag_owner_tracker is None:
      raise ValueError(f"tracker-kind tag observation missing owner: {obs.tag_id!r}")
    owner = keys.pose_key(obs.tag_owner_tracker, obs.keyframe_index)
    if not values.exists(owner):
      raise KeyError(
        f"observation references missing owner pose: "
        f"{obs.tag_owner_tracker!r}@{obs.keyframe_index}"
      )
    graph.add(
      make_cross_tracker_factor(
        observer,
        owner,
        obs.camera_to_tag,
        camera_extrinsic,
        scene.tag_on_tracker,
        noise.cross_tracker_translation,
        noise.cross_tracker_rotation,
      )
    )
