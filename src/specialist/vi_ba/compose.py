"""Per-tracker GTSAM symbol-key allocation for multi-tracker VI-BA.

``TrackerKeys`` allocates collision-free symbol namespaces per tracker so the
per-tracker subgraphs compose into one joint smoother without key collisions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import gtsam

__all__ = ["TrackerKeys"]

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
