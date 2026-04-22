"""Per-tracker GTSAM symbol-key allocation for multi-tracker VI-BA."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import gtsam

__all__ = ["TrackerKeys"]

_ANCHOR_CHAR = "A"
_POSE_CHAR = "p"
_VEL_CHAR = "v"
_BIAS_CHAR = "b"
_LANDMARK_CHAR = "l"

_MAX_KEYFRAMES_PER_TRACKER = 100_000
_MAX_LANDMARKS_PER_TRACKER = 1_000_000


class TrackerKeys:
  """Collision-free GTSAM keys per tracker; the anchor lives at ``symbol('A', 0)``."""

  def __init__(
    self,
    tracker_names: Sequence[str],
  ) -> None:
    if len(set(tracker_names)) != len(tracker_names):
      raise ValueError(f"tracker names must be unique: {list(tracker_names)}")
    self._names: list[str] = list(tracker_names)
    self._kf_step = int(_MAX_KEYFRAMES_PER_TRACKER)
    self._lm_step = int(_MAX_LANDMARKS_PER_TRACKER)

  def tracker_index(self, name: str) -> int:
    if name not in self._names:
      raise KeyError(f"unknown tracker {name!r}")
    return self._names.index(name)

  def _kf_key(self, char: str, name: str, keyframe_idx: int) -> int:
    """Collision-free key for a per-keyframe variable in one tracker's slice."""
    return int(
      gtsam.symbol(char, self.tracker_index(name) * self._kf_step + keyframe_idx)
    )

  def pose_key(self, name: str, keyframe_idx: int) -> int:
    return self._kf_key(_POSE_CHAR, name, keyframe_idx)

  def velocity_key(self, name: str, keyframe_idx: int) -> int:
    return self._kf_key(_VEL_CHAR, name, keyframe_idx)

  def bias_key(self, name: str, keyframe_idx: int) -> int:
    return self._kf_key(_BIAS_CHAR, name, keyframe_idx)

  def _lm_key(self, name: str, half: int, track_id: int) -> int:
    """A landmark key in a tracker's front (``half=0``) or rear (``half=1``) slot."""
    return int(
      gtsam.symbol(
        _LANDMARK_CHAR,
        (self.tracker_index(name) * 2 + half) * self._lm_step + track_id,
      )
    )

  def front_landmark_key(self, name: str, track_id: int) -> int:
    return self._lm_key(name, 0, track_id)

  def rear_landmark_key(self, name: str, track_id: int) -> int:
    return self._lm_key(name, 1, track_id)

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
