"""Per-tracker GTSAM symbol-key allocation for multi-tracker VI-BA.

``TrackerKeys`` allocates a collision-free symbol namespace per tracker.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import gtsam

__all__ = ["TrackerKeys"]

_BASE_TAG_CHAR = "G"
_POSE_CHAR = "p"
_VEL_CHAR = "v"
_BIAS_CHAR = "b"
_LANDMARK_CHAR = "l"
_ANGLE_CHAR = "a"

_MAX_KEYFRAMES_PER_TRACKER = 100_000
_MAX_LANDMARKS_PER_TRACKER = 1_000_000


class TrackerKeys:
  """Each tracker gets a unique slice of the integer-key space, indexed by
  ``_MAX_KEYFRAMES_PER_TRACKER`` for poses/velocities/biases and
  ``_MAX_LANDMARKS_PER_TRACKER`` for each lens's landmarks. Anchor tag corners
  are constants baked into unary factors, not variables, so they hold no
  keys."""

  def __init__(
    self,
    tracker_names: Sequence[str],
  ) -> None:
    if len(set(tracker_names)) != len(tracker_names):
      raise ValueError(f"tracker names must be unique: {list(tracker_names)}")
    self._names: list[str] = list(tracker_names)

  def tracker_index(self, name: str) -> int:
    if name not in self._names:
      raise KeyError(f"unknown tracker {name!r}")
    return self._names.index(name)

  def _kf_key(self, char: str, name: str, keyframe_idx: int) -> int:
    """Collision-free key for a per-keyframe variable in one tracker's slice."""
    return int(
      gtsam.symbol(
        char, self.tracker_index(name) * _MAX_KEYFRAMES_PER_TRACKER + keyframe_idx
      )
    )

  def pose_key(self, name: str, keyframe_idx: int) -> int:
    return self._kf_key(_POSE_CHAR, name, keyframe_idx)

  def velocity_key(self, name: str, keyframe_idx: int) -> int:
    return self._kf_key(_VEL_CHAR, name, keyframe_idx)

  def bias_key(self, name: str, keyframe_idx: int) -> int:
    return self._kf_key(_BIAS_CHAR, name, keyframe_idx)

  def _lm_key(self, name: str, half: int, track_id: int) -> int:
    """Collision-free key for a landmark in one tracker's front (``half=0``) or
    rear (``half=1``) lens slot."""
    return self._slot_key(
      _LANDMARK_CHAR, name, half, _MAX_LANDMARKS_PER_TRACKER, track_id
    )

  def front_landmark_key(self, name: str, track_id: int) -> int:
    return self._lm_key(name, 0, track_id)

  def rear_landmark_key(self, name: str, track_id: int) -> int:
    return self._lm_key(name, 1, track_id)

  def theta_key(self, name: str, finger_idx: int, keyframe_idx: int) -> int:
    """Key for a finger's revolute joint angle (scalar) at one keyframe.
    ``finger_idx`` 0 = left, 1 = right; collision-free per tracker and finger."""
    return self._slot_key(
      _ANGLE_CHAR, name, finger_idx, _MAX_KEYFRAMES_PER_TRACKER, keyframe_idx
    )

  def _slot_key(self, char: str, name: str, half: int, width: int, index: int) -> int:
    """One integer key in the tracker's ``half`` slice of a ``width``-wide slot."""
    return int(
      gtsam.symbol(char, (self.tracker_index(name) * 2 + half) * width + index)
    )

  def base_tag_key(self, name: str) -> int:
    """Key for a tracker's base-tag pose in its body frame (one static ``Pose3``
    shared across all keyframes; the base + finger tags reproject through it)."""
    return int(gtsam.symbol(_BASE_TAG_CHAR, self.tracker_index(name)))

  def vi_graph_kwargs(self, name: str) -> dict[str, Callable[[int], int]]:
    """Per-tracker key callables to pass to ``build_vi_graph``."""
    return {
      "pose_key_fn": lambda i: self.pose_key(name, i),
      "velocity_key_fn": lambda i: self.velocity_key(name, i),
      "bias_key_fn": lambda i: self.bias_key(name, i),
      "front_landmark_key_fn": lambda i: self.front_landmark_key(name, i),
      "rear_landmark_key_fn": lambda i: self.rear_landmark_key(name, i),
    }
