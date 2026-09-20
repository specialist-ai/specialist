"""GTSAM key namespace shared by the graph builders and the solvers."""

from __future__ import annotations

__all__ = [
  "ANGLE_CHAR",
  "BIAS_CHAR",
  "LANDMARK_CHAR",
  "MAX_KEYFRAMES_PER_TRACKER",
  "PER_KEYFRAME_CHARS",
  "POSE_CHAR",
  "VELOCITY_CHAR",
]

LANDMARK_CHAR = "l"
POSE_CHAR = "p"
VELOCITY_CHAR = "v"
BIAS_CHAR = "b"
ANGLE_CHAR = "a"
PER_KEYFRAME_CHARS = frozenset({POSE_CHAR, VELOCITY_CHAR, BIAS_CHAR, ANGLE_CHAR})
# A per-keyframe key's index is its tracker's slot times this, plus the keyframe.
MAX_KEYFRAMES_PER_TRACKER = 100_000
