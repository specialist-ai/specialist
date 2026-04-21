"""Symbol shorthand for the factor graph.

Each tracker is assigned one lowercase letter on a first-come first-served
basis (``a``, ``b``, ``c`` ...). A pose variable at keyframe ``idx`` of
tracker ``name`` is ``symbol(char, idx)``; the anchor tag uses the
uppercase ``A`` symbol so it cannot collide with any tracker.

Lowercase for trackers and uppercase for the anchor follows GTSAM's own
examples and keeps the keys short and human-readable when dumped.
"""

from __future__ import annotations

from collections.abc import Sequence

import gtsam

_MAX_TRACKERS = 26
_ANCHOR_CHAR = "A"


class KeyMap:
  """Assigns GTSAM symbol characters to tracker names on construction."""

  def __init__(self, tracker_names: Sequence[str]) -> None:
    if len(tracker_names) > _MAX_TRACKERS:
      raise ValueError(
        f"at most {_MAX_TRACKERS} trackers supported, got {len(tracker_names)}"
      )
    if len(set(tracker_names)) != len(tracker_names):
      raise ValueError(f"tracker names must be unique: {tracker_names}")
    self._name_to_char: dict[str, str] = {
      name: chr(ord("a") + i) for i, name in enumerate(tracker_names)
    }

  def tracker_char(self, name: str) -> str:
    """Return the single-character symbol assigned to ``name``."""
    if name not in self._name_to_char:
      raise KeyError(f"unknown tracker {name!r}")
    return self._name_to_char[name]

  def tracker_pose_key(self, name: str, idx: int) -> int:
    """GTSAM symbol key for ``name``'s pose at keyframe ``idx``."""
    return int(gtsam.symbol(self.tracker_char(name), idx))

  def anchor_key(self) -> int:
    """GTSAM symbol key for the anchor tag pose variable."""
    return int(gtsam.symbol(_ANCHOR_CHAR, 0))
