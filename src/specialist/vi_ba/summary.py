"""The run summary the track layer carries as properties."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Summary", "TrackerSummary"]


@dataclass(frozen=True)
class TrackerSummary:
  """One tracker's line in the run summary."""

  name: str
  n_keyframes: int
  anchor_obs: int
  bounding_box_diagonal: float
  path_length: float


@dataclass(frozen=True)
class Summary:
  """The run's header: what was processed, and the solve's own account of
  how it went."""

  sources: tuple[str, ...]
  tracker_names: tuple[str, ...]
  cross_tracker_obs: int
  initial_graph_error: float
  final_graph_error: float
  solver_iterations: int
  # The worst secondary anchor's post-solve re-estimate move off its seed
  # (metres, radians); NaN when the take has no secondary anchor.
  secondary_anchor_move_translation: float = float("nan")
  secondary_anchor_move_rotation: float = float("nan")
  # Every defaulted flag this run overrode, as "dotted.name=value". Two layers
  # are comparable exactly when both tuples are empty.
  non_default_flags: tuple[str, ...] = ()
