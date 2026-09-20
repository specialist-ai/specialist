"""Nested tyro config dataclasses for ``estimate.py`` and ``prepare_tracker``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from specialist.vi_ba.urdf_joints import RevoluteJoint

__all__ = [
  "AnchorCfg",
  "Args",
  "FrontendCfg",
  "GpuMode",
  "GripperCfg",
  "SolverCfg",
  "TrackerArgs",
  "validate",
]

GpuMode = Literal["require", "auto", "off"]


@dataclass
class FrontendCfg:
  orb_n_features: int = 4000
  # Run DIS optical flow on the GPU, falling back loudly to cv2 without torch/CUDA.
  gpu_flow: bool = True

  ransac_threshold: float = 4.5
  # Fraction of the frame the dense flow is computed at.
  flow_scale: float = 0.25

  min_track_length: int = 3
  # Bearing half-angle (rad) past which the essential-matrix RANSAC drops a match.
  feature_max_half_angle: float = float(np.deg2rad(70.0))

  reproj_sigma: float = 0.5

  min_parallax: float = float(np.deg2rad(0.5))
  # Max metric depth (m) for kept landmarks in the final (scaled) triangulation.
  max_landmark_depth: float = 5.0

  # Re-match ORB features between nearby non-adjacent keyframes to rejoin broken tracks.
  loop_closure: bool = True
  # Keyframes a landmark's observations may span; 0 is unbounded.
  loop_closure_window: int = 300

  # Triangulate the revisit loop landmarks from the anchor-covisible views alone.
  anchored_loop_closure: bool = True
  # Cap on triangulated landmarks, keeping the longest tracks.
  max_landmarks: int = 4000
  # Cap on observations per track in the joint BA: endpoints plus an even middle.
  max_obs_per_track: int = 8
  # Per-thread AprilTag detectors; off, one detector serves every walk in turn.
  parallel_tag_detect: bool = True


@dataclass
class AnchorCfg:
  # Static wall-tag ids, each a solved pose; the first names the world frame.
  tag_ids: tuple[int, ...]

  tag_size: float = 0.1

  # On-axis corner-reprojection noise floor (px), widened with corner radius.
  corner_reproj_sigma: float = 1.05

  @property
  def enabled_tag_ids(self) -> tuple[int, ...]:
    """``tag_ids`` with the negative (disabled) sentinel filtered out."""
    return tuple(int(t) for t in self.tag_ids if int(t) >= 0)


@dataclass
class GripperCfg:
  tag_size: float = 0.01
  keypoint_offset: tuple[float, float, float] = (0.1095, 0.0, -0.018)


@dataclass
class SolverCfg:
  max_iterations: int = 30
  velocity_prior_sigma: float = 0.1

  # Schur/cuDSS: "require" aborts without it, "auto" falls back to CPU, "off" skips it.
  gpu: GpuMode = "require"


@dataclass
class TrackerArgs:
  """One tracker's ``prepare_tracker`` configuration."""

  base_rrd: Path
  tracker: str  # this tracker's entity name inside the base recording
  anchor: AnchorCfg
  keyframe_every: int  # 0 = auto (round(fps/10))
  joints: dict[str, RevoluteJoint]
  frontend: FrontendCfg
  gripper: GripperCfg
  # Where the lenses' encoded samples stage; unset, a temporary directory.
  stage: Path | None = None


@dataclass
class Args:
  """One run; the base recording carries the session id, window and tracker names."""

  base: Path
  anchor: AnchorCfg
  # Keyframe stride in frames, a multiple of the ~0.1 s lattice; 0 picks the lattice.
  keyframe_every: int = 0

  output_dir: Path = Path("logs")
  urdf_path: Path | None = None
  # Replace this session's existing estimate and viz layers; off refuses.
  overwrite: bool = False
  # Where the lenses' encoded samples stage; unset, under output_dir.
  tmpdir: Path | None = None

  frontend: FrontendCfg = field(default_factory=FrontendCfg)
  gripper: GripperCfg = field(default_factory=GripperCfg)
  solver: SolverCfg = field(default_factory=SolverCfg)


def validate(args: Args) -> None:
  """Raise ``ValueError`` when the run configuration breaks an invariant."""
  if args.solver.max_iterations < 1:
    raise ValueError(
      f"--solver.max-iterations must be positive, got {args.solver.max_iterations}"
    )
  if not args.anchor.enabled_tag_ids:
    raise ValueError(
      "--anchor.tag-ids must name at least one enabled anchor id; the solve "
      f"reads {list(args.anchor.enabled_tag_ids)} from {list(args.anchor.tag_ids)}"
    )
  if len(set(args.anchor.tag_ids)) != len(args.anchor.tag_ids):
    raise ValueError(
      f"--anchor.tag-ids must be distinct, got {list(args.anchor.tag_ids)}"
    )
