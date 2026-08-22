"""Nested config dataclasses for ``estimate.py`` and ``prepare_tracker``.

Tyro picks up nested ``@dataclass`` fields and exposes them as dotted CLI
flags (e.g. ``--frontend.orb-n-features 500``).
"""

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
  # Compute the DIS optical flow on the GPU; falls back loudly to the cv2 path
  # when torch/CUDA is unavailable.
  gpu_flow: bool = True

  ransac_threshold: float = 4.5
  # Fraction of the frame the dense flow is computed at.
  flow_scale: float = 0.25

  min_track_length: int = 3
  # Drop feature correspondences past this bearing half-angle (rad) from the
  # essential-matrix RANSAC.
  feature_max_half_angle: float = float(np.deg2rad(70.0))

  reproj_sigma: float = 0.5
  # Padding (px) around a masked tag's detected box; the tag overlay draws the
  # padded boxes, none when masking is off.
  mask_tag_pad: int = 45
  # Exclude non-anchor tags (the grippers' own plates) from feature detection.
  # The essential-matrix RANSAC and the parallax filter also reject features
  # that move with the gripper, so this is belt-and-braces.
  mask_tags: bool = True

  min_parallax: float = float(np.deg2rad(0.5))
  # Max metric depth (m) for kept landmarks in the final (scaled) triangulation.
  max_landmark_depth: float = 5.0

  mask_rear_rects: tuple[tuple[int, int, int, int], ...] = ((728, 1425, 2153, 2880),)
  init_min_shared_tracks: int = 30
  vo_min_registered_ratio: float = 0.7
  # Run the rear lens's landmark frontend (flow, features, triangulation). Its
  # tags are always detected; this only governs the visual landmarks, which the
  # gripper body and the static mask cover much of.
  rear_landmarks: bool = False
  # Re-match ORB features between spatially-close, non-adjacent keyframes to
  # recover tracks the intervening motion breaks.
  loop_closure: bool = True

  # Triangulate the revisit loop landmarks from the anchor-covisible views alone.
  anchored_loop_closure: bool = True
  # Cap on triangulated landmarks per lens (keeps the longest-track,
  # best-constrained landmarks). The dense far-field front structure is
  # redundant; the cap bounds the joint BA size.
  max_landmarks_per_lens: int = 4000
  # Cap on reprojection observations per track fed to the joint BA (endpoints +
  # evenly-spaced middle).
  max_obs_per_track: int = 8
  # Per-thread AprilTag detectors so concurrent tag walks overlap.
  parallel_tag_detect: bool = True


@dataclass
class AnchorCfg:
  # Static wall-tag ids. Every one is a pose the bundle solves for; the first
  # additionally names the world frame.
  tag_ids: tuple[int, ...]

  tag_size: float = 0.1

  # The anchor enters the bundle by reprojecting the tag's four known-metric
  # corners (a fixed-spacing calibration target), which pins absolute scale.
  corner_reproj_sigma: float = 1.5
  # Drop anchor sightings past this bearing half-angle (rad); the fisheye
  # calibration is least constrained at the extreme edge.
  max_half_angle: float = float(np.deg2rad(75.0))

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

  # Schur/cuDSS solver mode. "require" aborts when the backend cannot load;
  # "auto" falls back to the CPU optimizer with a notice; "off" never tries it.
  gpu: GpuMode = "require"


@dataclass
class TrackerArgs:
  """Per-tracker prep knobs. Composes the nested configs above."""

  base_rrd: Path  # the session's base recording
  tracker: str  # this tracker's entity name inside the base recording
  anchor: AnchorCfg
  keyframe_every: int  # 0 = auto (round(fps/10))
  joints: dict[str, RevoluteJoint]
  frontend: FrontendCfg
  gripper: GripperCfg
  # True when sibling captures prepare concurrently; gates per-thread AprilTag
  # detectors.
  parallel_siblings: bool = False


@dataclass
class Args:
  """One run's full configuration: the base recording, the keyframe stride, the
  anchor tag ids, and the nested per-stage knobs above. The recording carries
  the session window, the tracker names, and the session id."""

  base: Path
  anchor: AnchorCfg
  # 0 = auto: a ~0.10s keyframe interval via round(fps/10). Set positive to
  # override.
  keyframe_every: int = 0

  output_dir: Path = Path("logs")
  urdf_path: Path | None = None
  open_viewer: bool = True
  # Replace an existing track for this session; off refuses.
  overwrite: bool = False

  frontend: FrontendCfg = field(default_factory=FrontendCfg)
  gripper: GripperCfg = field(default_factory=GripperCfg)
  solver: SolverCfg = field(default_factory=SolverCfg)


def validate(args: Args) -> None:
  """Every invariant the run configuration must satisfy before any prep starts.
  ``validate_discovered`` asserts the discovered gripper tag ids."""
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
