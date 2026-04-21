"""Configuration dataclasses for specialist scenarios.

``NoiseConfig`` holds per-source noise sigmas; ``SceneConfig`` holds the
rigid geometry of the tracker (tag + fisheye extrinsics) and the anchor-tag
pose. ``ScenarioConfig`` is the top-level runnable scenario consumed by
example scripts via ``tyro.cli``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Vec3 = tuple[float, float, float]


@dataclass
class NoiseConfig:
  """Per-source noise sigmas for synthetic measurements.

  The anchor prior is the gauge fix and is intentionally tight.
  """

  odometry_translation: float = 3e-3
  odometry_rotation: float = math.radians(0.1)
  anchor_translation: float = 1e-2
  anchor_rotation: float = math.radians(0.5)
  cross_tracker_translation: float = 5e-3
  cross_tracker_rotation: float = math.radians(1.0)
  anchor_prior_translation: float = 1e-6
  anchor_prior_rotation: float = 1e-6


@dataclass
class SceneConfig:
  """Rigid geometry of the scene: anchor pose and on-tracker extrinsics.

  The same ``tag_on_tracker`` extrinsic is used for every tracker; trackers
  differ only in their name and trajectory. Two fisheyes per tracker model an
  Insta360 back-to-back arrangement.
  """

  # Anchor sits at the world origin with its normal pointing up (+Z), so it
  # reads as a floor-mounted tag that the trackers look down on.
  anchor_tag_xyz: Vec3 = (0.0, 0.0, 0.0)
  anchor_tag_rpy: Vec3 = (0.0, 0.0, 0.0)

  # Tracker-carried tag normal points up (+Z in tracker body frame) so another
  # tracker looking down from above presents its camera to this tag's face.
  tag_on_tracker_xyz: Vec3 = (0.0, 0.05, 0.0)
  tag_on_tracker_rpy: Vec3 = (0.0, 0.0, 0.0)

  camera_front_xyz: Vec3 = (0.0, 0.0, 0.02)
  camera_front_rpy: Vec3 = (0.0, 0.0, 0.0)
  camera_back_xyz: Vec3 = (0.0, 0.0, -0.02)
  camera_back_rpy: Vec3 = (math.pi, 0.0, 0.0)

  # Kept under 180° so the Pinhole frustum in the Rerun viewer renders cleanly;
  # the real Insta360 fisheye is ~195° but that wraps past a plane and reads
  # as noise in a wireframe.
  fov: float = math.radians(120.0)
  min_distance: float = 0.05
  max_distance: float = 8.0

  anchor_tag_id_pattern: str = r"^anchor$"
  tracker_tag_id_pattern: str = r"^tracker_(\w+)$"


@dataclass
class ScenarioConfig:
  """Top-level scenario: trackers, trajectory timing, scene, noise, seed."""

  tracker_names: tuple[str, ...] = ("left", "right")
  duration: float = 10.0
  rate: float = 10.0
  workspace_radius: float = 0.6
  workspace_center_xyz: Vec3 = (0.0, 0.0, 1.0)
  seed: int = 0

  scene: SceneConfig = field(default_factory=SceneConfig)
  noise: NoiseConfig = field(default_factory=NoiseConfig)
