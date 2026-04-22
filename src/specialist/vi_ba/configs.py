"""Nested config dataclasses for ``track.py`` and ``prepare_tracker``.

Tyro picks up nested ``@dataclass`` fields and exposes them as dotted CLI
flags (e.g. ``--frontend.orb-n-features 500``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrontendCfg:
  orb_n_features: int = 1000
  ransac_threshold: float = 4.5
  min_track_length: int = 3
  max_half_angle: float = float(np.deg2rad(70.0))
  reproj_sigma: float = 2.25
  mask_tag_pad: int = 45
  min_parallax: float = float(np.deg2rad(0.5))
  # Max metric depth (m) for kept landmarks in the final (scaled) triangulation.
  max_landmark_depth: float = 5.0
  mask_rear_rects: tuple[tuple[int, int, int, int], ...] = ((728, 1425, 2153, 2880),)
  init_min_shared_tracks: int = 60


@dataclass
class ImuCfg:
  gyro_noise_sigma: float = 0.1
  accel_noise_sigma: float = 1.0
  gyro_random_walk: float = 1.0e-7
  accel_random_walk: float = 1.0e-7
  imu_to_body: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
  )


@dataclass
class AnchorCfg:
  tag_id: int
  tag_size: float = 0.1
  translation_sigma: float = 0.002
  rotation_sigma: float = 0.005
  prior_translation_sigma: float = 1.0e-4
  prior_rotation_sigma: float = 1.0e-4
  # σ multiplier = (1 + max(0, r − pivot) / scale)² for tag image radius r.
  sigma_radius_pivot: float = 0.5
  sigma_radius_scale: float = 0.1
  # Huber threshold; caps single-frame PnP outliers. None disables.
  huber_k: float | None = 1.5


@dataclass
class GripperCfg:
  tag_size: float = 0.01
  keypoint_offset: tuple[float, float, float] = (0.1095, 0.0, -0.018)


@dataclass
class CrossTrackerCfg:
  translation_sigma: float = 0.005
  rotation_sigma: float = 0.0175


@dataclass
class SolverCfg:
  max_iterations: int = 30
  velocity_prior_sigma: float = 0.1


@dataclass
class WindowCfg:
  # Seconds of clip to process; None = the full capture.
  duration: float | None
  master_start: float = 0.0
  # 0 = auto: a ~0.10s keyframe interval via round(fps/10). Set positive to
  # override.
  keyframe_every: int = 0
