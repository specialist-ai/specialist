"""Nested config dataclasses for ``track.py`` and ``prepare_tracker``.

Tyro picks up nested ``@dataclass`` fields and exposes them as dotted CLI
flags (e.g. ``--frontend.orb-n-features 500``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrontendCfg:
  orb_n_features: int = 1000
  ransac_threshold_px: float = 3.0
  min_track_length: int = 3
  max_half_angle_deg: float = 70.0
  noise_sigma_px: float = 1.5
  mask_tag_pad_px: int = 30
  min_parallax_deg: float = 0.5
  mask_rear_rects: tuple[tuple[int, int, int, int], ...] = ((485, 950, 1435, 1920),)
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
  tag_id: int = 0
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
class LmCfg:
  max_iterations: int = 30
  velocity_prior_sigma: float = 0.1


@dataclass
class WindowCfg:
  master_start: float = 0.0
  duration: float | None = 6.0
  # 0 = auto: a ~0.10s keyframe interval via round(fps/10) (6 at 60fps, 3 at
  # 30fps) so a sub-60fps clip is not undersampled. Set positive to override.
  keyframe_every: int = 0
