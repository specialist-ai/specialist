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
  # Max metric depth (m) for kept landmarks in the final (scaled) triangulation;
  # culls far lab/ceiling structure. Raise to keep more distant geometry.
  max_landmark_depth: float = 5.0
  mask_rear_rects: tuple[tuple[int, int, int, int], ...] = ((485, 950, 1435, 1920),)
  init_min_shared_tracks: int = 60
  # Loop closure: re-match ORB features between spatially-close, non-adjacent
  # keyframes (the fingertip writing revisits the workspace) to recover tracks
  # the fast lifts break, adding wide-baseline landmarks that tie the revisits
  # together in the bundle and cancel the drift accumulated across each lift.
  loop_closure: bool = True
  loop_closure_max_dist: float = 0.1
  loop_closure_min_kf_gap: int = 15
  # Also run loop closure in anchor-seeded mode (it otherwise only runs under
  # visual odometry). When the anchor leaves view, the trajectory tail dead-
  # reckons on the IMU and drifts; if that tail revisits earlier anchored poses
  # the loop-closure landmarks tie it back to the well-localized trajectory
  # without needing the anchor.
  anchored_loop_closure: bool = True
  # Use incremental visual odometry (front-lens SfM) for the trajectory shape
  # instead of the two-keyframe straight-line interpolation, and let the anchor
  # enter only as a joint-graph factor (no per-keyframe PnP pose overwrite).
  # Triangulates both lenses against the vision trajectory so the near-field
  # rear structure survives to constrain scale.
  use_visual_odometry: bool = True
  # Cap on triangulated landmarks per lens in visual-odometry mode (keeps the
  # longest-track, best-constrained landmarks). The dense far-field front
  # structure is redundant; capping keeps the joint BA tractable.
  vo_max_landmarks_per_lens: int = 400
  # Cap on reprojection observations per track fed to the joint BA in
  # visual-odometry mode (endpoints + evenly-spaced middle, preserving the
  # track's baseline). Long VO tracks otherwise add ~15 factors each; capping
  # the factor count keeps BA runtime near the anchor-snapped baseline.
  vo_max_obs_per_track: int = 8
  # DIS dense flow (default) vs gyro-seeded pyramidal KLT. KLT consumes the
  # IMU rotation predictor (DIS ignores it); flip to False to stress-test the
  # gyro-aided frontend through rotation.
  use_dis_flow: bool = True
  # When the anchor blacks out, the default pipeline drops every landmark not
  # co-visible with an anchor keyframe, so vision contributes nothing through
  # the gap (pure IMU dead-reckon). Set True to keep gap landmarks
  # (triangulated from anchor-covisible views) and let the joint BA refine
  # poses and structure together through the blackout.
  keep_gap_landmarks: bool = False
  # Restrict keep_gap to the front (world-facing) lens. The rear lens mostly
  # images the gripper/hand, so its gap landmarks are unreliable.
  keep_gap_front_only: bool = False
  # Parallax gate (deg) on the anchor-covisible view subset used to triangulate
  # gap landmarks; 0 disables. Drops short-baseline gap landmarks.
  gap_min_parallax_deg: float = 0.5


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
  # Per-frame single-tag PnP is only ~cm-accurate at ~1 m (and laterally weak),
  # so the anchor enters as a soft frame/scale reference, not a hard per-frame
  # pose; visual odometry owns the trajectory shape. A tight σ here forces the
  # inflated PnP onto the trajectory (the session_03 2x over-scale).
  translation_sigma: float = 0.02
  rotation_sigma: float = 0.05
  prior_translation_sigma: float = 1.0e-4
  prior_rotation_sigma: float = 1.0e-4
  # σ multiplier = (1 + max(0, r − pivot) / scale)² for tag image radius r.
  sigma_radius_pivot: float = 0.5
  sigma_radius_scale: float = 0.1
  # Huber threshold; caps single-frame PnP outliers. None disables.
  huber_k: float | None = 1.5
  # Ablation bench: drop every anchor observation whose keyframe time falls
  # in [blackout_start, blackout_end] (seconds, clip clock). Both None = no
  # blackout. Lets the full-anchor run serve as pseudo-GT while a candidate
  # run flies blind through the window on VI alone.
  blackout_start: float | None = None
  blackout_end: float | None = None
  # Keep only every Nth anchor observation (1 = all). Simulates sparser anchor
  # placement to study how dense anchors must be for reliable tracking.
  keep_every: int = 1


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
