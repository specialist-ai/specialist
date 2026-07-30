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
  # Compute the DIS optical flow on the GPU; falls back loudly to the cv2 path
  # when torch/CUDA is unavailable.
  gpu_flow: bool = True
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


@dataclass
class ImuCfg:
  # Noise sigmas at ~5-10x the measured noise floor (Allan deviation on a
  # stationary capture: gyro ~0.0025 rad/s/sqrtHz, accel ~0.001 m/s2/sqrtHz).
  # Held a few x above physical so the accelerometer's residual systematic error
  # (timing / extrinsic) is not over-weighted, which would distort the metric
  # scale it provides.
  gyro_noise_sigma: float = 0.0125
  accel_noise_sigma: float = 0.01
  gyro_random_walk: float = 1.0e-7
  accel_random_walk: float = 1.0e-7
  imu_to_body: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
  )


@dataclass
class AnchorCfg:
  # Static wall-tag ids; the first is the reference anchor that pins the
  # world gauge. Additional ids ground trackers that co-sight them.
  tag_ids: tuple[int, ...]
  tag_size: float = 0.1

  @property
  def enabled_tag_ids(self) -> tuple[int, ...]:
    """``tag_ids`` with the negative (disabled) sentinel filtered out."""
    return tuple(int(t) for t in self.tag_ids if int(t) >= 0)

  # The anchor enters the bundle by reprojecting the tag's four known-metric
  # corners (a fixed-spacing calibration target), which pins absolute scale.
  corner_reproj_sigma: float = 1.5


@dataclass
class GripperCfg:
  tag_size: float = 0.01
  keypoint_offset: tuple[float, float, float] = (0.1095, 0.0, -0.018)
  # Finger joint-angle recovery: per-corner reprojection σ (px) and the soft
  # equal-and-opposite symmetry prior σ (rad) on θ_left + θ_right. The tag pose
  # in the clamp slot comes from the URDF tag link (urdf_joints.joint_T_tag0).
  finger_reproj_sigma: float = 1.5
  finger_symmetry_sigma: float = 0.05
  # The base-tag pose in the body frame is a BA variable, pinned by reprojecting
  # the base tag's corners (σ px) and by a loose prior seeded from PnP (rotation
  # rad / translation m) that anchors its gauge.
  base_tag_reproj_sigma: float = 1.5
  base_tag_prior_sigma_rot: float = 0.3
  base_tag_prior_sigma_trans: float = 0.05


@dataclass
class SolverCfg:
  max_iterations: int = 30
  velocity_prior_sigma: float = 0.1
  # Prefer the Schur/cuDSS GPU solver; falls back to the gtsam optimizer.
  gpu: bool = True


@dataclass
class WindowCfg:
  # Seconds of clip to process; None = the full capture.
  duration: float | None
  master_start: float = 0.0
  # 0 = auto: a ~0.10s keyframe interval via round(fps/10). Set positive to
  # override.
  keyframe_every: int = 0
