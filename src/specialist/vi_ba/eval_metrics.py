"""Autoresearch evaluation metrics — pure-numpy library, zero gtsam.

This module is the *only* place where the autoresearch metric is
defined. It consumes the raw observations emitted by ``scripts/track.py``
(poses, landmarks, observations as numpy arrays) and produces the
metric values consumed by the keep-rule.

The module is **deletable**: nothing in ``src/`` or ``scripts/track.py``
imports from it. Only ``scripts/eval_track.py`` depends on this module.
That property is load-bearing for the autoresearch contract — the
pipeline cannot read the metric to game it.

Geometry constants used by both the BA and the metric (most notably
``_EXTREME_RADIUS_FRAC``) live as private duplicates here and in
``track.py``; the duplication is intentional. A docstring on each
duplicate names the partner constant; if one moves, update the other
by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from specialist.loader.mei_model import MeiParams, mei_project

Float64Array = npt.NDArray[np.float64]


# Quality bar for the primary ``zero_lm`` metric. Hardcoded so an
# iteration cannot loosen ``min_track_length`` / parallax / depth
# in Cfg and game the metric — these thresholds are recomputed for
# every run from scratch.
QB_MIN_TRACK_LENGTH = 3
QB_MIN_PARALLAX_DEG = 1.0
QB_MAX_REPROJ_RMSE_PX = 2.0
QB_DEPTH_RANGE_M = (0.1, 50.0)
QB_DEDUP_RADIUS_M = 0.005

# AprilTag corner-centroid radius beyond which the metric ignores the
# revisit (treats the kf as "no confident anchor" for kp segment
# classification, drops it from anc_med / anc_max). Normalized so the
# fisheye optical FOV (a circle of radius ``min(h, w) / 2`` inscribed
# in the rectangular sensor) maps to ``[0, 1]``.
#
# MUST stay in sync with ``scripts/track.py``'s ``_EXTREME_RADIUS_FRAC``
# (the BA inflates anchor σ by 7× past the same radius for the same
# physical reason — IPPE PnP residuals there are dominated by fisheye
# distortion, not by drift).
_EXTREME_RADIUS_FRAC = 0.65


# ---------------------------------------------------------------------------
# SE(3) helpers (pure numpy; mirrors gtsam.Pose3 semantics)
# ---------------------------------------------------------------------------


def compose_se3(A: Float64Array, B: Float64Array) -> Float64Array:
  """``A @ B`` for 4x4 SE(3) matrices."""
  out = np.eye(4, dtype=np.float64)
  out[:3, :3] = A[:3, :3] @ B[:3, :3]
  out[:3, 3] = A[:3, :3] @ B[:3, 3] + A[:3, 3]
  return out


def invert_se3(T: Float64Array) -> Float64Array:
  out = np.eye(4, dtype=np.float64)
  out[:3, :3] = T[:3, :3].T
  out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
  return out


def transform_world_to_local(
  T_world_x: Float64Array, p_world: Float64Array
) -> Float64Array:
  """``T_world_x.transformTo(p_world)`` — express ``p_world`` in x's
  local frame. ``T_world_x`` follows GTSAM convention (rotation maps
  x-frame vectors into world-frame)."""
  return T_world_x[:3, :3].T @ (p_world - T_world_x[:3, 3])


# ---------------------------------------------------------------------------
# Helpers used by multiple metrics
# ---------------------------------------------------------------------------


def aggregate_rmse(per_kf: list[float]) -> float:
  """RMS of the finite per-keyframe RMSE values."""
  finite = [v for v in per_kf if np.isfinite(v)]
  return float(np.sqrt(np.mean(np.square(finite)))) if finite else float("nan")


def radius_frac(pixel_center: Float64Array, frame_shape: tuple[int, ...]) -> float:
  """Fisheye-correct distance from image-center, normalized so the lens
  optical-circle edge is ``1.0``. The fisheye projects a circle of
  radius ``min(h, w) / 2`` onto the rectangular sensor; corners of the
  rectangle are outside the lens and unreachable for tag detection,
  so normalizing by ``min`` (not ``max``) is the right scale for
  "fraction from optical center to optical edge"."""
  h, w = frame_shape[:2]
  cx, cy = float(w) / 2.0, float(h) / 2.0
  return float(
    np.sqrt((pixel_center[0] - cx) ** 2 + (pixel_center[1] - cy) ** 2)
    / (min(w, h) / 2.0)
  )


# ---------------------------------------------------------------------------
# Reprojection RMSE
# ---------------------------------------------------------------------------


def per_keyframe_reproj_rmse(
  body_poses: list[Float64Array],
  body_T_lens: Float64Array,
  params: MeiParams,
  tracks: dict[int, dict[int, Float64Array]],
  landmarks: dict[int, Float64Array],
  n_frames: int,
) -> list[float]:
  """Per-kf RMSE of the reprojection residuals for every track that has
  a triangulated landmark and pose at that kf."""
  buckets: list[list[float]] = [[] for _ in range(n_frames)]
  for tid, obs in tracks.items():
    if tid not in landmarks:
      continue
    xyz = landmarks[tid]
    for fi, pt in obs.items():
      if not (0 <= fi < n_frames):
        continue
      lens_pose = compose_se3(body_poses[fi], body_T_lens)
      p_lens = transform_world_to_local(lens_pose, xyz)
      pred = mei_project(p_lens[None, :], params)[0]
      if np.all(np.isfinite(pred)):
        buckets[fi].append(float(np.linalg.norm(pred - pt)))
  return [
    float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    for errs in buckets
  ]


# ---------------------------------------------------------------------------
# Quality-bar landmark count (the primary metric driver)
# ---------------------------------------------------------------------------


def quality_landmark_counts_per_kf(
  body_poses: list[Float64Array],
  body_T_lens: Float64Array,
  params: MeiParams,
  tracks: dict[int, dict[int, Float64Array]],
  landmarks: dict[int, Float64Array],
  n_frames: int,
) -> tuple[list[int], list[list[Float64Array]]]:
  """Per-kf count of landmarks that pass the fixed quality bar.

  Returns ``(counts, world_xyz_per_kf)``. The caller dedupes across
  lenses by spatial proximity (see ``dedup_counts_by_position``)."""
  counts = [0] * n_frames
  xyz_per_kf: list[list[Float64Array]] = [[] for _ in range(n_frames)]
  depth_lo, depth_hi = QB_DEPTH_RANGE_M
  for tid, obs in tracks.items():
    if len(obs) < QB_MIN_TRACK_LENGTH:
      continue
    if tid not in landmarks:
      continue
    xyz = landmarks[tid]
    valid_kfs: list[int] = []
    bearings: list[Float64Array] = []
    sq_residuals: list[float] = []
    for fi, pt in obs.items():
      if not (0 <= fi < n_frames):
        continue
      lens_pose = compose_se3(body_poses[fi], body_T_lens)
      p_lens = transform_world_to_local(lens_pose, xyz)
      depth = float(np.linalg.norm(p_lens))
      if p_lens[2] <= 0.0 or depth < depth_lo or depth > depth_hi:
        continue
      pred = mei_project(p_lens[None, :], params)[0]
      if not np.all(np.isfinite(pred)):
        continue
      sq_residuals.append(float(np.sum((pred - pt) ** 2)))
      cam_pos = lens_pose[:3, 3]
      bearing = xyz - cam_pos
      bn = np.linalg.norm(bearing)
      if bn <= 0.0:
        continue
      bearings.append(bearing / bn)
      valid_kfs.append(fi)
    if len(valid_kfs) < QB_MIN_TRACK_LENGTH:
      continue
    rmse = float(np.sqrt(np.mean(sq_residuals)))
    if rmse >= QB_MAX_REPROJ_RMSE_PX:
      continue
    max_angle_deg = 0.0
    for i in range(len(bearings)):
      for j in range(i + 1, len(bearings)):
        cos_a = float(np.clip(np.dot(bearings[i], bearings[j]), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cos_a)))
        if ang > max_angle_deg:
          max_angle_deg = ang
    if max_angle_deg < QB_MIN_PARALLAX_DEG:
      continue
    for fi in valid_kfs:
      counts[fi] += 1
      xyz_per_kf[fi].append(xyz)
  return counts, xyz_per_kf


def dedup_counts_by_position(
  per_lens_xyz_per_kf: list[list[list[Float64Array]]],
  n_frames: int,
) -> list[int]:
  """Combine per-lens accepted-landmark world positions into a per-kf
  count with positions within ``QB_DEDUP_RADIUS_M`` collapsed to one."""
  out = [0] * n_frames
  r2 = QB_DEDUP_RADIUS_M * QB_DEDUP_RADIUS_M
  for k in range(n_frames):
    kept: list[Float64Array] = []
    for lens_xyzs in per_lens_xyz_per_kf:
      for xyz in lens_xyzs[k]:
        if any(float(np.sum((xyz - prev) ** 2)) < r2 for prev in kept):
          continue
        kept.append(xyz)
    out[k] = len(kept)
  return out


# ---------------------------------------------------------------------------
# Confident anchor visibility (kp segment classifier + anc gate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TagDetection:
  """Minimal detection record for the confidence filter."""

  tag_id: int
  corners: Float64Array  # (4, 2) image pixels


def compute_anchor_confidently_visible(
  n_frames: int,
  anchor_obs_per_kf: dict[int, list[str]],
  front_tag_detections: list[list[TagDetection]],
  rear_tag_detections: list[list[TagDetection]],
  front_frame_shape: tuple[int, ...],
  rear_frame_shape: tuple[int, ...],
  anchor_tag_id: int,
) -> list[bool]:
  """Per-kf flag: ``True`` iff the anchor tag is detected at this kf on
  at least one lens AND its corner-centroid radius is within
  ``_EXTREME_RADIUS_FRAC``. Use this — not the raw "anchor visible"
  signal — for kp segment classification and for the anc_med /
  anc_max metric. Peripheral revisits never count as ground truth."""
  out = [False] * n_frames
  for kf, sides in anchor_obs_per_kf.items():
    if not (0 <= kf < n_frames):
      continue
    for side in sides:
      detections = front_tag_detections if side == "front" else rear_tag_detections
      shape = front_frame_shape if side == "front" else rear_frame_shape
      if kf >= len(detections):
        continue
      det = next(
        (d for d in detections[kf] if int(d.tag_id) == int(anchor_tag_id)),
        None,
      )
      if det is None:
        continue
      centroid = np.asarray(det.corners, dtype=np.float64).mean(axis=0)
      if radius_frac(centroid, shape) <= _EXTREME_RADIUS_FRAC:
        out[kf] = True
        break
  return out


# ---------------------------------------------------------------------------
# Anchor-revisit drift (gap exit)
# ---------------------------------------------------------------------------


def compute_gap_drift(
  body_poses: list[Float64Array],
  ref_anchor: Float64Array,
  body_T_front_lens: Float64Array,
  body_T_rear_lens: Float64Array,
  anchor_obs_per_kf: dict[int, dict[str, Float64Array]],
  has_confident_anchor: list[bool],
  kf_times: Float64Array,
  n_frames: int,
) -> list[dict[str, object]]:
  """For each gap (a confident-anchor kf followed by ≥1 unanchored kfs
  followed by another confident-anchor kf), the L2 distance at the gap
  exit between the optimized body translation and the body translation
  implied by the raw anchor PnP alone — i.e. how far the IMU+landmark
  chain drifted while the anchor was lost."""
  drifts: list[dict[str, object]] = []
  i = 0
  while i < n_frames:
    if has_confident_anchor[i] and i + 1 < n_frames and not has_confident_anchor[i + 1]:
      pre = i
      j = i + 1
      while j < n_frames and not has_confident_anchor[j]:
        j += 1
      if j < n_frames:
        sides = anchor_obs_per_kf[j]
        if "front" in sides:
          body_T_lens = body_T_front_lens
          cam_T_tag = sides["front"]
        else:
          body_T_lens = body_T_rear_lens
          cam_T_tag = sides["rear"]
        body_T_tag = compose_se3(body_T_lens, cam_T_tag)
        anchor_pose = compose_se3(ref_anchor, invert_se3(body_T_tag))
        lm_p = body_poses[j][:3, 3]
        anc_p = anchor_pose[:3, 3]
        drift = float(np.linalg.norm(lm_p - anc_p))
        drifts.append(
          {
            "gap_kfs": (pre, j),
            "gap_t": (float(kf_times[pre]), float(kf_times[j])),
            "drift": drift,
          }
        )
      i = j
    else:
      i += 1
  return drifts


# ---------------------------------------------------------------------------
# Inter-tracker keypoint distance (taped fingertips invariant)
# ---------------------------------------------------------------------------


def keypoint_world_position_per_kf(
  body_poses: list[Float64Array],
  body_T_gripper_tag: Float64Array | None,
  keypoint_offset: Float64Array,
) -> list[Float64Array]:
  """Per-kf fingertip world position. The offset is in gripper-tag frame
  (matches viz / physical geometry), so we compose the body pose with
  ``body_T_gripper_tag`` first when it is available."""
  out: list[Float64Array] = []
  for p in body_poses:
    tag_pose = (
      compose_se3(p, body_T_gripper_tag) if body_T_gripper_tag is not None else p
    )
    R = tag_pose[:3, :3]
    t = tag_pose[:3, 3]
    out.append(t + R @ keypoint_offset)
  return out


def per_pair_keypoint_distances(
  keypoint_world_by_tracker: dict[str, list[Float64Array]],
  tracker_order: list[str],
) -> tuple[dict[str, list[float]], dict[str, dict[str, float]]]:
  """Per-tracker-pair, per-kf inter-fingertip distance + summary stats.
  ``tracker_order`` controls the pair-key direction (``a↔b`` with a < b
  in the input order); pass the same order ``track.py`` uses to keep
  pair keys stable across runs."""
  per_kf: dict[str, list[float]] = {}
  stats: dict[str, dict[str, float]] = {}
  for i, name_i in enumerate(tracker_order):
    for name_j in tracker_order[i + 1 :]:
      if (
        name_i not in keypoint_world_by_tracker
        or name_j not in keypoint_world_by_tracker
      ):
        continue
      common = min(
        len(keypoint_world_by_tracker[name_i]), len(keypoint_world_by_tracker[name_j])
      )
      if common == 0:
        continue
      kpi = np.stack(keypoint_world_by_tracker[name_i][:common], axis=0)
      kpj = np.stack(keypoint_world_by_tracker[name_j][:common], axis=0)
      dists = np.linalg.norm(kpi - kpj, axis=1)
      pair_key = f"{name_i}↔{name_j}"
      per_kf[pair_key] = [float(x) for x in dists]
      stats[pair_key] = {
        "n": int(common),
        "mean": float(dists.mean()),
        "median": float(np.median(dists)),
        "p95": float(np.percentile(dists, 95)),
        "max": float(dists.max()),
      }
  return per_kf, stats
