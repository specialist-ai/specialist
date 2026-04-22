"""Visual frontend for VI-BA: ORB detect/match/essential-RANSAC and
KLT optical-flow tracking. Camera-model-agnostic — RANSAC uses
caller-supplied unit bearings. Each lens runs independently."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "Frame",
  "chain_tracks",
  "detect_orb",
  "dis_flow_track",
  "essential_ransac_inliers",
  "klt_track",
  "match_pair",
]


def _bilinear_sample_flow(
  flow: npt.NDArray[np.float32], xy: Float64Array
) -> Float64Array:
  """Bilinear-interpolated dense-flow lookup at subpixel positions ``xy``."""
  h, w = flow.shape[:2]
  if xy.shape[0] == 0:
    return np.empty((0, 2), dtype=np.float64)
  x = np.clip(xy[:, 0], 0.0, float(w) - 1.001)
  y = np.clip(xy[:, 1], 0.0, float(h) - 1.001)
  x0 = x.astype(np.int64)
  y0 = y.astype(np.int64)
  fx = (x - x0).astype(np.float64)[:, None]
  fy = (y - y0).astype(np.float64)[:, None]
  f00 = flow[y0, x0].astype(np.float64)
  f01 = flow[y0, x0 + 1].astype(np.float64)
  f10 = flow[y0 + 1, x0].astype(np.float64)
  f11 = flow[y0 + 1, x0 + 1].astype(np.float64)
  return (
    (1.0 - fx) * (1.0 - fy) * f00
    + fx * (1.0 - fy) * f01
    + (1.0 - fx) * fy * f10
    + fx * fy * f11
  )


def _in_bounds(xy: Float64Array, h: int, w: int) -> npt.NDArray[np.bool_]:
  return (xy[:, 0] >= 1.0) & (xy[:, 0] < w - 1) & (xy[:, 1] >= 1.0) & (xy[:, 1] < h - 1)


def _mask_ok(xy: Float64Array, mask: np.ndarray) -> npt.NDArray[np.bool_]:
  ix = np.round(xy[:, 0]).astype(np.int64)
  iy = np.round(xy[:, 1]).astype(np.int64)
  h, w = mask.shape
  ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
  vals = np.zeros(len(xy), dtype=np.uint8)
  vals[ok] = mask[iy[ok], ix[ok]]
  return vals > 0


def _detect_corners(
  img: np.ndarray,
  mask: np.ndarray | None,
  max_corners: int,
  quality_level: float,
  min_distance_px: float,
) -> Float64Array:
  if max_corners <= 0:
    return np.empty((0, 2), dtype=np.float64)
  corners = cv2.goodFeaturesToTrack(
    img,
    maxCorners=int(max_corners),
    qualityLevel=quality_level,
    minDistance=float(min_distance_px),
    mask=mask,
  )
  if corners is None:
    return np.empty((0, 2), dtype=np.float64)
  return corners.reshape(-1, 2).astype(np.float64)


def dis_flow_track(
  frames: list[np.ndarray],
  n_features: int,
  masks: list[np.ndarray] | None = None,
  fb_error_threshold_px: float = 1.0,
  min_distance_px: float = 8.0,
  quality_level: float = 0.01,
  redetect_ratio: float = 0.5,
) -> tuple[list[Frame], list[npt.NDArray[np.int64]]]:
  """Sparse-track propagation via DIS dense optical flow — wholesale
  replacement for ``klt_track``. Same Shi-Tomasi seeding and redetect
  logic; the per-pair propagation step is dense flow sampled bilinearly
  at each prior-kf pixel position. The dense flow is computed
  multi-scale across the whole frame, so global rotation patterns are
  captured before per-track sampling, removing KLT's per-window
  convergence-radius limit (DIS finds large displacements natively, so no
  rotation predictor is needed).
  """
  if masks is not None and len(masks) != len(frames):
    raise ValueError(f"masks length {len(masks)} != frames length {len(frames)}")
  if not frames:
    return [], []

  dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)  # type: ignore[attr-defined]
  dis.setVariationalRefinementIterations(25)
  per_frame_pts: list[list[np.ndarray]] = [[] for _ in frames]
  matches_between: list[npt.NDArray[np.int64]] = []

  for k, img in enumerate(frames):
    mask_k = None if masks is None else masks[k]

    if k > 0 and per_frame_pts[k - 1]:
      prev_xy = np.asarray(per_frame_pts[k - 1], dtype=np.float64)
      prev_img = np.ascontiguousarray(frames[k - 1])
      cur_img = np.ascontiguousarray(img)
      flow_fwd = dis.calc(prev_img, cur_img, None)
      flow_bwd = dis.calc(cur_img, prev_img, None)
      disp_fwd = _bilinear_sample_flow(flow_fwd, prev_xy)
      nxt_xy = prev_xy + disp_fwd
      disp_bwd = _bilinear_sample_flow(flow_bwd, nxt_xy)
      back_xy = nxt_xy + disp_bwd
      fb = np.linalg.norm(back_xy - prev_xy, axis=1)
      ok = np.isfinite(fb) & (fb < fb_error_threshold_px)
      h, w = img.shape[:2]
      ok &= _in_bounds(nxt_xy, h, w)
      if mask_k is not None:
        ok &= _mask_ok(nxt_xy, mask_k)
      matches: list[tuple[int, int]] = []
      for prev_idx in np.where(ok)[0]:
        cur_idx = len(per_frame_pts[k])
        per_frame_pts[k].append(nxt_xy[prev_idx])
        matches.append((int(prev_idx), cur_idx))
      matches_between.append(
        np.asarray(matches, dtype=np.int64)
        if matches
        else np.empty((0, 2), dtype=np.int64)
      )
    elif k > 0:
      matches_between.append(np.empty((0, 2), dtype=np.int64))

    if per_frame_pts[k]:
      pts_subpix = np.asarray(per_frame_pts[k], dtype=np.float32).reshape(-1, 1, 2)
      cv2.cornerSubPix(
        img,
        pts_subpix,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
      )
      per_frame_pts[k] = list(pts_subpix.reshape(-1, 2).astype(np.float64))

    n_active = len(per_frame_pts[k])
    need_redetect = k == 0 or n_active < int(round(redetect_ratio * n_features))
    if need_redetect:
      if mask_k is None:
        redetect_mask = np.full(img.shape[:2], 255, dtype=np.uint8)
      else:
        redetect_mask = mask_k.copy()
      for p in per_frame_pts[k]:
        cv2.circle(
          redetect_mask,
          (int(round(float(p[0]))), int(round(float(p[1])))),
          int(round(float(min_distance_px))),
          0,
          thickness=-1,
        )
      new_corners = _detect_corners(
        img,
        redetect_mask,
        max_corners=n_features - n_active,
        quality_level=quality_level,
        min_distance_px=min_distance_px,
      )
      for p in new_corners:
        per_frame_pts[k].append(p.copy())

  frames_out: list[Frame] = []
  for pts in per_frame_pts:
    kps = (
      np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2), dtype=np.float64)
    )
    frames_out.append(
      Frame(
        keypoints=kps,
        descriptors=np.empty((kps.shape[0], 32), dtype=np.uint8),
      )
    )
  return frames_out, matches_between


@dataclass(frozen=True)
class Frame:
  """One keyframe's ORB output: 2D keypoints and 32-byte descriptors."""

  keypoints: Float64Array
  descriptors: np.ndarray


def detect_orb(
  image: np.ndarray,
  n_features: int,
  fast_threshold: int,
  mask: np.ndarray | None = None,
) -> Frame:
  """Run OpenCV ORB on a single grayscale frame.

  ``mask`` is the standard OpenCV per-pixel ``uint8`` mask: ``255`` keeps a
  pixel eligible for detection, ``0`` excludes it. Used to suppress
  keypoints on body-attached regions (gripper, mounted tags) that would
  otherwise bias the BA toward zero motion.
  """
  orb = cv2.ORB_create(  # type: ignore[attr-defined]
    nfeatures=n_features, fastThreshold=fast_threshold
  )
  kps, desc = orb.detectAndCompute(image, mask)
  if kps is None or len(kps) == 0 or desc is None:
    return Frame(
      keypoints=np.empty((0, 2), dtype=np.float64),
      descriptors=np.empty((0, 32), dtype=np.uint8),
    )
  return Frame(
    keypoints=np.asarray([k.pt for k in kps], dtype=np.float64),
    descriptors=np.asarray(desc, dtype=np.uint8),
  )


def match_pair(a: Frame, b: Frame, ratio: float) -> npt.NDArray[np.int64]:
  """Lowe ratio-test BFMatch between two frames.

  Returns an ``(M, 2)`` array of ``(query_idx_in_a, train_idx_in_b)``.
  """
  if a.descriptors.size == 0 or b.descriptors.size == 0:
    return np.empty((0, 2), dtype=np.int64)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  knn = bf.knnMatch(a.descriptors, b.descriptors, k=2)
  good: list[tuple[int, int]] = []
  for pair in knn:
    if len(pair) < 2:
      continue
    m0, m1 = pair
    if m0.distance < ratio * m1.distance:
      good.append((int(m0.queryIdx), int(m0.trainIdx)))
  return np.asarray(good, dtype=np.int64) if good else np.empty((0, 2), dtype=np.int64)


def essential_ransac_inliers(
  pts_a: Float64Array,
  pts_b: Float64Array,
  bearings_a: Float64Array,
  bearings_b: Float64Array,
  threshold_px: float,
  fx: float,
  max_half_angle: float = np.pi / 2.0 - 1e-6,
) -> npt.NDArray[np.bool_]:
  """RANSAC essential-matrix inlier mask.

  ``pts_*`` and ``bearings_*`` must be row-aligned. Bearings are unit
  vectors in each lens frame (the caller unprojects through its own MEI/KB
  model). ``threshold_px / fx`` sets the inlier threshold in
  normalised-image units. Correspondences whose bearing z-component falls
  below ``cos(max_half_angle)`` are excluded from the solve — they
  are numerically unstable once divided by a near-zero z.
  """
  n = pts_a.shape[0]
  if n < 8 or pts_b.shape[0] != n:
    return np.zeros(n, dtype=bool)

  cos_min = float(np.cos(max_half_angle))
  in_fov = (bearings_a[:, 2] > cos_min) & (bearings_b[:, 2] > cos_min)
  if int(in_fov.sum()) < 8:
    return np.zeros(n, dtype=bool)

  norm_a = (bearings_a[in_fov][:, :2] / bearings_a[in_fov][:, 2:3]).astype(np.float64)
  norm_b = (bearings_b[in_fov][:, :2] / bearings_b[in_fov][:, 2:3]).astype(np.float64)
  thresh = float(threshold_px / fx)

  E, mask = cv2.findEssentialMat(
    norm_a,
    norm_b,
    focal=1.0,
    pp=(0.0, 0.0),
    method=cv2.RANSAC,
    prob=0.999,
    threshold=thresh,
  )
  out = np.zeros(n, dtype=bool)
  if E is None or mask is None:
    return out
  in_fov_idx = np.where(in_fov)[0]
  mask_flat = mask.flatten().astype(bool)
  out[in_fov_idx[mask_flat]] = True
  return out


def klt_track(
  frames: list[np.ndarray],
  n_features: int,
  masks: list[np.ndarray] | None = None,
  window_size: int = 21,
  max_pyramid_level: int = 3,
  fb_error_threshold_px: float = 1.0,
  min_distance_px: float = 8.0,
  quality_level: float = 0.01,
  redetect_ratio: float = 0.5,
  rotation_predictor: "Callable[[int, Float64Array], Float64Array] | None" = None,
) -> tuple[list[Frame], list[npt.NDArray[np.int64]]]:
  """Lucas-Kanade pyramidal optical-flow tracker with FB consistency
  check. Re-seeds via ``cv2.goodFeaturesToTrack`` whenever the active
  pool drops below ``redetect_ratio * n_features``. Returns
  ``(frames_out, matches_between)`` in the same shape as
  ``detect_orb`` + ``match_pair`` so ``chain_tracks`` works unchanged.
  ``masks[k]`` filters new detections and tracked points that drift in.

  ``rotation_predictor(k, prev_pixels)`` (optional) returns ``(N, 2)``
  predicted pixel positions at frame ``k`` for the given prev-frame
  pixels using IMU-integrated rotation. KLT then runs with
  ``OPTFLOW_USE_INITIAL_FLOW`` so the search starts at the predicted
  location. Non-finite predictions fall back to the prev pixel.
  """
  if masks is not None and len(masks) != len(frames):
    raise ValueError(f"masks length {len(masks)} != frames length {len(frames)}")
  if not frames:
    return [], []

  lk_win = (window_size, window_size)
  lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
  per_frame_pts: list[list[np.ndarray]] = [[] for _ in frames]
  matches_between: list[npt.NDArray[np.int64]] = []

  for k, img in enumerate(frames):
    mask_k = None if masks is None else masks[k]

    if k > 0 and per_frame_pts[k - 1]:
      prev = np.asarray(per_frame_pts[k - 1], dtype=np.float32).reshape(-1, 1, 2)
      flags_klt = 0
      initial_nxt = prev.copy()
      if rotation_predictor is not None:
        pred = np.asarray(
          rotation_predictor(k, prev.reshape(-1, 2).astype(np.float64)),
          dtype=np.float64,
        ).reshape(-1, 2)
        finite = np.all(np.isfinite(pred), axis=1)
        guess = prev.reshape(-1, 2).astype(np.float32, copy=True)
        guess[finite] = pred[finite].astype(np.float32)
        initial_nxt = guess.reshape(-1, 1, 2)
        flags_klt = cv2.OPTFLOW_USE_INITIAL_FLOW
      nxt, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
        frames[k - 1],
        img,
        prev,
        initial_nxt,
        winSize=lk_win,
        maxLevel=max_pyramid_level,
        criteria=lk_criteria,
        flags=flags_klt,
      )
      back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
        img,
        frames[k - 1],
        nxt,
        None,
        winSize=lk_win,
        maxLevel=max_pyramid_level,
        criteria=lk_criteria,
      )
      fb = np.linalg.norm(back.reshape(-1, 2) - prev.reshape(-1, 2), axis=1).astype(
        np.float64
      )
      ok = (
        (st_fwd.flatten() == 1) & (st_bwd.flatten() == 1) & (fb < fb_error_threshold_px)
      )
      nxt_xy = nxt.reshape(-1, 2).astype(np.float64)
      h, w = img.shape[:2]
      ok &= _in_bounds(nxt_xy, h, w)
      if mask_k is not None:
        ok &= _mask_ok(nxt_xy, mask_k)
      matches: list[tuple[int, int]] = []
      for prev_idx in np.where(ok)[0]:
        cur_idx = len(per_frame_pts[k])
        per_frame_pts[k].append(nxt_xy[prev_idx])
        matches.append((int(prev_idx), cur_idx))
      matches_between.append(
        np.asarray(matches, dtype=np.int64)
        if matches
        else np.empty((0, 2), dtype=np.int64)
      )
    elif k > 0:
      matches_between.append(np.empty((0, 2), dtype=np.int64))

    # Snap survivors back to local corner peaks; KLT drifts a few tenths
    # of a pixel along intensity gradients per kf, which biases the LM.
    if per_frame_pts[k]:
      pts_subpix = np.asarray(per_frame_pts[k], dtype=np.float32).reshape(-1, 1, 2)
      cv2.cornerSubPix(
        img,
        pts_subpix,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
      )
      per_frame_pts[k] = list(pts_subpix.reshape(-1, 2).astype(np.float64))

    n_active = len(per_frame_pts[k])
    need_redetect = k == 0 or n_active < int(round(redetect_ratio * n_features))
    if need_redetect:
      if mask_k is None:
        redetect_mask = np.full(img.shape[:2], 255, dtype=np.uint8)
      else:
        redetect_mask = mask_k.copy()
      for p in per_frame_pts[k]:
        cv2.circle(
          redetect_mask,
          (int(round(float(p[0]))), int(round(float(p[1])))),
          int(round(float(min_distance_px))),
          0,
          thickness=-1,
        )
      new_corners = _detect_corners(
        img,
        redetect_mask,
        max_corners=n_features - n_active,
        quality_level=quality_level,
        min_distance_px=min_distance_px,
      )
      for p in new_corners:
        per_frame_pts[k].append(p.copy())

  frames_out: list[Frame] = []
  for pts in per_frame_pts:
    kps = (
      np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2), dtype=np.float64)
    )
    frames_out.append(
      Frame(
        keypoints=kps,
        descriptors=np.empty((kps.shape[0], 32), dtype=np.uint8),
      )
    )
  return frames_out, matches_between


def chain_tracks(
  frames: list[Frame],
  matches_between: list[npt.NDArray[np.int64]],
) -> dict[int, dict[int, Float64Array]]:
  """Chain pairwise matches into per-landmark tracks.

  ``matches_between[k]`` is the ``(M, 2)`` inlier-match array for
  ``frames[k] → frames[k+1]``. Returns ``{track_id: {frame_idx: pt2d}}``.
  """
  if len(matches_between) != max(len(frames) - 1, 0):
    raise ValueError(
      f"expected {len(frames) - 1} match arrays, got {len(matches_between)}"
    )
  tracks: dict[int, dict[int, Float64Array]] = {}
  lookup: dict[tuple[int, int], int] = {}
  next_id = 0
  for k, pairs in enumerate(matches_between, start=1):
    if pairs.shape[0] == 0:
      continue
    for a_idx, b_idx in pairs:
      prev_key = (k - 1, int(a_idx))
      if prev_key in lookup:
        tid = lookup[prev_key]
      else:
        tid = next_id
        next_id += 1
        tracks[tid] = {k - 1: frames[k - 1].keypoints[int(a_idx)]}
        lookup[prev_key] = tid
      tracks[tid][k] = frames[k].keypoints[int(b_idx)]
      lookup[(k, int(b_idx))] = tid
  return tracks
