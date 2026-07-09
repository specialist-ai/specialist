"""Visual frontend for VI-BA: DIS optical-flow tracking with essential-RANSAC
filtering, plus ORB detect/match for loop-closure re-matching.
Camera-model-agnostic — RANSAC uses caller-supplied unit bearings. Each lens
runs independently."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

import cv2
import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]
_R = TypeVar("_R")

__all__ = [
  "Frame",
  "chain_tracks",
  "detect_orb",
  "dis_flow_track",
  "essential_ransac_inliers",
  "match_pair",
  "parallel_map",
]


@dataclass(frozen=True)
class Frame:
  """One keyframe's 2D keypoints and their descriptors."""

  keypoints: Float64Array
  descriptors: np.ndarray


def parallel_map(fn: Callable[..., _R], *iterables: Iterable[object]) -> list[_R]:
  """Ordered ``ThreadPoolExecutor.map`` over ``iterables`` for native per-item
  work that releases the GIL. Results come back in input order."""
  columns = [list(it) for it in iterables]
  if len({len(column) for column in columns}) > 1:
    raise ValueError("parallel_map iterables must have equal length")
  if not columns[0]:
    return []
  workers = min(len(columns[0]), max(4, (os.cpu_count() or 8) // 8))
  with ThreadPoolExecutor(max_workers=workers) as executor:
    return list(executor.map(fn, *columns))


def _bilinear_sample_flow(
  flow: npt.NDArray[np.float32], xy: Float64Array
) -> Float64Array:
  """Bilinear-interpolated dense-flow lookup at subpixel positions ``xy``."""
  h, w = flow.shape[:2]
  if xy.shape[0] == 0:
    return np.empty((0, 2))
  x = np.clip(xy[:, 0], 0.0, float(w) - 1.001)
  y = np.clip(xy[:, 1], 0.0, float(h) - 1.001)
  x0 = x.astype(np.int64)
  y0 = y.astype(np.int64)
  fx = (x - x0)[:, None]
  fy = (y - y0)[:, None]
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


_PRECOMPUTE_CONCURRENCY = 1

# DIS variational-refinement iterations: the subpixel-polish pass, and the bulk
# of DIS runtime.
_DIS_VARIATIONAL_ITERS = 10
# The backward flow feeds only the forward-backward consistency gate, never a
# track coordinate.
_DIS_BACKWARD_VARIATIONAL_ITERS = 2


def set_precompute_concurrency(n: int) -> None:
  """Tell the flow precompute how many trackers prepare concurrently, so it can
  shrink its per-lens worker pool. 1 = the single-tracker default."""
  global _PRECOMPUTE_CONCURRENCY
  _PRECOMPUTE_CONCURRENCY = max(1, n)


def _flow_chunk_size(frames: list[np.ndarray]) -> int:
  """How many consecutive flow pairs to precompute and hold at once, bounded so
  the cached fields fit a fraction of RAM. Each pair holds forward+backward
  ``(H, W, 2)`` float32 fields, and the two lenses (times any concurrent trackers)
  share memory. On a big-RAM host this is the whole sequence in one chunk; on a
  small host it caps peak memory so the precompute does not exhaust RAM."""
  n = len(frames)
  if n < 2:
    return 1
  h, w = frames[0].shape[:2]
  per_pair_bytes = h * w * 2 * 4 * 2  # (H, W, 2) float32, forward + backward
  total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
  lanes = 2 * _PRECOMPUTE_CONCURRENCY  # two lenses run concurrently per tracker
  chunk = int(total_ram * 0.25 / (per_pair_bytes * lanes))
  return max(1, min(n - 1, chunk))


def _precompute_dis_flows(
  frames: list[np.ndarray],
  ks: range | None = None,
) -> dict[int, tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:
  """Forward+backward DIS flow for the consecutive pairs ``ks`` (each ``k`` is the
  pair ``(k-1, k)``; default = every pair), computed in parallel. Each pair's flow
  depends only on its two frames — not on the sequential track propagation that
  consumes it — so this is bit-identical to per-pair serial ``dis.calc`` (a fresh
  ``DISOpticalFlow`` with no warm-start flow carries no cross-call state); the
  parallelism just keeps the cores busy. Callers request bounded chunks (see
  ``_flow_chunk_size``) so the cached fields do not exhaust RAM."""
  if ks is None:
    ks = range(1, len(frames))
  pair_ks = list(ks)
  if not pair_ks:
    return {}

  def calc_pair(
    k: int,
  ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)  # type: ignore[attr-defined]
    prev = np.ascontiguousarray(frames[k - 1])
    cur = np.ascontiguousarray(frames[k])
    dis.setVariationalRefinementIterations(_DIS_VARIATIONAL_ITERS)
    forward = dis.calc(prev, cur, None)
    dis.setVariationalRefinementIterations(_DIS_BACKWARD_VARIATIONAL_ITERS)
    backward = dis.calc(cur, prev, None)
    return forward, backward

  # A single DIS.calc saturates ~32 threads; concurrent calls only help when
  # there are cores to spare. Thread and worker counts never change the output.
  n_cpu = os.cpu_count() or 32
  cv2.setNumThreads(max(n_cpu // 8, min(n_cpu, 32)))
  workers = max(1, n_cpu // 16 // _PRECOMPUTE_CONCURRENCY)
  with ThreadPoolExecutor(max_workers=workers) as executor:
    return dict(zip(pair_ks, executor.map(calc_pair, pair_ks), strict=True))


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
  min_distance: float,
) -> Float64Array:
  if max_corners <= 0:
    return np.empty((0, 2))
  corners = cv2.goodFeaturesToTrack(
    img,
    maxCorners=int(max_corners),
    qualityLevel=quality_level,
    minDistance=float(min_distance),
    mask=mask,
  )
  if corners is None:
    return np.empty((0, 2))
  return corners.reshape(-1, 2).astype(np.float64)


def dis_flow_track(
  frames: list[np.ndarray],
  n_features: int,
  masks: list[np.ndarray] | None = None,
  fb_error_threshold: float = 1.0,
  min_distance: float = 8.0,
  quality_level: float = 0.01,
  redetect_ratio: float = 0.5,
) -> tuple[list[Frame], list[npt.NDArray[np.int64]]]:
  """Sparse-track propagation via DIS dense optical flow: Shi-Tomasi seeding
  with redetect, and a per-pair propagation step that samples the dense flow
  bilinearly at each prior-kf pixel position.
  """
  if masks is not None and len(masks) != len(frames):
    raise ValueError(f"masks length {len(masks)} != frames length {len(frames)}")
  if not frames:
    return [], []

  # Precompute each consecutive pair's flow in parallel, in RAM-bounded chunks
  # (see _flow_chunk_size); the loop below just samples the cached fields. Each
  # pair's flow is independent of the sequential propagation, so this stays
  # bit-identical (the precompute also sets the cv2 thread budget for the lighter
  # ops below).
  chunk_size = _flow_chunk_size(frames)
  flows: dict[int, tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]] = {}
  per_frame_pts: list[list[np.ndarray]] = [[] for _ in frames]
  matches_between: list[npt.NDArray[np.int64]] = []

  for k, img in enumerate(frames):
    mask_k = None if masks is None else masks[k]
    if k >= 1 and k not in flows:
      flows = _precompute_dis_flows(frames, range(k, min(k + chunk_size, len(frames))))

    if k > 0 and per_frame_pts[k - 1]:
      prev_xy = np.asarray(per_frame_pts[k - 1], dtype=np.float64)
      flow_fwd, flow_bwd = flows[k]
      disp_fwd = _bilinear_sample_flow(flow_fwd, prev_xy)
      nxt_xy = prev_xy + disp_fwd
      disp_bwd = _bilinear_sample_flow(flow_bwd, nxt_xy)
      back_xy = nxt_xy + disp_bwd
      fb = np.linalg.norm(back_xy - prev_xy, axis=1)
      ok = np.isfinite(fb) & (fb < fb_error_threshold)
      h, w = img.shape[:2]
      ok &= _in_bounds(nxt_xy, h, w)
      if mask_k is not None:
        ok &= _mask_ok(nxt_xy, mask_k)
      sel = np.where(ok)[0]
      per_frame_pts[k] = list(nxt_xy[sel])
      matches_between.append(
        np.column_stack([sel, np.arange(sel.size)]).astype(np.int64)
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
          int(round(float(min_distance))),
          0,
          thickness=-1,
        )
      new_corners = _detect_corners(
        img,
        redetect_mask,
        max_corners=n_features - n_active,
        quality_level=quality_level,
        min_distance=min_distance,
      )
      for p in new_corners:
        per_frame_pts[k].append(p.copy())

  frames_out: list[Frame] = []
  for pts in per_frame_pts:
    kps = np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2))
    frames_out.append(
      Frame(
        keypoints=kps,
        descriptors=np.empty((kps.shape[0], 32), dtype=np.uint8),
      )
    )
  return frames_out, matches_between


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
      keypoints=np.empty((0, 2)),
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

  ba = bearings_a[in_fov]
  bb = bearings_b[in_fov]
  norm_a = (ba[:, :2] / ba[:, 2:3]).astype(np.float64)
  norm_b = (bb[:, :2] / bb[:, 2:3]).astype(np.float64)
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
