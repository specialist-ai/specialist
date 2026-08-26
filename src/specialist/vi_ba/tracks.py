"""One lens's tracks: the visual frontend that chains them, the ORB loop
closure that reconnects the ones fast motion breaks, the multi-view
triangulation that turns them into landmarks, and the caps that bound how many
of each reach the bundle."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import cv2
import gtsam
import numpy as np
import numpy.typing as npt
from specialist_camera import MeiParams, mei_unproject, unit_rows

from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.configs import FrontendCfg
from specialist.vi_ba.features import (
  Frame,
  Rect,
  chain_tracks,
  dis_flow_track,
  essential_ransac_inliers,
  match_pair,
  parallel_map,
)
from specialist.vi_ba.geometry import translation
from specialist.vi_ba.init import triangulate_track_multiview

Float64Array = npt.NDArray[np.float64]

# ORB loop-closure track recovery (``loop_closure_tracks``).
_LOOP_N_NEIGHBORS = 8  # spatial-neighbour revisit candidates per keyframe
_LOOP_MATCH_RATIO = 0.8  # Lowe ratio for the revisit ORB matches
_LOOP_MIN_INLIERS = 10  # matches a revisit pair must keep to count
_LOOP_MAX_DIST = 0.1  # metres between keyframes for a revisit candidate
_LOOP_MIN_KF_GAP = 15  # keyframes a revisit must span, or it is just motion
_LOOP_RANSAC_PX = 4.5  # homography RANSAC reprojection tolerance
_LOOP_MIN_VIEWS = 3  # keyframes a union-found track must span to be kept

__all__ = [
  "add_loop_landmarks",
  "bearings_by_track",
  "cap_landmarks",
  "filter_tracks_by_parallax",
  "loop_closure_tracks",
  "run_lens_frontend",
  "subsample_obs",
  "triangulate_lens",
  "unproject_tracks",
]


def run_lens_frontend(
  frames: Sequence[np.ndarray],
  params: MeiParams,
  cfg: FrontendCfg,
  masks: list[list[Rect]] | None = None,
  *,
  label: str | None = None,
) -> tuple[list[Float64Array], dict[int, dict[int, Float64Array]]]:
  """DIS flow → essential-RANSAC → chain for one lens, returning the per-frame
  keypoints and the chained tracks. ``masks[k]`` (optional, the blocked
  rectangles for that keyframe) suppresses new detections and filters tracked
  points that drift into them."""
  if masks is not None and len(masks) != len(frames):
    raise ValueError(f"masks length {len(masks)} != frames length {len(frames)}")
  keypoints, matches_between = dis_flow_track(
    frames,
    n_features=cfg.orb_n_features,
    masks=masks,
    gpu_flow=cfg.gpu_flow,
    flow_scale=cfg.flow_scale,
    label=label,
  )

  def ransac_pair(k: int) -> npt.NDArray[np.int64]:
    m = matches_between[k - 1]
    if m.shape[0] == 0:
      return np.empty((0, 2), dtype=np.int64)
    # The cv2 RNG is thread-local, so a per-pair seed keeps RANSAC
    # deterministic.
    cv2.setRNGSeed(k)
    inliers = essential_ransac_inliers(
      mei_unproject(keypoints[k - 1][m[:, 0]], params),
      mei_unproject(keypoints[k][m[:, 1]], params),
      threshold_px=cfg.ransac_threshold,
      fx=params.fx,
      max_half_angle=cfg.feature_max_half_angle,
    )
    return np.asarray(m[inliers], dtype=np.int64)

  filtered = parallel_map(ransac_pair, range(1, len(keypoints)))
  tracks = chain_tracks(keypoints, filtered)
  n_kp = sum(len(k) for k in keypoints)
  n_match = sum(int(m.shape[0]) for m in matches_between)
  n_in = sum(int(f.shape[0]) for f in filtered)
  scope = f"{label}/frontend" if label else "frontend"
  print(
    f"  [{scope}] funnel keypoints={n_kp} matched={n_match} "
    f"ransac_inliers={n_in} chained_tracks={len(tracks)}"
  )
  return keypoints, tracks


def unproject_tracks(
  tracks: dict[int, dict[int, Float64Array]],
  params: MeiParams,
  min_obs: int,
  normalize: bool,
) -> Iterator[tuple[int, list[int], Float64Array]]:
  """Yield ``(tid, sorted kfs, bearings)`` for each track with at least
  ``min_obs`` observations. ``normalize`` unit-normalizes the bearing rows."""
  spans: list[tuple[int, list[int], int, int]] = []
  flat_pixels: list[Float64Array] = []
  for tid, obs in tracks.items():
    if len(obs) < min_obs:
      continue
    kfs = sorted(obs)
    start = len(flat_pixels)
    flat_pixels.extend(obs[k] for k in kfs)
    spans.append((tid, kfs, start, len(flat_pixels)))
  if not spans:
    return
  bearings = mei_unproject(np.asarray(flat_pixels, dtype=np.float64), params)
  if normalize:
    bearings = unit_rows(bearings)
  for tid, kfs, start, end in spans:
    yield tid, kfs, bearings[start:end]


def filter_tracks_by_parallax(
  tracks: dict[int, dict[int, Float64Array]],
  params: MeiParams,
  min_parallax: float,
) -> tuple[dict[int, dict[int, Float64Array]], dict[int, dict[int, Float64Array]]]:
  """Split tracks into ``(kept, rejected)`` by max bearing-angle change.
  Body-attached features are stationary in image space and get rejected."""
  if min_parallax < 0.0:
    raise ValueError(f"min_parallax must be ≥ 0, got {min_parallax}")
  cos_min = float(np.cos(min_parallax))
  kept: dict[int, dict[int, Float64Array]] = {}
  rejected = {tid: obs for tid, obs in tracks.items() if len(obs) < 2}
  for tid, _, bearings in unproject_tracks(tracks, params, min_obs=2, normalize=True):
    dots = np.clip(bearings @ bearings.T, -1.0, 1.0)
    if float(dots.min()) <= cos_min:
      kept[tid] = tracks[tid]
    else:
      rejected[tid] = tracks[tid]
  return kept, rejected


def triangulate_lens(
  tracks: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  extrinsic: gtsam.Pose3,
  params: MeiParams,
  min_track_length: int,
  min_landmark_depth: float = 0.15,
  max_landmark_depth: float = 5.0,
) -> dict[int, Float64Array]:
  """Triangulate every track by geometric closest-point across all observations.
  ``min_landmark_depth`` / ``max_landmark_depth`` gate the median-view
  depth — callers triangulating against unscaled visual-frame poses
  should widen these bounds."""
  if not tracks:
    return {}
  lens_poses = [pose.compose(extrinsic) for pose in body_poses]
  lens_rotations = np.stack([pose.rotation().matrix() for pose in lens_poses])
  lens_centers = np.stack([translation(pose) for pose in lens_poses])
  out: dict[int, Float64Array] = {}
  for tid, kfs, bearings in unproject_tracks(
    tracks, params, min_obs=min_track_length, normalize=False
  ):
    xyz = triangulate_track_multiview(
      bearings,
      lens_rotations[kfs],
      lens_centers[kfs],
      min_landmark_depth=min_landmark_depth,
      max_landmark_depth=max_landmark_depth,
    )
    if xyz is not None:
      out[tid] = xyz
  return out


def bearings_by_track(
  tracks: dict[int, dict[int, Float64Array]], params: MeiParams
) -> dict[int, dict[int, Float64Array]]:
  """``{tid: {kf: unit bearing}}`` for the visual-odometry front-end."""
  return {
    tid: dict(zip(kfs, bearings, strict=True))
    for tid, kfs, bearings in unproject_tracks(
      tracks, params, min_obs=1, normalize=True
    )
  }


def subsample_obs(
  track: dict[int, Float64Array], max_obs: int
) -> dict[int, Float64Array]:
  """Keep at most ``max_obs`` observations (endpoints + evenly-spaced middle),
  preserving the track's baseline."""
  kfs = sorted(track)
  if max_obs <= 0 or len(kfs) <= max_obs:
    return track
  idx = np.unique(np.linspace(0, len(kfs) - 1, max_obs).round().astype(int))
  return {kfs[i]: track[kfs[i]] for i in idx}


def cap_landmarks(
  points: dict[int, Float64Array],
  tracks: dict[int, dict[int, Float64Array]],
  cap: int,
) -> dict[int, Float64Array]:
  """Keep at most ``cap`` landmarks, preferring the longest tracks. No-op when
  already under the cap."""
  if cap <= 0 or len(points) <= cap:
    return points
  keep = set(sorted(points, key=lambda t: -len(tracks.get(t, {})))[:cap])
  return {t: p for t, p in points.items() if t in keep}


def loop_closure_tracks(
  orb: list[Frame],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  base_track_id: int,
  cfg: FrontendCfg,
) -> dict[int, dict[int, Float64Array]]:
  """Recover tracks broken by fast lifts via ORB loop closure.

  ORB features are re-matched between spatially-close, non-adjacent keyframes,
  geometrically verified, and union-found into multi-view tracks spanning the
  lifts."""
  centers = np.asarray([translation(p.compose(lens.extrinsic)) for p in body_poses])
  n = len(orb)
  parent: dict[tuple[int, int], tuple[int, int]] = {}

  def find(x: tuple[int, int]) -> tuple[int, int]:
    parent.setdefault(x, x)
    root = x
    while parent[root] != root:
      root = parent[root]
    while parent[x] != root:
      parent[x], x = root, parent[x]
    return root

  pairs: list[tuple[int, int]] = []
  for i in range(n):
    if len(orb[i].keypoints) == 0:
      continue
    dist = np.linalg.norm(centers - centers[i], axis=1)
    cand = [
      int(j)
      for j in np.argsort(dist)
      if j > i and (j - i) > _LOOP_MIN_KF_GAP and dist[j] <= _LOOP_MAX_DIST
    ][:_LOOP_N_NEIGHBORS]
    pairs.extend((i, j) for j in cand)

  def verify_pair(i: int, j: int) -> npt.NDArray[np.int64] | None:
    m = match_pair(orb[i], orb[j], _LOOP_MATCH_RATIO)
    if len(m) < _LOOP_MIN_INLIERS:
      return None
    pa = orb[i].keypoints[m[:, 0]]
    pb = orb[j].keypoints[m[:, 1]]
    # Revisits are near-identical small-baseline views, so the essential matrix
    # is degenerate; a homography is the right model. The cv2 RNG is
    # thread-local, so a per-pair seed keeps RANSAC deterministic.
    cv2.setRNGSeed(i * n + j)
    _, mask = cv2.findHomography(pa, pb, cv2.RANSAC, _LOOP_RANSAC_PX)
    if mask is None:
      return None
    inl = mask.ravel().astype(bool)
    if int(inl.sum()) < _LOOP_MIN_INLIERS:
      return None
    return np.asarray(m[inl], dtype=np.int64)

  verified = parallel_map(verify_pair, [i for i, _ in pairs], [j for _, j in pairs])
  for (i, j), matches in zip(pairs, verified, strict=True):
    if matches is None:
      continue
    for a_idx, b_idx in matches:
      ra = find((i, int(a_idx)))
      rb = find((j, int(b_idx)))
      if ra != rb:
        parent[ra] = rb

  comps: dict[tuple[int, int], dict[int, int]] = {}
  for node in list(parent):
    comps.setdefault(find(node), {})[node[0]] = node[1]
  tracks: dict[int, dict[int, Float64Array]] = {}
  tid = base_track_id + 1
  for by_kf in comps.values():
    if len(by_kf) < _LOOP_MIN_VIEWS:
      continue
    # Pixel coordinates, as the optical-flow tracks carry.
    tracks[tid] = {
      kf: np.asarray(orb[kf].keypoints[kp], dtype=np.float64)
      for kf, kp in by_kf.items()
    }
    tid += 1
  return tracks


def add_loop_landmarks(
  points: dict[int, Float64Array],
  kept: dict[int, dict[int, Float64Array]],
  loop_tracks: dict[int, dict[int, Float64Array]],
  body_poses: list[gtsam.Pose3],
  lens: LensSetup,
  cfg: FrontendCfg,
  *,
  name: str,
  anchor_kfs: set[int] | None = None,
) -> dict[int, Float64Array]:
  """Triangulate revisit ``loop_tracks`` and merge them into the lens's
  landmarks (``points``) and tracks (``kept``). Mutates both.

  When ``anchor_kfs`` is given, each landmark is triangulated from its
  anchor-covisible views alone, while the full track still enters ``kept`` as
  reprojection factors."""
  if anchor_kfs:
    tri_tracks = {
      t: {k: v for k, v in obs.items() if k in anchor_kfs}
      for t, obs in loop_tracks.items()
    }
  else:
    tri_tracks = loop_tracks
  loop_pts = triangulate_lens(
    tri_tracks,
    body_poses,
    lens.extrinsic,
    lens.params,
    cfg.min_track_length,
    max_landmark_depth=cfg.max_landmark_depth,
  )
  points.update(loop_pts)
  kept.update(loop_tracks)
  print(
    f"  [{name}/loop] +{len(loop_pts)} landmarks from {len(loop_tracks)} revisit tracks"
  )
  return loop_pts
