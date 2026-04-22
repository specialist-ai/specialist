"""Visual-only bundle adjustment smoke test.

Verifies that ORB detection + descriptor matching + two-view init +
triangulation + GTSAM bundle adjustment produces a coherent up-to-scale
trajectory on raw Insta360 X3 fisheye frames, before we commit to the full
VI-BA. Uses ``gtsam.Cal3Fisheye`` (which is Kannala-Brandt) plus
``gtsam.GenericProjectionFactorCal3Fisheye`` with the KB calibration that
``camera_model.py`` produced — that way we exercise the full geometric
pipeline today and can swap in a custom MEI projection factor later.

Pipeline:
  1. Decode N keyframes from one fisheye half over a short window.
  2. ORB detect + BFMatcher + ratio test between adjacent keyframes.
  3. Essential-matrix RANSAC on KB-unprojected bearings for outlier rejection.
  4. Chain surviving pairwise matches into per-landmark tracks.
  5. Two-view init on (first, last) keyframe via essential matrix; SLERP +
     linear translation interpolation for intermediate keyframes.
  6. Triangulate each track from its earliest and latest observation.
  7. Build GTSAM factor graph: Pose3 per keyframe, Point3 per landmark,
     GenericProjectionFactorCal3Fisheye per observation, tight PriorFactor
     on keyframes 0 and N-1 to fix the gauge (first pose + absorb scale).
  8. Optimise with Levenberg-Marquardt; log trajectory + landmarks to Rerun.

Prerequisite: ``scripts/camera_model.py`` has produced
``logs/kb_calib_<stem>.yaml`` for this X3 unit.

Status: experimental — gating probe before investing in full VI-BA.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import gtsam
import numpy as np
import numpy.typing as npt
import rerun as rr
import tyro
import yaml
from gtsam.symbol_shorthand import L, X
from scipy.spatial.transform import Rotation as Rscp
from scipy.spatial.transform import Slerp

from specialist.loader.kb_model import KbParams, kb_unproject
from specialist.loader.parse_insv import load_insv

Float64Array = npt.NDArray[np.float64]


@dataclass
class BaArgs:
  """CLI arguments for the visual-BA smoke probe."""

  input_insv: Path
  kb_yaml: Path
  output_dir: Path = Path("logs")
  lens: str = "front"  # 'front' (Lens 1, left half) or 'rear' (Lens 2, right half)
  start: float = 0.0
  duration: float = 10.0
  keyframe_every: int = 3
  orb_n_features: int = 2000
  orb_fast_threshold: int = 12
  match_ratio: float = 0.7
  ransac_threshold_px: float = 3.0
  min_track_length: int = 3
  max_half_angle_deg: float = 70.0
  noise_sigma_px: float = 1.5
  max_lm_iterations: int = 50
  init_min_shared_tracks: int = 60


@dataclass
class BaSummary:
  source: str
  lens: str
  start: float
  duration: float
  n_keyframes: int
  n_tracks_chained: int
  n_tracks_triangulated: int
  mean_track_length: float
  init_pair_inliers: int
  initial_graph_error: float
  final_graph_error: float
  lm_iterations: int
  final_rmse_px: float
  trajectory_length: float
  verdict: str = "unknown"


def _load_kb(yaml_path: Path) -> KbParams:
  d = yaml.safe_load(yaml_path.read_text())
  return KbParams(
    fx=float(d["Camera1.fx"]),
    fy=float(d["Camera1.fy"]),
    cx=float(d["Camera1.cx"]),
    cy=float(d["Camera1.cy"]),
    k1=float(d["Camera1.k1"]),
    k2=float(d["Camera1.k2"]),
    k3=float(d["Camera1.k3"]),
    k4=float(d["Camera1.k4"]),
  )


def _decode_keyframes(
  video_path: Path,
  lens: str,
  start: float,
  duration: float,
  fps: float,
  keyframe_every: int,
) -> list[Float64Array]:
  if lens not in ("front", "rear"):
    raise ValueError(f"lens must be 'front' or 'rear', got {lens!r}")
  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    raise RuntimeError(f"OpenCV cannot open {video_path}")
  start_frame = int(round(start * fps))
  total_frames = int(round(duration * fps))
  cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  half = width // 2
  frames: list[Float64Array] = []
  for i in range(total_frames):
    ok, frame = cap.read()
    if not ok:
      break
    if i % keyframe_every == 0:
      crop = frame[:, :half] if lens == "front" else frame[:, half:]
      frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
  cap.release()
  return frames


def _extract_orb(
  frames: list[Float64Array], n_features: int, fast_threshold: int
) -> tuple[list[Float64Array], list[np.ndarray]]:
  orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=fast_threshold)
  pts: list[Float64Array] = []
  descs: list[np.ndarray] = []
  for f in frames:
    kps, d = orb.detectAndCompute(f, None)
    if kps is None or len(kps) == 0 or d is None:
      pts.append(np.empty((0, 2), dtype=np.float64))
      descs.append(np.empty((0, 32), dtype=np.uint8))
      continue
    pts.append(np.asarray([k.pt for k in kps], dtype=np.float64))
    descs.append(d)
  return pts, descs


def _match_pair(desc_a: np.ndarray, desc_b: np.ndarray, ratio: float) -> np.ndarray:
  if desc_a.size == 0 or desc_b.size == 0:
    return np.empty((0, 2), dtype=np.int64)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  knn = bf.knnMatch(desc_a, desc_b, k=2)
  good: list[tuple[int, int]] = []
  for pair in knn:
    if len(pair) < 2:
      continue
    a, b = pair
    if a.distance < ratio * b.distance:
      good.append((a.queryIdx, a.trainIdx))
  return np.asarray(good, dtype=np.int64) if good else np.empty((0, 2), dtype=np.int64)


def _essential_ransac_inliers(
  pts_a: Float64Array,
  pts_b: Float64Array,
  kb: KbParams,
  max_half_angle: float,
  threshold_px: float,
) -> np.ndarray:
  """RANSAC essential-matrix inliers on KB-unprojected bearings."""
  n = pts_a.shape[0]
  if n < 8:
    return np.zeros(n, dtype=bool)
  ba = kb_unproject(pts_a, kb)
  bb = kb_unproject(pts_b, kb)
  cos_min = float(np.cos(max_half_angle))
  in_fov = (ba[:, 2] > cos_min) & (bb[:, 2] > cos_min)
  if in_fov.sum() < 8:
    return np.zeros(n, dtype=bool)
  norm_a = (ba[in_fov][:, :2] / ba[in_fov][:, 2:3]).astype(np.float64)
  norm_b = (bb[in_fov][:, :2] / bb[in_fov][:, 2:3]).astype(np.float64)
  # Threshold expressed in normalised-image units (divide pixel threshold by fx).
  thresh = float(threshold_px / kb.fx)
  E, mask = cv2.findEssentialMat(
    norm_a,
    norm_b,
    focal=1.0,
    pp=(0.0, 0.0),
    method=cv2.RANSAC,
    prob=0.999,
    threshold=thresh,
  )
  full_mask = np.zeros(n, dtype=bool)
  if E is None or mask is None:
    return full_mask
  in_fov_idx = np.where(in_fov)[0]
  mask_flat = mask.flatten().astype(bool)
  for i, is_inlier in enumerate(mask_flat):
    if is_inlier:
      full_mask[in_fov_idx[i]] = True
  return full_mask


def _chain_tracks(
  kpts: list[Float64Array],
  descs: list[np.ndarray],
  kb: KbParams,
  args: BaArgs,
) -> tuple[dict[int, dict[int, Float64Array]], int]:
  """Return ({track_id: {frame_idx: pt2d}}, total_inlier_matches)."""
  max_ha = float(np.deg2rad(args.max_half_angle_deg))
  tracks: dict[int, dict[int, Float64Array]] = {}
  lookup: dict[tuple[int, int], int] = {}
  next_id = 0
  total_inliers = 0
  for k in range(1, len(kpts)):
    if kpts[k].shape[0] == 0 or kpts[k - 1].shape[0] == 0:
      continue
    matches = _match_pair(descs[k - 1], descs[k], args.match_ratio)
    if matches.shape[0] == 0:
      continue
    inliers = _essential_ransac_inliers(
      kpts[k - 1][matches[:, 0]],
      kpts[k][matches[:, 1]],
      kb,
      max_ha,
      args.ransac_threshold_px,
    )
    good = matches[inliers]
    total_inliers += int(good.shape[0])
    for a_idx, b_idx in good:
      prev_key = (k - 1, int(a_idx))
      if prev_key in lookup:
        tid = lookup[prev_key]
      else:
        tid = next_id
        next_id += 1
        tracks[tid] = {k - 1: kpts[k - 1][a_idx]}
        lookup[prev_key] = tid
      tracks[tid][k] = kpts[k][b_idx]
      lookup[(k, int(b_idx))] = tid
  return tracks, total_inliers


def _pick_init_pair(
  tracks: dict[int, dict[int, Float64Array]],
  n_frames: int,
  min_shared: int,
) -> tuple[int, int, int]:
  """Pick the widest (first=0, last=K) pair that still has ≥min_shared tracks.

  Returns (first, last, n_shared). Scans K from n_frames-1 down to 1 and
  returns the first K that meets the threshold.
  """
  # Bucket tracks by which of the 0 vs K constraint they satisfy.
  first = 0
  for last in range(n_frames - 1, 0, -1):
    shared = sum(1 for obs in tracks.values() if first in obs and last in obs)
    if shared >= min_shared:
      return first, last, shared
  # Fall back to the pair with the most shared observations, even if < threshold.
  best_last, best_shared = 1, 0
  for last in range(1, n_frames):
    shared = sum(1 for obs in tracks.values() if first in obs and last in obs)
    if shared > best_shared:
      best_last, best_shared = last, shared
  return first, best_last, best_shared


def _two_view_init(
  tracks: dict[int, dict[int, Float64Array]],
  kb: KbParams,
  first: int,
  last: int,
  max_half_angle: float,
) -> tuple[Float64Array, Float64Array, int]:
  """Essential-matrix init between two keyframes; returns (R_cw_last, t_cw_last, inliers)."""
  pts0: list[Float64Array] = []
  pts1: list[Float64Array] = []
  for _, obs in tracks.items():
    if first in obs and last in obs:
      pts0.append(obs[first])
      pts1.append(obs[last])
  if len(pts0) < 8:
    raise RuntimeError(
      f"two-view init needs ≥8 shared observations between keyframes "
      f"{first} and {last}; got {len(pts0)}. Increase --duration or "
      f"shrink --keyframe-every."
    )
  p0 = np.asarray(pts0, dtype=np.float64)
  p1 = np.asarray(pts1, dtype=np.float64)
  b0 = kb_unproject(p0, kb)
  b1 = kb_unproject(p1, kb)
  cos_min = float(np.cos(max_half_angle))
  ok = (b0[:, 2] > cos_min) & (b1[:, 2] > cos_min)
  if ok.sum() < 8:
    raise RuntimeError(
      f"only {int(ok.sum())} in-FOV correspondences survived filtering; "
      f"relax --max-half-angle-deg or pick a different window"
    )
  norm_0 = (b0[ok][:, :2] / b0[ok][:, 2:3]).astype(np.float64)
  norm_1 = (b1[ok][:, :2] / b1[ok][:, 2:3]).astype(np.float64)
  E, _ = cv2.findEssentialMat(
    norm_0,
    norm_1,
    focal=1.0,
    pp=(0.0, 0.0),
    method=cv2.RANSAC,
    prob=0.999,
    threshold=0.01,
  )
  if E is None:
    raise RuntimeError("essential matrix failed on the init pair")
  n_inliers, R, t, _ = cv2.recoverPose(E, norm_0, norm_1)
  return R.astype(np.float64), t.flatten().astype(np.float64), int(n_inliers)


def _interpolate_poses(
  R_cw_last: Float64Array, t_cw_last: Float64Array, n_frames: int
) -> list[gtsam.Pose3]:
  """Generate body-to-world poses via SLERP (rotation) + linear (translation).

  ``R_cw_last, t_cw_last`` are the world→camera transform for the last keyframe
  when keyframe 0 sits at the origin; we invert to body-to-world for GTSAM.
  """
  key_rots = Rscp.from_matrix(np.stack([np.eye(3), R_cw_last]))
  slerp = Slerp([0.0, 1.0], key_rots)
  out: list[gtsam.Pose3] = []
  for i in range(n_frames):
    alpha = i / max(n_frames - 1, 1)
    R_cw = slerp([alpha])[0].as_matrix()
    t_cw = alpha * t_cw_last
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    out.append(gtsam.Pose3(gtsam.Rot3(R_wc), gtsam.Point3(*t_wc)))
  return out


def _pose_to_world_from_camera(pose_body_to_world: gtsam.Pose3) -> Float64Array:
  """3x4 world-to-camera matrix in normalised-image convention (for cv2.triangulatePoints)."""
  T_wb = pose_body_to_world.matrix()
  T_bw = np.linalg.inv(T_wb)
  return T_bw[:3, :4].astype(np.float64)


def _triangulate_tracks(
  tracks: dict[int, dict[int, Float64Array]],
  poses: list[gtsam.Pose3],
  kb: KbParams,
  min_track_length: int,
) -> dict[int, Float64Array]:
  """Per-track DLT triangulation from earliest + latest observation."""
  points: dict[int, Float64Array] = {}
  for tid, obs in tracks.items():
    frames = sorted(obs.keys())
    if len(frames) < min_track_length:
      continue
    f0, f1 = frames[0], frames[-1]
    bearing0 = kb_unproject(obs[f0][None, :], kb)[0]
    bearing1 = kb_unproject(obs[f1][None, :], kb)[0]
    if bearing0[2] < 0.1 or bearing1[2] < 0.1:
      continue
    norm_0 = bearing0[:2] / bearing0[2]
    norm_1 = bearing1[:2] / bearing1[2]
    P0 = _pose_to_world_from_camera(poses[f0])
    P1 = _pose_to_world_from_camera(poses[f1])
    tri = cv2.triangulatePoints(
      P0,
      P1,
      np.asarray(norm_0, dtype=np.float64).reshape(2, 1),
      np.asarray(norm_1, dtype=np.float64).reshape(2, 1),
    )
    if tri[3, 0] == 0.0:
      continue
    xyz = (tri[:3, 0] / tri[3, 0]).astype(np.float64)
    if not np.all(np.isfinite(xyz)):
      continue
    # Cheirality: reject if behind either camera.
    point_cam0 = poses[f0].transformTo(gtsam.Point3(*xyz))
    point_cam1 = poses[f1].transformTo(gtsam.Point3(*xyz))
    if float(np.asarray(point_cam0)[2]) < 0.05:
      continue
    if float(np.asarray(point_cam1)[2]) < 0.05:
      continue
    points[tid] = xyz
  return points


def _build_graph(
  poses: list[gtsam.Pose3],
  points: dict[int, Float64Array],
  tracks: dict[int, dict[int, Float64Array]],
  kb: KbParams,
  args: BaArgs,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  graph = gtsam.NonlinearFactorGraph()
  values = gtsam.Values()

  cal = gtsam.Cal3Fisheye(kb.fx, kb.fy, 0.0, kb.cx, kb.cy, kb.k1, kb.k2, kb.k3, kb.k4)
  pixel_noise = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(1.345),
    gtsam.noiseModel.Isotropic.Sigma(2, args.noise_sigma_px),
  )
  # Tight priors on first + last pose to fix the 6-DOF + scale gauge.
  gauge = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4], dtype=np.float64)
  )
  graph.push_back(gtsam.PriorFactorPose3(X(0), poses[0], gauge))
  graph.push_back(gtsam.PriorFactorPose3(X(len(poses) - 1), poses[-1], gauge))

  # Weak priors on every other pose so intermediate keyframes with few/no
  # observations stay anchored to their interpolated init — prevents
  # "leftover key" elimination failures when fast motion starves some frames.
  weak = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float64)
  )
  for i in range(1, len(poses) - 1):
    graph.push_back(gtsam.PriorFactorPose3(X(i), poses[i], weak))

  for i, pose in enumerate(poses):
    values.insert(X(i), pose)
  for tid, xyz in points.items():
    values.insert(L(tid), gtsam.Point3(*xyz))
    for fi, pt in tracks[tid].items():
      graph.push_back(
        gtsam.GenericProjectionFactorCal3Fisheye(
          gtsam.Point2(float(pt[0]), float(pt[1])),
          pixel_noise,
          X(fi),
          L(tid),
          cal,
        )
      )
  return graph, values


def _final_rmse(
  values: gtsam.Values,
  tracks: dict[int, dict[int, Float64Array]],
  kb: KbParams,
) -> float:
  cal = gtsam.Cal3Fisheye(kb.fx, kb.fy, 0.0, kb.cx, kb.cy, kb.k1, kb.k2, kb.k3, kb.k4)
  errs: list[float] = []
  for tid, obs in tracks.items():
    if not values.exists(L(tid)):
      continue
    p_world = values.atPoint3(L(tid))
    for fi, pt in obs.items():
      if not values.exists(X(fi)):
        continue
      cam = gtsam.PinholeCameraCal3Fisheye(values.atPose3(X(fi)), cal)
      try:
        predicted = cam.project(p_world)
      except Exception:
        continue
      errs.append(float(np.linalg.norm(np.asarray(predicted, dtype=np.float64) - pt)))
  return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def _trajectory_length(poses: list[gtsam.Pose3]) -> float:
  pts = [np.asarray(p.translation(), dtype=np.float64) for p in poses]
  return float(sum(np.linalg.norm(pts[i] - pts[i - 1]) for i in range(1, len(pts))))


def _log_to_rerun(
  poses: list[gtsam.Pose3],
  points_world: dict[int, Float64Array],
) -> None:
  rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
  if points_world:
    xyz = np.asarray(list(points_world.values()), dtype=np.float64)
    rr.log(
      "world/points",
      rr.Points3D(xyz, radii=0.005, colors=[80, 200, 80]),
      static=True,
    )
  traj = np.asarray([np.asarray(p.translation(), dtype=np.float64) for p in poses])
  rr.log(
    "world/trajectory",
    rr.LineStrips3D([traj], colors=[80, 130, 230], radii=0.003),
    static=True,
  )
  for i, pose in enumerate(poses):
    rr.set_time("keyframe", sequence=i)
    q = pose.rotation().toQuaternion()
    rr.log(
      "world/camera",
      rr.Transform3D(
        translation=np.asarray(pose.translation(), dtype=np.float64),
        quaternion=rr.Quaternion(xyzw=[q.x(), q.y(), q.z(), q.w()]),
      ),
    )


def _verdict(summary: BaSummary) -> str:
  if summary.n_tracks_triangulated < 50:
    return (
      f"red: only {summary.n_tracks_triangulated} tracks triangulated — "
      f"feature matching too weak for BA"
    )
  if summary.final_graph_error > summary.initial_graph_error:
    return "red: BA diverged (final graph error > initial)"
  if summary.final_rmse_px > 5.0:
    return (
      f"yellow: BA converged but reprojection RMSE "
      f"{summary.final_rmse_px:.2f} px — intrinsics / outliers suspect"
    )
  if summary.trajectory_length < 0.05:
    return (
      "yellow: BA converged but trajectory is nearly static — try a window "
      "with more camera motion"
    )
  return (
    f"green: up-to-scale trajectory {summary.trajectory_length:.2f} m, "
    f"{summary.n_tracks_triangulated} landmarks, RMSE "
    f"{summary.final_rmse_px:.2f} px"
  )


def main(args: BaArgs) -> None:
  args.output_dir.mkdir(parents=True, exist_ok=True)
  data = load_insv(args.input_insv)
  kb = _load_kb(args.kb_yaml)

  frames = _decode_keyframes(
    args.input_insv,
    args.lens,
    args.start,
    args.duration,
    data.frame_rate,
    args.keyframe_every,
  )
  if len(frames) < 3:
    raise RuntimeError(f"only {len(frames)} keyframes decoded — window too short")

  kpts, descs = _extract_orb(frames, args.orb_n_features, args.orb_fast_threshold)
  tracks, _ = _chain_tracks(kpts, descs, kb, args)
  tracks_kept = {
    tid: obs for tid, obs in tracks.items() if len(obs) >= args.min_track_length
  }
  if not tracks_kept:
    raise RuntimeError("no tracks survived chaining — feature matching failed")

  # Trim to the keyframe range actually constrained by tracks. On fast-motion
  # captures tracks die out quickly and a BA over unconstrained later keyframes
  # is ill-posed.
  max_kf_with_tracks = max(max(obs.keys()) for obs in tracks_kept.values())
  frames_trimmed = frames[: max_kf_with_tracks + 1]
  tracks_trimmed = {
    tid: {fi: pt for fi, pt in obs.items() if fi <= max_kf_with_tracks}
    for tid, obs in tracks_kept.items()
  }
  tracks_trimmed = {
    tid: obs for tid, obs in tracks_trimmed.items() if len(obs) >= args.min_track_length
  }
  if len(frames_trimmed) < 3:
    raise RuntimeError(
      f"only {len(frames_trimmed)} keyframes have tracks — feature matching "
      f"too fragile, try larger --orb-n-features or smaller --keyframe-every"
    )
  if len(frames_trimmed) < len(frames):
    print(
      f"tracks die out after keyframe {max_kf_with_tracks}; BA restricted to "
      f"{len(frames_trimmed)}/{len(frames)} keyframes"
    )

  init_first, init_last, init_shared = _pick_init_pair(
    tracks_trimmed, len(frames_trimmed), args.init_min_shared_tracks
  )
  if init_shared < 8:
    raise RuntimeError(
      f"no (0, K) keyframe pair has ≥8 shared tracks (best: {init_shared}). "
      f"Shorten --keyframe-every or pick a more feature-rich window."
    )
  print(
    f"init pair: keyframes {init_first}→{init_last} with {init_shared} shared tracks"
  )
  R_cw_last, t_cw_last, init_inliers = _two_view_init(
    tracks_trimmed,
    kb,
    first=init_first,
    last=init_last,
    max_half_angle=float(np.deg2rad(args.max_half_angle_deg)),
  )
  poses_init = _interpolate_poses(R_cw_last, t_cw_last, len(frames_trimmed))
  points_init = _triangulate_tracks(
    tracks_trimmed, poses_init, kb, args.min_track_length
  )
  tracks_tri = {tid: tracks_trimmed[tid] for tid in points_init}

  graph, values = _build_graph(poses_init, points_init, tracks_tri, kb, args)
  initial_error = float(graph.error(values))

  params = gtsam.LevenbergMarquardtParams()
  params.setMaxIterations(args.max_lm_iterations)
  optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
  result = optimizer.optimize()

  final_error = float(graph.error(result))
  iterations = int(optimizer.iterations())

  poses_final = [result.atPose3(X(i)) for i in range(len(poses_init))]
  points_final = {
    tid: np.asarray(result.atPoint3(L(tid)), dtype=np.float64) for tid in points_init
  }
  rmse = _final_rmse(result, tracks_tri, kb)
  traj_len = _trajectory_length(poses_final)
  mean_track_len = (
    float(np.mean([len(obs) for obs in tracks_tri.values()])) if tracks_tri else 0.0
  )

  summary = BaSummary(
    source=str(args.input_insv),
    lens=args.lens,
    start=args.start,
    duration=args.duration,
    n_keyframes=len(frames_trimmed),
    n_tracks_chained=len(tracks_trimmed),
    n_tracks_triangulated=len(tracks_tri),
    mean_track_length=mean_track_len,
    init_pair_inliers=init_inliers,
    initial_graph_error=initial_error,
    final_graph_error=final_error,
    lm_iterations=iterations,
    final_rmse_px=rmse,
    trajectory_length=traj_len,
  )
  summary.verdict = _verdict(summary)

  stamp = time.strftime("%Y%m%d_%H%M%S")
  rrd_path = args.output_dir / f"ba_smoke_{stamp}.rrd"
  json_path = args.output_dir / f"ba_smoke_{stamp}.json"
  rr.init("specialist.ba_smoke", spawn=False)
  rr.save(str(rrd_path))
  _log_to_rerun(poses_final, points_final)

  json_path.write_text(json.dumps(asdict(summary), indent=2))
  print(json.dumps(asdict(summary), indent=2))
  print(f"\nWrote {rrd_path} and {json_path}")
  print(f"Verdict: {summary.verdict}")


if __name__ == "__main__":
  main(tyro.cli(BaArgs))
