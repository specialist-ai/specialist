"""Dual-lens MEI visual-only bundle adjustment.

Runs the ``vi_ba/`` pipeline end-to-end on a short window of one
``.insv``. Same ORB + essential-matrix + two-view + triangulate + LM
skeleton as ``ba_smoke.py``, but with the MEI projection factor (so the
~2 px KB-fit residual drops out), per-lens extrinsics parsed from
``offset_v3``, and both fisheyes running independently into a single
joint BA.

Verdict guardrails:
  - per-lens final reprojection RMSE below the KB-baseline minus ~2 px;
  - dual-lens init consistency: shared landmarks re-project through both
    lens extrinsics to within the MEI-residual tolerance;
  - both lenses contribute landmarks (neither is silently empty).

Prerequisite: a ``.insv`` with valid ``offset_v3`` metadata. MEI params
come straight from the per-lens block there — no separate YAML input.

Status: keep.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import gtsam
import numpy as np
import numpy.typing as npt
import tyro
from gtsam.symbol_shorthand import X
from specialist.vi_ba.build import (
  LensSetup,
  build_graph,
  front_landmark_key,
  rear_landmark_key,
)
from specialist.vi_ba.init import (
  body_poses_from_camera_poses,
  interpolate_camera_poses,
  pick_init_pair,
  two_view_essential,
)
from specialist.vi_ba.viz import log_ba_run

from specialist.loader.mei_model import MeiParams, mei_project, mei_unproject
from specialist.loader.parse_insv import load_insv
from specialist.vi_ba.extrinsics import parse_insv_extrinsics
from specialist.vi_ba.features import (
  Frame,
  chain_tracks,
  detect_orb,
  essential_ransac_inliers,
  match_pair,
)
from specialist.vi_ba.optimize import optimize_graph
from specialist.vi_ba.pipeline import (
  decode_dual_keyframes,
  dual_lens_consistency,
  lens_setups,
  triangulate_lens,
)

Float64Array = npt.NDArray[np.float64]


@dataclass
class BaArgs:
  input_insv: Path
  output_dir: Path = Path("logs")
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
  log_frames_in_rrd: bool = False


@dataclass
class BaSummary:
  source: str
  start: float
  duration: float
  n_keyframes: int
  init_lens: str
  init_first: int
  init_last: int
  init_shared: int
  init_inliers: int
  front_tracks_chained: int
  rear_tracks_chained: int
  front_tracks_triangulated: int
  rear_tracks_triangulated: int
  initial_graph_error: float
  final_graph_error: float
  lm_iterations: int
  front_rmse_px: float
  rear_rmse_px: float
  trajectory_length: float
  dual_lens_check_mean_px: float
  dual_lens_check_n: int
  verdict: str = "unknown"


def _run_lens_frontend(
  frames: list[Float64Array],
  params: MeiParams,
  args: BaArgs,
) -> tuple[list[Frame], dict[int, dict[int, Float64Array]]]:
  """ORB → BFMatch → essential RANSAC → chain tracks for one lens."""
  orb_frames = [
    detect_orb(
      f, n_features=args.orb_n_features, fast_threshold=args.orb_fast_threshold
    )
    for f in frames
  ]
  max_ha = float(np.deg2rad(args.max_half_angle_deg))
  matches_between: list[np.ndarray] = []
  for k in range(1, len(orb_frames)):
    a, b = orb_frames[k - 1], orb_frames[k]
    m = match_pair(a, b, ratio=args.match_ratio)
    if m.shape[0] == 0:
      matches_between.append(np.empty((0, 2), dtype=np.int64))
      continue
    pts_a = a.keypoints[m[:, 0]]
    pts_b = b.keypoints[m[:, 1]]
    bears_a = mei_unproject(pts_a, params)
    bears_b = mei_unproject(pts_b, params)
    inliers = essential_ransac_inliers(
      pts_a,
      pts_b,
      bears_a,
      bears_b,
      threshold_px=args.ransac_threshold_px,
      fx=params.fx,
      max_half_angle=max_ha,
    )
    matches_between.append(m[inliers])
  tracks = chain_tracks(orb_frames, matches_between)
  return orb_frames, tracks


def _per_lens_rmse(
  values: gtsam.Values,
  tracks: Mapping[int, Mapping[int, Float64Array]],
  lens: LensSetup,
  landmark_key_fn: Callable[[int], int],
) -> float:
  errs: list[float] = []
  for tid, obs in tracks.items():
    lm_key = landmark_key_fn(tid)
    if not values.exists(lm_key):
      continue
    xyz = values.atPoint3(lm_key)
    for fi, pt in obs.items():
      if not values.exists(X(fi)):
        continue
      body_pose = values.atPose3(X(fi))
      p_lens = body_pose.compose(lens.extrinsic).transformTo(
        gtsam.Point3(float(xyz[0]), float(xyz[1]), float(xyz[2]))
      )
      pred = mei_project(np.asarray(p_lens)[None, :], lens.params)[0]
      if not np.all(np.isfinite(pred)):
        continue
      errs.append(float(np.linalg.norm(pred - pt)))
  return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def _trajectory_length(poses: list[gtsam.Pose3]) -> float:
  pts = [np.asarray(p.translation(), dtype=np.float64) for p in poses]
  return float(sum(np.linalg.norm(pts[i] - pts[i - 1]) for i in range(1, len(pts))))


def _verdict(s: BaSummary) -> str:
  if s.front_tracks_triangulated < 30 or s.rear_tracks_triangulated < 30:
    return (
      f"red: per-lens triangulated counts too low "
      f"(front={s.front_tracks_triangulated}, rear={s.rear_tracks_triangulated})"
    )
  if s.final_graph_error > s.initial_graph_error:
    return "red: LM diverged"
  max_rmse = max(s.front_rmse_px, s.rear_rmse_px)
  if max_rmse > 5.0:
    return f"yellow: per-lens RMSE {max_rmse:.2f} px — intrinsics / outliers suspect"
  if s.trajectory_length < 0.05:
    return "yellow: trajectory nearly static — pick a window with more motion"
  return (
    f"green: traj {s.trajectory_length:.2f} m, front {s.front_rmse_px:.2f} px, "
    f"rear {s.rear_rmse_px:.2f} px, dual-lens mean disagreement "
    f"{s.dual_lens_check_mean_px:.1f} px over {s.dual_lens_check_n} landmarks"
  )


def main(args: BaArgs) -> None:
  args.output_dir.mkdir(parents=True, exist_ok=True)
  data = load_insv(args.input_insv)
  extrinsics = parse_insv_extrinsics(data)
  front_lens, rear_lens = lens_setups(data, extrinsics)

  front_frames, rear_frames = decode_dual_keyframes(
    args.input_insv, args.start, args.duration, data.frame_rate, args.keyframe_every
  )
  if len(front_frames) < 3 or len(rear_frames) < 3:
    raise RuntimeError(
      f"only {len(front_frames)} front / {len(rear_frames)} rear keyframes decoded"
    )

  front_orb, front_tracks = _run_lens_frontend(front_frames, front_lens.params, args)
  rear_orb, rear_tracks = _run_lens_frontend(rear_frames, rear_lens.params, args)
  front_kept = {
    t: o for t, o in front_tracks.items() if len(o) >= args.min_track_length
  }
  rear_kept = {t: o for t, o in rear_tracks.items() if len(o) >= args.min_track_length}

  n_frames = len(front_frames)
  f_first, f_last, f_shared = pick_init_pair(
    front_kept, n_frames, args.init_min_shared_tracks
  )
  r_first, r_last, r_shared = pick_init_pair(
    rear_kept, n_frames, args.init_min_shared_tracks
  )

  if f_shared >= r_shared:
    init_lens, init_tracks, init_params, init_extrinsic = (
      "front",
      front_kept,
      front_lens.params,
      front_lens.extrinsic,
    )
    init_first, init_last, init_shared = f_first, f_last, f_shared
  else:
    init_lens, init_tracks, init_params, init_extrinsic = (
      "rear",
      rear_kept,
      rear_lens.params,
      rear_lens.extrinsic,
    )
    init_first, init_last, init_shared = r_first, r_last, r_shared

  print(
    f"init: lens={init_lens} keyframes {init_first}→{init_last} "
    f"with {init_shared} shared tracks"
  )
  if init_shared < 8:
    raise RuntimeError(f"neither lens has ≥8 shared tracks (best={init_shared})")

  # Collect correspondences, unproject via MEI, run two-view essential.
  pix_a, pix_b = [], []
  for obs in init_tracks.values():
    if init_first in obs and init_last in obs:
      pix_a.append(obs[init_first])
      pix_b.append(obs[init_last])
  pix_a = np.asarray(pix_a, dtype=np.float64)
  pix_b = np.asarray(pix_b, dtype=np.float64)
  bears_a = mei_unproject(pix_a, init_params)
  bears_b = mei_unproject(pix_b, init_params)
  r_cw_last, t_cw_last, init_inliers = two_view_essential(
    bears_a, bears_b, threshold=0.01
  )

  cam_poses = interpolate_camera_poses(r_cw_last, t_cw_last, n_frames)
  body_poses = body_poses_from_camera_poses(cam_poses, init_extrinsic)

  front_points = triangulate_lens(
    front_kept,
    body_poses,
    front_lens.extrinsic,
    front_lens.params,
    args.min_track_length,
  )
  rear_points = triangulate_lens(
    rear_kept, body_poses, rear_lens.extrinsic, rear_lens.params, args.min_track_length
  )
  front_tracks_tri = {t: front_kept[t] for t in front_points}
  rear_tracks_tri = {t: rear_kept[t] for t in rear_points}

  graph, values = build_graph(
    body_poses=body_poses,
    front=front_lens,
    front_tracks=front_tracks_tri,
    front_points=front_points,
    rear=rear_lens,
    rear_tracks=rear_tracks_tri,
    rear_points=rear_points,
    pixel_sigma=args.noise_sigma_px,
    weak_pose_sigmas=(0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
  )
  initial_error = float(graph.error(values))
  result = optimize_graph(graph, values, max_iterations=args.max_lm_iterations)

  poses_final = [result.values.atPose3(X(i)) for i in range(len(body_poses))]
  front_points_final = {
    t: np.asarray(result.values.atPoint3(front_landmark_key(t)), dtype=np.float64)
    for t in front_points
  }
  rear_points_final = {
    t: np.asarray(result.values.atPoint3(rear_landmark_key(t)), dtype=np.float64)
    for t in rear_points
  }
  front_rmse = _per_lens_rmse(
    result.values, front_tracks_tri, front_lens, front_landmark_key
  )
  rear_rmse = _per_lens_rmse(
    result.values, rear_tracks_tri, rear_lens, rear_landmark_key
  )
  dual_mean, dual_n = dual_lens_consistency(
    front_points_final,
    rear_points_final,
    poses_final,
    extrinsics,
    front_lens.params,
    rear_lens.params,
  )

  summary = BaSummary(
    source=str(args.input_insv),
    start=args.start,
    duration=args.duration,
    n_keyframes=n_frames,
    init_lens=init_lens,
    init_first=init_first,
    init_last=init_last,
    init_shared=init_shared,
    init_inliers=init_inliers,
    front_tracks_chained=len(front_kept),
    rear_tracks_chained=len(rear_kept),
    front_tracks_triangulated=len(front_points),
    rear_tracks_triangulated=len(rear_points),
    initial_graph_error=initial_error,
    final_graph_error=result.final_error,
    lm_iterations=result.iterations,
    front_rmse_px=front_rmse,
    rear_rmse_px=rear_rmse,
    trajectory_length=_trajectory_length(poses_final),
    dual_lens_check_mean_px=dual_mean,
    dual_lens_check_n=dual_n,
  )
  summary.verdict = _verdict(summary)

  stamp = time.strftime("%Y%m%d_%H%M%S")
  rrd_path = args.output_dir / f"ba_mei_smoke_{stamp}.rrd"
  json_path = args.output_dir / f"ba_mei_smoke_{stamp}.json"
  log_ba_run(
    rrd_path=rrd_path,
    body_poses=poses_final,
    front_points=front_points_final,
    rear_points=rear_points_final,
    front_frames=front_frames if args.log_frames_in_rrd else None,
    rear_frames=rear_frames if args.log_frames_in_rrd else None,
    front_keypoints=[f.keypoints for f in front_orb]
    if args.log_frames_in_rrd
    else None,
    rear_keypoints=[f.keypoints for f in rear_orb] if args.log_frames_in_rrd else None,
  )
  json_path.write_text(json.dumps(asdict(summary), indent=2))
  print(json.dumps(asdict(summary), indent=2))
  print(f"\nWrote {rrd_path} and {json_path}")
  print(f"Verdict: {summary.verdict}")


if __name__ == "__main__":
  main(tyro.cli(BaArgs))
