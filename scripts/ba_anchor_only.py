"""Anchor-only baseline: AprilTag PnP → ``world_T_body`` per keyframe,
logged to Rerun. No visual tracks, no IMU, no LM. The world frame is
pinned to the anchor tag so the trajectory is whatever the hand did
relative to it."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from specialist.loader.parse_insv import load_insv
from specialist.vi_ba.anchor_detection import anchor_pose_from_tags, detect_all_tags
from specialist.vi_ba.extrinsics import parse_insv_extrinsics
from specialist.vi_ba.pipeline import (
  decode_dual_keyframes,
  keyframe_timestamps,
  scaled_mei,
)
from specialist.vi_ba.viz import log_ba_run


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--input-insv", type=Path, required=True)
  ap.add_argument("--start", type=float, default=0.0)
  ap.add_argument("--duration", type=float, default=None)
  ap.add_argument("--keyframe-every", type=int, default=9)
  ap.add_argument("--anchor-tag-id", type=int, default=0)
  ap.add_argument("--anchor-tag-size", type=float, default=0.1)
  ap.add_argument("--anchor-lens", choices=["front", "rear"], default="rear")
  ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
  args = ap.parse_args()

  args.logs_dir.mkdir(parents=True, exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  rrd_path = args.logs_dir / f"ba_anchor_only_{ts}.rrd"
  json_path = args.logs_dir / f"ba_anchor_only_{ts}.json"

  data = load_insv(args.input_insv)
  extrinsics = parse_insv_extrinsics(data)
  fps = data.frame_rate

  print(f"[anchor-only] decoding keyframes from {args.input_insv.name}")
  t0 = time.time()
  front_frames, rear_frames = decode_dual_keyframes(
    args.input_insv, args.start, args.duration, fps, args.keyframe_every
  )
  print(
    f"[anchor-only] decoded {len(rear_frames)} keyframes in {time.time() - t0:.1f}s"
  )

  if args.anchor_lens == "rear":
    frames = rear_frames
    body_T_lens = extrinsics.rear_T_body_lens
    mei = scaled_mei(extrinsics.rear, data.frame_width, lens_idx=1)
  else:
    frames = front_frames
    body_T_lens = extrinsics.front_T_body_lens
    mei = scaled_mei(extrinsics.front, data.frame_width, lens_idx=0)

  print(f"[anchor-only] detecting tags on {args.anchor_lens} lens ({len(frames)} kf)")
  t0 = time.time()
  per_kf = detect_all_tags(frames)
  print(f"[anchor-only] tag detection done in {time.time() - t0:.1f}s")

  cam_T_tag_per_kf = anchor_pose_from_tags(
    per_kf, args.anchor_tag_id, args.anchor_tag_size, mei
  )
  n_visible = sum(
    1 for dets in per_kf if any(d.tag_id == args.anchor_tag_id for d in dets)
  )
  kept_idx = sorted(cam_T_tag_per_kf.keys())
  poses_world_body = [
    body_T_lens.compose(cam_T_tag_per_kf[kf]).inverse() for kf in kept_idx
  ]
  per_kf_overlay = [
    [(d.tag_id, d.center, d.corners) for d in per_kf[kf]] for kf in kept_idx
  ]

  if not poses_world_body:
    print(
      f"[anchor-only] tag {args.anchor_tag_id} not detected in any keyframe — "
      "nothing to log"
    )
    return

  kf_times_all = keyframe_timestamps(args.start, fps, args.keyframe_every, len(frames))
  kept_times = kf_times_all[kept_idx]
  kept_frames = [frames[i] for i in kept_idx]

  traj = np.array([p.translation() for p in poses_world_body], dtype=np.float64)
  bbox_diag = float(np.linalg.norm(traj.max(axis=0) - traj.min(axis=0)))
  path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
  start_to_end = float(np.linalg.norm(traj[-1] - traj[0]))
  first_kept = kept_idx[0]
  last_kept = kept_idx[-1]

  print(
    f"[anchor-only] kept {len(poses_world_body)} / {len(frames)} keyframes "
    f"({n_visible} with tag visible; first kf {first_kept}, last kf {last_kept})"
  )
  print(
    f"[anchor-only] bbox_diag={bbox_diag:.3f} path_len={path_len:.3f} "
    f"start_to_end={start_to_end:.3f}"
  )

  if args.anchor_lens == "rear":
    front_kw: dict = {}
    rear_kw: dict = {
      "rear_frames": kept_frames,
      "rear_tag_detections": per_kf_overlay,
    }
  else:
    front_kw = {
      "front_frames": kept_frames,
      "front_tag_detections": per_kf_overlay,
    }
    rear_kw = {}

  log_ba_run(
    rrd_path=rrd_path,
    body_poses=poses_world_body,
    front_points={},
    rear_points={},
    keyframe_times=kept_times,
    anchor_tag_size=args.anchor_tag_size,
    **front_kw,
    **rear_kw,
  )

  json_path.write_text(
    json.dumps(
      {
        "kept_keyframe_indices": [int(i) for i in kept_idx],
        "kept_keyframe_times": [float(t) for t in kept_times.tolist()],
        "body_positions_world": [
          [float(x) for x in p.translation()] for p in poses_world_body
        ],
        "source": str(args.input_insv),
        "start": args.start,
        "duration": args.duration,
        "anchor_lens": args.anchor_lens,
        "anchor_tag_id": args.anchor_tag_id,
        "anchor_tag_size": args.anchor_tag_size,
        "n_keyframes_total": len(frames),
        "n_keyframes_with_tag": n_visible,
        "n_keyframes_kept": len(poses_world_body),
        "first_kept_keyframe": first_kept,
        "last_kept_keyframe": last_kept,
        "trajectory_path_length": path_len,
        "bounding_box_diagonal": bbox_diag,
        "start_to_end_displacement": start_to_end,
      },
      indent=2,
    )
  )

  print(f"[anchor-only] wrote {rrd_path.name} + {json_path.name}")


if __name__ == "__main__":
  main()
