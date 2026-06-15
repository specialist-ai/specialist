"""Autoresearch evaluation script — reads raw observations from
``track.py``'s JSON, calls ``specialist.vi_ba.eval_metrics``, writes a
companion metrics JSON.

The whole point of this two-script split is that the pipeline cannot
read or game the metric. ``track.py`` knows nothing about
``eval_metrics``; this script knows nothing about gtsam. If the
metric definition needs to change, edit only ``eval_metrics.py`` and
rerun ``eval_track.py`` against the existing raw JSON — no need to
rerun the BA.

Usage:
    uv run python scripts/eval_track.py path/to/track_<stamp>.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tyro

from specialist.loader.mei_model import MeiParams
from specialist.vi_ba.eval_metrics import (
  TagDetection,
  aggregate_rmse,
  compute_anchor_confidently_visible,
  compute_gap_drift,
  dedup_counts_by_position,
  keypoint_world_position_per_kf,
  per_keyframe_reproj_rmse,
  per_pair_keypoint_distances,
  quality_landmark_counts_per_kf,
  trajectory_position_drift,
)

Float64Array = npt.NDArray[np.float64]


def _pose_from_dict(d: Any) -> Float64Array:
  T = np.eye(4, dtype=np.float64)
  T[:3, :3] = np.asarray(d["R"], dtype=np.float64)
  T[:3, 3] = np.asarray(d["t"], dtype=np.float64)
  return T


def _mei_from_dict(d: Any) -> MeiParams:
  return MeiParams(
    fx=float(d["fx"]),
    fy=float(d["fy"]),
    cx=float(d["cx"]),
    cy=float(d["cy"]),
    k1=float(d["k1"]),
    k2=float(d["k2"]),
    k3=float(d["k3"]),
    p1=float(d["p1"]),
    p2=float(d["p2"]),
    xi=float(d["xi"]),
  )


def _tracks_from_dict(d: Any) -> dict[int, dict[int, Float64Array]]:
  return {
    int(tid): {int(kf): np.asarray(p, dtype=np.float64) for kf, p in kf_obs.items()}
    for tid, kf_obs in d.items()
  }


def _landmarks_from_dict(d: Any) -> dict[int, Float64Array]:
  return {int(tid): np.asarray(xyz, dtype=np.float64) for tid, xyz in d.items()}


def _tag_detections_from_dict(per_kf: Any) -> list[list[TagDetection]]:
  return [
    [
      TagDetection(
        tag_id=int(d["tag_id"]),
        corners=np.asarray(d["corners"], dtype=np.float64),
      )
      for d in tags
    ]
    for tags in per_kf
  ]


def _keypoint_world_from_raw(raw: Any) -> dict[str, list[Float64Array]]:
  """Reconstruct each tracker's per-kf fingertip world trajectory from a
  raw ``track.py`` JSON (used to load a reference/pseudo-GT run)."""
  names = list(raw["tracker_names"])
  offset = np.asarray(raw["gripper_keypoint_offset"], dtype=np.float64)
  out: dict[str, list[Float64Array]] = {}
  for name in names:
    poses = [_pose_from_dict(p) for p in raw["body_poses_by_tracker"][name]]
    btg_dict = raw["body_T_base_tag_by_tracker"][name]
    btg = _pose_from_dict(btg_dict) if btg_dict is not None else None
    out[name] = keypoint_world_position_per_kf(poses, btg, offset)
  return out


def _anchor_obs_kf_to_sides(
  anchor_obs_list: Any,
) -> dict[int, dict[str, Float64Array]]:
  """``track.py`` flattens ``anchor_obs_per_kf`` to a list of
  ``{kf, side, cam_T_tag}``; rebuild the kf→side→pose map."""
  out: dict[int, dict[str, Float64Array]] = {}
  for entry in anchor_obs_list:
    kf = int(entry["kf"])
    side = str(entry["side"])
    out.setdefault(kf, {})[side] = _pose_from_dict(entry["cam_T_tag"])
  return out


def main(
  track_json: Path,
  reference_json: Path | None = None,
  blackout: tuple[float, float] | None = None,
) -> None:
  raw = json.loads(track_json.read_text())
  tracker_names: list[str] = list(raw["tracker_names"])
  ref_anchor = _pose_from_dict(raw["ref_anchor_pose_world"])
  anchor_tag_id = int(raw["anchor_tag_id"])
  keypoint_offset = np.asarray(raw["gripper_keypoint_offset"], dtype=np.float64)

  metrics: dict[str, object] = {}
  metrics["source_track_json"] = str(track_json)
  metrics["tracker_names"] = tracker_names

  body_poses_by_tracker: dict[str, list[Float64Array]] = {
    name: [_pose_from_dict(p) for p in raw["body_poses_by_tracker"][name]]
    for name in tracker_names
  }
  n_frames_by_tracker = {
    name: len(body_poses_by_tracker[name]) for name in tracker_names
  }

  anchor_obs_kf_sides_by_tracker: dict[str, dict[int, dict[str, Float64Array]]] = {
    name: _anchor_obs_kf_to_sides(raw["anchor_obs_by_tracker"][name])
    for name in tracker_names
  }

  per_kf_anchor_visible_by_tracker = {
    name: [
      k in anchor_obs_kf_sides_by_tracker[name]
      for k in range(n_frames_by_tracker[name])
    ]
    for name in tracker_names
  }
  metrics["per_kf_anchor_visible_by_tracker"] = per_kf_anchor_visible_by_tracker

  tag_detections_by_tracker: dict[str, dict[str, list[list[TagDetection]]]] = {
    name: {
      "front": _tag_detections_from_dict(
        raw["tag_detections_by_tracker"][name]["front"]
      ),
      "rear": _tag_detections_from_dict(raw["tag_detections_by_tracker"][name]["rear"]),
    }
    for name in tracker_names
  }

  per_kf_anchor_confidently_visible_by_tracker: dict[str, list[bool]] = {}
  for name in tracker_names:
    n = n_frames_by_tracker[name]
    sides_per_kf: dict[int, list[str]] = {
      kf: list(d.keys()) for kf, d in anchor_obs_kf_sides_by_tracker[name].items()
    }
    front_shape = tuple(raw["lens_by_tracker"][name]["front"]["frame_shape"])
    rear_shape = tuple(raw["lens_by_tracker"][name]["rear"]["frame_shape"])
    per_kf_anchor_confidently_visible_by_tracker[name] = (
      compute_anchor_confidently_visible(
        n_frames=n,
        anchor_obs_per_kf=sides_per_kf,
        front_tag_detections=tag_detections_by_tracker[name]["front"],
        rear_tag_detections=tag_detections_by_tracker[name]["rear"],
        front_frame_shape=front_shape,
        rear_frame_shape=rear_shape,
        anchor_tag_id=anchor_tag_id,
      )
    )
  metrics["per_kf_anchor_confidently_visible_by_tracker"] = (
    per_kf_anchor_confidently_visible_by_tracker
  )

  # Per-kf landmark count (post quality bar, deduped across lenses).
  per_kf_landmark_count_by_tracker: dict[str, list[int]] = {}
  per_tracker_rmse: dict[str, dict[str, float]] = {}
  for name in tracker_names:
    n = n_frames_by_tracker[name]
    front_extr = _pose_from_dict(raw["lens_by_tracker"][name]["front"]["extrinsic"])
    rear_extr = _pose_from_dict(raw["lens_by_tracker"][name]["rear"]["extrinsic"])
    front_params = _mei_from_dict(raw["lens_by_tracker"][name]["front"]["params"])
    rear_params = _mei_from_dict(raw["lens_by_tracker"][name]["rear"]["params"])
    front_tracks = _tracks_from_dict(raw["tracks_by_tracker"][name]["front"])
    rear_tracks = _tracks_from_dict(raw["tracks_by_tracker"][name]["rear"])
    front_lms = _landmarks_from_dict(raw["landmarks_by_tracker"][name]["front"])
    rear_lms = _landmarks_from_dict(raw["landmarks_by_tracker"][name]["rear"])
    poses = body_poses_by_tracker[name]

    _, front_xyz = quality_landmark_counts_per_kf(
      poses, front_extr, front_params, front_tracks, front_lms, n
    )
    _, rear_xyz = quality_landmark_counts_per_kf(
      poses, rear_extr, rear_params, rear_tracks, rear_lms, n
    )
    per_kf_landmark_count_by_tracker[name] = dedup_counts_by_position(
      [front_xyz, rear_xyz], n
    )

    front_rmse_per_kf = per_keyframe_reproj_rmse(
      poses, front_extr, front_params, front_tracks, front_lms, n
    )
    rear_rmse_per_kf = per_keyframe_reproj_rmse(
      poses, rear_extr, rear_params, rear_tracks, rear_lms, n
    )
    per_tracker_rmse[name] = {
      "front_rmse_px": aggregate_rmse(front_rmse_per_kf),
      "rear_rmse_px": aggregate_rmse(rear_rmse_per_kf),
    }
  metrics["per_kf_landmark_count_by_tracker"] = per_kf_landmark_count_by_tracker
  metrics["per_tracker_rmse"] = per_tracker_rmse

  # Anchor-revisit drift per tracker.
  gap_drift_by_tracker: dict[str, list[dict[str, object]]] = {}
  for name in tracker_names:
    n = n_frames_by_tracker[name]
    front_extr = _pose_from_dict(raw["lens_by_tracker"][name]["front"]["extrinsic"])
    rear_extr = _pose_from_dict(raw["lens_by_tracker"][name]["rear"]["extrinsic"])
    kf_times = np.asarray(raw["kf_times_by_tracker"][name], dtype=np.float64)
    drifts = compute_gap_drift(
      body_poses=body_poses_by_tracker[name],
      ref_anchor=ref_anchor,
      body_T_front_lens=front_extr,
      body_T_rear_lens=rear_extr,
      anchor_obs_per_kf=anchor_obs_kf_sides_by_tracker[name],
      has_confident_anchor=per_kf_anchor_confidently_visible_by_tracker[name],
      kf_times=kf_times,
      n_frames=n,
    )
    gap_drift_by_tracker[name] = drifts
    if drifts:
      drift_norms = np.asarray([d["drift"] for d in drifts])
      worst = drifts[int(np.argmax(drift_norms))]
      gap_t = worst["gap_t"]
      assert isinstance(gap_t, tuple)
      print(
        f"  [{name}] anchor-revisit error: {len(drifts)} gaps, "
        f"max={drift_norms.max() * 1000:.1f} mm at "
        f"t={gap_t[0]:.2f}-{gap_t[1]:.2f}s, "
        f"median={float(np.median(drift_norms)) * 1000:.1f} mm"
      )
  metrics["gap_drift_by_tracker"] = gap_drift_by_tracker

  # Inter-tracker keypoint distance.
  keypoint_world_by_tracker: dict[str, list[Float64Array]] = {}
  for name in tracker_names:
    btg_dict = raw["body_T_base_tag_by_tracker"][name]
    btg = _pose_from_dict(btg_dict) if btg_dict is not None else None
    keypoint_world_by_tracker[name] = keypoint_world_position_per_kf(
      body_poses_by_tracker[name], btg, keypoint_offset
    )
  per_kf_keypoint_distance_by_pair, keypoint_distance_stats = (
    per_pair_keypoint_distances(keypoint_world_by_tracker, tracker_names)
  )
  metrics["per_kf_keypoint_distance_by_pair"] = per_kf_keypoint_distance_by_pair
  metrics["keypoint_distance_stats"] = keypoint_distance_stats
  if keypoint_distance_stats:
    print("\n=== taped-keypoint world-frame distance ===")
    for pair, s in keypoint_distance_stats.items():
      print(
        f"  {pair}: mean={s['mean'] * 1000:.1f}mm "
        f"median={s['median'] * 1000:.1f}mm "
        f"p95={s['p95'] * 1000:.1f}mm "
        f"max={s['max'] * 1000:.1f}mm  (n={int(s['n'])})"
      )

  # Fingertip drift vs a reference run (the full-anchor pseudo-GT). Both live
  # in the same anchor world, so the per-kf L2 is the candidate's VI drift.
  if reference_json is not None:
    ref_kp = _keypoint_world_from_raw(json.loads(reference_json.read_text()))
    drift_by_tracker: dict[str, dict[str, object]] = {}
    print("\n=== fingertip drift vs reference (pseudo-GT) ===")
    for name in tracker_names:
      if name not in ref_kp:
        continue
      kf_times = np.asarray(raw["kf_times_by_tracker"][name], dtype=np.float64)
      d = trajectory_position_drift(
        keypoint_world_by_tracker[name], ref_kp[name], kf_times, window=blackout
      )
      drift_by_tracker[name] = d
      ov = d["overall"]
      assert isinstance(ov, dict)
      msg = (
        f"  [{name}] overall median={ov['median'] * 1000:.1f}mm "
        f"max={ov['max'] * 1000:.1f}mm (n={int(ov['n'])})"
      )
      win = d.get("window")
      if isinstance(win, dict):
        bounds = d["window_bounds"]
        assert isinstance(bounds, list)
        msg += (
          f" | blackout {bounds[0]:.0f}-{bounds[1]:.0f}s "
          f"median={win['median'] * 1000:.1f}mm max={win['max'] * 1000:.1f}mm "
          f"(n={int(win['n'])})"
        )
      print(msg)
    metrics["fingertip_drift_vs_reference"] = drift_by_tracker

  out_path = track_json.with_suffix(".eval.json")
  out_path.write_text(json.dumps(metrics, indent=2, default=str))
  print(f"\nWrote {out_path}")


if __name__ == "__main__":
  tyro.cli(main)
