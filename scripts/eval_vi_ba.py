"""Compare a VI-BA run's body trajectory and landmark cloud against the
anchor-only PnP baseline. Prints trajectory error, landmark stats, and
the per-gap anchor-revisit residual. Re-runs ``ba_anchor_only.py`` to
produce a fresh baseline JSON sidecar if none matches this clip."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun_bindings as rb
import tyro


@dataclass(frozen=True)
class Trajectory:
  times: np.ndarray  # (N,)
  positions: np.ndarray  # (N, 3)


def _latest_json_for(prefix: str, source_path: Path, logs_dir: Path) -> Path | None:
  """Find most recent JSON sidecar for ``prefix`` whose source matches.

  ``ba_anchor_only.py`` writes ``"source": str`` (one input);
  ``track.py`` writes ``"sources": [str, ...]`` (one or more).
  """
  src_str = str(source_path)
  cands = sorted(logs_dir.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime)
  for p in reversed(cands):
    try:
      data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
      continue
    if data.get("source") == src_str or src_str in (data.get("sources") or []):
      return p
  return None


def _anchor_world_to_tag_transform(
  json_path: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
  """Read ``anchor_pose_world`` from a VI-BA json. Returns ``(R_tag_world,
  t_tag)`` such that ``p_tag = R_tag_world @ (p_world - t_tag)``. Returns
  ``None`` if the json lacks the field (positions already in tag-frame)."""
  data = json.loads(json_path.read_text())
  pose = data.get("anchor_pose_world")
  if pose is None:
    return None
  t = np.asarray(pose["translation"], dtype=np.float64)
  qx, qy, qz, qw = pose["quaternion_xyzw"]
  R = np.array(
    [
      [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
      [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
      [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ],
    dtype=np.float64,
  )
  return R.T, t  # tag = R^T (world - t)


def _load_traj_from_json(
  json_path: Path, tracker_name: str = "tracker_0"
) -> Trajectory:
  """Load a tracker's body trajectory in tag frame.

  ``ba_anchor_only.py`` writes ``body_positions_world`` as a flat list and
  ``kept_keyframe_times``; ``track.py`` writes them keyed by tracker name.
  """
  data = json.loads(json_path.read_text())
  body = data["body_positions_world"]
  pos = np.asarray(
    body[tracker_name] if isinstance(body, dict) else body,
    dtype=np.float64,
  )
  xform = _anchor_world_to_tag_transform(json_path)
  if xform is not None:
    R_tag_world, t = xform
    pos = (pos - t) @ R_tag_world.T
  if "kept_keyframe_times" in data:
    t_arr = np.asarray(data["kept_keyframe_times"], dtype=np.float64)
  elif "keyframe_times_by_tracker" in data:
    t_arr = np.asarray(
      data["keyframe_times_by_tracker"][tracker_name], dtype=np.float64
    )
  else:
    t_arr = np.arange(pos.shape[0], dtype=np.float64)
  return Trajectory(times=t_arr, positions=pos)


def _load_landmarks_from_rrd(
  rrd_path: Path, json_path: Path | None = None
) -> np.ndarray:
  """All static landmark world positions (front + rear) from a VI-BA rrd.
  If ``json_path`` is given and contains an ``anchor_pose_world`` field,
  landmarks are transformed into tag-frame to match the eval baseline."""
  rec = rb.load_recording(str(rrd_path))
  out: list[np.ndarray] = []
  for c in rec.chunks():
    if not c.entity_path.startswith("/world/landmarks/"):
      continue
    rt = c.to_record_batch()
    if "Points3D:positions" not in rt.column_names:
      continue
    for row in rt.column("Points3D:positions").to_pylist():
      if row is None:
        continue
      for xyz in row:
        out.append(np.asarray(xyz, dtype=np.float64))
  pts = np.stack(out, axis=0) if out else np.empty((0, 3), dtype=np.float64)
  if json_path is not None and pts.size:
    xform = _anchor_world_to_tag_transform(json_path)
    if xform is not None:
      R_tag_world, t = xform
      pts = (pts - t) @ R_tag_world.T
  return pts


def _ensure_anchor_only(
  clip: Path,
  start: float,
  duration: float | None,
  keyframe_every: int,
  logs_dir: Path,
  anchor_tag_id: int,
  anchor_tag_size: float,
  anchor_lens: str,
) -> Path:
  """Re-run anchor-only if no fresh JSON sidecar exists for this clip."""
  existing = _latest_json_for("ba_anchor_only", clip, logs_dir)
  if existing is not None:
    data = json.loads(existing.read_text())
    cached_start_ok = abs(data.get("start", -1) - start) < 1e-6
    cached_duration = data.get("duration")
    duration_ok = (cached_duration is None and duration is None) or (
      cached_duration is not None
      and duration is not None
      and abs(cached_duration - duration) < 1e-6
    )
    if cached_start_ok and duration_ok:
      return existing
  cmd = [
    "uv",
    "run",
    "python",
    str(Path(__file__).parent / "ba_anchor_only.py"),
    "--input-insv",
    str(clip),
    "--start",
    str(start),
    *(["--duration", str(duration)] if duration is not None else []),
    "--keyframe-every",
    str(keyframe_every),
    "--anchor-tag-id",
    str(anchor_tag_id),
    "--anchor-tag-size",
    str(anchor_tag_size),
    "--anchor-lens",
    anchor_lens,
    "--logs-dir",
    str(logs_dir),
  ]
  subprocess.run(cmd, check=True, capture_output=True)
  found = _latest_json_for("ba_anchor_only", clip, logs_dir)
  if found is None:
    raise RuntimeError(f"anchor-only ran but no JSON found for {clip}")
  return found


def _align_to_baseline_times(
  baseline: Trajectory, candidate: Trajectory, dt_tol: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
  """Match candidate keyframes to baseline by nearest-time within ``dt_tol``.

  Returns ``(baseline_pos_matched, candidate_pos_matched)`` of equal length.
  """
  base_pos: list[np.ndarray] = []
  cand_pos: list[np.ndarray] = []
  for i, t in enumerate(baseline.times):
    j = int(np.argmin(np.abs(candidate.times - t)))
    if abs(candidate.times[j] - t) <= dt_tol:
      base_pos.append(baseline.positions[i])
      cand_pos.append(candidate.positions[j])
  if not base_pos:
    return (
      np.empty((0, 3), dtype=np.float64),
      np.empty((0, 3), dtype=np.float64),
    )
  return np.stack(base_pos, axis=0), np.stack(cand_pos, axis=0)


def evaluate(
  clip: Path,
  start: float,
  duration: float | None,
  keyframe_every: int,
  anchor_tag_id: int,
  anchor_tag_size: float,
  anchor_lens: str,
  logs_dir: Path = Path("logs"),
) -> None:
  baseline_json = _ensure_anchor_only(
    clip,
    start,
    duration,
    keyframe_every,
    logs_dir,
    anchor_tag_id,
    anchor_tag_size,
    anchor_lens,
  )
  vi_json = _latest_json_for("track", clip, logs_dir)
  if vi_json is None:
    raise RuntimeError(
      f"no VI-BA JSON found for {clip} in {logs_dir}; run track.py first"
    )
  vi_rrd = vi_json.with_suffix(".rrd")

  baseline = _load_traj_from_json(baseline_json)
  candidate = _load_traj_from_json(vi_json)
  base_pos, cand_pos = _align_to_baseline_times(baseline, candidate)
  if base_pos.shape[0] == 0:
    print(f"[{clip.name}] no overlapping keyframes between baseline and VI-BA")
    return

  err = np.linalg.norm(base_pos - cand_pos, axis=1) * 1000.0  # mm
  base_dist = np.linalg.norm(base_pos, axis=1)
  cand_dist = np.linalg.norm(cand_pos, axis=1)

  landmarks = _load_landmarks_from_rrd(vi_rrd, vi_json)
  lm_dist = np.linalg.norm(landmarks, axis=1) if landmarks.size else np.array([])
  if landmarks.size:
    diffs = landmarks[:, None, :] - cand_pos[None, :, :]
    lm_to_traj = np.linalg.norm(diffs, axis=2).min(axis=1)
  else:
    lm_to_traj = np.array([])

  # Verdict thresholds chosen for tabletop handheld:
  # - mean trajectory error < 20 mm = green
  # - mean trajectory error < 50 mm = yellow
  # - "collision" = >25% of landmarks within 5 cm of any body pose
  cam_dist_med = float(np.median(base_dist))
  lm_dist_med = float(np.median(lm_dist)) if lm_dist.size else float("nan")
  collision_frac = float((lm_to_traj < 0.05).mean()) if lm_to_traj.size else 0.0
  collision = collision_frac > 0.25
  err_mean = float(err.mean())
  err_max = float(err.max())
  err_p95 = float(np.percentile(err, 95))

  if err_mean < 20.0 and not collision:
    verdict = "GREEN"
  elif err_mean < 50.0:
    verdict = "YELLOW"
  else:
    verdict = "RED"

  range_label = (
    f"{start:.0f}-end" if duration is None else f"{start:.0f}-{start + duration:.0f}"
  )
  print(f"=== {clip.name} ({range_label} s) ===")
  print(f"  matched keyframes: {base_pos.shape[0]} of {baseline.positions.shape[0]}")
  print("  trajectory error vs anchor-only (mm):")
  print(f"    mean={err_mean:6.1f}  max={err_max:6.1f}  p95={err_p95:6.1f}")
  vi_data = json.loads(vi_json.read_text())
  drifts_by = vi_data.get("gap_drift_by_tracker", {})
  drifts: list[dict] = []
  for ds in drifts_by.values():
    drifts.extend(ds)
  if drifts:
    drifts_mm = np.asarray([d["drift"] for d in drifts]) * 1000.0
    worst = drifts[int(np.argmax(drifts_mm))]
    t0, t1 = worst["gap_t"]
    print(
      f"  anchor-revisit error: {len(drifts)} gaps, "
      f"max={drifts_mm.max():.1f} mm at t={t0:.2f}-{t1:.2f}s, "
      f"median={float(np.median(drifts_mm)):.1f} mm"
    )
  print(
    f"  camera-to-anchor distance (m): baseline={cam_dist_med:.3f}  "
    f"candidate={float(np.median(cand_dist)):.3f}"
  )
  if lm_dist.size:
    print(
      f"  landmark distance from anchor (m): "
      f"min={float(lm_dist.min()):.3f}  "
      f"p50={lm_dist_med:.3f}  "
      f"p95={float(np.percentile(lm_dist, 95)):.3f}  "
      f"max={float(lm_dist.max()):.3f}  "
      f"(n={lm_dist.size})"
    )
    print(
      f"  landmark distance to nearest body pose (m): "
      f"min={float(lm_to_traj.min()):.3f}  "
      f"p50={float(np.median(lm_to_traj)):.3f}  "
      f"p95={float(np.percentile(lm_to_traj, 95)):.3f}  "
      f"frac<5cm={collision_frac * 100:.1f}%"
    )
    if collision:
      print(
        f"  ⚠ {collision_frac * 100:.1f}% of landmarks within 5 cm of a body "
        f"pose — likely body-attached artifacts"
      )
  print(f"  verdict: {verdict}")
  print(f"  baseline: {baseline_json.name}")
  print(f"  candidate: {vi_json.name}")


@dataclass
class Args:
  clip: Path
  start: float = 0.0
  duration: float | None = None
  keyframe_every: int = 9
  anchor_tag_id: int = 0
  anchor_tag_size: float = 0.1
  anchor_lens: str = "rear"
  logs_dir: Path = Path("logs")


def main(args: Args) -> None:
  evaluate(
    clip=args.clip,
    start=args.start,
    duration=args.duration,
    keyframe_every=args.keyframe_every,
    anchor_tag_id=args.anchor_tag_id,
    anchor_tag_size=args.anchor_tag_size,
    anchor_lens=args.anchor_lens,
    logs_dir=args.logs_dir,
  )


if __name__ == "__main__":
  main(tyro.cli(Args))
