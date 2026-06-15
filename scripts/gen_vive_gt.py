"""Build the HTC Vive ground-truth ghost (``.npz`` of gripper poses + times) that
``track.py --ground-truth-npy`` overlays in the Rerun blueprint.

Reads the SteamVR tracker ``/tf`` from a mocap-session rosbag/mcap, time-syncs it
to a finished run by velocity-profile correlation (offset + clock-rate), maps the
tracker pose to the gripper pose via the measured rigid mount, and aligns the
result into the run's anchor world with a robust (Huber-IRLS) Umeyama sim(3) so
high-residual keyframes (poorly tracked / fast motion) don't drag the fit off the
well-tracked segments.

The tracker→gripper mount (measured): the tracker origin sits at
``(-0.045, -0.040, -0.010)`` in the gripper frame, with tracker ``+x`` = gripper
``+z`` and tracker ``+y`` = gripper ``-x`` (hence tracker ``+z`` = gripper ``-y``).
``_T_TG`` is the gripper origin in the tracker frame and ``_R_TG`` right-multiplies
the tracker orientation to yield the gripper orientation; the viz then applies the
gripper→fingertip ``keypoint_offset`` in that gripper frame, so the ghost must
carry the gripper orientation (not the raw tracker orientation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

_TRACKER = "LHR-EDF0CCFB"
_T_TG = np.array([0.010, -0.045, -0.040])
_R_TG = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _umeyama_weighted(
  src: np.ndarray, dst: np.ndarray, w: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
  w = w / w.sum()
  mu_s = (src * w[:, None]).sum(0)
  mu_d = (dst * w[:, None]).sum(0)
  sc, dc = src - mu_s, dst - mu_d
  u, dv, vt = np.linalg.svd((dc * w[:, None]).T @ sc)
  ww = np.eye(3)
  if np.linalg.det(u @ vt) < 0:
    ww[-1, -1] = -1
  r = u @ ww @ vt
  s = float(np.trace(np.diag(dv) @ ww) / (w[:, None] * sc**2).sum())
  return s, r, mu_d - s * r @ mu_s


def _robust_sim3(
  src: np.ndarray, dst: np.ndarray, iters: int = 10
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
  """Huber-IRLS sim(3): down-weights high-residual (outlier) keyframes."""
  w = np.ones(len(src))
  s, r, t = _umeyama_weighted(src, dst, w)
  for _ in range(iters):
    res = np.linalg.norm((s * (r @ src.T).T + t) - dst, axis=1)
    k = 1.345 * (float(np.median(res)) + 1e-9)
    w = np.where(res <= k, 1.0, k / (res + 1e-9))
    s, r, t = _umeyama_weighted(src, dst, w)
  return s, r, t, w


def _read_tracker(bag: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  # Group every /tf transform by child frame, then pick the gripper tracker: the
  # known serial if present, otherwise the dominant frame that is neither the HMD
  # nor a lighthouse base station. Sessions label the tracker by serial (e.g.
  # "LHR-...") or semantically (e.g. "right_wrist"), so auto-detect handles both.
  by_frame: dict[str, list[tuple[float, list[float], list[float]]]] = {}
  with AnyReader([bag]) as reader:
    conns = [c for c in reader.connections if c.topic == "/tf"]
    for c, _, raw in reader.messages(connections=conns):
      msg = reader.deserialize(raw, c.msgtype)
      for tf in msg.transforms:
        tr, q = tf.transform.translation, tf.transform.rotation
        by_frame.setdefault(tf.child_frame_id, []).append(
          (
            tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9,
            [tr.x, tr.y, tr.z],
            [q.x, q.y, q.z, q.w],
          )
        )
  if _TRACKER in by_frame:
    frame = _TRACKER
  else:
    candidates = {
      f: rows for f, rows in by_frame.items() if f != "hmd" and not f.startswith("LHB-")
    }
    if not candidates:
      raise RuntimeError(f"no tracker frame found in /tf; saw {sorted(by_frame)}")
    frame = max(candidates, key=lambda f: len(candidates[f]))
  rows = by_frame[frame]
  te_arr = np.array([r[0] for r in rows])
  tp_arr = np.array([r[1] for r in rows])
  tq_arr = np.array([r[2] for r in rows])
  order = np.argsort(te_arr)
  return te_arr[order], tp_arr[order], tq_arr[order]


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("run_json", type=Path, help="track.py output JSON")
  ap.add_argument("bag", type=Path, help="mocap-session rosbag/mcap directory")
  ap.add_argument("out_npz", type=Path, help="ghost .npz (poses, times) for track.py")
  args = ap.parse_args()

  d = json.loads(args.run_json.read_text())
  name = d["tracker_names"][0]
  kt = np.array(d["kf_times_by_tracker"][name], float)
  bt = np.array(d["body_T_base_tag_by_tracker"][name]["t"], float)
  pg = np.array(
    [
      np.array(p["R"], float) @ bt + np.array(p["t"], float)
      for p in d["body_poses_by_tracker"][name]
    ]
  )
  te, tp, tq = _read_tracker(args.bag)

  # Time sync: velocity-profile correlation, refined over offset + clock-rate.
  pv = np.linalg.norm(np.diff(pg, axis=0), axis=1) / np.diff(kt)
  ve = np.linalg.norm(np.diff(tp, axis=0), axis=1) / np.diff(te)
  tmid = 0.5 * (kt[1:] + kt[:-1])

  def corr(off: float, rate: float) -> float:
    vs = np.interp(tmid * rate + off, te[1:], ve, left=0, right=0)
    if np.std(vs) < 1e-6:
      return -1.0
    return float(np.corrcoef(pv, vs)[0, 1])

  best = max(
    ((corr(o, 1.0), o, 1.0) for o in np.arange(te[0] - kt[-1], te[-1] - kt[0], 0.05)),
    key=lambda x: x[0],
  )
  for _ in range(4):
    _, o0, r0 = best
    grid = [
      (corr(o0 + do, r0 + dr), o0 + do, r0 + dr)
      for do in np.linspace(-0.05, 0.05, 11)
      for dr in np.linspace(-0.002, 0.002, 9)
    ]
    best = max([best, *grid], key=lambda x: x[0])
  cc, off, rate = best

  slerp = Slerp(te, Rot.from_quat(tq))
  qe = kt * rate + off
  cov = (qe >= te[0]) & (qe <= te[-1])
  grip_p = np.full((len(kt), 3), np.nan)
  grip_r: list[np.ndarray | None] = [None] * len(kt)
  for k in np.where(cov)[0]:
    rk = slerp([qe[k]])[0].as_matrix()
    pk = np.array([np.interp(qe[k], te, tp[:, j]) for j in range(3)])
    grip_p[k] = rk @ _T_TG + pk
    grip_r[k] = rk @ _R_TG
  idx = np.where(cov & ~np.isnan(grip_p[:, 0]))[0]

  s, r, t, w = _robust_sim3(pg[idx], grip_p[idx])
  res = np.linalg.norm((s * (r @ pg[idx].T).T + t) - grip_p[idx], axis=1) * 1000
  poses = np.empty((len(idx), 4, 4))
  for n, k in enumerate(idx):
    poses[n] = np.eye(4)
    poses[n, :3, :3] = r.T @ grip_r[k]
    poses[n, :3, 3] = (r.T @ (grip_p[k] - t)) / s
  np.savez(args.out_npz, poses=poses, times=kt[idx])
  print(
    f"saved {args.out_npz} poses={len(idx)} corr={cc:.3f} off={off:.2f} rate={rate:.4f} "
    f"sim3_scale={s:.3f} robust-ATE median={np.median(res):.0f}mm "
    f"inliers={(w > 0.5).sum()}/{len(w)}"
  )


if __name__ == "__main__":
  main()
