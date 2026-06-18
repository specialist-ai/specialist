"""Synthetic-scene test for the incremental monocular SfM front-end.

A known camera sweep observes a 3D point cloud; the projected unit bearings are
fed to ``incremental_sfm`` and the recovered camera trajectory must match the
ground truth up to a similarity transform (monocular scale/gauge are free).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rscp

from specialist.vi_ba.init import incremental_sfm


def _scene(n_frames: int = 30, n_points: int = 250, noise: float = 0.0, seed: int = 0):
  rng = np.random.default_rng(seed)
  # camera sweeps laterally 0.6 m and pans ~20° (T_world_camera = R_i, C_i)
  centers = np.stack(
    [np.linspace(-0.3, 0.3, n_frames), np.zeros(n_frames), np.zeros(n_frames)], axis=1
  )
  yaw = np.linspace(-0.18, 0.18, n_frames)
  rots = [Rscp.from_euler("y", a).as_matrix() for a in yaw]  # cam→world
  # point cloud: near surface (parallax) + far wall, all in +z half-space
  pts = np.column_stack(
    [
      rng.uniform(-0.6, 0.6, n_points),
      rng.uniform(-0.35, 0.35, n_points),
      rng.uniform(0.3, 3.0, n_points),
    ]
  )
  bearings: dict[int, dict[int, np.ndarray]] = {}
  for tid, x in enumerate(pts):
    obs: dict[int, np.ndarray] = {}
    for k in range(n_frames):
      x_cam = rots[k].T @ (x - centers[k])  # world→cam
      if x_cam[2] <= 0.05:
        continue
      if abs(x_cam[0] / x_cam[2]) > 1.4 or abs(x_cam[1] / x_cam[2]) > 1.4:
        continue
      b = x_cam / np.linalg.norm(x_cam)
      if noise:
        b = b + rng.normal(0, noise, 3)
        b = b / np.linalg.norm(b)
      obs[k] = b
    if len(obs) >= 3:
      bearings[tid] = obs
  return bearings, centers, rots


def _umeyama_ate(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
  mu_s, mu_d = src.mean(0), dst.mean(0)
  sc, dc = src - mu_s, dst - mu_d
  u, dv, vt = np.linalg.svd(dc.T @ sc / len(src))
  w = np.eye(3)
  if np.linalg.det(u @ vt) < 0:
    w[-1, -1] = -1
  r = u @ w @ vt
  s = np.trace(np.diag(dv) @ w) / ((sc**2).sum() / len(src))
  t = mu_d - s * r @ mu_s
  return np.linalg.norm((s * (r @ src.T).T + t) - dst, axis=1)


def test_recovers_trajectory_noiseless():
  n = 30
  bearings, centers, _ = _scene(n_frames=n, noise=0.0)
  cam_poses, pmap = incremental_sfm(bearings, n)
  assert len(cam_poses) >= int(0.8 * n), f"only registered {len(cam_poses)}/{n}"
  assert len(pmap) >= 50
  ks = sorted(cam_poses)
  rec = np.array([np.asarray(cam_poses[k].translation()) for k in ks])
  ate = _umeyama_ate(rec, centers[ks])
  extent = float(np.linalg.norm(centers.max(0) - centers.min(0)))
  assert np.median(ate) < 0.02 * extent, (
    f"median ATE {np.median(ate):.4f} of {extent:.3f}"
  )


def test_recovers_trajectory_noisy():
  n = 30
  bearings, centers, _ = _scene(n_frames=n, noise=1e-3)
  cam_poses, _ = incremental_sfm(bearings, n)
  assert len(cam_poses) >= int(0.7 * n)
  ks = sorted(cam_poses)
  rec = np.array([np.asarray(cam_poses[k].translation()) for k in ks])
  ate = _umeyama_ate(rec, centers[ks])
  extent = float(np.linalg.norm(centers.max(0) - centers.min(0)))
  assert np.median(ate) < 0.08 * extent


def test_no_bootstrap_returns_empty():
  # too few shared tracks → cannot bootstrap
  bearings = {0: {0: np.array([0.0, 0.0, 1.0]), 1: np.array([0.0, 0.0, 1.0])}}
  cam_poses, pmap = incremental_sfm(bearings, 5)
  assert cam_poses == {} and pmap == {}
