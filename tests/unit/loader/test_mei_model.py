"""Round-trip tests for the MEI projection."""

from __future__ import annotations

import numpy as np
import pytest

from specialist.loader.mei_model import (
  MeiParams,
  mei_project,
  mei_unproject,
)


def _x3_like_params() -> MeiParams:
  """Plausible Insta360 intrinsics in 1920x1920 per-lens coords."""
  return MeiParams(
    fx=540.0,
    fy=540.0,
    cx=960.0,
    cy=960.0,
    k1=0.39,
    k2=1.24,
    k3=-3.86,
    p1=-1.0e-5,
    p2=-2.16e-3,
    xi=1.948,
  )


def _undistorted_params() -> MeiParams:
  """Pinhole on the unit sphere with no distortion, so the invariants are easy."""
  return MeiParams(
    fx=400.0,
    fy=400.0,
    cx=320.0,
    cy=240.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
    xi=0.0,
  )


def test_mei_unprojects_principal_point_to_optical_axis() -> None:
  params = _undistorted_params()
  pixels = np.array([[320.0, 240.0]])
  bearing = mei_unproject(pixels, params)
  assert bearing.shape == (1, 3)
  assert np.allclose(bearing, [[0.0, 0.0, 1.0]], atol=1e-9)


def test_mei_round_trip_pinhole_offcenter() -> None:
  params = _undistorted_params()
  rng = np.random.default_rng(0)
  pixels = rng.uniform([100.0, 80.0], [540.0, 400.0], size=(50, 2))
  bearing = mei_unproject(pixels, params)
  reprojected = mei_project(bearing, params)
  assert np.allclose(reprojected, pixels, atol=1e-6)


def _forward_rays(rng: np.random.Generator, n: int, half_angle: float) -> np.ndarray:
  """Sample unit rays inside a forward cone of given half-angle."""
  cos_min = np.cos(half_angle)
  cos_t = rng.uniform(cos_min, 1.0, size=n)
  sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
  phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
  return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)


def test_mei_round_trip_with_distortion() -> None:
  """ray -> pixel -> ray within tight tolerance, staying inside the camera FOV."""
  params = _x3_like_params()
  rng = np.random.default_rng(42)
  rays = _forward_rays(rng, n=80, half_angle=np.deg2rad(85.0))
  pixels = mei_project(rays, params)
  assert np.isfinite(pixels).all()
  recovered = mei_unproject(pixels, params)
  assert np.isfinite(recovered).all()
  assert np.max(np.abs(recovered - rays)) < 1e-6


def test_mei_unit_bearings_have_unit_norm() -> None:
  """Unprojected bearings are unit vectors when the source pixels are in-domain."""
  params = _x3_like_params()
  rng = np.random.default_rng(1)
  rays = _forward_rays(rng, n=20, half_angle=np.deg2rad(80.0))
  pixels = mei_project(rays, params)
  bearing = mei_unproject(pixels, params)
  norms = np.linalg.norm(bearing, axis=1)
  assert np.allclose(norms, 1.0, atol=1e-9)


def test_mei_project_rejects_bad_shape() -> None:
  params = _undistorted_params()
  with pytest.raises(ValueError):
    mei_project(np.zeros((3,)), params)


def test_mei_unproject_rejects_bad_shape() -> None:
  params = _undistorted_params()
  with pytest.raises(ValueError):
    mei_unproject(np.zeros((3,)), params)
