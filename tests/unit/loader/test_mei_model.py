"""Round-trip tests for the MEI projection."""

from __future__ import annotations

import numpy as np
import pytest

from specialist.loader.mei_model import (
  MeiParams,
  mei_project,
  mei_project_with_jacobian,
  mei_unproject,
)


def _x3_like_params() -> MeiParams:
  """Plausible Insta360 X3 intrinsics in 1920x1920 per-lens coords."""
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
  """Pinhole on the unit sphere, no tangential, no distortion. Easy invariants."""
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


def test_mei_round_trip_pinhole_centered() -> None:
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


def test_mei_project_with_jacobian_pixel_matches_batch_project() -> None:
  """The single-point analytical projector returns the same pixel as the
  batched ``mei_project`` for the X3-like params at a few in-FOV points."""
  params = _x3_like_params()
  for p in [
    np.array([0.05, 0.05, 1.0]),
    np.array([0.5, 0.2, 1.5]),
    np.array([1.0, -0.3, 0.8]),
    np.array([1.5, 0.0, 0.2]),
  ]:
    pixel_a, _ = mei_project_with_jacobian(p, params)
    pixel_b = mei_project(p[None, :], params)[0]
    assert np.allclose(pixel_a, pixel_b, atol=1e-12)


def test_mei_project_jacobian_matches_central_finite_differences() -> None:
  """Analytical 2×3 Jacobian matches central finite differences on
  ``mei_project`` to ~1e-5 relative across the FOV including a point
  beyond the 90° half-angle."""
  params = _x3_like_params()
  test_points = [
    np.array([0.01, 0.02, 1.5]),  # near-axial
    np.array([0.4, 0.3, 1.0]),
    np.array([1.0, -0.2, 0.6]),
    np.array([1.6, 0.0, 0.3]),
    np.array([1.0, 0.0, -0.1]),  # past 90° half-angle, super-fisheye
  ]
  eps = 1e-6
  for p in test_points:
    _, jac = mei_project_with_jacobian(p, params)
    fd = np.zeros((2, 3), dtype=np.float64)
    for i in range(3):
      d = np.zeros(3)
      d[i] = eps
      pix_p = mei_project((p + d)[None, :], params)[0]
      pix_m = mei_project((p - d)[None, :], params)[0]
      fd[:, i] = (pix_p - pix_m) / (2.0 * eps)
    rel = np.linalg.norm(jac - fd) / max(np.linalg.norm(fd), 1.0)
    assert rel < 1e-4, f"point {p}: |jac - fd|/|fd| = {rel:.2e}"


def test_mei_project_with_jacobian_returns_nan_behind_mirror() -> None:
  # With ξ > 1 (X3) the unit-sphere shift covers the full forward
  # hemisphere, so the cheirality check only trips at ξ = 0 (pinhole)
  # for points with z < 0.
  params = _undistorted_params()  # xi=0
  pixel, jac = mei_project_with_jacobian(np.array([0.1, 0.0, -1.0]), params)
  assert np.all(np.isnan(pixel))
  assert np.all(np.isnan(jac))


def test_mei_from_camera_matrix_validates_shapes() -> None:
  with pytest.raises(ValueError):
    MeiParams.from_camera_matrix(np.zeros((2, 2)), np.zeros((6,)))
  with pytest.raises(ValueError):
    MeiParams.from_camera_matrix(np.eye(3), np.zeros((5,)))
