"""Round-trip tests for the Kannala-Brandt projection."""

from __future__ import annotations

import numpy as np
import pytest

from specialist.loader.kb_model import KbParams, kb_project, kb_unproject


def _wide_fisheye_params() -> KbParams:
  return KbParams(
    fx=320.0,
    fy=320.0,
    cx=640.0,
    cy=640.0,
    k1=-0.01,
    k2=0.005,
    k3=-0.001,
    k4=0.0001,
  )


def _pinhole_params() -> KbParams:
  return KbParams(
    fx=500.0, fy=500.0, cx=320.0, cy=240.0, k1=0.0, k2=0.0, k3=0.0, k4=0.0
  )


def test_kb_round_trip_centered() -> None:
  params = _pinhole_params()
  reprojected = kb_project(np.array([[0.0, 0.0, 1.0]]), params)
  assert np.allclose(reprojected, [[params.cx, params.cy]])


def test_kb_round_trip_pinhole() -> None:
  params = _pinhole_params()
  rng = np.random.default_rng(0)
  pixels = rng.uniform(
    [params.cx - 200.0, params.cy - 150.0],
    [params.cx + 200.0, params.cy + 150.0],
    size=(50, 2),
  )
  bearing = kb_unproject(pixels, params)
  reprojected = kb_project(bearing, params)
  assert np.allclose(reprojected, pixels, atol=1e-6)


def test_kb_round_trip_distorted() -> None:
  params = _wide_fisheye_params()
  rng = np.random.default_rng(1)
  pixels = rng.uniform(
    [params.cx - 400.0, params.cy - 400.0],
    [params.cx + 400.0, params.cy + 400.0],
    size=(80, 2),
  )
  bearing = kb_unproject(pixels, params)
  reprojected = kb_project(bearing, params)
  assert np.allclose(reprojected, pixels, atol=1e-4)


def test_kb_handles_back_of_camera() -> None:
  """KB models theta from optical axis, so z<0 (rays behind) should still project."""
  params = _wide_fisheye_params()
  pts = np.array([[0.5, 0.5, -0.5]])
  pix = kb_project(pts, params)
  assert np.isfinite(pix).all()


def test_kb_unit_bearings_have_unit_norm() -> None:
  params = _wide_fisheye_params()
  rng = np.random.default_rng(2)
  pixels = rng.uniform(
    [params.cx - 200.0, params.cy - 200.0],
    [params.cx + 200.0, params.cy + 200.0],
    size=(20, 2),
  )
  bearing = kb_unproject(pixels, params)
  norms = np.linalg.norm(bearing, axis=1)
  assert np.allclose(norms, 1.0, atol=1e-9)


def test_kb_array_round_trip() -> None:
  params = _wide_fisheye_params()
  again = KbParams.from_array(params.to_array())
  assert again == params


def test_kb_from_array_validates_shape() -> None:
  with pytest.raises(ValueError):
    KbParams.from_array(np.zeros((4,)))


def test_kb_project_rejects_bad_shape() -> None:
  params = _pinhole_params()
  with pytest.raises(ValueError):
    kb_project(np.zeros((3,)), params)


def test_kb_unproject_rejects_bad_shape() -> None:
  params = _pinhole_params()
  with pytest.raises(ValueError):
    kb_unproject(np.zeros((3,)), params)
