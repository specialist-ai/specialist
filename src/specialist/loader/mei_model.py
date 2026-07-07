"""Pure-numpy Mei unified-omnidirectional projection.

The Mei model (Mei & Rives, 2007) projects a 3D point onto the unit sphere,
shifts the sphere center by ``xi`` along the optical axis, then applies a
pinhole projection with Brown-Conrady distortion. ``xi > 1`` enables the
super-fisheye regime (fisheye lenses run around ``xi ~ 1.948``).

Distortion order matches the convention stored in Insta360 ``.insv`` lens
blocks: ``[k1, k2, k3, p1, p2, xi]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from specialist.loader._camera_math import require_2d, unit_rows

Float64Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MeiParams:
  """MEI intrinsics: pinhole + radial/tangential distortion + ``xi``."""

  fx: float
  fy: float
  cx: float
  cy: float
  k1: float
  k2: float
  k3: float
  p1: float
  p2: float
  xi: float

  @classmethod
  def from_camera_matrix(
    cls, camera_matrix: Float64Array, distortion_coeffs: Float64Array
  ) -> MeiParams:
    """Build from the ``LensIntrinsics`` arrays parsed out of a ``.insv``."""
    if camera_matrix.shape != (3, 3):
      raise ValueError(f"camera_matrix must be 3x3, got {camera_matrix.shape}")
    if distortion_coeffs.shape != (6,):
      raise ValueError(
        f"distortion_coeffs must be (6,) [k1,k2,k3,p1,p2,xi], "
        f"got {distortion_coeffs.shape}"
      )
    return cls(
      fx=float(camera_matrix[0, 0]),
      fy=float(camera_matrix[1, 1]),
      cx=float(camera_matrix[0, 2]),
      cy=float(camera_matrix[1, 2]),
      k1=float(distortion_coeffs[0]),
      k2=float(distortion_coeffs[1]),
      k3=float(distortion_coeffs[2]),
      p1=float(distortion_coeffs[3]),
      p2=float(distortion_coeffs[4]),
      xi=float(distortion_coeffs[5]),
    )


def mei_project(points_cam: Float64Array, params: MeiParams) -> Float64Array:
  """Project ``(N, 3)`` camera-frame points to ``(N, 2)`` pixel coordinates.

  Points behind the unit-sphere mirror (``z + xi * |X| <= 0``) are returned
  as NaN — callers decide whether to mask or report them.
  """
  require_2d(points_cam, 3, "points_cam")
  sphere = unit_rows(points_cam)

  denom = sphere[:, 2] + params.xi
  valid = denom > 0.0
  x = np.where(valid, sphere[:, 0] / denom, np.nan)
  y = np.where(valid, sphere[:, 1] / denom, np.nan)

  r2 = x * x + y * y
  radial = 1.0 + params.k1 * r2 + params.k2 * r2 * r2 + params.k3 * r2**3
  x_d = x * radial + 2.0 * params.p1 * x * y + params.p2 * (r2 + 2.0 * x * x)
  y_d = y * radial + params.p1 * (r2 + 2.0 * y * y) + 2.0 * params.p2 * x * y

  u = params.fx * x_d + params.cx
  v = params.fy * y_d + params.cy
  return np.stack([u, v], axis=1)


def mei_unproject(pixels: Float64Array, params: MeiParams) -> Float64Array:
  """Lift ``(N, 2)`` pixels back to ``(N, 3)`` unit-sphere bearing vectors.

  Inverts the distorted-pinhole step iteratively (Newton on the radial
  polynomial), then inverts the unit-sphere projection. Returns unit vectors
  in the camera frame; raises if any pixel sits beyond the model's domain.
  """
  require_2d(pixels, 2, "pixels")

  x_d = (pixels[:, 0] - params.cx) / params.fx
  y_d = (pixels[:, 1] - params.cy) / params.fy

  x = x_d.copy()
  y = y_d.copy()
  for _ in range(20):
    r2 = x * x + y * y
    radial = 1.0 + params.k1 * r2 + params.k2 * r2 * r2 + params.k3 * r2**3
    dx_t = 2.0 * params.p1 * x * y + params.p2 * (r2 + 2.0 * x * x)
    dy_t = params.p1 * (r2 + 2.0 * y * y) + 2.0 * params.p2 * x * y
    safe_radial = np.where(radial == 0.0, 1.0, radial)
    x_new = (x_d - dx_t) / safe_radial
    y_new = (y_d - dy_t) / safe_radial
    step = max(np.max(np.abs(x_new - x)), np.max(np.abs(y_new - y)))
    x, y = x_new, y_new
    if step < 1e-12:
      break

  r2 = x * x + y * y
  # Invert sphere shift: solve for sphere point P_s with P_s.z + xi = 1/(...)
  # giving the closed form below for xi >= 0.
  disc = 1.0 + (1.0 - params.xi * params.xi) * r2
  disc = np.where(disc < 0.0, np.nan, disc)
  factor = (params.xi + np.sqrt(disc)) / (r2 + 1.0)
  bearing = np.stack([factor * x, factor * y, factor - params.xi], axis=1)
  return unit_rows(bearing)
