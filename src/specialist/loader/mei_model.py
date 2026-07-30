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

__all__ = ["MeiParams", "mei_project", "mei_unproject"]


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
  in the camera frame; pixels beyond the model's domain (negative
  discriminant) come back NaN, matching ``mei_project``.
  """
  require_2d(pixels, 2, "pixels")

  x_d = (pixels[:, 0] - params.cx) / params.fx
  y_d = (pixels[:, 1] - params.cy) / params.fy

  x, y = x_d, y_d
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
  # Invert the sphere shift: writing the sphere point as
  # (factor * x, factor * y, factor - xi), unit norm makes factor a root of
  # factor**2 * (r2 + 1) - 2 * factor * xi + xi**2 - 1 = 0. The larger root is
  # the sphere intersection the forward projection came from; both roots are
  # positive once xi > 1. A negative discriminant puts the pixel outside the
  # model's domain, so the bearing comes back NaN.
  disc = 1.0 + (1.0 - params.xi * params.xi) * r2
  disc = np.where(disc < 0.0, np.nan, disc)
  factor = (params.xi + np.sqrt(disc)) / (r2 + 1.0)
  bearing = np.stack([factor * x, factor * y, factor - params.xi], axis=1)
  return unit_rows(bearing)
