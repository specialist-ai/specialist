"""Pure-numpy Kannala-Brandt projection.

KB (Kannala & Brandt, 2006) is a fisheye model that expresses the angle
``theta`` between the incoming ray and the optical axis as an odd polynomial
in radius:

    theta_d = theta * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8),  t = theta

then maps the distorted radius onto the image plane via ``(fx, fy, cx, cy)``.

This module supplies the projection + unprojection used by ``camera_model.py``
when fitting a KB approximation to factory MEI intrinsics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class KbParams:
  """KB intrinsics: pinhole + four polynomial coefficients."""

  fx: float
  fy: float
  cx: float
  cy: float
  k1: float
  k2: float
  k3: float
  k4: float

  def to_array(self) -> Float64Array:
    return np.array(
      [self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.k3, self.k4],
      dtype=np.float64,
    )

  @classmethod
  def from_array(cls, x: Float64Array) -> KbParams:
    if x.shape != (8,):
      raise ValueError(f"KbParams.from_array expects (8,), got {x.shape}")
    return cls(
      fx=float(x[0]),
      fy=float(x[1]),
      cx=float(x[2]),
      cy=float(x[3]),
      k1=float(x[4]),
      k2=float(x[5]),
      k3=float(x[6]),
      k4=float(x[7]),
    )


def kb_project(points_cam: Float64Array, params: KbParams) -> Float64Array:
  """Project ``(N, 3)`` camera-frame points to ``(N, 2)`` pixel coordinates.

  KB handles back-of-camera rays (``z < 0``) — angle is measured from the
  optical axis and can exceed ``pi/2``.
  """
  if points_cam.ndim != 2 or points_cam.shape[1] != 3:
    raise ValueError(f"points_cam must be (N, 3), got {points_cam.shape}")

  x = points_cam[:, 0]
  y = points_cam[:, 1]
  z = points_cam[:, 2]
  r = np.sqrt(x * x + y * y)
  theta = np.arctan2(r, z)
  theta2 = theta * theta
  theta_d = theta * (
    1.0
    + params.k1 * theta2
    + params.k2 * theta2 * theta2
    + params.k3 * theta2**3
    + params.k4 * theta2**4
  )

  small_r = r < 1e-12
  scale = np.where(small_r, 0.0, theta_d / np.where(small_r, 1.0, r))
  u = params.fx * (scale * x) + params.cx
  v = params.fy * (scale * y) + params.cy
  return np.stack([u, v], axis=1)


def kb_unproject(pixels: Float64Array, params: KbParams) -> Float64Array:
  """Lift ``(N, 2)`` pixels back to ``(N, 3)`` unit bearing vectors.

  Inverts the polynomial radius -> angle relationship by Newton iteration.
  """
  if pixels.ndim != 2 or pixels.shape[1] != 2:
    raise ValueError(f"pixels must be (N, 2), got {pixels.shape}")

  x = (pixels[:, 0] - params.cx) / params.fx
  y = (pixels[:, 1] - params.cy) / params.fy
  theta_d = np.sqrt(x * x + y * y)

  theta = theta_d.copy()
  for _ in range(20):
    t2 = theta * theta
    poly = (
      1.0 + params.k1 * t2 + params.k2 * t2 * t2 + params.k3 * t2**3 + params.k4 * t2**4
    )
    deriv = (
      1.0
      + 3.0 * params.k1 * t2
      + 5.0 * params.k2 * t2 * t2
      + 7.0 * params.k3 * t2**3
      + 9.0 * params.k4 * t2**4
    )
    residual = theta * poly - theta_d
    step = residual / np.where(deriv == 0.0, 1.0, deriv)
    theta = theta - step
    if np.max(np.abs(step)) < 1e-12:
      break

  small = theta_d < 1e-12
  sin_t = np.sin(theta)
  cos_t = np.cos(theta)
  scale = np.where(small, 0.0, sin_t / np.where(small, 1.0, theta_d))
  bearing = np.stack([scale * x, scale * y, cos_t], axis=1)
  norms = np.linalg.norm(bearing, axis=1, keepdims=True)
  norms = np.where(norms == 0.0, 1.0, norms)
  return bearing / norms
