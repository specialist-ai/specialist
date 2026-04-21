"""Smooth SE(3) trajectory synthesis via cubic splines.

Random control points in a bounded workspace drive a natural cubic spline in
R^3 for position and a ``scipy.spatial.transform.RotationSpline`` (cubic in
rotation vector) for orientation. Both splines are evaluated in closed form,
so linear and angular velocity come from analytic derivatives rather than
finite differences.

Angular velocity is returned in the **body frame** — matching the convention
of ``RotationSpline``'s first-order derivative: ``R(t+dt) ≈ R(t) · exp([ω · dt]×)``.
"""

from __future__ import annotations

import gtsam
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline


class SmoothTrajectory:
  """Smooth SE(3) trajectory on ``[0, duration]``.

  The trajectory is defined by ``n_control_points`` uniformly-spaced control
  poses sampled from a uniform distribution over
  ``workspace_center + [-workspace_radius, +workspace_radius]^3`` in
  position and a uniform distribution over SO(3) in orientation. Between
  control points the trajectory is C² smooth.
  """

  def __init__(
    self,
    duration: float,
    workspace_radius: float,
    rng: np.random.Generator,
    n_control_points: int = 8,
    workspace_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
  ) -> None:
    if duration <= 0.0:
      raise ValueError(f"duration must be positive, got {duration}")
    if workspace_radius <= 0.0:
      raise ValueError(f"workspace_radius must be positive, got {workspace_radius}")
    if n_control_points < 4:
      raise ValueError(
        f"n_control_points must be >= 4 for cubic splines, got {n_control_points}"
      )

    self.duration = float(duration)
    t_ctrl = np.linspace(0.0, self.duration, n_control_points)
    p_ctrl = rng.uniform(
      -workspace_radius, workspace_radius, size=(n_control_points, 3)
    ) + np.asarray(workspace_center, dtype=np.float64)
    r_ctrl = Rotation.random(n_control_points, random_state=rng)
    self._position = CubicSpline(t_ctrl, p_ctrl, bc_type="natural")
    self._rotation = RotationSpline(t_ctrl, r_ctrl)

  def _check_t(self, t: float) -> float:
    if t < 0.0 or t > self.duration:
      raise ValueError(f"t={t} outside trajectory [0, {self.duration}]")
    return float(t)

  def pose(self, t: float) -> gtsam.Pose3:
    """World-from-body SE(3) pose at time ``t``."""
    t = self._check_t(t)
    pos = np.asarray(self._position(t), dtype=np.float64)
    rot_matrix = np.asarray(self._rotation(t).as_matrix(), dtype=np.float64)
    return gtsam.Pose3(gtsam.Rot3(rot_matrix), gtsam.Point3(*pos))

  def velocity(self, t: float) -> NDArray[np.float64]:
    """World-frame linear velocity (dp/dt) at time ``t``, shape ``(3,)``."""
    t = self._check_t(t)
    return np.asarray(self._position(t, 1), dtype=np.float64)

  def angular_velocity(self, t: float) -> NDArray[np.float64]:
    """Body-frame angular velocity at time ``t``, shape ``(3,)``."""
    t = self._check_t(t)
    return np.asarray(self._rotation(t, 1), dtype=np.float64)
