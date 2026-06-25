"""Tests for the native MEI projection factor (``_mei_factor_cpp``).

A binary GTSAM factor on ``(body_pose, world_landmark)`` with the body → lens
extrinsic baked in at construction. Residual:
``mei_project((T_world_body · T_body_lens)⁻¹ · L) − pixel``.

Guardrail properties:

1. **Zero residual** at the exact projection, with both a front (identity)
   and a rear (roll 90°, tz −0.026 m) extrinsic.
2. **Jacobian self-consistency**: the factor's linearization, evaluated at
   ≥5 points spanning the FOV including one beyond 90° half-angle, predicts
   the residual change over a small tangent step to within `1e-4` absolute.
   This finite-difference check independently validates the analytical
   Jacobians compiled into the C++ factor.
3. **Cheirality past 90°**: ``mei_project`` + ±1 px noise + ``mei_unproject``
   at 95° half-angle recovers a bearing in the same forward hemisphere as
   the input — a standing guardrail for the super-fisheye regime.
"""

from __future__ import annotations

import math

import gtsam
import numpy as np
import numpy.typing as npt
import pytest

from specialist.loader.mei_model import MeiParams, mei_project, mei_unproject

try:
  from specialist.vi_ba._mei_factor_cpp import make_mei_factor_cpp
except ImportError:
  make_mei_factor_cpp = None

pytestmark = pytest.mark.skipif(
  make_mei_factor_cpp is None,
  reason="native MEI factor not built (run `uv run build-mei-factor`)",
)

_POSE_KEY = 0
_LM_KEY = 1


def _params() -> MeiParams:
  # Realistic values (from the committed offset_v3 fixture).
  return MeiParams(
    fx=4625.45,
    fy=4623.58,
    cx=2968.61,
    cy=2994.87,
    k1=0.39252487,
    k2=1.23661351,
    k3=-3.86204076,
    p1=-1.075e-05,
    p2=-0.00215951,
    xi=1.94817,
  )


def _noise() -> gtsam.noiseModel.Base:
  # Isotropic σ=1 px → whitened == unwhitened; makes Jacobian introspection easy.
  return gtsam.noiseModel.Isotropic.Sigma(2, 1.0)


def _factor(
  pixel: npt.NDArray[np.float64], extrinsic: gtsam.Pose3, params: MeiParams
) -> gtsam.NonlinearFactor:
  p = params
  return make_mei_factor_cpp(
    _POSE_KEY,
    _LM_KEY,
    np.asarray(pixel, dtype=np.float64).reshape(2),
    _noise(),
    extrinsic,
    p.fx,
    p.fy,
    p.cx,
    p.cy,
    p.k1,
    p.k2,
    p.k3,
    p.p1,
    p.p2,
    p.xi,
  )


def _rear_extrinsic() -> gtsam.Pose3:
  return gtsam.Pose3(
    gtsam.Rot3.Ypr(0.0, 0.0, math.pi / 2), gtsam.Point3(0.0, 0.0, -0.026248)
  )


def _values(pose: gtsam.Pose3, lm: npt.NDArray[np.float64]) -> gtsam.Values:
  v = gtsam.Values()
  v.insert(_POSE_KEY, pose)
  v.insert(_LM_KEY, gtsam.Point3(float(lm[0]), float(lm[1]), float(lm[2])))
  return v


def _project_through(
  body_pose: gtsam.Pose3,
  landmark_world: npt.NDArray[np.float64],
  extrinsic: gtsam.Pose3,
  params: MeiParams,
) -> npt.NDArray[np.float64]:
  T_world_lens = body_pose.compose(extrinsic)
  p_lens = T_world_lens.transformTo(
    gtsam.Point3(
      float(landmark_world[0]), float(landmark_world[1]), float(landmark_world[2])
    )
  )
  return mei_project(np.asarray(p_lens)[None, :], params)[0]


def test_zero_residual_front_identity_extrinsic() -> None:
  params = _params()
  extrinsic = gtsam.Pose3()
  body_pose = gtsam.Pose3()
  landmark = np.array([0.3, 0.1, 2.0])  # in front of lens
  pixel = _project_through(body_pose, landmark, extrinsic, params)

  factor = _factor(pixel, extrinsic, params)
  err = factor.unwhitenedError(_values(body_pose, landmark))
  assert np.allclose(err, 0.0, atol=1e-10)


def test_zero_residual_rear_extrinsic() -> None:
  params = _params()
  extrinsic = _rear_extrinsic()
  body_pose = gtsam.Pose3()
  # A point in front of the *rear* lens (forward = extrinsic.rotation() * +z):
  # Put the landmark 1.5 m out along the lens's +z, then into world.
  p_in_lens = np.array([0.1, -0.1, 1.5])
  T_world_lens = extrinsic  # body == world here
  landmark_world = np.asarray(T_world_lens.transformFrom(gtsam.Point3(*p_in_lens)))
  pixel = _project_through(body_pose, landmark_world, extrinsic, params)

  factor = _factor(pixel, extrinsic, params)
  err = factor.unwhitenedError(_values(body_pose, landmark_world))
  assert np.allclose(err, 0.0, atol=1e-10)


def test_nonzero_residual_when_pose_perturbed() -> None:
  params = _params()
  extrinsic = gtsam.Pose3()
  body_pose = gtsam.Pose3()
  landmark = np.array([0.3, 0.1, 2.0])
  pixel = _project_through(body_pose, landmark, extrinsic, params)

  factor = _factor(pixel, extrinsic, params)

  perturbed = body_pose.retract(np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0]))
  err = factor.unwhitenedError(_values(perturbed, landmark))
  assert np.linalg.norm(err) > 1.0  # 5 cm translation ≫ 1 px noise


_FOV_LANDMARKS: tuple[tuple[npt.NDArray[np.float64], str], ...] = (
  (np.array([0.01, 0.01, 2.0]), "near-axial"),
  (np.array([0.5, 0.2, 1.5]), "~20° off axis"),
  (np.array([1.0, 0.3, 0.9]), "~50° off axis"),
  (np.array([1.6, 0.0, 0.3]), "~80° off axis"),
  (np.array([1.0, 0.0, -0.1]), "past 90° half-angle (super-fisheye)"),
)


def _assert_fd_jacobian(
  factor: gtsam.NonlinearFactor,
  body_pose: gtsam.Pose3,
  landmark: npt.NDArray[np.float64],
  jac: npt.NDArray[np.float64],
  kind: str,
  label: str,
) -> None:
  """Finite-difference check: ``jac · δ ≈ r(x ⊞ δ) − r(x)`` along each axis of a
  ``"pose"`` (retract) or ``"landmark"`` (additive) perturbation."""
  base = factor.unwhitenedError(_values(body_pose, landmark))
  for axis in range(jac.shape[1]):
    delta = np.zeros(jac.shape[1])
    delta[axis] = 1e-5
    stepped = (
      _values(body_pose.retract(delta), landmark)
      if kind == "pose"
      else _values(body_pose, landmark + delta)
    )
    observed = factor.unwhitenedError(stepped) - base
    assert np.allclose(jac @ delta, observed, atol=1e-4), (
      f"{label}: {kind} axis {axis}: pred={jac @ delta}, obs={observed}"
    )


def test_jacobian_self_consistency_across_fov() -> None:
  """At ≥5 FOV-spanning landmarks, J·δ ≈ r(x ⊞ δ) − r(x) for small δ."""
  params = _params()
  extrinsic = gtsam.Pose3()
  body_pose = gtsam.Pose3()

  for landmark, label in _FOV_LANDMARKS:
    pixel = _project_through(body_pose, landmark, extrinsic, params)
    # Shift pixel so the base residual is non-zero, mirroring how LM evaluates
    # the Jacobian away from the optimum during optimization.
    factor = _factor(pixel + np.array([2.0, -3.0]), extrinsic, params)
    a, _ = factor.linearize(_values(body_pose, landmark)).jacobian()
    assert a.shape == (2, 9), f"{label}: unexpected Jacobian shape {a.shape}"
    _assert_fd_jacobian(factor, body_pose, landmark, a[:, :6], "pose", label)
    _assert_fd_jacobian(factor, body_pose, landmark, a[:, 6:], "landmark", label)


def test_mei_cheirality_past_90_degrees_preserves_hemisphere() -> None:
  """Project a 95°-half-angle ray + ±1 px noise + unproject ⇒ same hemisphere.

  Standing guardrail for the super-fisheye cheirality regime.
  """
  params = _params()
  rng = np.random.default_rng(seed=42)

  half_angle = math.radians(95.0)
  # A ray at 95° half-angle: z is negative, xy magnitude is large.
  base_ray = np.array([math.sin(half_angle), 0.0, math.cos(half_angle)])
  assert base_ray[2] < 0.0  # past horizon; sanity.

  # Project at ~1.5 m depth, collect pixels over 100 noise realizations.
  depth = 1.5
  cam_points = (base_ray * depth)[None, :]
  pixel_true = mei_project(cam_points, params)[0]
  assert np.all(np.isfinite(pixel_true)), "MEI must represent 95° half-angle"

  for _ in range(100):
    noise = rng.normal(scale=1.0, size=2)
    pixel_noisy = pixel_true + noise
    bearing = mei_unproject(pixel_noisy[None, :], params)[0]
    # Hemisphere check: recovered z should be close to base z, NOT flipped sign.
    assert bearing[2] < 0.2, (
      f"cheirality flip: base z={base_ray[2]:.3f}, recovered z={bearing[2]:.3f}"
    )
    assert np.linalg.norm(bearing - base_ray) < 0.05
