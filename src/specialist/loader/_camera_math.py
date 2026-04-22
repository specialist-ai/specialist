"""Shared numpy helpers for the fisheye camera-model loaders (MEI, KB)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]


def unit_rows(v: Float64Array) -> Float64Array:
  """Normalise each row of ``v`` to unit length, leaving zero rows unchanged."""
  norms = np.linalg.norm(v, axis=1, keepdims=True)
  norms = np.where(norms == 0.0, 1.0, norms)
  return v / norms


def require_2d(arr: Float64Array, cols: int, name: str) -> None:
  """Validate that ``arr`` is ``(N, cols)``; raise ValueError otherwise."""
  if arr.ndim != 2 or arr.shape[1] != cols:
    raise ValueError(f"{name} must be (N, {cols}), got {arr.shape}")
