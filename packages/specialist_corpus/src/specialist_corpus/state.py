"""Which rows a state stream keeps, how a dense series lands on the step
lattice, the gripper's opening, and which camera the policy saw a hand from.
Pure numpy."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

__all__ = ["finger", "hands", "image_rows", "lattice_mean", "nearest", "opening"]

_PAIR_TOLERANCE = 0.05
_MIN_ANGLE = 3.0
_MIN_AGREEMENT = 0.75


def nearest(
  times: Float64Array, queries: Float64Array
) -> tuple[Int64Array, Float64Array]:
  """The index into ascending ``times`` nearest each query, and the gap."""
  if not len(times):
    raise ValueError("no times to search")
  hi = np.clip(np.searchsorted(times, queries), 0, len(times) - 1)
  lo = np.clip(hi - 1, 0, len(times) - 1)
  index = np.where(np.abs(times[hi] - queries) < np.abs(times[lo] - queries), hi, lo)
  return index.astype(np.int64), np.abs(times[index] - queries)


def image_rows(
  times: dict[str, Float64Array],
  fps: float,
  stride: int,
  window: tuple[float, float],
  grid: Int64Array,
) -> dict[str, Int64Array]:
  """Each lens's frame rows the stream keeps: one set of session rows
  (``round(t * fps)``) that every lens has a frame on, out of every
  ``stride``-th row of the estimate's keyframe rows ``grid`` from its first,
  whose nominal time lies inside ``window`` to half a frame."""
  if stride < 1:
    raise ValueError(f"stride must be at least 1, got {stride}")
  wanted = grid[(grid - grid[0]) % stride == 0]
  wanted = wanted[
    (wanted >= math.ceil(window[0] * fps - 0.5))
    & (wanted <= math.floor(window[1] * fps + 0.5))
  ]
  session = {s: np.rint(np.asarray(t) * fps).astype(np.int64) for s, t in times.items()}
  for rows in session.values():
    wanted = wanted[np.isin(wanted, rows)]
  if not wanted.size:
    raise ValueError(f"no frame of every lens falls in the window {window} s")
  return {
    s: np.searchsorted(rows, wanted).astype(np.int64) for s, rows in session.items()
  }


def lattice_mean(
  times: Float64Array, values: Float64Array, f_step: float
) -> tuple[Float64Array, Int64Array, Float64Array]:
  """``values`` averaged onto the ``1 / f_step`` lattice of the session clock:
  ``(times, steps, means)``. Each row is the mean of the samples whose nearest
  lattice point is that step, stamped at the mean of their own times; the first
  and last steps, covered by only part of their window, are dropped."""
  t = np.asarray(times, dtype=np.float64)
  v = np.asarray(values, dtype=np.float64)
  steps = np.rint(t * f_step).astype(np.int64)
  labels, first = np.unique(steps, return_index=True)
  if len(labels) <= 2:
    empty = np.empty(0)
    return empty, np.empty(0, dtype=np.int64), np.empty((0, *v.shape[1:]))
  counts = np.diff(np.append(first, len(steps)))
  sums = np.add.reduceat(v, first, axis=0)
  t_sums = np.add.reduceat(t, first)
  keep = slice(1, -1)
  shape = (-1, *(1,) * (v.ndim - 1))
  means: Float64Array = sums[keep] / counts[keep].reshape(shape)
  centres: Float64Array = t_sums[keep] / counts[keep]
  return centres, labels[keep], means


def finger(
  times: Float64Array, angles: Float64Array
) -> tuple[Float64Array, Float64Array]:
  """One finger's angle series on the rows its tag was seen: the non-finite
  rows dropped, so a span and an interpolation read only measurements."""
  t = np.asarray(times, dtype=np.float64)
  a = np.asarray(angles, dtype=np.float64)
  keep = np.isfinite(a)
  if not keep.any():
    raise ValueError("the finger was never seen")
  return t[keep], a[keep]


def opening(
  left: tuple[Float64Array, Float64Array],
  right: tuple[Float64Array, Float64Array],
  at: Float64Array,
) -> Float64Array:
  """The coupled finger joint angle on the ``at`` rows: the two fingers mirror
  each other, so ``q = (left - right) / 2`` in the left joint's sign, each
  finger's angle interpolated from the rows its tag was seen at, never past
  its own span."""
  (t_left, a_left), (t_right, a_right) = left, right
  if at[0] < max(t_left[0], t_right[0]) or at[-1] > min(t_left[-1], t_right[-1]):
    raise ValueError("a row falls outside the solved fingers' span")
  return (np.interp(at, t_left, a_left) - np.interp(at, t_right, a_right)) / 2.0


def hands(
  times: dict[str, Float64Array], translations: dict[str, Float64Array]
) -> tuple[str, str]:
  """``(left, right)`` camera slots of a two-camera take, from the keyframe
  times and world positions.

  The reference anchor is behind the operator, so seen from the world origin
  the gripper counter-clockwise of the other is on the operator's left. A
  mean angle under 3 degrees between the grippers, or a side the paired
  keyframes agree on less than three quarters of the time, is refused.
  """
  if len(times) != 2:
    raise ValueError(f"hand detection needs two cameras, got {sorted(times)}")
  a, b = sorted(times)
  ta, tb = (
    np.asarray(times[a], dtype=np.float64),
    np.asarray(times[b], dtype=np.float64),
  )
  index, gap = nearest(tb, ta)
  paired = gap <= _PAIR_TOLERANCE
  pa = np.asarray(translations[a], dtype=np.float64)[paired, :2]
  pb = np.asarray(translations[b], dtype=np.float64)[index[paired], :2]
  norm = np.linalg.norm(pa, axis=1) * np.linalg.norm(pb, axis=1)
  sine = (pa[:, 0] * pb[:, 1] - pa[:, 1] * pb[:, 0])[norm > 0] / norm[norm > 0]
  if not sine.size:
    raise ValueError(f"{a} and {b} share no keyframes to compare")
  mean = float(sine.mean())
  angle = math.degrees(math.asin(max(-1.0, min(1.0, mean))))
  agreement = float(np.mean(np.sign(sine) == np.sign(mean)))
  if abs(angle) < _MIN_ANGLE or agreement < _MIN_AGREEMENT:
    raise ValueError(
      f"cannot tell the hands apart: seen from the anchor {b} sits {angle:+.1f} deg "
      f"from {a}, and the keyframes agree on the side {agreement:.0%} of the time"
    )
  return (b, a) if mean > 0.0 else (a, b)
