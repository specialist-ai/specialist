"""Tests for track-length-adaptive reprojection-noise scaling."""

from __future__ import annotations

import math

from specialist.vi_ba.build import _reproj_noise_scale


def test_scale_one_at_reference() -> None:
  assert math.isclose(_reproj_noise_scale(8, ref_obs=8, max_mult=4.0), 1.0)


def test_short_track_inflated() -> None:
  s = _reproj_noise_scale(2, ref_obs=8, max_mult=4.0)
  assert s > 1.0  # few observations -> less reliable -> looser


def test_long_track_tightened_but_floored() -> None:
  s = _reproj_noise_scale(64, ref_obs=8, max_mult=4.0)
  assert s < 1.0
  assert s >= 0.5  # floor so we never wildly over-trust


def test_clamped_to_max() -> None:
  assert _reproj_noise_scale(1, ref_obs=8, max_mult=4.0) <= 4.0


def test_monotonic_decreasing_in_obs() -> None:
  vals = [_reproj_noise_scale(k, ref_obs=8, max_mult=4.0) for k in (2, 4, 8, 16, 32)]
  assert all(a >= b for a, b in zip(vals, vals[1:], strict=False))
