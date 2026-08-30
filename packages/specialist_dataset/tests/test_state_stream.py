"""The rows a state stream keeps, and the hand each camera rode."""

from __future__ import annotations

import numpy as np
import pytest
from specialist_dataset.state import (
  finger,
  hands,
  image_rows,
  lattice_mean,
  nearest,
  opening,
)


def test_lattice_mean_averages_around_each_step_and_stamps_the_true_time() -> None:
  """Steps 0 and 3 are covered by only part of their window, so they go; each
  kept row is the mean of the samples nearest its step, stamped at the mean of
  their own times, never at the lattice point."""
  times = np.array([0.002, 0.006, 0.011, 0.014, 0.019, 0.023, 0.031])
  values = np.array([[4.0], [9.0], [1.0], [2.0], [3.0], [10.0], [7.0]])
  centres, steps, means = lattice_mean(times, values, 100.0)
  np.testing.assert_array_equal(steps, [1, 2])
  np.testing.assert_allclose(
    centres, [(0.006 + 0.011 + 0.014) / 3, (0.019 + 0.023) / 2]
  )
  np.testing.assert_allclose(means[:, 0], [4.0, 6.5])
  # A series of one column keeps its shape, and a hole in the samples is a
  # missing step, never a wrong mean.
  gappy = np.array([0.002, 0.011, 0.032, 0.041])
  _, steps, means = lattice_mean(gappy, np.array([1.0, 2.0, 3.0, 4.0]), 100.0)
  np.testing.assert_array_equal(steps, [1, 3])
  np.testing.assert_allclose(means, [2.0, 3.0])
  # Fewer than three steps leaves nothing whole.
  assert lattice_mean(times[:3], values[:3], 100.0)[1].size == 0


def test_nearest_refuses_nothing_to_search() -> None:
  with pytest.raises(ValueError, match="no times"):
    nearest(np.array([]), np.array([1.0]))


def test_opening_is_half_the_mirrored_finger_difference() -> None:
  t = np.arange(4) * 0.1
  left = finger(t, np.array([0.2, 0.3, 0.1, 0.0]))
  right = finger(t, -np.array([0.2, 0.3, 0.1, 0.0]))
  np.testing.assert_allclose(opening(left, right, t), [0.2, 0.3, 0.1, 0.0])
  # Each finger is interpolated from the rows its own tag was seen at, so a
  # row one finger missed reads through its neighbours.
  nan = float("nan")
  left = finger(t, np.array([0.2, nan, 0.0, 0.0]))
  np.testing.assert_allclose(left[0], t[[0, 2, 3]])
  np.testing.assert_allclose(
    opening(left, right, np.array([0.1])), [(0.1 - (-0.3)) / 2]
  )
  # Never past either finger's span.
  with pytest.raises(ValueError, match="span"):
    opening(left, finger(t[:3], np.zeros(3)), t)
  with pytest.raises(ValueError, match="never seen"):
    finger(t, np.full(4, nan))


def test_image_rows_keeps_every_stride_th_keyframe_row_inside_the_window() -> None:
  fps = 30.0
  # Frames from session row -3 to row 20, at the frame period; keyframes on
  # every third row from row 0.
  times = (np.arange(-3, 21) / fps).astype(np.float64)
  grid = np.arange(0, 21, 3)
  rows = image_rows({"cam00": times}, fps, 3, (0.0, 0.5), grid)["cam00"]
  # Session rows 0, 3, 6, 9, 12, 15 fall in [0.0, 0.5] s; row 18 is at 0.6 s.
  np.testing.assert_allclose(times[rows] * fps, [0, 3, 6, 9, 12, 15], atol=1e-9)
  rows = image_rows({"cam00": times}, fps, 6, (0.0, 0.5), grid)["cam00"]
  np.testing.assert_allclose(times[rows] * fps, [0, 6, 12], atol=1e-9)
  # A grid off the session lattice keeps its own phase.
  rows = image_rows({"cam00": times}, fps, 6, (0.0, 0.5), grid + 2)["cam00"]
  np.testing.assert_allclose(times[rows] * fps, [2, 8, 14], atol=1e-9)


def test_image_rows_are_one_set_of_session_rows_every_lens_has() -> None:
  """One lens starts mid-stride and earlier, and the other's frame times sit a
  hair off the grid; both keep the same session rows."""
  fps = 30.0
  early = np.arange(-7, 30) / fps
  late = np.arange(0, 30) / fps + 0.004
  grid = np.arange(-9, 31, 3)
  rows = image_rows({"a": early, "b": late}, fps, 3, (-0.5, 0.75), grid)
  np.testing.assert_allclose(early[rows["a"]] * fps, np.arange(0, 23, 3), atol=1e-6)
  np.testing.assert_allclose(
    late[rows["b"]] * fps, np.arange(0, 23, 3) + 0.12, atol=1e-6
  )
  with pytest.raises(ValueError, match="window"):
    image_rows({"a": early}, fps, 3, (5.0, 6.0), grid)
  with pytest.raises(ValueError, match="stride"):
    image_rows({"a": early}, fps, 0, (0.0, 1.0), grid)


def _take(left_y: float, right_y: float, offset: float = 0.0) -> tuple[dict, dict]:
  """Two grippers 0.5 m in front of the anchor along +x, one at each ``y``;
  with the operator facing +x, +y is their left."""
  t = np.arange(20) * 0.1
  wobble = 0.01 * np.sin(np.arange(20))
  times = {"cam00": t, "cam01": t + offset}
  translations = {
    "cam00": np.stack([0.5 + wobble, np.full(20, left_y), np.zeros(20)], axis=1),
    "cam01": np.stack([0.5 - wobble, np.full(20, right_y), np.zeros(20)], axis=1),
  }
  return times, translations


def test_hands_name_the_counter_clockwise_gripper_left() -> None:
  assert hands(*_take(+0.1, -0.1)) == ("cam00", "cam01")
  assert hands(*_take(-0.1, +0.1)) == ("cam01", "cam00")
  assert hands(*_take(+0.1, -0.1, offset=0.01)) == ("cam00", "cam01")


def test_hands_refuse_a_thin_margin_or_a_split_verdict() -> None:
  with pytest.raises(ValueError, match="cannot tell"):
    hands(*_take(+0.01, -0.005))
  times, translations = _take(+0.1, -0.1)
  translations["cam01"][::2, 1] = +0.3
  with pytest.raises(ValueError, match="cannot tell"):
    hands(times, translations)
  with pytest.raises(ValueError, match="two cameras"):
    hands({"cam00": times["cam00"]}, {"cam00": translations["cam00"]})
