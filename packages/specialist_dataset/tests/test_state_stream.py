"""The rows a state stream keeps, and the hand each camera rode."""

from __future__ import annotations

import numpy as np
import pytest
from specialist_dataset.state import (
  finger,
  hands,
  image_frames,
  lattice_brackets,
  lerp,
  nearest,
  opening,
)


def test_lattice_brackets_place_each_row_between_its_two_samples() -> None:
  times = np.array([0.0002, 0.0012, 0.0021, 0.0032, 0.0044])
  rows, before, weight = lattice_brackets(times, 1000.0, (0.0, 1.0))
  np.testing.assert_array_equal(rows, [1, 2, 3, 4])
  np.testing.assert_array_equal(before, [0, 1, 2, 3])
  np.testing.assert_allclose(weight, [0.8, 0.8 / 0.9, 0.9 / 1.1, 0.8 / 1.2])
  # A series linear in time comes back exactly at the instants, whatever its shape.
  np.testing.assert_allclose(lerp(times * 1000.0, before, weight), [1.0, 2.0, 3.0, 4.0])
  np.testing.assert_allclose(
    lerp(np.stack([times, -times], axis=1) * 1000.0, before, weight)[:, 1],
    [-1.0, -2.0, -3.0, -4.0],
  )
  # A row on a sample is that sample.
  rows, before, weight = lattice_brackets(
    np.array([0.0, 0.001, 0.002]), 1000.0, (0.0, 1.0)
  )
  np.testing.assert_array_equal(rows, [0, 1, 2])
  np.testing.assert_allclose(weight, [0.0, 0.0, 1.0])
  # Only the rows inside the span are served, and a gap outside it is no gap.
  rows, _, _ = lattice_brackets(
    np.array([0.0, 0.001, 0.004, 0.005]), 1000.0, (0.0035, 0.01)
  )
  np.testing.assert_array_equal(rows, [4, 5])
  with pytest.raises(ValueError, match="gap"):
    lattice_brackets(np.array([0.0, 0.001, 0.004]), 1000.0, (0.0, 1.0))
  with pytest.raises(ValueError, match="strictly increasing"):
    lattice_brackets(np.array([0.0, 0.001, 0.001]), 1000.0, (0.0, 1.0))
  with pytest.raises(ValueError, match="no row"):
    lattice_brackets(times, 1000.0, (0.0041, 0.0042))


def test_nearest_refuses_nothing_to_search() -> None:
  with pytest.raises(ValueError, match="no times"):
    nearest(np.array([]), np.array([1.0]))


def test_opening_is_half_the_mirrored_finger_difference() -> None:
  t = np.arange(4) * 0.1
  left = finger(t, np.array([0.2, 0.3, 0.1, 0.0]))
  right = finger(t, -np.array([0.2, 0.3, 0.1, 0.0]))
  np.testing.assert_allclose(opening(left, right, t), [0.2, 0.3, 0.1, 0.0])
  # A row one finger missed interpolates from that finger's own neighbouring rows.
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


def test_image_frames_keeps_every_frame_inside_the_window() -> None:
  fps = 30.0
  times = (np.arange(-3, 21) / fps).astype(np.float64)
  frames = image_frames({"cam00": times}, fps, (0.0, 0.5))["cam00"]
  # Frames 0..15 fall in [0.0, 0.5] s to half a frame; frame 16 is at 0.533 s.
  np.testing.assert_allclose(times[frames] * fps, np.arange(16), atol=1e-9)


def test_image_frames_are_one_set_of_session_frames_every_lens_has() -> None:
  """An early lens and an off-grid lens keep the same session frames."""
  fps = 30.0
  early = np.arange(-7, 30) / fps
  late = np.arange(0, 30) / fps + 0.004
  frames = image_frames({"a": early, "b": late}, fps, (-0.5, 0.75))
  np.testing.assert_allclose(early[frames["a"]] * fps, np.arange(24), atol=1e-6)
  np.testing.assert_allclose(late[frames["b"]] * fps, np.arange(24) + 0.12, atol=1e-6)
  with pytest.raises(ValueError, match="window"):
    image_frames({"a": early}, fps, (5.0, 6.0))


def _take(left_y: float, right_y: float, offset: float = 0.0) -> tuple[dict, dict]:
  """Two grippers 0.5 m ahead of the anchor along +x; +y is the operator's left."""
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
