"""Tests for specialist.synthesis.visibility."""

from __future__ import annotations

import math

import gtsam

from specialist.synthesis.visibility import is_tag_visible
from tests._helpers import pose

FOV = math.pi  # 180°
MIN = 0.1
MAX = 5.0
IDENTITY = gtsam.Pose3()
PI = math.pi


def test_tag_straight_ahead_visible() -> None:
  tracker = pose((0, 0, 0), (0, 0, 0))
  tag = pose((0, 0, 1.0), (PI, 0, 0))  # tag +Z flipped to face camera
  assert is_tag_visible(tracker, IDENTITY, tag, FOV, MIN, MAX)


def test_tag_too_close() -> None:
  tracker = pose((0, 0, 0), (0, 0, 0))
  tag = pose((0, 0, 0.05), (PI, 0, 0))
  assert not is_tag_visible(tracker, IDENTITY, tag, FOV, MIN, MAX)


def test_tag_too_far() -> None:
  tracker = pose((0, 0, 0), (0, 0, 0))
  tag = pose((0, 0, 10.0), (PI, 0, 0))
  assert not is_tag_visible(tracker, IDENTITY, tag, FOV, MIN, MAX)


def test_tag_outside_narrow_fov() -> None:
  tracker = pose((0, 0, 0), (0, 0, 0))
  tag = pose((3.0, 0, 1.0), (PI, 0, 0))
  narrow = math.radians(30.0)
  assert not is_tag_visible(tracker, IDENTITY, tag, narrow, MIN, MAX)


def test_tag_same_pos_wide_fov_visible() -> None:
  tracker = pose((0, 0, 0), (0, 0, 0))
  tag = pose((3.0, 0, 1.0), (PI, 0, 0))
  wide = math.radians(150.0)
  assert is_tag_visible(tracker, IDENTITY, tag, wide, MIN, MAX)


def test_tag_facing_away_invisible() -> None:
  """Tag normal pointing away from camera means the back of the tag is presented."""
  tracker = pose((0, 0, 0), (0, 0, 0))
  tag = pose((0, 0, 1.0), (0, 0, 0))  # tag +Z is +Z world, same as view direction
  assert not is_tag_visible(tracker, IDENTITY, tag, FOV, MIN, MAX)


def test_back_fisheye_extrinsic() -> None:
  """Fisheye extrinsic that flips the camera should see tags behind the tracker."""
  tracker = pose((0, 0, 0), (0, 0, 0))
  back_fisheye = pose((0, 0, 0), (PI, 0, 0))
  tag_behind = pose(
    (0, 0, -1.0), (0, 0, 0)
  )  # tag +Z = world +Z, camera now faces -Z → sees tag
  assert is_tag_visible(tracker, back_fisheye, tag_behind, FOV, MIN, MAX)


def test_moving_tracker_shifts_visibility() -> None:
  tracker_a = pose((0, 0, 0), (0, 0, 0))
  tracker_b = pose((0, 0, 4.5), (0, 0, 0))
  tag = pose((0, 0, 1.0), (PI, 0, 0))
  assert is_tag_visible(tracker_a, IDENTITY, tag, FOV, MIN, MAX)
  # From tracker_b, tag is at z = -3.5 in world; camera facing +z won't see it.
  assert not is_tag_visible(tracker_b, IDENTITY, tag, FOV, MIN, MAX)


def test_distance_at_boundary() -> None:
  """Boundary: distance == min or max should be considered visible."""
  tracker = pose((0, 0, 0), (0, 0, 0))
  tag_min = pose((0, 0, MIN), (PI, 0, 0))
  tag_max = pose((0, 0, MAX), (PI, 0, 0))
  assert is_tag_visible(tracker, IDENTITY, tag_min, FOV, MIN, MAX)
  assert is_tag_visible(tracker, IDENTITY, tag_max, FOV, MIN, MAX)
