"""Each tracker and lens owns a disjoint slice of the integer-key space."""

from __future__ import annotations

import gtsam
import pytest

from specialist.vi_ba.compose import (
  _MAX_LANDMARKS_PER_TRACKER,
  LANDMARK_CHAR,
  TrackerKeys,
)


def test_the_lens_slots_do_not_overlap() -> None:
  keys = TrackerKeys(["tracker_0", "tracker_1"])
  issued = {
    keys.front_landmark_key("tracker_0", 7),
    keys.rear_landmark_key("tracker_0", 7),
    keys.front_landmark_key("tracker_1", 7),
    keys.rear_landmark_key("tracker_1", 7),
  }
  assert len(issued) == 4


def test_a_track_id_that_outgrows_its_slot_is_refused() -> None:
  """Ids grow monotonically, so a long capture can reach the next tracker's slot."""
  keys = TrackerKeys(["tracker_0", "tracker_1"])
  assert keys.front_landmark_key("tracker_0", _MAX_LANDMARKS_PER_TRACKER - 1)
  for track_id in (_MAX_LANDMARKS_PER_TRACKER, _MAX_LANDMARKS_PER_TRACKER + 1, -1):
    with pytest.raises(ValueError, match="landmark slot"):
      keys.front_landmark_key("tracker_0", track_id)


def test_the_collision_it_prevents_is_real() -> None:
  """An id one past the front slot produces exactly the rear slot's key."""
  keys = TrackerKeys(["tracker_0", "tracker_1"])
  overflowed = int(gtsam.symbol(LANDMARK_CHAR, _MAX_LANDMARKS_PER_TRACKER + 5))
  assert overflowed == keys.rear_landmark_key("tracker_0", 5)
