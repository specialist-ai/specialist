"""Tests for specialist.graph.keys."""

from __future__ import annotations

import gtsam
import pytest

from specialist.graph.keys import KeyMap


def test_first_come_first_served_char_assignment() -> None:
  km = KeyMap(("left", "right"))
  assert km.tracker_char("left") == "a"
  assert km.tracker_char("right") == "b"


def test_char_stable_on_rebuild() -> None:
  km1 = KeyMap(("left", "right"))
  km2 = KeyMap(("left", "right"))
  assert km1.tracker_char("left") == km2.tracker_char("left")
  assert km1.tracker_pose_key("left", 5) == km2.tracker_pose_key("left", 5)


def test_order_changes_char() -> None:
  km_left_first = KeyMap(("left", "right"))
  km_right_first = KeyMap(("right", "left"))
  assert km_left_first.tracker_char("left") != km_right_first.tracker_char("left")


def test_tracker_pose_key_round_trips_through_gtsam_symbol() -> None:
  km = KeyMap(("left",))
  key = km.tracker_pose_key("left", 42)
  assert key == gtsam.symbol("a", 42)


def test_anchor_key_distinct_from_tracker_keys() -> None:
  km = KeyMap(("left", "right"))
  assert km.anchor_key() != km.tracker_pose_key("left", 0)
  assert km.anchor_key() != km.tracker_pose_key("right", 0)


def test_anchor_key_stable() -> None:
  assert KeyMap(("a",)).anchor_key() == KeyMap(("b",)).anchor_key()


def test_unknown_tracker_raises() -> None:
  km = KeyMap(("left",))
  with pytest.raises(KeyError):
    km.tracker_pose_key("right", 0)
  with pytest.raises(KeyError):
    km.tracker_char("right")


def test_duplicate_names_rejected() -> None:
  with pytest.raises(ValueError):
    KeyMap(("left", "left"))


def test_too_many_trackers_rejected() -> None:
  with pytest.raises(ValueError):
    KeyMap(tuple(f"t{i}" for i in range(27)))


def test_tracker_char_never_collides_with_anchor() -> None:
  km = KeyMap(tuple(f"t{i}" for i in range(26)))
  chars = {km.tracker_char(f"t{i}") for i in range(26)}
  # Anchor uses uppercase letter, trackers use lowercase — so no overlap.
  assert not any(c.isupper() for c in chars)
