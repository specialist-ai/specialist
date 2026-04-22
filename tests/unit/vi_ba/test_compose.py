"""``TrackerKeys``, the collision-free GTSAM key allocator for multi-tracker VI-BA."""

from __future__ import annotations

from specialist.vi_ba.compose import TrackerKeys


def test_tracker_keys_no_collision_across_trackers() -> None:
  keys = TrackerKeys(["alice", "bob"])
  ka_pose0 = keys.pose_key("alice", 0)
  kb_pose0 = keys.pose_key("bob", 0)
  assert ka_pose0 != kb_pose0
  assert keys.pose_key("alice", 0) != keys.velocity_key("alice", 0)
  assert keys.pose_key("alice", 0) != keys.bias_key("alice", 0)
  assert keys.front_landmark_key("alice", 0) != keys.rear_landmark_key("alice", 0)
  assert keys.front_landmark_key("alice", 0) != keys.front_landmark_key("bob", 0)
  # Anchor distinct from everything.
  anchor = keys.anchor_key()
  assert anchor not in {
    ka_pose0,
    kb_pose0,
    keys.velocity_key("alice", 0),
    keys.bias_key("alice", 0),
    keys.front_landmark_key("alice", 0),
  }


def test_tracker_keys_unknown_tracker_raises() -> None:
  keys = TrackerKeys(["alice"])
  try:
    keys.pose_key("bob", 0)
  except KeyError:
    return
  raise AssertionError("expected KeyError for unknown tracker")


def test_tracker_keys_duplicate_names_raises() -> None:
  try:
    TrackerKeys(["alice", "alice"])
  except ValueError:
    return
  raise AssertionError("expected ValueError for duplicate tracker names")
