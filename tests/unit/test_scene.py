"""Tests for specialist.synthesis.scene."""

from __future__ import annotations

import gtsam
import numpy as np
import pytest

from specialist.config import SceneConfig
from specialist.synthesis.scene import Scene, build_scene


def _identity_pose() -> gtsam.Pose3:
  return gtsam.Pose3()


def test_build_scene_from_default_config() -> None:
  scene = build_scene(SceneConfig(), tracker_names=("left", "right"))
  assert isinstance(scene, Scene)
  assert scene.tracker_names == ("left", "right")
  assert isinstance(scene.anchor_pose, gtsam.Pose3)
  assert isinstance(scene.tag_on_tracker, gtsam.Pose3)


def test_anchor_pose_matches_config() -> None:
  import math

  cfg = SceneConfig(
    anchor_tag_xyz=(0.5, -0.25, 1.5),
    anchor_tag_rpy=(0.0, 0.0, math.pi / 2),
  )
  scene = build_scene(cfg, tracker_names=("a",))
  t = scene.anchor_pose.translation()
  assert np.allclose(t, (0.5, -0.25, 1.5))


def test_tag_ids() -> None:
  scene = build_scene(SceneConfig(), tracker_names=("left", "right"))
  assert scene.anchor_tag_id == "anchor"
  assert scene.tracker_tag_id("left") == "tracker_left"
  assert scene.tracker_tag_id("right") == "tracker_right"


def test_tag_world_anchor_is_constant() -> None:
  scene = build_scene(SceneConfig(), tracker_names=("left",))
  tracker_poses = {"left": gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1.0, 2.0, 3.0))}
  anchor_world = scene.tag_world(scene.anchor_tag_id, tracker_poses)
  assert np.allclose(anchor_world.translation(), scene.anchor_pose.translation())


def test_tag_world_tracker_composes() -> None:
  cfg = SceneConfig(
    tag_on_tracker_xyz=(0.1, 0.0, 0.0), tag_on_tracker_rpy=(0.0, 0.0, 0.0)
  )
  scene = build_scene(cfg, tracker_names=("left",))
  tracker = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1.0, 0.0, 0.0))
  tag_world = scene.tag_world("tracker_left", {"left": tracker})
  expected = tracker.compose(scene.tag_on_tracker)
  assert np.allclose(tag_world.translation(), expected.translation())
  assert np.allclose(tag_world.rotation().matrix(), expected.rotation().matrix())


def test_tag_world_unknown_raises() -> None:
  scene = build_scene(SceneConfig(), tracker_names=("left",))
  with pytest.raises(KeyError):
    scene.tag_world("mystery", {"left": _identity_pose()})


def test_camera_world_front_and_back_differ() -> None:
  scene = build_scene(SceneConfig(), tracker_names=("left",))
  tracker = {"left": _identity_pose()}
  cam_f = scene.camera_world("left", "front", tracker)
  cam_b = scene.camera_world("left", "back", tracker)
  assert not np.allclose(cam_f.translation(), cam_b.translation()) or not np.allclose(
    cam_f.rotation().matrix(), cam_b.rotation().matrix()
  )


def test_camera_world_composes_with_tracker() -> None:
  cfg = SceneConfig(camera_front_xyz=(0.0, 0.0, 0.1), camera_front_rpy=(0.0, 0.0, 0.0))
  scene = build_scene(cfg, tracker_names=("left",))
  tracker = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1.0, 2.0, 3.0))
  cam = scene.camera_world("left", "front", {"left": tracker})
  expected_t = np.asarray(tracker.translation()) + np.array([0.0, 0.0, 0.1])
  assert np.allclose(cam.translation(), expected_t)


def test_camera_world_unknown_tracker_raises() -> None:
  scene = build_scene(SceneConfig(), tracker_names=("left",))
  with pytest.raises(KeyError):
    scene.camera_world("right", "front", {"left": _identity_pose()})


def test_fisheye_extrinsics_property_returns_both() -> None:
  scene = build_scene(SceneConfig(), tracker_names=("left",))
  front, back = scene.fisheye_extrinsics
  assert isinstance(front, gtsam.Pose3)
  assert isinstance(back, gtsam.Pose3)


def test_tag_id_pattern_roundtrip() -> None:
  """Regex patterns from config are exposed on the scene for the detection module."""
  scene = build_scene(SceneConfig(), tracker_names=("left",))
  import re

  assert re.match(scene.anchor_tag_id_pattern, "anchor")
  m = re.match(scene.tracker_tag_id_pattern, "tracker_left")
  assert m is not None and m.group(1) == "left"
