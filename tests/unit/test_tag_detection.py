"""Tests for specialist.synthesis.tag_detection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from specialist.config import NoiseConfig, SceneConfig
from specialist.synthesis.scene import build_scene
from specialist.synthesis.tag_detection import TagObservation, synthesize_tag_detections
from tests._helpers import pose


def _visible_scene(tracker_names: tuple[str, ...] = ("left", "right")) -> tuple:
  """Scene tuned so trackers sit below an anchor whose normal points down."""
  cfg = SceneConfig(
    anchor_tag_xyz=(0.0, 0.0, 1.0),
    anchor_tag_rpy=(math.pi, 0.0, 0.0),  # normal = -Z: faces cameras below
    tag_on_tracker_xyz=(0.0, 0.05, 0.0),
    tag_on_tracker_rpy=(math.pi, 0.0, 0.0),  # normal = -Z too
    camera_front_xyz=(0.0, 0.0, 0.0),
    camera_front_rpy=(0.0, 0.0, 0.0),
    camera_back_xyz=(0.0, 0.0, 0.0),
    camera_back_rpy=(math.pi, 0.0, 0.0),
    fov=math.radians(195.0),
    min_distance=0.01,
    max_distance=5.0,
  )
  scene = build_scene(cfg, tracker_names=tracker_names)
  return scene, cfg


def test_emits_observations_for_visible_anchor() -> None:
  scene, _ = _visible_scene(("left",))
  keyframes = {"left": [pose((0, 0, 0), (0, 0, 0))]}
  obs = synthesize_tag_detections(
    keyframes, scene, noise=NoiseConfig(), rng=np.random.default_rng(0)
  )
  anchor_obs = [o for o in obs if o.tag_kind == "anchor"]
  assert len(anchor_obs) >= 1
  assert all(isinstance(o, TagObservation) for o in anchor_obs)


def test_self_tag_observations_excluded() -> None:
  scene, _ = _visible_scene(("left", "right"))
  keyframes = {
    "left": [pose((0, 0, 0), (0, 0, 0))],
    "right": [pose((0.3, 0, 0), (0, 0, 0))],
  }
  obs = synthesize_tag_detections(
    keyframes, scene, noise=NoiseConfig(), rng=np.random.default_rng(0)
  )
  for o in obs:
    if o.tag_kind == "tracker":
      assert o.tag_owner_tracker != o.observer_tracker


def test_tag_kind_classification() -> None:
  scene, _ = _visible_scene(("left", "right"))
  keyframes = {
    "left": [pose((0, 0, 0), (0, 0, 0))],
    "right": [pose((0.3, 0, 0), (0, 0, 0))],
  }
  obs = synthesize_tag_detections(
    keyframes, scene, noise=NoiseConfig(), rng=np.random.default_rng(0)
  )
  for o in obs:
    if o.tag_id == "anchor":
      assert o.tag_kind == "anchor"
      assert o.tag_owner_tracker is None
    elif o.tag_id.startswith("tracker_"):
      assert o.tag_kind == "tracker"
      assert o.tag_owner_tracker == o.tag_id.removeprefix("tracker_")


def test_zero_noise_recovers_ground_truth() -> None:
  scene, _ = _visible_scene(("left",))
  tracker_pose = pose((0, 0, 0), (0, 0, 0))
  keyframes = {"left": [tracker_pose]}
  zero_noise = NoiseConfig(
    anchor_translation=0.0,
    anchor_rotation=0.0,
    cross_tracker_translation=0.0,
    cross_tracker_rotation=0.0,
  )
  obs = synthesize_tag_detections(
    keyframes, scene, noise=zero_noise, rng=np.random.default_rng(0)
  )
  for o in obs:
    tracker_poses = {"left": tracker_pose}
    camera_world = scene.camera_world("left", o.fisheye_side, tracker_poses)
    tag_world = scene.tag_world(o.tag_id, tracker_poses)
    gt = camera_world.between(tag_world)
    assert np.allclose(o.camera_to_tag.translation(), gt.translation(), atol=1e-10)
    assert np.allclose(
      o.camera_to_tag.rotation().matrix(), gt.rotation().matrix(), atol=1e-10
    )


def test_observations_skipped_when_out_of_range() -> None:
  scene, _ = _visible_scene(("left",))
  far_keyframes = {"left": [pose((0, 0, -100), (0, 0, 0))]}
  obs = synthesize_tag_detections(
    far_keyframes, scene, noise=NoiseConfig(), rng=np.random.default_rng(0)
  )
  anchor_obs = [o for o in obs if o.tag_kind == "anchor"]
  assert len(anchor_obs) == 0


def test_cross_tracker_observations_present() -> None:
  scene, _ = _visible_scene(("left", "right"))
  keyframes = {
    "left": [pose((-0.1, 0, 0), (0, 0, 0))],
    "right": [pose((0.1, 0, 0), (0, 0, 0))],
  }
  obs = synthesize_tag_detections(
    keyframes, scene, noise=NoiseConfig(), rng=np.random.default_rng(0)
  )
  cross = [o for o in obs if o.tag_kind == "tracker"]
  assert len(cross) >= 1
  pairs = {(o.observer_tracker, o.tag_owner_tracker) for o in cross}
  assert ("left", "right") in pairs or ("right", "left") in pairs


def test_same_rng_same_observations() -> None:
  scene, _ = _visible_scene(("left",))
  keyframes = {"left": [pose((0, 0, 0), (0, 0, 0)), pose((0.01, 0, 0), (0, 0, 0))]}
  a = synthesize_tag_detections(
    keyframes, scene, noise=NoiseConfig(), rng=np.random.default_rng(99)
  )
  b = synthesize_tag_detections(
    keyframes, scene, noise=NoiseConfig(), rng=np.random.default_rng(99)
  )
  assert len(a) == len(b)
  for x, y in zip(a, b, strict=True):
    assert x.tag_id == y.tag_id
    assert x.observer_tracker == y.observer_tracker
    assert x.fisheye_side == y.fisheye_side
    assert x.keyframe_index == y.keyframe_index
    assert np.allclose(x.camera_to_tag.translation(), y.camera_to_tag.translation())


def test_inconsistent_keyframe_counts_raise() -> None:
  scene, _ = _visible_scene(("left", "right"))
  with pytest.raises(ValueError):
    synthesize_tag_detections(
      {
        "left": [pose((0, 0, 0), (0, 0, 0))],
        "right": [pose((0, 0, 0), (0, 0, 0)), pose((0.1, 0, 0), (0, 0, 0))],
      },
      scene,
      noise=NoiseConfig(),
      rng=np.random.default_rng(0),
    )
