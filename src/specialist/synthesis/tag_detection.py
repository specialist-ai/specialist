"""Per-fisheye SE(3) tag observation synthesis.

For every (observer_tracker, fisheye_side, keyframe_index, tag_id) tuple,
if the tag is visible from that fisheye at that keyframe, emit a noisy
camera→tag SE(3) observation. Tag IDs are classified via the regex patterns
on ``Scene``:

- Anchor observations use the anchor noise model.
- Cross-tracker observations use the cross-tracker noise model.

Self-tag observations (observer_tracker == tag_owner_tracker) are excluded —
they carry no trajectory information given the fixed, pre-calibrated tag
extrinsic on each tracker.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import gtsam
import numpy as np

from specialist.config import NoiseConfig
from specialist.synthesis.noise import add_pose_noise
from specialist.synthesis.scene import FisheyeSide, Scene
from specialist.synthesis.visibility import is_tag_visible

TagKind = Literal["anchor", "tracker"]

_SIDES: tuple[FisheyeSide, ...] = ("front", "back")


@dataclass(eq=False)
class TagObservation:
  """One noisy camera→tag SE(3) observation from one fisheye at one keyframe."""

  observer_tracker: str
  fisheye_side: FisheyeSide
  keyframe_index: int
  tag_id: str
  tag_kind: TagKind
  tag_owner_tracker: str | None
  camera_to_tag: gtsam.Pose3


def _classify_tag(tag_id: str, scene: Scene) -> tuple[TagKind, str | None]:
  if re.fullmatch(scene.anchor_tag_id_pattern, tag_id):
    return ("anchor", None)
  m = re.fullmatch(scene.tracker_tag_id_pattern, tag_id)
  if m is not None:
    return ("tracker", m.group(1))
  raise KeyError(f"tag_id {tag_id!r} matches neither anchor nor tracker pattern")


def synthesize_tag_detections(
  keyframes_per_tracker: Mapping[str, Sequence[gtsam.Pose3]],
  scene: Scene,
  noise: NoiseConfig,
  rng: np.random.Generator,
) -> list[TagObservation]:
  """Emit per-fisheye SE(3) tag observations across every keyframe."""
  tracker_names = tuple(keyframes_per_tracker.keys())
  if not tracker_names:
    return []

  n_keyframes = len(next(iter(keyframes_per_tracker.values())))
  for name, poses in keyframes_per_tracker.items():
    if len(poses) != n_keyframes:
      raise ValueError(
        f"tracker {name!r} has {len(poses)} keyframes; expected {n_keyframes}"
      )

  all_tag_ids: list[str] = [scene.anchor_tag_id] + [
    scene.tracker_tag_id(n) for n in scene.tracker_names
  ]
  sides_and_extrinsics = tuple(zip(_SIDES, scene.fisheye_extrinsics, strict=True))

  observations: list[TagObservation] = []
  for k in range(n_keyframes):
    tracker_poses_at_k = {n: keyframes_per_tracker[n][k] for n in tracker_names}
    for observer in tracker_names:
      tracker_world = tracker_poses_at_k[observer]
      for side, extrinsic in sides_and_extrinsics:
        camera_world = scene.camera_world(observer, side, tracker_poses_at_k)
        for tag_id in all_tag_ids:
          kind, owner = _classify_tag(tag_id, scene)
          if kind == "tracker" and owner == observer:
            continue
          tag_world = scene.tag_world(tag_id, tracker_poses_at_k)
          if not is_tag_visible(
            tracker_world,
            extrinsic,
            tag_world,
            scene.fov,
            scene.min_distance,
            scene.max_distance,
          ):
            continue
          gt_cam_to_tag = camera_world.between(tag_world)
          if kind == "anchor":
            t_sigma = noise.anchor_translation
            r_sigma = noise.anchor_rotation
          else:
            t_sigma = noise.cross_tracker_translation
            r_sigma = noise.cross_tracker_rotation
          noisy = add_pose_noise(gt_cam_to_tag, t_sigma, r_sigma, rng)
          observations.append(
            TagObservation(
              observer_tracker=observer,
              fisheye_side=side,
              keyframe_index=k,
              tag_id=tag_id,
              tag_kind=kind,
              tag_owner_tracker=owner,
              camera_to_tag=noisy,
            )
          )
  return observations
