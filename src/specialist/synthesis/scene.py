"""Scene geometry: anchor tag pose and on-tracker extrinsics.

The scene is a rigid collection of poses — the anchor tag in the world frame,
and a shared tag + two fisheye cameras (Insta360 back-to-back) in the tracker
body frame. All trackers carry identical rigs; they differ only in their
trajectories. Tag IDs follow the patterns in ``SceneConfig`` so bundles can
extend without code changes: the anchor matches ``^anchor$`` and per-tracker
tags match ``^tracker_(\\w+)$``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import gtsam

from specialist.config import SceneConfig

Vec3 = tuple[float, float, float]
FisheyeSide = Literal["front", "back"]


def _pose_from_xyz_rpy(xyz: Vec3, rpy: Vec3) -> gtsam.Pose3:
  rot = gtsam.Rot3.RzRyRx(float(rpy[0]), float(rpy[1]), float(rpy[2]))
  return gtsam.Pose3(rot, gtsam.Point3(float(xyz[0]), float(xyz[1]), float(xyz[2])))


@dataclass(eq=False)
class Scene:
  """Rigid geometry for a scenario: anchor + per-tracker extrinsics.

  ``tag_on_tracker`` and the two fisheye extrinsics are shared across all
  trackers — trackers are physically identical, they only differ in their
  world-frame trajectories.
  """

  anchor_pose: gtsam.Pose3
  tag_on_tracker: gtsam.Pose3
  camera_front_on_tracker: gtsam.Pose3
  camera_back_on_tracker: gtsam.Pose3
  tracker_names: tuple[str, ...]
  fov: float
  min_distance: float
  max_distance: float
  anchor_tag_id_pattern: str
  tracker_tag_id_pattern: str
  anchor_tag_id: str = "anchor"

  @property
  def fisheye_extrinsics(self) -> tuple[gtsam.Pose3, gtsam.Pose3]:
    return (self.camera_front_on_tracker, self.camera_back_on_tracker)

  def tracker_tag_id(self, tracker_name: str) -> str:
    return f"tracker_{tracker_name}"

  def tag_world(
    self, tag_id: str, tracker_poses: Mapping[str, gtsam.Pose3]
  ) -> gtsam.Pose3:
    """World-frame pose of ``tag_id`` given current world-frame tracker poses."""
    if tag_id == self.anchor_tag_id:
      return self.anchor_pose
    for name in self.tracker_names:
      if tag_id == self.tracker_tag_id(name):
        return tracker_poses[name].compose(self.tag_on_tracker)
    raise KeyError(f"unknown tag_id: {tag_id!r}")

  def camera_world(
    self,
    tracker_name: str,
    side: FisheyeSide,
    tracker_poses: Mapping[str, gtsam.Pose3],
  ) -> gtsam.Pose3:
    """World-frame pose of one fisheye camera on ``tracker_name``."""
    if tracker_name not in tracker_poses:
      raise KeyError(f"no pose for tracker {tracker_name!r}")
    extrinsic = (
      self.camera_front_on_tracker if side == "front" else self.camera_back_on_tracker
    )
    return tracker_poses[tracker_name].compose(extrinsic)


def build_scene(config: SceneConfig, tracker_names: tuple[str, ...]) -> Scene:
  """Materialize a ``Scene`` from the flat ``SceneConfig`` + tracker names."""
  return Scene(
    anchor_pose=_pose_from_xyz_rpy(config.anchor_tag_xyz, config.anchor_tag_rpy),
    tag_on_tracker=_pose_from_xyz_rpy(
      config.tag_on_tracker_xyz, config.tag_on_tracker_rpy
    ),
    camera_front_on_tracker=_pose_from_xyz_rpy(
      config.camera_front_xyz, config.camera_front_rpy
    ),
    camera_back_on_tracker=_pose_from_xyz_rpy(
      config.camera_back_xyz, config.camera_back_rpy
    ),
    tracker_names=tuple(tracker_names),
    fov=config.fov,
    min_distance=config.min_distance,
    max_distance=config.max_distance,
    anchor_tag_id_pattern=config.anchor_tag_id_pattern,
    tracker_tag_id_pattern=config.tracker_tag_id_pattern,
  )
