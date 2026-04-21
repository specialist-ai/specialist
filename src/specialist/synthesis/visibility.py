"""Visibility predicate: does a tag appear in one fisheye at one instant?

A tag is visible iff all three conditions hold:

1. **Distance** — the camera-to-tag Euclidean distance is within
   ``[min_distance, max_distance]``.
2. **Angular FOV** — the angle between the camera's optical axis (local +Z)
   and the camera-to-tag direction is within ``fov / 2``.
3. **Tag facing** — the tag's normal (local +Z) points toward the camera, so
   the marker's visible face is presented (the reverse side is blank).

Frame conventions: GTSAM Pose3 follows the column-major rotation convention
where ``R[:, 2]`` is the +Z axis of the body in world coordinates. The
tracker body pose is composed with the fisheye's body-frame extrinsic to get
the camera's world pose; the tag's world pose provides its position and
normal directly.
"""

from __future__ import annotations

import gtsam
import numpy as np


def is_tag_visible(
  tracker_world: gtsam.Pose3,
  fisheye_extrinsic: gtsam.Pose3,
  tag_world: gtsam.Pose3,
  fov: float,
  min_distance: float,
  max_distance: float,
) -> bool:
  """Return True iff the tag is within the fisheye's visibility cone."""
  camera_world = tracker_world.compose(fisheye_extrinsic)
  cam_pos = np.asarray(camera_world.translation(), dtype=np.float64)
  tag_pos = np.asarray(tag_world.translation(), dtype=np.float64)
  delta = tag_pos - cam_pos
  distance = float(np.linalg.norm(delta))

  if distance < min_distance or distance > max_distance:
    return False

  direction = delta / distance
  cam_axis = np.asarray(camera_world.rotation().matrix()[:, 2], dtype=np.float64)
  cos_angle = float(np.dot(cam_axis, direction))
  cos_half_fov = float(np.cos(fov / 2.0))
  if cos_angle < cos_half_fov:
    return False

  tag_normal = np.asarray(tag_world.rotation().matrix()[:, 2], dtype=np.float64)
  to_camera = -direction
  if float(np.dot(tag_normal, to_camera)) <= 0.0:
    return False

  return True
