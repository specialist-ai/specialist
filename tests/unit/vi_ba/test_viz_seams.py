"""The decomposed viz seams are drivable one at a time."""

from __future__ import annotations

from typing import Any

import gtsam
import numpy as np
from rerun.chunk import Chunk

from specialist.vi_ba.viz import (
  CameraEstimate,
  Layers,
  LensEstimate,
  _camera_indexes,
  _gripper_streams,
  _trajectory_streams,
)


def _data() -> CameraEstimate:
  times = np.linspace(0.0, 0.2, 3)
  poses = [gtsam.Pose3(gtsam.Rot3(), np.array([0.1 * i, 0.0, 0.0])) for i in range(3)]
  hr = np.linspace(0.0, 0.2, 5)
  data = CameraEstimate(
    slot="cam00",
    body_poses=poses,
    keyframe_times=times,
    body_T_base_tag=gtsam.Pose3(),
    body_T_rear_lens=gtsam.Pose3(),
    base_tag_id=2000,
    keyframe_rows=np.arange(len(times), dtype=np.int64) * 3,
    highrate_times=hr,
    highrate_positions=np.stack([hr, np.zeros(5), np.zeros(5)], axis=1),
    highrate_quats=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (5, 1)),
    highrate_vels=np.zeros((5, 3)),
  )
  data.lenses = {"front": LensEstimate()}
  data.trajectory_offset_in_gripper = (0.0, 0.0, 0.05)
  return data


def _strips(chunks: list[Chunk]) -> list[Any]:
  """The ribbon's line strips, from whichever of the entity's chunks carries them."""
  for chunk in chunks:
    batch = chunk.to_record_batch()
    if "LineStrips3D:strips" in batch.schema.names:
      return list(batch.column("LineStrips3D:strips").to_pylist())
  raise AssertionError("no chunk carried the ribbon")


def test_the_propagated_gripper_frame_is_what_the_trajectory_rides() -> None:
  """The trail applies its keypoint offset in the frame ``_gripper_streams`` returns."""
  data = _data()
  indexes = _camera_indexes(data, 1.0 / 30.0, timeline_start=0.0)
  gripper_hr = _gripper_streams(Layers([], []), data, indexes, "cam00_base")
  assert gripper_hr is not None
  translations, rotations = gripper_hr
  assert translations.shape == (5, 3)
  assert rotations.shape == (5, 3, 3)

  chunks: list[Chunk] = []
  _trajectory_streams(Layers(chunks, chunks), data, gripper_hr, (1, 2, 3))
  offset = np.asarray(data.trajectory_offset_in_gripper)
  expected = translations + rotations @ offset
  strips = _strips(chunks)
  assert np.allclose([s[0][-1] for s in strips], expected)
