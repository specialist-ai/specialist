"""Evidence that the decomposed seams are drivable one at a time."""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
from rerun.experimental import Chunk

from specialist.vi_ba.viz import (
  LensVizData,
  TrackerVizData,
  _gripper_streams,
  _tracker_indexes,
  _trajectory_streams,
  _urdf_model,
)


def _data(highrate: bool) -> TrackerVizData:
  times = np.linspace(0.0, 0.2, 3)
  poses = [gtsam.Pose3(gtsam.Rot3(), np.array([0.1 * i, 0.0, 0.0])) for i in range(3)]
  data = TrackerVizData(name="t0", body_poses=poses, keyframe_times=times)
  data.lenses = {"front": LensVizData()}
  data.urdf_root_poses = list(poses)
  data.trajectory_offset_in_gripper = (0.0, 0.0, 0.05)
  if highrate:
    hr = np.linspace(0.0, 0.2, 5)
    data.highrate_times = hr
    data.highrate_positions = np.stack([hr, np.zeros(5), np.zeros(5)], axis=1)
    data.highrate_quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (5, 1))
  return data


def test_a_urdf_that_cannot_be_read_places_no_model(tmp_path: Path) -> None:
  broken = tmp_path / "broken.urdf"
  broken.write_text("<robot name=")
  chunks: list[Chunk] = []
  assert _urdf_model(chunks.append, chunks.append, _data(False), broken) is None
  assert chunks == []


def test_the_propagated_gripper_frame_is_what_the_trajectory_rides(
  tmp_path: Path,
) -> None:
  """The rotations and translations `_gripper_streams` returns are the frame the
  trail's keypoint offset is applied in, so the two agree by construction."""
  data = _data(True)
  indexes = _tracker_indexes(data, 1.0 / 30.0)
  gripper_hr = _gripper_streams(lambda _: None, data, indexes, "t0_base")
  assert gripper_hr is not None
  translations, rotations = gripper_hr
  assert translations.shape == (5, 3)
  assert rotations.shape == (5, 3, 3)

  chunks: list[Chunk] = []
  _trajectory_streams(chunks.append, data, indexes, gripper_hr, (1, 2, 3))
  offset = np.asarray(data.trajectory_offset_in_gripper)
  expected = translations + rotations @ offset
  strips = chunks[0].to_record_batch().column("LineStrips3D:strips").to_pylist()
  assert np.allclose([s[0][-1] for s in strips], expected)


def test_without_an_imu_the_trail_falls_back_to_the_keyframes() -> None:
  data = _data(False)
  indexes = _tracker_indexes(data, 1.0 / 30.0)
  chunks: list[Chunk] = []
  _trajectory_streams(chunks.append, data, indexes, None, (1, 2, 3))
  strips = chunks[0].to_record_batch().column("LineStrips3D:strips").to_pylist()
  assert len(strips) == len(data.keyframe_times)
