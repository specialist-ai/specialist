"""Evidence that the decomposed seams are drivable one at a time."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gtsam
import numpy as np
import pytest
from rerun.experimental import Chunk

from specialist.vi_ba.viz import (
  CameraEstimate,
  Layers,
  LensEstimate,
  _camera_indexes,
  _gripper_streams,
  _trajectory_streams,
  _urdf_model,
)
from tests.record_fixtures import urdf_file


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


def test_a_urdf_that_cannot_be_read_stops_the_run(tmp_path: Path) -> None:
  """A URDF that fails to parse raises."""
  broken = tmp_path / "broken.urdf"
  broken.write_text("<robot name=")
  chunks: list[Chunk] = []
  with pytest.raises(RuntimeError):
    _urdf_model(Layers(chunks.append, chunks.append), _data(), broken)


def test_a_missing_mesh_costs_pictures_not_answers(tmp_path: Path) -> None:
  """A mesh that cannot be read still leaves the joints, their definitions and
  the model's provenance in the layer."""
  path = tmp_path / "robot.urdf"
  path.write_text(
    '<?xml version="1.0"?><robot name="gripper">'
    '<link name="base"><visual><geometry>'
    '<mesh filename="nowhere.stl"/></geometry></visual></link>'
    "</robot>"
  )
  chunks: list[Chunk] = []
  tree, root_frame = _urdf_model(Layers(chunks.append, chunks.append), _data(), path)
  assert tree.name == "gripper"
  assert root_frame == "cam00_base"
  written = {
    name
    for chunk in chunks
    for name in chunk.to_record_batch().schema.names
    if not name.startswith("rerun.controls.")
  }
  assert "urdf_sha" in written, "the model's provenance did not survive"


def _child_frames(chunks: list[Chunk]) -> set[str]:
  """Every frame the chunks define an edge onto."""
  frames = set()
  for chunk in chunks:
    batch = chunk.to_record_batch()
    if "Transform3D:child_frame" in batch.schema.names:
      column = batch.column("Transform3D:child_frame").to_pylist()
      frames |= {f for row in column if row for f in row}
  return frames


def test_geometry_that_cannot_be_read_does_not_cost_the_fixed_joint_edges(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A fixed joint places the tag frame the solve reprojects onto, and the edge
  survives geometry the reader cannot open."""
  path = urdf_file(tmp_path)
  from rerun.urdf import UrdfTree

  def boom(self: Any, link: Any) -> list[str]:
    raise RuntimeError("mesh unreadable")

  monkeypatch.setattr(UrdfTree, "get_collision_geometry_paths", boom)
  chunks: list[Chunk] = []
  _urdf_model(Layers(chunks.append, chunks.append), _data(), path)
  frames = _child_frames(chunks)
  assert "cam00_base_tag" in frames, f"the fixed joint edge is gone: {sorted(frames)}"


def test_a_missing_mesh_does_not_cost_the_fixed_joint_edges(tmp_path: Path) -> None:
  """A fixed joint places the tag frame the solve reprojects onto, so it is an
  answer, and it survives geometry the reader cannot open."""
  path = tmp_path / "robot.urdf"
  path.write_text(
    '<?xml version="1.0"?><robot name="gripper">'
    '<link name="base"><visual><geometry>'
    '<mesh filename="nowhere.stl"/></geometry></visual></link>'
    '<link name="base_tag"/>'
    '<joint name="base_tag" type="fixed"><parent link="base"/>'
    '<child link="base_tag"/><origin xyz="0 0.01 0.02" rpy="0 0 0"/></joint>'
    "</robot>"
  )
  chunks: list[Chunk] = []
  _urdf_model(Layers(chunks.append, chunks.append), _data(), path)
  frames = set()
  for chunk in chunks:
    batch = chunk.to_record_batch()
    if "Transform3D:child_frame" in batch.schema.names:
      column = batch.column("Transform3D:child_frame").to_pylist()
      frames |= {f for row in column if row for f in row}
  assert "cam00_base_tag" in frames, f"the fixed joint edge is gone: {sorted(frames)}"


def _strips(chunks: list[Chunk]) -> list[Any]:
  """The ribbon's line strips, whichever chunk carries them: the entity also
  holds a frame declaration and a colour, each in its own chunk."""
  for chunk in chunks:
    batch = chunk.to_record_batch()
    if "LineStrips3D:strips" in batch.schema.names:
      return list(batch.column("LineStrips3D:strips").to_pylist())
  raise AssertionError("no chunk carried the ribbon")


def test_the_propagated_gripper_frame_is_what_the_trajectory_rides(
  tmp_path: Path,
) -> None:
  """The rotations and translations `_gripper_streams` returns are the frame the
  trail's keypoint offset is applied in, so the two agree by construction."""
  data = _data()
  indexes = _camera_indexes(data, 1.0 / 30.0, timeline_start=0.0)
  gripper_hr = _gripper_streams(
    Layers(lambda _: None, lambda _: None), data, indexes, "cam00_base"
  )
  assert gripper_hr is not None
  translations, rotations = gripper_hr
  assert translations.shape == (5, 3)
  assert rotations.shape == (5, 3, 3)

  chunks: list[Chunk] = []
  _trajectory_streams(Layers(chunks.append, chunks.append), data, gripper_hr, (1, 2, 3))
  offset = np.asarray(data.trajectory_offset_in_gripper)
  expected = translations + rotations @ offset
  strips = _strips(chunks)
  assert np.allclose([s[0][-1] for s in strips], expected)
