"""The naming authority: how a camera earns its slot, and what a slot is not."""

from __future__ import annotations

import numpy as np
import pytest

from specialist.record import schema


def test_the_slot_is_the_serial_rank_not_the_argument_order() -> None:
  """Cameras are handed to conversion in whatever order the command listed
  them. Ranking their serials is what makes a re-conversion of the same
  cameras hand out the same slots, so a corpus keyed by slot cannot be
  re-numbered by a differently written command line."""
  serials = ["IAQEB2601V3F6K", "IAQEB2601UVVAJ", "IAQEB2601A0000"]
  assert schema.camera_slots(serials) == ["cam02", "cam01", "cam00"]
  # The same cameras in any order earn the same slots.
  reversed_order = list(reversed(serials))
  assert schema.camera_slots(reversed_order) == ["cam00", "cam01", "cam02"]
  by_serial = dict(zip(serials, schema.camera_slots(serials), strict=True))
  assert by_serial == dict(
    zip(reversed_order, schema.camera_slots(reversed_order), strict=True)
  )


def test_a_serial_that_cannot_identify_a_camera_refuses() -> None:
  """Falling back to argument order for an unidentifiable camera is the one
  behaviour the rank exists to prevent, so it raises instead."""
  for serials in ([""], ["A", ""], ["A", "A"], ["", ""]):
    with pytest.raises(ValueError, match="one distinct serial"):
      schema.camera_slots(serials)


def test_a_slot_is_dense_within_its_take() -> None:
  """The rank is over the take's own serials, so the same camera converted
  alongside a different one numbers differently. Anything comparing across
  takes keys on the recorded serial, never on the slot."""
  assert schema.camera_slots(["Z"]) == ["cam00"]
  assert schema.camera_slots(["Z", "A"]) == ["cam01", "cam00"]


def test_a_camera_path_reports_its_slot_and_nothing_else_does() -> None:
  """``slot_of`` is how a reader attributes an entity to a camera; a path under
  another root must not answer, or world-frame entities read as cameras."""
  assert schema.slot_of(f"/{schema.video('cam00', 'front')}") == "cam00"
  assert schema.slot_of(schema.state("cam07", "pose")) == "cam07"
  assert schema.slot_of(f"/{schema.anchor(3)}") is None
  assert schema.slot_of("/estimate") is None
  assert schema.slots_in(
    [schema.video("cam01", "rear"), schema.imu("cam00"), "/world/anchor/0"]
  ) == ["cam00", "cam01"]


def test_an_anchor_path_reports_its_tag_and_nothing_else_does() -> None:
  """``anchor_ids_in`` is how a consumer with only entity paths -- catalog
  registration building the corpus blueprint -- learns which tags to draw."""
  assert schema.anchor_id_of(f"/{schema.anchor(3)}") == 3
  assert schema.anchor_id_of(schema.anchor(0)) == 0
  assert schema.anchor_id_of(f"/{schema.video('cam00', 'front')}") is None
  assert schema.anchor_id_of("/world") is None
  assert schema.anchor_id_of(f"/{schema.anchor(3)}/label") is None
  assert schema.anchor_id_of(f"/{schema.anchor(3)}/") == 3
  assert schema.anchor_id_of("/world/anchor/\uff13") is None
  assert schema.anchor_id_of("/world/anchor/label") is None
  assert schema.anchor_id_of("/world/floor/3") is None
  assert schema.anchor_ids_in(["/world/anchor/3", "/world/anchor/8"]) == [3, 8]
  assert schema.anchor_ids_in(
    [schema.anchor(1), f"/{schema.anchor(0)}", schema.anchor(1), schema.imu("cam00")]
  ) == [0, 1]


def test_the_keyframe_bucket_lands_on_the_keyframe_row() -> None:
  """Latest-at on the review timeline must resolve to the frame the solver
  actually placed, so bucket k has to end exactly on row k*stride."""
  stride = schema.keyframe_stride(60.0)
  assert stride == 6
  rows = schema.keyframe_index(np.arange(0, 19, dtype=np.int64), stride)
  assert list(rows[:8]) == [0, 1, 1, 1, 1, 1, 1, 2]
  assert rows[stride] == 1 and rows[2 * stride] == 2
