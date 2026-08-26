"""Read an estimate layer into numpy, refusing a layer of another schema version."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from rerun.experimental import Chunk, RrdReader
from specialist_camera import MeiParams, Side, rescale

from specialist_dataset import schema
from specialist_dataset.poses import matrices_from_poses, quats_to_matrices

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "Anchors",
  "Gripper",
  "KeyframePoses",
  "Lens",
  "check_frame_alignment",
  "lens_from_row",
  "newest_values",
  "read_anchors",
  "read_grippers",
  "read_keyframe_poses",
  "read_lens",
  "read_tag_ids",
]

_ANCHOR = re.compile(rf"/{schema.WORLD}/anchor/(\d+)")

# Past this, a keyframe pairs its overlays with the wrong video frame.
_ALIGNMENT_TOLERANCE = 1e-3


def _state_pose_paths(paths: list[str]) -> dict[str, str]:
  """The recording's own spelling of each camera's keyframe-pose entity."""
  found = {}
  for path in paths:
    slot = schema.slot_of(path)
    if slot is not None and path.strip("/") == schema.state(slot, "pose"):
      found[slot] = path
  return found


@dataclass(frozen=True)
class KeyframePoses:
  """Each camera's pose at each keyframe."""

  slots: list[str]
  kf_times_by_slot: dict[str, Float64Array]
  kf_rotations_by_slot: dict[str, Float64Array]  # (n, 3, 3)
  kf_translations_by_slot: dict[str, Float64Array]  # (n, 3)


def _pose_rows(chunk: Chunk) -> tuple[Float64Array, Float64Array, Float64Array]:
  """(times, translations, quats_xyzw) of one temporal Transform3D chunk."""
  batch = chunk.to_record_batch()
  names = batch.schema.names

  def vectors(component: str, width: int) -> Float64Array:
    # The writer stores one instance per row, so both nesting levels flatten away.
    values = batch.column(names.index(component))
    flat = values.flatten().flatten().to_numpy(zero_copy_only=False)
    out: Float64Array = flat.astype(np.float64).reshape(len(values), width)
    return out

  return (
    schema.seconds(batch.column(names.index(schema.TIMELINE))),
    vectors(schema.POSE_TRANSLATION, 3),
    vectors(schema.POSE_QUATERNION, 4),
  )


def check_frame_alignment(
  frame_times: dict[str, Float64Array], layer_rrd: Path
) -> None:
  """Assert each camera's keyframe timestamps land on its frame times."""
  run = read_keyframe_poses(layer_rrd)
  for slot, kf_times in run.kf_times_by_slot.items():
    if slot not in frame_times:
      raise ValueError(f"{slot} has no video frame times")
    times = frame_times[slot]
    positions = np.searchsorted(times, kf_times)
    lo = times[np.clip(positions - 1, 0, len(times) - 1)]
    hi = times[np.clip(positions, 0, len(times) - 1)]
    nearest = np.minimum(np.abs(kf_times - lo), np.abs(kf_times - hi))
    worst = float(nearest.max())
    if worst > _ALIGNMENT_TOLERANCE:
      raise ValueError(
        f"{slot}: keyframe times miss the video frames by up to "
        f"{worst * 1e3:.2f} ms (tolerance {_ALIGNMENT_TOLERANCE * 1e3:.2f} ms)"
      )


def read_keyframe_poses(rrd_path: Path) -> KeyframePoses:
  """Load the estimate layer's keyframe poses and slots."""
  store = RrdReader(str(rrd_path)).store()
  paths = store.schema().entity_paths()
  state_paths = _state_pose_paths(paths)
  if not state_paths:
    raise ValueError(f"{rrd_path} holds no {schema.state('<slot>', 'pose')} entity")
  rows_by_slot: dict[str, list[tuple[Float64Array, ...]]] = {}
  keyframe_stream = store.stream().filter(
    content=list(state_paths.values()),
    components=[schema.POSE_TRANSLATION, schema.POSE_QUATERNION],
  )
  by_path = {path: slot for slot, path in state_paths.items()}
  for chunk in keyframe_stream.to_chunks():
    rows_by_slot.setdefault(by_path[chunk.entity_path], []).append(_pose_rows(chunk))
  kf_times = {}
  kf_rotations = {}
  kf_translations = {}
  for slot, rows in rows_by_slot.items():
    times, translations, quats = (np.concatenate(c) for c in zip(*rows, strict=True))
    order = np.argsort(times, kind="stable")
    kf_times[slot] = times[order]
    kf_translations[slot] = translations[order]
    kf_rotations[slot] = quats_to_matrices(quats[order])
  return KeyframePoses(
    slots=sorted(kf_times),
    kf_times_by_slot=kf_times,
    kf_rotations_by_slot=kf_rotations,
    kf_translations_by_slot=kf_translations,
  )


def newest_values(chunks: Iterable[Chunk]) -> dict[str, Any]:
  """Component name -> the value with the highest row id over ``chunks``."""
  out: dict[str, Any] = {}
  newest: dict[str, bytes] = {}
  for chunk in chunks:
    batch = chunk.to_record_batch()
    names = batch.schema.names
    row_ids = batch.column(names.index("rerun.controls.RowId")).to_pylist()
    for name, column in zip(names, batch.columns, strict=True):
      if name.startswith("rerun.controls.") or name == schema.TIMELINE:
        continue
      for i, value in enumerate(column.to_pylist()):
        if value is not None and row_ids[i] > newest.get(name, b""):
          newest[name] = row_ids[i]
          out[name] = value
  return out


@dataclass(frozen=True)
class Lens:
  """One lens at the decode resolution."""

  params: MeiParams
  body_from_lens: Float64Array  # (4, 4)


def lens_from_row(row: dict[str, Any], decoded_width: int, side: Side) -> Lens:
  """One stored lens entity as a lens at the decode resolution."""

  def scalar(name: str) -> float:
    return float(np.asarray(row[name], dtype=np.float64).reshape(-1)[0])

  raw = MeiParams(
    **{
      name: scalar(name)
      for name in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2", "xi")
    }
  )
  body_from_lens = matrices_from_poses(
    np.asarray(row[schema.POSE_TRANSLATION], dtype=np.float64),
    np.asarray(row[schema.POSE_QUATERNION], dtype=np.float64),
  )[0]
  return Lens(
    params=rescale(raw, scalar("frame_width"), decoded_width, side),
    body_from_lens=body_from_lens,
  )


def read_lens(base_rrd: Path, slot: str, side: Side, decoded_width: int) -> Lens:
  """One lens of one base layer, at the resolution its frames decode to."""
  entity = schema.lens(slot, side)
  store = RrdReader(str(base_rrd)).store()
  row = newest_values(store.stream().filter(content=f"/{entity}").to_chunks())
  if not row:
    raise ValueError(f"{base_rrd} holds no /{entity}")
  return lens_from_row(row, decoded_width, side)


@dataclass(frozen=True)
class Gripper:
  """One slot's solved gripper: base-tag pose on the body and each finger's opening."""

  base_tag: Float64Array  # (4, 4)
  fingers: dict[str, tuple[Float64Array, Float64Array]]
  """Side to its ``(times, angles)``, on the recording's clock."""


@dataclass(frozen=True)
class Anchors:
  """The solved anchor tags, and the id of the one the solve held fixed."""

  reference: int
  poses: dict[int, Float64Array]
  """Tag id to its ``(4, 4)`` world pose."""


def _entity_row(store: Any, entity: str) -> dict[str, Any]:
  """Component name to newest value at one entity."""
  return newest_values(store.stream().filter(content=entity).to_chunks())


def _static_pose(store: Any, entity: str) -> Float64Array:
  """The pose an entity carries statically."""
  row = _entity_row(store, entity)
  if schema.POSE_TRANSLATION not in row:
    raise ValueError(f"no static pose at {entity}")
  pose: Float64Array = matrices_from_poses(
    np.asarray(row[schema.POSE_TRANSLATION], dtype=np.float64),
    np.asarray(row[schema.POSE_QUATERNION], dtype=np.float64),
  )[0]
  return pose


def _series(
  store: Any, entity: str, component: str
) -> tuple[Float64Array, Float64Array]:
  """One temporal scalar column of one entity, in time order across its chunks."""
  times: list[Float64Array] = []
  values: list[Float64Array] = []
  stream = store.stream().filter(content=entity, components=[component])
  for chunk in stream.to_chunks():
    batch = chunk.to_record_batch()
    names = batch.schema.names
    if component not in names or schema.TIMELINE not in names:
      continue
    column = batch.column(names.index(component))
    times.append(schema.seconds(batch.column(names.index(schema.TIMELINE))))
    values.append(
      column.flatten().to_numpy(zero_copy_only=False).astype(np.float64).ravel()
    )
  if not values:
    raise ValueError(f"no {component} at {entity}")
  time, value = np.concatenate(times), np.concatenate(values)
  order = np.argsort(time, kind="stable")
  return time[order], value[order]


def read_grippers(rrd_path: Path) -> dict[str, Gripper]:
  """Each slot's gripper: its base tag and its finger angles."""
  store = RrdReader(str(rrd_path)).store()
  paths = {p.strip("/") for p in store.schema().entity_paths()}
  out: dict[str, Gripper] = {}
  for slot in schema.slots_in(paths):
    if schema.base_tag(slot) not in paths:
      continue
    fingers = {
      side: _series(store, f"/{schema.gripper_part(slot, f'{side}_finger')}", "angle")
      for side in ("left", "right")
    }
    out[slot] = Gripper(
      base_tag=_static_pose(store, f"/{schema.base_tag(slot)}"), fingers=fingers
    )
  return out


def read_anchors(rrd_path: Path) -> Anchors:
  """The solved anchor poses by tag id, and which one the solve held fixed."""
  store = RrdReader(str(rrd_path)).store()
  poses: dict[int, Float64Array] = {}
  reference: int | None = None
  for entity in sorted(store.schema().entity_paths()):
    if not _ANCHOR.fullmatch(entity):
      continue
    row = _entity_row(store, entity)
    tag_id = int(np.asarray(row["tag_id"]).reshape(-1)[0])
    poses[tag_id] = _static_pose(store, entity)
    if bool(np.asarray(row["is_reference"]).reshape(-1)[0]):
      reference = tag_id
  if reference is None:
    raise ValueError(f"{rrd_path} marks no anchor as the reference")
  return Anchors(reference=reference, poses=poses)


def read_tag_ids(rrd_path: Path) -> dict[str, dict[str, int]]:
  """Per slot, each gripper body's tag id, skipping a body with an absent or zero id."""
  store = RrdReader(str(rrd_path)).store()
  paths = {p.strip("/") for p in store.schema().entity_paths()}
  out: dict[str, dict[str, int]] = {}
  for slot in schema.slots_in(paths):
    mapping: dict[str, int] = {}
    for body, entity in (
      ("base_tag", schema.base_tag(slot)),
      ("left_finger_tag", schema.joint_tag(slot, "left_finger")),
      ("right_finger_tag", schema.joint_tag(slot, "right_finger")),
    ):
      recorded = _entity_row(store, f"/{entity}").get("tag_id")
      if recorded is not None and int(np.asarray(recorded).reshape(-1)[0]) > 0:
        mapping[body] = int(np.asarray(recorded).reshape(-1)[0])
    out[slot] = mapping
  return out
