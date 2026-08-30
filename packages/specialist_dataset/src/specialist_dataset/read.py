"""Query a dataset layer: what a recording holds, as numpy.

Columns are addressed by the selectors ``schema`` mints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import rerun as rr
from specialist_camera import MeiParams, Side, rescale

from specialist_dataset import schema
from specialist_dataset.poses import matrices_from_poses, quats_to_matrices

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "KeyframePoses",
  "Lens",
  "Recording",
  "lens_from_row",
  "read_keyframe_poses",
]

# Past this, a keyframe pairs its overlays with the wrong video frame.
_DATASET = "recording"
_ANCHOR = re.compile(rf"/{schema.WORLD}/anchor/(\d+):")


def _values(table: pa.Table, column: str, width: int | None = None) -> Float64Array:
  """One column's values, flattened out of the component's per-row nesting.

  Rows that never carried the component drop out.
  """
  array = table.column(column).combine_chunks()
  array = array.drop_null()
  while pa.types.is_list(array.type) or pa.types.is_fixed_size_list(array.type):
    array = array.flatten()
  flat = np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)
  return flat.reshape(-1, width) if width else flat


def _rows_with(table: pa.Table, column: str) -> pa.Table:
  """The rows where one column carried a value."""
  return table.filter(pa.compute.is_valid(table.column(column).combine_chunks()))


class Recording:
  """One layer of a dataset, open for queries.

  Dropping it closes the dataset, so a reader holds it for as long as it reads.
  """

  def __init__(self, rrd_path: Path) -> None:
    self.path = rrd_path
    self._server = rr.server.Server(datasets={_DATASET: [str(rrd_path)]})
    self._dataset = rr.catalog.CatalogClient(self._server.url()).get_dataset(_DATASET)

  def __enter__(self) -> Recording:
    return self

  def __exit__(self, *exception: object) -> None:
    return None

  def reader(self, contents: str | list[str]) -> Any:
    """A query over the named entities, indexed by the recording's clock."""
    return self._dataset.filter_contents(contents).reader(index=schema.TIMELINE)

  def statics(self, contents: str | list[str]) -> pa.Table:
    """Every static component of the named entities, one row."""
    return self._dataset.filter_contents(contents).reader(index=None).to_arrow_table()

  def series(self, contents: str | list[str]) -> pa.Table:
    """The named entities over the recording's clock, in index order."""
    table = (
      self._dataset.filter_contents(contents)
      .reader(index=schema.TIMELINE)
      .to_arrow_table()
    )
    return table.sort_by(schema.TIMELINE)

  def row(self, entity: str) -> dict[str, Any]:
    """Component name to value for one entity's statics, as stored.

    Values come back untyped: a component is not always a number.
    """
    table = self.statics(f"/{entity}")
    prefix = f"/{entity}:"
    out: dict[str, Any] = {}
    for name in table.schema.names:
      if not name.startswith(prefix):
        continue
      values = table.column(name).combine_chunks().drop_null().to_pylist()
      if values:
        out[name[len(prefix) :]] = values[-1]
    return out

  def property(self, name: str) -> dict[str, Any]:
    """Field name to value for one recording property, as stored."""
    table = self._dataset.segment_table().to_arrow_table()
    prefix = f"property:{name}:"
    out: dict[str, Any] = {}
    for column in table.schema.names:
      if column.startswith(prefix):
        values = table.column(column).combine_chunks().drop_null().to_pylist()
        if values:
          out[column[len(prefix) :]] = values[0]
    return out

  def values(
    self, entity: str, component: str, width: int | None = None
  ) -> tuple[Float64Array, Float64Array]:
    """One component's series on one entity: ``(times, values)`` in index
    order, ``values`` as ``(rows, width)``."""
    column = schema.selector(entity, component)
    table = self.series(f"/{entity}")
    if column not in table.schema.names:
      raise ValueError(f"{self.path} has no {component} at /{entity}")
    rows = _rows_with(table, column)
    flat = _values(rows, column)
    return self.seconds(rows), flat.reshape(rows.num_rows, width or -1)

  def scalars(
    self, entity: str, width: int | None = None
  ) -> tuple[Float64Array, Float64Array]:
    """One entity's scalar series: ``(times, values)`` in index order."""
    return self.values(entity, schema.SCALARS, width)

  def entity_paths(self) -> set[str]:
    """Every entity the recording holds, without a leading slash."""
    return {p.strip("/") for p in self._dataset.schema().entity_paths()}

  def seconds(self, table: pa.Table) -> Float64Array:
    """The index column of a series, in seconds."""
    return schema.seconds(table.column(schema.TIMELINE).combine_chunks())


@dataclass(frozen=True)
class KeyframePoses:
  """Each camera's pose at each keyframe."""

  slots: list[str]
  kf_times_by_slot: dict[str, Float64Array]
  kf_rotations_by_slot: dict[str, Float64Array]  # (n, 3, 3)
  kf_translations_by_slot: dict[str, Float64Array]  # (n, 3)


@dataclass(frozen=True)
class Lens:
  """One lens at the decode resolution."""

  params: MeiParams
  body_from_lens: Float64Array  # (4, 4)


def _pose_at(table: pa.Table, entity: str) -> Float64Array:
  """The pose one entity carries, as ``(4, 4)``."""
  translation = schema.selector(entity, schema.POSE_TRANSLATION)
  if translation not in table.schema.names:
    raise ValueError(f"no pose at /{entity}")
  matrices: Float64Array = matrices_from_poses(
    _values(table, translation, 3),
    _values(table, schema.selector(entity, schema.POSE_QUATERNION), 4),
  )
  pose: Float64Array = matrices[0]
  return pose


def read_keyframe_poses(rrd_path: Path) -> KeyframePoses:
  """Load the estimate layer's keyframe poses and slots."""
  with Recording(rrd_path) as recording:
    slots = sorted(
      slot
      for slot in schema.slots_in(recording.entity_paths())
      if schema.state(slot, "pose") in recording.entity_paths()
    )
    if not slots:
      raise ValueError(f"{rrd_path} holds no {schema.state('<slot>', 'pose')} entity")
    kf_times, kf_rotations, kf_translations = {}, {}, {}
    for slot in slots:
      entity = schema.state(slot, "pose")
      translation = schema.selector(entity, schema.POSE_TRANSLATION)
      table = _rows_with(recording.series(f"/{entity}"), translation)
      kf_times[slot] = recording.seconds(table)
      kf_translations[slot] = _values(table, translation, 3)
      kf_rotations[slot] = quats_to_matrices(
        _values(table, schema.selector(entity, schema.POSE_QUATERNION), 4)
      )
  return KeyframePoses(
    slots=slots,
    kf_times_by_slot=kf_times,
    kf_rotations_by_slot=kf_rotations,
    kf_translations_by_slot=kf_translations,
  )


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
