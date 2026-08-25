"""Read an estimate layer back into numpy: the scorers' view of a run.

The solver's exact keyframe poses live at ``camera/<slot>/state/pose``; the
IMU-rate propagated stream on ``camera/<slot>`` itself is for consumers that
want a smooth trajectory. The solved ``body -> base tag`` transform sits
statically under the camera, and the slots are read off the entity paths.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from rerun.experimental import Chunk, RrdReader

from specialist.record import schema
from specialist.vi_ba.geometry import quats_to_matrices

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "KeyframePoses",
  "check_frame_alignment",
  "read_keyframe_poses",
]

# Past this, a keyframe pairs its overlays with the wrong video frame.
_ALIGNMENT_TOLERANCE = 1e-3


def _paths_by_slot(paths: list[str], mint: Callable[[str], str]) -> dict[str, str]:
  """The recording's own spelling of one per-camera entity, per slot. Matching
  against the minted path rather than a suffix keeps ``schema`` the only place
  that decides what an entity is called."""
  found = {}
  for path in paths:
    slot = schema.slot_of(path)
    if slot is not None and path.strip("/") == mint(slot):
      found[slot] = path
  return found


@dataclass(frozen=True)
class KeyframePoses:
  """What a scorer reads back out of the estimate layer: each camera's pose at
  each keyframe. A slice of the estimate, not the whole of it -- the biases,
  joint angles and landmarks stay in the layer."""

  slots: list[str]
  kf_times_by_slot: dict[str, Float64Array]
  kf_rotations_by_slot: dict[str, Float64Array]  # (n, 3, 3)
  kf_translations_by_slot: dict[str, Float64Array]  # (n, 3)


def _pose_rows(chunk: Chunk) -> tuple[Float64Array, Float64Array, Float64Array]:
  """(times, translations, quats_xyzw) of one temporal Transform3D chunk, each
  column taken whole out of arrow rather than a row at a time."""
  batch = chunk.to_record_batch()
  names = batch.schema.names

  def vectors(component: str, width: int) -> Float64Array:
    # A component column nests the instances of a row inside the row, and the
    # writer stores one instance per row: both levels flatten away.
    values = batch.column(names.index(component))
    flat = values.flatten().flatten().to_numpy(zero_copy_only=False)
    out: Float64Array = flat.astype(np.float64).reshape(len(values), width)
    return out

  return (
    schema.seconds(batch.column(names.index(schema.TIMELINE))),
    vectors("Transform3D:translation", 3),
    vectors("Transform3D:quaternion", 4),
  )


def check_frame_alignment(
  frame_times: dict[str, Float64Array], layer_rrd: Path
) -> None:
  """Assert every keyframe timestamp in the layer lands on a base-recording
  frame time, per camera.

  A timestamp between frames means the viewer pairs that keyframe's overlays
  with the previous frame, and the solve windows its IMU on a clock that
  matches no image."""
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
  state_paths = _paths_by_slot(paths, lambda slot: schema.state(slot, "pose"))
  rows_by_slot: dict[str, list[tuple[Float64Array, ...]]] = {}
  if state_paths:
    keyframe_stream = store.stream().filter(
      content=list(state_paths.values()),
      components=["Transform3D:translation", "Transform3D:quaternion"],
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
