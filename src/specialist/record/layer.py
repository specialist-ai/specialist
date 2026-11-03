"""The layers a solve writes: what a consumer reads, and what it looks at.

Both share their ``recording_id`` with the session's base recording and carry no
``RecordingInfo``: a second one on that id collides with the base segment's on
merge.

The estimate layer carries quantities, the viz layer carries renderings. Every
name here is written by exactly one layer.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import rerun as rr
from rerun.chunk import Chunk, LazyChunkStream, OptimizationProfile
from specialist_corpus import schema

from specialist.vi_ba.summary import CameraSummary, Summary
from specialist.vi_ba.viz import Layers, estimate_chunks
from specialist.vi_ba.viz_build import RunEstimate

__all__ = ["git_sha", "write_estimate_layers"]


def git_sha() -> str:
  """The commit this solve ran from, or ``unknown``. Never empty: an empty
  value writes no column."""
  try:
    return subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=Path(__file__).parent,
      capture_output=True,
      text=True,
      timeout=5.0,
      check=True,
    ).stdout.strip()
  except Exception:  # noqa: BLE001 - provenance is optional; the layers are not
    return "unknown"


def _run_chunks(
  summary: Summary, per_camera: list[CameraSummary], estimated_at: str
) -> list[Chunk]:
  """The run's own account as properties, so a catalog reads it without opening
  the segment, and each camera's numbers on its ``fit`` entity.

  The per-camera numbers stay on entities: a property is per recording, so they
  would flatten to one column per camera and grow with the take's camera count.
  ``segment_table()`` has no projection pushdown, so every query pays for every
  property column any layer ever wrote."""
  chunks = []
  for camera in per_camera:
    fields = asdict(camera)
    entity = schema.fit(fields.pop("slot"))
    columns = cast("Any", rr.AnyValues).columns(**{k: [v] for k, v in fields.items()})
    chunks.append(Chunk.from_columns(entity, [], columns))
  # A camera the run never measured would poison max() with its NaN.
  measured = [c.front_rmse for c in per_camera if not math.isnan(c.front_rmse)]
  account = rr.AnyValues(
    **asdict(summary),
    worst_rmse=max(measured) if measured else float("nan"),
    git_sha=git_sha(),
    estimated_at=estimated_at,
    rerun_version=rr.__version__,
  )
  return [*chunks, Chunk.from_property(schema.ESTIMATE_PROPERTIES, account)]


def write_estimate_layers(
  estimate_path: Path,
  viz_path: Path,
  estimate: RunEstimate,
  summary: Summary,
  anchor_tag_size: float,
  session_id: str,
  urdf_path: Path,
  timeline_start: float,
  frame_period: float,
) -> None:
  """Write the run's estimate and viz layers."""
  estimated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
  estimate_path.parent.mkdir(parents=True, exist_ok=True)
  viz_path.parent.mkdir(parents=True, exist_ok=True)
  partials = {
    path: path.parent / (path.name + ".partial") for path in (estimate_path, viz_path)
  }
  kept: list[Chunk] = []
  drawn: list[Chunk] = []
  estimate_chunks(
    Layers(estimate=kept.append, viz=drawn.append),
    estimate.cameras,
    estimate.anchor_poses_world,
    estimate.reference_tag_id,
    anchor_tag_size,
    urdf_path=urdf_path,
    timeline_start=timeline_start,
    frame_period=frame_period,
  )
  kept += _run_chunks(summary, estimate.per_camera, estimated_at)
  ids = {"application_id": schema.APPLICATION_ID, "recording_id": session_id}
  try:
    LazyChunkStream.from_iter(drawn).write_rrd(partials[viz_path], **ids)
    LazyChunkStream.from_iter(kept).collect(
      optimize=OptimizationProfile.OBJECT_STORE
    ).write_rrd(partials[estimate_path], **ids)
    for path, partial in partials.items():
      partial.replace(path)
  finally:
    for partial in partials.values():
      partial.unlink(missing_ok=True)
