"""The layers a solve writes: what a consumer reads, and what it looks at.

Both share their ``recording_id`` with the session's base recording so a
catalog merges all three, and both streams are constructed with
``send_properties=False`` because a second ``RecordingInfo`` on the same id
collides with the base segment's on merge. The solver and per-tracker quality
numbers go through named properties instead; the session's own identity and
sources stay in the base recording.

The split is by consumer. The track layer carries the solve's answers: body
poses, the keyframes they were solved at, joint angles, gripper opening, and
the frames those are expressed in. The viz layer carries what is only ever
drawn: the trajectory ribbon, the frontend's image overlays, the diagnostic
scalars and the robot meshes. A catalog registering both shows the session
whole; a query naming the track layer's entities reads none of the rest.
"""

from __future__ import annotations

import socket
from dataclasses import asdict
from pathlib import Path
from typing import Any

import rerun as rr
from rerun.experimental import (
  Chunk,
  LazyChunkStream,
  OptimizationProfile,
  send_chunks,
)

from specialist.record import schema
from specialist.vi_ba.summary import Summary, TrackerSummary
from specialist.vi_ba.viz import run_blueprint, solution_chunks
from specialist.vi_ba.viz_build import RunOutputs

__all__ = ["write_solution_layers"]

_VIEWER_PORT = 9876


def _split_group(
  stream: rr.RecordingStream,
  group: str,
  entity: str,
  values: dict[str, Any],
  headline: tuple[str, ...],
) -> None:
  """The headline keys become the group's segment-table columns; the rest
  lands on the static entity, queryable but never a column."""
  stream.send_property(group, rr.AnyValues(**{k: values[k] for k in headline}))
  rest = {k: v for k, v in values.items() if k not in headline}
  rr.log(entity, rr.AnyValues(**rest), static=True, recording=stream)


def _send_run_properties(
  summary: Summary, per_tracker: list[TrackerSummary], stream: rr.RecordingStream
) -> None:
  """The run's numbers, to the layer that keeps them.

  Everything the summary carries is written: a number the solve computed and
  the artifact does not record is a number nobody can ask about later.
  """
  run = asdict(summary)
  stream.send_property("session", rr.AnyValues(sources=list(run.pop("sources"))))
  _split_group(
    stream, "solver", "run_summary", run, ("cross_tracker_obs", "solver_iterations")
  )
  for tracker in per_tracker:
    _split_group(
      stream,
      f"quality_{tracker.name}",
      f"run_summary/{tracker.name}",
      asdict(tracker),
      ("front_rmse",),
    )


def _stream(session_id: str) -> rr.RecordingStream:
  return rr.RecordingStream(
    schema.APPLICATION_ID, recording_id=session_id, send_properties=False
  )


def _viewer_sink() -> list[rr.GrpcSink]:
  """A sink into a viewer, or nothing when one cannot be opened.

  The port is probed first: a gRPC sink nothing ever answers fails its whole
  stream on flush.
  """
  try:
    rr.spawn(connect=False)
    socket.create_connection(("127.0.0.1", _VIEWER_PORT), timeout=5.0).close()
  except Exception as exc:  # noqa: BLE001 - the viewer is optional; the layers are not
    print(f"  [layer] viewer not opened ({exc}); writing the layers without it")
    return []
  return [rr.GrpcSink(url=f"rerun+http://127.0.0.1:{_VIEWER_PORT}/proxy")]


def write_solution_layers(
  track_path: Path,
  viz_path: Path,
  outputs: RunOutputs,
  summary: Summary,
  reference_tag_id: int,
  anchor_tag_size: float,
  session_id: str,
  mask_pad_px: int,
  open_viewer: bool,
  urdf_path: Path | None = None,
  timeline_start: float | None = None,
  frame_period: float | None = None,
) -> None:
  """Write the run's track and viz layers (and open the viewer by default)."""
  blueprint = run_blueprint(outputs.viz_trackers)
  track_path.parent.mkdir(parents=True, exist_ok=True)
  viz_path.parent.mkdir(parents=True, exist_ok=True)
  partials = {
    path: path.parent / (path.name + ".partial") for path in (track_path, viz_path)
  }
  for partial in partials.values():
    partial.unlink(missing_ok=True)

  viewer = _viewer_sink() if open_viewer else []

  # The viz layer is most of the bytes and is only ever drawn, so its chunks go
  # to the file as they are built; the track layer is small enough to shape for
  # the object store a catalog reads it from.
  kept: list[Chunk] = []
  try:
    with _stream(session_id) as viz_stream:
      viz_stream.set_sinks(rr.FileSink(partials[viz_path]), *viewer)
      viz_stream.send_blueprint(blueprint, make_active=True, make_default=True)
      solution_chunks(
        kept.append,
        lambda chunk: send_chunks([chunk], recording=viz_stream),
        outputs.viz_trackers,
        outputs.anchor_poses_world,
        reference_tag_id,
        anchor_tag_size,
        mask_pad_px=mask_pad_px,
        urdf_path=urdf_path,
        timeline_start=timeline_start,
        frame_period=frame_period,
      )

    store = LazyChunkStream.from_iter(kept).collect(
      optimize=OptimizationProfile.OBJECT_STORE
    )
    with _stream(session_id) as track_stream:
      track_stream.set_sinks(rr.FileSink(partials[track_path]), *viewer)
      send_chunks(store, recording=track_stream)
      _send_run_properties(summary, outputs.per_tracker, track_stream)
    for path, partial in partials.items():
      partial.replace(path)
  finally:
    for partial in partials.values():
      partial.unlink(missing_ok=True)
