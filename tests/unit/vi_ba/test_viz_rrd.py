"""The rrd written by the layer writer round-trips through the reader."""

from dataclasses import fields
from pathlib import Path

import gtsam
import numpy as np
import pytest
from rerun.experimental import Chunk, RrdReader

from specialist.vi_ba.summary import TrackerSummary
from specialist.vi_ba.viz import TrackerVizData, _overlay_index, solution_chunks
from tests.record_fixtures import make_summary, urdf_file, viz_tracker, write_run

# Every entity a solve writes, and the layer that must carry it. An entity a
# future solve adds and routes nowhere would otherwise join the track layer in
# silence, which is the one thing the split exists to prevent.
_TRACK = (
  "/world",
  "/world/anchor_tag",
  "/world/anchor_tag/frame",
  "/tracker_0/body",
  "/tracker_0/body_keyframes",
  "/tracker_0/gripper",
  "/tracker_0/gripper_opening",
  "/tracker_1/body",
)
_VIZ = (
  "/tracker_0/trajectory",
  "/tracker_0/body/marker",
  "/tracker_0/landmarks/front",
  "/tracker_0/landmarks/rear",
  "/tracker_0/tf_static",
  "/tracker_0/gripper/visual_geometries/base/visual_0",
  "/metrics/distance_from_start/tracker_0",
  "/tracker_1/trajectory",
)


def _two_trackers() -> list[TrackerVizData]:
  return [viz_tracker("tracker_0", highrate=True), viz_tracker("tracker_1", offset=1.0)]


@pytest.fixture(scope="module")
def solved_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
  """One two-tracker run with a URDF model: the layers every check that only
  reads a finished run shares."""
  tmp = tmp_path_factory.mktemp("solved")
  return write_run(tmp, "sess_route", _two_trackers(), urdf_path=urdf_file(tmp))


def test_each_layer_carries_exactly_what_it_owns(solved_run: tuple[Path, Path]) -> None:
  """The producer names the layer for each chunk it builds, so a new entity
  that nobody routes lands nowhere by accident rather than in the answers."""
  track_path, viz_path = solved_run
  track = set(RrdReader(str(track_path)).store().schema().entity_paths())
  viz = set(RrdReader(str(viz_path)).store().schema().entity_paths())

  assert len(RrdReader(str(track_path)).recordings()) == 1
  assert not track & viz, f"{sorted(track & viz)} in both layers"
  assert not set(_TRACK) - track, f"missing from track: {sorted(set(_TRACK) - track)}"
  assert not set(_VIZ) - viz, f"missing from viz: {sorted(set(_VIZ) - viz)}"
  assert not set(_VIZ) & track, f"decoration in track: {sorted(set(_VIZ) & track)}"
  assert not set(_TRACK) & viz, f"answers in viz: {sorted(set(_TRACK) & viz)}"


def test_the_session_clock_carries_durations_in_every_layer(
  solved_run: tuple[Path, Path],
) -> None:
  """The session clock is indexed by durations, which a reader divides by a
  nanosecond."""
  seen = 0
  for path in solved_run:
    for chunk in RrdReader(str(path)).store().stream().to_chunks():
      schema = chunk.to_record_batch().schema
      if "video_time" in schema.names:
        dtype = schema.field("video_time").type
        assert "duration" in str(dtype), (path.parent.name, dtype)
        seen += 1
  assert seen, "no chunk carried the session clock"


def test_the_dashboard_ships_with_what_it_draws(solved_run: tuple[Path, Path]) -> None:
  """The dashboard is the viz layer's: it plots the overlays and the metrics,
  which the track layer does not carry, and two files claiming the same
  default would leave whichever loaded last to win."""
  track_path, viz_path = solved_run
  assert RrdReader(str(viz_path)).blueprints()
  assert not RrdReader(str(track_path)).blueprints()


def test_overlay_index_bounds_each_keyframe_to_one_frame() -> None:
  """Data rows sit at keyframe times (carrying the review timeline when
  given); clearing rows follow one frame period later on the video clock
  alone, so a 30 fps video never wears an overlay computed from an older
  keyframe and the review timeline never sees a clearing row."""
  kf_times = np.array([0.0, 0.1, 0.2])
  kf_col = object()
  data, clear = _overlay_index(kf_times, 1.0 / 30.0, [], kf_col)  # type: ignore[arg-type]
  assert len(data) == 2 and data[1] is kf_col
  assert clear is not None and len(clear) == 1
  fallback = [object()]
  same, none = _overlay_index(None, 1.0 / 30.0, fallback, None)  # type: ignore[arg-type]
  assert same is fallback and none is None


def test_several_mask_rects_reach_the_layer(tmp_path: Path) -> None:
  """A capture may mask more than one static region, and every box in the set
  is one row of the same chunk."""
  _track, viz = write_run(
    tmp_path,
    "sess_mask",
    _two_trackers(),
    mask_rects=((0, 0, 10, 10), (20, 20, 40, 40)),
  )
  masks = {
    chunk.entity_path: chunk.num_rows
    for chunk in RrdReader(str(viz)).store().stream().to_chunks()
    if chunk.entity_path.endswith("/static_mask")
  }
  assert masks and all(rows == 1 for rows in masks.values()), masks


def test_a_model_that_fails_to_write_is_not_mistaken_for_a_missing_model(
  tmp_path: Path,
) -> None:
  """A sink that raises on a mesh chunk raises out of ``estimate_chunks``: the
  optional-model guard covers building the model, not writing it."""
  data = viz_tracker("t0")
  data.urdf_root_poses = list(data.body_poses)

  def die(chunk: Chunk) -> None:
    if "geometries" in chunk.entity_path:
      raise OSError(28, "No space left on device")

  with pytest.raises(OSError):
    solution_chunks(
      die,
      die,
      [data],
      {0: gtsam.Pose3()},
      0,
      0.1,
      mask_pad_px=-1,
      urdf_path=urdf_file(tmp_path),
    )


def test_a_viewer_that_cannot_open_does_not_cost_the_run_its_layers(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A box with no display cannot show the run, and the run is worth more than
  the look at it."""

  def no_display(**kwargs: object) -> None:
    raise RuntimeError("no display")

  monkeypatch.setattr("rerun.sinks._spawn_viewer", no_display)
  for path in write_run(tmp_path, "sess_headless", _two_trackers(), open_viewer=True):
    assert path.exists()


def test_every_number_the_summary_carries_reaches_the_layer(tmp_path: Path) -> None:
  """Every field of ``Summary`` and ``CameraSummary`` reaches the layer."""
  quality = TrackerSummary(
    name="tracker_0",
    n_keyframes=3,
    anchor_obs=2,
    bounding_box_diagonal=1.5,
    path_length=2.5,
    front_rmse=0.4,
    rear_rmse=0.6,
  )
  summary = make_summary("tracker_0")
  track, _viz = write_run(
    tmp_path,
    "sess_props",
    [viz_tracker("tracker_0", highrate=True)],
    per_tracker=[quality],
    summary=summary,
  )
  columns = RrdReader(str(track)).store().schema().component_columns()
  written = {
    (c.entity_path, c.component)
    for c in columns
    if "__properties" in c.entity_path or "run_summary" in c.entity_path
  }
  # The curated headline set becomes segment-table columns; every other number
  # reaches the layer as the static run_summary entities instead.
  headline_homes = {
    "sources": "/__properties/session",
    "cross_tracker_obs": "/__properties/solver",
    "solver_iterations": "/__properties/solver",
  }
  for field in fields(summary):
    home = headline_homes.get(field.name, "/run_summary")
    assert (home, field.name) in written, field.name
  for field in fields(quality):
    home = (
      "/__properties/quality_tracker_0"
      if field.name == "front_rmse"
      else "/run_summary/tracker_0"
    )
    assert (home, field.name) in written, field.name


def test_training_fields_carry_the_keyframe_index(tmp_path: Path) -> None:
  """Every per-timestep field carries the keyframe timeline beside the session
  clock, so a dataloader can join one training row by integer keyframe
  index."""
  track, _viz = write_run(
    tmp_path,
    "sess_clock",
    [viz_tracker("tracker_0")],
    frame_period=1.0 / 30.0,
  )
  timelines: dict[str, set[str]] = {}
  for chunk in RrdReader(str(track)).store().stream().to_chunks():
    timelines.setdefault(chunk.entity_path, set()).update(chunk.timeline_names)
  for field in ("body_keyframes", "gripper_opening"):
    entities = [e for e in timelines if field in e]
    assert entities, field
    for entity in entities:
      assert {"video_time", "keyframe"} <= timelines[entity], (
        entity,
        timelines[entity],
      )
