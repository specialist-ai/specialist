"""The trajectory ribbon carries one step per sample; the window is the view's."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.experimental import RrdReader

from specialist.vi_ba.viz import (
  _trail_segments,
  blueprint_for_names,
  segment_card_blueprint,
)


def test_each_sample_carries_the_step_that_reached_it() -> None:
  traj = np.arange(18, dtype=np.float64).reshape(6, 3)
  times = np.arange(6, dtype=np.float64) / 10.0
  segments = _trail_segments(traj, times, 1.0)
  assert segments.shape == (6, 2, 3)
  # The first sample has no predecessor, so its step stands still.
  assert np.array_equal(segments[0], np.stack([traj[0], traj[0]]))
  for i in range(1, len(traj)):
    assert np.array_equal(segments[i], np.stack([traj[i - 1], traj[i]]))


def test_a_break_in_the_samples_is_a_break_in_the_path() -> None:
  """A step longer than the window stands still, so the path breaks at a
  dropout and spans only the samples the run has positions for."""
  traj = np.arange(12, dtype=np.float64).reshape(4, 3)
  times = np.array([0.0, 0.1, 5.0, 5.1])
  segments = _trail_segments(traj, times, 1.0)
  assert np.array_equal(segments[2], np.stack([traj[2], traj[2]]))
  assert np.array_equal(segments[1], np.stack([traj[0], traj[1]]))
  assert np.array_equal(segments[3], np.stack([traj[2], traj[3]]))


def _ranged(blueprint: rrb.Blueprint, rrd: Path) -> set[str]:
  """The entity paths the blueprint gives a visible time range."""
  with rr.RecordingStream("app", recording_id="bp", send_properties=False) as stream:
    stream.save(str(rrd))
    stream.send_blueprint(blueprint, make_active=True, make_default=True)
  reader = RrdReader(str(rrd))
  columns = reader.store(store=reader.blueprints()[0]).schema().component_columns()
  return {c.entity_path for c in columns if c.component == "VisibleTimeRanges:ranges"}


def test_every_view_that_draws_the_ribbon_sets_the_window(tmp_path: Path) -> None:
  """The samples carry one step each, so every view that draws the ribbon sets
  a visible time range."""
  views = {
    "dashboard": blueprint_for_names(["t0"]),
    "card": segment_card_blueprint(["t0"]),
  }
  for name, blueprint in views.items():
    ranged = _ranged(blueprint, tmp_path / f"{name}.rrd")
    assert any(p.endswith("overrides/t0/trajectory/visualizers") for p in ranged), (
      f"{name}: {ranged}"
    )
