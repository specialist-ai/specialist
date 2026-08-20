"""The run-configuration invariants ``validate`` enforces before any prep."""

from __future__ import annotations

from pathlib import Path

import pytest

from specialist.vi_ba.configs import AnchorCfg, Args, validate


def _args(**overrides: object) -> Args:
  fields: dict[str, object] = {
    "base": Path("session_base.rrd"),
    "anchor": AnchorCfg(tag_ids=(1,)),
    "base_tag_ids": (2000,),
  }
  fields.update(overrides)
  return Args(**fields)  # type: ignore[arg-type]


def test_accepts_a_minimal_single_tracker_run() -> None:
  validate(_args(), n_trackers=1)


def test_rejects_missing_base_tag_ids() -> None:
  """Each tracker is identified by its base tag, so the run cannot proceed
  without one per tracker."""
  with pytest.raises(ValueError, match="base-tag-ids"):
    validate(_args(base_tag_ids=()), n_trackers=1)


@pytest.mark.parametrize(
  "field", ["base_tag_ids", "left_finger_tag_ids", "right_finger_tag_ids"]
)
def test_rejects_per_tracker_list_shorter_than_the_trackers(field: str) -> None:
  """A short id list would silently pair the wrong tag with the wrong tracker."""
  overrides: dict[str, object] = {"base_tag_ids": (2000, 2003)}
  overrides[field] = (2000,)
  with pytest.raises(ValueError, match="must match"):
    validate(_args(**overrides), n_trackers=2)


def test_rejects_duplicate_base_tag_ids() -> None:
  """Two trackers wearing the same base tag cannot be told apart in the
  cross-tracker factors."""
  with pytest.raises(ValueError, match="distinct per tracker"):
    validate(_args(base_tag_ids=(2000, 2000)), n_trackers=2)


def test_rejects_nonpositive_solver_iterations() -> None:
  """A zero or negative LM budget silently produces an unrefined solve."""
  from specialist.vi_ba.configs import SolverCfg

  with pytest.raises(ValueError, match="max-iterations"):
    validate(_args(solver=SolverCfg(max_iterations=0)), n_trackers=1)


def test_rejects_empty_anchor_tag_ids() -> None:
  with pytest.raises(ValueError, match="at least one enabled anchor"):
    validate(_args(anchor=AnchorCfg(tag_ids=())), n_trackers=1)


def test_rejects_anchor_tag_ids_that_are_all_disabled() -> None:
  """A list of nothing but the disabled sentinel leaves no enabled anchor."""
  with pytest.raises(ValueError, match="at least one enabled anchor"):
    validate(_args(anchor=AnchorCfg(tag_ids=(-1,))), n_trackers=1)


def test_rejects_duplicate_anchor_tag_ids() -> None:
  with pytest.raises(ValueError, match="must be distinct"):
    validate(_args(anchor=AnchorCfg(tag_ids=(1, 1))), n_trackers=1)


def test_tag_id_disjointness_validated() -> None:
  """Anchor, base, and nonzero finger tag ids must be pairwise disjoint; a
  zero finger id means disabled and never clashes."""
  validate(_args(anchor=AnchorCfg(tag_ids=(1, 0))), n_trackers=1)
  validate(
    _args(anchor=AnchorCfg(tag_ids=(1, 0)), left_finger_tag_ids=(0,)), n_trackers=1
  )
  with pytest.raises(ValueError, match="2000"):
    validate(_args(anchor=AnchorCfg(tag_ids=(1, 2000))), n_trackers=1)
  with pytest.raises(ValueError, match="1"):
    validate(
      _args(anchor=AnchorCfg(tag_ids=(1, 0)), left_finger_tag_ids=(1,)), n_trackers=1
    )
  with pytest.raises(ValueError, match="2000"):
    validate(
      _args(anchor=AnchorCfg(tag_ids=(1, 0)), right_finger_tag_ids=(2000,)),
      n_trackers=1,
    )
