"""Tests for the per-session reliability verdict (under-constraint detector)."""

from __future__ import annotations

from specialist.vi_ba.reliability import reliability_verdict


def test_well_constrained_is_reliable() -> None:
  level, _ = reliability_verdict(anchor_coverage=0.97, n_front=712, n_rear=164)
  assert level == "reliable"


def test_starved_workspace_flagged() -> None:
  level, reason = reliability_verdict(anchor_coverage=0.85, n_front=523, n_rear=8)
  assert level == "under-constrained"
  assert "workspace" in reason


def test_sparse_anchor_flagged() -> None:
  level, reason = reliability_verdict(anchor_coverage=0.45, n_front=43, n_rear=80)
  assert level == "under-constrained"
  assert "anchor" in reason


def test_multiple_issues_listed() -> None:
  level, reason = reliability_verdict(anchor_coverage=0.45, n_front=43, n_rear=8)
  assert level == "under-constrained"
  assert "anchor" in reason and "workspace" in reason


def test_reason_nonempty() -> None:
  for args in [(0.97, 712, 164), (0.4, 10, 5)]:
    _, reason = reliability_verdict(*args)
    assert reason
