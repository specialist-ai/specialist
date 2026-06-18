"""Calibration regression test for the reliability verdict.

These (anchor_coverage, n_front, n_rear, fingertip_error_mm) tuples are the real
session_02 anchor-subsampling ablation series (each run evaluated against the
full-anchor pseudo-GT). They tie the verdict's thresholds to *measured* accuracy:
every capture that drifted past ~1 cm must be flagged "under-constrained", and the
full-anchor capture (mm-level) must read "reliable". A future threshold change that
would silently re-classify a known-degraded capture as trustworthy fails here.
"""

from __future__ import annotations

import pytest

from specialist.vi_ba.reliability import reliability_verdict

# (label, anchor_coverage, n_front, n_rear, mean_fingertip_error_mm)
ABLATION = [
  ("curve_ke2", 0.48, 453, 143, 8.1),
  ("curve_ke8", 0.12, 78, 29, 30.4),
  ("curve_ke16", 0.06, 31, 3, 22.3),
  ("sw_b_4", 0.24, 205, 46, 22.4),
  ("sw_b_8", 0.12, 12, 2, 53.2),
  ("sw_b_16", 0.06, 4, 0, 79.4),
  ("sw_b_32", 0.03, 0, 0, 130.1),
  ("sw_kg_4", 0.24, 182, 78, 12.2),
  ("sw_kg_8", 0.12, 78, 29, 30.2),
  ("sw_kg_16", 0.06, 31, 3, 44.6),
  ("sw_kg_32", 0.03, 5, 0, 110.7),
]


@pytest.mark.parametrize("label,cov,n_front,n_rear,err_mm", ABLATION)
def test_degraded_captures_are_flagged(
  label: str, cov: float, n_front: int, n_rear: int, err_mm: float
) -> None:
  """Every ablation run drifted past 1 cm; all must be flagged under-constrained."""
  assert err_mm > 8.0  # guards the premise: these are all degraded captures
  level, _ = reliability_verdict(cov, n_front, n_rear)
  assert level == "under-constrained", f"{label} ({err_mm:.0f}mm) wrongly read reliable"


def test_full_anchor_capture_is_reliable() -> None:
  """The full-anchor session_02 run (mm-level vs GT) must read reliable."""
  level, _ = reliability_verdict(anchor_coverage=0.99, n_front=712, n_rear=164)
  assert level == "reliable"


def test_starved_lens_flagged_despite_good_coverage() -> None:
  """A lens starved of landmarks is flagged even when anchor coverage passes
  (the session_03 mode: good coverage, co-moving workspace → few landmarks).
  Exercises the landmark floors, which the cov<0.6 ablation rows never trigger."""
  level, reason = reliability_verdict(anchor_coverage=0.85, n_front=712, n_rear=8)
  assert level == "under-constrained"
  assert "workspace lens" in reason
