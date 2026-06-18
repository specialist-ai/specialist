"""Per-session reliability verdict — a cheap under-constraint detector built
from pipeline-internal signals (anchor visibility + per-lens landmark counts).

This is NOT the ground-truth eval metric (that lives in eval_metrics.py and is
deliberately not imported by the pipeline). This verdict only reads the
pipeline's own inputs, so it is safe for track.py to consume: it tells the
operator when a capture is under-constrained (sparse anchor and/or a lens
starved of static-world landmarks) *before* the trajectory is trusted.

Thresholds are grounded in the session_02 anchor-subsampling calibration
(tests/unit/test_reliability_calibration.py): trajectory error rises steeply
below ~0.5 anchor coverage. The landmark floors separately catch a lens starved
of static-world structure (e.g. a workspace lens imaging the co-moving write
surface) even when anchor coverage is adequate.
"""

from __future__ import annotations

MIN_ANCHOR_COVERAGE = 0.6
MIN_REAR_LANDMARKS = 30
MIN_FRONT_LANDMARKS = 100


def reliability_verdict(
  anchor_coverage: float,
  n_front: int,
  n_rear: int,
  min_anchor_cov: float = MIN_ANCHOR_COVERAGE,
  min_rear: int = MIN_REAR_LANDMARKS,
  min_front: int = MIN_FRONT_LANDMARKS,
) -> tuple[str, str]:
  """Return (level, reason). ``level`` is "reliable" or "under-constrained".
  ``anchor_coverage`` = fraction of keyframes with an anchor sighting."""
  issues: list[str] = []
  if anchor_coverage < min_anchor_cov:
    issues.append(f"sparse anchor ({anchor_coverage:.0%} of keyframes)")
  if n_rear < min_rear:
    issues.append(f"starved workspace lens ({n_rear} landmarks)")
  if n_front < min_front:
    issues.append(f"starved front lens ({n_front} landmarks)")
  if not issues:
    return ("reliable", "anchor-dense + rich static vision")
  return ("under-constrained", "; ".join(issues))
