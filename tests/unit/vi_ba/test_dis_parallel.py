"""Bit-identity test for the parallel DIS flow precompute (``_iter_dis_flows``).

``_iter_dis_flows`` runs each pair on its own DISOpticalFlow across a thread
pool. Its output must be byte-for-byte identical to a serial loop that reuses
one DISOpticalFlow across pairs, because the frontend's subpixel precision
depends on the exact flow field.
"""

from __future__ import annotations

import cv2
import numpy as np

from specialist.vi_ba.features import (
  _DIS_BACKWARD_VARIATIONAL_ITERS,
  _DIS_VARIATIONAL_ITERS,
  _iter_dis_flows,
)
from tests.frame_fixtures import translating_frames


def test_precompute_matches_shared_serial_dis() -> None:
  """Fresh-per-pair parallel flow equals shared-instance serial flow, exactly."""
  frames = translating_frames(5, shift=3, seed=2, height=256, width=256)

  # Serial reference: one DISOpticalFlow reused across pairs.
  dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
  ref: dict[int, tuple[np.ndarray, np.ndarray]] = {}
  for k in range(1, len(frames)):
    a = np.ascontiguousarray(frames[k - 1])
    b = np.ascontiguousarray(frames[k])
    dis.setVariationalRefinementIterations(_DIS_VARIATIONAL_ITERS)
    fwd = dis.calc(a, b, None)
    dis.setVariationalRefinementIterations(_DIS_BACKWARD_VARIATIONAL_ITERS)
    ref[k] = (fwd, dis.calc(b, a, None))

  par = dict(_iter_dis_flows(frames, range(1, len(frames))))

  assert set(par) == set(ref)
  for k in ref:
    assert np.array_equal(par[k][0], ref[k][0]), f"fwd flow differs at pair {k}"
    assert np.array_equal(par[k][1], ref[k][1]), f"bwd flow differs at pair {k}"


def test_precompute_short_sequences_return_empty() -> None:
  assert dict(_iter_dis_flows([], range(1, 0))) == {}
  assert dict(_iter_dis_flows([np.zeros((16, 16), dtype=np.uint8)], range(1, 1))) == {}
