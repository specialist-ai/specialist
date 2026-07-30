"""Synthetic frame sequences shared by the vi_ba frontend unit tests."""

from __future__ import annotations

import cv2
import numpy as np

# The margin left over after cropping ``width`` centred columns keeps the
# np.roll wrap-around seam out of every returned frame.
_BASE_WIDTH = 320


def translating_frames(
  n: int, shift: int, seed: int, height: int = 240, width: int = 240
) -> list[np.ndarray]:
  """``n`` textured frames translated by ``shift`` px per frame: pure
  translation, so DIS flow and essential RANSAC make every match an inlier."""
  rng = np.random.default_rng(seed)
  base = rng.integers(0, 255, size=(height, _BASE_WIDTH), dtype=np.uint8)
  base = cv2.GaussianBlur(base, (0, 0), sigmaX=2.0)
  lo = (_BASE_WIDTH - width) // 2
  return [np.roll(base, shift * k, axis=1)[:, lo : lo + width].copy() for k in range(n)]
