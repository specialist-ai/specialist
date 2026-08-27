"""The lens as the policy sees it: the block mean and the disc."""

from __future__ import annotations

import numpy as np
import pytest
from specialist_camera import MASK_RADIUS, block_mean, disc_mask, to_image


def test_block_mean_is_the_exact_mean_rounded_half_up() -> None:
  image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
  expected = image.reshape(2, 2, 3, 2, 3).mean(axis=(1, 3))
  np.testing.assert_array_equal(block_mean(image, 2), np.floor(expected + 0.5))
  np.testing.assert_array_equal(block_mean(image, 1), image)
  assert block_mean(np.array([[[0], [1]], [[1], [1]]], dtype=np.uint8), 2)[0, 0, 0] == 1
  assert block_mean(np.full((2, 2, 1), 255, dtype=np.uint8), 2)[0, 0, 0] == 255
  with pytest.raises(ValueError, match="divide"):
    block_mean(np.zeros((10, 10, 3), dtype=np.uint8), 3)


def test_disc_mask_measures_pixel_centres_in_widths() -> None:
  # A 4x4 image and a radius of one pixel keep the middle four pixels.
  expected = np.zeros((4, 4), dtype=bool)
  expected[1:3, 1:3] = True
  np.testing.assert_array_equal(disc_mask(4, 4, 0.25), expected)
  assert disc_mask(48, 48, 0.4).mean() == pytest.approx(np.pi * 0.4**2, abs=0.02)
  # The default disc is the inscribed circle: it touches the edges, not the corners.
  assert MASK_RADIUS == 0.5
  mask = disc_mask(240, 240)
  assert mask[120, 0] and mask[0, 120] and not mask[0, 0]


def test_to_image_block_averages_then_blanks_outside_the_disc() -> None:
  out = to_image(np.full((24, 24, 3), 200, dtype=np.uint8), 2, 0.25)
  assert out.shape == (12, 12, 3) and out.dtype == np.uint8
  inside = disc_mask(12, 12, 0.25)
  assert np.all(out[inside] == 200) and np.all(out[~inside] == 0)
