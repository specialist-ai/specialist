"""The GPU loop-closure matcher returns cv2's brute-force answer."""

from __future__ import annotations

import numpy as np
import pytest

from specialist.vi_ba import gpu_match
from specialist.vi_ba.features import detect_orb
from tests.frame_fixtures import match_pair


def _cuda() -> bool:
  try:
    import torch

    return bool(torch.cuda.is_available())
  except Exception:  # noqa: BLE001 - no torch, no GPU tests
    return False


def _block_frames(n: int) -> list[np.ndarray]:
  """Frames of an 8 px random block pattern shifted by 3 px each, rich in FAST
  corners and in exact descriptor ties between shifts."""
  rng = np.random.default_rng(5)
  cells = rng.integers(0, 2, (64, 70), dtype=np.uint8) * 255
  base = np.kron(cells, np.ones((8, 8), dtype=np.uint8))
  return [np.ascontiguousarray(base[:, 3 * k : 3 * k + 512]) for k in range(n)]


@pytest.mark.skipif(not _cuda(), reason="needs CUDA")
def test_gpu_matches_equal_cv2_brute_force() -> None:
  frames = [detect_orb(f, 2000, 20) for f in _block_frames(6)]
  assert min(len(f.keypoints) for f in frames) > 500
  pairs = [(i, j) for i in range(6) for j in range(6) if i != j]
  out = gpu_match._match_pairs_cuda(frames, pairs, 0.8)
  n_matches = 0
  for (i, j), got in zip(pairs, out, strict=True):
    expected = match_pair(frames[i], frames[j], 0.8)
    assert np.array_equal(got, expected), (i, j)
    n_matches += len(expected)
  assert n_matches > 100


@pytest.mark.skipif(not _cuda(), reason="needs CUDA")
def test_gpu_matches_equal_cv2_with_uneven_and_tied_trains() -> None:
  """Train frames of different sizes share one product, and frames made of
  repeated descriptors tie on distance everywhere."""
  frames = [detect_orb(f, 2000, 20) for f in _block_frames(5)]
  rng = np.random.default_rng(11)
  for k, keep in enumerate((1700, 900, 300, 2, 1200)):
    order = rng.permutation(len(frames[k].keypoints))[:keep]
    frames[k] = type(frames[k])(
      keypoints=frames[k].keypoints[order], descriptors=frames[k].descriptors[order]
    )
  tied = frames[0].descriptors[rng.integers(0, 40, 600)]
  frames.append(type(frames[0])(keypoints=np.zeros((600, 2)), descriptors=tied))
  n = len(frames)
  pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
  out = gpu_match._match_pairs_cuda(frames, pairs, 0.8)
  for (i, j), got in zip(pairs, out, strict=True):
    assert np.array_equal(got, match_pair(frames[i], frames[j], 0.8)), (i, j)
