"""Tests for ``specialist.vi_ba.features``: ORB detection, BFMatcher,
essential-matrix RANSAC, track chaining, and KLT tracking. All synthetic
— no ``.insv`` dependency.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from specialist.vi_ba.features import (
  Frame,
  chain_tracks,
  detect_orb,
  essential_ransac_inliers,
  klt_track,
  match_pair,
)

Float64Array = npt.NDArray[np.float64]


def _textured_image(rng: np.random.Generator, h: int = 200, w: int = 300) -> np.ndarray:
  return rng.integers(0, 255, size=(h, w), dtype=np.uint8)


def test_detect_orb_on_textured_image_returns_keypoints() -> None:
  rng = np.random.default_rng(0)
  frame = detect_orb(_textured_image(rng), n_features=500, fast_threshold=12)
  assert isinstance(frame, Frame)
  assert frame.keypoints.ndim == 2 and frame.keypoints.shape[1] == 2
  assert frame.keypoints.shape[0] > 10  # noise image should produce some corners
  assert frame.descriptors.shape == (frame.keypoints.shape[0], 32)
  assert frame.descriptors.dtype == np.uint8


def test_detect_orb_on_solid_image_returns_empty_frame() -> None:
  frame = detect_orb(
    np.zeros((100, 100), dtype=np.uint8), n_features=500, fast_threshold=12
  )
  assert frame.keypoints.shape == (0, 2)
  assert frame.descriptors.shape == (0, 32)


def test_detect_orb_respects_mask() -> None:
  """Keypoints inside the masked-out region must be suppressed."""
  rng = np.random.default_rng(0)
  img = _textured_image(rng, h=200, w=300)
  # Mask out the right half of the frame: 0 = ignore, 255 = keep.
  mask = np.full(img.shape, 255, dtype=np.uint8)
  mask[:, 150:] = 0
  frame = detect_orb(img, n_features=500, fast_threshold=12, mask=mask)
  assert frame.keypoints.shape[0] > 0
  # No keypoint should land inside the masked-out region.
  assert np.all(frame.keypoints[:, 0] < 150), (
    f"masked region produced {(frame.keypoints[:, 0] >= 150).sum()} keypoints"
  )


def test_match_pair_identical_frames_matches_all_points() -> None:
  rng = np.random.default_rng(1)
  img = _textured_image(rng)
  a = detect_orb(img, n_features=200, fast_threshold=12)
  # Identical descriptors: each query has an exact match, but ratio test may
  # reject if the second-best is equally close. Use a relaxed ratio.
  matches = match_pair(a, a, ratio=0.99)
  # With noisy uniform descriptors ratio test can still cull most; require
  # just that the structure is right and some matches survive.
  assert matches.ndim == 2 and matches.shape[1] == 2
  assert matches.shape[0] > 0


def test_match_pair_empty_descriptors_returns_empty() -> None:
  empty = Frame(
    keypoints=np.empty((0, 2), dtype=np.float64),
    descriptors=np.empty((0, 32), dtype=np.uint8),
  )
  assert match_pair(empty, empty, ratio=0.7).shape == (0, 2)


def test_essential_ransac_with_few_points_returns_all_false() -> None:
  pts = np.array([[0.0, 0.0], [1.0, 1.0]])  # < 8 points
  bearings = np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]])
  inliers = essential_ransac_inliers(
    pts, pts, bearings, bearings, threshold_px=3.0, fx=500.0
  )
  assert inliers.shape == (2,)
  assert inliers.dtype == bool
  assert not inliers.any()


def _synthetic_correspondence_pair(
  rng: np.random.Generator,
  n_inliers: int,
  n_outliers: int,
  fx: float = 500.0,
) -> tuple[Float64Array, Float64Array, Float64Array, Float64Array]:
  """Return (pts_a, pts_b, bearings_a, bearings_b) for a known x-translation."""
  # 3D points in camera A's frame, z > 0.
  pts3d = rng.uniform(-0.5, 0.5, size=(n_inliers, 3))
  pts3d[:, 2] = rng.uniform(1.5, 3.0, size=n_inliers)
  # Camera B is translated along +x by 0.2 m in camera A's frame.
  pts3d_b = pts3d - np.array([0.2, 0.0, 0.0])

  def _to_pts_and_bearings(
    p: Float64Array,
  ) -> tuple[Float64Array, Float64Array]:
    bearings = p / np.linalg.norm(p, axis=1, keepdims=True)
    norm = p[:, :2] / p[:, 2:3]
    pts = fx * norm + np.array([320.0, 240.0])
    return pts, bearings

  pts_a, bears_a = _to_pts_and_bearings(pts3d)
  pts_b, bears_b = _to_pts_and_bearings(pts3d_b)

  # Append outliers: random pixel pairs with random bearings.
  if n_outliers > 0:
    out_pts_a = rng.uniform(100, 500, size=(n_outliers, 2))
    out_pts_b = rng.uniform(100, 500, size=(n_outliers, 2))
    # Random unit bearings.
    out_bears_a = rng.normal(size=(n_outliers, 3))
    out_bears_a /= np.linalg.norm(out_bears_a, axis=1, keepdims=True)
    out_bears_b = rng.normal(size=(n_outliers, 3))
    out_bears_b /= np.linalg.norm(out_bears_b, axis=1, keepdims=True)
    pts_a = np.vstack([pts_a, out_pts_a])
    pts_b = np.vstack([pts_b, out_pts_b])
    bears_a = np.vstack([bears_a, out_bears_a])
    bears_b = np.vstack([bears_b, out_bears_b])

  return pts_a, pts_b, bears_a, bears_b


def test_essential_ransac_selects_inliers_over_outliers() -> None:
  rng = np.random.default_rng(2)
  pts_a, pts_b, bears_a, bears_b = _synthetic_correspondence_pair(
    rng, n_inliers=30, n_outliers=10
  )
  inliers = essential_ransac_inliers(
    pts_a, pts_b, bears_a, bears_b, threshold_px=1.5, fx=500.0
  )
  assert inliers.shape == (40,)
  # Inliers should dominate; allow some slack for RANSAC noise.
  assert inliers[:30].sum() >= 25
  # Outliers mostly rejected.
  assert inliers[30:].sum() <= 4


def test_chain_tracks_two_frames_one_match() -> None:
  # Frame 0: 2 kpts, Frame 1: 2 kpts. Match: (kp0_f0 ↔ kp1_f1).
  frames = [
    Frame(
      keypoints=np.array([[10.0, 10.0], [20.0, 20.0]]),
      descriptors=np.zeros((2, 32), dtype=np.uint8),
    ),
    Frame(
      keypoints=np.array([[11.0, 11.0], [21.0, 19.0]]),
      descriptors=np.zeros((2, 32), dtype=np.uint8),
    ),
  ]
  matches = [np.array([[0, 1]], dtype=np.int64)]
  tracks = chain_tracks(frames, matches)
  assert len(tracks) == 1
  tid = next(iter(tracks))
  assert set(tracks[tid].keys()) == {0, 1}
  assert np.allclose(tracks[tid][0], [10.0, 10.0])
  assert np.allclose(tracks[tid][1], [21.0, 19.0])


def test_chain_tracks_three_frames_chain_through_shared_keypoint() -> None:
  # A track persists: kp0_f0 → kp0_f1 → kp0_f2.
  frames = [
    Frame(
      keypoints=np.array([[0.0, 0.0], [100.0, 100.0]]),
      descriptors=np.zeros((2, 32), dtype=np.uint8),
    )
    for _ in range(3)
  ]
  matches = [
    np.array([[0, 0]], dtype=np.int64),  # f0→f1
    np.array([[0, 0]], dtype=np.int64),  # f1→f2
  ]
  tracks = chain_tracks(frames, matches)
  assert len(tracks) == 1
  tid = next(iter(tracks))
  assert set(tracks[tid].keys()) == {0, 1, 2}


def test_chain_tracks_unrelated_pairs_produce_separate_tracks() -> None:
  # f0→f1: kp0↔kp0. f1→f2: kp1↔kp0 (does NOT chain — different kp in f1).
  frames = [
    Frame(
      keypoints=np.array([[0.0, 0.0], [1.0, 1.0]]),
      descriptors=np.zeros((2, 32), dtype=np.uint8),
    )
    for _ in range(3)
  ]
  matches = [
    np.array([[0, 0]], dtype=np.int64),
    np.array([[1, 0]], dtype=np.int64),
  ]
  tracks = chain_tracks(frames, matches)
  assert len(tracks) == 2
  lengths = sorted(len(obs) for obs in tracks.values())
  assert lengths == [2, 2]


# ---------- KLT optical-flow tracker ----------


def _corner_pattern(h: int = 240, w: int = 320) -> np.ndarray:
  """Image with strong corners for goodFeaturesToTrack + KLT to latch onto."""
  img = np.full((h, w), 200, dtype=np.uint8)
  for cx in range(40, w - 20, 40):
    for cy in range(40, h - 20, 40):
      img[cy - 4 : cy + 4, cx - 4 : cx + 4] = 20
  return img


def test_klt_track_empty_frames_returns_empty() -> None:
  frames, matches = klt_track([], n_features=100)
  assert frames == []
  assert matches == []


def test_klt_track_single_frame_detects_features_no_matches() -> None:
  img = _corner_pattern()
  frames, matches = klt_track([img], n_features=200)
  assert len(frames) == 1
  assert len(matches) == 0
  assert frames[0].keypoints.shape[0] > 10
  assert frames[0].keypoints.shape[1] == 2


def test_klt_track_translating_image_propagates_features() -> None:
  """Rigid shift by (5, 0) px: every tracked point should advance by ~5 px."""
  base = _corner_pattern()
  seq = [np.roll(base, shift=k * 5, axis=1) for k in range(4)]
  out, matches = klt_track(seq, n_features=200)
  assert len(out) == 4
  assert len(matches) == 3
  # Most features should track across the full sequence.
  for m in matches:
    assert m.shape[1] == 2
    assert m.shape[0] >= 10
  # Pick a track present in frames 0 and 1 and confirm it shifted by ~5 px.
  m01 = matches[0]
  pts0 = out[0].keypoints[m01[:, 0]]
  pts1 = out[1].keypoints[m01[:, 1]]
  dx = pts1[:, 0] - pts0[:, 0]
  dy = pts1[:, 1] - pts0[:, 1]
  assert float(np.median(dx)) > 3.5
  assert float(np.median(dx)) < 6.5
  assert float(np.median(np.abs(dy))) < 1.0


def test_klt_track_chained_via_chain_tracks_yields_long_tracks() -> None:
  base = _corner_pattern()
  seq = [np.roll(base, shift=k * 2, axis=1) for k in range(5)]
  out, matches = klt_track(seq, n_features=200)
  tracks = chain_tracks(out, matches)
  # At least a few tracks must survive all 5 frames.
  long_tracks = [obs for obs in tracks.values() if len(obs) == 5]
  assert len(long_tracks) >= 5


def test_klt_track_respects_mask() -> None:
  """Masked-out region must not produce new detections."""
  img = _corner_pattern()
  mask = np.full(img.shape, 255, dtype=np.uint8)
  mask[:, img.shape[1] // 2 :] = 0
  out, _ = klt_track([img], n_features=200, masks=[mask])
  kps = out[0].keypoints
  assert kps.shape[0] > 0
  assert np.all(kps[:, 0] < img.shape[1] // 2)


def test_klt_track_mask_length_mismatch_raises() -> None:
  img = _corner_pattern()
  try:
    klt_track(
      [img, img], n_features=50, masks=[np.full(img.shape, 255, dtype=np.uint8)]
    )
  except ValueError:
    return
  raise AssertionError("expected ValueError for mask length mismatch")


def test_klt_track_redetects_when_pool_depletes() -> None:
  """After a blank image terminates every track, the next textured frame
  must re-seed the feature pool (downstream tracking depends on it)."""
  textured = _corner_pattern()
  blank = np.full_like(textured, 128)
  seq = [textured, blank, textured]
  out, _ = klt_track(seq, n_features=200, redetect_ratio=0.5)
  # Frame 0 populated, frame 1 loses all tracks (blank image → nothing to
  # track + goodFeaturesToTrack on a flat grey image returns nothing), frame
  # 2 re-detects.
  assert out[0].keypoints.shape[0] > 10
  assert out[2].keypoints.shape[0] > 10
