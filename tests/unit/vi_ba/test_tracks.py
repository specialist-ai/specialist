"""Tests for ``vi_ba.tracks``: the parallax filter and track unprojection."""

from __future__ import annotations

import numpy as np
import pytest

from specialist.loader.mei_model import MeiParams, mei_project, mei_unproject
from specialist.vi_ba.tracks import (
  filter_tracks_by_parallax,
  unproject_tracks,
)


def _params() -> MeiParams:
  return MeiParams(
    fx=400.0,
    fy=400.0,
    cx=480.0,
    cy=480.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
    xi=1.0,
  )


def _project_one(point_cam: np.ndarray, params: MeiParams) -> np.ndarray:
  return mei_project(point_cam.reshape(1, 3), params)[0]


def test_static_camera_track_kept_only_if_parallax_above_threshold() -> None:
  """A zero-parallax track, the same camera-frame point at every keyframe, is
  rejected; one that sweeps across the FOV is kept."""
  params = _params()
  body_mounted_pixel = _project_one(np.array([0.05, 0.0, 0.5]), params)
  static_track = {kf: body_mounted_pixel.copy() for kf in range(5)}

  moving_track = {
    kf: _project_one(np.array([0.05 + 0.05 * kf, 0.0, 0.5]), params) for kf in range(5)
  }

  tracks = {1: static_track, 2: moving_track}
  kept, rejected = filter_tracks_by_parallax(
    tracks, params, min_parallax=np.deg2rad(0.5)
  )
  assert 1 not in kept and 1 in rejected, "static-bearing track must be rejected"
  assert 2 in kept and 2 not in rejected, "wide-bearing-sweep track must be kept"


def test_threshold_zero_keeps_everything_with_two_or_more_obs() -> None:
  params = _params()
  obs = _project_one(np.array([0.1, 0.0, 0.5]), params)
  tracks = {1: {0: obs.copy(), 1: obs.copy()}}
  kept, rejected = filter_tracks_by_parallax(tracks, params, min_parallax=0.0)
  assert 1 in kept and 1 not in rejected


def test_single_observation_track_rejected() -> None:
  """A single-observation track has no pair to compute parallax and is
  rejected."""
  params = _params()
  tracks = {1: {0: np.array([100.0, 100.0])}}
  kept, rejected = filter_tracks_by_parallax(
    tracks, params, min_parallax=np.deg2rad(0.5)
  )
  assert 1 in rejected and 1 not in kept


def test_kept_and_rejected_are_disjoint_and_cover_input() -> None:
  params = _params()
  tracks = {
    i: {
      kf: _project_one(np.array([0.05 + 0.01 * i + 0.02 * kf, 0.0, 0.5]), params)
      for kf in range(3)
    }
    for i in range(10)
  }
  kept, rejected = filter_tracks_by_parallax(
    tracks, params, min_parallax=np.deg2rad(0.1)
  )
  assert set(kept.keys()).isdisjoint(rejected.keys())
  assert set(kept.keys()) | set(rejected.keys()) == set(tracks.keys())


def test_rejects_negative_threshold() -> None:
  params = _params()
  with pytest.raises(ValueError):
    filter_tracks_by_parallax({}, params, min_parallax=-1.0)


def test_unproject_tracks_batched_slices_match_per_track_scalar() -> None:
  """Batching every track's pixels through one ``mei_unproject`` call and
  slicing by span returns exactly what unprojecting each track alone would,
  with kfs sorted and tracks below ``min_obs`` dropped."""
  params = _params()
  tracks = {
    7: {
      2: np.array([120.0, 90.0]),
      0: np.array([300.0, 310.0]),
      1: np.array([50.0, 470.0]),
    },
    3: {5: np.array([460.0, 12.0]), 4: np.array([470.0, 495.0])},
    9: {0: np.array([240.0, 240.0])},  # single obs -> below min_obs
  }
  out = {
    tid: (kfs, bearings)
    for tid, kfs, bearings in unproject_tracks(
      tracks, params, min_obs=2, normalize=False
    )
  }
  assert set(out) == {7, 3}  # the single-observation track is filtered out
  for tid, (kfs, bearings) in out.items():
    assert kfs == sorted(tracks[tid])  # observations returned in keyframe order
    expected = mei_unproject(
      np.asarray([tracks[tid][k] for k in kfs], dtype=np.float64), params
    )
    assert np.array_equal(bearings, expected)  # batched slice == scalar unproject


def test_unproject_tracks_normalize_returns_unit_rows() -> None:
  params = _params()
  tracks = {1: {0: np.array([120.0, 90.0]), 1: np.array([300.0, 310.0])}}
  [(_, _, bearings)] = list(unproject_tracks(tracks, params, min_obs=2, normalize=True))
  assert np.allclose(np.linalg.norm(bearings, axis=1), 1.0)
