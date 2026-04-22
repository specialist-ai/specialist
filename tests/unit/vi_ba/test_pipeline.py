"""Tests for ``vi_ba.pipeline.filter_tracks_by_parallax``."""

from __future__ import annotations

import numpy as np
import pytest

from specialist.loader.mei_model import MeiParams, mei_project
from specialist.vi_ba.pipeline import filter_tracks_by_parallax


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
  """A track that observes the same camera-frame point across all keyframes
  has zero parallax — body-attached features look like this. It must be
  rejected. A track that sweeps across the FOV (large bearing change) is
  kept."""
  params = _params()
  body_mounted_pixel = _project_one(np.array([0.05, 0.0, 0.5]), params)
  static_track = {kf: body_mounted_pixel.copy() for kf in range(5)}

  moving_track = {
    kf: _project_one(np.array([0.05 + 0.05 * kf, 0.0, 0.5]), params) for kf in range(5)
  }

  tracks = {1: static_track, 2: moving_track}
  kept, rejected = filter_tracks_by_parallax(tracks, params, min_parallax_deg=0.5)
  assert 1 not in kept and 1 in rejected, "static-bearing track must be rejected"
  assert 2 in kept and 2 not in rejected, "wide-bearing-sweep track must be kept"


def test_threshold_zero_keeps_everything_with_two_or_more_obs() -> None:
  params = _params()
  obs = _project_one(np.array([0.1, 0.0, 0.5]), params)
  tracks = {1: {0: obs.copy(), 1: obs.copy()}}
  kept, rejected = filter_tracks_by_parallax(tracks, params, min_parallax_deg=0.0)
  assert 1 in kept


def test_single_observation_track_rejected() -> None:
  """A single-observation track has no pair to compute parallax — must
  be rejected as unfilterable."""
  params = _params()
  tracks = {1: {0: np.array([100.0, 100.0])}}
  kept, rejected = filter_tracks_by_parallax(tracks, params, min_parallax_deg=0.5)
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
  kept, rejected = filter_tracks_by_parallax(tracks, params, min_parallax_deg=0.1)
  assert set(kept.keys()).isdisjoint(rejected.keys())
  assert set(kept.keys()) | set(rejected.keys()) == set(tracks.keys())


def test_rejects_negative_threshold() -> None:
  params = _params()
  with pytest.raises(ValueError):
    filter_tracks_by_parallax({}, params, min_parallax_deg=-1.0)
