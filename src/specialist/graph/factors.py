"""Factor-construction helpers wrapping GTSAM factors with the right noise models.

Each helper produces a single GTSAM factor ready to ``add`` to a
``NonlinearFactorGraph``. The noise model is always a diagonal SE(3) noise
in GTSAM's ``[ω, v]`` rotation-first tangent convention.

The observation helpers fold the camera-on-tracker and tag-on-tracker
extrinsics into the measurement so the resulting factor lives on the
tracker pose variables directly. Since those extrinsics are rigid
isometries the diagonal tangent-space noise model is preserved under
transformation (the eigenvalues of the covariance are invariant under
rotation of the 6D tangent frame).
"""

from __future__ import annotations

import gtsam
import numpy as np


def _sigmas(
  translation_sigma: float, rotation_sigma: float, huber_k: float | None = None
) -> gtsam.noiseModel.Base:
  sigmas = np.array(
    [
      rotation_sigma,
      rotation_sigma,
      rotation_sigma,
      translation_sigma,
      translation_sigma,
      translation_sigma,
    ],
    dtype=np.float64,
  )
  base = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
  if huber_k is None:
    return base
  if huber_k < 0:
    # Negative encodes Cauchy: |huber_k| is the Cauchy k.
    return gtsam.noiseModel.Robust.Create(
      gtsam.noiseModel.mEstimator.Cauchy.Create(-huber_k), base
    )
  return gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(huber_k), base
  )


def make_anchor_prior(
  anchor_key: int,
  anchor_pose: gtsam.Pose3,
  translation_sigma: float,
  rotation_sigma: float,
) -> gtsam.PriorFactorPose3:
  """Tight-σ prior pinning the anchor-tag pose — the gauge fix for the graph."""
  return gtsam.PriorFactorPose3(
    anchor_key, anchor_pose, _sigmas(translation_sigma, rotation_sigma)
  )


def make_odometry_factor(
  key_i: int,
  key_j: int,
  relative_pose: gtsam.Pose3,
  translation_sigma: float,
  rotation_sigma: float,
) -> gtsam.BetweenFactorPose3:
  """BetweenFactor on consecutive tracker keyframes from a noisy relative-pose measurement."""
  return gtsam.BetweenFactorPose3(
    key_i,
    key_j,
    relative_pose,
    _sigmas(translation_sigma, rotation_sigma),
  )


def make_anchor_observation_factor(
  tracker_key: int,
  anchor_key: int,
  camera_to_anchor: gtsam.Pose3,
  camera_extrinsic: gtsam.Pose3,
  translation_sigma: float,
  rotation_sigma: float,
  huber_k: float | None = None,
) -> gtsam.BetweenFactorPose3:
  """BetweenFactor linking ``tracker_key`` to ``anchor_key`` via one camera observation.

  ``camera_extrinsic`` composes with the camera→anchor measurement to give
  the tracker→anchor relative pose. ``huber_k`` (σ-normalized) enables a
  Huber robust loss to cap single-frame PnP outliers.
  """
  tracker_to_anchor = camera_extrinsic.compose(camera_to_anchor)
  return gtsam.BetweenFactorPose3(
    tracker_key,
    anchor_key,
    tracker_to_anchor,
    _sigmas(translation_sigma, rotation_sigma, huber_k=huber_k),
  )


def make_cross_tracker_factor(
  observer_key: int,
  tag_owner_key: int,
  camera_to_tag: gtsam.Pose3,
  observer_camera_extrinsic: gtsam.Pose3,
  tag_on_owner_extrinsic: gtsam.Pose3,
  translation_sigma: float,
  rotation_sigma: float,
  huber_k: float | None = None,
) -> gtsam.BetweenFactorPose3:
  """BetweenFactor linking two trackers via one tracker observing the other's tag.

  Composes the observer's camera extrinsic with the camera→tag measurement
  and then the inverse of the tag-on-owner extrinsic to get the
  observer→tag_owner relative pose the BetweenFactor expects.
  """
  observer_to_owner = observer_camera_extrinsic.compose(camera_to_tag).compose(
    tag_on_owner_extrinsic.inverse()
  )
  return gtsam.BetweenFactorPose3(
    observer_key,
    tag_owner_key,
    observer_to_owner,
    _sigmas(translation_sigma, rotation_sigma, huber_k=huber_k),
  )
