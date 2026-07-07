"""AprilTag wall-anchor detection: per-keyframe ``camera → tag`` SE(3)
recovered by MEI-unprojecting the four corners and running planar PnP.
The pose pins metric scale + world-frame gauge for VI-BA."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import gtsam
import numpy as np
import numpy.typing as npt
from pupil_apriltags import Detector

from specialist.loader.mei_model import MeiParams, mei_unproject

Float64Array = npt.NDArray[np.float64]

_TAG_FAMILY = "tagStandard41h12"

__all__ = [
  "TagDetection",
  "anchor_pose_from_tags",
  "detect_all_tags",
  "estimate_camera_to_tag_pose",
  "rotate_prior",
  "tag_corners_local",
]


def tag_corners_local(tag_size: float) -> Float64Array:
  """The tag's four corners in the tag frame, ``(4, 3)``.

  pupil_apriltags corner order ``[BL, BR, TR, TL]`` (image-CCW from the
  bottom-left); tag-frame axes ``+x right``, ``+y up``, ``+z out``.
  """
  half = float(tag_size / 2.0)
  return np.array(
    [
      [-half, -half, 0.0],
      [+half, -half, 0.0],
      [+half, +half, 0.0],
      [-half, +half, 0.0],
    ],
    dtype=np.float64,
  )


@dataclass(frozen=True)
class TagDetection:
  """One AprilTag hit in one keyframe: id, center, corners (CCW image order)."""

  tag_id: int
  center: Float64Array  # (2,)
  corners: Float64Array  # (4, 2)


def estimate_camera_to_tag_pose(
  corners_px: Float64Array,
  tag_size: float,
  params: MeiParams,
  prev_cam_T_tag: gtsam.Pose3 | None = None,
) -> gtsam.Pose3 | None:
  """Camera → tag SE(3) from MEI-unprojected corners + planar IPPE PnP.

  ``SOLVEPNP_IPPE`` returns both reflectional candidates; ``prev_cam_T_tag``
  (the caller's gyro-propagated prior) selects between them, so the choice
  stays valid through fast rotation.
  """
  bearings = mei_unproject(corners_px.astype(np.float64), params)
  if not np.all(np.isfinite(bearings)) or np.any(bearings[:, 2] <= 1e-3):
    return None
  normalized = (bearings[:, :2] / bearings[:, 2:3]).astype(np.float64).reshape(-1, 1, 2)
  obj = tag_corners_local(tag_size).reshape(-1, 1, 3)
  k = np.eye(3)
  dist = np.zeros(4)
  n_sols, rvecs, tvecs, _ = cv2.solvePnPGeneric(
    obj, normalized, k, dist, flags=cv2.SOLVEPNP_IPPE
  )
  if n_sols == 0:
    return None
  poses: list[gtsam.Pose3] = []
  for rvec, tvec in zip(rvecs, tvecs, strict=True):
    rmat, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    poses.append(gtsam.Pose3(gtsam.Rot3(rmat), gtsam.Point3(*t)))
  # Bootstrap cheirality: tag +z must point toward the camera (R[:, 2] · t < 0).
  cheir = [
    p
    for p in poses
    if float(p.rotation().matrix()[:, 2] @ np.asarray(p.translation())) < 0.0
  ]
  if cheir:
    poses = cheir
  if prev_cam_T_tag is None or len(poses) == 1:
    return poses[0]
  return min(
    poses,
    key=lambda p: float(
      np.linalg.norm(gtsam.Rot3.Logmap(prev_cam_T_tag.rotation().between(p.rotation())))
    ),
  )


def anchor_pose_from_tags(
  tags_per_kf: list[list[TagDetection]],
  tag_id: int,
  tag_size: float,
  params: MeiParams,
  kf_rotations: list[Float64Array],
) -> dict[int, gtsam.Pose3]:
  """``{kf_idx: camera→tag SE(3)}`` from pre-detected tag lists, threading
  a gyro-propagated prior to disambiguate IPPE temporally.

  ``kf_rotations[k]`` (one 3x3 per keyframe, gyro-derived) maps camera-frame
  vectors at keyframe ``k-1`` into keyframe ``k``; the threaded prior is
  rotated through the increments accumulated since its sighting."""
  out: dict[int, gtsam.Pose3] = {}
  prev: gtsam.Pose3 | None = None
  prev_kf = 0
  for kf, tags in enumerate(tags_per_kf):
    for det in tags:
      if det.tag_id != tag_id:
        continue
      prior = (
        rotate_prior(prev, kf_rotations, prev_kf, kf) if prev is not None else None
      )
      pose = estimate_camera_to_tag_pose(
        det.corners, tag_size, params, prev_cam_T_tag=prior
      )
      if pose is not None:
        out[kf] = pose
        prev = pose
        prev_kf = kf
        break
  return out


def rotate_prior(
  prev: gtsam.Pose3,
  kf_rotations: list[Float64Array],
  prev_kf: int,
  kf: int,
) -> gtsam.Pose3:
  """Predict a ``cam_T_tag`` prior at keyframe ``kf`` from the pose sighted at
  ``prev_kf``: a pure camera rotation ``ΔR`` maps ``(R, t)`` to
  ``(ΔR·R, ΔR·t)``, with ``ΔR`` composed from the per-keyframe increments."""
  acc = np.eye(3)
  for j in range(prev_kf + 1, kf + 1):
    acc = kf_rotations[j] @ acc
  return gtsam.Pose3(gtsam.Rot3(acc), gtsam.Point3(0, 0, 0)).compose(prev)


def detect_all_tags(
  frames: list[np.ndarray],
) -> list[list[TagDetection]]:
  """Per-keyframe AprilTag list: every detected tag id + center + corners."""
  detector = Detector(families=_TAG_FAMILY)
  out: list[list[TagDetection]] = []
  for frame in frames:
    img = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.append(
      [
        TagDetection(
          tag_id=int(t.tag_id),
          center=np.asarray(t.center, dtype=np.float64),
          corners=np.asarray(t.corners, dtype=np.float64),
        )
        for t in detector.detect(img)
      ]
    )
  return out
