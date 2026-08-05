"""AprilTag wall-anchor detection: per-keyframe ``camera → tag`` SE(3)
recovered by MEI-unprojecting the four corners and running planar PnP.
The pose pins metric scale + world-frame gauge for VI-BA."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import gtsam
import numpy as np
import numpy.typing as npt
from pupil_apriltags import Detector

from specialist.loader.mei_model import MeiParams, mei_unproject
from specialist.vi_ba.features import cpu_budget

Float64Array = npt.NDArray[np.float64]

_TAG_FAMILY = "tagStandard41h12"
# Cap on the detector's own thread pool.
_DETECT_THREADS = 48
_detect_slots = threading.Lock()
_detector: Detector | None = None
_parallel = False
_local = threading.local()

__all__ = [
  "TagDetection",
  "anchor_pose_from_tags",
  "detect_all_tags",
  "estimate_camera_to_tag_pose",
  "tag_center_bearing",
  "tag_corners_in_frame",
  "tag_corners_local",
]


def _detect_threads() -> int:
  """Width of the detector's pool on this host."""
  return min(_DETECT_THREADS, cpu_budget())


def configure_detector(parallel: bool) -> None:
  """Choose per-thread tag detectors (``parallel``) or one shared behind a lock.
  Each instance costs ~160 MB."""
  global _parallel
  _parallel = parallel


@contextmanager
def _detect_slot() -> Iterator[None]:
  """Hold the process's single tag-detection slot."""
  if _parallel:
    yield
    return
  with _detect_slots:
    yield


class _OrderedTeardownDetector(Detector):  # type: ignore[misc]
  """Destroy the detector before its tag families: the binding's own order is a
  use-after-free that corrupts the heap."""

  def __del__(self) -> None:
    detector = getattr(self, "tag_detector_ptr", None)
    if detector is None:
      return
    self.libc.apriltag_detector_destroy.restype = None
    self.libc.apriltag_detector_destroy(detector)
    for family, tf in self.tag_families.items():
      destroy = getattr(self.libc, f"{family}_destroy")
      destroy.restype = None
      destroy(tf)
    self.tag_detector_ptr = None


def _shared_detector() -> Detector:
  """This thread's detector when parallel, else the process-wide one."""
  global _detector
  if _parallel:
    got = getattr(_local, "detector", None)
    if got is None:
      got = _OrderedTeardownDetector(
        families=_TAG_FAMILY, nthreads=max(1, _detect_threads() // 2)
      )
      _local.detector = got
    return got
  with _detect_slots:
    if _detector is None:
      _detector = _OrderedTeardownDetector(
        families=_TAG_FAMILY, nthreads=_detect_threads()
      )
    return _detector


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


def tag_corners_in_frame(pose: gtsam.Pose3, tag_size: float) -> Float64Array:
  """A tag's four metric corners transformed by ``pose`` (e.g. a tag's world
  pose, or a joint's tag-at-rest pose), ``(4, 3)``."""
  return np.array(
    [pose.transformFrom(gtsam.Point3(*c)) for c in tag_corners_local(tag_size)],
    dtype=np.float64,
  )


def tag_center_bearing(det: TagDetection, params: MeiParams) -> Float64Array:
  """Lens-frame bearing of a tag's center, the cheirality ray for MEI factors."""
  bearing = mei_unproject(
    np.asarray(det.center, dtype=np.float64).reshape(1, 2), params
  )[0]
  return np.asarray(bearing, dtype=np.float64)


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
    prior = _rotate_prior(prev, kf_rotations, prev_kf, kf) if prev is not None else None
    for det in tags:
      if det.tag_id != tag_id:
        continue
      pose = estimate_camera_to_tag_pose(
        det.corners, tag_size, params, prev_cam_T_tag=prior
      )
      if pose is not None:
        out[kf] = pose
        prev = pose
        prev_kf = kf
        break
  return out


def _rotate_prior(
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
  """Per-keyframe AprilTag list: every detected tag id + center + corners.

  The native detector is not safe to enter from two threads at once.
  """
  detector = _shared_detector()
  with _detect_slot():
    return [
      [
        TagDetection(
          tag_id=int(t.tag_id),
          center=np.asarray(t.center, dtype=np.float64),
          corners=np.asarray(t.corners, dtype=np.float64),
        )
        for t in detector.detect(
          frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )
      ]
      for frame in frames
    ]
