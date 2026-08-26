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
from specialist_camera import MeiParams, mei_project, mei_unproject
from specialist_corpus.fiducials import TAG_FAMILY, tag_corners_local

from specialist.vi_ba.features import cpu_budget

Float64Array = npt.NDArray[np.float64]

# Cap on the detector's own thread pool.
_DETECT_THREADS = 48
_detect_slots = threading.Lock()
_detector: Detector | None = None
_parallel = False
_local = threading.local()

__all__ = [
  "TagDetection",
  "anchor_pose_from_tags",
  "configure_detector",
  "detect_all_tags",
  "estimate_camera_to_tag_pose",
  "tag_center_bearing",
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
        families=TAG_FAMILY, nthreads=max(1, _detect_threads() // 2)
      )
      _local.detector = got
    return got
  with _detect_slots:
    if _detector is None:
      _detector = _OrderedTeardownDetector(
        families=TAG_FAMILY, nthreads=_detect_threads()
      )
    return _detector


@dataclass(frozen=True)
class TagDetection:
  """One AprilTag hit in one keyframe: id, center, corners (CCW image order)."""

  tag_id: int
  center: Float64Array  # (2,)
  corners: Float64Array  # (4, 2)


def tag_center_bearing(det: TagDetection, params: MeiParams) -> Float64Array:
  """Lens-frame bearing of a tag's center, the cheirality ray for MEI factors.
  A center the lens cannot unproject yields a zero bearing, which the factors
  read as "no ray"."""
  bearing = mei_unproject(
    np.asarray(det.center, dtype=np.float64).reshape(1, 2), params
  )[0]
  if not np.all(np.isfinite(bearing)):
    return np.zeros(3, dtype=np.float64)
  return np.asarray(bearing, dtype=np.float64)


def estimate_camera_to_tag_pose(
  corners_px: Float64Array,
  tag_size: float,
  params: MeiParams,
) -> gtsam.Pose3 | None:
  """Camera → tag SE(3) from MEI-unprojected corners + planar IPPE PnP.

  ``SOLVEPNP_IPPE`` returns both reflectional candidates; cheirality drops the
  back-facing one and the corner reprojection RMS ranks the survivors. Returns
  ``None`` for a sighting with no finite pose.
  """
  bearings = mei_unproject(corners_px.astype(np.float64), params)
  if not np.all(np.isfinite(bearings)) or np.any(bearings[:, 2] <= 1e-3):
    return None
  normalized = (bearings[:, :2] / bearings[:, 2:3]).astype(np.float64).reshape(-1, 1, 2)
  obj = tag_corners_local(tag_size).reshape(-1, 1, 3)
  k = np.eye(3)
  dist = np.zeros(4)
  _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
    obj, normalized, k, dist, flags=cv2.SOLVEPNP_IPPE
  )
  poses: list[gtsam.Pose3] = []
  for rvec, tvec in zip(rvecs, tvecs, strict=True):
    rmat, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    if np.all(np.isfinite(rmat)) and np.all(np.isfinite(t)):
      poses.append(gtsam.Pose3(gtsam.Rot3(rmat), gtsam.Point3(*t)))
  if not poses:
    return None
  # Bootstrap cheirality: tag +z must point toward the camera (R[:, 2] · t < 0).
  cheir = [
    p
    for p in poses
    if float(p.rotation().matrix()[:, 2] @ np.asarray(p.translation())) < 0.0
  ]
  if cheir:
    poses = cheir
  if len(poses) == 1:
    return poses[0]
  local = tag_corners_local(tag_size)
  points = np.concatenate(
    [(p.rotation().matrix() @ local.T).T + np.asarray(p.translation()) for p in poses]
  )
  pixels = mei_project(points, params).reshape(len(poses), 4, 2)
  rms = np.sqrt(np.mean(np.sum((pixels - corners_px) ** 2, axis=2), axis=1))
  return poses[int(np.argmin(rms))]


def anchor_pose_from_tags(
  tags_per_kf: list[list[TagDetection]],
  tag_id: int,
  tag_size: float,
  params: MeiParams,
) -> dict[int, gtsam.Pose3]:
  """``{kf_idx: camera→tag SE(3)}`` from pre-detected tag lists."""
  out: dict[int, gtsam.Pose3] = {}
  for kf, tags in enumerate(tags_per_kf):
    for det in tags:
      if det.tag_id != tag_id:
        continue
      pose = estimate_camera_to_tag_pose(det.corners, tag_size, params)
      if pose is not None:
        out[kf] = pose
        break
  return out


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
