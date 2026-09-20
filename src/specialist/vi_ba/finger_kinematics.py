"""Finger-tag kinematics: each tag rides a revolute joint off the base tag,
``body_T_tag(theta) = body_T_base_tag · base_T_joint · Rz(theta) · joint_T_tag0``."""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt
from specialist_camera import MeiParams, mei_project
from specialist_corpus.fiducials import tag_corners_in_frame

from specialist.vi_ba.anchor_detection import (
  TagDetection,
  anchor_pose_from_tags,
  tag_center_bearings,
)
from specialist.vi_ba.build import LensSetup
from specialist.vi_ba.geometry import average_pose, consensus_pose, rotate_about_z
from specialist.vi_ba.tag_discovery import GripperTags
from specialist.vi_ba.urdf_joints import FINGER_JOINTS, RevoluteJoint

Float64Array = npt.NDArray[np.float64]

__all__ = [
  "FingerKfObs",
  "FingerReproj",
  "build_finger_reproj",
  "pin_angles_to_keyframes",
  "recover_gripper_tags",
]

_THETA_SCAN_SAMPLES = 48


@dataclass(frozen=True)
class FingerKfObs:
  """One keyframe's finger-tag corner detection, ready for a reprojection factor."""

  theta_seed: float
  pixels: Float64Array  # (4, 2) detected corners, order [BL, BR, TR, TL]
  obs_bearing: Float64Array  # (3,) tag direction in the lens frame (cheirality)


@dataclass(frozen=True)
class FingerReproj:
  """Everything the joint graph needs to constrain one finger's joint angle."""

  base_T_joint: gtsam.Pose3  # URDF joint origin in the base frame; its z is the axis
  corners_in_pivot: Float64Array  # (4, 3) tag corners in the joint frame at theta=0
  observations: dict[int, FingerKfObs]


def _theta_scan(
  corners_in_pivot: Float64Array,
  body_T_joint: gtsam.Pose3,
  lens: LensSetup,
  lower: float,
  upper: float,
) -> tuple[Float64Array, Float64Array]:
  """The scanned angles over ``[lower, upper]`` and the corners' pixels at each."""
  lens_T_body = lens.extrinsic.inverse()
  r_lb = lens_T_body.rotation().matrix()
  t_lb = np.asarray(lens_T_body.translation(), dtype=np.float64)
  r_bj = body_T_joint.rotation().matrix()
  t_bj = np.asarray(body_T_joint.translation(), dtype=np.float64)
  thetas = np.linspace(lower, upper, _THETA_SCAN_SAMPLES)
  rz = rotate_about_z(corners_in_pivot, thetas)
  p_lens = (rz @ r_bj.T + t_bj) @ r_lb.T + t_lb
  projected = mei_project(p_lens.reshape(-1, 3), lens.params).reshape(
    p_lens.shape[:2] + (2,)
  )
  return thetas, projected


def _seed_thetas(
  pixels: Float64Array,
  corners_in_pivot: Float64Array,
  body_T_joint: gtsam.Pose3,
  lens: LensSetup,
  lower: float,
  upper: float,
) -> Float64Array:
  """Per ``(m, 4, 2)`` sighting, the scanned angle that best reprojects it."""
  thetas, projected = _theta_scan(corners_in_pivot, body_T_joint, lens, lower, upper)
  errors = np.sum((projected[None] - pixels[:, None]) ** 2, axis=(2, 3))
  # A behind-the-mirror corner projects to NaN; an all-NaN scan falls back to
  # the lower bound.
  best: Float64Array = thetas[
    np.argmin(np.where(np.isnan(errors), np.inf, errors), axis=1)
  ]
  return best


def build_finger_reproj(
  joint: RevoluteJoint,
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  finger_tag_id: int,
  body_T_base_tag: gtsam.Pose3,
  tag_size: float,
  n_keyframes: int,
) -> FingerReproj | None:
  """One finger's reprojection inputs from the URDF, ``None`` when its tag is never
  detected."""
  assert joint.joint_T_tag0 is not None
  base_T_joint = joint.base_T_joint
  body_T_joint_seed = body_T_base_tag.compose(base_T_joint)
  corners_in_pivot = tag_corners_in_frame(joint.joint_T_tag0.matrix(), tag_size)
  seen: list[tuple[int, TagDetection]] = []
  for kf in range(min(n_keyframes, len(tags_per_kf))):
    det = next((d for d in tags_per_kf[kf] if d.tag_id == finger_tag_id), None)
    if det is not None:
      seen.append((kf, det))
  if not seen:
    return None
  pixels = np.stack([np.asarray(det.corners, dtype=np.float64) for _, det in seen])
  thetas = _seed_thetas(
    pixels, corners_in_pivot, body_T_joint_seed, lens, joint.lower, joint.upper
  )
  bearings = tag_center_bearings([det for _, det in seen], lens.params)
  observations = {
    kf: FingerKfObs(theta_seed=float(theta), pixels=px, obs_bearing=bearing)
    for (kf, _), theta, px, bearing in zip(seen, thetas, pixels, bearings, strict=True)
  }
  return FingerReproj(
    base_T_joint=base_T_joint,
    corners_in_pivot=corners_in_pivot,
    observations=observations,
  )


def pin_angles_to_keyframes(
  frame_times: Float64Array,
  frame_angles: Float64Array,
  keyframe_times: Float64Array,
  keyframe_angles: Float64Array,
) -> Float64Array:
  """The per-frame angles bent linearly within each interval to pass through the
  solved keyframe angles."""
  pinned = np.asarray(frame_angles, dtype=np.float64).copy()
  raw_at_kf = np.interp(keyframe_times, frame_times, frame_angles)
  offsets = keyframe_angles - raw_at_kf
  for k in range(len(keyframe_times) - 1):
    left, right = offsets[k], offsets[k + 1]
    if not (np.isfinite(left) and np.isfinite(right)):
      continue
    span = keyframe_times[k + 1] - keyframe_times[k]
    inside = (frame_times >= keyframe_times[k]) & (frame_times <= keyframe_times[k + 1])
    fraction = (frame_times[inside] - keyframe_times[k]) / span
    pinned[inside] = frame_angles[inside] + (1 - fraction) * left + fraction * right
  return pinned


def frame_finger_angles(
  joint: RevoluteJoint,
  tags_per_frame: list[list[TagDetection]],
  lens: LensSetup,
  finger_tag_id: int,
  body_T_base_tag: gtsam.Pose3,
  tag_size: float,
) -> tuple[npt.NDArray[np.int64], Float64Array]:
  """Detection-only joint angles for every frame where the finger tag is seen,
  as ``(frame indices, angles)``, from the same scan the keyframe seeds use."""
  assert joint.joint_T_tag0 is not None
  body_T_joint = body_T_base_tag.compose(joint.base_T_joint)
  corners_in_pivot = tag_corners_in_frame(joint.joint_T_tag0.matrix(), tag_size)
  frames: list[int] = []
  pixels: list[Float64Array] = []
  for i, dets in enumerate(tags_per_frame):
    det = next((d for d in dets if d.tag_id == finger_tag_id), None)
    if det is None:
      continue
    frames.append(i)
    pixels.append(np.asarray(det.corners, dtype=np.float64))
  if not frames:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
  angles = _seed_thetas(
    np.stack(pixels), corners_in_pivot, body_T_joint, lens, joint.lower, joint.upper
  )
  return np.asarray(frames, dtype=np.int64), np.asarray(angles, dtype=np.float64)


def _pool_pose(poses: list[gtsam.Pose3], what: str) -> gtsam.Pose3:
  """The agreeing cluster's pose, or the plain median when the pool splits."""
  agreed = consensus_pose(poses)
  if agreed is not None:
    return agreed
  print(f"  [warn] [gripper] {what} sightings split; seeding on the median")
  return average_pose(poses)


def _joint_angles_per_kf(
  tags_per_kf: list[list[TagDetection]],
  params: MeiParams,
  cam_T_base: gtsam.Pose3,
  finger_tag_id: int,
  tag_size: float,
  n_keyframes: int,
) -> list[float]:
  """``Logmap(base_tag_T_finger)[2]`` per keyframe — the signed rotation about
  the URDF revolute-joint axis (``+z`` by URDF convention), against the base
  tag's pose averaged over its sightings."""
  out: list[float] = [float("nan")] * n_keyframes
  cam_T_finger_per_kf = anchor_pose_from_tags(
    tags_per_kf[:n_keyframes], finger_tag_id, tag_size, params
  )
  for kf, cam_T_finger in cam_T_finger_per_kf.items():
    base_tag_T_finger = cam_T_base.between(cam_T_finger)
    out[kf] = float(gtsam.Rot3.Logmap(base_tag_T_finger.rotation())[2])
  return out


def recover_gripper_tags(
  tags: GripperTags,
  joints: dict[str, RevoluteJoint],
  tags_per_kf: list[list[TagDetection]],
  lens: LensSetup,
  tag_size: float,
  n_keyframes: int,
  tags_per_frame: list[list[TagDetection]],
  frame_times: Float64Array,
) -> tuple[
  gtsam.Pose3,
  dict[str, list[float]],
  dict[str, FingerReproj],
  dict[str, tuple[Float64Array, Float64Array]],
]:
  """The base tag's body pose, each finger's per-keyframe and per-frame joint angles,
  and the finger reprojection inputs."""
  base_tag_obs = anchor_pose_from_tags(
    tags_per_kf[:n_keyframes],
    tag_id=tags.base,
    tag_size=tag_size,
    params=lens.params,
  )
  if not base_tag_obs:
    raise ValueError(
      f"base tag {tags.base} was discovered but never resolved a pose "
      f"across {n_keyframes} rear keyframes — its base-tag, finger, and "
      f"cross-tracker factors would all be silently disabled"
    )
  cam_T_base = _pool_pose(list(base_tag_obs.values()), "cam_T_base")
  body_T_base_tag = _pool_pose(
    [lens.extrinsic.compose(p) for p in base_tag_obs.values()], "body_T_base_tag"
  )

  joint_angles: dict[str, list[float]] = {}
  finger_reproj: dict[str, FingerReproj] = {}
  frame_angles: dict[str, tuple[Float64Array, Float64Array]] = {}
  for joint_name, finger_id in zip(FINGER_JOINTS, tags.fingers, strict=True):
    if finger_id <= 0:
      continue
    joint_angles[joint_name] = _joint_angles_per_kf(
      tags_per_kf,
      lens.params,
      cam_T_base,
      finger_tag_id=finger_id,
      tag_size=tag_size,
      n_keyframes=n_keyframes,
    )
    joint = joints[joint_name]
    reproj = build_finger_reproj(
      joint,
      tags_per_kf,
      lens,
      finger_tag_id=finger_id,
      body_T_base_tag=body_T_base_tag,
      tag_size=tag_size,
      n_keyframes=n_keyframes,
    )
    if reproj is None:
      print(
        f"  [warn] [gripper] finger tag {finger_id} ({joint_name}) never detected "
        f"in the rear lens; its angle stays PnP-only, unrefined by the bundle"
      )
    else:
      finger_reproj[joint_name] = reproj
    indices, angles = frame_finger_angles(
      joint,
      tags_per_frame,
      lens,
      finger_tag_id=finger_id,
      body_T_base_tag=body_T_base_tag,
      tag_size=tag_size,
    )
    if len(indices):
      frame_angles[joint_name] = (frame_times[indices], angles)
  return body_T_base_tag, joint_angles, finger_reproj, frame_angles
