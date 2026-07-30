"""Per-lens extrinsics decoded from an Insta360 ``offset_v3`` array."""

from __future__ import annotations

import math
from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.loader.parse_insv import InsvData

__all__ = ["Extrinsics", "LensOffset", "parse_insv_extrinsics", "parse_offset_v3"]

_BLOCK_SIZE = 19
_TOTAL_SIZE = 1 + 2 * _BLOCK_SIZE + 1


@dataclass(frozen=True)
class LensOffset:
  """One lens's ``offset_v3`` block in raw stitched-sensor units, the rear ``cx``
  counted from the stitched left edge."""

  xi: float
  fx: float
  fy: float
  cx: float
  cy: float
  yaw_deg: float
  pitch_deg: float
  roll_deg: float
  tx: float
  ty: float
  tz: float
  k1: float
  k2: float
  k3: float
  p1: float
  p2: float
  frame_width: int
  frame_height: int

  def to_pose3(self) -> gtsam.Pose3:
    """``body_T_lens``, with ``(tx, ty, tz)`` the body origin in the lens frame."""
    rot = gtsam.Rot3.Ypr(
      math.radians(self.yaw_deg),
      math.radians(self.pitch_deg),
      math.radians(self.roll_deg),
    )
    t_factory = gtsam.Point3(self.tx, self.ty, self.tz)
    return gtsam.Pose3(rot, -rot.rotate(t_factory))


@dataclass(frozen=True)
class Extrinsics:
  """Both lenses; the front lens's rotation gains a 180° turn about body z."""

  front: LensOffset
  rear: LensOffset

  @property
  def body_T_front_lens(self) -> gtsam.Pose3:
    raw = self.front.to_pose3()
    flip_z = gtsam.Rot3.Rz(math.pi)
    return gtsam.Pose3(flip_z.compose(raw.rotation()), raw.translation())

  @property
  def body_T_rear_lens(self) -> gtsam.Pose3:
    return self.rear.to_pose3()


def _decode_block(block: npt.NDArray[np.float64]) -> LensOffset:
  return LensOffset(
    xi=float(block[0]),
    fx=float(block[1]),
    fy=float(block[2]),
    cx=float(block[3]),
    cy=float(block[4]),
    yaw_deg=float(block[5]),
    pitch_deg=float(block[6]),
    roll_deg=float(block[7]),
    tx=float(block[8]),
    ty=float(block[9]),
    tz=float(block[10]),
    k1=float(block[11]),
    k2=float(block[12]),
    k3=float(block[13]),
    p1=float(block[14]),
    p2=float(block[15]),
    frame_width=int(block[16]),
    frame_height=int(block[17]),
  )


def parse_offset_v3(arr: npt.NDArray[np.float64]) -> Extrinsics:
  """Decode the 40-float ``offset_v3`` array into per-lens extrinsics."""
  arr = np.asarray(arr, dtype=np.float64)
  if arr.shape != (_TOTAL_SIZE,):
    raise ValueError(f"offset_v3 must have shape ({_TOTAL_SIZE},); got {arr.shape}")
  if int(arr[0]) != 2:
    raise ValueError(f"offset_v3 lens count must be 2; got {arr[0]}")
  front = _decode_block(arr[1 : 1 + _BLOCK_SIZE])
  rear = _decode_block(arr[1 + _BLOCK_SIZE : 1 + 2 * _BLOCK_SIZE])
  return Extrinsics(front=front, rear=rear)


def parse_insv_extrinsics(data: InsvData) -> Extrinsics:
  """Decode ``InsvData.offset_v3`` into per-lens extrinsics."""
  return parse_offset_v3(data.offset_v3)
