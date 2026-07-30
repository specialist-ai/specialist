"""Per-lens extrinsics decoded from an Insta360 ``offset_v3`` array.

``offset_v3`` is a 40-float blob: ``[lens_count, front_block(19),
rear_block(19), trailer]``. Each 19-float block packs xi, fx, fy, cx, cy,
yaw_deg, pitch_deg, roll_deg, tx, ty, tz, k1..k3, p1, p2, frame_width,
frame_height, _unknown. Intrinsics are reported in the **stitched raw
sensor frame** (11904 × 5952); decoded-video callers must rescale fx, fy,
cx, cy by the raw-to-decoded ratio. The rear lens cx is measured from the
stitched left edge — subtract the half-width to get its lens-local center.

Rotation uses ``gtsam.Rot3.Ypr`` (Z-Y-X intrinsic). The 180° physical
opposition between front and rear is absent from the per-lens rotations;
``Extrinsics`` applies it.
"""

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
  """One lens's decoded ``offset_v3`` block (raw stitched-sensor units)."""

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
    """Body → lens SE(3) transform built from ``(yaw, pitch, roll, t)``.

    Insta360 stores ``(tx, ty, tz)`` as the body origin's position in the
    lens frame (PnP-style), so the lens origin in body coordinates is
    ``-R @ t``."""
    rot = gtsam.Rot3.Ypr(
      math.radians(self.yaw_deg),
      math.radians(self.pitch_deg),
      math.radians(self.roll_deg),
    )
    t_factory = gtsam.Point3(self.tx, self.ty, self.tz)
    return gtsam.Pose3(rot, -rot.rotate(t_factory))


@dataclass(frozen=True)
class Extrinsics:
  """Both lenses; rear keeps the factory rotation, front is flipped 180°
  around body z."""

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
