"""The single source of record entity paths, timeline names, and how the
timeline decodes.

A camera slot (``cam00``, ``cam01``, …) is an opaque handle: which physical
camera, gripper or arm it stands for is recorded as a value and may differ per
take.

A session is indexed by five clocks:

1. ``video_time`` (``TIMELINE``) — seconds on the session clock.
0. ``row`` (``ROW_TIMELINE``, state layers) — ``round(t * f_row)``, the
   integer row of the state stream's lattice; the true time stays on
   ``video_time`` beside it.
2. Session frame row — ``round(t * fps)``, integer, phase-locked across
   cameras; ``keyframe_index`` maps it to the next clock.
3. ``keyframe`` timeline bucket — ``ceil(row / stride)`` with
   ``stride = keyframe_stride(fps)``.
4. Solver keyframe ordinal ``kf`` — dense ``0..n_keyframes-1``, the index
   behind every solver structure. It steps by the run's ``keyframe_every``,
   only required to be a multiple of ``stride``, so ``kf`` equals
   ``keyframe_index(row)`` exactly when the stride is the auto one and the
   grid phase is zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from specialist_camera import SIDES as SIDES
from specialist_camera import Side as Side

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

APPLICATION_ID = "specialist"
# One property group per layer: a name has exactly one writer after a merge.
BASE_PROPERTIES = "base"
ESTIMATE_PROPERTIES = "estimate"
VIZ_PROPERTIES = "viz"
TIMELINE = "video_time"
KEYFRAME_TIMELINE = "keyframe"
ROW_TIMELINE = "row"
# An image's own capture instant on the row lattice, a value beside its blob.
CAPTURE_ROW = "capture_row"
NS_PER_SECOND = 1_000_000_000

CAMERA_ROOT = "camera"
WORLD = "world"
# The frame every world-referenced quantity names. An entity under a posed
# parent must declare it, or it rides a pose it is not expressed in.
WORLD_FRAME = "world"
BASE = "base"
ESTIMATE = "estimate"
VIZ = "viz"
MOCAP = "mocap"
STATE = "state"
ASSETS = "assets"


def state_layer(scale: int, f_row: float) -> str:
  """The directory one state configuration lives in: ``state_s12_r100`` for 12
  lens pixels per stored pixel and 100 rows per second."""
  return f"{STATE}_s{scale}_r{round(f_row)}"


def layer_dir(corpus: Path, layer: str) -> Path:
  """Where one layer's recordings sit in a layer-first corpus."""
  return corpus / layer


def layer_rrd(corpus: Path, layer: str, session: str) -> Path:
  """Where one session's layer sits in a layer-first corpus."""
  return layer_dir(corpus, layer) / f"{session}.rrd"


def asset_rrd(corpus: Path, asset: str) -> Path:
  """Where one dataset asset, a recording every session shares, sits."""
  return corpus / ASSETS / f"{asset}.rrd"


def camera_slots(serials: Sequence[str]) -> list[str]:
  """One handle per capture: the rank of its serial among the take's serials.

  The rank is dense within the take, so a different set of cameras numbers
  differently: anything comparing across takes keys on the stored
  ``serial_number``, never on the slot.
  """
  if not all(serials) or len(set(serials)) != len(serials):
    raise ValueError(f"cameras need one distinct serial each, got {list(serials)}")
  rank = {serial: i for i, serial in enumerate(sorted(serials))}
  return [f"cam{rank[serial]:02d}" for serial in serials]


def keyframe_stride(fps: float) -> int:
  """The auto keyframe interval: ``round(fps / 10)`` frames, about 0.10 s."""
  return max(1, round(fps / 10.0))


def keyframe_index(
  session_rows: npt.NDArray[np.int64], stride: int
) -> npt.NDArray[np.int64]:
  """``ceil(row / stride)`` per session frame row: the ``keyframe`` timeline
  value."""
  return (session_rows + stride - 1) // stride


def times_ns(column: pa.Array | pa.ChunkedArray) -> Int64Array:
  """A duration column as integer nanoseconds. A read that would copy raises."""
  values = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
  out: Int64Array = values.to_numpy(zero_copy_only=True).astype("int64")
  return out


def seconds(column: pa.Array | pa.ChunkedArray) -> Float64Array:
  """A duration column in seconds."""
  out: Float64Array = times_ns(column) / NS_PER_SECOND
  return out


def camera(slot: str) -> str:
  """The camera body: what it is, and its solved pose over time.

  This entity defines the body frame, and everything under it rides that pose.
  A child holding a world-frame quantity must say so with ``WORLD_FRAME``."""
  return f"{CAMERA_ROOT}/{slot}"


def lens(slot: str, side: str) -> str:
  """One lens: its pose in the camera body and its projection model."""
  return f"{camera(slot)}/{side}"


def lens_seat(slot: str, side: str) -> str:
  """The body -> lens rotation the bundle solved."""
  return f"{camera(slot)}/{side}_seat"


def video(slot: str, side: str) -> str:
  """A lens's encoded frames. A leaf: it carries no temporal children."""
  return f"{lens(slot, side)}/video"


HANDS = ("left", "right")


def state_hand(hand: str) -> str:
  """What the policy sees of one hand; which camera it was rides on it as a
  value."""
  return f"{STATE}/{hand}"


def state_image(hand: str) -> str:
  """The hand's rear-lens images at the policy's resolution and rate, blanked
  outside the fisheye disc."""
  return f"{state_hand(hand)}/image"


def state_pose(hand: str) -> str:
  """The gripper's base-tag frame in the world."""
  return f"{state_hand(hand)}/pose"


def state_lin_vel(hand: str) -> str:
  """The base-tag origin's velocity in the world frame, m/s."""
  return f"{state_hand(hand)}/lin_vel"


def state_ang_vel(hand: str) -> str:
  """The gripper's angular velocity in the world frame, rad/s."""
  return f"{state_hand(hand)}/ang_vel"


def state_opening(hand: str) -> str:
  """The coupled finger joint angle, rad, in the left finger's sign."""
  return f"{state_hand(hand)}/opening"


def state_finger(hand: str, joint: str) -> str:
  """One finger's joint edge, the opening put through the model."""
  return f"{state_hand(hand)}/{joint}"


TAGS_DETECTION = "tags"
KEYPOINTS_DETECTION = "keypoints"
REJECTED_KEYPOINTS_DETECTION = "keypoints_rejected"
DETECTION_KINDS = (TAGS_DETECTION, KEYPOINTS_DETECTION, REJECTED_KEYPOINTS_DETECTION)

TAG_AXES_OVERLAY = "tag_axes"
LANDMARK_REPROJECTION_OVERLAY = "landmark_reprojection"
OVERLAY_KINDS = (TAG_AXES_OVERLAY, LANDMARK_REPROJECTION_OVERLAY)


def detection(slot: str, side: str, kind: str) -> str:
  """What the lens measured, in its own pixels."""
  return f"{lens(slot, side)}/detection/{kind}"


def tag_center(slot: str, side: str) -> str:
  """A detected tag's centre pixel, the cheirality ray's origin."""
  return f"{detection(slot, side, TAGS_DETECTION)}/center"


def overlay(slot: str, side: str, kind: str) -> str:
  """A picture the solve draws in a lens's pixels: nothing reads it back."""
  return f"{lens(slot, side)}/overlay/{kind}"


def imu(slot: str) -> str:
  """The IMU: the rotation from its raw axes to the camera body, and its
  factory error terms."""
  return f"{camera(slot)}/imu"


def imu_signal(slot: str, kind: str) -> str:
  """A raw inertial stream, ``gyro`` or ``accel``, on the IMU's own axes."""
  return f"{imu(slot)}/{kind}"


def imu_bias(slot: str, kind: str) -> str:
  """One solved bias stream, ``gyro`` or ``accel``. The bias is on the body
  axes and the raw stream on the IMU's own, so correcting one with the other
  takes the rotation stored on the parent too."""
  return f"{imu(slot)}/bias/{kind}"


def exposure(slot: str) -> str:
  return f"{camera(slot)}/exposure"


def autoexposure(slot: str) -> str:
  return f"{camera(slot)}/autoexposure"


def lin_vel(slot: str) -> str:
  """Linear velocity in the world frame, at IMU rate."""
  return f"{camera(slot)}/lin_vel"


def base_tag(slot: str) -> str:
  """The gripper tag bolted to the camera body. Which physical tag it is
  rides on the entity as a value."""
  return f"{camera(slot)}/base_tag"


def joint_tag(slot: str, joint: str) -> str:
  """The tag riding one revolute joint, on the link it turns with."""
  return f"{gripper_part(slot, joint)}/tag"


def state_frame(slot: str) -> str:
  """The frame the solved keyframe pose defines, named apart from the
  propagated one: Rerun takes one definition per frame."""
  return f"{slot}_state_body"


def state(slot: str, quantity: str) -> str:
  """One quantity of the solved state at keyframes: ``pose`` or ``lin_vel``."""
  return f"{camera(slot)}/state/{quantity}"


def gripper(slot: str) -> str:
  """The gripper this camera rode, in the world frame."""
  return f"{camera(slot)}/gripper"


def link_frame(slot: str, link: str) -> str:
  """The frame of one model link on the gripper this camera rode."""
  return f"{slot}_{link}"


def gripper_part(slot: str, part: str) -> str:
  """Something rigidly under the gripper: ``base_tag``, a finger, ``opening``,
  a URDF link, or ``tf_static``."""
  return f"{gripper(slot)}/{part}"


def fit(slot: str, metric: str | None = None) -> str:
  """The camera's account of the solve: whole-take numbers on the parent, one
  per-keyframe series per child."""
  return f"{camera(slot)}/fit/{metric}" if metric else f"{camera(slot)}/fit"


def landmark(slot: str, side: str) -> str:
  """The map one lens contributed, in the world frame it names for itself."""
  return f"{camera(slot)}/landmark/{side}"


def factor(slot: str, kind: str) -> str:
  """The rays of a factor that fired, in the world frame it names for itself."""
  return f"{camera(slot)}/factor/{kind}"


def trajectory(slot: str) -> str:
  return f"{camera(slot)}/trajectory"


def anchor(tag_id: int) -> str:
  """A world-static tag, addressed by the tag it is."""
  return f"{WORLD}/anchor/{tag_id}"


def slot_of(path: str) -> str | None:
  """The camera slot an entity path belongs to, or ``None`` when it belongs to
  no camera."""
  parts = path.strip("/").split("/")
  return parts[1] if len(parts) > 1 and parts[0] == CAMERA_ROOT else None


def slots_in(paths: Iterable[str]) -> list[str]:
  """The camera slots a recording holds, read off its entity paths."""
  return sorted({slot for slot in map(slot_of, paths) if slot})


# The stored spelling of the components a consumer selects on, addressed as
# "/entity/path:Component:field".
SCALARS = "Scalars:scalars"
POSE_TRANSLATION = "Transform3D:translation"
POSE_QUATERNION = "Transform3D:quaternion"


def selector(entity: str, component: str) -> str:
  """One field of one entity, fully qualified."""
  return f"/{entity}:{component}"
