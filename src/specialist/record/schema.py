"""The single source of record entity paths, timeline names, and how the
timeline decodes.

Every layer of a session (matched by ``recording_id``) builds its paths through
these helpers, so no two writers can spell one entity differently and no
consumer has to guess.

A camera slot (``cam00``, ``cam01``, …) is an opaque handle. It says nothing
about which physical camera, which gripper, or which arm; those are recorded as
values, discovered later, and may differ per take. ``camera_slots`` mints the
handles by sorting on camera serial so a re-conversion cannot permute them.

A session is indexed by four clocks:

1. ``video_time`` (``TIMELINE``) — seconds on the session clock.
2. Session frame row — ``round(t * fps)``, integer, phase-locked across
   cameras; ``keyframe_index`` maps it to the next clock.
3. ``keyframe`` timeline bucket — ``ceil(row / stride)`` with
   ``stride = keyframe_stride(fps)``, the review cursor's step.
4. Solver keyframe ordinal ``kf`` — dense ``0..n_keyframes-1``, the index
   behind every solver structure (poses, tracks, theta keys). It steps by
   the run's ``keyframe_every``, which is only required to be a multiple of
   ``stride``, so ``kf`` equals ``keyframe_index(row)`` exactly when the
   stride is the auto one and the grid phase is zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pyarrow as pa

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

# The two sides of a stitched dual-fisheye capture; every per-lens entity and
# payload is keyed by one of these.
Side = Literal["front", "rear"]
SIDES: tuple[Side, Side] = ("front", "rear")

APPLICATION_ID = "specialist"
TIMELINE = "video_time"
# Sequence timeline bucketing video frames to solver keyframes: frame ``i``
# carries index ``ceil(i / stride)``, so latest-at on this timeline lands on
# frame ``stride * k`` exactly and the cursor steps keyframe to keyframe.
KEYFRAME_TIMELINE = "keyframe"
NS_PER_SECOND = 1_000_000_000

CAMERA_ROOT = "camera"
WORLD = "world"
# The frame every world-referenced quantity names. An entity that does not name
# a frame inherits an implicit one from its path, so a world-frame entity under
# a posed parent has to declare this or it rides a pose it is not expressed in.
WORLD_FRAME = "world"
ESTIMATE = "estimate"


def camera_slots(serials: Sequence[str]) -> list[str]:
  """One handle per capture: the rank of its serial among the take's serials.

  Ranking a recorded fact rather than the order the captures were passed in is
  what stops a re-conversion of the same cameras from moving one to another
  slot, so a missing or shared serial refuses rather than falling back to
  argument order. The rank is dense within the take, so a DIFFERENT set of
  cameras numbers differently: anything comparing across takes keys on the
  stored ``serial_number``, never on the slot.
  """
  if not all(serials) or len(set(serials)) != len(serials):
    raise ValueError(f"cameras need one distinct serial each, got {list(serials)}")
  rank = {serial: i for i, serial in enumerate(sorted(serials))}
  return [f"cam{rank[serial]:02d}" for serial in serials]


def keyframe_stride(fps: float) -> int:
  """The auto keyframe interval: ~0.10 s via ``round(fps / 10)`` (3 at 30 fps,
  6 at 60 fps). The converter buckets the ``keyframe`` timeline with this, and
  the solve's auto stride matches it by construction."""
  return max(1, round(fps / 10.0))


def keyframe_index(
  session_rows: npt.NDArray[np.int64], stride: int
) -> npt.NDArray[np.int64]:
  """``ceil(row / stride)`` per session frame row: the ``keyframe`` timeline
  value."""
  return (session_rows + stride - 1) // stride


def times_ns(column: pa.Array | pa.ChunkedArray) -> Int64Array:
  """A duration column as integer nanoseconds, taken whole out of arrow. A
  chunked column is joined first so ``zero_copy_only`` can hold, which makes a
  silently copied read an error rather than a doubled peak."""
  values = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
  out: Int64Array = values.to_numpy(zero_copy_only=True).astype("int64")
  return out


def seconds(column: pa.Array | pa.ChunkedArray) -> Float64Array:
  """A duration column in seconds."""
  out: Float64Array = times_ns(column) / NS_PER_SECOND
  return out


def camera(slot: str) -> str:
  """The camera body: what it is (serial, model, firmware, capture telemetry)
  and where it is (its solved pose over time).

  This entity is the body frame. Everything bolted to the camera hangs under
  it -- the lenses, the IMU, the base tag -- and rides that pose for free,
  because an entity with no frame of its own is placed by composing the
  transforms along its path. Anything under here holding a world-frame
  quantity must therefore say so with ``WORLD_FRAME``."""
  return f"{CAMERA_ROOT}/{slot}"


def lens(slot: str, side: str) -> str:
  """One lens: its pose in the camera body and its projection model."""
  return f"{camera(slot)}/{side}"


def video(slot: str, side: str) -> str:
  """A lens's encoded frames. A leaf: a training field reading it must see no
  temporal children, or the sample index gains rows no frame stands behind."""
  return f"{lens(slot, side)}/video"


TAGS_DETECTION = "tags"
KEYPOINTS_DETECTION = "keypoints"
REJECTED_KEYPOINTS_DETECTION = "keypoints_rejected"
DETECTION_KINDS = (TAGS_DETECTION, KEYPOINTS_DETECTION, REJECTED_KEYPOINTS_DETECTION)

TAG_AXES_OVERLAY = "tag_axes"
LANDMARK_REPROJECTION_OVERLAY = "landmark_reprojection"
STATIC_MASK_OVERLAY = "static_mask"
OVERLAY_KINDS = (TAG_AXES_OVERLAY, LANDMARK_REPROJECTION_OVERLAY, STATIC_MASK_OVERLAY)


def detection(slot: str, side: str, kind: str) -> str:
  """What the lens measured in its own pixels: the pixels a factor reads."""
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
  """One solved bias stream, ``gyro`` or ``accel``, beside the raw stream it
  corrects. The bias is on the body axes the solve preintegrates in and the raw
  stream is on the IMU's own, so correcting one with the other takes the
  rotation stored on the parent as well."""
  return f"{imu(slot)}/bias/{kind}"


def exposure(slot: str) -> str:
  return f"{camera(slot)}/exposure"


def autoexposure(slot: str) -> str:
  return f"{camera(slot)}/autoexposure"


def lin_vel(slot: str) -> str:
  """Linear velocity in the world frame, at IMU rate."""
  return f"{camera(slot)}/lin_vel"


def base_tag(slot: str) -> str:
  """The gripper tag bolted to the camera body, a child of the body it is
  measured from. Which physical tag it is rides on the entity as a value."""
  return f"{camera(slot)}/base_tag"


def joint_model(slot: str, joint: str) -> str:
  """What the model says about one revolute joint. A child of the link, whose
  own instance count is its pose's: the axis alone is three numbers wide."""
  return f"{gripper_part(slot, joint)}/model"


def joint_tag(slot: str, joint: str) -> str:
  """The tag riding one revolute joint, on the link it turns with."""
  return f"{gripper_part(slot, joint)}/tag"


def state_frame(slot: str) -> str:
  """The frame the solved keyframe pose defines: the body frame as the solver
  estimates it, named apart from the propagated one the mounted parts hang
  from, because Rerun takes one definition per frame."""
  return f"{slot}_state_body"


def state(slot: str, quantity: str) -> str:
  """One quantity of the solved state at keyframes: ``pose`` or ``lin_vel``,
  distinct from the propagated series on the camera itself, which is a
  different estimate at a different rate."""
  return f"{camera(slot)}/state/{quantity}"


def gripper(slot: str) -> str:
  """The gripper this camera rode, in the world frame."""
  return f"{camera(slot)}/gripper"


def gripper_part(slot: str, part: str) -> str:
  """Something rigidly under the gripper: ``base_tag``, a finger, ``opening``,
  a URDF link, or ``tf_static``."""
  return f"{gripper(slot)}/{part}"


def fit(slot: str, metric: str | None = None) -> str:
  """The camera's own account of the solve: the whole-take quality numbers on
  the parent, and one per-keyframe diagnostic series per child."""
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
  """A world-static tag, addressed by the tag it is. Which one pins the gauge
  is a component on it, never the shape of its name."""
  return f"{WORLD}/anchor/{tag_id}"


def anchor_id_of(path: str) -> int | None:
  """The tag an anchor path addresses, or ``None`` when the path is not one."""
  parts = path.strip("/").split("/")
  if len(parts) != 3 or parts[:2] != [WORLD, "anchor"]:
    return None
  tag = parts[2]
  return int(tag) if tag.isascii() and tag.isdigit() else None


def anchor_ids_in(paths: Iterable[str]) -> list[int]:
  """The world-static tags a recording holds, read off its entity paths."""
  return sorted({tag for tag in map(anchor_id_of, paths) if tag is not None})


def slot_of(path: str) -> str | None:
  """The camera slot an entity path belongs to, or ``None`` when it belongs to
  no camera."""
  parts = path.strip("/").split("/")
  return parts[1] if len(parts) > 1 and parts[0] == CAMERA_ROOT else None


def slots_in(paths: Iterable[str]) -> list[str]:
  """The camera slots a recording holds, read off its entity paths. The
  catalog's schema is a union over segments and cannot answer this per
  segment."""
  return sorted({slot for slot in map(slot_of, paths) if slot})
