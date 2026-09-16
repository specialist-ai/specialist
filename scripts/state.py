"""Write a session's state layer from the base recording and the estimate.

The layer holds the state of the session, keyed by hand: proprioceptive, each
hand's pose, twist and finger opening; and object, which only the images
record. Per hand: the rear lens block-averaged to the policy's resolution,
blanked outside the fisheye disc, every frame on a state row inside the
usable window, one PNG each; the base-tag frame's pose and twist in the world
frame; and the coupled finger angle.

The usable window is read from the operator's bracket gestures: it opens one
margin after the later hand's start hold at a wall anchor ends, and closes
the margin before the fingertips first strike the desk. A hold ending past
the capture's first tenth or a strike before its last is refused.

A trainer indexes the layer by the integer ``row`` timeline,
``round(t * f_row)``, whose rate is the one a policy's loop will run at.
A row holds the estimate's trajectory at its own instant, from the two
samples that bracket it: position and twist linearly, the rotation along the
arc between them. A bracket wider than a dropout's width refuses. The
opening is interpolated onto the same rows from the solved finger angles,
never past their span. An image's row identifies its sample: both hands'
frames of one shared session frame carry that frame's nominal time rounded,
while each frame's own capture row rides beside it as a value and its true
capture time stays on the session clock untouched. Both hands cover one
dense row range.

Which camera rode which hand is detected from the estimate and written into
the paths, so a trainer's fields are the same for every session; the slot
rides on each hand as a value. The recording property ``state`` carries the
layer's whole per-session account, read without opening the segment; only
per-hand facts ride entities.

Usage:
    uv run python scripts/state.py --base corpus/base/<session-id>.rrd \
        --output-dir corpus
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr
import tyro
from rerun.experimental import Chunk, LazyChunkStream, OptimizationProfile, send_chunks
from scipy.spatial.transform import Rotation, Slerp
from specialist_camera import MASK_RADIUS, to_image
from specialist_corpus import schema
from specialist_corpus.frames import LensVideo, decode_rows
from specialist_corpus.read import Recording, read_anchors, read_keyframe_poses
from specialist_corpus.state import (
  finger,
  hands,
  hold,
  image_frames,
  lattice_brackets,
  lerp,
  opening,
  tap,
)

from specialist.record.layer import git_sha
from specialist.record.session import SessionRecording, fps_from_times
from specialist.vi_ba.geometry import canonical

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]
State = tuple[Float64Array, Float64Array, Float64Array, Float64Array, Float64Array]

_SIDE: schema.Side = "rear"


@dataclass
class Args:
  """One session's state layer."""

  base: Path

  # The session's estimate layer; by default estimate/<session-id>.rrd beside
  # the base.
  estimate: Path | None = None

  # Corpus root; the layer lands as state_s<scale>_r<f-row>/<session-id>.rrd.
  output_dir: Path = Path("logs")

  # Lens pixels per stored pixel.
  scale: int = 12

  # Rows per second: the rate a policy's loop runs at, which every rate it
  # reads must divide and no target's feedback may fall below.
  f_row: float = 100.0

  # Seconds kept clear of the bracket gestures at each end of the window.
  margin: float = 1.0

  # The slot on the operator's left; overrides the geometric detection.
  left: str | None = None

  # Replace an existing layer.
  overwrite: bool = False


@contextmanager
def refusing(where: str = "") -> Iterator[None]:
  """A corpus the writer cannot use stops the run with its own message."""
  try:
    yield
  except ValueError as exc:
    raise SystemExit(f"{where}: {exc}" if where else str(exc)) from exc


def finger_pair(estimate: Recording, slot: str) -> tuple[Any, Any]:
  """Both fingers' solved angle series."""
  left, right = (
    estimate.values(schema.gripper_part(slot, f"{side}_finger"), "angle", 1)
    for side in ("left", "right")
  )
  return finger(left[0], left[1][:, 0]), finger(right[0], right[1][:, 0])


def stored_width(session: SessionRecording, slots: list[str], scale: int) -> int:
  """The width one hand's frames occupy once scaled. Both lenses decode the
  same square frame: one image plane holds both hands side by side, and the
  fisheye disc is a circle only on a square."""
  shapes = {slot: session.decoded_shape(slot, _SIDE) for slot in slots}
  for slot, (height, width) in shapes.items():
    if width % scale or height % scale:
      raise SystemExit(
        f"--scale {scale} does not divide {slot}'s {width}x{height} frames"
      )
  if len(set(shapes.values())) != 1:
    raise SystemExit(f"the lenses decode to different frames, {shapes}")
  height, width = shapes[slots[0]]
  if height != width:
    raise SystemExit(f"the rear lens decodes {height}x{width} frames, not a square")
  return width // scale


def hand_slots(
  chosen: str | None,
  slots: list[str],
  keyframes: dict[str, Any],
  places: dict[str, Any],
) -> tuple[str | None, str]:
  """Which slot rode the operator's left hand, and which the right. A lone
  camera rode the right."""
  if len(slots) == 1:
    return None, slots[0]
  if chosen is None:
    try:
      return hands(keyframes, places)
    except ValueError as exc:
      raise SystemExit(f"{exc}; --left <slot> says which is left") from exc
  if chosen not in slots:
    raise SystemExit(f"--left must name one of {slots}, got {chosen}")
  return chosen, next(s for s in slots if s != chosen)


def image_blobs(video: LensVideo, frames: Int64Array, scale: int) -> list[bytes]:
  """One PNG per frame of ``frames``, ascending."""
  return [
    cv2.imencode(".png", to_image(frame.to_ndarray(format="rgb24"), scale)[:, :, ::-1])[
      1
    ].tobytes()
    for _, frame in decode_rows(video, frames)
  ]


def clock(times: Float64Array, rows: Int64Array) -> list[rr.TimeColumn]:
  """The session clock at the instants a series was measured, and the row each
  is filed under: the true time stays beside the index."""
  ns = np.rint(times * schema.NS_PER_SECOND).astype("timedelta64[ns]")
  return [
    rr.TimeColumn(schema.TIMELINE, duration=ns),
    rr.TimeColumn(schema.ROW_TIMELINE, sequence=rows),
  ]


def frame_rows(
  times: Float64Array, fps: float, f_row: float
) -> tuple[Int64Array, Int64Array]:
  """Each frame's row label, its shared session frame's nominal time rounded,
  and its capture row, its own shutter time rounded."""
  nominal = np.rint(times * fps) / fps
  return (
    np.rint(nominal * f_row).astype(np.int64),
    np.rint(times * f_row).astype(np.int64),
  )


def angular_velocity(times: Float64Array, rotations: Rotation) -> Float64Array:
  """The world-frame rotation rate of a trajectory, by central difference over
  its own rows; the two ends take the one-sided difference."""
  if len(times) < 3:
    raise ValueError(f"a rotation rate needs three rows, got {len(times)}")
  span = (rotations[2:] * rotations[:-2].inv()).as_rotvec()
  out: Float64Array = np.empty((len(times), 3))
  out[1:-1] = span / (times[2:] - times[:-2])[:, None]
  out[0] = (rotations[1] * rotations[0].inv()).as_rotvec() / (times[1] - times[0])
  out[-1] = (rotations[-1] * rotations[-2].inv()).as_rotvec() / (times[-1] - times[-2])
  return out


def lattice_state(
  state: State, f_row: float, window: tuple[float, float]
) -> tuple[Int64Array, State]:
  """The state at each row's instant inside ``window``, from the two samples
  that bracket it: position and twist linearly, the rotation along the arc
  between them, stored with ``w >= 0``."""
  t, position, orientation, lin_vel, ang_vel = state
  rows, before, weight = lattice_brackets(t, f_row, window)
  instants = rows / f_row
  turn = Slerp(t, Rotation.from_quat(orientation))(instants)
  return rows, (
    instants,
    lerp(position, before, weight),
    canonical(turn.as_quat()),
    lerp(lin_vel, before, weight),
    lerp(ang_vel, before, weight),
  )


def gripper_state(estimate: Recording, slot: str) -> State:
  """The base-tag frame's pose and twist in the world frame, on the rows the
  estimate solved them at: ``(times, position, orientation, lin_vel, ang_vel)``.
  The trajectory and its velocity are composed with the tag's mounting, and the
  rotation rate is the trajectory's own, so the twist and the pose describe one
  motion."""
  body = schema.camera(slot)
  t, translation = estimate.values(body, schema.POSE_TRANSLATION, 3)
  t_rot, quaternion = estimate.values(body, schema.POSE_QUATERNION, 4)
  t_vel, v_world = estimate.values(schema.lin_vel(slot), "lin_vel", 3)
  if not (np.array_equal(t, t_rot) and np.array_equal(t, t_vel)):
    raise SystemExit(
      f"{slot}: the trajectory's translation, rotation and velocity sit on "
      "different rows"
    )
  tag = estimate.row(schema.base_tag(slot))
  if not tag:
    raise SystemExit(f"{estimate.path} holds no base tag for {slot}")
  world_body = Rotation.from_quat(quaternion)
  body_tag = Rotation.from_quat(np.asarray(tag[schema.POSE_QUATERNION]).reshape(4))
  arm = world_body.apply(np.asarray(tag[schema.POSE_TRANSLATION]).reshape(3))
  omega = angular_velocity(t, world_body)
  return (
    t,
    translation + arm,
    (world_body * body_tag).as_quat(),
    v_world + np.cross(omega, arm),
    omega,
  )


def usable_window(
  hold_ends: list[float],
  strike: float,
  margin: float,
  times: dict[str, Float64Array],
  keyframes: dict[str, Float64Array],
  fingers: dict[str, tuple[Any, Any]],
) -> tuple[float, float]:
  """The span the layer keeps: one margin after the later hold and one before
  the strike, within what every lens, every keyframe span and every solved
  finger covers, so every kept row has a frame of each lens and an opening."""
  return (
    max(
      max(hold_ends) + margin,
      max(float(t[0]) for t in times.values()),
      max(k[0] for k in keyframes.values()),
      max(float(t[0]) for pair in fingers.values() for t, _ in pair),
    ),
    min(
      strike - margin,
      min(float(t[-1]) for t in times.values()),
      min(k[-1] for k in keyframes.values()),
      min(float(t[-1]) for pair in fingers.values() for t, _ in pair),
    ),
  )


def frames_on_rows(
  times: dict[str, Float64Array],
  frames: dict[str, Int64Array],
  fps: float,
  f_row: float,
  lo: int,
  hi: int,
) -> npt.NDArray[np.bool_]:
  """Which shared frames have a state row at their label and at every lens's
  capture row."""
  kept = np.ones(len(next(iter(frames.values()))), dtype=bool)
  for slot, picked in frames.items():
    label, capture = frame_rows(times[slot][picked], fps, f_row)
    kept &= (label >= lo) & (label <= hi) & (capture >= lo) & (capture <= hi)
  return kept


def hand_chunks(
  hand: str,
  slot: str,
  frames: tuple[Float64Array, Int64Array, Int64Array],
  blobs: list[bytes],
  rows: Int64Array,
  state: State,
  opening_rows: Float64Array,
  width: int,
) -> list[Chunk]:
  """Everything the layer holds for one hand, each series at the times it was
  measured."""
  t, position, orientation, lin_vel, ang_vel = state
  frame_times, label, capture = frames
  # One image plane holds both hands, left then right.
  column = float(schema.HANDS.index(hand) * width)
  return [
    Chunk.from_columns(
      schema.state_hand(hand), indexes=[], columns=rr.AnyValues.columns(slot=[slot])
    ),
    Chunk.from_columns(
      schema.state_image(hand),
      indexes=[],
      columns=rr.Transform3D.columns(translation=[[column, 0.0, 0.0]]),
    ),
    Chunk.from_columns(
      schema.state_image(hand),
      indexes=clock(frame_times, label),
      columns=[
        *rr.EncodedImage.columns(blob=blobs, media_type=["image/png"] * len(blobs)),
        *rr.AnyValues.columns(capture_row=capture),
      ],
    ),
    Chunk.from_columns(
      schema.state_pose(hand),
      indexes=clock(t, rows),
      columns=rr.Transform3D.columns(translation=position, quaternion=orientation),
    ),
    Chunk.from_columns(
      schema.state_lin_vel(hand),
      indexes=clock(t, rows),
      columns=rr.Scalars.columns(scalars=lin_vel),
    ),
    Chunk.from_columns(
      schema.state_ang_vel(hand),
      indexes=clock(t, rows),
      columns=rr.Scalars.columns(scalars=ang_vel),
    ),
    Chunk.from_columns(
      schema.state_opening(hand),
      indexes=clock(t, rows),
      columns=rr.Scalars.columns(scalars=opening_rows[:, None]),
    ),
  ]


def _one(value: Any) -> str:
  return str(np.asarray(value).reshape(-1)[0])


def layer_account(
  args: Args,
  estimate: Recording,
  slot: str,
  converted_at: int,
  layer: str,
  window: tuple[float, float],
  covered: float,
) -> dict[str, Any]:
  """The layer's per-session account, as the ``state`` recording property:
  which configuration it is, how it was made, when the session was converted,
  solved and written, and how much of the capture the usable window keeps.
  Per-hand facts ride entities."""
  run = estimate.property(schema.ESTIMATE_PROPERTIES)
  return {
    "config": layer,
    "f_row": args.f_row,
    "scale": args.scale,
    "mask_radius": MASK_RADIUS,
    "side": _SIDE,
    "margin": args.margin,
    "duration": window[1] - window[0],
    "kept": (window[1] - window[0]) / covered,
    "git_sha": git_sha(),
    "rerun_version": rr.__version__,
    "estimate_git_sha": _one(run["git_sha"]),
    "urdf_sha": _one(estimate.row(schema.gripper(slot)).get("urdf_sha", ["unknown"])),
    "converted_at": converted_at,
    "estimated_at": _one(run["estimated_at"]),
    "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
  }


def write_layer(
  path: Path,
  session_id: str,
  account: dict[str, Any],
  chunks: list[Chunk],
) -> None:
  """Write the layer under the session's id, through a partial file that only a
  complete write renames into place."""
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.parent / (path.name + ".partial")
  store = LazyChunkStream.from_iter(chunks).collect(
    optimize=OptimizationProfile.OBJECT_STORE
  )
  try:
    with rr.RecordingStream(
      schema.APPLICATION_ID, recording_id=session_id, send_properties=False
    ) as stream:
      stream.set_sinks(rr.FileSink(partial))
      stream.send_property(schema.STATE, rr.AnyValues(**account))
      send_chunks(store, recording=stream)
    partial.replace(path)
  finally:
    partial.unlink(missing_ok=True)


def main(args: Args) -> None:
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  session = SessionRecording(args.base)
  slots = session.slots
  if not 1 <= len(slots) <= 2:
    raise SystemExit(f"the state layer takes one or two cameras, got {slots}")
  estimate_path = args.estimate or schema.layer_rrd(
    args.base.parent.parent, schema.ESTIMATE, session.session_id
  )
  poses = read_keyframe_poses(estimate_path)
  if missing := [s for s in slots if s not in poses.kf_times_by_slot]:
    raise SystemExit(f"{estimate_path} holds no keyframe poses for {missing}")
  keyframes = {s: poses.kf_times_by_slot[s] for s in slots}

  times = {slot: session.lens_times(slot, _SIDE) for slot in slots}
  fps = fps_from_times(times[slots[0]])
  width = stored_width(session, slots, args.scale)
  layer = schema.state_layer(args.scale, args.f_row)
  out_path = schema.layer_rrd(args.output_dir, layer, session.session_id)
  if out_path.exists() and not args.overwrite:
    raise SystemExit(f"{out_path} exists; pass --overwrite to replace it")

  estimate = Recording(estimate_path)
  anchors = np.stack(
    [pose[:3, 3] for pose in read_anchors(estimate_path).poses.values()]
  )
  imus = {slot: session.imu(slot) for slot in slots}
  span = (
    min(float(imu.t[0]) for imu in imus.values()),
    max(float(imu.t[-1]) for imu in imus.values()),
  )
  grippers: dict[str, State] = {}
  fingers: dict[str, tuple[Any, Any]] = {}
  hold_ends: list[float] = []
  for slot in slots:
    with refusing(slot):
      fingers[slot] = finger_pair(estimate, slot)
      gripper = grippers[slot] = gripper_state(estimate, slot)
      distance = np.linalg.norm(gripper[1][:, None] - anchors[None], axis=2).min(axis=1)
      hold_ends.append(
        hold(gripper[0], distance, np.linalg.norm(gripper[3], axis=1), span)
      )
  with refusing():
    strike = tap(
      {slot: imu.t for slot, imu in imus.items()},
      {slot: imu.accel for slot, imu in imus.items()},
    )

  window = usable_window(hold_ends, strike, args.margin, times, keyframes, fingers)
  with refusing():
    frames = image_frames(times, fps, window)
  left, right = hand_slots(
    args.left, slots, keyframes, {s: poses.kf_translations_by_slot[s] for s in slots}
  )
  named = f"right {right}" if left is None else f"left {left}, right {right}"
  print(
    f"session {session.session_id}: {layer}, window [{window[0]:.2f}, "
    f"{window[1]:.2f}] s, {named}"
  )

  covered = min(float(t[-1]) for t in times.values()) - max(
    float(t[0]) for t in times.values()
  )
  account = layer_account(
    args, estimate, right, session.converted_at, layer, window, covered
  )

  states: dict[str, tuple[Int64Array, State]] = {}
  for slot in slots:
    with refusing(slot):
      states[slot] = lattice_state(grippers[slot], args.f_row, window)
  lo = max(int(rows[0]) for rows, _ in states.values())
  hi = min(int(rows[-1]) for rows, _ in states.values())
  kept = frames_on_rows(times, frames, fps, args.f_row, lo, hi)
  frames = {slot: picked[kept] for slot, picked in frames.items()}

  chunks: list[Chunk] = []
  rode = [(h, c) for h, c in zip(schema.HANDS, (left, right), strict=True) if c]
  for hand, slot in rode:
    started = time.time()
    video = session.lens_video(slot, _SIDE)
    if not np.array_equal(video.times, times[slot]):
      raise RuntimeError(
        f"{slot}: the rear lens's frame times changed between two reads"
      )
    frame_times = times[slot][frames[slot]]
    blobs = image_blobs(video, frames[slot], args.scale)
    all_rows, (t, position, orientation, lin_vel, ang_vel) = states[slot]
    held = (all_rows >= lo) & (all_rows <= hi)
    state = (t[held], position[held], orientation[held], lin_vel[held], ang_vel[held])
    with refusing(slot):
      opened = opening(*fingers[slot], state[0])
    chunks += hand_chunks(
      hand,
      slot,
      (frame_times, *frame_rows(frame_times, fps, args.f_row)),
      blobs,
      all_rows[held],
      state,
      opened,
      width,
    )
    print(
      f"  {hand} ({slot}): {len(blobs)} images, {sum(map(len, blobs)) / 1e6:.1f} MB, "
      f"rows [{lo}, {hi}], {time.time() - started:.0f} s"
    )
  write_layer(out_path, session.session_id, account, chunks)
  print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main(tyro.cli(Args))
