"""Write a session's state layer from its base recording and estimate layer.

Usage:
    uv run python scripts/state.py --base dataset/base/<session-id>.rrd \
        --output-dir dataset
"""

from __future__ import annotations

import sys
import tempfile
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
from specialist_dataset import schema
from specialist_dataset.frames import LensVideo, decode_rows
from specialist_dataset.read import Recording, read_anchors, read_keyframe_poses
from specialist_dataset.state import (
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
# (times, position, xyzw orientation, lin_vel, ang_vel), in the world frame.
State = tuple[Float64Array, Float64Array, Float64Array, Float64Array, Float64Array]

_SIDE: schema.Side = "rear"


@dataclass
class Args:
  """One session's state layer."""

  base: Path

  # The estimate layer; defaults to estimate/<session-id>.rrd in the base's dataset.
  estimate: Path | None = None

  # Dataset root; the layer lands as state_s<scale>_r<f-row>/<session-id>.rrd.
  output_dir: Path = Path("logs")

  # Lens pixels per stored pixel.
  scale: int = 12

  # Rows per second: the rate a policy's loop runs at.
  f_row: float = 100.0

  # Seconds kept clear of the bracket gestures at each end of the window.
  margin: float = 1.0

  # The slot on the operator's left; overrides the geometric detection.
  left: str | None = None

  # Replace an existing layer.
  overwrite: bool = False


@contextmanager
def refusing(where: str = "") -> Iterator[None]:
  """Exit with any ``ValueError``'s message, prefixed by ``where``."""
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
  """Scaled frame width per hand; refuses unequal, non-square or indivisible frames."""
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
  """The slots on the ``(left, right)`` hands; a lone camera rode the right."""
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
  """The session-clock and ``row`` index columns of a series measured at ``times``."""
  ns = np.rint(times * schema.NS_PER_SECOND).astype("timedelta64[ns]")
  return [
    rr.TimeColumn(schema.TIMELINE, duration=ns),
    rr.TimeColumn(schema.ROW_TIMELINE, sequence=rows),
  ]


def frame_rows(
  times: Float64Array, fps: float, f_row: float
) -> tuple[Int64Array, Int64Array]:
  """Each frame's label row (nominal session-frame time) and capture row (own time)."""
  nominal = np.rint(times * fps) / fps
  return (
    np.rint(nominal * f_row).astype(np.int64),
    np.rint(times * f_row).astype(np.int64),
  )


def angular_velocity(times: Float64Array, rotations: Rotation) -> Float64Array:
  """The world-frame rotation rate by central difference, one-sided at the two ends."""
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
  """Rows in ``window`` and the state there: rotation slerped, the rest lerped."""
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
  """The base-tag frame's world pose and twist on the estimate's rows as a ``State``."""
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
  """The kept span, a margin inside the hold and strike and covered by every series."""
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
  """The shared frames whose label and capture rows all fall in ``[lo, hi]``."""
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
  """Every chunk the layer holds for one hand, each series at its measured times."""
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
  """The ``state`` recording property: config, provenance, stamps and kept fraction."""
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
  """Write the layer under the session's id, renaming a partial file once complete."""
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
  args.output_dir.mkdir(parents=True, exist_ok=True)
  stage = tempfile.TemporaryDirectory(prefix="lenses-", dir=args.output_dir)
  session = SessionRecording(args.base, Path(stage.name))
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
  session.close()
  stage.cleanup()
  print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main(tyro.cli(Args))
