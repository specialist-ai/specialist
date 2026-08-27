"""Write a session's state layer from the base recording and the estimate.

The layer holds the state of the session, keyed by hand: proprioceptive, each
hand's pose, twist and finger opening; and object, which only the images
record. Per hand: the rear lens block-averaged to the policy's resolution,
blanked outside the fisheye disc and kept on the keyframe grid inside the cut
window, one PNG per kept row; the base-tag frame's pose and twist in the world
frame; and the coupled finger angle.

A trainer indexes the layer by the integer ``step`` timeline,
``round(t * F_STEP)`` at 100 rows per second. The pose and twist are the
estimate's trajectory averaged onto that lattice, each row stamped at the mean
of its own sample times; the opening is interpolated onto the same rows from
the solved finger angles, never past their span. An image's step identifies
its sample: both hands' frames of one shared session row carry the row's
nominal time rounded, while each camera's true capture time stays on the
session clock untouched. Both hands cover one dense step range, or the writer
refuses. A timestamp is never moved to make two streams look simultaneous:
rates are converted here, once, and every row says the instant it
represents.

Which camera rode which hand is detected from the estimate and written into
the paths, so a trainer's fields are the same for every session; the slot
rides on each hand as a value. The recording property ``state`` records when
the session was captured, solved and written; how the layer was made rides on
the ``state`` entity.

Usage:
    uv run python scripts/state.py --base corpus/base/<session-id>.rrd \
        --output-dir corpus
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr
import tyro
from rerun.experimental import Chunk, LazyChunkStream, OptimizationProfile, send_chunks
from scipy.spatial.transform import Rotation
from specialist_camera import MASK_RADIUS, to_image
from specialist_corpus import schema
from specialist_corpus.frames import LensVideo, decode_rows
from specialist_corpus.read import Recording, read_keyframe_poses
from specialist_corpus.state import finger, hands, image_rows, lattice_mean, opening

from specialist.record.layer import git_sha
from specialist.record.session import SessionRecording, fps_from_times
from specialist.vi_ba.geometry import canonical

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]
State = tuple[Float64Array, Float64Array, Float64Array, Float64Array, Float64Array]

_SIDE: schema.Side = "rear"
F_STEP = 100.0


@dataclass
class Cut:
  """Seconds dropped at each end of the session."""

  head: float = 1.0
  tail: float = 2.0


@dataclass
class Args:
  """One session's state layer."""

  base: Path

  # The session's estimate layer; by default estimate/<session-id>.rrd beside
  # the base.
  estimate: Path | None = None

  # Corpus root; the layer lands as state_s<scale>_i<image-stride>/<session-id>.rrd.
  output_dir: Path = Path("logs")

  # Lens pixels per stored pixel.
  scale: int = 12

  # Capture frames per stored image, a multiple of the estimate's
  # keyframe spacing; that spacing unless given.
  image_stride: int | None = None

  cut: Cut = field(default_factory=Cut)

  # The slot on the operator's left; overrides the geometric detection.
  left: str | None = None

  # Replace an existing layer.
  overwrite: bool = False


def image_blobs(video: LensVideo, rows: Int64Array, scale: int) -> list[bytes]:
  """One PNG per row of ``rows``, ascending."""
  return [
    cv2.imencode(".png", to_image(frame.to_ndarray(format="rgb24"), scale)[:, :, ::-1])[
      1
    ].tobytes()
    for _, frame in decode_rows(video, rows)
  ]


def clock(times: Float64Array) -> list[rr.TimeColumn]:
  """The session clock at the instants a series was measured, and the step each
  row labels: the true time stays beside the rounded index."""
  ns = np.rint(times * schema.NS_PER_SECOND).astype("timedelta64[ns]")
  return [
    rr.TimeColumn(schema.TIMELINE, duration=ns),
    rr.TimeColumn(
      schema.STEP_TIMELINE, sequence=np.rint(times * F_STEP).astype(np.int64)
    ),
  ]


def frame_clock(
  times: Float64Array, rows: Int64Array, fps: float, stride: int
) -> list[rr.TimeColumn]:
  """The camera's true capture times on the session clock, and the labels of
  the shared session ``rows`` the frames sample: the step from each row's
  nominal time, and the keyframe the row is."""
  ns = np.rint(times * schema.NS_PER_SECOND).astype("timedelta64[ns]")
  return [
    rr.TimeColumn(schema.TIMELINE, duration=ns),
    rr.TimeColumn(
      schema.STEP_TIMELINE,
      sequence=np.rint(rows / fps * F_STEP).astype(np.int64),
    ),
    rr.TimeColumn(
      schema.KEYFRAME_TIMELINE, sequence=schema.keyframe_index(rows, stride)
    ),
  ]


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


def hemisphere(quats: Float64Array) -> Float64Array:
  """``quats`` with each sign chosen to stay on its predecessor's hemisphere,
  so a mean over neighbours is a mean of one rotation, not of q and -q."""
  flips = np.where(np.sum(quats[1:] * quats[:-1], axis=1) < 0.0, -1.0, 1.0)
  out: Float64Array = quats * np.concatenate([[1.0], np.cumprod(flips)])[:, None]
  return out


def bin_state(state: State, f_step: float) -> tuple[Int64Array, State]:
  """The state averaged onto the step lattice: ``(steps, state)``, each row
  stamped at the mean of its own sample times."""
  t, position, orientation, lin_vel, ang_vel = state
  stacked = np.concatenate(
    [position, hemisphere(orientation), lin_vel, ang_vel], axis=1
  )
  times, steps, means = lattice_mean(t, stacked, f_step)
  quats = means[:, 3:7]
  quats = canonical(quats / np.linalg.norm(quats, axis=1, keepdims=True))
  return steps, (times, means[:, :3], quats, means[:, 7:10], means[:, 10:13])


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
    hemisphere((world_body * body_tag).as_quat()),
    v_world + np.cross(omega, arm),
    omega,
  )


def inside(times: Float64Array, window: tuple[float, float]) -> npt.NDArray[np.bool_]:
  out: npt.NDArray[np.bool_] = (times >= window[0]) & (times <= window[1])
  return out


def hand_chunks(
  hand: str,
  slot: str,
  frames: tuple[Float64Array, Int64Array],
  fps: float,
  keyframe_stride: int,
  blobs: list[bytes],
  state: State,
  opening_rows: Float64Array,
  width: int,
) -> list[Chunk]:
  """Everything the layer holds for one hand, each series at the times it was
  measured."""
  t, position, orientation, lin_vel, ang_vel = state
  frame_times, frame_rows = frames
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
      indexes=frame_clock(frame_times, frame_rows, fps, keyframe_stride),
      columns=rr.EncodedImage.columns(
        blob=blobs, media_type=["image/png"] * len(blobs)
      ),
    ),
    Chunk.from_columns(
      schema.state_pose(hand),
      indexes=clock(t),
      columns=rr.Transform3D.columns(translation=position, quaternion=orientation),
    ),
    Chunk.from_columns(
      schema.state_lin_vel(hand),
      indexes=clock(t),
      columns=rr.Scalars.columns(scalars=lin_vel),
    ),
    Chunk.from_columns(
      schema.state_ang_vel(hand),
      indexes=clock(t),
      columns=rr.Scalars.columns(scalars=ang_vel),
    ),
    Chunk.from_columns(
      schema.state_opening(hand),
      indexes=clock(t),
      columns=rr.Scalars.columns(scalars=opening_rows[:, None]),
    ),
  ]


def _one(value: Any) -> str:
  return str(np.asarray(value).reshape(-1)[0])


def layer_account(estimate: Recording, started_at: int, layer: str) -> dict[str, Any]:
  """What a catalog sorts and filters a segment by: which configuration the
  layer is, the row rate a trainer indexes by, and when the session was
  captured, solved and written.

  A property is read without opening the segment and ``segment_table()`` has no
  projection pushdown, so every query pays for every property column any layer
  ever wrote. How the layer was made rides on the ``state`` entity instead.
  """
  run = {
    **estimate.row(schema.ESTIMATE),
    **estimate.property(schema.ESTIMATE_PROPERTIES),
  }
  return {
    "config": layer,
    "f_step": F_STEP,
    "started_at": started_at,
    "estimated_at": _one(run["estimated_at"]),
    "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
  }


def layer_settings(
  args: Args, stride: int, estimate: Recording, slot: str
) -> dict[str, Any]:
  """How the layer was made, on the ``state`` entity."""
  run = {
    **estimate.row(schema.ESTIMATE),
    **estimate.property(schema.ESTIMATE_PROPERTIES),
  }
  return {
    "scale": args.scale,
    "image_stride": stride,
    "mask_radius": MASK_RADIUS,
    "side": _SIDE,
    "cut_head": args.cut.head,
    "cut_tail": args.cut.tail,
    "git_sha": git_sha(),
    "rerun_version": rr.__version__,
    "estimate_git_sha": _one(run["git_sha"]),
    "urdf_sha": _one(estimate.row(schema.gripper(slot)).get("urdf_sha", ["unknown"])),
  }


def write_layer(
  path: Path,
  session_id: str,
  account: dict[str, Any],
  settings: dict[str, Any],
  chunks: list[Chunk],
) -> None:
  """Write the layer under the session's id."""
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
      rr.log(schema.STATE, rr.AnyValues(**settings), static=True, recording=stream)
      send_chunks(store, recording=stream)
    partial.replace(path)
  finally:
    partial.unlink(missing_ok=True)


def main(args: Args) -> None:
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  if args.image_stride is not None and args.image_stride < 1:
    raise SystemExit(f"--image-stride must be at least 1, got {args.image_stride}")
  session = SessionRecording(args.base)
  slots = session.slots
  if len(slots) != 2:
    raise SystemExit(f"the state layer takes a two-camera session, got {slots}")
  estimate_path = args.estimate or schema.layer_rrd(
    args.base.parent.parent, schema.ESTIMATE, session.session_id
  )
  poses = read_keyframe_poses(estimate_path)
  if missing := [s for s in slots if s not in poses.kf_times_by_slot]:
    raise SystemExit(f"{estimate_path} holds no keyframe poses for {missing}")
  keyframes = {s: poses.kf_times_by_slot[s] for s in slots}

  times = {slot: session.lens_times(slot, _SIDE) for slot in slots}
  fps = fps_from_times(times[slots[0]])
  shapes = {slot: session.decoded_shape(slot, _SIDE) for slot in slots}
  for slot, (height, width) in shapes.items():
    if width % args.scale or height % args.scale:
      raise SystemExit(
        f"--scale {args.scale} does not divide {slot}'s {width}x{height} frames"
      )
  # The hands sit side by side one frame apart, so the two must be one size,
  # and the disc is a circle only on a square one.
  if len(set(shapes.values())) != 1:
    raise SystemExit(f"the lenses decode to different frames: {shapes}")
  if shapes[slots[0]][0] != shapes[slots[0]][1]:
    raise SystemExit(f"the rear lens does not decode square frames: {shapes}")
  grid = np.rint(keyframes[slots[0]] * fps).astype(np.int64)
  for k in keyframes.values():
    grid = np.intersect1d(grid, np.rint(k * fps).astype(np.int64))
  spacing = round(float(np.median(np.diff(grid))))
  stride = args.image_stride or spacing
  if stride % spacing:
    raise SystemExit(
      f"the estimate keyframes every {spacing} frames; --image-stride {stride} "
      "is not a multiple of that"
    )
  layer = schema.state_layer(args.scale, stride)
  out_path = schema.layer_rrd(args.output_dir, layer, session.session_id)
  if out_path.exists() and not args.overwrite:
    raise SystemExit(f"{out_path} exists; pass --overwrite to replace it")

  estimate = Recording(estimate_path)
  states: dict[str, tuple[Int64Array, State]] = {}
  fingers: dict[str, tuple[Any, Any]] = {}
  for slot in slots:
    try:
      series = [
        estimate.values(schema.gripper_part(slot, f"{side}_finger"), "angle", 1)
        for side in ("left", "right")
      ]
      fingers[slot] = (
        finger(series[0][0], series[0][1][:, 0]),
        finger(series[1][0], series[1][1][:, 0]),
      )
    except ValueError as exc:
      raise SystemExit(f"{slot}: {exc}") from exc
    states[slot] = bin_state(gripper_state(estimate, slot), F_STEP)

  # The cut trims the span every lens covers; the keyframes and the solved
  # fingers bound it further, so the opening covers every kept row.
  window = (
    max(
      max(float(t[0]) for t in times.values()) + args.cut.head,
      max(k[0] for k in keyframes.values()),
      max(float(t[0]) for pair in fingers.values() for t, _ in pair),
    ),
    min(
      min(float(t[-1]) for t in times.values()) - args.cut.tail,
      min(k[-1] for k in keyframes.values()),
      min(float(t[-1]) for pair in fingers.values() for t, _ in pair),
    ),
  )
  try:
    rows = image_rows(times, fps, stride, window, grid)
  except ValueError as exc:
    raise SystemExit(str(exc)) from exc
  drift = float(
    np.ptp(times[slots[0]][rows[slots[0]]] - times[slots[1]][rows[slots[1]]])
  )
  if drift > 0.5 / F_STEP:
    raise SystemExit(
      f"the cameras' clocks drift {drift * 1e3:.1f} ms across the take, more "
      "than half a step; a shared row index would lie"
    )
  if args.left is None:
    try:
      left, right = hands(
        keyframes, {s: poses.kf_translations_by_slot[s] for s in slots}
      )
    except ValueError as exc:
      raise SystemExit(f"{exc}; --left <slot> says which is left") from exc
  elif args.left in slots:
    left, right = args.left, next(s for s in slots if s != args.left)
  else:
    raise SystemExit(f"--left must name one of {slots}, got {args.left}")
  print(
    f"session {session.session_id}: {layer}, window [{window[0]:.2f}, "
    f"{window[1]:.2f}] s, left {left}, right {right}"
  )

  account = layer_account(estimate, session.started_at, layer)
  settings = layer_settings(args, stride, estimate, left)
  stored_width = shapes[left][1] // args.scale
  keyframe_stride = schema.keyframe_stride(fps)

  # Both hands cover one step range: a trainer's row i means one instant.
  firsts, lasts = [], []
  for slot in slots:
    held = states[slot][0][inside(states[slot][1][0], window)]
    if not held.size:
      raise SystemExit(f"{slot}: no state row falls in the window {window} s")
    firsts.append(int(held[0]))
    lasts.append(int(held[-1]))
  lo, hi = max(firsts), min(lasts)

  chunks: list[Chunk] = []
  for hand, slot in zip(schema.HANDS, (left, right), strict=True):
    started = time.time()
    video = session.lens_video(slot, _SIDE)
    if not np.array_equal(video.times, times[slot]):
      raise RuntimeError(f"{slot}: the rear lens's rows changed between two reads")
    frame_times = times[slot][rows[slot]]
    shared_rows = np.rint(frame_times * fps).astype(np.int64)
    blobs = image_blobs(video, rows[slot], args.scale)
    steps, (t, position, orientation, lin_vel, ang_vel) = states[slot]
    sel = (steps >= lo) & (steps <= hi)
    if not np.array_equal(steps[sel], np.arange(lo, hi + 1)):
      holes = np.setdiff1d(np.arange(lo, hi + 1), steps[sel])
      raise SystemExit(
        f"{slot}: the state stream misses steps {holes[:6].tolist()} of "
        f"[{lo}, {hi}]; the trajectory has holes"
      )
    state = (t[sel], position[sel], orientation[sel], lin_vel[sel], ang_vel[sel])
    try:
      opened = opening(*fingers[slot], t[sel])
    except ValueError as exc:
      raise SystemExit(f"{slot}: {exc}") from exc
    chunks += hand_chunks(
      hand,
      slot,
      (frame_times, shared_rows),
      fps,
      keyframe_stride,
      blobs,
      state,
      opened,
      stored_width,
    )
    print(
      f"  {hand} ({slot}): {len(blobs)} images, {sum(map(len, blobs)) / 1e6:.1f} MB, "
      f"steps [{lo}, {hi}], {time.time() - started:.0f} s"
    )
  write_layer(out_path, session.session_id, account, settings, chunks)
  print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main(tyro.cli(Args))
