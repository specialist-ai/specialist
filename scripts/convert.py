"""Convert one captured session's raw data into a base recording rrd.

One capture = one ``_00_`` master ``.insv`` (its ``_10_`` sibling is found
automatically). The first capture's clock is the session clock; later
captures are audio-synced onto it and their offsets recorded as properties.
The video streams are remuxed losslessly (never re-encoded) and land with
their real per-frame PTS; IMU keeps native per-sample timestamps.

Usage:
    uv run python scripts/convert.py --input-insvs camA_00.insv camB_00.insv \
        --session-id <session-id>
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from specialist.loader.audio_sync import sync_two_clips
from specialist.loader.parse_insv import load_insv
from specialist.record import schema
from specialist.record.base import (
  calibration_chunks,
  exposure_chunks,
  imu_chunks,
  write_base_recording,
)
from specialist.record.video import normalize_h264, video_chunks
from specialist.vi_ba.configs import ImuCfg
from specialist.vi_ba.pipeline import capture_duration


@dataclass
class Args:
  """Base-recording conversion of one session."""

  # One _00_ master .insv per capture; the first defines the session clock.
  input_insvs: tuple[Path, ...] = ()
  # Recording id shared by this session's base recording and track layers.
  session_id: str = ""
  # One name per capture; must match the solve's --tracker-names for the
  # track layer's entities to land on this base recording's.
  tracker_names: tuple[str, ...] = ()
  output_dir: Path = Path("logs")
  # Where the remuxed per-lens MP4s stage; needs roughly the session's video
  # size free until the write finishes.
  tmpdir: Path | None = None
  # Seconds to add to each capture's clock, one per input, for a take whose
  # clips share no audible event.
  audio_offsets: tuple[float, ...] = ()
  imu: ImuCfg = field(default_factory=ImuCfg)


def main(args: Args) -> None:
  if not args.input_insvs or not args.session_id:
    raise SystemExit("--input-insvs and --session-id are required")
  names = (
    list(args.tracker_names)
    if args.tracker_names
    else [f"tracker_{i}" for i in range(len(args.input_insvs))]
  )
  if len(names) != len(args.input_insvs):
    raise SystemExit("--tracker-names must match --input-insvs in length")
  rears = [p.with_name(p.name.replace("_00_", "_10_", 1)) for p in args.input_insvs]
  if args.audio_offsets:
    if len(args.audio_offsets) != len(args.input_insvs):
      raise SystemExit("--audio-offsets needs one value per --input-insvs")
    offsets = list(args.audio_offsets)
    print(f"audio offsets given: {offsets}")
  else:
    offsets = [0.0]
    for later in args.input_insvs[1:]:
      sync = sync_two_clips(args.input_insvs[0], later)
      offsets.append(sync.offset)
      print(
        f"audio sync: {later.name} offset={sync.offset:+.3f}s "
        f"peak_ratio={sync.peak_ratio:.2f}"
      )

  # The session window: from the latest capture's start to the earliest end,
  # so the record holds only the span every capture covers, both lenses
  # included (an interrupted rear file can be shorter than its master).
  durations = [
    min(capture_duration(master), capture_duration(rear))
    for master, rear in zip(args.input_insvs, rears, strict=True)
  ]
  window = (
    max(-off for off in offsets),
    min(dur - off for dur, off in zip(durations, offsets, strict=True)),
  )
  print(f"session window: [{window[0]:.3f}, {window[1]:.3f}] s")

  video_streams = []
  sidecars = []
  imu_to_body = np.asarray(args.imu.imu_to_body, dtype=np.float64)
  with tempfile.TemporaryDirectory(dir=args.tmpdir) as tmp:
    for master, rear, tracker, offset in zip(
      args.input_insvs, rears, names, offsets, strict=True
    ):
      insv = load_insv(master)
      for lens, src in (("front", master), ("rear", rear)):
        mp4 = normalize_h264(src, Path(tmp) / f"{tracker}_{lens}.mp4")
        video_streams.append(
          video_chunks(
            mp4,
            schema.image_entity(tracker, lens),
            schema.TIMELINE,
            time_shift=-offset,
            window=window,
          )
        )
      sidecars += imu_chunks(tracker, insv, time_shift=-offset, window=window)
      sidecars += exposure_chunks(tracker, insv, time_shift=-offset, window=window)
      sidecars += calibration_chunks(tracker, insv, imu_to_body)

    properties: dict[str, dict[str, Any]] = {
      "session": {
        "session_id": args.session_id,
        "sources": [p.name for p in args.input_insvs],
      },
      "audio_offset": dict(zip(names, offsets, strict=True)),
    }
    out_path = args.output_dir / "base" / f"{args.session_id}.rrd"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_base_recording(out_path, args.session_id, video_streams, sidecars, properties)
  size_mb = out_path.stat().st_size / 1e6
  print(f"Wrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
  main(tyro.cli(Args))
