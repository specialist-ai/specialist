"""Convert one session's ``.insv`` captures into a base recording.

Usage:
    uv run python scripts/convert.py --input-insvs camA_00.insv camB_00.insv \
        --session-id <session-id>
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

from specialist.loader.audio_sync import sync_two_clips
from specialist.loader.capture import probe_capture
from specialist.loader.parse_insv import load_insv
from specialist.record import schema
from specialist.record.base import (
  calibration_chunks,
  capture_chunk,
  exposure_chunks,
  imu_chunks,
  write_base_recording,
)
from specialist.record.video import normalize_h264, video_chunks


@dataclass
class Args:
  """Base-recording conversion of one session."""

  # One _00_ master .insv per capture; the first defines the session clock.
  input_insvs: tuple[Path, ...]
  # Recording id shared by this session's base recording and its layers.
  session_id: str
  # Dataset root; the base recording lands as ``base/<session-id>.rrd``.
  output_dir: Path = Path("logs")
  # Staging directory for the remuxed MP4s; needs about the session's video size free.
  tmpdir: Path | None = None
  # Seconds to add to each capture's clock, for clips that share no audible event.
  audio_offsets: tuple[float, ...] = ()
  # Replace an existing base recording.
  overwrite: bool = False

  # Catalog selector, stored as a segment property a query filters on.
  task: str = ""
  date: str = ""
  operator: str = ""
  site: str = ""

  # Clamp the session to its first N seconds, for short test conversions.
  duration: float | None = None


def main(args: Args) -> None:
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  out_path = args.output_dir / "base" / f"{args.session_id}.rrd"
  if out_path.exists() and not args.overwrite:
    raise SystemExit(
      f"{out_path} exists; pass --overwrite to replace it (a registered base "
      f"must then be re-registered with register.py --replace-base "
      f"{args.session_id}, or its byte-range index goes stale)"
    )
  captures = [probe_capture(insv) for insv in args.input_insvs]
  if args.audio_offsets:
    if len(args.audio_offsets) != len(args.input_insvs):
      raise SystemExit("--audio-offsets needs one value per --input-insvs")
    offsets = list(args.audio_offsets)
    confidences = [float("nan")] * len(offsets)
    method = "given"
    print(f"audio offsets given: {offsets}")
  else:
    offsets, confidences, method = [0.0], [float("inf")], "audio"
    for later in args.input_insvs[1:]:
      sync = sync_two_clips(args.input_insvs[0], later)
      offsets.append(sync.offset)
      confidences.append(sync.peak_ratio)
      print(
        f"audio sync: {later.name} offset={sync.offset:+.3f}s "
        f"peak_ratio={sync.peak_ratio:.2f}"
      )

  window = (
    max(-off for off in offsets),
    min(cap.duration - off for cap, off in zip(captures, offsets, strict=True)),
  )
  if args.duration is not None:
    window = (window[0], min(window[1], window[0] + args.duration))
  print(f"session window: [{window[0]:.3f}, {window[1]:.3f}] s")

  video_streams = []
  sidecars = []
  loaded = [load_insv(master) for master in args.input_insvs]
  slots = schema.camera_slots(
    [str(insv.capture.get("serial_number", "")) for insv in loaded]
  )
  print(f"camera slots: {dict(zip(slots, args.input_insvs, strict=True))}")
  with tempfile.TemporaryDirectory(dir=args.tmpdir) as tmp:
    for master, capture, insv, slot, offset, confidence in zip(
      args.input_insvs, captures, loaded, slots, offsets, confidences, strict=True
    ):
      for lens, src in (("front", master), ("rear", capture.rear_path)):
        mp4 = normalize_h264(src, Path(tmp) / f"{slot}_{lens}.mp4")
        video_streams.append(
          video_chunks(mp4, schema.video(slot, lens), time_shift=-offset, window=window)
        )
      sidecars += imu_chunks(slot, insv, time_shift=-offset, window=window)
      sidecars += exposure_chunks(slot, insv, time_shift=-offset, window=window)
      sidecars += calibration_chunks(slot, insv)
      sidecars.append(
        capture_chunk(slot, insv, master.name, -offset, confidence, method)
      )

    selectors = {
      "task": args.task,
      "date": args.date,
      "operator": args.operator,
      "site": args.site,
    }
    properties: dict[str, dict[str, Any]] = {
      "base": {key: value for key, value in selectors.items() if value}
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_base_recording(out_path, args.session_id, video_streams, sidecars, properties)
  size_mb = out_path.stat().st_size / 1e6
  print(f"Wrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
  main(tyro.cli(Args))
