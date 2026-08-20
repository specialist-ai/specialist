"""Frame decoding and the ``.rrd`` + JSON summary tail the probe scripts share."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr

from specialist.vi_ba.pipeline import probe_capture

UInt8Array = npt.NDArray[np.uint8]


def read_lens_grays(
  video_path: Path, lens: str, *, max_frames: int
) -> tuple[list[UInt8Array], float]:
  """Grey frames from one lens, with the container frame rate.

  ``video_path`` is the ``_00_`` master; ``lens`` picks the master (``front``)
  or its ``_10_`` sibling (``rear``).
  """
  if lens not in ("front", "rear"):
    raise ValueError(f"lens must be 'front' or 'rear', got {lens!r}")
  rear_path, fps, _ = probe_capture(video_path)
  path = video_path if lens == "front" else rear_path
  cap = cv2.VideoCapture(str(path))
  if not cap.isOpened():
    raise RuntimeError(f"OpenCV failed to open {path}")
  try:
    grays: list[UInt8Array] = []
    while len(grays) < max_frames:
      ok, frame = cap.read()
      if not ok:
        break
      grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8))
  finally:
    cap.release()
  return grays, fps


def verdict(problems: Sequence[str]) -> str:
  return "green" if not problems else "yellow: " + "; ".join(problems)


def finalize_probe(
  name: str,
  output_dir: Path,
  summary: Any,
  log_fn: Callable[[], None],
) -> None:
  """Save one probe's ``.rrd`` + JSON summary under ``name``.

  ``log_fn`` streams this probe's entities while the recording is open.
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  stamp = time.strftime("%Y%m%d_%H%M%S")
  rrd_path = output_dir / f"{name}_{stamp}.rrd"
  json_path = output_dir / f"{name}_{stamp}.json"

  rr.init(f"specialist-{name}", spawn=False)
  rr.save(str(rrd_path))
  log_fn()
  rr.disconnect()

  payload = json.dumps(asdict(summary), indent=2)
  json_path.write_text(payload)
  print(payload)
  print(f"\nWrote {rrd_path} and {json_path}")
  print(f"Verdict: {summary.verdict}")
