"""Frame decoding and the ``.rrd`` + JSON summary tail the probe scripts share."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
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
  video_path: Path, lens: str, *, max_frames: int | None = None
) -> list[UInt8Array]:
  """Grey frames of ``video_path`` or, for ``rear``, its ``_10_`` sibling."""
  if lens not in ("front", "rear"):
    raise ValueError(f"lens must be 'front' or 'rear', got {lens!r}")
  rear_path, _, _ = probe_capture(video_path)
  path = video_path if lens == "front" else rear_path
  cap = cv2.VideoCapture(str(path))
  if not cap.isOpened():
    raise RuntimeError(f"OpenCV failed to open {path}")
  try:
    grays: list[UInt8Array] = []
    while max_frames is None or len(grays) < max_frames:
      ok, frame = cap.read()
      if not ok:
        break
      grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
  finally:
    cap.release()
  return grays


def finalize_probe(
  name: str,
  app_id: str,
  output_dir: Path,
  summary: Any,
  log_fn: Callable[[], None],
  *,
  extra_lines: list[str] | None = None,
) -> None:
  """Save a probe's ``.rrd``, logged by ``log_fn``, and its JSON summary."""
  stamp = time.strftime("%Y%m%d_%H%M%S")
  rrd_path = output_dir / f"{name}_{stamp}.rrd"
  json_path = output_dir / f"{name}_{stamp}.json"

  rr.init(app_id, spawn=False)
  rr.save(str(rrd_path))
  log_fn()

  payload = json.dumps(asdict(summary), indent=2)
  json_path.write_text(payload)
  print(payload)
  print(f"\nWrote {rrd_path} and {json_path}")
  for line in extra_lines or []:
    print(line)
  print(f"Verdict: {summary.verdict}")
