"""Raw dual-lens capture probing.

A capture is a square ``_00_`` master ``.insv`` carrying the telemetry, plus
its ``_10_`` rear sibling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

__all__ = ["CaptureInfo", "probe_capture"]


@dataclass(frozen=True)
class CaptureInfo:
  """What one probe of a capture pair reads: the rear-lens file, the container
  frame rate, the native per-lens width, and the seconds both lens files
  cover."""

  rear_path: Path
  fps: float
  width: int
  duration: float


def _video_props(path: Path, *props: int) -> list[float]:
  """Container properties read via one short-lived ``VideoCapture``."""
  cap = cv2.VideoCapture(str(path))
  if not cap.isOpened():
    raise RuntimeError(f"OpenCV cannot open {path}")
  values = [float(cap.get(prop)) for prop in props]
  cap.release()
  return values


def probe_capture(video_path: Path) -> CaptureInfo:
  """Read a capture's container facts, one pass per lens file.

  ``video_path`` is the square ``_00_`` master; its rear sibling swaps ``_00_``
  for ``_10_`` in the filename. The frame rate comes from the container, which
  unlike the telemetry does not round the NTSC rate. The duration counts only
  the frames both files hold."""
  raw_w, raw_h, fps, front_frames = _video_props(
    video_path,
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS,
    cv2.CAP_PROP_FRAME_COUNT,
  )
  width, height = int(raw_w), int(raw_h)
  rear_path = video_path.with_name(video_path.name.replace("_00_", "_10_", 1))
  if width != height or rear_path == video_path or not rear_path.is_file():
    raise RuntimeError(
      f"{video_path.name} ({width}x{height}) is not a square per-lens master "
      f"with sibling {rear_path.name}"
    )
  rear_frames = _video_props(rear_path, cv2.CAP_PROP_FRAME_COUNT)[0]
  return CaptureInfo(
    rear_path=rear_path,
    fps=fps,
    width=width,
    duration=min(int(front_frames), int(rear_frames)) / fps,
  )
