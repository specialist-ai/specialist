"""Raw dual-lens capture probing.

A capture is a square ``_00_`` master ``.insv`` carrying the telemetry, plus
its ``_10_`` rear sibling.
"""

from __future__ import annotations

from pathlib import Path

import cv2

__all__ = ["capture_duration", "probe_capture"]


def _video_props(path: Path, *props: int) -> list[float]:
  """Container properties read via one short-lived ``VideoCapture``."""
  cap = cv2.VideoCapture(str(path))
  if not cap.isOpened():
    raise RuntimeError(f"OpenCV cannot open {path}")
  values = [float(cap.get(prop)) for prop in props]
  cap.release()
  return values


def probe_capture(video_path: Path) -> tuple[Path, float, int]:
  """Locate a capture's rear-lens file and read the container frame rate
  and native per-lens width.

  ``video_path`` is the square ``_00_`` master carrying the telemetry; its
  rear sibling swaps ``_00_`` for ``_10_`` in the filename. The frame rate
  comes from the container: the telemetry rounds the NTSC rate (30.0 for a
  30000/1001 stream), which would stretch keyframe timestamps against the
  IMU clock. Raises ``RuntimeError`` unless the video is square and its
  sibling exists."""
  raw_w, raw_h, fps = _video_props(
    video_path, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS
  )
  width, height = int(raw_w), int(raw_h)
  rear_path = video_path.with_name(video_path.name.replace("_00_", "_10_", 1))
  if width != height or rear_path == video_path or not rear_path.is_file():
    raise RuntimeError(
      f"{video_path.name} ({width}x{height}) is not a square per-lens master "
      f"with sibling {rear_path.name}"
    )
  return rear_path, fps, width


def _shared_frame_count(front_path: Path, rear_path: Path) -> int:
  """Frames present in both lens files; the camera may close one file of the
  pair a frame before the other."""
  return min(
    int(_video_props(path, cv2.CAP_PROP_FRAME_COUNT)[0])
    for path in (front_path, rear_path)
  )


def capture_duration(video_path: Path) -> float:
  """Seconds decodable from a capture, bounded by the shorter lens file."""
  rear_path, fps, _ = probe_capture(video_path)
  return _shared_frame_count(video_path, rear_path) / fps
