"""Split a side-by-side dual-fisheye Insta360 video into per-lens streams.

The X3 records both fisheyes packed side-by-side at 3840x1920 (1920x1920 per
lens). Downstream single-camera processing needs one lens at a time; this
helper crops the left and right halves into two separate ``.mp4`` files
using ``ffmpeg``.

``ffmpeg`` is invoked via ``subprocess`` (stream-copy is impossible across a
crop, so the output is re-encoded). The function returns the two output paths
once ``ffmpeg`` exits cleanly; otherwise it raises with the ``ffmpeg`` stderr.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def split_dual_fisheye(
  video_path: str | Path,
  out_dir: str | Path,
  *,
  crf: int = 18,
  preset: str = "medium",
) -> tuple[Path, Path]:
  """Crop side-by-side dual-fisheye into ``<stem>_front.mp4`` + ``_rear.mp4``.

  Writes the two halves into ``out_dir`` and returns ``(front, rear)``.
  """
  video_path = Path(video_path)
  if not video_path.is_file():
    raise FileNotFoundError(f"video not found: {video_path}")
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    raise RuntimeError(
      "ffmpeg not found on PATH. Install via 'brew install ffmpeg' or apt."
    )

  front = out_dir / f"{video_path.stem}_front.mp4"
  rear = out_dir / f"{video_path.stem}_rear.mp4"

  cmd = [
    ffmpeg,
    "-y",
    "-i",
    str(video_path),
    "-filter_complex",
    "[0:v]crop=iw/2:ih:0:0[front];[0:v]crop=iw/2:ih:iw/2:0[rear]",
    "-map",
    "[front]",
    "-c:v",
    "libx264",
    "-crf",
    str(crf),
    "-preset",
    preset,
    str(front),
    "-map",
    "[rear]",
    "-c:v",
    "libx264",
    "-crf",
    str(crf),
    "-preset",
    preset,
    str(rear),
  ]
  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
  return front, rear
