"""The single source of record entity paths, timeline names, and how the
timeline decodes.

Base recordings, track layers, the viewer logging, and the record readers
all build paths through these helpers so the two files of a session (matched
by ``recording_id``) can never drift apart.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pyarrow as pa

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

APPLICATION_ID = "specialist"
TIMELINE = "video_time"
# Sequence timeline bucketing video frames to solver keyframes: frame ``i``
# carries index ``ceil(i / stride)``, so latest-at on this timeline lands on
# frame ``stride * k`` exactly and the cursor steps keyframe to keyframe.
KEYFRAME_TIMELINE = "keyframe"
NS_PER_SECOND = 1_000_000_000

# What each entity is called under the tracker that owns it, and the roots the
# shared entities hang off: a consumer matching a path and the code that writes
# it spell it from here.
IMAGE_ROOT = "image"
BODY = "body"
BODY_KEYFRAMES = "body_keyframes"
BASE_TAG = f"{BODY}/base_tag"
BODY_MARKER = f"{BODY}/marker"
LANDMARKS = "landmarks"
FACTOR_LINES = "factor_lines"
TRAJECTORY = "trajectory"
TF_STATIC = "tf_static"
METRICS_ROOT = "metrics"


def keyframe_stride(fps: float) -> int:
  """The auto keyframe interval: ~0.10 s via ``round(fps / 10)`` (3 at 30 fps,
  6 at 60 fps). The converter buckets the ``keyframe`` timeline with this, and
  the solve's auto stride matches it by construction."""
  return max(1, round(fps / 10.0))


def keyframe_index(
  frame_rows: npt.NDArray[np.int64], stride: int
) -> npt.NDArray[np.int64]:
  """``ceil(row / stride)`` per frame row: the ``keyframe`` timeline value."""
  return (frame_rows + stride - 1) // stride


def times_ns(column: pa.Array | pa.ChunkedArray) -> Int64Array:
  """A duration column as integer nanoseconds, taken whole out of arrow. A
  chunked column is joined first so ``zero_copy_only`` can hold, which makes a
  silently copied read an error rather than a doubled peak."""
  values = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
  out: Int64Array = values.to_numpy(zero_copy_only=True).astype("int64")
  return out


def seconds(column: pa.Array | pa.ChunkedArray) -> Float64Array:
  """A duration column in seconds."""
  out: Float64Array = times_ns(column) / NS_PER_SECOND
  return out


def image_root(tracker: str) -> str:
  return f"{IMAGE_ROOT}/{tracker}"


def image_entity(tracker: str, lens: str) -> str:
  return f"{image_root(tracker)}/{lens}"


def imu_entity(tracker: str, kind: str) -> str:
  return f"imu/{tracker}/{kind}"


def exposure_entity(tracker: str) -> str:
  return f"{image_root(tracker)}/exposure"


def ae_entity(tracker: str) -> str:
  return f"{image_root(tracker)}/ae"


def calibration_entity(tracker: str, part: str) -> str:
  return f"calibration/{tracker}/{part}"


def body_keyframes_entity(tracker: str) -> str:
  return f"{tracker}/{BODY_KEYFRAMES}"


def base_tag_entity(tracker: str) -> str:
  return f"{tracker}/{BASE_TAG}"
