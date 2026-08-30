"""The dataset fiducial convention: the tag family, and where corners sit."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]

__all__ = ["TAG_FAMILY", "tag_corners_in_frame", "tag_corners_local"]

TAG_FAMILY = "tagStandard41h12"


def tag_corners_local(tag_size: float) -> Float64Array:
  """The tag's four corners in the tag frame, ``(4, 3)``.

  pupil_apriltags corner order ``[BL, BR, TR, TL]``; tag-frame axes ``+x``
  right, ``+y`` up, ``+z`` out.
  """
  half = float(tag_size / 2.0)
  return np.array(
    [
      [-half, -half, 0.0],
      [+half, -half, 0.0],
      [+half, +half, 0.0],
      [-half, +half, 0.0],
    ],
    dtype=np.float64,
  )


def tag_corners_in_frame(pose: Float64Array, tag_size: float) -> Float64Array:
  """A tag's four corners through a ``(4, 4)`` pose, ``(4, 3)``."""
  matrix = np.asarray(pose, dtype=np.float64)
  corners = tag_corners_local(tag_size)
  return corners @ matrix[:3, :3].T + matrix[:3, 3]
