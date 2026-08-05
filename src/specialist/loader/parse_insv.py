"""Parse an Insta360 ``.insv`` file via ``telemetry-parser``.

Wraps ``telemetry_parser.Parser`` in a typed view of the bits we actually
use: IMU arrays with a single shared timestamp axis, the factory MEI
intrinsics, the rolling-shutter readout time, and the raw ``offset_v3``
array that encodes per-lens extrinsics.

``telemetry-parser`` exposes only one ``fisheye_params`` block per file, even
though the camera has two physical lenses. The dual-lens extrinsics and per-lens
intrinsics live in the raw ``offset_v3`` array (40 floats packed as
``[lens_count, front_block(19), rear_block(19), trailer]``) — callers that
need per-lens calibration decode it via ``specialist.vi_ba.extrinsics``;
``LensIntrinsics`` returns the single legacy MEI block as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import telemetry_parser

__all__ = ["InsvData", "LensIntrinsics", "load_insv"]


@dataclass(frozen=True)
class LensIntrinsics:
  """Factory MEI intrinsics for one fisheye, decoded from ``Lens.Data``.

  ``camera_matrix`` is the 3x3 pinhole projection. ``distortion_coeffs`` holds
  ``[k1, k2, k3, p1, p2, xi]`` — the Mei unified-omnidirectional model with a
  Brown-Conrady distortion polynomial, where ``xi`` is the unit-sphere mirror
  parameter.
  """

  camera_matrix: npt.NDArray[np.float64]
  distortion_coeffs: npt.NDArray[np.float64]
  calib_width: int
  calib_height: int
  distortion_model: str

  @property
  def xi(self) -> float:
    return float(self.distortion_coeffs[5])


@dataclass(frozen=True)
class InsvData:
  """All parsed bits from one ``.insv`` file.

  Timestamps are seconds on the video clock; the IMU pre-roll is negative.
  ``gyro_deg`` is deg/s (XYZ, as the source stores it), ``accel`` is m/s^2
  (XYZ); both share ``imu_timestamps``. ``rolling_shutter_readout`` is in
  seconds. ``offset_v3`` carries per-lens MEI intrinsics, the body -> lens
  pose (yaw, pitch, roll, tx, ty, tz) and Brown-Conrady distortion.
  ``gyro_calib_numbers`` is 6 floats: 3 gyro bias (rad/s), then 3 accel bias
  (m/s^2).
  """

  source_path: Path
  imu_timestamps: npt.NDArray[np.float64]
  gyro_deg: npt.NDArray[np.float64]
  accel: npt.NDArray[np.float64]
  intrinsics: LensIntrinsics
  rolling_shutter_readout: float
  offset_v3: npt.NDArray[np.float64]
  gyro_calib_numbers: npt.NDArray[np.float64]
  # ``capture`` is everything else the recording keeps; the exposure durations
  # (s) and the auto-exposure reports are on the video clock.
  capture: dict[str, Any] = field(default_factory=dict)
  exposure_timestamps: npt.NDArray[np.float64] = field(
    default_factory=lambda: np.empty(0)
  )
  exposure_durations: npt.NDArray[np.float64] = field(
    default_factory=lambda: np.empty(0)
  )
  ae_timestamps: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
  ae_stats: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)


_PARSED_FIELDS = frozenset({"offset_v3", "gyro_calib"})
_SUPERSEDED = frozenset(
  {
    "offset",
    "offset_v2",
    "original_offset",
    "original_offset_v2",
    "original_offset_v3",
    "original_offset_3d",
  }
)
_CONTAINER_DERIVED = frozenset(
  {
    "dimension",
    "resolution_size",
    "frame_rate",
    "total_time",
    "total_frames",
    "file_size",
  }
)
_ZERO_MEANS_UNSET = frozenset({"creation_time", "export_time"})


def capture_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
  """The capture facts worth keeping, flattened one level so every value is a
  scalar, a string, or a numeric array."""
  kept: dict[str, Any] = {}

  def put(key: str, value: Any) -> None:
    if value is None or value == "" or value == [] or value == {}:
      return
    if isinstance(value, dict):
      for sub, nested in value.items():
        put(f"{key}_{sub}", nested)
      return
    if isinstance(value, (list, tuple)):
      arr = np.asarray(value)
      if arr.dtype.kind in "if":
        kept[key] = arr.astype(np.float64)
      else:
        kept[key] = str(value)
      return
    kept[key] = value if isinstance(value, (bool, int, float)) else str(value)

  for key, value in metadata.items():
    if key in _PARSED_FIELDS or key in _SUPERSEDED or key in _CONTAINER_DERIVED:
      continue
    if key in _ZERO_MEANS_UNSET and not value:
      continue
    put(key, value)
  return kept


def load_insv(path: str | Path) -> InsvData:
  """Parse a ``.insv`` file and return its typed view."""
  path = Path(path)
  if not path.is_file():
    raise FileNotFoundError(f"insv not found: {path}")

  parser = telemetry_parser.Parser(str(path))
  raw = parser.telemetry()
  if not isinstance(raw, list) or not raw:
    raise ValueError(f"telemetry-parser returned empty telemetry for {path}")
  meta = raw[0]

  imu_samples = parser.normalized_imu()
  n = len(imu_samples)
  if n == 0:
    raise ValueError(f"no IMU samples in {path}")
  ts = np.array([s["timestamp_ms"] for s in imu_samples], dtype=np.float64) * 1e-3
  gyro = np.array([s["gyro"] for s in imu_samples], dtype=np.float64)
  accel = np.array([s["accl"] for s in imu_samples], dtype=np.float64)

  lens = meta["Lens"]["Data"]
  fisheye = lens["fisheye_params"]
  intrinsics = LensIntrinsics(
    camera_matrix=np.asarray(fisheye["camera_matrix"], dtype=np.float64),
    distortion_coeffs=np.asarray(fisheye["distortion_coeffs"], dtype=np.float64),
    calib_width=int(lens["calib_dimension"]["w"]),
    calib_height=int(lens["calib_dimension"]["h"]),
    distortion_model=str(lens["distortion_model"]),
  )

  default_meta = meta["Default"]["Metadata"]
  exposure = meta.get("Exposure", {}).get("Data", [])
  expo_t = np.array([e["t"] for e in exposure], dtype=np.float64)
  expo_v = np.array([e["v"] for e in exposure], dtype=np.float64)
  expo_order = np.argsort(expo_t, kind="stable")

  # The AE reports sit on the camera's boot clock; the first frame's boot
  # timestamp (us) moves them onto the video clock.
  reports = meta["Default"].get("AAAData", [])
  first_frame_us = float(default_meta.get("first_frame_timestamp") or 0.0)
  ae_t = np.empty(0)
  ae_stats: dict[str, npt.NDArray[np.float64]] = {}
  if reports and first_frame_us > 0.0:
    ae_t = (
      np.array([r["t"] for r in reports], dtype=np.float64) * 1e-3
      - first_frame_us * 1e-6
    )
    ae_order = np.argsort(ae_t, kind="stable")
    ae_t = ae_t[ae_order]
    ae_stats = {
      name: np.array([float(r["v"][name]) for r in reports], dtype=np.float64)[ae_order]
      for name in sorted(reports[0]["v"])
    }

  return InsvData(
    source_path=path,
    imu_timestamps=ts,
    gyro_deg=gyro,
    accel=accel,
    intrinsics=intrinsics,
    rolling_shutter_readout=float(lens["frame_readout_time"]) * 1e-3,
    offset_v3=np.asarray(default_meta["offset_v3"], dtype=np.float64),
    gyro_calib_numbers=np.asarray(
      default_meta.get("gyro_calib", {}).get("numbers", [0.0] * 6), dtype=np.float64
    ),
    capture=capture_metadata(default_meta),
    exposure_timestamps=expo_t[expo_order],
    exposure_durations=expo_v[expo_order],
    ae_timestamps=ae_t,
    ae_stats=ae_stats,
  )
