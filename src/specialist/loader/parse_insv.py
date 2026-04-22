"""Parse an Insta360 ``.insv`` file via ``telemetry-parser``.

Wraps ``telemetry_parser.Parser`` so the probe scripts get a typed view of
the bits we actually use: IMU arrays with a single shared timestamp axis,
the factory MEI intrinsics, the rolling-shutter readout time, and the raw
``offset_v3`` array that encodes per-lens extrinsics.

``telemetry-parser`` exposes only one ``fisheye_params`` block per file, even
though the X3 has two physical lenses. The dual-lens extrinsics and per-lens
intrinsics live in the raw ``offset_v3`` array (40 floats packed as
``[lens_count, front_block(19), rear_block(19), trailer]``) — callers that
need per-lens calibration decode it via ``specialist.vi_ba.extrinsics``;
``LensIntrinsics`` returns the single legacy MEI block as-is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import telemetry_parser


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
  """All parsed bits from one ``.insv`` file that the probes consume.

  Timestamps are seconds, on the same clock as the video (the IMU pre-roll
  shows up as negative values). ``gyro_deg`` is deg/s (XYZ — non-SI to
  match the .insv source), ``accel`` is m/s^2 (XYZ); both arrays share
  ``imu_timestamps``.

  ``rolling_shutter_readout`` is the factory ``frame_readout_time`` from
  the lens block, converted to seconds. ``offset_v3`` is the raw 40-float
  array carrying per-lens MEI intrinsics + body → lens (yaw, pitch, roll,
  tx, ty, tz) + Brown-Conrady distortion — decoded by
  ``specialist.vi_ba.extrinsics.parse_offset_v3``. ``gyro_calib_numbers``
  is the factory IMU-bias calibration: 6 floats, first 3 gyro bias
  (rad/s), last 3 accel bias (m/s²).
  """

  source_path: Path
  camera: str
  model: str
  imu_timestamps: npt.NDArray[np.float64]
  gyro_deg: npt.NDArray[np.float64]  # raw .insv units are deg/s, not SI
  accel: npt.NDArray[np.float64]
  intrinsics: LensIntrinsics
  rolling_shutter_readout: float
  frame_width: int
  frame_height: int
  frame_rate: float
  offset_v3: npt.NDArray[np.float64]
  gyro_calib_numbers: npt.NDArray[np.float64]
  raw_telemetry: dict[str, Any]


def load_insv(path: str | Path) -> InsvData:
  """Parse a ``.insv`` file and return the typed view used by the probes."""
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
  ts = np.empty(n, dtype=np.float64)
  gyro = np.empty((n, 3), dtype=np.float64)
  accel = np.empty((n, 3), dtype=np.float64)
  for i, s in enumerate(imu_samples):
    ts[i] = float(s["timestamp_ms"]) * 1e-3
    gx, gy, gz = s["gyro"]
    ax, ay, az = s["accl"]
    gyro[i, 0], gyro[i, 1], gyro[i, 2] = float(gx), float(gy), float(gz)
    accel[i, 0], accel[i, 1], accel[i, 2] = float(ax), float(ay), float(az)

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
  dim = default_meta["dimension"]
  return InsvData(
    source_path=path,
    camera=str(parser.camera),
    model=str(parser.model),
    imu_timestamps=ts,
    gyro_deg=gyro,
    accel=accel,
    intrinsics=intrinsics,
    rolling_shutter_readout=float(lens["frame_readout_time"]) * 1e-3,
    frame_width=int(dim["x"]),
    frame_height=int(dim["y"]),
    frame_rate=float(default_meta["frame_rate"]),
    offset_v3=np.asarray(default_meta["offset_v3"], dtype=np.float64),
    gyro_calib_numbers=np.asarray(
      default_meta.get("gyro_calib", {}).get("numbers", [0.0] * 6), dtype=np.float64
    ),
    raw_telemetry=meta,
  )
