"""Parse an Insta360 ``.insv`` file via ``telemetry-parser``.

Wraps ``telemetry_parser.Parser`` so the probe scripts get a typed view of
the bits we actually use: IMU arrays with a single shared timestamp axis,
the factory MEI intrinsics, the rolling-shutter readout time, and the raw
``offset_v3`` array that encodes per-lens extrinsics.

``telemetry-parser`` exposes only one ``fisheye_params`` block per file, even
though the camera has two physical lenses. The dual-lens extrinsics and per-lens
intrinsics live in the raw ``offset_v3`` array (40 floats packed as
``[lens_count, front_block(19), rear_block(19), trailer]``) — callers that
need per-lens calibration decode it via ``specialist.vi_ba.extrinsics``;
``LensIntrinsics`` returns the single legacy MEI block as-is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
  ts = np.empty(n)
  gyro = np.empty((n, 3))
  accel = np.empty((n, 3))
  for i, s in enumerate(imu_samples):
    ts[i] = float(s["timestamp_ms"]) * 1e-3
    gyro[i] = s["gyro"]
    accel[i] = s["accl"]

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
  )
