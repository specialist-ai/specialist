"""Camera-model probe.

Two coupled checks against the factory MEI intrinsics from a ``.insv``:

  - MEI reprojection: detect AprilGrid corners on a calibration frame, solve
    pose via MEI-undistorted bearings, project the 3D corners back through
    ``mei_project``, and report mean / max / periphery reprojection error.
  - KB fit: sample rays uniformly in half-angle, project them through MEI,
    fit Kannala-Brandt to the resulting (ray, pixel) pairs via
    ``scipy.optimize.least_squares``, and report residual vs. half-angle.

The two are coupled: if MEI reprojection is red, the KB fit is meaningless.
The script refuses to write the KB YAML if MEI reprojection exceeds the
red threshold.

Capture: AprilGrid printed at known size, framed several times across the
fisheye FOV.

Outputs: ``logs/camera_model_<ts>.rrd`` (reprojection heatmap + KB residual
curve), ``logs/camera_model_<ts>.json`` summary, and on green/yellow MEI:
``logs/kb_calib_<stem>.yaml`` (Kannala-Brandt intrinsics YAML).

Status: keep — regression check for new camera units / firmware.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr
import tyro
import yaml
from pupil_apriltags import Detector
from scipy.optimize import least_squares

from specialist.loader.kb_model import KbParams, kb_project
from specialist.loader.mei_model import MeiParams, mei_project, mei_unproject
from specialist.loader.parse_insv import InsvData, load_insv

Float64Array = npt.NDArray[np.float64]


@dataclass
class CameraModelArgs:
  """CLI arguments for the camera-model probe."""

  input_insv: Path
  output_dir: Path = Path("logs")
  lens: str = "front"  # 'front' (Lens 1, left half) or 'rear' (Lens 2, right half)
  tag_size: float = 0.04
  tag_family: str = "tag36h11"
  tag_spacing: float = 0.012
  tag_rows: int = 6
  tag_cols: int = 6
  frame_index: int = 0
  kb_max_half_angle_deg: float = 95.0
  kb_n_samples: int = 2000
  mei_red_threshold_px: float = 10.0


@dataclass
class MeiReprojectionResult:
  n_corners: int
  mean_error_px: float
  max_error_px: float
  periphery_error_px: float
  status: str  # green / yellow / red


@dataclass
class KbFitResult:
  params: list[float]
  residual_max_under_70deg_px: float
  residual_max_under_90deg_px: float
  residual_overall_px: float
  half_angle_first_pixel_residual_deg: float


@dataclass
class CameraModelSummary:
  source: str
  intrinsics: dict[str, float]
  mei: MeiReprojectionResult | None
  kb: KbFitResult | None
  kb_yaml_path: str | None = None
  verdict: str = "unknown"
  notes: list[str] = field(default_factory=list)


def _build_grid_3d(
  tag_size: float, tag_spacing: float, rows: int, cols: int
) -> dict[int, Float64Array]:
  """Each tag has 4 corners on the Z=0 plane, in metres.

  Corner order is TL, TR, BR, BL — chosen empirically to match what the rear
  X3 lens delivers. pupil_apriltags returns the four image-frame corners
  going CCW from the lower-left (image y-down: LL, LR, UR, UL); pairing
  detector index 0 with our world-frame TL means the image-LL corner
  corresponds to the physical top-left of the printed grid. The rear-lens
  side-by-side packing inverts chirality once compared to the natural-view
  convention, which flips bottom↔top in the world-frame mapping.
  """
  grid: dict[int, Float64Array] = {}
  pitch = tag_size + tag_spacing
  for r in range(rows):
    for c in range(cols):
      tag_id = r * cols + c
      x0 = c * pitch
      y0 = r * pitch
      x1 = x0 + tag_size
      y1 = y0 + tag_size
      corners = np.array(
        [
          [x0, y0, 0.0],  # TL  ← detector index 0 (image LL)
          [x1, y0, 0.0],  # TR  ← detector index 1 (image LR)
          [x1, y1, 0.0],  # BR  ← detector index 2 (image UR)
          [x0, y1, 0.0],  # BL  ← detector index 3 (image UL)
        ],
        dtype=np.float64,
      )
      grid[tag_id] = corners
  return grid


def _detect_apriltags(
  gray: Float64Array, family: str
) -> list[tuple[int, Float64Array]]:
  detector = Detector(
    families=family,
    nthreads=4,
    quad_decimate=1.0,
    refine_edges=1,
  )
  detections = detector.detect(gray.astype(np.uint8))
  results: list[tuple[int, Float64Array]] = []
  for d in detections:
    results.append((int(d.tag_id), np.asarray(d.corners, dtype=np.float64)))
  return results


def _read_fisheye_frame(video_path: Path, frame_index: int, lens: str) -> Float64Array:
  if lens not in ("front", "rear"):
    raise ValueError(f"lens must be 'front' or 'rear', got {lens!r}")
  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    raise RuntimeError(f"OpenCV failed to open {video_path}")
  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
  ok, frame = cap.read()
  cap.release()
  if not ok or frame is None:
    raise RuntimeError(f"failed to read frame {frame_index} from {video_path}")
  width = frame.shape[1]
  half = width // 2
  cropped = frame[:, :half] if lens == "front" else frame[:, half:]
  return cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY).astype(np.float64)


def _solve_pose_from_bearings(
  bearings: Float64Array, points_3d: Float64Array
) -> tuple[Float64Array, Float64Array]:
  """Linear pose via DLT-style approach + nonlinear refinement.

  ``bearings`` are unit vectors in the camera frame. We minimise
  ``angle(bearing, R*X + t)`` over (R, t).
  """
  rvec = np.zeros(3, dtype=np.float64)
  tvec = np.array([0.0, 0.0, 1.0], dtype=np.float64)

  def residuals(x: Float64Array) -> Float64Array:
    rv = x[:3]
    tv = x[3:]
    R, _ = cv2.Rodrigues(rv)
    pts_cam = (R @ points_3d.T + tv.reshape(3, 1)).T
    norms = np.linalg.norm(pts_cam, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    pred = pts_cam / norms
    return (pred - bearings).reshape(-1)

  result = least_squares(residuals, np.concatenate([rvec, tvec]), method="lm")
  R, _ = cv2.Rodrigues(result.x[:3])
  return R, result.x[3:]


def _mei_reprojection(
  gray: Float64Array,
  intrinsics: MeiParams,
  args: CameraModelArgs,
) -> tuple[MeiReprojectionResult, Float64Array, Float64Array]:
  detections = _detect_apriltags(gray, args.tag_family)
  if not detections:
    return (
      MeiReprojectionResult(
        n_corners=0,
        mean_error_px=float("nan"),
        max_error_px=float("nan"),
        periphery_error_px=float("nan"),
        status="red",
      ),
      np.empty((0, 2)),
      np.empty((0, 2)),
    )
  grid = _build_grid_3d(args.tag_size, args.tag_spacing, args.tag_rows, args.tag_cols)

  pixels: list[Float64Array] = []
  points_3d: list[Float64Array] = []
  for tag_id, corners in detections:
    if tag_id not in grid:
      continue
    pixels.append(corners)
    points_3d.append(grid[tag_id])
  if not pixels:
    return (
      MeiReprojectionResult(
        n_corners=0,
        mean_error_px=float("nan"),
        max_error_px=float("nan"),
        periphery_error_px=float("nan"),
        status="red",
      ),
      np.empty((0, 2)),
      np.empty((0, 2)),
    )
  pixels_arr = np.concatenate(pixels, axis=0)
  points_arr = np.concatenate(points_3d, axis=0)

  bearings = mei_unproject(pixels_arr, intrinsics)
  finite = np.isfinite(bearings).all(axis=1)
  bearings = bearings[finite]
  pixels_arr = pixels_arr[finite]
  points_arr = points_arr[finite]

  R, t = _solve_pose_from_bearings(bearings, points_arr)
  pts_cam = (R @ points_arr.T + t.reshape(3, 1)).T
  reproj = mei_project(pts_cam, intrinsics)

  errors = np.linalg.norm(reproj - pixels_arr, axis=1)
  finite = np.isfinite(errors)
  errors = errors[finite]
  pixels_arr = pixels_arr[finite]
  reproj = reproj[finite]

  if errors.size == 0:
    status = "red"
  elif float(np.nanmean(errors)) < 1.0:
    status = "green"
  elif float(np.nanmean(errors)) < args.mei_red_threshold_px:
    status = "yellow"
  else:
    status = "red"

  cx_img, cy_img = intrinsics.cx, intrinsics.cy
  radii = np.linalg.norm(pixels_arr - np.array([cx_img, cy_img]), axis=1)
  periphery_mask = radii > np.percentile(radii, 75)
  periphery_err = (
    float(np.median(errors[periphery_mask])) if periphery_mask.any() else float("nan")
  )

  return (
    MeiReprojectionResult(
      n_corners=int(errors.size),
      mean_error_px=float(np.mean(errors)),
      max_error_px=float(np.max(errors)),
      periphery_error_px=periphery_err,
      status=status,
    ),
    pixels_arr,
    reproj,
  )


def _sample_rays_in_cone(
  n: int, half_angle: float, rng: np.random.Generator
) -> Float64Array:
  cos_min = float(np.cos(half_angle))
  cos_t = rng.uniform(cos_min, 1.0, size=n)
  sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
  phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
  return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)


def _fit_kb_to_mei(
  intrinsics: MeiParams, args: CameraModelArgs
) -> tuple[KbFitResult, Float64Array, Float64Array]:
  rng = np.random.default_rng(0)
  rays = _sample_rays_in_cone(
    args.kb_n_samples, np.deg2rad(args.kb_max_half_angle_deg), rng
  )
  mei_pixels = mei_project(rays, intrinsics)
  finite = np.isfinite(mei_pixels).all(axis=1)
  rays = rays[finite]
  mei_pixels = mei_pixels[finite]

  x0 = np.array(
    [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
  )

  def residuals(x: Float64Array) -> Float64Array:
    pred = kb_project(rays, KbParams.from_array(x))
    return (pred - mei_pixels).reshape(-1)

  result = least_squares(residuals, x0, method="lm", max_nfev=200)
  fitted = KbParams.from_array(result.x)
  reproj = kb_project(rays, fitted)

  errors = np.linalg.norm(reproj - mei_pixels, axis=1)
  half_angles_deg = np.degrees(np.arccos(np.clip(rays[:, 2], -1.0, 1.0)))

  under_70 = half_angles_deg < 70.0
  under_90 = half_angles_deg < 90.0
  res_70 = float(np.max(errors[under_70])) if under_70.any() else float("nan")
  res_90 = float(np.max(errors[under_90])) if under_90.any() else float("nan")

  order = np.argsort(half_angles_deg)
  errs_sorted = errors[order]
  ang_sorted = half_angles_deg[order]
  first_pixel = ang_sorted[errs_sorted > 1.0]
  first_pixel_deg = float(first_pixel[0]) if first_pixel.size else float("nan")

  return (
    KbFitResult(
      params=fitted.to_array().tolist(),
      residual_max_under_70deg_px=res_70,
      residual_max_under_90deg_px=res_90,
      residual_overall_px=float(np.max(errors)),
      half_angle_first_pixel_residual_deg=first_pixel_deg,
    ),
    half_angles_deg,
    errors,
  )


def _write_kb_yaml(
  path: Path, kb: KbFitResult, data: InsvData, intrinsics: MeiParams
) -> None:
  fx, fy, cx, cy, k1, k2, k3, k4 = kb.params
  doc = {
    "Camera.type": "KannalaBrandt8",
    "Camera1.fx": fx,
    "Camera1.fy": fy,
    "Camera1.cx": cx,
    "Camera1.cy": cy,
    "Camera1.k1": k1,
    "Camera1.k2": k2,
    "Camera1.k3": k3,
    "Camera1.k4": k4,
    "Camera.width": data.frame_width // 2,
    "Camera.height": data.frame_height,
    "Camera.fps": data.frame_rate,
    "Camera.RGB": 1,
    "_source": str(data.source_path),
    "_factory_mei_xi": intrinsics.xi,
  }
  path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _verdict(summary: CameraModelSummary) -> str:
  if summary.mei is None or summary.mei.n_corners == 0:
    return "red: no AprilTag corners detected — wrong family/grid or wrong frame"
  if summary.mei.status == "red":
    return f"red: MEI mean {summary.mei.mean_error_px:.1f} px (max {summary.mei.max_error_px:.1f} px)"
  if summary.kb is None:
    return f"yellow: MEI {summary.mei.status} but KB fit skipped"
  if summary.kb.residual_max_under_70deg_px > 1.0:
    return (
      f"yellow: MEI {summary.mei.status}; KB residual {summary.kb.residual_max_under_70deg_px:.2f} px "
      f"already exceeds 1 px under 70°"
    )
  return (
    f"green: MEI mean {summary.mei.mean_error_px:.2f} px; "
    f"KB <1 px up to {summary.kb.half_angle_first_pixel_residual_deg:.0f}°"
  )


def _log_to_rerun(
  pixels: Float64Array,
  reproj: Float64Array,
  kb_angles: Float64Array,
  kb_errors: Float64Array,
) -> None:
  if pixels.size > 0:
    rr.log(
      "mei/corners/observed",
      rr.Points2D(pixels, colors=[120, 220, 160], radii=2.0),
    )
    rr.log(
      "mei/corners/reprojected",
      rr.Points2D(reproj, colors=[230, 80, 200], radii=2.0),
    )
  if kb_angles.size > 0:
    order = np.argsort(kb_angles)
    for ang, err in zip(kb_angles[order], kb_errors[order], strict=True):
      rr.set_time("kb_half_angle_deg", duration=float(ang))
      rr.log("kb/residual_px", rr.Scalars([float(err)]))


def main(args: CameraModelArgs) -> None:
  args.output_dir.mkdir(parents=True, exist_ok=True)
  data = load_insv(args.input_insv)
  intrinsics = MeiParams.from_camera_matrix(
    data.intrinsics.camera_matrix, data.intrinsics.distortion_coeffs
  )

  notes: list[str] = []
  gray = _read_fisheye_frame(args.input_insv, args.frame_index, args.lens)
  mei_result, pixels, reproj = _mei_reprojection(gray, intrinsics, args)

  kb_result: KbFitResult | None = None
  kb_angles = np.empty((0,), dtype=np.float64)
  kb_errors = np.empty((0,), dtype=np.float64)
  kb_yaml_path: str | None = None

  if mei_result.status != "red":
    kb_result, kb_angles, kb_errors = _fit_kb_to_mei(intrinsics, args)
    yaml_path = args.output_dir / f"kb_calib_{args.input_insv.stem}.yaml"
    _write_kb_yaml(yaml_path, kb_result, data, intrinsics)
    kb_yaml_path = str(yaml_path)
  else:
    notes.append("KB YAML not written: MEI reprojection failed")

  summary = CameraModelSummary(
    source=str(args.input_insv),
    intrinsics={
      "fx": intrinsics.fx,
      "fy": intrinsics.fy,
      "cx": intrinsics.cx,
      "cy": intrinsics.cy,
      "xi": intrinsics.xi,
    },
    mei=mei_result,
    kb=kb_result,
    kb_yaml_path=kb_yaml_path,
    notes=notes,
  )
  summary.verdict = _verdict(summary)

  stamp = time.strftime("%Y%m%d_%H%M%S")
  rrd_path = args.output_dir / f"camera_model_{stamp}.rrd"
  json_path = args.output_dir / f"camera_model_{stamp}.json"

  rr.init("specialist.camera_model", spawn=False)
  rr.save(str(rrd_path))
  _log_to_rerun(pixels, reproj, kb_angles, kb_errors)

  json_path.write_text(json.dumps(asdict(summary), indent=2))
  print(json.dumps(asdict(summary), indent=2))
  print(f"\nWrote {rrd_path} and {json_path}")
  if kb_yaml_path:
    print(f"KB calibration: {kb_yaml_path}")
  print(f"Verdict: {summary.verdict}")


if __name__ == "__main__":
  main(tyro.cli(CameraModelArgs))
