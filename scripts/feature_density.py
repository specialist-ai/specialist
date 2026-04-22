"""Feature-density probe.

Runs OpenCV ORB per frame on one fisheye (front or rear), reports features-
per-frame distribution, coarse track lifetime via nearest-neighbour matching
across frames, and a spatial heatmap of where features land in the image.
Flags intervals where feature count falls below a floor as likely VIO
failure zones.

Capture: representative manipulation episode.

Outputs: ``logs/feature_density_<ts>.rrd`` (feature count stream + heatmap),
JSON summary, stdout verdict on whether the scene yields enough features to
keep a VIO alive.

Status: keep — run once per scene class.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr
import tyro

Float64Array = npt.NDArray[np.float64]


@dataclass
class FeatureArgs:
  """CLI arguments for the feature-density probe."""

  input_insv: Path
  output_dir: Path = Path("logs")
  lens: str = "front"  # 'front' (Lens 1, left half) or 'rear' (Lens 2, right half)
  max_frames: int = 400
  orb_n_features: int = 1500
  orb_fast_threshold: int = 12
  low_feature_floor: int = 80
  heatmap_bins: int = 32


@dataclass
class FeatureSummary:
  source: str
  lens: str
  n_frames: int
  mean_features: float
  median_features: float
  p10_features: float
  p90_features: float
  mean_track_lifetime_frames: float
  fraction_low_feature_frames: float
  low_feature_floor: int
  verdict: str = "unknown"


def _iter_frames(
  video_path: Path, max_frames: int, lens: str
) -> tuple[list[Float64Array], int, int]:
  """Decode ``max_frames`` grey frames from the chosen fisheye half."""
  if lens not in ("front", "rear"):
    raise ValueError(f"lens must be 'front' or 'rear', got {lens!r}")
  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    raise RuntimeError(f"OpenCV failed to open {video_path}")
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  half = width // 2
  grays: list[Float64Array] = []
  i = 0
  while i < max_frames:
    ok, frame = cap.read()
    if not ok:
      break
    crop = frame[:, :half] if lens == "front" else frame[:, half:]
    grays.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64))
    i += 1
  cap.release()
  return grays, half, height


def _extract_features(
  grays: list[Float64Array], n_features: int, fast_threshold: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
  orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=fast_threshold)
  all_kp: list[np.ndarray] = []
  all_desc: list[np.ndarray] = []
  for gray in grays:
    kp, desc = orb.detectAndCompute(gray.astype(np.uint8), None)
    if kp is None or desc is None:
      all_kp.append(np.empty((0, 2), dtype=np.float32))
      all_desc.append(np.empty((0, 32), dtype=np.uint8))
      continue
    all_kp.append(np.asarray([p.pt for p in kp], dtype=np.float32))
    all_desc.append(desc)
  return all_kp, all_desc


def _mean_track_lifetime(descs: list[np.ndarray], ratio: float = 0.75) -> float:
  """Crude: for each frame, ratio-test BFMatch to the previous frame; a track
  that persists across k consecutive frames contributes lifetime k."""
  if len(descs) < 2:
    return 0.0
  matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
  lifetimes: list[int] = []
  live_lengths: dict[int, int] = {}
  for k in range(1, len(descs)):
    if descs[k - 1].size == 0 or descs[k].size == 0:
      live_lengths.clear()
      continue
    pairs = matcher.knnMatch(descs[k - 1], descs[k], k=2)
    matched_new: dict[int, int] = {}
    for m in pairs:
      if len(m) < 2:
        continue
      a, b = m
      if a.distance < ratio * b.distance:
        matched_new[a.trainIdx] = a.queryIdx
    new_live: dict[int, int] = {}
    for train_idx, query_idx in matched_new.items():
      prev_len = live_lengths.get(query_idx, 1)
      new_live[train_idx] = prev_len + 1
    ended = set(live_lengths) - set(matched_new.values())
    for q in ended:
      lifetimes.append(live_lengths[q])
    live_lengths = new_live
  lifetimes.extend(live_lengths.values())
  return float(np.mean(lifetimes)) if lifetimes else 0.0


def _heatmap(
  kpts_per_frame: list[np.ndarray], width: int, height: int, bins: int
) -> Float64Array:
  heatmap = np.zeros((bins, bins), dtype=np.float64)
  for kp in kpts_per_frame:
    if kp.size == 0:
      continue
    x_bin = np.clip((kp[:, 0] / width * bins).astype(int), 0, bins - 1)
    y_bin = np.clip((kp[:, 1] / height * bins).astype(int), 0, bins - 1)
    for xb, yb in zip(x_bin.tolist(), y_bin.tolist(), strict=True):
      heatmap[yb, xb] += 1.0
  return heatmap


def _verdict(summary: FeatureSummary) -> str:
  if summary.median_features < summary.low_feature_floor:
    return (
      f"red: median {summary.median_features:.0f} features "
      f"below floor {summary.low_feature_floor} — VIO will struggle"
    )
  if summary.fraction_low_feature_frames > 0.1:
    return (
      f"yellow: {summary.fraction_low_feature_frames * 100:.0f}% of frames "
      f"below {summary.low_feature_floor} features"
    )
  return f"green: median {summary.median_features:.0f} features, mean track length {summary.mean_track_lifetime_frames:.1f}"


def _log_to_rerun(
  n_per_frame: list[int],
  heatmap: Float64Array,
) -> None:
  for k, n in enumerate(n_per_frame):
    rr.set_time("frame", sequence=k)
    rr.log("features/count", rr.Scalars([float(n)]))
  rr.log("features/heatmap", rr.Tensor(heatmap), static=True)


def main(args: FeatureArgs) -> None:
  args.output_dir.mkdir(parents=True, exist_ok=True)
  grays, width, height = _iter_frames(args.input_insv, args.max_frames, args.lens)
  if len(grays) < 3:
    raise RuntimeError(f"only {len(grays)} frames decoded")

  kpts, descs = _extract_features(grays, args.orb_n_features, args.orb_fast_threshold)
  n_per_frame = [int(k.shape[0]) for k in kpts]
  counts = np.asarray(n_per_frame, dtype=np.float64)

  lifetime = _mean_track_lifetime(descs)
  heatmap = _heatmap(kpts, width, height, args.heatmap_bins)

  summary = FeatureSummary(
    source=str(args.input_insv),
    lens=args.lens,
    n_frames=len(grays),
    mean_features=float(np.mean(counts)),
    median_features=float(np.median(counts)),
    p10_features=float(np.percentile(counts, 10)),
    p90_features=float(np.percentile(counts, 90)),
    mean_track_lifetime_frames=lifetime,
    fraction_low_feature_frames=float(np.mean(counts < args.low_feature_floor)),
    low_feature_floor=args.low_feature_floor,
  )
  summary.verdict = _verdict(summary)

  stamp = time.strftime("%Y%m%d_%H%M%S")
  rrd_path = args.output_dir / f"feature_density_{stamp}.rrd"
  json_path = args.output_dir / f"feature_density_{stamp}.json"

  rr.init("specialist.feature_density", spawn=False)
  rr.save(str(rrd_path))
  _log_to_rerun(n_per_frame, heatmap)

  json_path.write_text(json.dumps(asdict(summary), indent=2))
  print(json.dumps(asdict(summary), indent=2))
  print(f"\nWrote {rrd_path} and {json_path}")
  print(f"Verdict: {summary.verdict}")


if __name__ == "__main__":
  main(tyro.cli(FeatureArgs))
