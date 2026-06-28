# specialist

Multi-camera AprilTag SLAM for humanoid data collection.

Each humanoid gripper carries an Insta360 (two back-to-back fisheye cameras) plus an AprilTag mounted on the body. A large wall- or ceiling-mounted anchor tag provides the world gauge. Per-tracker VIO supplies odometry; the cameras see the anchor and the other trackers' tags. A GTSAM `NonlinearFactorGraph` fuses all three signals into globally-consistent, inter-tracker-consistent trajectories.

The pipeline runs on real Insta360 X3 captures end to end: the loader parses `.insv` video + telemetry, and `scripts/track.py` runs per-tracker visual-inertial bundle adjustment against the wall anchor. See **Probes** below for the real-data characterization scripts the pipeline is built on.

## Install

```
uv sync --extra dev
```

This pulls `gtsam>=4.3a0` (the alpha — it is the only release with cp312 wheels), `numpy`, `scipy`, `rerun-sdk`, `tyro`, `telemetry-parser`, `opencv-python`, `pupil-apriltags`, `allantools`, `pyyaml`, and `imageio-ffmpeg` (wheel-bundled ffmpeg for audio clap-sync), plus the dev tools `pytest`, `ruff`, `mypy`, and `pre-commit`. `robot-descriptions` (gripper URDF assets) is a private SSH git dependency.

## Test

```
uv run pytest
```

Unit tests per `loader` / `vi_ba` module, plus integration tests covering the VI-BA pipeline on a synthetic scene and IMU preintegration on a real 30-second stationary window. Loader smoke tests that depend on a `.insv` fixture skip cleanly when no capture is available; VI-BA real-data assertions use tiny fixtures (40-float `offset_v3` JSON, 30-s IMU `.npz`) committed under `tests/fixtures/vi_ba/`.

## Visualize

Open a saved `.rrd` in the Rerun viewer:

```
uv run rerun logs/track_<stamp>.rrd
```

## Layout

```
src/specialist/
├── loader/      parse_insv, mei_model, audio_sync
├── vi_ba/       per-tracker visual-inertial bundle adjustment — single_tracker
│                (per-tracker prep), pipeline + features (frontend), init
│                (incremental SfM + VI alignment), build / compose (graph),
│                mei_factor, imu, anchor_detection, extrinsics, optimize,
│                reliability, configs, viz
├── graph/       factors (shared GTSAM factor constructors)
└── synthesis/   tag_detection (the cross-tracker observation record)

tests/
├── unit/        loader + vi_ba module tests
└── integration/ VI-BA pipeline on a synthetic scene, IMU preintegration

scripts/
├── imu.py              monotonicity, sample rate, gyro/accel alignment, Allan deviation
├── video_imu_timing.py LK-flow / gyro cross-correlation + rolling-shutter shear
└── track.py            VI-BA driver for any number of trackers (N ≥ 1)
```

## Probes

`scripts/` holds real-data probes that answer open questions about Insta360 X3 captures before the SLAM pipeline is built on top. Each probe is a standalone `tyro`-driven CLI that writes a Rerun `.rrd` and a JSON summary to `logs/` and prints a one-line verdict. Pass `--input-insv <path>` per run.

```
uv run python scripts/imu.py --input-insv path/to/stationary.insv
uv run python scripts/video_imu_timing.py --input-insv path/to/yaw.insv
uv run python scripts/track.py --input-insvs path/to/manipulation.insv --base-tag-ids <base-tag-id> --anchor.tag-id <anchor-tag-id> --window.master-start 2.0 --window.duration 10.0
```

