# specialist

Multi-camera AprilTag SLAM for humanoid data collection.

Each humanoid gripper carries an Insta360 (two back-to-back fisheye cameras) plus an AprilTag mounted on the body. A large wall- or ceiling-mounted anchor tag provides the world gauge. Per-tracker VIO supplies odometry; the cameras see the anchor and the other trackers' tags. A GTSAM `NonlinearFactorGraph` fuses all three signals into globally-consistent, inter-tracker-consistent trajectories.

This repo currently exercises the factor-graph algorithms on synthetic SE(3) data; real VIO output and AprilTag detections from Insta360 video will land through a loader module later. See **Probes** below for the real-data characterization scripts that back this plan.

## Install

```
uv sync --extra dev
```

This pulls `gtsam>=4.3a0` (the alpha — it is the only release with cp312 wheels), `numpy`, `scipy`, `rerun-sdk`, `tyro`, `telemetry-parser`, `opencv-python`, `pupil-apriltags`, `allantools`, and `pyyaml`, plus the dev tools `pytest`, `ruff`, `mypy`, and `pre-commit`.

## Test

```
uv run pytest
```

162 tests: unit tests per synthesis / graph / eval / loader / vi_ba module, plus integration tests covering noiseless recovery, within-CRB-noise recovery, mid-trajectory anchor blackout, the cross-tracker-only regime where the anchor is seen once at init, the VI-BA pipeline on a synthetic scene, and IMU preintegration on a real 30-second stationary window. Loader smoke tests that depend on a `.insv` fixture skip cleanly when no capture is available; VI-BA real-data assertions use tiny fixtures (40-float `offset_v3` JSON, 30-s IMU `.npz`) committed under `tests/fixtures/vi_ba/`.

## Run

Generate one synthetic scenario, optimize it, and write a `.rrd` plus a JSON summary:

```
uv run python examples/run_synthetic.py
```

Override any scenario parameter via tyro-generated CLI flags:

```
uv run python examples/run_synthetic.py \
  --scenario.seed 7 \
  --scenario.duration 20 \
  --scenario.noise.odometry-translation 0.005
```

## Visualize

Open a saved `.rrd` in the Rerun viewer:

```
uv run python examples/visualize_run.py logs/specialist_run.rrd
```

The viewer shows ground-truth and estimated trajectories per tracker (green vs blue), the anchor and on-body tags, observation rays (yellow = anchor, magenta = cross-tracker), and scalar streams for per-tracker ATE and LM convergence metadata.

## Tune

All tunable knobs live on `ScenarioConfig` in `src/specialist/config.py`. The nested `NoiseConfig` holds per-source noise sigmas (odometry, anchor detection, cross-tracker detection, anchor prior); the nested `SceneConfig` holds anchor pose, per-tracker rig extrinsics, and fisheye FOV / distance bounds. Tests scale their tolerances via the Cramér-Rao bound, so changing σ values doesn't break the suite.

## Layout

```
src/specialist/
├── config.py        NoiseConfig, SceneConfig, ScenarioConfig
├── synthesis/       trajectory, scene, visibility, odometry, tag_detection
├── graph/           keys, factors, build, optimize
├── eval/            metrics (ATE, RPE), crb
├── loader/          parse_insv, mei_model, kb_model, split_lenses
├── vi_ba/           per-tracker visual-inertial bundle adjustment (extrinsics,
│                    mei_factor, features, init, build, optimize, imu,
│                    pipeline, viz)
├── runner.py        end-to-end scenario driver
└── viz.py           Rerun logging helpers

tests/
├── unit/            one test file per source module
└── integration/     noiseless, with_noise, blackout, cross_tracker_only

examples/
├── run_synthetic.py   tyro CLI → scenario → .rrd + summary JSON
└── visualize_run.py   open a saved .rrd

scripts/
├── imu.py              monotonicity, sample rate, gyro/accel alignment, Allan deviation
├── video_imu_timing.py LK-flow / gyro cross-correlation + rolling-shutter shear
├── camera_model.py     MEI reprojection check + Mei→Kannala-Brandt fit (writes KB calibration YAML)
├── feature_density.py  ORB feature density, track lifetime, low-feature flagging
├── ba_smoke.py         visual-only bundle adjustment on a short fisheye window
├── ba_mei_smoke.py     dual-lens MEI bundle adjustment (per-lens extrinsics)
└── track.py            VI-BA driver for any number of trackers (N ≥ 1)
```

## Probes

`scripts/` holds real-data probes that answer open questions about Insta360 X3 captures before the SLAM pipeline is built on top. Each probe is a standalone `tyro`-driven CLI that writes a Rerun `.rrd` and a JSON summary to `logs/` and prints a one-line verdict. Pass `--input-insv <path>` per run.

```
uv run python scripts/imu.py --input-insv path/to/stationary.insv
uv run python scripts/video_imu_timing.py --input-insv path/to/yaw.insv
uv run python scripts/camera_model.py --input-insv path/to/aprilgrid.insv --lens rear --tag-size 0.088 --tag-spacing 0.0264
uv run python scripts/feature_density.py --input-insv path/to/manipulation.insv
uv run python scripts/ba_smoke.py --input-insv path/to/manipulation.insv --kb-yaml logs/kb_calib_<stem>.yaml --start 2.0 --duration 10.0
uv run python scripts/ba_mei_smoke.py --input-insv path/to/manipulation.insv --start 2.0 --duration 10.0
uv run python scripts/track.py --input-insvs path/to/manipulation.insv --gripper-tag-ids 6 --window.master-start 2.0 --window.duration 10.0
```

`camera_model.py` writes a Kannala-Brandt YAML that `ba_smoke.py` consumes for single-lens visual-only BA. `ba_mei_smoke.py` swaps in the MEI projection factor and runs both fisheyes jointly using the per-lens extrinsics decoded from `offset_v3` — no separate calibration YAML needed. `track.py` builds on that by pulling IMU preintegration (`CombinedImuFactor` chain) into the graph and recovering metric scale + gravity via a VINS-Mono-style linear alignment between a short visual-only pre-pass and the IMU integrations. It accepts any number of `.insv` inputs (one tracker per input) and audio-syncs them via clap detection when `N ≥ 2`.

Findings live in `FINDINGS.md` at the repo root — the scripts produce data, the findings log captures what we concluded. Each script's docstring carries a `Status:` line (`keep` / `one-shot` / `experimental`); a `one-shot` script is removed in a commit that cites the FINDINGS entry's SHA.
