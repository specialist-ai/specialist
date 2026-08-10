# specialist

6DoF tracking of gripper-mounted Insta360 dual-fisheye + IMU cameras in an
AprilTag wall-anchor frame, for humanoid fingertip data collection.

Each gripper carries an Insta360 (two back-to-back fisheye cameras) plus an IMU,
with AprilTags rigidly mounted on the gripper body. A wall- or ceiling-mounted
anchor tag pins the world frame and metric scale. Per tracker, a GTSAM factor
graph fuses visual odometry, IMU preintegration, and tag-corner reprojection;
with multiple grippers, sightings of each other's tags tie the trackers
together. The output is globally-consistent metric trajectories and finger
joint angles.

## Install

```
uv sync
```

That is the whole environment story: the `dev` and `gpu` dependency groups are
defaults, so a plain `uv sync` (or any `uv run`) converges every box to the
full declared environment — the CUDA solve stack lands wherever its linux
markers apply, and nothing is ever installed into the venv by hand. Solves
default to `--solver.gpu require`, which aborts at the first solve when that stack
stack cannot load; `--solver.gpu auto` falls back to the CPU optimizer with a
notice. Reserve
`uv run --no-sync` for timed benchmark legs, where a run must not spend its
wall clock on dependency resolution.

Pulls `gtsam>=4.3a0` (the alpha is the only release with cp312 wheels), `numpy`,
`scipy`, `rerun-sdk`, `tyro`, `telemetry-parser`, `opencv-python`,
`pupil-apriltags`, `allantools`, `pyyaml`, plus dev tools `pytest`, `ruff`,
`mypy`, `pre-commit`.

`uv sync` also compiles the native MEI bundle-adjustment factor
(`_mei_factor_cpp`) — a C++ port of the reprojection factor that makes bundle
adjustment ~88× faster (~35% off total wall-clock) with a bit-identical optimum.
This needs a C++17 compiler, `git`, and Boost headers (`libboost-dev`, or
`brew install boost` on macOS). The build clones the matching gtsam headers into
a gitignored `.build/` cache the first time (the gtsam wheel ships no headers).

It is compiled per machine (the `.so` is architecture-specific and gitignored),
so run `uv sync` on each host. To rebuild after editing the C++ source:

```
uv run build-mei-factor
```

The extension shares pybind11 types with the gtsam wheel, so it must match that
wheel's ABI: pybind11 is pinned to the 2.x series, and `-fabi-version` plus the
libstdc++ string ABI are auto-detected from the installed wheel in `_buildext.py`
(needed because manylinux wheels are built with an older GCC than a typical host).

## Test

```
uv run pytest
```

Unit tests per `vi_ba` / `loader` module plus integration
tests (the VI-BA pipeline on a synthetic scene, IMU preintegration on a real
stationary window). Real-data assertions use tiny committed fixtures; tests that
need a full `.insv` skip cleanly when no capture is present.

## Run

`scripts/track.py` is the pipeline. One `.insv` per tracker; with `N ≥ 2` the
clips are audio-synced by envelope correlation. Base tag ids must be distinct per
tracker (cross-tracker disambiguation).

Single tracker against a wall anchor:

```
uv run python scripts/track.py \
  --input-insvs path/to/clip.insv \
  --base-tag-ids <base-tag-id> --anchor.tag-id <anchor-tag-id> \
  --window.duration None --output-dir logs/run
```

Two grippers (cross-tag multi-tracker):

```
uv run python scripts/track.py \
  --input-insvs path/to/g0.insv path/to/g1.insv \
  --base-tag-ids <g0-base-tag-id> <g1-base-tag-id> --anchor.tag-id <anchor-tag-id> \
  --left-finger-tag-ids <g0-left-finger-tag-id> <g1-left-finger-tag-id> --right-finger-tag-ids <g0-right-finger-tag-id> <g1-right-finger-tag-id> \
  --window.duration None --output-dir logs/run
```

`track.py` writes a per-keyframe trajectory JSON and a Rerun `.rrd`.
`--log-frames-in-rrd` adds the camera image panels and the gripper URDF mesh.
View a result with `uv run rerun logs/run/track_<stamp>.rrd`.

## Layout

```
src/specialist/
├── vi_ba/       the pipeline — single_tracker (per-tracker prep), pipeline +
│                features (frontend), init (incremental SfM + VI alignment),
│                build / compose (graph) + the native cpp/ MEI factor,
│                finger_kinematics + urdf_joints (gripper joint recovery), imu,
│                anchor_detection, extrinsics, optimize, landmark_cull,
│                reliability, configs, viz
└── loader/      parse_insv, mei_model, audio_sync

scripts/
├── track.py            the VI-BA driver (N ≥ 1 trackers)
├── imu.py              IMU rate / noise / Allan-deviation probe
├── video_imu_timing.py video↔IMU clock + rolling-shutter probe
```
