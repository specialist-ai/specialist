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
`scipy`, `rerun-sdk`, `av`, `tyro`, `telemetry-parser`, `opencv-python`,
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

A session goes through three steps. The raw captures convert once into a base
recording; every solve reads that recording and writes a track layer beside
it; registration puts both on a catalog. Records are laid out layer-first —
one directory per layer, one file per session — so a whole layer registers
with one glob and any layer can be refetched or regenerated without touching
the rest:

```
base/<session-id>.rrd    # remuxed per-lens video, IMU, calibration, offsets
track/<session-id>.rrd   # the solve: trajectories, landmarks, joint angles
mocap/<session-id>.rrd   # ground-truth layer (eval branch)
```

**1. Convert.** One `_00_` master `.insv` per tracker (its `_10_` rear sibling
is found automatically). The first capture's clock is the session clock; the
rest are audio-synced onto it, and the recording is trimmed to the span every
capture covers.

```
uv run python scripts/convert.py \
  --input-insvs path/to/g0_00.insv path/to/g1_00.insv \
  --session-id <session-id> --tracker-names tracker_0 tracker_1 \
  --output-dir logs
```

This writes `logs/base/<session-id>.rrd`: the remuxed per-lens video with its
real PTS, the native-rate IMU, the calibration, and the audio offsets. Pass
`--audio-offsets` to supply the shifts by hand when a take's clips share no
audible event.

**2. Solve.** `track.py` reads the base recording — it carries the tracker
names, the session window, and the session id, so none of them are passed
again. Base tag ids must be distinct per tracker (cross-tracker
disambiguation); the first `--anchor.tag-ids` entry is the reference anchor
that pins the world gauge.

```
uv run python scripts/track.py --base logs/base/<session-id>.rrd \
  --anchor.tag-ids <anchor-tag-id> --base-tag-ids <g0-base-tag-id> <g1-base-tag-id> \
  --left-finger-tag-ids <g0-left> <g1-left> \
  --right-finger-tag-ids <g0-right> <g1-right> \
  --output-dir logs
```

A single tracker is the same command with one id per list and no cross-tracker
factors. `--keyframe-every` overrides the automatic ~0.10 s keyframe stride,
and `--no-open-viewer` keeps a headless box from spawning one.

The run writes `logs/track/<session-id>.rrd`, a track layer sharing the base
recording's id: trajectories, keyframe poses, landmarks, finger joint angles,
the solved anchor poses, and the run summary. Re-running a session refuses to
overwrite its track unless `--overwrite` is passed, so point experimental runs
at their own `--output-dir`. It spawns a viewer on the way out; to re-open a layer later,
load it beside its base — the shared id merges the two — with
`uv run rerun logs/base/<session-id>.rrd logs/track/<session-id>.rrd`.

**3. Register** (optional) the pair on a Rerun catalog, which installs the
standard dashboard as the dataset default. A layer whose recording id matches
no base segment is refused; a layer attaches only to the base segment that
shares its recording id.

```
uv run python scripts/register.py --catalog-url rerun+http://host:port \
  --dataset <dataset> --root logs
```

## Layout

```
src/specialist/
├── vi_ba/       the solve — single_tracker (per-tracker prep), tracks +
│                features + gpu_flow (frontend), init (incremental SfM + VI
│                alignment), build / compose (graph) + the native cpp/ MEI
│                factor, optimize + gpu_lm (LM), reference_solve (the
│                multi-tracker world gauge), finger_kinematics + urdf_joints +
│                gripper_opening (gripper joint recovery), imu,
│                anchor_detection, extrinsics, landmark_cull, highrate,
│                geometry, configs, summary, viz + viz_build
├── record/      the rrd data layer — schema (entity paths), base + video
│                (session recording), layer (track), session + frames +
│                read (reading them back)
└── loader/      parse_insv, mei_model, capture, audio_sync

scripts/
├── convert.py          captures → one session base recording
├── track.py            the VI-BA driver (N ≥ 1 trackers) → track layer
├── register.py         a layer-first corpus root → a Rerun catalog dataset
├── imu.py              IMU rate / noise / Allan-deviation probe
├── video_imu_timing.py video↔IMU clock + rolling-shutter probe
```
