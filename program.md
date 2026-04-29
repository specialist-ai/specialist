# Autoresearch — VI-BA visual frontend through rotation

## Goal

The pipeline at `src/specialist/vi_ba/` and `scripts/track.py` produces sub-cm
trajectories *only when the wall AprilTag is continuously visible*. The
moment the anchor leaves the frame for any meaningful stretch, the trajectory
blows up — see baseline below where one tracker sees the anchor in 20/113
keyframes and ends with a 19.7 m bbox.

**Make VI-BA produce sub-cm tracking through arbitrarily long anchor-blackout
windows.** This must work for any rig (single tracker, multi-tracker, with or
without scene overlap). The benchmark below is one regression test, not the
target setup.

## The one problem to solve — visual frontend collapses during rotation

**Confirmed empirically (run_1777511931 inspection):** during rotation
sweeps, the per-keyframe landmark observation count drops to literally
**zero** for tens of consecutive keyframes:

- tracker_0: 48 of 113 kfs (42%) have zero landmark obs; one block runs
  kf 65-74 (10 consecutive zero kfs in the trailing window).
- tracker_1: 40 of 113 kfs (35%) have zero; blocks at kf 5-10, 42-45.

In those windows the body pose at each kf is constrained ONLY by IMU chain
+ smoothness + occasional cross-tracker, with NO visual factors firing at
all. IMU integrates gravity-residual into phantom translation and there is
nothing to pull it back. **This is the bottleneck. Nothing else matters
until per-tracker zero-landmark-kf count is below 6/113 (~5%).**

The user has visualised this directly: when the Insta360 rotates, the
2-D image shows nearly zero landmark reprojections.

### Root cause hypothesis

KLT track chains break when frame-to-frame rotation exceeds the LK pyramid's
effective radius. Even with the IMU rotation predictor seeding KLT initial
pixel positions, fast rotation rotates features in/out of the masked /
extreme-periphery / cheirality-fail regions. When all features die, the
next kf re-detects fresh corners that have no link to the previous kf's
pool — so the new tracks need ≥3 more kfs to satisfy `min_track_length`,
but the rotation is still ongoing and they die again before reaching
length 3.

### Acceptance test for any proposed change

State in one sentence the mechanism by which more landmarks survive
rotation. For tweaks to the existing pipeline, this means naming a
lifecycle stage (Detect → KLT → FB check → Redetect → Chain →
Triangulate). For wholesale replacements, it means naming why the new
method is rotation-tolerant where the current one isn't (rotation-invariant
descriptor / affine warp model / dense flow / equivariant features /
fisheye-native, etc.). "Try X and see what happens" is not an acceptance
test.

## Resume protocol — read first

Before doing anything, read the last 30 lines of `autoresearch.results.md`.

- If the last `attempting:` line has no trailing `kept|reverted|crashed`
  line, a previous session started an experiment without recording the
  outcome. Re-parse `logs/autoresearch/run_LATEST/track_*.eval.json`, run
  `git status`, and finish that iteration before starting a new
  hypothesis.
- The journal *is* the state. There is no other memory between sessions.
- Each line: `<ISO-date> | attempted: <hypothesis> | files: <paths> |
  result: kept|reverted|crashed | kp[A=<mm> G=<mm> T=<mm>] anc[med=<mm>
  max=<mm>] zero_lm={t0:N,t1:N}/113 | <reason>`.
- The journal's first line is `best: kp[A=<mm> G=<mm> T=<mm>] anc[med=<mm>
  max=<mm>] zero_lm={t0:N,t1:N}/113` — update it on every kept change.
  "Improvement" is strictly against `best:`.

## In scope — anything in the visual frontend

The visual frontend is **completely not working** during rotation
(`zero_lm = 48/40 of 113`). Tweaking knobs inside the existing
Shi-Tomasi + KLT + IMU-rotation-seeded pipeline has had ~30 iterations
with zero movement on `zero_lm`. **Wholesale frontend replacement is
not just allowed — it is encouraged.** The current pipeline is a
default, not a contract.

### Wholesale replacement (encouraged)

Swap the entire feature → match → triangulate pipeline. The interface
the rest of VI-BA needs is per-kf, per-lens landmark observations
`(landmark_id, pixel)` with stable landmark IDs across kfs. Anything
that produces that interface is fair game. Representative candidates:

- **SuperPoint + SuperGlue / LightGlue** — learned, rotation-invariant
  keypoints with learned matching. Robust across large rotations.
- **LoFTR / DKM / RoMa** — transformer-based dense pairwise matching;
  multi-kf chains built from pairwise matches.
- **RAFT / GMFlow optical flow** — dense per-pair flow; chains by
  following flow vectors, replacing KLT entirely.
- **Fisheye-native KLT / descriptors** — Insta360 lenses are heavily
  distorted; tracking on the unit sphere instead of the rectified
  image plane may be the actual root-cause fix.
- **etc.** — DROID-SLAM frontend, direct/photometric (DSO, LSD-SLAM),
  affine-warp KLT, equivariant feature CNNs, multi-frame batched
  matching. All in scope.

A wholesale swap is multi-iteration by nature (~3–5 iters: plumb
weights → integrate into chain/triangulate → parameter shake-out →
IMU/mask integration). Each iteration must pass tests and run the
canonical command. **Keep rule is relaxed during integration:** any
iteration that increases per-kf landmark count on rotation kfs counts
as kept, even if kp/anc are flat or slightly regressed within gates.
The 6-knob budget applies *after* integration, not during.

### Smaller-scope changes within the existing pipeline

Still allowed; understand they refine a default that may be wrong. Each
counts as one structural change for the 6-knob budget:

- **Detect:** corner detector params; alternative detectors (ORB, AKAZE,
  SuperPoint as a drop-in). BRISK regressed catastrophically in
  run_1777531891 — different seed positions shifted the whole KLT chain.
- **KLT track:** window size, pyramid levels, min eigenvalue, min
  distance. IMU-rotation-predictor variants for *seeding pixel locations*
  only — NOT IMU σ tuning.
- **FB check:** forward-backward error threshold.
- **Redetect:** redetect ratio / threshold; descriptor-match recovery
  on KLT collapse.
- **Chain:** cross-kf descriptor matching across non-consecutive kfs;
  chaining logic; RANSAC thresholds; parallax filter; `min_track_length`
  variants.
- **Triangulate:** multi-view DLT acceptance; parallax / depth
  thresholds.

New visual factor types are allowed only if they demonstrably increase
landmark observation count (run_1777529809 confirmed anchor-center
reprojection is NOT this — it constrains the anchor pose, not landmarks).

## Out of scope — explicit distraction list

These ate 30+ iterations with zero progress on `zero_lm`. Forbidden, not
"lower priority":

- **Anchor PnP / anchor obs σ / anchor obs reweighting / anchor
  periphery σ multipliers** — anchor pipeline is fine when the tag is
  visible; the bottleneck is what happens between sightings.
- **IMU σ tuning** (gyro/accel noise, random walk). The 30-min
  stationary-recording calibration is suspended.
- **IMU bias as online state in the joint LM.**
- **Factor-graph composition changes** (between-pose factors,
  smoothness priors, cross-tracker σ, robust kernel arrangements).
- **LM stage modifications** (iterative re-tri pass count, multi-stage
  rewrites of the joint LM).
- **B-spline / continuous-time trajectory.**
- **SVO-style probabilistic depth filter.**
- **Tightly-coupled VIO at the visual frontend stage.**

The last three big rewrites are legitimate methods that may be
revisited *after* `zero_lm` is below 6/113, but not before — they don't
address KLT chain breakage during rotation.

## Forbidden — always

- **Metric-derived priors.** Any prior, factor, threshold, or expected
  value derived from the evaluation metric or a previous run's metric
  value. A legitimate prior is one that could be written down before
  the first run with no knowledge of any metric value (raw IMU, raw
  pixels, raw tag pixel detections, factory bias metadata, physical
  constants).
- **Cross-tracker keypoint / body-pose / velocity coupling at any
  keyframe** — the taped fingertip distance *is* `keypoint_median_mm`.
  Anything that constrains it inside the optimizer drives the metric
  to zero by construction.
- **Per-tracker `keypoint_offset` overrides; learned offsets; post-hoc
  keypoint adjustment.**
- **Anchor PnP injected into gap-exit body pose initialization.**
- **One tracker's body pose / velocity used as a prior for another.**
- **IMU bias copied from the canonical clip as a per-device default.**
  Bias prior must come from per-device factory metadata
  (`gyro_calib_numbers`).
- **`audio_sync` output used as anything other than a time offset.**
- **Clip-specific tag IDs / durations / file stems** hardcoded inside
  `src/` or `scripts/`.
- **Front-lens ORB/KLT mask.** Only `mask_rear_rects` is allowed.
- **Loop closure / map merging / place recognition.**

Cheat-baseline reference (NOT to be merged): with a kp-distance prior
(expected=22mm, σ ranging 25→1mm), `kp_T` floors at ~22mm — consistent
with body-T-gripper-tag PnP error + fingertip offset uncertainty.
Anything below 22mm without ground truth requires a tighter rigid-body
gripper model.

## Run command

The pipeline (`scripts/track.py`) writes only raw observations
(poses, landmarks, observations, tag detections, lens intrinsics).
The autoresearch metric is computed by `scripts/eval_track.py`,
which reads `track_<stamp>.json` and writes `track_<stamp>.eval.json`.
This split is load-bearing — `track.py` cannot read or game the
metric, so an iteration's "improvement" cannot loosen the rule that
defines it.

```
STAMP=$(date +%s)
RUN=logs/autoresearch/run_$STAMP
mkdir -p "$RUN"
timeout --signal=KILL 900 uv run python scripts/track.py \
  --input-insvs /Users/qiayuanl/Downloads/fingertip/VID_20260428_014500_00_003.insv \
                /Users/qiayuanl/Downloads/fingertip/VID_20260428_014501_00_003.insv \
  --gripper-tag-ids 2000 2003 \
  --left-finger-tag-ids 2002 2004 \
  --right-finger-tag-ids 2001 2005 \
  --window.duration None \
  --no-log-frames-in-rrd \
  --output-dir "$RUN" \
  > "$RUN/run.log" 2>&1
TRACK_JSON=$(ls "$RUN"/track_*.json | head -n1)
uv run python scripts/eval_track.py "$TRACK_JSON" >> "$RUN/run.log" 2>&1
```

Hard 15-min wall (`timeout`) on the BA. Exit 124 is a crash for this
iteration — revert and increment crash counter. Don't change anything
on this command.

## Metrics

### Primary: zero-landmark-kf count, against a fixed quality bar

`zero_lm` is hackable: lower `min_track_length` from 3 to 2, loosen
the parallax filter, allow duplicate landmark IDs — any of these
inflates per-kf count without representing better data.

To prevent threshold-loosening from gaming the metric, `zero_lm` is
computed against a **fixed reference quality bar that does not read
from the iteration's `Cfg`.** The bar is hardcoded in
`scripts/track.py`. A landmark counts at kf `k` iff *all* of:

- `track_length ≥ 3` (full chain, NOT iter's `min_track_length`)
- parallax across observing kfs ≥ 1° (recomputed; NOT iter's filter)
- post-triangulation reprojection RMSE < 2 px (recomputed)
- depth ∈ [0.1 m, 50 m]
- distinct landmark ID (deduped by 3D position within ε = 5 mm)

```python
zero_lm[name] = sum(1 for k in range(n) if quality_landmarks[name][k] == 0)
```

Target: per-tracker `zero_lm` below 6/113 (~5%).

Current baseline (commit `b341685`, **old direct-count definition**):
tracker_0 48/113, tracker_1 40/113. The first iteration after this
revision must dump the new quality-bar count to JSON and rebaseline. A
worse baseline number under the new bar (e.g. 90/40) is just truth, not
regression.

### Secondary: segment-split kp_*_mm and anc_*_mm

Tag-visibility class per kf uses **`per_kf_anchor_confidently_visible_by_tracker`**
(NOT the raw `per_kf_anchor_visible_by_tracker`, which the visualizer
keeps for display). The "confident" signal is True only when the
AprilTag is detected at a kf AND its corner-centroid radius is within
`_EXTREME_RADIUS_FRAC = 0.65` of the fisheye optical edge
(`min(h, w) / 2`) — peripheral fisheye PnP is unreliable enough that
revisits at the edge produce body-pose-vs-PnP residuals dominated by
PnP noise rather than IMU+landmark drift, so a tag at the edge should
not be treated as ground truth at the metric level. Threshold lives
in `src/specialist/vi_ba/eval_metrics.py` (with a private duplicate
in `scripts/track.py` that the BA uses to inflate anchor σ at the
same radius — duplicates must be kept in sync by hand).

Class per kf, across both trackers:

- **A (anchored):** every tracker has confident anchor at `k`.
- **G (gap, interior):** at least one tracker has no confident anchor
  at `k`, but every tracker has at least one confident-anchor kf at
  index `> k`.
- **T (trailing):** at least one tracker has no confident-anchor kf at
  any index `≥ k`.

For each segment S ∈ {A,G,T} and each tracker pair, then `max` across
pairs:

```python
kp_S_mm = max(median(per_kf_keypoint_distance[pair][k] for k in seg(S))
              for pair in pairs) * 1000
```

Anchor revisit (per-tracker, all gaps). The gap_drift_by_tracker
entries are themselves filtered by the confidence gate: a gap exit
only counts if the tag at the revisit kf is at a confident image
position. Peripheral revisits never enter the metric.

```python
all_drifts = [d["drift"] for ts in J["gap_drift_by_tracker"].values() for d in ts]
anc_med_mm = median(all_drifts) * 1000
anc_max_mm = max(all_drifts) * 1000
```

### Keep rule (asymmetric)

A change is **kept** iff *all* of:

- Sanity: `cross_obs >= 2`, `final_error <= initial_error`, every tracker
  has at least one gap, `kp_*_mm < 1000`, `anc_max_mm < 500` (else crash).
- One of the two improvement modes below.

**Mode A — zero_lm-led (strict kp/anc gate):** `zero_lm[t0]` or
`zero_lm[t1]` reduces by ≥5% vs `best`, AND **no segment in {kp_A_mm,
kp_G_mm, kp_T_mm, anc_med_mm} regresses at all** (`m ≤ best.m`). If
the new "landmarks" were noise, the LM would absorb them as kp_G/kp_T
degradation; the strict gate forces them to carry information.

**Mode B — kp/anc-led (5% gate):** `zero_lm` is within ±5% of `best`,
some segment beats `best` by ≥5%, no other segment regresses by more
than 5%.

`anc_max_mm` is logged but is not a hard gate — it can spike with the
trailing tail.

Otherwise: revert.

### Loading metrics

```python
import json
# Metrics live in the eval JSON, not the raw track JSON.
M = json.loads(open(f"{RUN}/track_<stamp>.eval.json").read())
# confident-anchor signal drives kp segment classification + anc_med/max
anchor_vis = M["per_kf_anchor_confidently_visible_by_tracker"]   # {name: [bool]}
landmarks  = M["per_kf_landmark_count_by_tracker"]   # {name: [int]}, post-quality-bar
kp_pair    = M["per_kf_keypoint_distance_by_pair"]   # {"a↔b": [float]}
gap_drift  = M["gap_drift_by_tracker"]               # {name: [{drift, gap_kfs, gap_t}]}
n = min(len(v) for v in anchor_vis.values())
zero_lm = {name: sum(1 for c in landmarks[name] if c == 0) for name in landmarks}
```

## The loop

Every iteration is one fresh Claude Code session. Steps:

1. Read this file. Read `autoresearch.results.md`. Run `git log --oneline -20`.
2. Form one specific hypothesis that passes the acceptance test above.
   Append `<ISO-date> | attempting: <one sentence> | …` to the journal.
3. Edit code under `src/specialist/` or `scripts/track.py` (no other
   directories; no edits to `program.md` or `tests/`).
4. `uv run pytest tests/ -q`. If fail, fix the change so tests pass.
   Don't disable tests. Don't `--no-verify`.
5. Run the canonical command above. Wait for it to exit.
6. Parse metrics from `$RUN/track_*.eval.json` per the keep rule.
   - **Keep:** `uv run ruff check --fix src scripts`,
     `uv run mypy src/specialist scripts/track.py`. If mypy fails →
     revert as a regression. Otherwise commit:
     `Add <change> with kp[A=<X> G=<Y> T=<Z>]mm anc[med=<M> max=<X>]mm
     zero_lm={t0:N,t1:N}/113`. Update the `best:` line. Append
     `result: kept …` to the `attempting:` line.
   - **Revert:** `git reset --hard HEAD && git clean -fd src scripts`.
     Append `result: reverted …`.
   - **Crash** (timeout 124, no JSON, NaN, sanity-gate fail,
     `final_error > initial_error`): revert, append `result: crashed …`.
7. Goto 1.

## Iteration budget pattern

**Every iteration must propose a structural change first.** Hyperparameter
tuning is slack — 6 knob iterations to refine the most recent structural
change. You cannot enter knob-tuning mode without a fresh structural change
preceding it. Once a structure's 6-knob budget is exhausted (kept or not),
the next iteration *must* introduce a new structure.

A structural change introduces a new lifecycle stage variant, a new
detector, a new matching strategy, a new chain-recovery strategy, **or a
wholesale frontend replacement**. Tuning σ / threshold / count / window
of an existing stage counts against the 6-iteration budget for the
structure that introduced it.

## NEVER STOP

Prime directive: **never stop iterating**. Do not declare STUCK. Do not
say "we've reached a deep local minimum." The only exits are the stop
conditions below.

When a structure exhausts its 6-iteration budget without a kept change,
**immediately propose a new structural change within the visual frontend**.
After two consecutive structural changes inside the existing
Shi-Tomasi+KLT pipeline have failed, **escalate to a wholesale
replacement** — do not keep tweaking the same default. The existing
pipeline is "completely not working" during rotation; default loyalty
is the wrong instinct.

Returning to the user with "no further improvement found" is a violation
of this directive. The user does not need status updates between
iterations; they need the next attempt.

## Stop conditions

- `zero_lm[t0] < 6 AND zero_lm[t1] < 6` AND `keypoint_median_mm < 10`
  AND `anchor_revisit_max_mm < 50` → write `SUCCESS: …` to the journal,
  exit 0.
- 3 consecutive crashes → write `BLOCKED: …`, exit 0.
- Repository in an unrecoverable state (rebase conflict, missing data
  file, hooks broken) → write `BLOCKED: …`, exit 0.

The loop runs forever otherwise — there is no STUCK exit and no
human-review gate.

## Facts

- Canonical clip pair is ~17 s, 113 keyframes per tracker, ~6 min wall
  time per iteration with frames disabled.
- `body_T_gripper_tag` is the rigid SE(3) from each tracker's body
  frame to its body-rigid AprilTag, recovered once during prep.
- `mask_rear_rects` blacks out the body-tag pixels in the rear lens
  before ORB so visual tracks aren't dominated by the body-rigid tag.
- Kept changes are individual commits on `master`. Browse them with
  `git log --grep='zero_lm='`.
- One iteration = one `track.py` run start-to-finish. No parallel
  edits while the run is in flight.
