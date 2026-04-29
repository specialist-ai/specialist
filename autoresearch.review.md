# Autoresearch review — experiment summary + program.md improvement brainstorm

Snapshot date: 2026-05-02. Source: `autoresearch.results.md` (462 iterations,
2026-04-28 → 2026-05-01) and `program.md` (metrics + keep rule).

This file is decision-support material for revising `program.md`. It is NOT a
metric definition and is NOT read by the pipeline or the eval scripts.

---

## 1. Experiment summary

Total iterations: ~462 across 4 sessions.

| Outcome   | Count | Share |
|-----------|-------|-------|
| Reverted  | 425   | 92%   |
| Kept      | 19    | 4%    |
| Crashed   | 18    | 4%    |

Per session:

| Date       | Iters | Notable |
|------------|-------|---------|
| 2026-04-28 | 151   | First 5 kept commits; first STUCK |
| 2026-04-29 | 142   | Corrected-metric era begins |
| 2026-04-30 | 122   | Fixed-quality-bar zero_lm; DIS dense-flow integration |
| 2026-05-01 | 80    | 30+ structural attempts, 0 kept; session terminus note |

Loop-control events: 11 STUCK declarations, 3 PLATEAU notes, 3 RESUME events.

### The 19 kept changes

| #  | Change | Result after |
|----|--------|--------------|
| 1  | IMU-forward NavState init for gap kfs | kp=410.8 anc=251.9 (pre-frame-fix scale) |
| 2  | Full-track multi-view DLT | kp=380.3 anc=244.8 |
| 3  | IMU-rotation-predicted KLT initial flow | kp=417.1 anc=174.7 |
| 4  | Iterative re-triangulation | kp=394.0 anc=174.6 |
| 5  | Trailing-kf pose smoothness (σ=30cm/5°) | kp=358.6 anc=173.2 |
| 6  | Tighter trailing smoothness (σ=15cm/2.5°) | kp=330.8 anc=174.6 |
| 7  | Velocity smoothness at trailing kfs | kp=312.7 anc=173.8 |
| 8  | Tighter pose smoothness 15→10cm | kp=287.9 anc=176.6 |
| 9  | 3-pass iterative re-triangulation | kp=271.8 anc=177.2 |
| 10 | Segment-split instrumentation (no pipeline change) | first A/G/T split |
| 11 | **Keypoint-offset frame fix** (gripper-tag frame) | kp[A=23.7 G=88.5 T=166]; metric re-baselined |
| 12 | Rotation-only prep-mean trailing smoothness | kp[T=129.9] |
| 13 | Velocity-delta prep-mean smoothness | kp[T=102.7] |
| 14 | Extended prep-mean to interior gaps | kp[G=65.3 T=104.4] |
| 15 | body_T_gripper_tag averaging | kp[T=78.9] |
| 16 | 4:1 anchor-weighted DLT + min_depth 0.30m | kp[T=70.4] |
| 17 | Cross-finger constraint (user override on anc_med gate) | kp[A=22.8 G=54.7 T=41.0] |
| 18 | Fixed-quality-bar zero_lm instrumentation | zero_lm={t0:56, t1:42} new baseline |
| 19 | DIS dense optical flow (wholesale KLT swap) | zero_lm={t0:48, t1:33} |
| 20 | DIS variational iters 5→25 | **current best** |

Current best (run_1777547850):
`kp[A=23.0 G=43.1 T=49.1] anc[med=38.9 max=296.9] zero_lm={t0:48, t1:31}/113`.
Target gate: `zero_lm[t0] < 6 AND zero_lm[t1] < 6` — far from met.

(Numbering above counts 20 kept lines; the grep tally is 19 "kept/KEPT" result
lines plus the instrumentation-only entries — treat the table as the authority.)

### Arc shape

- Iters 1–9 (Apr 28): geometric cleanup, kp 384→272mm, anc 267→177mm.
- Iter 11 (Apr 28): keypoint-offset frame fix revealed the old metric inflated
  values by ~150mm (body_T_gripper_tag offset). kp_A dropped 271→24mm.
  Pre- and post-fix numbers are NOT directly comparable.
- Iters 12–17 (Apr 29): prep-mean smoothness + cross-finger constraint drove
  kp_T 166→41mm under the corrected metric.
- Iters 18–20 (Apr 30): focus shifted to zero_lm (fixed quality bar); DIS dense
  flow + variational tuning halved zero_lm and BA final error.
- Session 4 (May 1, 80 iters, 0 kept): structural exhaustion in the
  OpenCV-feature + DIS/KLT space; many near-misses blocked by the keep rule.

Keep rate 4%, but ~30 intervention categories were systematically falsified
(Tukey/Cauchy/GNC kernels, Dogleg, IPPE_SQUARE/SQPNP, ORB descriptor frontends,
IMU σ tuning post-lift, multiple smoothness scopes, NCC patch filters, etc.).

---

## 2. Five-agent brainstorm: program.md improvements

Each agent took a distinct lens and grounded proposals in journal entries.

### 2a. Keep-rule (gating logic)
- Auto-keep strict-Pareto with no minimum delta — iter 151 (FB-mean 0.7px) was
  strict-Pareto across all 6 segments, rejected only because nothing beat 5%.
- Asymmetric per-segment thresholds — kp_A ±3% (low-variance), kp_T ±10%
  (basin-volatile), derived from intra-iter LM noise.
- 3-iter exploration window for `STRUCTURE:` swaps — iter 180 KLT (kp_T −21%,
  anc_med −52%) reverted alone; its knob sweep was scored against `best:` not
  against KLT, measuring the wrong delta.
- Per-tracker zero_lm decoupling — many iters help t0 / break t1; strict
  bilateral gate is too brittle for an asymmetric frontend.
- **Top pick:** auto-keep strict Pareto with a frozen `best:` snapshot per
  structural budget.

### 2b. Metric definitions
- Drop kp_A from the gate — parked at 22.5–23.0mm structural floor
  (program.md:183-185 names it as gripper-PnP + fingertip-offset noise).
- Replace binary zero_lm==0 with a graded shortage score — many iters move
  mean lm/kf substantially without crossing the 0/1 boundary.
- Soften track_length≥3 to a weighted gradient (length-2 chains at half-weight).
- Replace anc_max with anc_p95 — single-PnP outliers dominate; iter 166
  PRESET_FAST crashed at 510mm despite anc_med=30.5.
- **Top pick:** graded zero_lm score (preserves anti-gaming, breaks the binary
  plateau).

### 2c. Search strategy
- Persistent do-not-retry ledger keyed on (knob, baseline-hash) — iter 124, 113,
  ZUPT×3 came back as no-ops already tested.
- Mandatory baseline-shift re-explore queue after each kept change — iter 333
  body_T_gripper_tag became valid only under a new baseline.
- Auto-fire wholesale rewrite after 3 reverts in one lifecycle stage —
  SuperPoint+LightGlue recommended at lines 1100/1447/1501/1512, never tried.
- Hypothesis-prediction contract with PREDICTED/SURPRISE self-grading.
- **Single biggest waste:** ~80–100 iters on OpenCV detector permutations after
  the first explicit "wholesale recommendation".

### 2d. Forbidden-list audit
- Restore IMU-σ ban with a narrow carve-out — iters 237/243/261/263 post-lift
  all reverted/crashed (~6 wasted iterations).
- Pre-authorize a wholesale PyTorch frontend with a 5-iter integration budget +
  relaxed keep rule — blocked only by protocol, not by hypothesis.
- Add explicit "no metric-definition edits" rule — _EXTREME_RADIUS_FRAC is
  duplicated between eval_metrics.py and track.py; an iter could widen it.
- Add "no new factor type without clone()" rule — iter 175 GNC crashed for this.
- **Top pick:** pre-authorize the wholesale PyTorch swap.

### 2e. Data and ground truth
- Synthetic-render clip with known-truth poses — replays canonical IMU stream
  through a synthetic MEI-fisheye renderer; gives an oracle to break basin ties.
- Re-record with 3–4 wall AprilTags — every PLATEAU note names tracker_0's 8s
  unanchored trailing motion as the binding constraint.
- Stationary-bookend recording protocol — 2s stationary at start/end so ZUPT,
  leading-V, body_T_gripper_tag averaging actually fire (currently no-ops).
- CAD-measured fingertip offset / rigid jig — scalar-distance invariant can be
  satisfied by sliding both fingertips along the same wrong axis.
- Sensitivity-analysis baseline to derive principled per-segment noise floors.
- **Top pick:** synthetic-render clip (subsumes 2/3/6/7, no field re-recording).

---

## 3. Cross-cutting themes (≥3 agents independently)

1. kp_A is at a known structural floor and shouldn't gate anymore.
2. The wholesale-PyTorch-frontend recommendation appears in 3+ terminus notes,
   blocked only by protocol.
3. The 5% gate has no principled noise basis — derive it from measured variance.
4. Single-clip benchmarking causes "no-op for this clip" rejections of sound
   mechanisms (ZUPT, leading-V, multi-tag bookkeeping).
5. anc_max is single-PnP-noise dominated and should be p90/p95.

---

## 4. Suggested minimum viable program.md revision

1. Drop kp_A from the keep rule (keep as a sanity bound only).
2. Add "Mode C — basin-flip": allow one-segment regression ≤15% when zero_lm
   wins ≥10%, with a 3-iter burn-in to claw it back.
3. Replace anc_max with anc_p95 in both the gate and the sanity check.
4. Pre-authorize one wholesale-frontend swap with a 5-iter integration window
   and relaxed keep rule.
5. Make the per-segment threshold derived from a one-time sensitivity-analysis
   run, not a flat 5%.
