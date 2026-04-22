# Findings

Running log of what each probe run actually concluded. Entries are appended in
the same commit that produced the result. Use `git log -- FINDINGS.md` to
recover the commit SHA.

---

### 2026-04-21 — imu.py
Capture: VID_20260421_205659_00_001_5min_stationary.insv (304 s on a table)
- 304,092 IMU samples at 1 kHz (median Δt exactly 1 ms; rate jitter 1.35%; max gap 1.19 ms). No gaps. Gyro/accel share one per-sample timestamp axis — alignment is automatic.
- Stationary span covers essentially the full capture (287 k samples). Per-axis gyro σ = (0.060, 0.078, 0.061) °/s; accel σ = (0.014, 0.013, 0.013) m/s² — sensible MEMS-grade noise floors.
- Overlapping Allan deviation runs τ = 2 ms → 75 s. Short-τ slope is the expected −½ (white noise). White-noise density ≈ 2.4–2.8 × 10⁻³ °/s/√Hz per axis; bias instability floor 5 × 10⁻⁴ to 1.9 × 10⁻³ °/s between τ = 20–60 s.
- **Verdict:** green — telemetry-parser output usable for VIO; IMU noise sane.
- Script status: keep.

### 2026-04-21 — video_imu_timing.py
Captures:
- VID_20260421_210237_00_002_yaw_flick.insv (front lens, 400 frames)
- VID_20260421_210305_00_003_yaw_smooth.insv (front lens, 600 frames)
- Flow/gyro cross-correlation peaks at 0.88 on yaw_flick, **0.99 on yaw_smooth**; estimated video↔IMU offset 5–10 ms, inside the sub-50-ms bound the firmware doc claims. Video and IMU clocks are aligned for SLAM purposes.
- Rolling-shutter shear measurement: yaw_flick yields 35.6 ms (13 frames used) vs. factory 15.73 ms — a consistent ~2× bias; yaw_smooth yields no measurement because the front-lens scene has almost no straight vertical edges (the fisheye bends them into curves). The ~2× bias is an estimator artefact: a fisheye's deg/pixel varies radically across the FOV and the current code uses a constant approximation. The factory value is not refuted, but it is not confirmed either.
- **Verdict:** green on video↔IMU clock alignment; open on rolling-shutter readout — estimator needs to rectify before the shear → readout math holds. Next step: apply `mei_unproject` to the line segments so the shear is measured in angular space, not pixel space.
- Script status: keep (clock check); RS estimator needs a follow-up that uses the camera model.

### 2026-04-22 — camera_model.py
Capture: VID_20260421_214108_00_006_aprilgrid.insv (rear lens, frame 1500), grid 6×6, tag size 8.8 cm, spacing 2.64 cm
- 92 corners detected (≈23 tags of 36). The detection survives severe fisheye distortion at the tag periphery, confirming `tag36h11` is the right family and `pupil_apriltags` works on raw fisheye frames.
- **Corner-order bug found and fixed.** `pupil_apriltags` returns image-frame corners going CCW from lower-left; pairing detector index 0 with our world-frame TL (not BL as I originally wrote) drops mean reprojection from 28 px → **7.9 px**, max from 49 px → 18.9 px. The rear-lens side-by-side packing introduces the chirality flip that makes TL the right target for index 0; documented in the `_build_grid_3d` docstring.
- MEI reprojection (after fix): mean 7.9 px, max 18.9 px, periphery 10.2 px — factory intrinsics are directionally right but carry ~8 px residual. Remaining error is likely a mix of: (a) the factory `camera_matrix`'s fx≈2·fy aspect, which may need per-axis rescaling for the 1920×1920 per-lens crop (A/B test of fx halved or fy doubled doesn't fix it alone, so it's not a clean scalar issue), (b) corner-detection noise at the heavily distorted periphery.
- KB fit (samples rays, projects through MEI, fits Kannala-Brandt): converges; residual 1.88 px under 70° half-angle, 3.40 px under 90°, 3.95 px overall. KB can represent the factory MEI to ~2 px across the useful fisheye half-angle, so **KB is not the bottleneck** for downstream VIO.
- **Verdict:** yellow — factory MEI is close enough for a VIO smoke test but not tight enough for high-precision work. KB YAML is written and usable; treat the ~8 px intrinsic residual as a known caveat and revisit once we have a Kalibr-refined calibration.
- Script status: keep.
- Output: `logs/kb_calib_VID_20260421_214108_00_006_aprilgrid.yaml`.

### 2026-04-22 — ba_smoke.py
Capture: VID_20260421_210650_00_004_mani_example.insv (front lens, 2-12 s window, 200 keyframes at 1-per-3 frames)
- ORB-2000 + BFMatcher/ratio-0.7 + essential-matrix RANSAC on KB-unprojected bearings chained **27,875 tracks**; **11,467 survived triangulation + cheirality**. Mean track length ≈ 5 frames.
- Two-view init on the widest feasible keyframe pair (0→6, 65 shared tracks) produced rotation + unit-scale translation. Tracks die out after ~6 keyframes (~0.3 s) on this fast-motion manipulation capture, so middle keyframes with 0-1 observations rely on **weak priors (σ = 0.3 m / ~17°) around their linearly-interpolated init** to stay eliminable — without these, GTSAM fails with "Leftover keys after elimination".
- Levenberg-Marquardt converged in **12 iterations**; graph error 464,776 → 88,445 (5.3×). Final reprojection **RMSE 2.79 px** across the triangulated landmarks.
- Up-to-scale trajectory length **1.23 m** (gauge = init-pair baseline); the Rerun visualization traces gripper motion consistent with the hand-held recording.
- **Verdict:** green — visual-only BA on raw X3 fisheye works end-to-end with GTSAM's `Cal3Fisheye` and the KB calibration `camera_model.py` produced. The bottleneck is feature lifetime, not geometry: this is exactly why the IMU is load-bearing for the VI-BA we're about to build — preintegration factors will replace the weak priors that are currently papering over unconstrained keyframes.
- Next: add `gtsam.ImuFactor` preintegration between keyframes; swap `Cal3Fisheye` for a custom MEI projection factor so we stop absorbing the ~2 px KB-vs-MEI fit residual.
- Script status: keep.

### 2026-04-22 — feature_density.py
Capture: VID_20260421_210650_00_004_mani_example.insv (front lens, 600 frames)
- ORB saturates every configured cap I tried: default 1500 across all 600 frames (p10 = p90 = 1500), 5000 across 300 frames, 20,000 across 200 frames. The scene is extremely feature-rich — the manipulation setup (cluttered lab with equipment, textured floor, people) has no feature-starved frames.
- 0 % of frames below the 80-feature floor.
- Mean track lifetime ≈ 4.4–4.6 frames at 60 fps (~75 ms). Short-ish, but measured via conservative BFMatch with ratio 0.75 — real VIO uses geometric gating that retains longer tracks. Absolute number is informative for low-texture scenes; for this scene class it is a floor, not a ceiling.
- **Verdict:** green — feature density is not a risk for VIO on manipulation scenes. The live question is tracking stability during fast gripper motion, which this probe cannot answer — that belongs to the VI-BA.
- Script status: keep.
