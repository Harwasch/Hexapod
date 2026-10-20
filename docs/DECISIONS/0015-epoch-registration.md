# ADR 0015 — Epoch registration: how an epoch gets into its site's frame, and what its verdict may claim

**Status:** accepted · **Date:** 2026-10-19 · **Decision record:** D4 and the D6 amendment of the
splat-pipeline build (M3b.1)

## Context

A site is captured more than once (ADR 0013): each capture is an epoch, and every epoch must sit in
the site's one frame for the time slider to compare them. Until now an epoch was placed by its own
GNSS alone: an EXIF similarity (consumer GPS, floored at ±5 m), a phone video's one location (±10
m), or a hand-dropped pin. D4 asks for more, and for honesty about it: GNSS priors in bundle
adjustment, a **visual** alignment of a new epoch to an earlier one that does not depend on GNSS
precision, check targets that measure accuracy rather than consistency, and on every epoch a verdict
(`survey` | `coarse` | `unaligned`) the web shows as a label — "survey" for an RTK drone only, a
phone that aligns "Phone video, aligned visually, ±X m", a failure never shown as aligned.

## Decision

1. **GNSS pose priors** (`priors.py`, `colmap4.py priors`; `pose_priors: auto`, COLMAP 4.2 only):
   each frame's EXIF fix is a CARTESIAN `PosePrior` about the georeference's own origin (the site
   frame, else the fixes' median — `priors.frame_origin`, the origin `exif_gps` uses, so the two
   agree by construction). Sigma: an RTK _fixed_ solution's own std from `frames_meta.json`
   (`dji.rtk`, flag 50 only), else `gps.hAccuracyM`, else `prior_std_m`, else 5 m; vertical from the
   RTK std, else `prior_std_v_m`, else ×2 (first principles: satellites are above the horizon);
   unconstrained (1 km) when a fix has no altitude or its datum is not the frame's. COLMAP's robust
   loss at its own 7.815. `poses.json` records every sigma's source and claims a metric scale only
   when priors were used.
2. **A seeded Sim3** (`sim3.py`): every similarity this milestone fits is a numpy RANSAC/LMedS over
   3-point samples, enumerated when small and drawn from `default_rng(seed)` otherwise, ties to the
   earlier sample. Byte-identical on one machine; `exif_gps` uses it with `aligner: ransac` (the
   recipe's), `model_aligner` stays for A/B.
3. **Visual alignment, into a fixed reference** (`colmap4.py register`, `alignment.py`): the new
   epoch's frames (≤ 60 a side, sampled evenly) are matched against the reference epoch's and
   registered into the reference's model with `fix_existing_frames` and its camera constant; the
   reference model goes into this epoch's east/north/up by its own `placement.json` and the exact
   ECEF reframe between the two origins (`exif.reframe_enu`); a seeded LMedS Sim3 then maps this
   epoch's own pose model onto its registered cameras. **Not a joint re-solve of both epochs:** that
   would move epoch 1 every time epoch N arrives, and an epoch's published tiles must not change
   under it. Verified pycolmap 4.2.0 names: `PosePrior`, `data_t`, `sensor_t`, `Database.open /
write_pose_prior / clear_pose_priors / read_two_view_geometries / read_all_{images,cameras,rigs,
frames}`, `pair_id_to_image_pair`, `CameraMode.PER_FOLDER`, `Reconstruction.add_camera / add_rig
/ add_frame / add_image / register_frame`, `triangulate_points`, `IncrementalPipelineOptions.
fix_existing_frames / constant_cameras / multiple_models / use_prior_position /
use_robust_loss_on_prior_position`; the Modal image's self-check builds each one.
4. **Acceptance** (visual): a majority of sampled frames registered, ≥ 6 correspondences agree, and
   they are ≥ half. "Agree" is within 3·√6 pixel footprints (each centre one footprint per axis;
   their 3D difference, three sigma) — **not** LMedS's own threshold, which is set from the median
   and so always admits half the set. Otherwise the epoch falls back to its GNSS frame and says why.
5. **Check targets** (`check_targets.py`): clicks (`px` at an `imageSize`, or `uv`, on an original
   upload name or a kept frame) are triangulated by DLT plus Gauss-Newton after undoing
   SIMPLE_RADIAL, refused under two views or COLMAP's 1.5°, and scored horizontal, vertical (same
   datum only) and total against **three sigma of the declared uncertainties**: horizontal
   3·√2·√(anchor² + target² + pick²), vertical 3·√(…), the pick one pixel's footprint. First
   principles, not calibrated on any capture.
6. **Verdict**: `survey` = the `rtk-drone` tier, aligned, ≥ 3 check targets within tolerance, and
   an anchor that constrains height (GNSS priors whose vertical was left unconstrained by a datum
   mismatch score no vertical and cannot be survey); `coarse` = aligned otherwise; `unaligned`
   otherwise — **including an epoch whose site had a reference it could not be aligned to, whatever
   its own GNSS** ("Drone, not aligned: placed by its GPS, ±5 m"), and one whose registration
   crashed, which is never a failed run. It keeps the georeference's frame (`frameSource:
georef`). **An RTK epoch never trades its own GNSS frame for a reference that is not `survey`**
   (the review's fix 11; the default until Harrison decides otherwise): it keeps `gnss-prior-ba`
   and records the visual fit as `visualCheck`. `uncertaintyM` starts from what anchors the frame —
   √(reference² + (3·rms)²) aligned visually, the priors' sigma or the georeference's own (≥ 5 m)
   with GNSS — and check targets (their RMSE, their own sigma) can only raise it; an unaligned
   epoch carries the georeference's own. Confidence is a heuristic score in [0, 1], documented as
   not a probability.
7. **Placement**: `place` applies the alignment's frame when it aligned visually, and then never
   adds `georef.siteFrame.offsetM` (the review's fix 2: it would move a located phone epoch by its
   GPS error twice). `catalog` carries the alignment, less per-image residuals, to the API, where
   M3b.2 writes it onto `epochs.alignment`; the web reads only `label`, `verdict`, `uncertaintyM`.
8. **Recipe** `photo-reconstruct` v13: `normalize, pose, georeference, align, mask, train, …`
   (`align` on cpu4). A gate failing there fails before any L4 time; a Refine resuming at `train`
   keeps poses, georeference and alignment, which is right because the poses are unchanged. A new
   recipe input, `site_context`, is seeded by every caller (empty until M3b.2 seeds a reference).
9. **Gates are all off** (`fail_unaligned`, `max_check_target_rmse_m`,
   `fail_check_targets_out_of_tolerance`, `max_alignment_rms_m`, `max_gnss_visual_offset_m`, and
   pose's `fail_below_registered_fraction`, `max_mean_reprojection_error_px`): no recorded
   measurement backs a threshold. A `QualityGateError` is bad input and never retried.

## Measured on this machine (16 cores, pycolmap 4.2.0; the synthetic tree, not a real capture)

`experiments/registration_demo.py`, one run after the review fixes (matching and mapping vary
run to run with threads; six runs of it and six of `test_registration_epochs.py` all passed, every
epoch's camera RMSE against the truth inside the uncertainty it claimed):

| epoch           | method        | verdict   | registered | residual rms | truth RMSE | check RMSE h / v | tolerance h / v | ±       |
| --------------- | ------------- | --------- | ---------- | ------------ | ---------- | ---------------- | --------------- | ------- |
| e1 RTK drone    | gnss-prior-ba | survey    | —          | 0.026 m      | 0.015 m    | 0.021 / 0.023 m  | 0.145 / 0.106 m | 0.031 m |
| e2 RTK drone    | visual        | survey    | 30/30      | 0.013 m      | 0.018 m    | 0.010 / 0.007 m  | 0.181 / 0.131 m | 0.049 m |
| e2 phone `.mov` | visual        | coarse    | 30/30      | 0.018 m      | 0.028 m    | —                | —               | 0.062 m |
| unrelated scene | none          | unaligned | 0/30       | —            | —          | —                | —               | 10 m    |

The whole demo (five epochs, rendering included) takes 33-45 s; registering one epoch into the
reference about 3 s (extract 0.5, match 1.3, triangulate 0.1, map 0.8). The reference moved
≤ 2·10⁻¹⁴ m every time. Findings that changed the design (some from the independent review, which saw the demo test fail 3 of 6 runs before them): a sampled reference frame that no verified pair touched crashed the fixed-reference mapping, so only matched reference frames are seeded; COLMAP 4 turns EXIF GPS into WGS84 pose
priors at extraction, which broke the fixed-reference mapping, so `register` clears them (the
alignment is visual by decision); and e2's eight targets scored 1.3 cm while its reference is
worth 3 cm, hence the anchor floor in item 6.

## Consequences

- Training now runs in metric east/north/up when priors were used. gsplat scales its learning
  rates by the camera spread, so it should not matter — **unverified on a GPU** until the first
  Modal smoke (open question 3; `pose_priors: auto` ships on).
- An RTK drone epoch with no check targets is `coarse` ("accuracy unverified without ≥ 3 check
  targets"), open question 1. The reference is chosen by M3b.2 (best verdict, oldest), open
  question 2.
- A wrong declared sigma — an RTK float reported as fixed — makes every tolerance here wrong with
  it; the `rtk-drone` platform comes only from M3a's fix-flag evidence.
- Not bit-reproducible end to end: never byte-gate `epoch_alignment.json`.
- Deferred: automatic target detection (M3c, OpenCV is only in tools/captures), control targets as
  BA constraints, splat-ingest alignment (no frames; `unaligned`), vocabulary-tree pairing for large
  joint registrations (sampled to 60 a side).
