# Beach full data comparison

Edit-only preparation; **no Modal work was launched**. Preserve the existing `beach`
scene and the 38 test images/cameras. The new `beach-full` scene inherits their exact
pixel, intrinsic and extrinsic fingerprint. Training pools are different; evaluation is
identical. `upgrade-excl` uses the original 900×1600 images. `full-data` uses decoded
1080×1920 images, without upsampling the old images or shrinking the new ones.

## Measured locally and estimated remotely

The inclusive exclusion window is **±0.1 s** (about 3 video frames each side, the near-identical neighbours). It removes **7 of 262** original pool frames and about 11% of the full video. ±0.2 s would remove 31 of 262 (about 23% of the video) and ±0.5 s 95 of 262 (about 57%), so ±0.5 s would have tested a different, data-starved regime. The budget figures below were computed at ±0.5 s and leaving
**167**. The original test images are never training inputs. The auto budget after
cutting tracks to those 167 views is **2,519,371** splats. At native resolution the same
tracks give a **2,863,388** proxy, below the **7,247,091** L4 memory ceiling (24 GB model).
The expanded reconstruction has different tracks: freeze reports its actual budget;
training recomputes it. There is no fixed `budget_max` in either variant.

Schedule: 30,000 steps × sqrt(pool / 60) above 60 frames; below 60 the existing linear
rule and 0.25 floor apply. For this experiment the ceiling is **60,000 steps**. This
means **50,049 steps** for upgrade-excl and up to 60,000 for full-data. No internal
validation images, no PSNR early stop. Both render the untouched bench tests only after
training. No PPISP, dense initialization, priors, anisotropy penalty, far-field sphere,
pose optimization or quality gate is enabled.

Upgrade measured 4,368 s / 55,800 steps at 3.456 M splats and 900×1600. Scaling by pixels
and splat budget gives approximately **50–60 minutes / $0.80–$0.96** for upgrade-excl,
and **1.7–4.2 hours / $1.63–$4.03** for full-data (the upper end assumes the native memory
ceiling binds). These are proxies, not measured native throughput. Twenty PLY snapshots,
seed preview, dataset preparation and final evaluation are allowed for in the estimates.
CPU freeze is estimated at **0.5–2 hours / $0.13–$0.51**. Four-row rescore is estimated
at **30–60 minutes / $0.48–$0.96**, extrapolating the previous three-row call's ~24 minutes.

The commands below reserve at most **$7.74** in the bench ledger, including its startup
and refused-restart allowances: CPU freeze $0.523, baseline $1.341, full-data $4.373,
rescore $1.500. All use the same **$8 tag cap**. Each function has a hard timeout;
unknown outcomes retain reservations, and a restarted training/freeze attempt is refused.
These are the repository's compute prices, not an invoice guarantee; leave the remaining
~$0.26 for image-build/storage overhead. Do not increase a timeout or launch extra arms
against this budget. A timeout yields a failed run, not permission to spend more.

Rates checked against [Modal's pricing](https://modal.com/pricing): L4 $0.000222/s,
CPU $0.0000131/core/s, RAM $0.00000222/GiB/s. With the bench reservations this is
$0.957456/hour for L4 + 2 CPU + 8 GiB, and $0.252576/hour for 4 CPU + 8 GiB.

Actual source FPS, decoded count, selection counts and registration coverage cannot be
reported before the authorized CPU freeze. `full_data.json` records them, every frame's
PTS offset/sharpness/exposure, selection and training reason, and the recomputed forecast.

## Freeze behavior

- Locate IMG_3450.mov first under the source run, then under capture uploads if unique.
  Ambiguous filenames fail; `--source-key` can specify the exact uploaded object. R2 is
  list/get only, pinned by ETag and size. Nothing writes to R2.
- Decode every frame without an fps filter or frame cap. Reuse normalize's HDR detection
  and `video.TONEMAP_FILTER`. Parse ffmpeg `showinfo` PTS for actual timing.
- Drop a frame for relative Laplacian sharpness below 0.5 of its ±5-frame median, or
  thumbnail MAE ≤1 eight-bit code level against the last retained frame. The thumbnail
  is at most 160 px. These are recorded conservative rules, **not a tuned optimum**.
- Re-extract SIFT with the pose stage's settings: 4096 features, 1600 px, octave 0,
  four CPU threads. Those limits affect registration features, not training images.
  Match each new frame against four temporal backbone neighbours on either side.
- Re-triangulate backbone matches at their fixed poses; register new cameras with
  pycolmap 4.2 and bundle adjustment fixing every backbone pose and both camera
  calibrations. Scale native K by the image-size ratio. Fail if a backbone is lost or
  moved. Undo COLMAP's normalization from a fixed camera orientation and baseline
  (also works for planar/linear paths), check every backbone pose, then restore the
  exact original quaternion encoding.
- Keep all backbone images for pose provenance, but only registered native frames enter
  the new training pool. Apply the same inclusive time exclusion to both variants.
  Every rejected frame remains in metadata, with its reason. No top-K frame cap.

A cached pycolmap 4.2 wheel enabled a local CPU smoke test: **4/4 new images registered**
against **14 fixed beach backbone cameras**, 7,192 points, mean reprojection error
**0.677 px**, **5.76 s**. These were resized frozen photos, not decoded original frames.
Largest pre-restoration drift was 2.40e-15 scene units in camera centres and 2.23e-16
in rotation coefficients; final stored poses are exactly the originals. The guard uses
the repository's existing 1e-9 fixed-reference criterion, not a tolerance fitted to this run.
The complete source-video extraction/registration coverage remains for the CPU freeze.
Evidence: `tools/pipeline/.tmp/full-register-beach-final/result.json`.

## Commands

From the repository root, after committing and pushing:

```bash
export TMPDIR="$PWD/tools/pipeline/.tmp"
mkdir -p "$TMPDIR"
cd tools/pipeline

uv run python -m experiments.bench freeze-full \
  --scene beach-full --exclude-test-seconds 0.5 \
  --tag beach-full-data --total-cap 8 --cap 0.53

HEXAPOD_BENCH_FULL_TIMEOUT=4800 uv run python -m experiments.bench run \
  --scene beach --variants experiments/variants/beach-full.yaml \
  --variant upgrade-excl --schedule full --tier l4 --seeds 42 --live \
  --tag beach-full-data --total-cap 8 --cap 1.35

HEXAPOD_BENCH_FULL_TIMEOUT=16200 uv run python -m experiments.bench run \
  --scene beach-full --variants experiments/variants/beach-full.yaml \
  --variant full-data --schedule full --tier l4 --seeds 42 --live \
  --tag beach-full-data --total-cap 8 --cap 4.38
```

Read the freeze result's `fullData.trainingForecast` before training. Keep the two run IDs
printed by the run commands. They are generated at invocation; substitute them below:

```bash
UPGRADE_EXCL_RUN=<upgrade-excl-run-id>
FULL_DATA_RUN=<full-data-run-id>
HEXAPOD_BENCH_RESCORE_TIMEOUT=5400 uv run python -m experiments.bench rescore \
  --scene beach --ply trained \
  --runs 924f540e243a4a09bc8fa32280f5e436 28deb3c9bfba4b90afe272bb614dddf5 \
  "$UPGRADE_EXCL_RUN" "$FULL_DATA_RUN" \
  --tag beach-full-data --total-cap 8 --cap 1.51
```

This is exactly four PLY rows: A, upgrade, upgrade-excl, full-data. `--ply trained` avoids
also selecting faded exports. If additional PLY kinds are wanted later, split by PLY kind
into calls of at most four rows and make a separate budget decision; that is not included
in this $8 run. Rescore raw/aligned/cc, bottom-third, brightness and distance fields remain.

Run the view server in another terminal, from the repo root (one server per run):

```bash
export TMPDIR="$PWD/tools/pipeline/.tmp"
TAILSCALE_ADDR=$(tailscale ip -4)
cd apps/api
uv run python -m app.scripts.bench_live "$UPGRADE_EXCL_RUN" --host "$TAILSCALE_ADDR" --port 8791
# Or, for the full-data run:
uv run python -m app.scripts.bench_live "$FULL_DATA_RUN" --host "$TAILSCALE_ADDR" --port 8792
```

Supply the run-ID variables in that terminal as well. Both runs retain trained PLYs and
use the merged live console's 20 snapshots, seed points, quality series and final scores.
No README metric field names have changed.

## Production defaults

`test_every: 0` trains every frame. Set `test_every >= 2`, `converge: true` and/or
`holdout_error: true` explicitly when validation is wanted. Quality defaults to `mark`,
retaining opacity and geometry and adding `quality_tier` (0 weak, 1 context, 2 keep) to
PLYs through canonical placement. Existing support/confidence grids remain available to
the viewer; SPZ does not encode an arbitrary new per-splat property. No web edits are part
of this work. Refine keeps the preview mask as evidence and allows recovery outside it;
`restrict_preview_mask: true` restores explicit region restriction.

Packaging removes nonfinite positions/opacities and alpha below `opacity_min` (still
0.02 by default). Its 1.5×p99.5 radius deletion is gone. Counts and reasons are reported,
as are the auto budget/memory clamp, schedule ceiling, holdout and preview restriction.
The training image also checks the real pinned Dataset's all-frame adapter on CPU.
