# Drone captures: from photos to the map

`tools/captures` turns a folder of drone photos into three representations of one place
(textured mesh, point cloud, Gaussian splat) that the console lets you switch between. This
is the pipeline the test-run sites came from.

```
photos ──▶ OpenDroneMap ──▶ 3d_tiles/model (b3dm mesh)       ──▶ build_site.py ──▶ data/tiles/<slug>/mesh
                       ├──▶ odm_georeferenced_model.laz      ──▶                ──▶ data/tiles/<slug>/pointcloud
                       └──▶ opensfm/ (cameras, sparse points)
                                  └──▶ OpenSplat ──▶ splat.ply ──▶                ──▶ data/tiles/<slug>/splat
                                                                                  └──▶ data/tiles/<slug>/site.json
                                                                                          │
                                        app.seed.publish ◀───────────────────────────────┘
                                                │
                                                └──▶ object storage  sites/<slug>/…  +  catalog.json
```

`site.json` is the capture's own registration document: one file per capture, beside the
tiles it describes. It replaced the shared `data/tiles/captures.json` manifest in A9, along
with the 104 MB of tiles that used to be committed next to it.

**Where the tiles are served from is one rule: a capture is served from your checkout
when the checkout has it, and from object storage under `sites/<slug>/` otherwise.** In
development that means `synthetic-tree` and anything you have just built come off disk over
the API's `/api/v1/tiles/` mount, and the three migrated drone captures come from the
bucket. Production never reads a checkout — the container image built from
`infra/api.Dockerfile` copies `apps/api/` and nothing else, so `data/tiles` is not in it and
never was, which is why every capture 404'd there before A9. `TILES_BASE_URL` overrides
everything, and is how you put a CDN in front of the bucket.

Either way the seeder (`apps/api/app/seed/captures.py`) creates one site per capture, with
`clipsWorld` on so the global 3D world is cut away under it. Sites show up in **Sites**
like any other, with the representation switcher and the usual quality and provenance
metadata.

## What became of `captures.json`

It is not the interface any more. A capture becomes a site by being **registered**: the
pipeline's `catalog` stage writes `registration.json` and the worker turns it into rows
(`apps/api/app/worker/registration.py`). Nothing writes a shared manifest.

Two one-shot remnants carry the old captures forward rather than orphaning them:

- **`apps/api/app/seed/legacy_captures.json`** — the four pre-pipeline captures
  (`brighton-beach`, `mygla`, `sheffield-park`, `synthetic-tree`), frozen exactly as the old
  manifest described them. Their attribution, licence, capture date, image count and
  estimated GSD are real and were not recoverable from the tiles, so they are archived
  verbatim. Nothing writes this file and a fifth capture does not go in it.
- **`data/tiles/<slug>/site.json`** — the same shape, one per capture, written by
  `build_site.py`. A locally built capture still seeds with no bucket and no publish step.

The three drone captures' tiles were deleted from `HEAD`. They remain in git history, so a
clone that needs them back can restore and publish them:

```bash
git log --diff-filter=D --format=%H -1 -- data/tiles/mygla   # the commit that removed them
git checkout <that commit>^ -- data/tiles/mygla
cd apps/api && uv run python -m app.seed.publish --slug mygla
```

`synthetic-tree` stays in the repository because it is a CI fixture with a byte-identity
gate on it, not a capture — but it is published to the bucket like the others, because the
container does not have it either.

## Running it

Prerequisites: Docker (for `opendronemap/odm`), an OpenSplat build, and the tool's Python
deps (`cd tools/captures && uv sync`, or `pip install -e .`).

```bash
# 1. ODM: mesh tiles, georeferenced point cloud, cameras.
docker run --rm -v /path/to/<dataset>:/datasets/code opendronemap/odm --project-path /datasets \
  --3d-tiles --pc-quality high --feature-quality high --mesh-size 300000 --dsm --skip-report

# 2. OpenSplat on ODM's OpenSfM reconstruction (image paths inside opensfm/ are /datasets/code/...).
ln -s /path/to/<dataset> /datasets/code
# OpenSplat only densifies until half the run by default and every 500 steps, which leaves a
# short CPU run with a few thousand gaussians; refine more often and for longer.
opensplat /path/to/<dataset>/opensfm -n 3000 --downscale-factor 4 --sh-degree 0 \
  --refine-every 100 --densify-from 500 --densify-until 2500 -o /path/to/<dataset>/splat.ply

# 3. Height offset: EXIF altitudes are not on the WGS84 ellipsoid. Sample the model's ground,
#    sample the console's terrain at the same points (browser console: Cesium.sampleTerrainMostDetailed),
#    and use the median difference.
#
#    This hand loop is still how *this* script places a capture, because build_site.py bakes
#    the offset into the tileset's own transform. It is not how the pipeline places one any
#    more: `splat_ground` measures the same cells, they travel with the registered asset as
#    `renderConfig.groundSamples`, and the viewer does this subtraction itself against
#    whatever terrain it has (apps/web/src/cesium/placement.ts). A capture that goes through
#    `tools/pipeline` needs no --height-offset at all.
python ground_samples.py /path/to/<dataset>/odm_georeferencing/odm_georeferenced_model.laz

# 4. Assemble the site folder and its site.json.
python build_site.py /path/to/<dataset> <slug> --name "…" --attribution "…" --license-name "…" \
  --source-url … --captured 2016 --splat /path/to/<dataset>/splat.ply --height-offset -2.1

# 5. Seed and look. In development this is all of it: the API serves the tiles off disk.
cd apps/api && uv run python -m app.seed

# 6. Deploying it: upload the tiles and refresh the offline catalog. Needs OBJECT_STORAGE_*.
uv run python -m app.seed.publish --slug <slug>
```

`build_site.py` does the work the raw outputs need before Cesium will place them:

- **Mesh.** ODM's b3dm tiles carry east-north-up vertices but the tileset says nothing about
  the up axis, so Cesium's glTF y-up convention lays the mesh on its side 160 m off. Each
  mesh node gets a z-up matrix, every bounding box is recomputed from the geometry, and the
  root transform is rebuilt from the OpenSfM reference point plus the height offset. Textures
  (one 16k PNG per tile, about 13 MB) are re-encoded as JPEG capped at 2048 px, which takes
  a site from ~150 MB to ~20 MB.
- **Point cloud.** The LAZ is reprojected from its UTM zone into the same local frame and
  written as a quadtree of quantized `pnts` tiles (refine ADD, each node a random sample of
  what is under it), capped at one million points.
- **Splat.** The PLY is packed as SPZ inside glTF tiles with the `KHR_gaussian_splatting`
  extensions Cesium 1.145 loads, floaters and near-transparent gaussians dropped, and the
  same frame and offset applied (`splat_tiles.py`). Every other gaussian is kept: an adaptive
  octree of at most 100k a tile (refine ADD, each parent an even subset of what is under it,
  geometric error the cell size that subset was spread at), so a small scan is one tile and a
  big one is a hierarchy the viewer draws as much of as its budget allows.

## The synthetic tree

`synthetic_tree.py` is the odd one out: it invents a capture rather than converting one. The
Living Survey deformer assigns each splat to the nearest node of a skeleton rig, and on a real
capture there is nothing to score that assignment against — nobody knows which splat _should_
belong to which branch. So the tool builds a tree whose answer is known by construction and
writes the answer down.

```bash
cd tools/captures
uv run python synthetic_tree.py ../../data/tiles/synthetic-tree --splats 12000       # committed
uv run python synthetic_tree.py ../../data/tiles/synthetic-tree-large --splats 50000 # gitignored
```

Each run writes `source/splat.ply` (the 3DGS layout `splat_tiles.py` reads), `source/rig.json`
(the `MotionRig` schema in `packages/world/src/rig.ts`), `source/labels.json` (the true node
index per splat, in PLY order), `source/positions.f32`, and a single-tile `splat/tileset.json`
built by `splat_tiles.convert` — so the app and Playwright can load the tree from disk with no
API. Both sizes share one 214-node skeleton; only the splat density differs.

Only the committed size is in the catalog. `synthetic-tree-large` is gitignored local output
with no `site.json`, so it is not seeded and therefore has no rig claim; before A9 it had a
`LIVING_RIGS` entry that could never fire because nothing seeded it either. To put it in the
console, write it a `site.json` with `"rig": "../source/rig.json"` on its splat asset.

Two properties the rest of the sprint leans on:

- **Byte-reproducible.** Seeded RNG, positions snapped to SPZ's 1/4096 m grid before anything
  is written (which also makes the decoded splats bit-identical to the PLY floats), and
  `pack_spz` already pins `gzip mtime=0`. The committed fixture is checked against a fresh run
  in CI, so it cannot go stale.
- **The checksum is a cross-language contract.** `rig.canonicalChecksum` is FNV-1a over the
  raw position bytes, computed by `checksum_positions` here and `checksumPositions` in
  `@twin/world`; the deformer refuses to move anything when the two disagree. Both sides read
  `source/checksum_vectors.json`, so a divergence fails CI instead of quietly freezing the
  tree. Note that `-0.0` and `+0.0` have different bytes: positions are normalised to `+0.0`
  because SPZ's integer round trip would otherwise change the digest without changing a value.

## Extracting a rig from a real capture

`skeleton.py` is the other half: given a 3DGS PLY of a real scene and a bounding region that
contains one tree, it isolates that tree and infers a skeleton from geometry alone, by height
banding plus connectivity clustering. A band cutting a tree crosses the trunk once and each
limb once, so the connected components _within_ a band are the separate limbs at that height;
each component becomes a node, parented to the node in a lower band whose points come nearest
to its own.

```bash
cd tools/captures
uv run python skeleton.py capture.ply ../../data/tiles/<slug> \
    --lat 44.944565 --lon -93.425903 \
    --cylinder 0 0 8          # keep splats within 8 m of the local origin
```

It writes the isolated `source/splat.ply`, `source/rig.json` and a single-tile `splat/`
tileset. Positions are snapped to SPZ's 1/4096 m grid before the PLY is written, so a real
capture's arbitrary coordinates still satisfy the checksum contract the deformer refuses on.
The rig's `sourceNote` names the file it came from and appears verbatim in the Inspector.

**What it is worth, measured.** Run it on the synthetic tree — the one input where the answer
is known — and it scores itself:

```bash
uv run python skeleton.py ../../data/tiles/synthetic-tree/source/splat.ply /tmp/rig \
    --lat 28.0389 --lon -82.6966 \
    --labels ../../data/tiles/synthetic-tree/source/labels.json \
    --truth-rig ../../data/tiles/synthetic-tree/source/rig.json
```

| metric                                                               | extracted | ceiling |
| -------------------------------------------------------------------- | --------: | ------: |
| adjusted Rand index (do splats that belong together stay together)   |     0.459 |   0.799 |
| band agreement (trunk / branch / leaf, which is what sets stiffness) |     0.601 |   0.989 |
| cluster purity                                                       |     0.597 |   0.876 |
| mean distance from a true joint to the nearest recovered one         |    0.14 m |     0 m |
| true joints recovered within 0.5 m                                   |      99 % |   100 % |

The ceiling column is the same scoring run with the _true_ rig substituted for the extracted
one, and it is not 1.0. Nearest-node assignment loses an eighth of the splats even given a
perfect skeleton, because a bark splat on the far side of a limb genuinely is nearer its
neighbour's node and three leaf sleeves on one fork genuinely overlap. Read the extracted
column against that ceiling, not against perfection: it recovers roughly seven tenths of what
nearest-node assignment can express, with a 189-node skeleton against the true 214.

**Two of these figures moved a long way in S9, in opposite directions**, when the fixture
stopped being a post with stubs and became a tree with 108 leaf clusters. Joint localisation
got much better — 0.14 m against 0.35 m, and 99 % of true joints within half a metre against
70 % — because the truth now has joints spread through the crown for the extractor's own nodes
to land near. Cluster agreement got worse — ARI 0.459 against 0.672 — because assigning a leaf
to one of 108 clusters 20 cm apart is a far harder question than assigning it to one of nine a
metre apart. The second is the honest number to quote about a real capture, and it is the one
that fell. `--max-nodes` rose from 36 to 200 to match.

Every radius in the extractor is a multiple of the cloud's own median nearest-neighbour
distance rather than a fixed number of metres. Fixed thresholds scored well on the
2,000-splat fixture they were tuned against and welded the whole canopy into a single blob on
the 50,000-splat one; density-relative radii score within 0.01 ARI of each other across that
25x change, which is the property that matters when the next input is a real scan of unknown
density.

Stiffness is interpolated from the same constants `syntheticTreeRig()` uses. It is a plausible
gradient — thicker and lower bends less — and it is **not** calibrated against how any real
tree moves. That is why the runtime labels the motion Simulated. What the runtime then does with
a rig, and what it refuses to do, is in [LIVING_SURVEY.md](LIVING_SURVEY.md).

## The real tree: what was tried, and what it would take

S6 set out to put a real scanned tree in the app. **It did not land, and the reason was
network egress, not the pipeline.** M0 (docs/LIVING_WORLD.md section 9) picked it up again on
2026-09-28 with the Hub reachable, found that **the dataset's published camera poses belong to
the wrong photos**, and built the path to train the tree anyway — solving its poses from the
photos — as a workflow that runs in GitHub Actions and on Modal
(`.github/workflows/minnetonka-tree.yml`). It has not run yet: nothing here has a GPU or the
credentials. The record is here so the run, and whoever reads its output, starts from it.

### The capture that should be used

**[Single Tree — High-Density Photogrammetry Dataset](https://github.com/Matt1Up/tree-photogrammetry-dataset)**,
Matthew Guertin, 2020. One mature deciduous tree in Minnetonka, Minnesota
(44.944565 N, 93.425903 W), flown with a DJI Mavic 2 Pro on 18 and 20 July 2020 in still air —
812 photos, 15.1 GB, of which 807 align. The low and mid tiers are hover passes at 0.5 m and
1.7 m with the gimbal angled up, which is what makes the trunk and the canopy underside
resolve rather than smear.

**Licence: CC BY 4.0**, verified three ways before anything else was attempted — the full
licence text in the repository's `LICENSE`, `license: CC-BY-4.0` in `CITATION.cff`, and
`license: cc-by-4.0` in the Hugging Face dataset card front matter. Attribution is to Matthew
Guertin (<https://mattguertin.com>).

It was chosen over the alternatives below because **the solved camera poses ship with it in
COLMAP format** (`poses/colmap/cameras.txt`, `images.txt`, and a 1.2 M-point RGB tie-point
cloud), so there would be no structure-from-motion run to pay for first. That turned out not to
hold — see the next section — and it is still the capture to use: a single tree flown up
close in still air, with the drone's own metadata in every photo, permissively licensed.

Pinned as read on 2026-09-28: Hub revision `5f9de5e4a1be429b192a928cf1359c066dadb4b3`
(`The_Tree`: 659 photos, 13,961,267,697 bytes; `colmap/points3D.txt` 58,072,943 bytes,
sha256 `f38ceae8…`), GitHub commit `942f050f7bb7fed530e61c80b3c72e5674986b48`, whose
`poses/colmap/{cameras,images}.txt` are byte-identical to the Hub's `colmap/` copies. Every
photo is checked against the LFS sha256 the Hub reports for it (the same digests as the
GitHub repository's `manifest/checksums.sha256`).

### The published poses are not these photos' poses

M0 reopened this on 2026-09-28, when `huggingface.co` was reachable, and the first thing it
did was check the poses against the photos — cheaply, without the 14 GB: every photo's
first 192 KiB holds its DJI XMP (barometric height above take-off, gimbal pitch and yaw,
capture time), which is enough to say whether a pose belongs to it. For the 655 `The_Tree`
photos (the four suspects dropped):

| check                                               | published poses         | a correct pose set |
| --------------------------------------------------- | ----------------------- | ------------------ |
| solved pitch − gimbal pitch, median / 90th pct      | 12.4° / 38.2°           | ≲ 1° / 2°          |
| up from gimbal pitch vs up from barometric height   | 11.6°                   | ≲ 1°               |
| barometric height vs height along up, RMS residual  | 1.87 m (corr. 0.41)     | ≲ 0.3 m            |
| gimbal compass vs solved heading, median residual   | 78°                     | a few degrees      |
| consecutive photos (4–8 s apart), median separation | 5.8 units (random 19.7) | small              |

Two photos looked at by eye settle it. `The_Tree-86.jpg` is taken from above the crown,
looking down; its pose (in `poses/xmp/The_Tree-86.xmp` too) is at the lowest tier looking up
8°. `The_Tree-542.jpg` is a low shot looking up at the trunk; its pose is at the top of the
orbit looking down 43°. The mismatch is in the XMP sidecars' names, upstream of the COLMAP
conversion — the converter pairs by file name, and its in-frame check passes for any camera
that looks at the middle of an orbit. The poses may well be a correct reconstruction; they
are not attached to the right photos, and a splat trained on them would be trained on ~half
its photos in the wrong place.

So the published model is not used. It is kept behind the **pose gate**
(`tools/pipeline/experiments/minnetonka.py`, `fit_frame`), which the `check` step runs on it
for anyone who wants to see the table above regenerated, and the poses are solved from the
photos by the repository's own `pose` stage instead.

### The path now: `.github/workflows/minnetonka-tree.yml`

One job per step, so each can be checked and re-run on its own. What a step keeps goes to the
private bucket under `experiments/minnetonka-tree/<tag>/`, which is how the next job (or a
later run with the same tag) picks it up. Drivers: `infra/modal/minnetonka.py` (I/O) over
`tools/pipeline/experiments/minnetonka.py` (the pure parts) and `tools/captures/real_tree.py`
(the post-train CPU step).

| step      | where              | what                                                                                                                                                                                                                                                                                                                                                                  | keeps                                                               |
| --------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `check`   | runner, no secrets | 192 KiB of each photo + the published COLMAP model (sha256-pinned) → the pose gate. ~130 MB, ~2 min. **Expected to fail.**                                                                                                                                                                                                                                            | artifact `tree-check`                                               |
| `prepare` | runner             | the Hub listing at the pinned revision checked against the pins (659 photos, 13.96 GB); the four suspect cameras not fetched; each of the other 655 (13.88 GB) fetched into memory, its size and sha256 checked against the LFS object id, its XMP read, shrunk to 1600 px (Lanczos, q95) and written as `The_Tree-0001.jpg`… — **no original ever touches the disk** | `frames/` (~0.5 GB), `meta.json`, `prepare.json`                    |
| `pose`    | Modal `cpu4`       | the `pose` stage (COLMAP 4.2 through pycolmap, photo-reconstruct's settings; sequential matching over the capture-ordered names, 15 neighbours, vocabulary-tree loop closure), then the pose gate on what it solved                                                                                                                                                   | `poses/`, `frame.json` (only if the gate passes), `pose.json`       |
| `train`   | Modal GPU (`l4`)   | the `train` stage with photo-reconstruct's Standard params — 30k-step schedule, gsplat MCMC, `cap_max: auto` (clamped to `budget_max`, 2M), `converge: true`, gsplat's default SH degree 3 — plus `blocks: 1` and an ROI of 12 m round the orbit's centre; the benchmark's path for a scene with known poses (`benchmark.cloud`, `execute_remotely`)                  | `train/trained.ply`, `train/train_metrics.json`, `train/train.json` |
| `rig`     | runner             | `real_tree.py`: into metres, the ground, the trunk measured, the tree isolated, ≤400k splats, `skeleton.py`, one tile, `site.json`                                                                                                                                                                                                                                    | `site/`; artifact `tree-site`                                       |
| `publish` | runner             | `app.seed.publish --slug minnetonka-tree --no-catalog` into the public bucket                                                                                                                                                                                                                                                                                         | `sites/minnetonka-tree/**`                                          |

**Dispatching it.** `workflow_dispatch` (inputs `steps`, `tag`, `poses`, `tier`, `budget_max`,
`roi_m`, `scale`, `crown_radius_m`, `max_splats`, `force`) works once the file is on the
default branch. Until then, as with `modal-benchmark.yml`, a commit on the working branch
carries a token: `[tree]` runs `check,prepare,pose,train,rig`; `[tree:check]` just the gate;
`[tree:rig,publish|tag=m0|scale=trunk-diameter-m=0.31]` re-rigs the kept PLY with an explicit
scale and publishes. Options after `|` are the dispatch inputs; the plan job validates every
one and refuses anything it does not know. A sensible order is `[tree:check]` (free, proves
the Hub and the pins), then `[tree:prepare,pose]` (cheap; read the gate), then
`[tree:train,rig]`, then look at the artifacts, then `[tree:publish]`.

**Cost and time** (estimates; nothing here has run on a GPU yet):

- `prepare`: ~14 GB in. Here, 51 of the photos (1.1 GB) streamed, checked and shrunk in 22 s,
  so ~5 min for all of them plus the upload of ~0.5 GB. Runner minutes only.
- `pose`: Modal `cpu4`, $0.25/h. The spool capture's 179 frames posed in 11–13 min with COLMAP
  4.2; 655 photos sequentially is ~4x the pairs and a superlinear mapper — budget 0.5–1.5 h,
  $0.15–0.40. If sequential matching registers under 70 % the stage would fill in
  exhaustively (214k pairs), which will not fit the 6-hour limit; the gate would say so first.
- `train`: L4, $0.80/h. The benchmark's reference point is ~1.5M gaussians in ~37 min at
  979×546. Here the frames are 1600×1066 (3.2× the pixels), the cap is 2M (1.3× the
  gaussians, and `converge` lets a 2M budget run up to √2 × 30k = 42k steps before its early
  stop): roughly 2.5–3.5× the per-step cost for 1.2–1.4× the steps, so **~1.5–3 h, $1.2–2.4**,
  plus a few minutes of cold start and ~0.5 GB up to the container.
- `rig`, `publish`: runner minutes.

In all, **about $1.5–3 and 2.5–5 hours** of wall time, most of it the `train` job waiting.
Its 355-minute timeout is GitHub's ceiling for a hosted job; a run that needs more wants a
smaller `budget_max` or `roi_m`, not a longer wait.

**What to check afterwards**, in order:

1. `tree-check` / `tree-prepare`: `prepare.json` — 655 photos, 13,875,363,394 bytes, one frame
   size (1600×1066).
2. The `pose` job summary: registered count, and the pose gate — every row `ok`. The gate is
   the evidence that the solved poses and the photos describe the same flight; `frame.json`
   (artifact `tree-pose`) has the residuals and the 20 worst photos.
3. The `train` summary and `train.json`: held-out PSNR/SSIM/LPIPS (every 8th frame, never
   trained on), the gaussian count, steps run and why it stopped, billed seconds and dollars.
   There is no published number to hold it to; what it is compared with is the Truck
   benchmark and the spool capture.
4. The `rig` summary and `source/real_tree.json` (artifact `tree-site`): the scale and how it
   was set, the trunk diameter and how consistent its three slabs were, the crown radius, the
   splat count, the rig's bands, and `crownToBase` ≥ 3. **Open `splat/` in the app before
   publishing**: `pnpm dev` with `TILES_BASE_URL` unset serves it from `data/tiles/`.
5. After `publish`, commit `data/tiles/minnetonka-tree/site.json` from the artifact (the
   tiles themselves never go in git; `.gitignore` keeps them out) and run `provision.yml`'s
   seed so the catalogue and `catalog.json` carry the site. Then, on real hardware: the tile
   is single, the deformer attaches (no refusal in the Living Survey panel), and the sway
   reads as _that_ tree. One development gotcha: a checkout holding only that `site.json`
   has a `data/tiles/minnetonka-tree/` directory, so the dev API serves the site from disk,
   where the tiles are not (`tiles_base_url`'s rule 2). Set `TILES_BASE_URL` to the public
   bucket's `…/sites`, or rebuild the folder locally from the kept PLY (`minnetonka.py fetch` +
   `real_tree.py`).

### Scale, up and heading: from the drone, not from GPS

The dataset's author could not get a scale from GPS (a 17 m footprint against metres of fix
error) and says so. Every photo also records its **barometric height above take-off**
(`RelativeAltitude`, 0.1 m steps) and its **gimbal pitch and yaw**, which the gimbal holds
against its own IMU and compass. From any pose set, `fit_frame` derives:

- **up**: the `u` for which `view · u = sin(gimbal pitch)` for every photo — linear least
  squares, robust to outliers; each photo's pitch is a measurement of gravity in its frame;
- **scale**: metres per model unit, the slope of barometric height against `u · centre`, one
  offset per flight (a gap of more than 2 minutes starts one), so a second take-off does not
  bend the line;
- **heading**: the rotation about `u` that best turns solved headings into compass ones;
- **origin**: on the axis the orbit looks at (least-squares intersection of the optical axes),
  z = 0 at the main flight's take-off.

Each is checked by a second measurement before anything is paid for: the up the barometer
finds on its own must agree to 3°, pitch residuals must be ≤ 3° median and ≤ 6° at the 90th
percentile, height residuals ≤ 0.6 m RMS with ≥ 90 % inliers over a ≥ 2 m span, and compass
residuals ≤ 8° median, ≤ 20° at the 90th percentile. A synthetic three-tier orbit in an
arbitrary similarity frame comes back to within 1 % in scale and 0.35 m in position; the same
orbit with half its metadata shuffled fails `pitch`, `height` and `heading`
(`tests/test_minnetonka.py`).

**On the real photos**, measured here: 51 consecutive `The_Tree` photos (470–520) streamed,
checked and shrunk to 1600 px, then the `pose` step rehearsed locally (COLMAP 3.9.1, 4 shared
cores, exhaustive — no vocabulary tree here; 20 min). It registered 30: the 7.9 m tier. The
low tier after the drone's 25-second descent did not join them, which is why the gate needs
the whole capture. Against those 30 solved poses the photos' own gimbal pitch disagrees by
**0.26° median, 0.35° at the 90th percentile**, and the compass by **0.34° / 0.65°** — an
order of magnitude inside the bars, where the published model is at 12° and 78°. The same 30
fail `heightSpan` (0.1 m of barometric height), as they should: a scale needs photos at more
than one height, and the full capture spans 1.3–7.9 m.

**The trunk is the cross-check, and the override.** `real_tree.py` measures the trunk in
three slabs 0.3–1.2 m above the lawn: a trunk is a hollow shell of splats, so its
cross-section is a ring, found by RANSAC over three-point circles and refined by least squares,
accepted only if splats cover half its circumference and it is several times denser than the
slab's splats spread evenly would be (a marker post beside the trunk, or haze, is not
mistaken for it — `tests/test_real_tree.py`). The median of the three is reported; on the
synthetic tree it comes back within 5 % of the true 0.297 m. **Nothing published gives this
tree's diameter**, so the barometric scale is the default and the trunk only checks it — a
mature street tree's trunk is tens of centimetres. If someone measures the real trunk
(a tape at the tree, or the FARO scan the dataset mentions but does not include),
`scale=trunk-diameter-m=D` rescales the whole splat about the trunk's foot so the measured
ring is `D`; `scale=metres-per-unit=X` replaces the barometric slope outright.

### The rig step

`real_tree.py` turns the trained PLY into a site folder, and writes what it measured into
`source/real_tree.json` at every step: the frame's similarity applied to every gaussian
(positions, log-scales and rotations — an ellipsoid, not a point); the ground (the densest
5 cm layer of opaque splats near the trunk — a lawn is a sheet); the trunk as above, whose
centre becomes the origin; the crown radius (98th percentile of the opaque splats above
2.5 m inside the camera ring, plus 0.3 m, or `crown_radius_m`); everything 0.2 m above the
lawn inside that cylinder, opaque, not isolated haze, and then **the packer's own floater rule
run to a fixed point**, so `splat_tiles.convert` drops nothing and the rig's checksum is over
exactly what the viewer decodes; at most 400k splats (the viewer's Standard Detail budget — a
single tile cannot be trimmed by it, so the tile must fit it), the least significant first;
**the deformer's `upright` refusal checked in Python first** (`uprightness`, a port of
`treeUprightness`); then `skeleton.extract` with its own isolation turned off, **single tile**
(`tile_gaussians=None`) because the deformer refuses anything else — M5 lifts that, and this
step will then want a level-of-detail tileset instead. `site.json` is the `build_site.py`
shape: attribution Matthew Guertin, `CC-BY-4.0`, the licence and source URLs, captured
2020-07-20, the splat asset's `ground_sample_distance_m` (median camera-to-crown distance over
the focal, at the 1600 px the splat was trained at) and `point_spacing_m`,
`"rig": "../source/rig.json"`, `clamp_to_ground: true` (the drone's altitudes are
metres-accurate at best, so the tile rests on the viewer's terrain), and a description that
says it is a photogrammetric reconstruction of a real tree and that the motion is simulated.

The splat keeps degree-0 colour only: `splat_tiles` packs SH0, as for every capture. The
degree-3 training is still what makes those colours right from every side.

### What happened in S6

The documentation, manifests and camera poses are on GitHub and were cloned here without
trouble. **The 15.1 GB of imagery, the 58 MB `points3D.txt` and the 80 MB `tiepoints.ply` are
on Hugging Face, and `huggingface.co` is a hard policy denial at this session's egress gateway**
— `CONNECT` is answered `403`, which the proxy documentation says to report rather than route
around. The same is true of every research data host that was tried: `zenodo.org`,
`figshare.com`, `data.goettingen-research-online.de` (BioDiv-3DTrees), `osf.io`,
`data.mendeley.com`, `dataverse.harvard.edu`, `opentopography.org`, `data.cyverse.org` (Open
Forest Observatory), `hub.dronedb.app` and `community.opendronemap.org`. What _was_ reachable:
`raw.githubusercontent.com`, `git clone` against `github.com`, and the Google Cloud Storage
JSON API. GitHub release assets and `codeload.github.com` were `403`; the GitHub search API is
scoped to this session's own repositories, so no discovery through it.

The Hugging Face MCP connector reaches the dataset and can read the PLY header — it is ASCII,
1,206,765 vertices with RGB — but relaying 80 MB of point data through tool results and back
out to disk is not a transfer mechanism, so that path was abandoned rather than half-run.

### Alternatives checked, and why each was rejected

| candidate                                                                         | reachable        | licence                                          | why not                                                                                                                                                                                                                           |
| --------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BioDiv-3DTrees (4,952 single-tree clouds, QSMs **and graph topology**)            | no — host `403`  | not verifiable from here                         | The QSM graphs would have been a ready-made skeleton, better than geometric extraction. Worth another attempt from a network that can reach Göttingen Research Online.                                                            |
| [AdTree](https://github.com/tudelft3d/AdTree) `data/tree1,7,13,33.xyz`            | yes, cloned      | GPL-3.0 on the repository, data licence unstated | Copyleft, not permissive; and 7k–15k points with no colour. A grey stick figure would be a poor scan passing as a survey.                                                                                                         |
| Mip-NeRF 360 `treehill` / `stump` (`storage.googleapis.com/gresearch/refraw360/`) | yes, 1.5–1.9 GB  | **no licence terms published**                   | Unlicensed is not permissive. Scene-scale rather than single-tree, and would still need GPU training.                                                                                                                             |
| ForestScan / Open Forest Observatory                                              | no — hosts `403` | not verifiable from here                         | No dataset publishing under the name ForestScan resolved to a reachable artefact; the Open Forest Observatory's data sits on CyVerse, which is blocked. Both are forest-plot scale in any case, not a single tree flown up close. |
| Committed splat assets in 3DGS viewer repositories                                | yes              | —                                                | There are none; they all fetch their demo scenes from INRIA or Hugging Face, both blocked, and the INRIA pre-trained models are research-use-only regardless.                                                                     |

### Why the captures already here cannot stand in

The obvious shortcut — pick a tree out of Sheffield Park or Brighton Beach and rig that — does
not survive contact with the numbers. Decoding the committed splat tiles gives a median
nearest-neighbour spacing of 0.16 m (Tokarzonka), 0.29 m (Brighton Beach) and 0.51 m
(Sheffield Park), over sites 160–220 m across. The densest 3 m-radius column anywhere in the
leafiest of them, Sheffield Park, holds **737 splats** — and most of that is ground. The
synthetic fixture puts 12,000 splats on one 6.3 m tree, and its large sibling 50,000.

A few hundred splats cannot carry branch structure, so the extractor would return a skeleton
of the noise. These are site captures at 1.4–2.7 cm GSD flown at altitude; a tree needs a
capture flown _for_ the tree, which is exactly what the Minnetonka set is.

### The S6 recipe, superseded

S6 left a hand recipe here: download everything, copy the published `cameras.txt` /
`images.txt` / `points3D.txt` into `sparse/0`, `mogrify` to 1600 px, run INRIA's `train.py`,
then `skeleton.py` and a hand-written `site.json`. Its step 2 would have trained on the
mismatched poses above. Every other step of it is now a job of the workflow, including the
wiring it listed — `site.json` with `captured`, `ground_sample_distance_m`, the rig claim,
attribution, licence and source URL, a description that says real reconstruction and
simulated motion, and tiles published rather than committed.

Two things only a human on real hardware can check, as before: that the splat tileset is
**single-tile** and the deformer attaches, and that the sway still reads as _that_ tree
rather than a generic sway — headless GL here is SwiftShader, and the stale draw order the
sorter cannot see needs a human eye.

## What `site.json` records

A capture's `site.json` is what the seeder builds its site from: boundary (convex hull of
the point cloud), centre, attribution and license, source URL, capture date, image count,
per-asset quality, and — for a capture that has one — the rig path. Ground sample distance is _estimated_ from the reconstructed camera
heights above the model's ground over the focal length in pixels (EXIF altitudes are often
relative to take-off) and labelled as such; the point spacing is measured from the cloud.

## Honest limits

- CPU-only OpenSplat at a quarter of the image resolution gives a coarse splat; it exists to
  exercise the format and switcher, not to compete with the mesh.
- The height offset is one number per site, so sloping sites can still sit a little above or
  below the terrain at the edges.
- Tile data no longer lives in the repository. `synthetic-tree` is the one exception, and it
  is a fixture with a byte-identity gate rather than a capture.
- **The only tree that moves in the app is the synthetic one** until
  `minnetonka-tree.yml` has run. The skeleton extractor has been scored against ground truth
  but has never been run on a real capture — see "The real tree" above for the licence
  checks, why the published poses are not used, and the workflow that finishes it.

## The test run

Three public DroneDB datasets, all DJI Phantom 3, processed on a 4-core CPU sandbox (no GPU).

| Site                          | Photos | Mesh            | Point cloud       | Splat                  | GSD (est.) | Height offset |
| ----------------------------- | -----: | --------------- | ----------------- | ---------------------- | ---------: | ------------: |
| Brighton Beach, Duluth        |     18 | 12 tiles, 22 MB | 1.0 M pts, 8.7 MB | 126k gaussians, 2.1 MB |  1.9 cm/px |        −2.1 m |
| Sheffield Park, Florida       |     32 | 12 tiles, 27 MB | 1.0 M pts, 8.7 MB | 120k gaussians, 2.0 MB |  2.7 cm/px |       +36.7 m |
| Tokarzonka reservoir, Istebna |     29 | 3 tiles, 19 MB  | 1.0 M pts, 8.7 MB | 57k gaussians, 0.9 MB  |  1.4 cm/px |        +111 m |

Timings: ODM 17–45 min per site (dense matching and meshing dominate), OpenSplat 3,000
steps 20–90 min depending on how many other jobs shared the cores, packaging under two
minutes per site.

What the run showed:

- **Meshes and point clouds are the useful outputs.** Both sit on the photorealistic world
  within a metre or two at all three sites once the height offset is applied, and the
  textured mesh is far sharper than the 3D world underneath it (2–3 cm/px against 10–20 cm).
- **CPU splats are proof of format, not of quality.** At quarter resolution and 3,000 steps
  they are blurry compared with the mesh from the same photos; a GPU run at full resolution
  with spherical harmonics is the real comparison, and that is a training-cost question,
  not a viewer one.
- **Georeferencing is the fragile step.** Two of three datasets had EXIF altitudes that were
  relative or wrong (0.4 m above take-off, 111 m off in the mountains), so the offset step
  is not optional. A tilted or partial reconstruction (Tokarzonka, flown in winter over a
  narrow dam) still lands where it should, but one number cannot fix its slope.
- **Sizes fit a repository for a test.** 30–40 MB a site with 2048 px JPEG textures, a
  million points and a splat; the raw ODM output was 150–190 MB a site.
