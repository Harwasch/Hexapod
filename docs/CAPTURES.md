# Drone captures: from photos to the map

`tools/captures` turns a folder of drone photos into three representations of one place
(textured mesh, point cloud, Gaussian splat) that the console serves itself and lets you
switch between. This is the pipeline the test-run sites under `data/tiles/` came from.

```
photos ──▶ OpenDroneMap ──▶ 3d_tiles/model (b3dm mesh)       ──▶ build_site.py ──▶ data/tiles/<slug>/mesh
                       ├──▶ odm_georeferenced_model.laz      ──▶                ──▶ data/tiles/<slug>/pointcloud
                       └──▶ opensfm/ (cameras, sparse points)
                                  └──▶ OpenSplat ──▶ splat.ply ──▶                ──▶ data/tiles/<slug>/splat
                                                                                  └──▶ data/tiles/captures.json
```

The API mounts `data/tiles` at `/api/v1/tiles/` and seeds one site per manifest entry
(`apps/api/app/seed/captures.py`), with `clipsWorld` on so the global 3D world is cut away
under the capture. Sites show up in **Sites** like any other, with the representation
switcher and the usual quality and provenance metadata.

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
python ground_samples.py /path/to/<dataset>/odm_georeferencing/odm_georeferenced_model.laz

# 4. Assemble the site folder and manifest entry.
python build_site.py /path/to/<dataset> <slug> --name "…" --attribution "…" --license-name "…" \
  --source-url … --captured 2016 --splat /path/to/<dataset>/splat.ply --height-offset -2.1

# 5. Seed and look.
cd apps/api && uv run python -m app.seed
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
- **Splat.** The PLY is packed as SPZ inside a single glTF tile with the
  `KHR_gaussian_splatting` extensions Cesium 1.145 loads, floaters and near-transparent
  gaussians dropped, and the same frame and offset applied (`splat_tiles.py`).

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

S6 set out to put a real scanned tree in the app. **It did not land, and the reason is
network egress, not the pipeline.** The record is here so the next attempt starts from it.

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

It suits this pipeline better than the alternatives below because **the solved camera poses
ship with it in COLMAP format** (`poses/colmap/cameras.txt`, `images.txt`, and a 1.2 M-point
RGB tie-point cloud), so there is no structure-from-motion run to pay for first.

### What actually happened

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

### The recipe to finish it on a GPU

Reconstruction here is CPU-only, and `docs/CAPTURES.md` already concludes that CPU splats are
proof of format, not of quality — 20–90 minutes per run at quarter resolution for a result
that is blurry next to the mesh from the same photos. A tree like this one, whose thin
structure is already near the resolution limit, is the worst possible subject for that. So
nothing blurry was shipped. On a machine with a GPU and unrestricted egress:

```bash
# 1. Images and poses. ~15.7 GB; --group The_Tree alone is 659 images and enough for the
#    crown, but the low and mid tiers are what make the trunk resolve.
pip install -U huggingface_hub
hf download Matt1up/tree-minnetonka-photogrammetry --repo-type dataset --local-dir tree
git clone https://github.com/Matt1Up/tree-photogrammetry-dataset tree-docs

# 2. Arrange as 3DGS expects. cameras.txt and images.txt come from the git repo,
#    points3D.txt from the Hugging Face colmap/ folder.
mkdir -p tree/sparse/0
cp tree-docs/poses/colmap/cameras.txt tree-docs/poses/colmap/images.txt tree/sparse/0/
cp tree/colmap/points3D.txt tree/sparse/0/

# 3. Downsample to ~1600 px wide. Nothing trains at 5464 px.
mogrify -path tree/images_1600 -resize 1600x tree/images/*.jpg

# 4. Train. 30k steps with spherical harmonics is the real comparison, not the 3,000-step
#    CPU run the test-run sites used.
python train.py -s tree -i images_1600 -m tree/output --iterations 30000
#    Drop the four suspect cameras first (poses/suspect_cameras.txt) — four wrong intrinsics
#    out of 807 put floaters in the scene.

# 5. Isolate the tree and build its rig. The scene frame has no verified scale (the dataset
#    says so explicitly), so measure the trunk in the cloud and scale to metres before this
#    step — the rig is in metres and the motion model assumes it.
cd tools/captures
uv run python skeleton.py tree/output/point_cloud/iteration_30000/point_cloud.ply \
    ../../data/tiles/minnetonka-tree \
    --lat 44.944565 --lon -93.425903 --height <ellipsoidal height> \
    --cylinder <east> <north> <radius>

# 6. Wire it up.
#    - add "minnetonka-tree": "../source/rig.json" to LIVING_RIGS in
#      apps/web/src/cesium/livingRigs.ts (an explicit table, deliberately not a probe)
#    - add the entry to data/tiles/captures.json, carrying "captured": "2020-07-20" and the
#      splat asset's ground_sample_distance_m — without both the Inspector reads
#      "No capture date or resolution recorded", which would undersell a real scan
#    - attribution "Matthew Guertin", license_name "CC-BY-4.0",
#      license_url https://creativecommons.org/licenses/by/4.0/,
#      source_url https://github.com/Matt1Up/tree-photogrammetry-dataset
#    - the description must say it is a photogrammetric reconstruction of a real tree and
#      that the motion is simulated
#    - if the tileset is large, gitignore it with the regeneration command beside it, the
#      way data/tiles/synthetic-tree-large already is
cd ../.. && cd apps/api && uv run pytest && uv run python -m app.seed
```

Two things to check when it renders, neither of which this machine can settle: that the splat
tileset is **single-tile** (the deformer refuses anything else, by design), and that the sway
still reads as _that_ tree rather than a generic sway — headless GL here is SwiftShader, and
the stale draw order the sorter cannot see needs a human eye on real hardware.

## What the manifest records

`data/tiles/captures.json` is the source of truth for the seeded sites: boundary (convex hull
of the point cloud), centre, attribution and license, source URL, capture date, image count,
and per-asset quality. Ground sample distance is _estimated_ from the reconstructed camera
heights above the model's ground over the focal length in pixels (EXIF altitudes are often
relative to take-off) and labelled as such; the point spacing is measured from the cloud.

## Honest limits

- CPU-only OpenSplat at a quarter of the image resolution gives a coarse splat; it exists to
  exercise the format and switcher, not to compete with the mesh.
- The height offset is one number per site, so sloping sites can still sit a little above or
  below the terrain at the edges.
- Tile data lives in the repository for the test run. Real sites belong in object storage or
  Cesium ion, registered by URL as `docs/ADDING_DATA.md` describes.
- **The only tree that moves in the app is the synthetic one.** The skeleton extractor has been
  scored against ground truth but has never been run on a real capture, because none could be
  reached from here — see "The real tree" above for the licence checks, the blocked hosts and
  the command to finish it.

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
