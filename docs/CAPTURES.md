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

## What the manifest records

`data/tiles/captures.json` is the source of truth for the seeded sites: boundary (convex hull
of the point cloud), centre, attribution and license, source URL, capture date, image count,
and per-asset quality. Ground sample distance is *estimated* from the reconstructed camera
heights above the model's ground over the focal length in pixels (EXIF altitudes are often
relative to take-off) and labelled as such; the point spacing is measured from the cloud.

## Honest limits

- CPU-only OpenSplat at a quarter of the image resolution gives a coarse splat; it exists to
  exercise the format and switcher, not to compete with the mesh.
- The height offset is one number per site, so sloping sites can still sit a little above or
  below the terrain at the edges.
- Tile data lives in the repository for the test run. Real sites belong in object storage or
  Cesium ion, registered by URL as `docs/ADDING_DATA.md` describes.

## The test run

Three public DroneDB datasets, all DJI Phantom 3, processed on a 4-core CPU sandbox (no GPU).

| Site | Photos | Mesh | Point cloud | Splat | GSD (est.) | Height offset |
| --- | ---: | --- | --- | --- | ---: | ---: |
| Brighton Beach, Duluth | 18 | 12 tiles, 22 MB | 1.0 M pts, 8.7 MB | 126k gaussians, 2.1 MB | 1.9 cm/px | −2.1 m |
| Sheffield Park, Florida | 32 | 12 tiles, 27 MB | 1.0 M pts, 8.7 MB | 120k gaussians, 2.0 MB | 2.7 cm/px | +36.7 m |
| Tokarzonka reservoir, Istebna | 29 | 3 tiles, 19 MB | 1.0 M pts, 8.7 MB | 57k gaussians, 0.9 MB | 1.4 cm/px | +111 m |

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
