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
opensplat /path/to/<dataset>/opensfm -n 3000 --downscale-factor 4 --sh-degree 0 -o /path/to/<dataset>/splat.ply

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
and per-asset quality. Ground sample distance is *estimated* from the cameras' EXIF (height
above the model's ground over the focal length in pixels) and labelled as such; the point
spacing is measured from the cloud.

## Honest limits

- CPU-only OpenSplat at a quarter of the image resolution gives a coarse splat; it exists to
  exercise the format and switcher, not to compete with the mesh.
- The height offset is one number per site, so sloping sites can still sit a little above or
  below the terrain at the edges.
- Tile data lives in the repository for the test run. Real sites belong in object storage or
  Cesium ion, registered by URL as `docs/ADDING_DATA.md` describes.
