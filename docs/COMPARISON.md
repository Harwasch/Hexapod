# Comparing reality-model formats

Four public sites, one app, so the look, feel and cost of each 3D format can be compared
side by side. All are Cesium ion sample assets that the CesiumJS evaluation token can
reach; attribution and terms are stored on each catalog record and shown in the About sheet.

| Site (Sites panel)                     | Format                         | Detail                | ion asset | Terms                            |
| -------------------------------------- | ------------------------------ | --------------------- | --------- | -------------------------------- |
| Melbourne: mesh vs point cloud         | Photogrammetry mesh            | 7.5 cm GSD, 2018      | 69380     | CC BY 4.0, City of Melbourne     |
|                                        | LiDAR point cloud              | 25 cm, ~300 M pts     | 43978     | CC BY 4.0, City of Melbourne     |
| San Francisco: high-resolution mesh    | Helicopter photogrammetry mesh | 2 to 5 cm + street    | 1415196   | Aerometrex, non-commercial trial |
| AGI HQ: close-range drone mesh         | Drone photogrammetry mesh      | cm-class, 40 MB       | 40866     | Cesium sample asset              |
| Chappes church: terrestrial laser scan | Ground LiDAR point cloud       | 1 cm precision, 37 MB | 16421     | Columbia Robotics Lab via Cesium |
| Boathouse: close-range Gaussian splat  | Gaussian splat                 | close range, 14 MB    | 3667783   | Cesium sample asset              |
| Cesium Gaussian splat demo (Redmond)   | Gaussian splat with LOD        | aerial campus, 2 GB   | 4547222   | Cesium sample asset              |

Google Photorealistic 3D Tiles is available as the world layer (Settings › World, feature
flag `VITE_ENABLE_PHOTOREALISTIC`, needs an ion token with Google access) for a global
mesh baseline.

## Resolution ceiling

None of the public samples reach millimetre resolution. The finest are the Aerometrex street
sections (about 2 cm), the Chappes scan (1 cm coordinate precision) and the Boathouse splat
(centimetre-class). Millimetre data comes from terrestrial scanners or close-range
photogrammetry of your own; upload it to Cesium ion (Add data › ion reconstruction) and it
appears here like any other site.

## Millimetre test object (no scanning needed)

`apps/web/scripts/fetch-sample-object.mjs` downloads a CC0 photoscanned object from Poly
Haven, packs it into a single-tile 3D Tiles tileset under `apps/web/public/samples/`
(gitignored, served by the Vite dev server), and registers it as a site through the API.
The viewer clamps it to the terrain, so only a longitude and latitude are needed:

```bash
pnpm samples          # rock_09 and weed_plant_02 next to each other on the Redmond demo site
# or one at a time, anywhere:
cd apps/web && node scripts/fetch-sample-object.mjs --slug potted_plant_04 --lon -122.1380 --lat 47.6448 --res 8k
```

Finding them afterwards: press `S` for the Sites panel and click the sample, or type
`fly to rock` / `fly to weed` in the command bar. Both sit a few metres apart, so once you
are at one, the other is in view.

`rock_09` is a 14 cm rock with 4K textures, about 0.035 mm per texel; `weed_plant_02` is a
2 m weed. Any Poly Haven model slug works (`https://polyhaven.com/models`). A phone scan
exported as glTF can be dropped into the same folder layout by hand: `model.glb` plus the
generated `tileset.json`, with the site registered through Add data.

While a model smaller than 30 m in radius is active the camera's zoom floor drops from
0.6 m to 5 mm and the near plane from 1 m to 1 cm, and a fly-to arrives at a few times the
object's radius. The scale readout in the command bar switches to mm/px below 1 cm.

## What to compare

- **Feel while dragging.** Open the Developer panel (`D`). "Motion fps" is the mean frame
  rate and 95th-percentile frame time measured only while the camera moves, reset every
  time a site becomes active. Drag for a few seconds at a similar altitude on each site.
- **Detail at rest.** Still frames render at native pixels with MSAA; zoom to a few metres
  and compare how much real detail each format holds. Ground / px in the panel is the
  scale.
- **Memory.** "Tileset memory" against the cache budget; the adaptive controller raises the
  screen-space error when a format overruns it.
- **Switching representation.** On Melbourne, the Splat / Mesh / Points switcher swaps
  the mesh and the point cloud without moving the camera.

## What to expect

- **Meshes** (Melbourne, San Francisco) are opaque triangles streamed by LOD: no per-frame
  CPU work, so they feel like Google Earth. Terrain is clipped under them using the tile
  coverage, so no z-fighting.
- **Point clouds** are drawn with attenuation and eye-dome lighting over the terrain; cost
  scales with visible points, gaps show the ground beneath.
- **Gaussian splats** are re-sorted on the CPU every camera change and alpha-blended, so
  their cost grows with splat count far faster than a mesh. They keep the terrain under
  them and are capped at a per-preset minimum screen-space error. The Boathouse shows a
  splat at its intended close range; the Redmond campus is an aerial capture and looks soft
  when you zoom beneath its native resolution.

Seeding: `cd apps/api && uv run python -m app.seed` adds the comparison sites to an
existing database (idempotent by slug).
