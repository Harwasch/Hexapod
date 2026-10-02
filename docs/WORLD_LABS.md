# World Labs: fit for Hexapod

Checked 2026-09-30 on branch `living-models`, from World Labs' own docs, OpenAPI spec, terms of
service (2026-01-21) and blog. Third-party benchmark claims are **UNVERIFIED**.

## What each product is

| Product                          | In                                                                                                                                                        | Out                                                                                                                            | Dynamics                                              | Access                                                           |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- | ---------------------------------------------------------------- |
| **Marble 1.0 / 1.1 / 1.1 Plus**  | text; 1 image; ≤ 4 placed or ≤ 8 "Auto Layout" images; one 360° panorama; ≤ 30 s / 100 MB video; Chisel blockout. **No splat, point-cloud or pose input** | SPZ (100k / 500k / ~2M splats), PLY, collider GLB, HQ mesh; OpenCV frame; `metric_scale_factor` **estimated**; no georeference | static                                                | app + World API                                                  |
| **World API**                    | as Marble; every generation goes through one panorama                                                                                                     | as Marble; ~5 min per world                                                                                                    | —                                                     | ~$0.18 (draft) – $1.28 (standard) per world; 3/min, 60/h default |
| **Atlas** (announced 2026-09-01) | text, 1 to more than 100 images, video, depth, **camera poses**                                                                                           | 1440p video ≤ 1 min, point clouds, 3DGS, per-frame depth; export format unannounced                                            | reconstructs rigid, articulated and deformable motion | **early access only**                                            |
| **RTFM**                         | ≥ 1 image                                                                                                                                                 | real-time video frames on one H100; no 3D                                                                                      | static scene, camera only                             | demo                                                             |
| **Spark 2.3** (MIT)              | splats                                                                                                                                                    | .RAD LOD streaming (64K-splat chunks), 16M GPU pool, 40–106M-splat demos; THREE.js                                             | —                                                     | open source                                                      |

**Reconstruction, precisely.** Marble's "Auto Layout" (`reconstruct_images: true`) is
generation conditioned on up to 8 overlapping photos, and unseen regions "are generated
plausibly". It returns no camera poses and estimates scale. **Atlas** is the product that
reconstructs from many views with poses, and it is gated.

**Corporate.** World Labs announced on 2026-09-28 that it is joining AMD (reported ~$8.2B,
all-stock, closing by end of 2026 subject to regulators). The announcement says nothing about
the future of Marble, the API or Spark.

**Terms.**

- Paid accounts own their outputs and may sublicense them, but "World Labs' rights shall take
  precedence" and attribution may be required.
- **Uploaded inputs may be used for training** (§3.6). The opt-out applies only going forward.
- No retention period is stated.
- Outputs "may not accurately represent real-world… dimensions".

## Fit per view

| View       | Role                                                                                                                                                                                                           | Verdict                              |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| **Static** | render panoramas or views from our scan at low-coverage spots → Marble → register the result (scale + ICP) into ENU → `inferred` tileset. It can't take our splat directly, and it can't be the measured layer | infill candidate, in the bake-off    |
| **Living** | Marble is static; Atlas motion reconstruction could one day supply `recorded` / `fitted-real` motion from ordinary video                                                                                       | watch Atlas                          |
| **Live**   | nothing takes in sensor streams                                                                                                                                                                                | none                                 |
| **Dream**  | pano edit ("after the storm") → a new world, as an illustrative scenario; Atlas camera-controlled video later                                                                                                  | illustrative only; route A candidate |
| **Render** | Spark could serve a close-up walk-in mode; Cesium stays for the globe (it now has hierarchical splat LOD in 3D Tiles)                                                                                          | optional                             |

Compared with the NVIDIA stack (Cosmos + Fixer + Lyra), which runs on our own GPUs under
commercial licences: Marble is faster to try and needs no GPU work. But customer imagery leaves
us, it doesn't take our splats as input, and the vendor's roadmap is uncertain.

## One-week evaluation (~$50 of credits, training opt-out enabled first)

| Day | Capture         | Test                                                                                                            | Measure                                                                                                                                |
| --- | --------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | —               | account, opt-out, a script for generate → export → metric and ground transform → OpenCV to ENU; apply for Atlas | —                                                                                                                                      |
| 2   | Spool table     | the phone video trimmed to 30 s; 8 Auto Layout frames; 4 directional frames                                     | Chamfer distance against our gsplat after scale-only alignment; scale error against the table's measured size; PSNR on held-out frames |
| 3   | Minnetonka tree | 8 drone frames; a panorama rendered from our splat                                                              | the generated back side against the real scan; error in tree height and crown width                                                    |
| 4   | Fort Clatsop    | panoramas from low-coverage spots + Expand; ICP into our tiles                                                  | seams; the S1 and S6 criteria against no fill                                                                                          |
| 5   | Fort Clatsop    | pano edit "after a winter storm" → world; Spark .RAD walk-in against Cesium LOD at 10⁷ splats                   | plausibility; frame rate                                                                                                               |
| 6–7 | —               | score; legal review of §3.3d and §3.6                                                                           | go / no-go per view                                                                                                                    |
