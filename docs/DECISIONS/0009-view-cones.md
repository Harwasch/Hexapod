# ADR 0009 — View cones: a splat is drawn from where it was seen

**Status:** accepted · **Date:** 2026-10-01 · branch `living-models`

## Context

Fort Clatsop (`phone-capture-2026-09-29`, 22.6M gaussians over ~150 m, uploaded as a trained
PLY through `splat-ingest`, so with no camera poses) was walked **inside** the fort. From the
globe's overview it looked like fog. The trees around the clearing were only seen from inside
it, and the 20–40 m canopy only from below. The trainer had also given the canopy
sky-coloured gaussians to explain the sky between its branches. The overview looks at all of
that from above and outside.

The camera-based quality stage (`quality.py`: views, spread, ground sampling distance) did
not run, because there were no poses. It also would not have caught this: the edge trees
were seen many times over a wide spread, but always **from one side**. What was missing was
the side, not the amount of evidence.

Measured from the published tiles:

- The 25th-percentile middle axis is 0.4 cm at eye level within 40 m.
- It is 7–11 cm in the canopy and 10–27 cm past 50 m.
- Detail coarsens with distance from the cameras, because a gaussian is about the size of the
  pixel it fitted.

## Decision

1. **Every scan gets a coarse grid of view cones at packaging time**
   (`tools/captures/view_cones.py`). Each cell gets an axis (from the observers to it) and a
   half-angle. The grid's longest axis is 96 cells. It is written as `viewcones.bin` (gzip
   RGBA8) and declared on the root tile's `extras.viewCones`, beside `collision.bin`.
2. **Without cameras, the finest-detail region stands in for the observers.** A cell is
   **seen from everywhere**, so no cone and no fade, when any of these hold:
   - it is near that region (2.5 cells);
   - the fine region surrounds it;
   - its detail is within 4× of the finest (the finest within 2 cells);
   - it has no judged detail.

   Only what is far coarser than the closest surfaces gets a cone. With camera poses, the
   camera centres replace the estimate (`observers=`); that is not wired yet.

3. **The globe fades a splat by its cell's cone.** It uses a new engine-patch hook,
   `GaussianSplatPrimitive.vertexVisibility`, independent of `vertexMotion`. The splat is
   drawn unchanged inside its cone and faded to nothing over 20° past the edge. Splats it
   hides are discarded before their covariance is fetched.

## Why not the alternatives

| Alternative                                      | Why not                                                                                                   |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| Delete the far and canopy gaussians when packing | They are right from inside: the immersive view would lose its surroundings                                |
| A second "overview" tileset                      | Doubles storage, and any static subset is still wrong from some side (the south edge seen from the south) |
| Direction-sector tilesets                        | Splats in different primitives are not sorted together: seams                                             |
| Generative fill of the unseen sides              | Fills, doesn't hide; it comes later (LIVING_ENGINE S1–S6) and still needs these cones to know where       |

## Consequences

- **Measured** on published tiles:
  - Fort Clatsop: 8.9 % of gaussians get a cone. The overview shows the fort in its clearing,
    and the inside views are unchanged (offline renderer).
  - Minnetonka (drone orbit): 0 %.
  - Three phone captures: 0–1.5 %.
  - Synthetic tree: 0 %.
  - Packing cost: two windowed passes, 80 MB peak on 1.5M gaussians; 70 s on Fort Clatsop's
    leaves.
- **Content can fool it.** A smooth surface next to fine detail can be several times coarser
  than its neighbours. The 4× contrast floor and the 2-cell envelope keep that from fading;
  a scan of truly mixed content could still get a cone it should not have. Every number is
  in `extras.viewCones` (`directionalShare`, `observerBox`), and `?viewCones=off` compares.
- **Where faded splats were, the globe shows what is under the clipped footprint** (flat
  imagery), not the photorealistic tiles. A footprint that follows the omni region is the
  next step.
- **Published scans need a backfill**: `splat_tiles.py viewcones <ply|tileset.json> <dir>
--tileset <dir>/tileset.json`, from the PLY or from the leaves when the PLY is gone.
