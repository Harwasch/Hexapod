# ADR 0010 — The Time Lantern: a cut in the world, the globe under it, dated imagery on it

**Status:** accepted · **Date:** 2026-10-16

## Context

The photorealistic world (Google Photorealistic 3D Tiles) is the ground in the default view, and
it has no date: Google does not publish when its captures were made. Public dated imagery of the
same ground exists -- NAIP aerial photographs every two years or so, Sentinel-2 scenes, USGS
topographic maps going back a century -- and the useful question on rangeland is "what was here
before". The Time Lantern answers it under the cursor: a circle in which the dated imagery shows
instead of today's 3D world, stepped through by year.

In the photorealistic world the globe (terrain and imagery) is drawn only inside site footprints,
by an inverse clipping polygon (ADR 0003), so imagery layers on the globe are invisible there
unless the globe is drawn where the lens is.

## Decision

Three engine pieces, all inside the scene (`apps/web/src/cesium`):

1. **A screen-space cut on the world tileset**, by the tileset's one `CustomShader`
   (`worldShader.ts`). Fragments inside the lens are discarded; in a feather band they are
   discarded with a probability falling to zero, chosen by interleaved gradient noise. `WorldShader`
   is the single owner of that shader: anything else that shades the Google world adds its
   uniforms there. It is attached on the first lens (one recompile of every tile's shader) and
   stays attached; a closed lens is a uniform.
2. **A globe-only ground window** (`ClippingManager.setLanternWindow`): a ring of ground points
   around the lens, found by casting rays through a circle slightly larger than the lens and
   meeting them with the terrain the globe has loaded. It is swapped as the lens moves. A reach
   rectangle (`setLanternReach`) widens the globe's limit while the lantern is open, so terrain and
   imagery anywhere the lens may go are loaded while still clipped away.
3. **Dated imagery as globe imagery layers**, limited to the reach: the current source on top, its
   neighbours in time loading underneath at alpha 0, and a dark grid under those that reads as
   "no record".

The rim (a DOM overlay) labels what the lens shows: source, date, licence, stated accuracy, and
"not checked against today's 3D world" on every source. The catalog
(`GET /api/v1/imagery/dated`) lists only real acquisitions: NAIP years from Planetary Computer,
the clearest dry-season Sentinel-2 scene per year, and USGS topographic sheets, which this API
tiles from their GeoTIFFs (`/api/v1/imagery/usgs-topo/tiles`).

## Alternatives rejected

- **A per-frame `ClippingPolygon` on the Google tileset.** A change to a tileset's clipping
  polygons marks every tile model for a rebake (`Cesium3DTileset._clippingPolygonsNeedRebake`) and
  resets draw commands when a tile's clipped state flips. Per frame, that is the whole world
  re-prepared on every mouse move. The globe's equivalent work is limited to the few tiles the
  polygon touches, which is why the window is globe-only.
- **A second `CesiumWidget` overlaid and masked with a CSS radial gradient.** It gives a true
  alpha feather, but costs a second WebGL context, a second terrain stream, a one-frame camera lag
  between the two, and entities drawn in one widget but not the other. ScanRendererHost shows how
  much machinery keeping two renderers in step takes.
- **A `PostProcessStage`.** By the time post-processing runs, the globe and the 3D tiles are one
  colour buffer; nothing there can tell which pixels were the Google mesh.
- **Translucency instead of discard.** A translucent material moves every Google tile into the
  translucent pass (order-independent transparency) on every frame.
- **Esri World Imagery Wayback and the Esri-hosted historical topo service.** Not openly licensed
  (Esri Master License Agreement), and a Wayback release date is not a capture date.

## Consequences

- Inside the lens the globe's terrain (Cesium World Terrain) is drawn where the mesh was cut away:
  trees and buildings disappear and old imagery lies on bare terrain. That is the intended meaning,
  and the rim says so.
- The feather is a stipple fixed in screen space, so it crawls slightly as the lens moves; the
  rim's glow covers most of it.
- The window's edge is aimed a few percent outside the lens. In that margin the terrain is drawn
  next to the mesh, and shows wherever it is the higher of the two; the rim's glow sits over it.
- Each window swap releases the globe's clipping data and rebuilds it for the tiles the window
  touches: a few milliseconds a frame while the lens moves, nothing at rest.
- The lantern is offered only in the photorealistic world, below 20 km, with WebGL 2 clipping
  on. In the open world the swipe Compare stays the tool.
- Nothing is measured against the Google mesh. Positional accuracy is the source's own statement
  (NAIP's FGDC record, the 1947 National Map Accuracy Standards for later USGS sheets), or "not
  stated". A control-point check against the mesh is a possible follow-up.
