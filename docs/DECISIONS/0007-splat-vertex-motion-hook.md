# ADR 0007 — Multi-tile splat motion, and a vertex-shader hook behind a flag

**Status:** accepted · **Date:** 2026-09-28 · **Extends:** [ADR 0006](0006-splat-texture-rewrite.md)

## Context

ADR 0006 moves a splat tileset by rewriting the position lanes of its attribute texture from the
CPU, and the deformer refused any tileset with more than one tile. M5 of
[LIVING_WORLD.md](../LIVING_WORLD.md#9-fastest-path-to-the-demo) needs both limits gone: a real
capture past 100k gaussians is a level-of-detail REPLACE hierarchy (`splat_tiles.py`, with merged
parents), and per-frame CPU cost is linear in the splat count — measured here at 32 ms for 1M
splats before the upload, against a frame budget of 16.

In CesiumJS 1.145 one `GaussianSplatPrimitive` per tileset concatenates every selected tile's baked
positions into one array, one texture and one sort; the selection changes with the camera. A
splat's index therefore means nothing across frames, and a merged parent's gaussians appear in no
leaf.

## Decision

**1. Bind per tile, at load time, by position.** Identity (the tile's un-baked positions digest into
`rig.tileChecksums`), rig binding (nearest node) and flutter identity (a key over the snapped
position) are derived from each tile's own positions, cached per tile content, and concatenated in
the snapshot's tile order (`apps/web/src/cesium/splatTiles.ts`). The aggregation order is read from
`primitive._selectedTileSet` and verified against sampled positions before use. Nothing
rig-specific is packaged except the checksums (`tools/captures/rig_tiles.py`).

Why not a per-tile binding sidecar: binding measured 0.25 µs a gaussian at scale here (about 25 ms
per 100k-gaussian tile, once per tile load, cached while the tile stays loaded), and the nearest-node
search is about half of it — the identity proof has to run either way. A sidecar would save that
half for 2 bytes a gaussian on the wire (~10 % of a tile's SPZ), but it would tie every tile to one
rig version: re-extracting a rig, which M0–M3 will do often, would mean re-packaging every tile.

**2. A minimal engine patch adds a vertex-shader motion hook**, `GaussianSplatPrimitive.vertexMotion`
(`patches/@cesium__engine@26.3.0.patch`, alongside the existing terrain-fill fix): an object whose
`addToShader(shaderBuilder, uniformMap, context)` is called on every draw-command build, adding
uniforms and a `splatVertexMotion(splatIndex, position)` function that the vertex shader applies to
each fetched position. About forty lines, in three places: the constructor/accessor, the
draw-command build, and a rebuild-before-push when the hook changes. The GPU path
(`splatGpuMotion.ts`) uploads per-node affine rows and flutter coefficients (16 KB for a 214-node
rig) and a per-snapshot binding texture; per-frame CPU work no longer depends on the splat count.

It sits behind `VITE_SPLAT_GPU_MOTION` until it has been looked at on real hardware, with the CPU
path as the fallback for an unpatched engine or a snapshot whose tiles do not share one bake
matrix.

## Why a patch now, when ADR 0006 avoided one

The texture rewrite needed no patch because the interception point was a property lookup on an
exported module. There is no such seam in the splat shader: `customShader` does not reach splats,
and the vertex shader source is an immutable ES import. The alternatives were string-replacing the
compiled program's source from a wrapper around `buildGSplatDrawCommand` — invisible, and it fails
silently when upstream reformats one line — or a pnpm patch, which fails **loudly at install** when
upstream changes the lines it touches. The patch is the smaller risk, and it is written to be
upstreamable as-is.

## Consequences

- Multi-tile REPLACE tilesets move; merged parents move with the node nearest their own centre.
- The never-write-canonical invariant is structural on the GPU path: no engine array or texture is
  written at all. Calm is exact on both paths (CPU: the engine's bytes are re-uploaded; GPU: the
  shader returns the fetched position untouched when inactive or for a node at rest).
- A patched file changes with every CesiumJS upgrade; `pnpm install` refuses a patch that no longer
  applies, and `hasVertexMotionHook` sends the deformer to the CPU path on an engine without it.
- Draw order still follows canonical positions on both paths (see LIVING_SURVEY.md, "Draw order").
