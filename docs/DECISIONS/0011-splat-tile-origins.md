# ADR 0011 — Per-tile origins: splat tiles that never wrap

**Status:** accepted (Part A: packer, Python readers, API, backfill; Part B: the web readers
and view.html) · **Date:** 2026-10-19

## Context

SPZ, the splat compression every tile here carries, stores each position as 24-bit two's
complement fixed point. At the 12 fractional bits `splat_tiles.pack_spz` always wrote, that is
±2,047.99976 m from the tile's frame. Every tileset the repo had packed was under 40 m from its
origin, so nothing noticed that the packer cast to int32 and kept the low 24 bits: a gaussian
3,000 m east read back at −1,096 m, silently. A drone survey of a ranch can be 5–10 km across.

## Decision

1. **Refuse, never wrap.** `pack_spz` and `split_objects.write_spz` raise on a position past
   `SPZ_FIXED_MAX · 2^-bits` (and on a non-finite one), naming the reach, the range and the
   remedy.
2. **A tile that does not fit gets an origin of its own.** A non-root tile whose centres reach
   past `TILE_FIT_M` = 2,047 m from the site origin (a metre short of the range, so a
   whole-metre origin and float32 rounding stay clear of the edge; first principles, not a
   measurement) is written relative to the centre of its centres' box, snapped to whole metres.
   The tileset states it as a 3D Tiles 1.1 `transform`, **translation only**, relative to the
   parent's origin. A tile whose own half-extent still passes the range takes fewer fractional
   bits, which the SPZ header has always carried (`fractional_bits_for`: 11 to 4,095 m, 10 to
   8,191 m, …).
3. **The root never moves** (departure R1 from the coordinator's D1). Its frame is the
   tileset's: the globe placement (`root.transform`), `root.extras.collision` and
   `root.extras.viewCones`, `instances.json` bounds and about twenty web call sites that read
   `root.computedTransform` as the scan's ENU frame. A root whose content passes the range
   takes fewer bits instead.
4. **"Fits" is judged in the site frame, from the tile's own content** (departure R2).
   `build_tiles` writes children before their parents exist, so a child cannot be tested
   against its parent's frame. The transform written is still `origin − parent.origin`.
5. **Tiles that fit keep origin 0 and 12 bits.** Every scan under ~4 km packs to the bytes it
   always did; no committed fixture changed, and no published tileset needs a repack.
6. **Checksums are over site-frame positions.** Every position binding (rig `tileChecksums`,
   `instances.json`, `skin.json`, `plants.json`) keys a tile by FNV-1a of
   `float32(local + composed offset)` (`splat_tiles.site_positions`). For an origin-0 tile
   that is the decoded array itself. It is exact wherever |site| < 4,096 m (an origin is whole
   metres and `local` is on the 2^-12 grid) and the correctly rounded float32 of the exact sum
   beyond, which `Float32Array` of the same double reproduces bit for bit. The site frame is
   each tileset's own root frame, so a split object's tileset (`split_objects`: its root is the
   scan's root times `T(origin)`, its one tile untransformed) keeps object-local checksums, as
   it always had.
7. **Every reader composes the translations.** `splat_tiles.tile_offsets` (Python),
   `app/worker/positions.tile_offsets` (API), and on the web `view/tiles.ts` `parseTileset`
   (Spark, PlayCanvas, the streamer, layers and objects) and `splatTiles.ts` `tileOffsetOf`
   (CesiumJS's bindings, which un-bake with `B · T(-offset)`; docs/CESIUM.md).
   Each refuses a child transform that is not a pure translation: a tile may be moved, never
   turned or scaled, because Cesium takes the spherical-harmonic frame from the first selected
   tile and `getRotation` of a computed transform ignores translation, so translations keep
   every tile in the root's SH frame.
8. **A 2^-b grid for b < 12 is a subset of the 2^-12 grid** (R3), so a reader that snaps to
   1/4096 m still recovers an 11-bit tile's values exactly; `SPZ_POSITION_QUANTUM_M` stays.
9. **splat-transform translates in its own frame** (R4). It reads SPZ in PLY space, a half
   turn about z, applies `--translate t` there and bakes back on write, so moving a leaf by `o`
   in ENU is `--translate=-ox,-oy,oz`. Derived from 3.8.0's source and then run with the pinned
   package on the fixture (`tools/captures/lod_leaves.py`). Before it builds anything the
   backfill checks the merge leaf by leaf: the count is the leaves' count, and each leaf's run
   of rows (splat-transform keeps its inputs' order) has that leaf's own site-frame box. A box
   of the whole merge is not enough: one leaf misplaced inside the site's extent passes it.

## The fixture

`data/tiles/synthetic-far` (`tools/captures/synthetic_far.py`): a 10.4 km track, a 1.8 km spur
and three 3 m plots, 7,901 gaussians in 18 tiles; 15 tiles carry a transform, nested up to four
deep; the root and both level-one parents are at 11 bits. `source/expected.json` holds each
tile's composed offset, fractional bits, count and site-frame checksum, the leaves that draw
each plot and the site box: what every reader, in every language, is tested against. It is
byte-gated (`tests/test_synthetic_far.py`) under the conftest's Sandybridge OpenBLAS pin.
`python synthetic_far.py report data/tiles/synthetic-far/splat` prints per tile its origin,
bits and decode error; the worst is exactly half a quantum, on the 11-bit tiles.

## Limits

- **Cesium bindings beyond ~2 km from the tileset centre.** Cesium bakes positions to float32
  (ulp 2.4e-4 m at 2–4 km, against a 1.2e-4 m half-quantum), so a far tile's un-baked
  checksum may not match and Cesium may refuse rig, instance and skin bindings there. An engine
  limit that predates this; PlayCanvas and Spark read SPZ directly and are not affected. Within
  2,048 m of the centre the un-bake is exact by construction (`splatFrames.test.ts`). On the
  fixture every tile happened to un-bake exactly, but the worst pre-snap error was 1.22e-4 m
  against a 1.2207e-4 m half step: luck, not a margin, so nothing relies on it.
- **Mixed fractional bits.** `split_objects` refuses an object gathered from tiles of different
  bits rather than re-quantise it.
- **`tools/pipeline/experiments/lod_compare.py` does not compose.** It reads
  `boundingVolume.box` and decodes tiles without their transforms, so its cut is wrong for a
  far site. It is an experiment, and M1b rewrites it (and adds synthetic-far to its parity
  run), so it is left to M1b rather than changed twice.
- **Spark memory.** Part B's 32-bit centres in view.html double Spark's per-splat memory (16 to
  32 bytes); `LOAD_FACTOR` and `MAX_STREAMED` are kept, not re-measured on a phone. The globe's
  Spark renderer keeps its accumulator at 16 bytes (`accumExtSplats` off), as before.
- **SOG precision** for very large sites is not addressed.
