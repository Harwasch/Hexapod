# ADR 0003 — Embed local models by clipping the coarse world with footprints

**Status:** accepted · **Date:** 2026-09-15

## Context

A splat or mesh placed over global terrain and imagery z-fights at the ground and lets
coarse geometry poke through. CesiumJS provides `ClippingPolygonCollection` for the globe
and for tilesets.

## Decision

Each asset declares `clipsWorld` and a `clipFootprint` policy. `catalog` uses the stored
footprint/site boundary; `tileset` derives the polygon from the streamed tileset's root
bounding volume, which is the only reliable extent for third-party assets (the public demo
uses this). `ClippingManager` applies the same polygons to the globe and, when active, to
the global photorealistic tileset.

## Consequences

- Coarse geometry disappears exactly where the detailed model takes over; no seams from
  terrain intersecting the model.
- A model that does not fully cover its clip polygon shows the void; authors should provide
  tight footprints. The developer panel can disable clipping to compare.
- Requires WebGL2 (checked at runtime).
