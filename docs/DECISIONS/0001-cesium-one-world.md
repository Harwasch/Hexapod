# ADR 0001 — One CesiumJS world, no separate 2D/3D viewers

**Status:** accepted · **Date:** 2026-09-15

## Context

The product promise is a continuous zoom from planet to centimetre without leaving the map.
Toggling between a 2D map library and a 3D scan viewer breaks that continuity and doubles
the interaction model.

## Decision

CesiumJS in `scene3DOnly` mode is the only renderer. A high downward camera provides the
map experience; tilt reveals terrain and 3D geometry; local reality models are 3D Tiles
placed geographically and clipped into the globe. Explore mode reuses the same camera.

## Consequences

- Everything (imagery, vector, 3D, measurements) shares one coordinate system and camera.
- We accept CesiumJS's bundle size (~1.1 MB gzip) and WebGL2 requirements for clipping and
  splats; both are documented and degrade gracefully.
- Cesium's 2D/Columbus modes and stock widgets are disabled.
