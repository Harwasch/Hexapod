# ADR 0004 — Delivery assets are not the canonical data model

**Status:** accepted · **Date:** 2026-09-15

## Context

Cesium ion asset IDs are convenient today, but the twin must outlive any single tiler or
CDN and eventually hold raw captures, LiDAR, sensor data and semantic entities.

## Decision

The catalog models **sites**, **assets** (with `provider` + `source` locator, provenance,
license, resolution, CRS, temporal validity) and **layers** (discriminated sources). An ion
ID is one `source` variant among several. Canonical data will live in S3-compatible
object storage behind the `ObjectStorage` abstraction; tiling is a derivation step.

## Consequences

- Switching tilers or hosting tilesets ourselves is a data change, not a schema change.
- Every dataset keeps organization, URL, license, attribution and dates; the UI surfaces
  them ("About this layer", inspector).
- Ion's reconstruction API is wrapped behind a `ReconstructionProvider` seam and only the
  documented parts are used.
