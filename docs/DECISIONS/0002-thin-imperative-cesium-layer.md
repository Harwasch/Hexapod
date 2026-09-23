# ADR 0002 — Thin imperative Cesium layer instead of a React wrapper

**Status:** accepted · **Date:** 2026-09-15

## Context

React wrappers around Cesium (Resium and similar) reconcile Cesium objects through React
props, which hides the lifecycle, restricts access to newer APIs (clipping polygons,
splats, controllers) and re-renders on every camera change.

## Decision

A dedicated `cesium/` subsystem owns the viewer: `CesiumSceneManager` plus focused managers
(Camera, Layers, Sites, Selection, Measurement, Clipping, Performance, Explore, Debug).
Managers emit typed events; one `SceneBridge` component mirrors them into Zustand stores
and pushes settings back. Components call managers through `useScene()`.

## Consequences

- The viewer is created and destroyed exactly once; event handlers are disposed centrally.
- React never re-renders the scene; camera telemetry is throttled to ~10 Hz.
- New Cesium features are one manager method away, not a wrapper release away.
