# ADR 0006 — Move splats by rewriting the attribute texture, not by shader or fork

**Status:** accepted · **Date:** 2026-09-21

## Context

The Living Survey needs a measured Gaussian splat tileset to move every frame while its
canonical positions stay untouched (see [LIVING_SURVEY.md](../LIVING_SURVEY.md)). Three routes
were available in CesiumJS 1.145, and the obvious one is closed:

- **`tileset.customShader` does not reach splat content.** Splats bypass the Model pipeline
  entirely: `Scene/GaussianSplatPrimitive.js` builds its own render resources, its own
  `PrimitiveGaussianSplatVS/FS` and its own DrawCommand, with zero `customShader` references.
- **Forking `@cesium/engine`** would work — the repo already carries one patch and has the pnpm
  machinery — but the splat subsystem is actively churning: roughly twenty changelog entries
  across recent releases, all stabilisation, none of them opening the shader. A fork there is a
  merge conflict every release, on the least stable part of the engine.
- **Rewriting the attribute texture from outside** is possible in principle, but
  `Texture.copyFrom` takes a contiguous rectangle and each splat's position texel sits next to
  its covariance/colour texel, so any rectangle covering the positions covers those too. Writing
  rows means supplying bytes we did not compute. Reimplementing the engine's packer was rejected
  outright: a lane-order mistake there corrupts geometry and colour silently.

## Decision

Rewrite the position lanes of the attribute texture, supplying the other lanes from the buffer
the engine itself packed.

`GaussianSplatPrimitive.generateSplatTexture` resolves `generateFromAttributes` as a property
lookup on the `GaussianSplatTextureGenerator` module object at call time, and that module is
exported from the public barrel. Replacing the property captures the exact `Uint32Array` the
engine goes on to upload. This needed **no fork at all**: `patches/@cesium__engine@26.3.0.patch`
is untouched, and the interception survives a production Rollup build through the app's own
`manualChunks` cesium rule.

Cross-validated three ways on a real capture (`mygla`, 57,410 splats): captured buffer against
`primitive._positions`, max absolute error 0.0 m; GPU readback against `_positions`, 0.0 m;
captured against readback, 0 differing words of 466,944.

`gl.readPixels` (`RGBA_INTEGER`/`UNSIGNED_INT`) was also proven correct and stays as the
documented recovery path for a snapshot whose creation the interception missed — a tileset that
loaded before installation. It is deliberately **not** on the steady-state path: the first
readback measured 762–898 ms, a synchronous stall that would land on the frame a person is
watching. Where that path would be needed the deformer reports `no-capture` and the tree stays
still instead.

Every engine internal this relies on is declared in one file,
`apps/web/src/cesium/splatInternals.ts`, versioned to CesiumJS 1.145 / `@cesium/engine` 26.3.0,
with a table naming each internal and its source line. None of it is in `Cesium.d.ts`.

## Consequences

- No engine fork to carry, and no CPU reimplementation of a packing format that changes.
- The dependency is on private internals, so a CesiumJS upgrade can break it. It breaks in one
  file, loudly, and the deformer's response to anything it does not recognise is to refuse and
  leave the survey static.
- The interception must be installed before the first splat tile loads, so
  `CesiumSceneManager` installs it at construction.
- Covariance and colour are never written, so a rewrite cannot corrupt appearance — only
  position, and only from an immutable canonical copy.
- The splat sorter reads `primitive._positions`, which this never touches, so draw order goes
  stale relative to displaced geometry. That is an accepted artifact, measured by `sortStaleness`
  in `@twin/world`, and it is what bounds the default wind strength.
