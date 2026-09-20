/**
 * Every CesiumJS internal the Living Survey depends on, declared once, in one place.
 *
 * **Version: CesiumJS 1.145 / `@cesium/engine` 26.3.0.** None of this is in `Cesium.d.ts` — a
 * grep for `GaussianSplatPrimitive`, `GaussianSplatTextureGenerator` or
 * `Cesium3DTileset.gaussianSplatPrimitive` finds nothing — and none of it is public API. The
 * splat subsystem is churning (about twenty changelog entries across recent releases), so the
 * honest thing is to name the dependency explicitly rather than spread `as` casts across the
 * feature. When an upgrade breaks it, it breaks here.
 *
 * What we rely on, and where it lives in the engine source:
 *
 * | Internal | Source | Why |
 * | --- | --- | --- |
 * | `Cesium3DTileset.gaussianSplatPrimitive` | `Cesium3DTileset.js` | the primitive for a splat tileset |
 * | `primitive._positions` | `GaussianSplatPrimitive.js:431` | baked positions, the deformation base |
 * | `primitive._numSplats` | `:437` | how many of them |
 * | `primitive._splatRowMask` / `_splatRowShift` | `:445-446` | texel addressing, matching `u_splatRowMask`/`u_splatRowShift` |
 * | `primitive.gaussianSplatTexture` | `:439` | the attribute texture we write |
 * | `primitive._snapshot.generation` | `:1955`, `:1039`, `:1766` | monotonic rebuild counter |
 * | `primitive._rootTransform` | `:1233` | ENU frame of the tileset, for the frame assertion |
 * | `primitive.selectedTileLength` | `:1974` | tiles aggregated into the snapshot |
 * | `tile.content._lastSplatTransform` | `:1411` | the bake matrix `B`, for un-baking to the rig's frame |
 * | `GaussianSplatTextureGenerator.generateFromAttributes` | exported at `cesium/Source/Cesium.js:638` | the interception point |
 *
 * What we deliberately do **not** touch: `tile.content.positions`, `primitive._positions`,
 * `snapshot.positions` and the glTF POSITION array are read-only to us, forever.
 * `transformTile` rewrites `tile.content.positions` in place from the pristine glTF attribute,
 * so a write there would make a deformation permanent and compound it across every rebuild,
 * silently destroying the measured geometry. The only things we write are our own staging
 * buffer and the GPU texture.
 */

import type { Cesium3DTileset } from "cesium";

import type { Mat4 } from "./splatFrames";

/**
 * The subset of `Renderer/Texture` we call. Structural on purpose: `Texture` is not exported
 * from the barrel either, and a structural type is what lets a unit test supply a fake.
 */
export interface SplatTexture {
  copyFrom(options: {
    source: { width: number; height: number; arrayBufferView: Uint32Array };
    xOffset?: number;
    yOffset?: number;
  }): void;
  isDestroyed(): boolean;
}

/** The snapshot fields we read. `generation` is the primary rebuild signal. */
export interface SplatSnapshot {
  readonly generation?: number;
}

/** The content fields we read. */
export interface SplatTileContent {
  /**
   * `inverse(rootTransform) · computedTransform · axisCorrection · worldTransform` — the matrix
   * `transformTile` baked into `content.positions`, cached by the engine so it can skip
   * re-baking. Un-baking by its inverse is what recovers the rig's local ENU frame.
   */
  readonly _lastSplatTransform?: Mat4;
}

/** The tile fields we read. `children` is how single-tile-ness is asserted. */
export interface SplatTile {
  readonly children?: readonly SplatTile[];
  readonly content?: SplatTileContent;
}

/** The primitive fields we read. All optional: none of them exist for the first few frames. */
export interface SplatPrimitive {
  readonly _positions?: Float32Array;
  readonly _numSplats?: number;
  readonly _splatRowMask?: number;
  readonly _splatRowShift?: number;
  readonly _snapshot?: SplatSnapshot;
  readonly _rootTransform?: Mat4;
  readonly selectedTileLength?: number;
  /**
   * Re-read on every write, never cached. The draw command's uniform closes over the texture
   * object at build time and the engine destroys and recreates it whenever the dimensions
   * change, so a held reference becomes a write into a destroyed texture.
   */
  readonly gaussianSplatTexture?: SplatTexture;
}

/** The shape `SplatDeformer` works against — a real `Cesium3DTileset`, or a test double. */
export interface SplatTilesetLike {
  readonly gaussianSplatPrimitive?: SplatPrimitive;
  readonly root?: SplatTile;
}

/**
 * Views a `Cesium3DTileset` as the internals above.
 *
 * One cast, here, instead of everywhere. `Cesium3DTileset` and `SplatTilesetLike` have no
 * declared overlap, so it goes through `unknown`.
 */
export function splatTilesetOf(tileset: Cesium3DTileset): SplatTilesetLike {
  return tileset as unknown as SplatTilesetLike;
}

/** The bake matrix `B` for a single-tile splat tileset, or `undefined` before it is baked. */
export function bakeTransformOf(tileset: SplatTilesetLike): Mat4 | undefined {
  const transform = tileset.root?.content?._lastSplatTransform;
  return transform?.length === 16 ? transform : undefined;
}

/**
 * Whether the tileset is the single-tile kind this prototype is restricted to.
 *
 * Splat snapshots aggregate over *selected* tiles, so a splat's index is stable only while tile
 * selection is. Our own `splat_tiles.py` emits single-node tilesets, where indices are stable;
 * anywhere else the rig assignment would silently point at different splats between frames.
 * Returns `undefined` while the root is not loaded yet, which is a wait, not a refusal.
 */
export function isSingleTile(tileset: SplatTilesetLike): boolean | undefined {
  const root = tileset.root;
  if (root === undefined) return undefined;
  return (root.children?.length ?? 0) === 0;
}
