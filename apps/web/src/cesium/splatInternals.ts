/**
 * Every CesiumJS internal the Living Survey depends on, declared once, in one place.
 *
 * **Version: CesiumJS 1.145 / `@cesium/engine` 26.3.0, with `patches/@cesium__engine@26.3.0.patch`.**
 * None of this is in `Cesium.d.ts` — a grep for `GaussianSplatPrimitive`,
 * `GaussianSplatTextureGenerator` or `Cesium3DTileset.gaussianSplatPrimitive` finds nothing — and
 * none of it is public API. The splat subsystem is churning (about twenty changelog entries
 * across recent releases), so the honest thing is to name the dependency explicitly rather than
 * spread `as` casts across the feature. When an upgrade breaks it, it breaks here.
 *
 * What we rely on, and where it lives in the engine source (line numbers are the patched file):
 *
 * | Internal | Source | Why |
 * | --- | --- | --- |
 * | `Cesium3DTileset.gaussianSplatPrimitive` | `Cesium3DTileset.js` | the primitive for a splat tileset |
 * | `primitive._positions` | `GaussianSplatPrimitive.js:431` | baked positions of every selected tile, aggregated |
 * | `primitive._colors` | `:434` | their RGBA; alpha is what `SplatCollider` calls solid |
 * | `primitive._numSplats` | `:437` | how many of them |
 * | `primitive._splatRowMask` / `_splatRowShift` | `:445-446` | texel addressing, matching `u_splatRowMask`/`u_splatRowShift` |
 * | `primitive.gaussianSplatTexture` | `:439` | the attribute texture the CPU path writes |
 * | `primitive._snapshot.generation` | `:2009` | monotonic rebuild counter |
 * | `primitive._rootTransform` | `:1835` | ENU frame of the tileset, for the frame assertion |
 * | `primitive._selectedTileSet` | `:2029` | the tiles the latest snapshot aggregated, **in aggregation order** |
 * | `primitive._pendingSnapshot` | `:2008` | a rebuild in flight, whose tiles `_selectedTileSet` already names |
 * | `tile.content._lastSplatTransform` | `:1444` | the bake matrix `B`, per tile, for un-baking to the rig's frame |
 * | `tile.content.positions` / `pointsLength` | `GaussianSplat3DTileContent.js:177,362` | a tile's baked positions, and its count |
 * | `primitive.vertexMotion` | **patch** | the vertex-shader motion hook the GPU path installs |
 * | `primitive.holdRebuilds` | **patch** | no new snapshot while the camera moves (`splatMotionGate.ts`) |
 * | `tileset.selectOffscreen` | **patch** (`Cesium3DTilesetBaseTraversal.js`) | a refining tile's out-of-view children drawn coarse |
 * | `tileset.focusWeight` / `focusConeRadians` | **patch** (`Cesium3DTile.js`) | detail spent on the centre of the view first |
 * | `scene.frameState.splatDecodesAllowed` | **patch** (`GltfSpzLoader.js`) | SPZ decodes that may start this frame |
 * | `GaussianSplatTextureGenerator.generateFromAttributes` | exported at `cesium/Source/Cesium.js:638` | the CPU path's interception point |
 *
 * **The aggregation order is the whole of multi-tile support.** A snapshot is
 * `concat(tile.content.positions for tile of tileset._selectedTiles)`, and `_selectedTileSet` is
 * `new Set(tileset._selectedTiles)` taken at the same moment — so iterating it gives the tiles in
 * the order their splats appear in `_positions`. It is refreshed when a rebuild *starts*, so it
 * describes the committed snapshot only while `_pendingSnapshot` is undefined; `snapshotTiles`
 * (`splatTiles.ts`) waits otherwise, and checks sampled positions against every tile besides.
 *
 * What we deliberately do **not** touch: `tile.content.positions`, `primitive._positions`,
 * `snapshot.positions` and the glTF POSITION array are read-only to us, forever.
 * `transformTile` rewrites `tile.content.positions` in place from the pristine glTF attribute,
 * so a write there would make a deformation permanent and compound it across every rebuild,
 * silently destroying the measured geometry. The only things we write are our own staging
 * buffer, the attribute texture (CPU path) and our own textures (GPU path).
 */

import type { Cesium3DTileset, Scene } from "cesium";

import type { Mat4 } from "./splatFrames";

/**
 * The subset of `Renderer/Texture` we call. Structural on purpose: `Texture` is not declared in
 * `Cesium.d.ts`, and a structural type is what lets a unit test supply a fake.
 */
export interface SplatTexture {
  copyFrom(options: {
    source: { width: number; height: number; arrayBufferView: Uint32Array | Float32Array };
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
  /** This tile's splat count. */
  readonly pointsLength?: number;
  /** This tile's baked positions: the slice of `_positions` it contributed. Never written. */
  readonly positions?: Float32Array;
}

/** The tile fields we read. */
export interface SplatTile {
  readonly children?: readonly SplatTile[];
  readonly content?: SplatTileContent;
}

/** What the patched engine calls on each draw-command build. See `splatGpuMotion.ts`. */
export interface SplatVertexMotion {
  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void;
}

/** The subset of `Renderer/ShaderBuilder` the hook uses. */
export interface SplatShaderBuilder {
  addUniform(type: string, identifier: string, destination?: number): void;
  addVertexLines(lines: string | readonly string[]): void;
}

/** The primitive fields we read. All optional: none of them exist for the first few frames. */
export interface SplatPrimitive {
  readonly _positions?: Float32Array;
  /** The committed snapshot's colours, RGBA bytes per splat (alpha is opacity). */
  readonly _colors?: Uint8Array;
  readonly _numSplats?: number;
  readonly _splatRowMask?: number;
  readonly _splatRowShift?: number;
  readonly _snapshot?: SplatSnapshot;
  readonly _pendingSnapshot?: unknown;
  readonly _rootTransform?: Mat4;
  readonly _selectedTileSet?: ReadonlySet<SplatTile>;
  readonly selectedTileLength?: number;
  /**
   * Re-read on every write, never cached. The draw command's uniform closes over the texture
   * object at build time and the engine destroys and recreates it whenever the dimensions
   * change, so a held reference becomes a write into a destroyed texture.
   */
  readonly gaussianSplatTexture?: SplatTexture;
  /** Present (as an accessor, initially `undefined`) only on the patched engine. */
  vertexMotion?: SplatVertexMotion;
  /** Patched engine: while true, the committed snapshot stays and no rebuild starts. */
  holdRebuilds?: boolean;
  isDestroyed?(): boolean;
}

/** The shape `SplatDeformer` works against — a real `Cesium3DTileset`, or a test double. */
export interface SplatTilesetLike {
  readonly gaussianSplatPrimitive?: SplatPrimitive;
  readonly root?: SplatTile;
}

/**
 * Views a `Cesium3DTileset` as the internals above.
 *
 * One place, instead of everywhere. Every field above is optional and `Cesium3DTileset.root`'s
 * declared shape (`children`, `content.pointsLength`) fits `SplatTile`, so no cast is needed —
 * which is also why nothing here can be trusted to exist and every read is checked.
 */
export function splatTilesetOf(tileset: Cesium3DTileset): SplatTilesetLike {
  return tileset;
}

/**
 * Whether this primitive carries the patch's `vertexMotion` hook. On the unpatched engine the
 * property does not exist at all; on the patched one it is an accessor on the prototype.
 */
export function hasVertexMotionHook(primitive: SplatPrimitive): boolean {
  return "vertexMotion" in primitive;
}

/** The frame-state field the patched SPZ loader reads. */
export interface SplatFrameState {
  /** SPZ decodes that may still start this frame; undefined for no cap. */
  splatDecodesAllowed?: number;
}

/** `scene.frameState` -- declared `@private` in the engine, absent from `Cesium.d.ts`. */
export function splatFrameStateOf(scene: Scene): SplatFrameState {
  return (scene as unknown as { frameState: SplatFrameState }).frameState;
}

/**
 * Has a splat tileset's traversal keep out-of-view children of a refining tile, coarse
 * (patched `Cesium3DTilesetBaseTraversal`): the snapshot then covers the whole scan, so one
 * held while the camera turns has no holes. See `splatMotionGate.ts`.
 */
export function keepOffscreenSplats(tileset: Cesium3DTileset): void {
  (tileset as unknown as { selectOffscreen: boolean }).selectOffscreen = true;
}

/** How much less detail the edge of the view gets than its centre (0..1). */
export const SPLAT_FOCUS_WEIGHT = 0.6;
/** Half-angle around the view direction that keeps full detail (about 11 degrees). */
export const SPLAT_FOCUS_CONE_RAD = 0.2;

/**
 * Spends a splat tileset's detail where the person looks (patched `Cesium3DTile`
 * `getScreenSpaceError`): full detail within a cone around the view direction, less toward
 * the edges, so under a budget the thing in the middle of the view refines first. Distance is
 * already in the error (a near tile's error is larger), so near and central come first.
 */
export function focusSplats(tileset: Cesium3DTileset): void {
  const patched = tileset as unknown as { focusWeight: number; focusConeRadians: number };
  patched.focusWeight = SPLAT_FOCUS_WEIGHT;
  patched.focusConeRadians = SPLAT_FOCUS_CONE_RAD;
}
