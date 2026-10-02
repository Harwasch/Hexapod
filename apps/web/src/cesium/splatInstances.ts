/**
 * Hides and highlights a scan's objects on the GPU (scene instances, `lib/instances.ts`).
 *
 * The segmentation step writes, beside the tiles, which instance each gaussian of each tile
 * belongs to: per tile checksum, run-length ids in the tile's own order (`instances.json`,
 * docs/SCENE_OBJECTS.md §4). The splat primitive draws every selected tile from one attribute
 * texture whose index `splatIndex` means nothing on its own -- it is a slot (incremental mode)
 * or a place in an aggregate, and changes with the selection -- so, as the deformer does
 * (`splatTiles.ts`), each tile is identified by un-baking its own positions and digesting
 * them, once per tile load, and its ids decoded from the runs (`decodeRuns`, shared with
 * `plants.json`) and written at its range.
 *
 * Two textures of our own, never the engine's:
 *
 * - **ids** (RGBA32UI, 1024 texels a row, four splats a texel): the instance id of every
 *   splat index, 0 for none or for a tile the file does not list. A tile that arrives is
 *   written at its range and only the rows it touched are uploaded; a tile that leaves has its
 *   range zeroed.
 * - **state** (RGBA8, 1024 texels a row, one per id): `r` hidden, `g` highlighted. Rewritten
 *   (a few KB) when what is hidden or highlighted changes.
 *
 * and two engine-patch hooks:
 *
 * - **hide** is a part of the primitive's visibility chain (`splatVisibility.ts`), so it
 *   composes with the view cones rather than replacing them: a hidden splat's weight is 0,
 *   and it is dropped before its covariance is fetched;
 * - **highlight** is the patch's `vertexColor` hook: a highlighted splat is tinted and lifted,
 *   and while anything is highlighted the rest are dimmed (and made more transparent, so the
 *   highlight shows through what stands in front of it), unless dimming is off.
 *
 * Neither does anything until something is hidden or highlighted (`u_instanceParams.x` is 0,
 * and each function returns at once). In incremental mode a tile's slot does not move while it
 * stays selected, so the ids stay valid across snapshots and are patched as tiles come and go;
 * in aggregated mode every snapshot reshuffles indices, so the hook is off for a snapshot until
 * its ids are written (a frame).
 */

import { checksumPositions } from "@twin/world";
import { BoundingSphere, Cartesian3, Matrix4, type Cesium3DTileset, type Scene } from "cesium";
import * as CesiumBarrel from "cesium";

import {
  instancesRefOf,
  loadInstances,
  tileInstanceIds,
  withDescendants,
  type InstancesDoc,
} from "@/lib/instances";
import { createLogger } from "@/lib/log";
import { useInstances } from "@/state/instances";

import { invertAffine, unbakePositions } from "./splatFrames";
import {
  splatTilesetOf,
  type SplatPrimitive,
  type SplatShaderBuilder,
  type SplatTilesetLike,
} from "./splatInternals";
import { sameMatrix, snapshotTiles, type SnapshotTile } from "./splatTiles";
import {
  addVisibilityPart,
  hasVisibilityPart,
  removeVisibilityPart,
  type SplatVisibilityPart,
  type VisibilityPrimitive,
} from "./splatVisibility";

const log = createLogger("instances");

/** Texels a row of either texture holds. */
export const INSTANCE_TEXTURE_WIDTH = 1024;
/** Splats an id texel holds (RGBA32UI: one id a channel). */
export const IDS_PER_TEXEL = 4;
/** Splats a row of the id texture holds. */
export const SPLATS_PER_ID_ROW = INSTANCE_TEXTURE_WIDTH * IDS_PER_TEXEL;

/** A texture we own. Structural, so a test can supply a fake. */
export interface InstanceTexture {
  copyFrom(options: {
    source: { width: number; height: number; arrayBufferView: ArrayBufferView };
    xOffset?: number;
    yOffset?: number;
  }): void;
  isDestroyed(): boolean;
  destroy(): void;
}

/** How the hook makes its textures and uniform values. The real one wraps CesiumJS. */
export interface InstanceGpu {
  /** An `RGBA32UI` texture. */
  createUintQuads(
    context: unknown,
    width: number,
    height: number,
    data: Uint32Array,
  ): InstanceTexture;
  /** An `RGBA8` texture, read normalised. */
  createBytes(context: unknown, width: number, height: number, data: Uint8Array): InstanceTexture;
  /** A `vec4` uniform's value. */
  vec4(x: number, y: number, z: number, w: number): unknown;
}

/** A primitive of the patched engine, with the visibility and colour accessors. */
export type InstancePrimitive = VisibilityPrimitive & { vertexColor?: SplatVertexColor };

/** What the patched engine calls on each draw-command build (`vertexColor`). */
export interface SplatVertexColor {
  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void;
}

/** How the highlight looks. */
export interface HighlightStyle {
  /** Linear RGB the highlight is pulled toward, and how far (`a`). */
  tint: readonly [number, number, number, number];
  /** While a highlight is active, everything else: colour times `x`, opacity times `y`. */
  dim: readonly [number, number];
}

export const HIGHLIGHT_STYLE: HighlightStyle = {
  tint: [1.0, 0.78, 0.25, 0.45],
  dim: [0.35, 0.5],
};

/**
 * Uniforms and helpers both hooks use, declared once whichever of them the build reaches
 * first: the visibility chain is built before the colour hook, and either may be absent.
 */
export const INSTANCE_HELPERS_GLSL = `
#ifndef HEXAPOD_SPLAT_INSTANCES
#define HEXAPOD_SPLAT_INSTANCES
uniform highp usampler2D u_instanceIds;
uniform highp sampler2D u_instanceState;
// x: ids are current and something is hidden or highlighted; y: the largest id;
// z: splat indices the id texture covers; w: something is highlighted.
uniform vec4 u_instanceParams;
uniform vec4 u_instanceTint;
uniform vec4 u_instanceDim;

uint splatInstanceId(uint splatIndex) {
    if (float(splatIndex) >= u_instanceParams.z) {
        return 0u;
    }
    uint q = splatIndex >> 2u;
    uvec4 t = texelFetch(u_instanceIds, ivec2(int(q & ${String(INSTANCE_TEXTURE_WIDTH - 1)}u), int(q >> ${String(Math.log2(INSTANCE_TEXTURE_WIDTH))}u)), 0);
    uint k = splatIndex & 3u;
    return k == 0u ? t.r : (k == 1u ? t.g : (k == 2u ? t.b : t.a));
}

vec4 splatInstanceState(uint id) {
    if (id == 0u || float(id) > u_instanceParams.y) {
        return vec4(0.0);
    }
    int i = int(id);
    return texelFetch(u_instanceState, ivec2(i & ${String(INSTANCE_TEXTURE_WIDTH - 1)}, i >> ${String(Math.log2(INSTANCE_TEXTURE_WIDTH))}), 0);
}
#endif
`;

/** The hide part of the visibility chain. */
export const INSTANCE_VISIBILITY_GLSL = `
float splatInstanceVisibility(uint splatIndex, vec3 position) {
    if (u_instanceParams.x < 0.5) {
        return 1.0;
    }
    return splatInstanceState(splatInstanceId(splatIndex)).r > 0.5 ? 0.0 : 1.0;
}
`;

/** The patched engine's `splatVertexColor`: the highlight, and the rest dimmed. */
export const INSTANCE_COLOR_GLSL = `
vec4 splatVertexColor(uint splatIndex, vec3 position, vec4 color) {
    if (u_instanceParams.x < 0.5 || u_instanceParams.w < 0.5) {
        return color;
    }
    if (splatInstanceState(splatInstanceId(splatIndex)).g > 0.5) {
        return vec4(mix(color.rgb, u_instanceTint.rgb, u_instanceTint.a) + 0.06, color.a);
    }
    return vec4(color.rgb * u_instanceDim.x, color.a * u_instanceDim.y);
}
`;

/** Rows of the id texture for `splats` indices (at least one). */
export function idTextureRows(splats: number): number {
  return Math.max(1, Math.ceil(splats / SPLATS_PER_ID_ROW));
}

/** Rows of the state texture for ids `0..maxId`. */
export function stateTextureRows(maxId: number): number {
  return Math.max(1, Math.ceil((maxId + 1) / INSTANCE_TEXTURE_WIDTH));
}

/**
 * The state texels: `r` 255 for a hidden id, `g` 255 for a highlighted one. Hidden and
 * highlighted sets are expected expanded to leaves (`withDescendants`); ids past `maxId` are
 * ignored.
 */
export function writeStateTexels(
  out: Uint8Array,
  maxId: number,
  hidden: Iterable<number>,
  highlighted: Iterable<number>,
): Uint8Array {
  out.fill(0);
  for (const id of hidden) if (id >= 1 && id <= maxId) out[id * 4] = 255;
  for (const id of highlighted) if (id >= 1 && id <= maxId) out[id * 4 + 1] = 255;
  return out;
}

/** What the shader computes for one splat: its opacity weight and colour. A reference. */
export function evaluateInstanceShading(
  id: number,
  state: Uint8Array,
  maxId: number,
  params: { active: boolean; highlightActive: boolean },
  color: readonly [number, number, number, number],
  style: HighlightStyle = HIGHLIGHT_STYLE,
): { visibility: number; color: [number, number, number, number] } {
  const texel = id >= 1 && id <= maxId ? id * 4 : -1;
  const hidden = texel >= 0 && (state[texel] ?? 0) > 127;
  const lit = texel >= 0 && (state[texel + 1] ?? 0) > 127;
  if (!params.active) return { visibility: 1, color: [...color] };
  const visibility = hidden ? 0 : 1;
  if (!params.highlightActive) return { visibility, color: [...color] };
  if (lit) {
    const [tr, tg, tb, ta] = style.tint;
    const mix = (a: number, b: number): number => a + (b - a) * ta + 0.06;
    return {
      visibility,
      color: [mix(color[0], tr), mix(color[1], tg), mix(color[2], tb), color[3]],
    };
  }
  return {
    visibility,
    color: [
      color[0] * style.dim[0],
      color[1] * style.dim[0],
      color[2] * style.dim[0],
      color[3] * style.dim[1],
    ],
  };
}

/** One tile's ids under one bake: `undefined` ids when the file does not list the tile. */
interface TileIds {
  readonly bake: readonly number[];
  readonly checksum: string;
  readonly ids: Uint32Array | undefined;
}

/** What `sync` did, for diagnostics and tests. */
export interface InstanceSyncResult {
  readonly changed: boolean;
  /** Tiles drawn now, and how many of them the file lists. */
  readonly tiles: number;
  readonly matched: number;
}

/** The hooks and textures for one scan. */
export class SplatInstances implements SplatVisibilityPart, SplatVertexColor {
  readonly visibilityFunction = "splatInstanceVisibility";
  /** First: one texel fetch, and a hidden splat skips everything after it. */
  readonly visibilityOrder = 0;
  readonly doc: InstancesDoc;
  readonly #gpu: InstanceGpu;
  readonly #tileset: SplatTilesetLike;
  readonly #cache = new WeakMap<object, TileIds>();
  #primitive: InstancePrimitive | undefined;
  #context: unknown;
  /** The id of every splat index, four a texel. */
  #ids = new Uint32Array(SPLATS_PER_ID_ROW);
  #idTexture: InstanceTexture | undefined;
  /** Rows of `#ids` written since the last upload, or `undefined`. */
  #dirtyRows: [number, number] | undefined;
  #idTextureRows = 0;
  readonly #state: Uint8Array;
  #stateTexture: InstanceTexture | undefined;
  #stateDirty = true;
  /** Where each drawn tile's ids are, by content object. */
  #placed = new Map<object, { start: number; count: number }>();
  #generation = -1;
  #positions: Float32Array | undefined;
  #incremental = false;
  #covered = 0;
  #anyHidden = false;
  #anyHighlighted = false;
  #params: unknown;
  #paramsKey = "";
  readonly #tint: unknown;
  #dim: unknown;
  #dimOthers = true;
  /** Tiles the file did not list, by checksum, logged once each. */
  readonly #unlisted = new Set<string>();
  /** Set while the colour hook is being built, so the shared `addToShader` adds its lines. */
  #buildingColor = false;
  /** Off draws every splat as it was; the hooks stay installed. */
  enabled = true;
  readonly style: HighlightStyle;

  constructor(
    doc: InstancesDoc,
    gpu: InstanceGpu,
    tileset: SplatTilesetLike,
    style = HIGHLIGHT_STYLE,
  ) {
    this.doc = doc;
    this.#gpu = gpu;
    this.#tileset = tileset;
    this.style = style;
    this.#state = new Uint8Array(INSTANCE_TEXTURE_WIDTH * stateTextureRows(doc.maxId) * 4);
    this.#tint = gpu.vec4(...style.tint);
    this.#dim = gpu.vec4(style.dim[0], style.dim[1], 0, 0);
  }

  /** The id each splat index carries now (a copy of the CPU side of the id texture). */
  idAt(splatIndex: number): number {
    return splatIndex < this.#covered ? (this.#ids[splatIndex] ?? 0) : 0;
  }

  /** Whether the shader would act this frame. */
  get active(): boolean {
    return this.#drawActive();
  }

  /** Called by the patched engine (through the visibility chain, and as `vertexColor`). */
  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void {
    this.#context = context;
    this.#ensureTextures();
    shaderBuilder.addVertexLines(INSTANCE_HELPERS_GLSL);
    uniformMap.u_instanceIds = () => this.#idTexture;
    uniformMap.u_instanceState = () => this.#stateTexture;
    uniformMap.u_instanceParams = () => this.#paramsValue();
    uniformMap.u_instanceTint = () => this.#tint;
    uniformMap.u_instanceDim = () => this.#dim;
    // One object is both hooks: tell them apart by the flag the chain sets before calling us.
    shaderBuilder.addVertexLines(
      this.#buildingColor ? INSTANCE_COLOR_GLSL : INSTANCE_VISIBILITY_GLSL,
    );
  }

  /** The `vertexColor` hook: a thin wrapper, so the visibility part and it share one object. */
  readonly colorHook: SplatVertexColor = {
    addToShader: (shaderBuilder, uniformMap, context) => {
      this.#buildingColor = true;
      try {
        this.addToShader(shaderBuilder, uniformMap, context);
      } finally {
        this.#buildingColor = false;
      }
    },
  };

  /**
   * Installs both hooks on `primitive`. False when the engine has neither; hiding works on an
   * engine without the colour hook, highlighting only with it.
   */
  install(primitive: InstancePrimitive): boolean {
    if (this.#primitive && this.#primitive !== primitive) this.uninstall();
    this.#primitive = primitive;
    const hides = hasVisibilityPart(primitive, this) || addVisibilityPart(primitive, this);
    let colours = false;
    if ("vertexColor" in primitive) {
      primitive.vertexColor ??= this.colorHook;
      colours = primitive.vertexColor === this.colorHook;
    }
    return hides || colours;
  }

  /** Whether both hooks are on the primitive. */
  get installed(): boolean {
    const primitive = this.#primitive;
    return hasVisibilityPart(primitive, this) && primitive?.vertexColor === this.colorHook;
  }

  uninstall(): void {
    const primitive = this.#primitive;
    this.#primitive = undefined;
    if (!primitive || primitive.isDestroyed?.() === true) return;
    removeVisibilityPart(primitive, this);
    if (primitive.vertexColor === this.colorHook) primitive.vertexColor = undefined;
  }

  /** What is hidden and highlighted, as the store has it (ids not yet expanded to leaves). */
  setState(
    hidden: ReadonlySet<number>,
    highlighted: ReadonlySet<number>,
    dimOthers: boolean,
  ): void {
    const hiddenLeaves = withDescendants(this.doc, hidden);
    const litLeaves = withDescendants(this.doc, highlighted);
    writeStateTexels(this.#state, this.doc.maxId, hiddenLeaves, litLeaves);
    this.#anyHidden = hiddenLeaves.size > 0;
    this.#anyHighlighted = litLeaves.size > 0;
    if (dimOthers !== this.#dimOthers) {
      this.#dimOthers = dimOthers;
      this.#dim = dimOthers
        ? this.#gpu.vec4(this.style.dim[0], this.style.dim[1], 0, 0)
        : this.#gpu.vec4(1, 1, 0, 0);
    }
    this.#stateDirty = true;
    this.#ensureTextures();
  }

  /**
   * Brings the ids up to the primitive's committed snapshot, and uploads what changed.
   * Call once a frame, before the render.
   */
  sync(): InstanceSyncResult {
    const unchanged = { changed: false, tiles: this.#placed.size, matched: 0 };
    const primitive: InstancePrimitive | undefined = this.#tileset.gaussianSplatPrimitive;
    if (!primitive) return unchanged;
    if (primitive !== this.#primitive || !this.installed) this.install(primitive);
    const positions = primitive._positions;
    const numSplats = primitive._numSplats ?? 0;
    const generation = primitive._snapshot?.generation ?? -1;
    const uploaded = this.#ensureTextures();
    if (
      positions === undefined ||
      numSplats <= 0 ||
      (generation === this.#generation && positions === this.#positions)
    ) {
      return { ...unchanged, changed: uploaded };
    }
    const listed = snapshotTiles(this.#tileset, primitive, positions, numSplats);
    if (listed.kind === "wait") return { ...unchanged, changed: uploaded };
    const incremental = primitive._tileSlots !== undefined;
    this.#reserve(numSplats);
    const next = new Map<object, { start: number; count: number }>();
    for (const tile of listed.tiles)
      next.set(tile.content, { start: tile.start, count: tile.count });
    if (!incremental || !this.#incremental) {
      // Every index may mean another splat now: start from nothing.
      this.#ids.fill(0);
      this.#markRows(0, Math.max(this.#covered, numSplats));
      this.#placed.clear();
    } else {
      for (const [content, slot] of this.#placed) {
        const now = next.get(content);
        if (now?.start === slot.start && now.count === slot.count) continue;
        this.#ids.fill(0, slot.start, slot.start + slot.count);
        this.#markRows(slot.start, slot.start + slot.count);
      }
    }
    let matched = 0;
    for (const tile of listed.tiles) {
      const before = this.#placed.get(tile.content);
      const ids = this.#tileIds(tile, positions);
      if (ids !== undefined) matched += 1;
      if (
        before?.start === tile.start &&
        before.count === tile.count &&
        incremental &&
        this.#incremental
      )
        continue;
      if (ids !== undefined) this.#ids.set(ids, tile.start);
      else this.#ids.fill(0, tile.start, tile.start + tile.count);
      this.#markRows(tile.start, tile.start + tile.count);
    }
    this.#placed = next;
    this.#incremental = incremental;
    this.#covered = numSplats;
    this.#generation = generation;
    this.#positions = positions;
    this.#ensureTextures();
    return { changed: true, tiles: listed.tiles.length, matched };
  }

  /** Takes the hooks off and frees the textures. */
  destroy(): void {
    this.uninstall();
    for (const texture of [this.#idTexture, this.#stateTexture]) {
      if (texture && !texture.isDestroyed()) texture.destroy();
    }
    this.#idTexture = undefined;
    this.#stateTexture = undefined;
  }

  #tileIds(tile: SnapshotTile, positions: Float32Array): Uint32Array | undefined {
    const cached = this.#cache.get(tile.content);
    if (cached && sameMatrix(cached.bake, tile.bake)) return cached.ids;
    const inverse = invertAffine(tile.bake);
    if (inverse === undefined) return undefined;
    const baked = positions.subarray(tile.start * 3, (tile.start + tile.count) * 3);
    const checksum = checksumPositions(unbakePositions(baked, inverse));
    const decoded = tileInstanceIds(this.doc, checksum);
    const ids = decoded?.length === tile.count ? decoded : undefined;
    if (ids === undefined && !this.#unlisted.has(checksum)) {
      this.#unlisted.add(checksum);
      log.warn("a drawn tile is not in instances.json; its splats carry no instance", {
        checksum,
      });
    }
    this.#cache.set(tile.content, { bake: Array.from(tile.bake), checksum, ids });
    return ids;
  }

  /** Grows the CPU id buffer (by half again) to hold `splats` indices. */
  #reserve(splats: number): void {
    const needed = idTextureRows(splats) * SPLATS_PER_ID_ROW;
    if (needed <= this.#ids.length) return;
    const grown = new Uint32Array(
      idTextureRows(Math.max(splats, Math.ceil((this.#ids.length * 3) / 2))) * SPLATS_PER_ID_ROW,
    );
    grown.set(this.#ids);
    this.#ids = grown;
  }

  #markRows(start: number, end: number): void {
    if (end <= start) return;
    const first = Math.floor(start / SPLATS_PER_ID_ROW);
    const last = Math.floor((end - 1) / SPLATS_PER_ID_ROW);
    const dirty = this.#dirtyRows;
    this.#dirtyRows = dirty ? [Math.min(dirty[0], first), Math.max(dirty[1], last)] : [first, last];
  }

  /** Creates what is missing and uploads what changed; true when anything was uploaded. */
  #ensureTextures(): boolean {
    const context = this.#context;
    if (context === undefined) return false;
    let uploaded = false;
    const rows = this.#ids.length / SPLATS_PER_ID_ROW;
    if (
      this.#idTexture === undefined ||
      this.#idTexture.isDestroyed() ||
      this.#idTextureRows !== rows
    ) {
      this.#idTexture?.destroy();
      this.#idTexture = this.#gpu.createUintQuads(context, INSTANCE_TEXTURE_WIDTH, rows, this.#ids);
      this.#idTextureRows = rows;
      this.#dirtyRows = undefined;
      uploaded = true;
    } else if (this.#dirtyRows !== undefined) {
      const [first, last] = this.#dirtyRows;
      const words = SPLATS_PER_ID_ROW;
      this.#idTexture.copyFrom({
        source: {
          width: INSTANCE_TEXTURE_WIDTH,
          height: last - first + 1,
          arrayBufferView: this.#ids.subarray(first * words, (last + 1) * words),
        },
        xOffset: 0,
        yOffset: first,
      });
      this.#dirtyRows = undefined;
      uploaded = true;
    }
    if (this.#stateTexture === undefined || this.#stateTexture.isDestroyed() || this.#stateDirty) {
      // A few KB: re-created rather than patched.
      this.#stateTexture?.destroy();
      this.#stateTexture = this.#gpu.createBytes(
        context,
        INSTANCE_TEXTURE_WIDTH,
        stateTextureRows(this.doc.maxId),
        this.#state,
      );
      this.#stateDirty = false;
      uploaded = true;
    }
    return uploaded;
  }

  #drawActive(): boolean {
    if (!this.enabled || !(this.#anyHidden || this.#anyHighlighted)) return false;
    if (this.#idTexture === undefined || this.#stateTexture === undefined) return false;
    if (this.#generation < 0) return false;
    const primitive = this.#primitive;
    if (!primitive) return false;
    // Incremental: a tile keeps its slot while it is drawn, so the ids stay right between
    // snapshots. Aggregated: only for the snapshot they were written for.
    return this.#incremental || primitive._snapshot?.generation === this.#generation;
  }

  #paramsValue(): unknown {
    const active = this.#drawActive();
    const key = `${String(active)}:${String(this.#anyHighlighted)}:${String(this.#covered)}`;
    if (key !== this.#paramsKey) {
      this.#paramsKey = key;
      this.#params = this.#gpu.vec4(
        active ? 1 : 0,
        this.doc.maxId,
        this.#covered,
        this.#anyHighlighted ? 1 : 0,
      );
    }
    return this.#params;
  }
}

interface Barrel {
  Texture?: new (options: Record<string, unknown>) => InstanceTexture;
  Sampler?: { NEAREST?: unknown };
  PixelFormat?: { RGBA?: number; RGBA_INTEGER?: number };
  PixelDatatype?: { UNSIGNED_BYTE?: number; UNSIGNED_INT?: number };
  Cartesian4?: new (x: number, y: number, z: number, w: number) => unknown;
}

/** The real `InstanceGpu`, or `undefined` when this CesiumJS build lacks what it needs. */
export function cesiumInstanceGpu(): InstanceGpu | undefined {
  const barrel = CesiumBarrel as unknown as Barrel;
  const { Texture, Cartesian4 } = barrel;
  const nearest = barrel.Sampler?.NEAREST;
  const rgba = barrel.PixelFormat?.RGBA;
  const rgbaInteger = barrel.PixelFormat?.RGBA_INTEGER;
  const bytes = barrel.PixelDatatype?.UNSIGNED_BYTE;
  const uint = barrel.PixelDatatype?.UNSIGNED_INT;
  if (
    typeof Texture !== "function" ||
    typeof Cartesian4 !== "function" ||
    nearest === undefined ||
    rgba === undefined ||
    rgbaInteger === undefined ||
    bytes === undefined ||
    uint === undefined
  ) {
    return undefined;
  }
  const create = (
    context: unknown,
    width: number,
    height: number,
    data: ArrayBufferView,
    pixelFormat: number,
    pixelDatatype: number,
  ): InstanceTexture =>
    new Texture({
      context,
      source: { width, height, arrayBufferView: data },
      pixelFormat,
      pixelDatatype,
      preMultiplyAlpha: false,
      skipColorSpaceConversion: true,
      flipY: false,
      sampler: nearest,
    });
  return {
    createUintQuads: (context, width, height, data) =>
      create(context, width, height, data, rgbaInteger, uint),
    createBytes: (context, width, height, data) =>
      create(context, width, height, data, rgba, bytes),
    vec4: (x, y, z, w) => new Cartesian4(x, y, z, w),
  };
}

// ---- Attachment --------------------------------------------------------------------------

/** Loaded scans, by asset id, for flying to an instance. */
const ATTACHED = new Map<string, { tileset: Cesium3DTileset; doc: InstancesDoc }>();

/**
 * The instances of `assetId`'s scan, once they loaded: what a dedicated splat renderer
 * (scanView/scanInstances.ts) draws from, since CesiumJS keeps the scan's tileset loaded (hidden)
 * under any renderer.
 */
export function instancesDocOf(assetId: string): InstancesDoc | undefined {
  return ATTACHED.get(assetId)?.doc;
}

/**
 * The world-space sphere around instance `id` of `assetId`'s scan: its bounds (tileset local
 * ENU metres) through the root tile's computed transform, which carries the model matrix.
 */
export function instanceSphere(assetId: string, id: number): BoundingSphere | undefined {
  const entry = ATTACHED.get(assetId);
  const instance = entry?.doc.byId.get(id);
  const root = entry?.tileset.root as { computedTransform?: Matrix4 } | undefined;
  if (!instance || !root?.computedTransform) return undefined;
  const { min, max } = instance.bounds;
  const corners: Cartesian3[] = [];
  for (const x of [min[0], max[0]])
    for (const y of [min[1], max[1]])
      for (const z of [min[2], max[2]])
        corners.push(
          Matrix4.multiplyByPoint(
            root.computedTransform,
            new Cartesian3(x, y, z),
            new Cartesian3(),
          ),
        );
  return BoundingSphere.fromPoints(corners);
}

export type LoadInstances = typeof loadInstances;

/**
 * Lets `tileset`'s objects be hidden and highlighted, when its root declares
 * `extras.instances`: the file is fetched once, its table put in the store
 * (`state/instances.ts`) under `assetId`, and the hooks installed on the splat primitive.
 * Returns the disposer. A tileset without instances, or an engine without the hooks, costs
 * nothing.
 */
export function attachInstances(
  tileset: Cesium3DTileset,
  scene: Pick<Scene, "preUpdate" | "requestRender">,
  assetId: string,
  gpu: InstanceGpu | undefined = cesiumInstanceGpu(),
  load: LoadInstances = loadInstances,
): () => void {
  const ref = instancesRefOf((tileset.root as { extras?: unknown } | undefined)?.extras);
  const url = (tileset as unknown as { resource?: { url?: string } }).resource?.url;
  if (!ref || !url || !gpu) return () => undefined;
  let hook: SplatInstances | undefined;
  let disposed = false;
  const push = (): void => {
    const entry = useInstances.getState().assets[assetId];
    if (!hook || !entry) return;
    hook.setState(entry.hidden, entry.highlighted, useInstances.getState().dimOthers);
    scene.requestRender();
  };
  const offStore = useInstances.subscribe((state, previous) => {
    const now = state.assets[assetId];
    const was = previous.assets[assetId];
    if (
      now?.hidden !== was?.hidden ||
      now?.highlighted !== was?.highlighted ||
      state.dimOthers !== previous.dimOthers
    ) {
      push();
    }
  });
  const offUpdate = scene.preUpdate.addEventListener(() => {
    if (hook?.sync().changed) scene.requestRender();
  });
  load(url, ref)
    .then((doc) => {
      if (disposed) return;
      hook = new SplatInstances(doc, gpu, splatTilesetOf(tileset));
      ATTACHED.set(assetId, { tileset, doc });
      useInstances.getState().setTable(assetId, doc);
      push();
      log.info("instances attached", {
        asset: assetId,
        instances: doc.instances.length,
        tiles: doc.tiles.size,
        issues: doc.issues.length,
      });
      if (doc.issues.length > 0) log.warn("instances.json had problems", { first: doc.issues[0] });
    })
    .catch((error: unknown) => {
      log.warn("instances did not load; nothing to search", {
        message: error instanceof Error ? error.message : String(error),
      });
    });
  return () => {
    disposed = true;
    offStore();
    offUpdate();
    hook?.destroy();
    hook = undefined;
    if (ATTACHED.get(assetId)?.tileset === tileset) ATTACHED.delete(assetId);
    useInstances.getState().setTable(assetId, null);
  };
}

/** The splat primitive of a tileset, as the hooks see it. For harnesses and tests. */
export function instancePrimitiveOf(tileset: Cesium3DTileset): SplatPrimitive | undefined {
  return splatTilesetOf(tileset).gaussianSplatPrimitive;
}
