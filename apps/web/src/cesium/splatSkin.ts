/**
 * Moves scene objects by their skins on the GPU (`lib/skin.ts`, docs/SCENE_OBJECTS.md §4).
 *
 * Every skinned splat is displaced in the vertex shader by
 *
 *     x' = x + Σ_j w_j(x) · Z_j · [x − origin; 1]
 *
 * as one part of the motion chain (`splatMotionChain.ts`), so it composes with the Living
 * Survey's rig, the visibility chain (hide, view cones) and the colour hook (highlight): they
 * all see the displaced position. Nothing canonical is written; a skin at rest costs one
 * texel fetch per splat, and with no skin moving none.
 *
 * As the instance ids do (`splatInstances.ts`), each drawn tile is identified by un-baking its
 * positions and digesting them, once per tile load, and its skin ids and weight rows written at
 * its range. Three textures of our own:
 *
 * - **skin ids** (RGBA32UI, 1024 texels a row, four splats a texel): the skin of every splat
 *   index, 0 for none.
 * - **weights** (RGBA32UI, 1024 texels a row, one splat a texel): the splat's row of
 *   `skin.bin` -- sixteen signed bytes, byte `k` handle `k + 1`'s weight. A file of wide rows
 *   (a skin of more than 16 handles) has a second such texture for bytes 16..31.
 * - **handles** (RGBA32F, 1024 texels a row, `TEXELS_PER_SKIN` a skin): texel 0 is
 *   `(moving, handle count, 0, 0)`, then three texels per handle, `Z_j`'s rows **folded into
 *   the baked frame** the shader sees (`foldHandle`): `A_b = L·A·L⁻¹`,
 *   `t_b = L·(t − A·o) − A_b·b` for the bake `B = (L, b)` and the rest origin `o`, so the
 *   shader's whole job is `Σ w_j (A_b x_b + t_b)`. Rewritten when a driver sets handles;
 *   only the rows of the skins that changed are uploaded.
 *
 * **Covariances.** The part also gives the chain its linear part `Σ_j w_j A_b,j`, so each
 * splat's covariance is drawn through `J = I + Σ_j w_j A_j` (the patch's
 * `splatVertexJacobian`). What is dropped is the weights' own gradient, `Σ_j Z_j[x;1] ∇w_jᵀ`:
 * exact for the constant handle (a rigid or affine motion of the whole object), and small for
 * the elastic ones while displacements stay small against the distance over which a weight
 * changes (`support[].radius` in `skin.json`), which is the regime wind drives.
 *
 * **Not re-sorted**: the sorter orders by rest positions, as on the rig's GPU path.
 *
 * **Two drivers at once.** A skin's handles are what its driver set (`setInstanceHandles`: the
 * wind, telemetry) plus an overlay (`setInstanceOverlay`: a poke, `skinPoke.ts`), so a poked
 * branch still sways and the wind lets go of nothing the poke holds.
 */

import { checksumPositions } from "@twin/world";
import type { Cesium3DTileset, Scene } from "cesium";

import { createLogger } from "@/lib/log";
import { loadMaterials, materialsRefOf, type MaterialTable } from "@/lib/skinMaterials";
import {
  HANDLE_FLOATS,
  MAX_SKIN_HANDLES,
  loadSkin,
  rowWeight,
  tileSkin,
  type SkinDoc,
  type SkinEntry,
  type SkinRef,
} from "@/lib/skin";
import { skinRefFor, variantsOf } from "@/lib/variants";
import { onPickChange, pickedVariant, useVariants, type VariantStatus } from "@/state/variants";

import { cesiumPickSource } from "./sceneSelect/cesiumPickSource";
import { CESIUM_PRIORITY, pickSourceOf, registerPickSource } from "./sceneSelect/pickSources";
import { invertAffine, unbakePositions, type Mat4 } from "./splatFrames";
import { cesiumMotionTextures } from "./splatGpuTextures";
import type { MotionTextureFactory, OwnedTexture } from "./splatGpuMotion";
import {
  splatTilesetOf,
  type SplatPrimitive,
  type SplatShaderBuilder,
  type SplatTilesetLike,
} from "./splatInternals";
import {
  addMotionPart,
  hasMotionPart,
  removeMotionPart,
  type SplatMotionPart,
} from "./splatMotionChain";
import { sameMatrix, snapshotTiles, type SnapshotTile } from "./splatTiles";

const log = createLogger("skin");

/** Texels a row of every skin texture holds. */
export const SKIN_TEXTURE_WIDTH = 1024;
/** Splats a skin-id texel holds. */
export const SKIN_IDS_PER_TEXEL = 4;
/** Splats a row of the skin-id texture holds. */
export const SPLATS_PER_SKIN_ID_ROW = SKIN_TEXTURE_WIDTH * SKIN_IDS_PER_TEXEL;
/** Splats a row of the weight texture holds. */
export const SPLATS_PER_WEIGHT_ROW = SKIN_TEXTURE_WIDTH;
/** Handle-texture texels per skin: a header, then three per handle (`MAX_SKIN_HANDLES`). */
export const TEXELS_PER_SKIN = 128;
/** Floats a handle texel holds. */
const FLOATS_PER_TEXEL = 4;
/** How far two tiles' bakes may differ and still share the folded handles. */
const BAKE_TOLERANCE = 1e-9;

/**
 * The shader side: a displacement and its linear part (`splatSkinJacobian`). `wide`: the file's
 * rows are two texels (a second weight texture, `u_skinWeights2`).
 */
export function skinGlsl(scale: number, wide = false): string {
  const width = String(SKIN_TEXTURE_WIDTH - 1);
  const shift = String(Math.log2(SKIN_TEXTURE_WIDTH));
  const second = wide
    ? `texelFetch(u_skinWeights2, ivec2(int(splatIndex & ${width}u), int(splatIndex >> ${shift}u)), 0)`
    : "uvec4(0u)";
  return `
// The skin's linear part at the splat splatSkinMotion last ran for: what the Jacobian returns.
mat3 splatSkinLinear = mat3(0.0);

vec4 splatSkinTexel(int index) {
    return texelFetch(u_skinHandles, ivec2(index & ${width}, index >> ${shift}), 0);
}

float splatSkinWeight(uvec4 words, uvec4 words2, int k) {
    uvec4 row = k < 16 ? words : words2;
    int kk = k & 15;
    uint word = kk < 4 ? row.x : (kk < 8 ? row.y : (kk < 12 ? row.z : row.w));
    // Byte kk & 3 of the word, sign-extended: an int8 weight.
    return float(int(word << uint(24 - 8 * (kk & 3))) >> 24) * ${scale.toPrecision(9)};
}

vec3 splatSkinMotion(uint splatIndex, vec3 position) {
    splatSkinLinear = mat3(0.0);
    if (u_skinActive < 0.5 || float(splatIndex) >= u_skinCovered) {
        return vec3(0.0);
    }
    uint q = splatIndex >> 2u;
    uvec4 ids = texelFetch(u_skinIds, ivec2(int(q & ${width}u), int(q >> ${shift}u)), 0);
    uint k = splatIndex & 3u;
    uint skin = k == 0u ? ids.r : (k == 1u ? ids.g : (k == 2u ? ids.b : ids.a));
    if (skin == 0u || float(skin) > u_skinMaxId) {
        return vec3(0.0);
    }
    int base = int(skin) * ${String(TEXELS_PER_SKIN)};
    vec4 head = splatSkinTexel(base);
    if (head.x < 0.5) {
        return vec3(0.0);
    }
    int handles = int(head.y);
    uvec4 words = texelFetch(
        u_skinWeights,
        ivec2(int(splatIndex & ${width}u), int(splatIndex >> ${shift}u)),
        0
    );
    uvec4 words2 = handles > 17 ? ${second} : uvec4(0u);
    vec4 xh = vec4(position, 1.0);
    vec3 delta = vec3(0.0);
    mat3 linear = mat3(0.0);
    for (int j = 0; j < ${String(MAX_SKIN_HANDLES)}; j++) {
        if (j >= handles) {
            break;
        }
        float w = j == 0 ? 1.0 : splatSkinWeight(words, words2, j - 1);
        if (w == 0.0) {
            continue;
        }
        int at = base + 1 + 3 * j;
        vec4 r0 = splatSkinTexel(at);
        vec4 r1 = splatSkinTexel(at + 1);
        vec4 r2 = splatSkinTexel(at + 2);
        delta += w * vec3(dot(r0, xh), dot(r1, xh), dot(r2, xh));
        // mat3 is column-major: column c holds row entries (r0[c], r1[c], r2[c]).
        linear += w * mat3(r0.x, r1.x, r2.x, r0.y, r1.y, r2.y, r0.z, r1.z, r2.z);
    }
    splatSkinLinear = linear;
    return delta;
}

mat3 splatSkinJacobian(uint splatIndex, vec3 position) {
    return splatSkinLinear;
}
`;
}

/**
 * `Z = [A | t]` (rest frame, about `origin`) folded into the baked frame of `bake`: the three
 * rows of `[A_b | t_b]`, `A_b = L·A·L⁻¹`, `t_b = L·(t − A·o) − A_b·b`.
 */
export function foldHandle(
  handle: ArrayLike<number>,
  at: number,
  origin: readonly [number, number, number],
  bake: Mat4,
  inverse: Mat4,
  out: Float32Array,
  outAt: number,
): void {
  const l = (r: number, c: number): number => bake[c * 4 + r] ?? 0;
  const li = (r: number, c: number): number => inverse[c * 4 + r] ?? 0;
  const a = (r: number, c: number): number => handle[at + r * 4 + c] ?? 0;
  const b = [bake[12] ?? 0, bake[13] ?? 0, bake[14] ?? 0];
  // L·A
  const la: number[] = [];
  for (let r = 0; r < 3; r += 1)
    for (let c = 0; c < 3; c += 1)
      la[r * 3 + c] = l(r, 0) * a(0, c) + l(r, 1) * a(1, c) + l(r, 2) * a(2, c);
  // A_b = (L·A)·L⁻¹
  const ab: number[] = [];
  for (let r = 0; r < 3; r += 1)
    for (let c = 0; c < 3; c += 1)
      ab[r * 3 + c] =
        (la[r * 3] ?? 0) * li(0, c) +
        (la[r * 3 + 1] ?? 0) * li(1, c) +
        (la[r * 3 + 2] ?? 0) * li(2, c);
  // u = t − A·o, then t_b = L·u − A_b·b
  const u = [0, 1, 2].map(
    (r) => a(r, 3) - (a(r, 0) * origin[0] + a(r, 1) * origin[1] + a(r, 2) * origin[2]),
  );
  for (let r = 0; r < 3; r += 1) {
    const o = outAt + r * FLOATS_PER_TEXEL;
    out[o] = ab[r * 3] ?? 0;
    out[o + 1] = ab[r * 3 + 1] ?? 0;
    out[o + 2] = ab[r * 3 + 2] ?? 0;
    out[o + 3] =
      l(r, 0) * (u[0] ?? 0) +
      l(r, 1) * (u[1] ?? 0) +
      l(r, 2) * (u[2] ?? 0) -
      ((ab[r * 3] ?? 0) * (b[0] ?? 0) +
        (ab[r * 3 + 1] ?? 0) * (b[1] ?? 0) +
        (ab[r * 3 + 2] ?? 0) * (b[2] ?? 0));
  }
}

/**
 * What the shader computes for one splat, from the same texture contents, in float64:
 * the displacement and the linear part. A reference for the tests.
 */
export function evaluateSkinMotion(
  handlesTexture: Float32Array,
  skin: number,
  words: ArrayLike<number>,
  scale: number,
  position: readonly [number, number, number],
  words2?: ArrayLike<number>,
): { displacement: [number, number, number]; linear: number[] } {
  const texel = (index: number): number[] =>
    [0, 1, 2, 3].map((k) => handlesTexture[index * FLOATS_PER_TEXEL + k] ?? 0);
  const displacement: [number, number, number] = [0, 0, 0];
  const linear = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  if (skin === 0) return { displacement, linear };
  const base = skin * TEXELS_PER_SKIN;
  const head = texel(base);
  if ((head[0] ?? 0) < 0.5) return { displacement, linear };
  const handles = head[1] ?? 0;
  for (let j = 0; j < Math.min(handles, MAX_SKIN_HANDLES); j += 1) {
    const w = j === 0 ? 1 : rowWeight(words, 0, j - 1, scale, words2);
    if (w === 0) continue;
    for (let r = 0; r < 3; r += 1) {
      const row = texel(base + 1 + 3 * j + r);
      displacement[r] =
        (displacement[r] ?? 0) +
        w *
          ((row[0] ?? 0) * position[0] +
            (row[1] ?? 0) * position[1] +
            (row[2] ?? 0) * position[2] +
            (row[3] ?? 0));
      for (let c = 0; c < 3; c += 1)
        linear[r * 3 + c] = (linear[r * 3 + c] ?? 0) + w * (row[c] ?? 0);
    }
  }
  return { displacement, linear };
}

/** One tile's skin data under one bake: `undefined` when the file does not list the tile. */
interface TileData {
  readonly bake: readonly number[];
  readonly skins: Uint32Array | undefined;
  readonly words: Uint32Array | undefined;
  readonly words2: Uint32Array | undefined;
}

/** What `sync` did, for diagnostics and tests. */
export interface SkinSyncResult {
  readonly changed: boolean;
  readonly tiles: number;
  readonly matched: number;
}

/** The skin part of one scan's motion chain. */
export class SplatSkinning implements SplatMotionPart {
  readonly motionFunction = "splatSkinMotion";
  #covariance = true;
  /** After the rig: the order is immaterial (both see the rest position). */
  readonly motionOrder = 10;
  readonly doc: SkinDoc;
  readonly #factory: MotionTextureFactory;
  readonly #tileset: SplatTilesetLike;
  readonly #cache = new WeakMap<object, TileData>();
  #primitive: SplatPrimitive | undefined;
  #context: unknown;
  #ids = new Uint32Array(SPLATS_PER_SKIN_ID_ROW);
  #words = new Uint32Array(SPLATS_PER_WEIGHT_ROW * 4);
  /** The second texel of a wide row (bytes 16..31), only for a file of wide rows. */
  #words2: Uint32Array | undefined;
  #idTexture: OwnedTexture | undefined;
  #weightTexture: OwnedTexture | undefined;
  #weightTexture2: OwnedTexture | undefined;
  #handleTexture: OwnedTexture | undefined;
  #idRows = 0;
  #weightRows = 0;
  #weightTexture2Rows = 0;
  /** Splat indices written since the last upload, `[start, end)`. */
  #dirty: [number, number] | undefined;
  readonly #handles: Float32Array;
  /** Per skin id, its handles as last set (rest frame), for re-folding under a new bake. */
  readonly #driven = new Map<number, Float64Array>();
  /** Per skin id, what its driver set (`setHandles`) and what is laid over it (a poke). */
  readonly #base = new Map<number, Float64Array>();
  readonly #overlay = new Map<number, Float64Array>();
  /** Skin ids whose texels changed since the last upload. */
  readonly #handlesDirty = new Set<number>();
  #handleTextureDirty = true;
  #placed = new Map<object, { start: number; count: number }>();
  #generation = -1;
  #positions: Float32Array | undefined;
  #incremental = false;
  #covered = 0;
  #motionVersion = 0;
  #bake: number[] | undefined;
  #inverse: number[] | undefined;
  readonly #unlisted = new Set<string>();
  /** Off draws every splat at rest; the part stays installed. */
  enabled = true;
  /**
   * The bake-off candidate this skin is (`extras.variants.skins`'s name), or null for the scan's
   * own: a candidate moves its scan instead of a Living Survey rig (`LivingSurveyManager`).
   */
  variant: string | null = null;
  /**
   * Fitted materials by instance (`materials.json`, `lib/skinMaterials.ts`), once loaded: what
   * a driver reads over its property priors. Empty while absent.
   */
  materials: MaterialTable = new Map();

  constructor(doc: SkinDoc, factory: MotionTextureFactory, tileset: SplatTilesetLike) {
    this.doc = doc;
    this.#factory = factory;
    this.#tileset = tileset;
    this.#handles = new Float32Array(
      SKIN_TEXTURE_WIDTH * handleTextureRows(doc.maxId) * FLOATS_PER_TEXEL,
    );
    if (doc.rowWords === 8) this.#words2 = new Uint32Array(SPLATS_PER_WEIGHT_ROW * 4);
  }

  /** The skin id each splat index carries now. */
  skinAt(splatIndex: number): number {
    return splatIndex < this.#covered ? (this.#ids[splatIndex] ?? 0) : 0;
  }

  /** The weight row each splat index carries now (four words; eight in a file of wide rows). */
  wordsAt(splatIndex: number): Uint32Array {
    const first = this.#words.slice(splatIndex * 4, splatIndex * 4 + 4);
    if (!this.#words2) return first;
    const out = new Uint32Array(8);
    out.set(first);
    out.set(this.#words2.subarray(splatIndex * 4, splatIndex * 4 + 4), 4);
    return out;
  }

  /** The handle texels as they would be uploaded, for tests. */
  get handleData(): Float32Array {
    return this.#handles;
  }

  /** The bake the handles are folded under, once a snapshot is bound. */
  get bake(): readonly number[] | undefined {
    return this.#bake;
  }

  /** The linear part's function, while covariances follow the skin (`covariance`). */
  get jacobianFunction(): string | undefined {
    return this.#covariance ? "splatSkinJacobian" : undefined;
  }

  /**
   * Whether splats' covariances follow the skin's linear part (the default). Off draws them
   * as measured, at their displaced positions: for comparing. Changing it rebuilds the draw
   * command.
   */
  get covariance(): boolean {
    return this.#covariance;
  }

  set covariance(value: boolean) {
    if (value === this.#covariance) return;
    this.#covariance = value;
    this.#motionVersion += 1;
    const primitive = this.#primitive;
    if (primitive && hasMotionPart(primitive, this)) {
      removeMotionPart(primitive, this);
      addMotionPart(primitive, this);
    }
  }

  /** Whether the shader would act this frame. */
  get active(): boolean {
    return this.#drawActive();
  }

  /**
   * Bumped whenever a driver sets handles or covariances are switched: what a dedicated splat
   * renderer polls to redraw the skins it applies itself (`scanView/scanMotion.ts`).
   */
  get motionVersion(): number {
    return this.#motionVersion;
  }

  /**
   * The handles every driven skin holds now, by skin id (rest frame, as `setHandles` took
   * them; a new array each time they are set): what any renderer draws the skins from.
   */
  get drivenSkins(): ReadonlyMap<number, Float64Array> {
    return this.#driven;
  }

  /** Whether any skin is moving. */
  get moving(): boolean {
    return this.#driven.size > 0;
  }

  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void {
    this.#context = context;
    this.#ensureTextures();
    const vertex = this.#factory.vertexDestination;
    shaderBuilder.addUniform("highp usampler2D", "u_skinIds", vertex);
    shaderBuilder.addUniform("highp usampler2D", "u_skinWeights", vertex);
    if (this.#words2) shaderBuilder.addUniform("highp usampler2D", "u_skinWeights2", vertex);
    shaderBuilder.addUniform("highp sampler2D", "u_skinHandles", vertex);
    shaderBuilder.addUniform("float", "u_skinActive", vertex);
    shaderBuilder.addUniform("float", "u_skinMaxId", vertex);
    shaderBuilder.addUniform("float", "u_skinCovered", vertex);
    shaderBuilder.addVertexLines(skinGlsl(this.doc.scale, this.#words2 !== undefined));
    uniformMap.u_skinIds = () => this.#idTexture;
    uniformMap.u_skinWeights = () => this.#weightTexture;
    if (this.#words2) uniformMap.u_skinWeights2 = () => this.#weightTexture2;
    uniformMap.u_skinHandles = () => this.#handleTexture;
    uniformMap.u_skinActive = () => (this.#drawActive() ? 1 : 0);
    uniformMap.u_skinMaxId = () => this.doc.maxId;
    uniformMap.u_skinCovered = () => this.#covered;
  }

  install(primitive: SplatPrimitive): boolean {
    if (this.#primitive && this.#primitive !== primitive) this.uninstall();
    this.#primitive = primitive;
    return hasMotionPart(primitive, this) || addMotionPart(primitive, this);
  }

  get installed(): boolean {
    return hasMotionPart(this.#primitive, this);
  }

  uninstall(): void {
    const primitive = this.#primitive;
    this.#primitive = undefined;
    if (primitive) removeMotionPart(primitive, this);
  }

  /**
   * Sets skin `skinId`'s handles: `handles` holds `Z_j` row-major (12 numbers a handle, rest
   * frame, about the skin's origin) for `j = 0..m−1`; fewer leave the rest at rest. `null`
   * puts the skin back at rest. The driver's whole interface (wind, telemetry, a slider).
   */
  setHandles(skinId: number, handles: ArrayLike<number> | null): void {
    this.#setLayer(this.#base, skinId, handles);
  }

  /**
   * Lays `handles` over skin `skinId`'s (what a poke adds to what the wind set: the two sum),
   * or `null` to take the overlay away. The same rest frame and layout as `setHandles`.
   */
  setOverlay(skinId: number, handles: ArrayLike<number> | null): void {
    this.#setLayer(this.#overlay, skinId, handles);
  }

  /** `setOverlay` for the skin of instance `instanceId`. False when it has none. */
  setInstanceOverlay(instanceId: number, handles: ArrayLike<number> | null): boolean {
    const skin = this.doc.byInstance.get(instanceId);
    if (!skin) return false;
    this.setOverlay(skin.id, handles);
    return true;
  }

  #setLayer(layer: Map<number, Float64Array>, skinId: number, handles: ArrayLike<number> | null) {
    const skin = this.doc.byId.get(skinId);
    if (!skin) return;
    // The same handles again (a driver ticking at a held clock) change nothing: no version
    // bump, so a dedicated renderer does not regenerate and re-sort for them.
    const current = layer.get(skinId);
    if (handles === null) {
      if (current === undefined) return;
      layer.delete(skinId);
    } else {
      const copy = new Float64Array(skin.handles * HANDLE_FLOATS);
      for (let i = 0; i < Math.min(copy.length, handles.length); i += 1) copy[i] = handles[i] ?? 0;
      if (current?.every((v, i) => Object.is(v, copy[i]))) return;
      layer.set(skinId, copy);
    }
    // What is drawn: the driver's handles plus the overlay (a new array each time).
    const base = this.#base.get(skinId);
    const over = this.#overlay.get(skinId);
    if (base === undefined && over === undefined) this.#driven.delete(skinId);
    else if (over === undefined) this.#driven.set(skinId, base ?? new Float64Array(0));
    else if (base === undefined) this.#driven.set(skinId, over);
    else
      this.#driven.set(
        skinId,
        base.map((v, i) => v + (over[i] ?? 0)),
      );
    this.#motionVersion += 1;
    this.#writeSkin(skin);
    this.#ensureTextures();
  }

  /** `setHandles` for the skin of instance `instanceId`. False when it has none. */
  setInstanceHandles(instanceId: number, handles: ArrayLike<number> | null): boolean {
    const skin = this.doc.byInstance.get(instanceId);
    if (!skin) return false;
    this.setHandles(skin.id, handles);
    return true;
  }

  /**
   * The handles instance `instanceId`'s driver last set (rest frame), or null at rest; an
   * overlay (`setInstanceOverlay`) is not included.
   */
  instanceHandles(instanceId: number): Float64Array | null {
    const skin = this.doc.byInstance.get(instanceId);
    const driven = skin ? this.#base.get(skin.id) : undefined;
    return driven ? driven.slice() : null;
  }

  /** What is laid over instance `instanceId`'s handles (a poke), or null. */
  instanceOverlay(instanceId: number): Float64Array | null {
    const skin = this.doc.byInstance.get(instanceId);
    const over = skin ? this.#overlay.get(skin.id) : undefined;
    return over ? over.slice() : null;
  }

  /** Every skin back at rest, overlays included. */
  rest(): void {
    for (const id of [...this.#base.keys()]) this.setHandles(id, null);
    for (const id of [...this.#overlay.keys()]) this.setOverlay(id, null);
  }

  /**
   * Brings the ids and weights up to the primitive's committed snapshot, and uploads what
   * changed. Call once a frame, before the render.
   */
  sync(): SkinSyncResult {
    const unchanged = { changed: false, tiles: this.#placed.size, matched: 0 };
    const primitive = this.#tileset.gaussianSplatPrimitive;
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
    const keep = incremental && this.#incremental;
    if (!keep) {
      this.#ids.fill(0);
      this.#words.fill(0);
      this.#words2?.fill(0);
      this.#mark(0, Math.max(this.#covered, numSplats));
      this.#placed.clear();
    } else {
      for (const [content, slot] of this.#placed) {
        const now = next.get(content);
        if (now?.start === slot.start && now.count === slot.count) continue;
        this.#clear(slot.start, slot.count);
      }
    }
    // The handles are folded under one bake: the first tile's, which every tile shares when
    // the tileset carries no per-tile transforms (the packer's output).
    const first = listed.tiles[0];
    if (first !== undefined && (this.#bake === undefined || !sameMatrix(this.#bake, first.bake))) {
      const inverse = invertAffine(first.bake);
      if (inverse !== undefined) {
        this.#bake = Array.from(first.bake);
        this.#inverse = inverse;
        for (const skin of this.doc.skins) this.#writeSkin(skin);
      }
    }
    let matched = 0;
    for (const tile of listed.tiles) {
      const before = this.#placed.get(tile.content);
      const data = this.#tileData(tile, positions);
      if (data.skins !== undefined) matched += 1;
      if (before?.start === tile.start && before.count === tile.count && keep) continue;
      if (data.skins !== undefined && data.words !== undefined) {
        this.#ids.set(data.skins, tile.start);
        this.#words.set(data.words, tile.start * 4);
        if (data.words2) this.#words2?.set(data.words2, tile.start * 4);
        this.#mark(tile.start, tile.start + tile.count);
      } else {
        this.#clear(tile.start, tile.count);
      }
    }
    this.#placed = next;
    this.#incremental = incremental;
    this.#covered = numSplats;
    this.#generation = generation;
    this.#positions = positions;
    this.#ensureTextures();
    return { changed: true, tiles: listed.tiles.length, matched };
  }

  destroy(): void {
    this.uninstall();
    for (const texture of [
      this.#idTexture,
      this.#weightTexture,
      this.#weightTexture2,
      this.#handleTexture,
    ]) {
      if (texture && !texture.isDestroyed()) texture.destroy();
    }
    this.#idTexture = undefined;
    this.#weightTexture = undefined;
    this.#weightTexture2 = undefined;
    this.#handleTexture = undefined;
  }

  #tileData(tile: SnapshotTile, positions: Float32Array): TileData {
    const cached = this.#cache.get(tile.content);
    if (cached && sameMatrix(cached.bake, tile.bake)) return cached;
    const none: TileData = {
      bake: Array.from(tile.bake),
      skins: undefined,
      words: undefined,
      words2: undefined,
    };
    const inverse = invertAffine(tile.bake);
    if (inverse === undefined) return none;
    const reference = this.#bake;
    if (reference !== undefined && !closeMatrix(reference, tile.bake)) {
      log.warn("a tile has its own transform; its skinned splats stay still", {});
      this.#cache.set(tile.content, none);
      return none;
    }
    const baked = positions.subarray(tile.start * 3, (tile.start + tile.count) * 3);
    const checksum = checksumPositions(unbakePositions(baked, inverse));
    const decoded = tileSkin(this.doc, checksum);
    const fits = decoded?.skins.length === tile.count;
    if (!fits && !this.#unlisted.has(checksum)) {
      this.#unlisted.add(checksum);
      log.info("a drawn tile is not in skin.json; its splats carry no skin", { checksum });
    }
    const data: TileData = {
      bake: Array.from(tile.bake),
      skins: fits ? decoded.skins : undefined,
      words: fits ? decoded.words : undefined,
      words2: fits ? decoded.words2 : undefined,
    };
    this.#cache.set(tile.content, data);
    return data;
  }

  /** Writes one skin's header and folded handles into the CPU texels. */
  #writeSkin(skin: SkinEntry): void {
    const base = skin.id * TEXELS_PER_SKIN * FLOATS_PER_TEXEL;
    this.#handles.fill(0, base, base + TEXELS_PER_SKIN * FLOATS_PER_TEXEL);
    const driven = this.#driven.get(skin.id);
    const bake = this.#bake;
    const inverse = this.#inverse;
    if (driven !== undefined && bake !== undefined && inverse !== undefined) {
      this.#handles[base] = 1;
      this.#handles[base + 1] = skin.handles;
      for (let j = 0; j < skin.handles; j += 1) {
        foldHandle(
          driven,
          j * HANDLE_FLOATS,
          skin.origin,
          bake,
          inverse,
          this.#handles,
          base + (1 + 3 * j) * FLOATS_PER_TEXEL,
        );
      }
    }
    this.#handlesDirty.add(skin.id);
  }

  #clear(start: number, count: number): void {
    this.#ids.fill(0, start, start + count);
    this.#words.fill(0, start * 4, (start + count) * 4);
    this.#words2?.fill(0, start * 4, (start + count) * 4);
    this.#mark(start, start + count);
  }

  #reserve(splats: number): void {
    const idNeeded = rowsFor(splats, SPLATS_PER_SKIN_ID_ROW) * SPLATS_PER_SKIN_ID_ROW;
    if (idNeeded > this.#ids.length) {
      const grown = new Uint32Array(
        rowsFor(Math.max(splats, Math.ceil((this.#ids.length * 3) / 2)), SPLATS_PER_SKIN_ID_ROW) *
          SPLATS_PER_SKIN_ID_ROW,
      );
      grown.set(this.#ids);
      this.#ids = grown;
    }
    const wordNeeded = rowsFor(splats, SPLATS_PER_WEIGHT_ROW) * SPLATS_PER_WEIGHT_ROW * 4;
    if (wordNeeded > this.#words.length) {
      const grown = new Uint32Array(
        rowsFor(Math.max(splats, Math.ceil((this.#words.length * 3) / 8)), SPLATS_PER_WEIGHT_ROW) *
          SPLATS_PER_WEIGHT_ROW *
          4,
      );
      grown.set(this.#words);
      this.#words = grown;
      if (this.#words2) {
        const second = new Uint32Array(grown.length);
        second.set(this.#words2);
        this.#words2 = second;
      }
    }
  }

  #mark(start: number, end: number): void {
    if (end <= start) return;
    const dirty = this.#dirty;
    this.#dirty = dirty ? [Math.min(dirty[0], start), Math.max(dirty[1], end)] : [start, end];
  }

  /** Creates what is missing and uploads what changed; true when anything was uploaded. */
  #ensureTextures(): boolean {
    const context = this.#context;
    if (context === undefined) return false;
    let uploaded = false;
    const idRows = this.#ids.length / SPLATS_PER_SKIN_ID_ROW;
    const weightRows = this.#words.length / (SPLATS_PER_WEIGHT_ROW * 4);
    const dirty = this.#dirty;
    if (this.#idTexture === undefined || this.#idTexture.isDestroyed() || this.#idRows !== idRows) {
      this.#idTexture?.destroy();
      this.#idTexture = this.#factory.createUintQuads(
        context,
        SKIN_TEXTURE_WIDTH,
        idRows,
        this.#ids,
      );
      this.#idRows = idRows;
      uploaded = true;
    } else if (dirty !== undefined) {
      uploadRows(this.#idTexture, this.#ids, dirty, SPLATS_PER_SKIN_ID_ROW, SPLATS_PER_SKIN_ID_ROW);
      uploaded = true;
    }
    if (
      this.#weightTexture === undefined ||
      this.#weightTexture.isDestroyed() ||
      this.#weightRows !== weightRows
    ) {
      this.#weightTexture?.destroy();
      this.#weightTexture = this.#factory.createUintQuads(
        context,
        SKIN_TEXTURE_WIDTH,
        weightRows,
        this.#words,
      );
      this.#weightRows = weightRows;
      uploaded = true;
    } else if (dirty !== undefined) {
      uploadRows(
        this.#weightTexture,
        this.#words,
        dirty,
        SPLATS_PER_WEIGHT_ROW,
        SPLATS_PER_WEIGHT_ROW * 4,
      );
      uploaded = true;
    }
    const second = this.#words2;
    if (second) {
      if (
        this.#weightTexture2 === undefined ||
        this.#weightTexture2.isDestroyed() ||
        this.#weightTexture2Rows !== weightRows
      ) {
        this.#weightTexture2?.destroy();
        this.#weightTexture2 = this.#factory.createUintQuads(
          context,
          SKIN_TEXTURE_WIDTH,
          weightRows,
          second,
        );
        this.#weightTexture2Rows = weightRows;
        uploaded = true;
      } else if (dirty !== undefined) {
        uploadRows(
          this.#weightTexture2,
          second,
          dirty,
          SPLATS_PER_WEIGHT_ROW,
          SPLATS_PER_WEIGHT_ROW * 4,
        );
        uploaded = true;
      }
    }
    this.#dirty = undefined;
    const handleRows = handleTextureRows(this.doc.maxId);
    if (
      this.#handleTexture === undefined ||
      this.#handleTexture.isDestroyed() ||
      this.#handleTextureDirty
    ) {
      this.#handleTexture?.destroy();
      this.#handleTexture = this.#factory.createFloat(
        context,
        SKIN_TEXTURE_WIDTH,
        handleRows,
        this.#handles,
      );
      this.#handleTextureDirty = false;
      this.#handlesDirty.clear();
      uploaded = true;
    } else if (this.#handlesDirty.size > 0) {
      const perRow = SKIN_TEXTURE_WIDTH / TEXELS_PER_SKIN;
      let first = Infinity;
      let last = -1;
      for (const id of this.#handlesDirty) {
        first = Math.min(first, Math.floor(id / perRow));
        last = Math.max(last, Math.floor(id / perRow));
      }
      const words = SKIN_TEXTURE_WIDTH * FLOATS_PER_TEXEL;
      this.#handleTexture.copyFrom({
        source: {
          width: SKIN_TEXTURE_WIDTH,
          height: last - first + 1,
          arrayBufferView: this.#handles.subarray(first * words, (last + 1) * words),
        },
        xOffset: 0,
        yOffset: first,
      });
      this.#handlesDirty.clear();
      uploaded = true;
    }
    return uploaded;
  }

  #drawActive(): boolean {
    if (!this.enabled || this.#driven.size === 0) return false;
    if (!this.#idTexture || !this.#weightTexture || !this.#handleTexture) return false;
    if (this.#words2 && !this.#weightTexture2) return false;
    if (this.#generation < 0 || this.#bake === undefined) return false;
    const primitive = this.#primitive;
    if (!primitive) return false;
    return this.#incremental || primitive._snapshot?.generation === this.#generation;
  }
}

/** Rows of a texture holding `count` items, `perRow` a row (at least one). */
function rowsFor(count: number, perRow: number): number {
  return Math.max(1, Math.ceil(count / perRow));
}

/** Rows of the handle texture for skin ids `0..maxId`. */
export function handleTextureRows(maxId: number): number {
  return rowsFor((maxId + 1) * TEXELS_PER_SKIN, SKIN_TEXTURE_WIDTH);
}

function uploadRows(
  texture: OwnedTexture,
  data: Uint32Array,
  [start, end]: [number, number],
  splatsPerRow: number,
  wordsPerRow: number,
): void {
  const first = Math.floor(start / splatsPerRow);
  const last = Math.floor((end - 1) / splatsPerRow);
  texture.copyFrom({
    source: {
      width: SKIN_TEXTURE_WIDTH,
      height: last - first + 1,
      arrayBufferView: data.subarray(first * wordsPerRow, (last + 1) * wordsPerRow),
    },
    xOffset: 0,
    yOffset: first,
  });
}

function closeMatrix(a: Mat4, b: Mat4): boolean {
  for (let i = 0; i < 16; i += 1) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    if (Math.abs(x - y) > BAKE_TOLERANCE * Math.max(1, Math.abs(x), Math.abs(y))) return false;
  }
  return true;
}

// ---- Attachment --------------------------------------------------------------------------

/** Attached skins, by asset id: what a driver (wind, telemetry, a harness) moves. */
const ATTACHED = new Map<string, SplatSkinning>();

/** The skin part of `assetId`'s scan, once its skin has loaded. */
export function skinningOf(assetId: string): SplatSkinning | undefined {
  return ATTACHED.get(assetId);
}

/** Every attached skin part, by asset id: what the wind drives. */
export function attachedSkins(): ReadonlyMap<string, SplatSkinning> {
  return ATTACHED;
}

const SKIN_LISTENERS = new Set<() => void>();

/** Calls `listener` whenever a skin part is attached or detached; returns the unsubscriber. */
export function onSkinsChanged(listener: () => void): () => void {
  SKIN_LISTENERS.add(listener);
  return () => SKIN_LISTENERS.delete(listener);
}

function skinsChanged(): void {
  for (const listener of [...SKIN_LISTENERS]) listener();
}

export type LoadSkin = typeof loadSkin;

/**
 * Lets `tileset`'s objects move by their skins, when its root declares `extras.skin` or offers
 * skins variants (`extras.variants.skins`, lib/variants.ts): the files are fetched and the part
 * joins the splat primitive's motion chain, synced each frame. Returns the disposer. A tileset
 * without a skin, or an engine without the hook, costs nothing. Drivers find the part with
 * `skinningOf(assetId)`.
 *
 * Which skin is the pick's (`state/variants.ts`): Today's `extras.skin`, or a variant's
 * `skin.json` and `skin.bin`. Picking another loads it and replaces the part; the drivers
 * follow the attached parts (`onSkinsChanged`: the wind makes a driver for the new one, and
 * keeps blowing), and a dedicated renderer rebinds its tiles to the new document
 * (scanMotion.ts). The scan's fitted materials (`extras.materials`) are keyed by instance,
 * not by skin, and go with whichever skin is drawn.
 */
export function attachSkin(
  tileset: Cesium3DTileset,
  scene: Pick<Scene, "preUpdate" | "requestRender">,
  assetId: string,
  factory: MotionTextureFactory | undefined = cesiumMotionTextures(),
  load: LoadSkin = loadSkin,
): () => void {
  const extras = (tileset.root as { extras?: unknown } | undefined)?.extras;
  const url = (tileset as unknown as { resource?: { url?: string } }).resource?.url;
  const variants = variantsOf(extras);
  const offersVariants = variants.skins.length > 0;
  const current = (): { ref: SkinRef | null; variant: string | null } => {
    const picked = offersVariants ? pickedVariant(assetId, "skins", variants) : null;
    return { ref: skinRefFor(extras, picked), variant: picked?.name ?? null };
  };
  if (!url || !factory || (current().ref === null && !offersVariants)) return () => undefined;
  let part: SplatSkinning | undefined;
  let disposed = false;
  const offUpdate = scene.preUpdate.addEventListener(() => {
    if (part?.sync().changed) scene.requestRender();
  });
  // A scan without objects has no pick source of CesiumJS's splats; the poke picks from one.
  const offPick = pickSourceOf(assetId)
    ? () => undefined
    : registerPickSource(assetId, cesiumPickSource(tileset), CESIUM_PRIORITY);
  const report = (status: VariantStatus | null): void => {
    if (offersVariants) useVariants.getState().setStatus(assetId, "skins", status);
  };
  // The fitted materials, loaded once: what a driver reads over its priors, whichever skin.
  let materials: MaterialTable | undefined;
  const materialsRef = materialsRefOf(extras);
  if (materialsRef) {
    loadMaterials(url, materialsRef)
      .then((table) => {
        if (disposed) return;
        materials = table;
        if (part) part.materials = table;
        log.info("skin materials attached", { asset: assetId, records: table.size });
      })
      .catch((error: unknown) => {
        log.warn("skin materials did not load; objects sway on their priors", {
          message: error instanceof Error ? error.message : String(error),
        });
      });
  }
  /** Takes the part off the primitive and out of the drivers' reach. */
  const detach = (): void => {
    const was = part;
    part = undefined;
    if (was === undefined) return;
    was.destroy();
    if (ATTACHED.get(assetId) === was) {
      ATTACHED.delete(assetId);
      skinsChanged();
    }
  };
  /** The skin drawn now (its uri, "" for none), and which load is the latest. */
  let shown: string | null = null;
  let serial = 0;
  const follow = (): void => {
    const { ref, variant } = current();
    const key = ref?.uri ?? "";
    if (key === shown) return;
    const swapping = shown !== null;
    shown = key;
    const mine = ++serial;
    if (ref === null) {
      detach();
      scene.requestRender();
      report({ state: "ready" });
      return;
    }
    report({ state: "loading" });
    load(url, ref)
      .then((doc) => {
        if (disposed || mine !== serial) return;
        // Replaced, not edited: the drivers key on the part (skinWind.ts, scanMotion.ts).
        const before = part;
        part = undefined;
        before?.destroy();
        const attached = new SplatSkinning(doc, factory, splatTilesetOf(tileset));
        if (materials) attached.materials = materials;
        attached.variant = variant;
        part = attached;
        ATTACHED.set(assetId, attached);
        skinsChanged();
        scene.requestRender();
        report({ state: "ready" });
        log.info("skin attached", {
          asset: assetId,
          variant,
          skins: doc.skins.length,
          tiles: doc.tiles.size,
          rows: doc.rows,
        });
        if (doc.issues.length > 0) log.warn("skin.json had problems", { first: doc.issues[0] });
      })
      .catch((error: unknown) => {
        if (disposed || mine !== serial) return;
        const message = error instanceof Error ? error.message : String(error);
        log.warn("skin did not load; objects stay still", { variant, message });
        // A pick that did not load is not drawn as if it had: nothing moves, and the panel says why.
        if (swapping) {
          detach();
          scene.requestRender();
        }
        report({ state: "error", message });
      });
  };
  const offVariant = offersVariants ? onPickChange(assetId, "skins", follow) : () => undefined;
  follow();
  return () => {
    disposed = true;
    offVariant();
    offUpdate();
    offPick();
    detach();
    report(null);
  };
}
