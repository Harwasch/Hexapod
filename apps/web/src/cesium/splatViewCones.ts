/**
 * Fades the parts of a splat scan seen from where the capture never looked.
 *
 * The packer writes, beside each tileset, a coarse grid of the directions each part of the
 * scan was seen from (`lib/viewCones.ts`, `tools/captures/view_cones.py`). This installs the
 * engine patch's `vertexVisibility` hook on the tileset's splat primitive (as one part of its
 * visibility chain, `splatVisibility.ts`, beside e.g. hidden instances): per splat, its
 * cell's cone against the direction the camera sees it from, a weight on its opacity. Inside
 * the cone nothing changes, so every view the capture had is drawn exactly as before.
 *
 * One texture of our own (RGBA8, `VIEW_CONES_TEXTURE_WIDTH` texels a row, uploaded once) and
 * four uniforms; per vertex one texel fetch and a few dozen flops, after the frustum test, so
 * splats off screen cost nothing more. It is independent of the motion hook (`vertexMotion`),
 * and is given the displaced position, so a swaying tree is judged where it is drawn.
 *
 * The scan's frame: a splat's position in the shader is in the draw command's model frame,
 * the frame its tile was baked into (`tile.content._lastSplatTransform`, `B`); the grid is in
 * the scan's local ENU metres, `B⁻¹ · p`. Every tile of a `splat_tiles.py` tileset shares one
 * `B`, read off the first selected tile that has one; until one has, the hook draws everything.
 *
 * Off with `?viewCones=off` in the page's URL, to compare.
 */

import {
  Cartesian4,
  Matrix4,
  PixelDatatype,
  PixelFormat,
  Sampler,
  ShaderDestination,
  Texture,
  type Cesium3DTileset,
} from "cesium";

import { createLogger } from "@/lib/log";
import {
  gridFromModel,
  loadViewCones,
  textureRows,
  VIEW_CONES_GLSL,
  VIEW_CONES_TEXTURE_WIDTH,
  viewConesMetaOf,
  type ViewConesMeta,
} from "@/lib/viewCones";

import { invertAffine, type Mat4 } from "./splatFrames";
import type { OwnedTexture } from "./splatGpuMotion";
import { splatTilesetOf, type SplatShaderBuilder, type SplatTile } from "./splatInternals";
import {
  addVisibilityPart,
  hasVisibilityPart,
  removeVisibilityPart,
  type SplatVisibilityPart,
  type VisibilityPrimitive,
} from "./splatVisibility";

export type { SplatVertexVisibility, VisibilityPrimitive } from "./splatVisibility";

const log = createLogger("viewCones");

/** How the hook makes its texture and uniform values. The real one wraps CesiumJS. */
export interface ViewConeGpu {
  /** An `RGBA8` texture, read normalised. */
  createBytes(context: unknown, width: number, height: number, data: Uint8Array): OwnedTexture;
  /** A `mat4` uniform's value from 16 column-major numbers. */
  mat4(values: readonly number[]): unknown;
  /** A `vec4` uniform's value. */
  vec4(x: number, y: number, z: number, w: number): unknown;
  /** `ShaderDestination.VERTEX`. */
  readonly vertexDestination: number;
}

/** Whether this page asked for the fade off (`?viewCones=off`). */
export function viewConesEnabled(search: string = globalThis.location?.search ?? ""): boolean {
  return new URLSearchParams(search).get("viewCones") !== "off";
}

/** The visibility hook for one scan: a part of its primitive's visibility chain. */
export class SplatViewCones implements SplatVisibilityPart {
  readonly visibilityFunction = "splatViewConeVisibility";
  /** After cheaper parts (a hidden instance is one texel fetch; a cone is two and an acos). */
  readonly visibilityOrder = 10;
  readonly meta: ViewConesMeta;
  readonly #rows: { data: Uint8Array; height: number };
  readonly #gpu: ViewConeGpu;
  #texture: OwnedTexture | undefined;
  #primitive: VisibilityPrimitive | undefined;
  #bake: number[] | undefined;
  #matrix: unknown;
  #identity: unknown;
  /** Off draws every splat at its own opacity; the hook stays installed. */
  enabled = true;

  constructor(meta: ViewConesMeta, texels: Uint8Array, gpu: ViewConeGpu) {
    this.meta = meta;
    this.#rows = textureRows(texels);
    this.#gpu = gpu;
  }

  /** Called (through the chain) by the patched engine on every draw-command build. */
  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void {
    if (this.#texture === undefined || this.#texture.isDestroyed()) {
      this.#texture = this.#gpu.createBytes(
        context,
        VIEW_CONES_TEXTURE_WIDTH,
        this.#rows.height,
        this.#rows.data,
      );
    }
    const vertex = this.#gpu.vertexDestination;
    shaderBuilder.addUniform("highp sampler2D", "u_viewCones", vertex);
    shaderBuilder.addUniform("mat4", "u_viewConeFromModel", vertex);
    shaderBuilder.addUniform("vec4", "u_viewConeDims", vertex);
    shaderBuilder.addUniform("float", "u_viewConeFade", vertex);
    shaderBuilder.addUniform("float", "u_viewConeActive", vertex);
    shaderBuilder.addVertexLines(VIEW_CONES_GLSL);
    const [nx, ny, nz] = this.meta.dims;
    const dims = this.#gpu.vec4(nx, ny, nz, VIEW_CONES_TEXTURE_WIDTH);
    uniformMap.u_viewCones = () => this.#texture;
    uniformMap.u_viewConeFromModel = () => this.#matrixValue();
    uniformMap.u_viewConeDims = () => dims;
    uniformMap.u_viewConeFade = () => this.meta.fadeDeg;
    uniformMap.u_viewConeActive = () => (this.enabled && this.#readBake() ? 1 : 0);
  }

  /** Installs the hook on `primitive`; false when this engine has no `vertexVisibility`. */
  install(primitive: VisibilityPrimitive): boolean {
    if (this.#primitive === primitive && hasVisibilityPart(primitive, this)) return true;
    if (this.#primitive && this.#primitive !== primitive) this.uninstall();
    if (!addVisibilityPart(primitive, this)) return false;
    this.#primitive = primitive;
    return true;
  }

  /** Whether the hook is in its primitive's visibility chain. */
  get installed(): boolean {
    return hasVisibilityPart(this.#primitive, this);
  }

  /** Takes the hook off its primitive, keeping the texture for a later `install`. */
  uninstall(): void {
    if (this.#primitive) removeVisibilityPart(this.#primitive, this);
    this.#primitive = undefined;
  }

  /** Takes the hook off its primitive and frees the texture. */
  destroy(): void {
    this.uninstall();
    if (this.#texture && !this.#texture.isDestroyed()) this.#texture.destroy();
    this.#texture = undefined;
  }

  /** The grid-from-model matrix of the current bake, rebuilt only when the bake changes. */
  #matrixValue(): unknown {
    if (!this.#readBake() || this.#matrix === undefined) {
      this.#identity ??= this.#gpu.mat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
      return this.#identity;
    }
    return this.#matrix;
  }

  /** Reads the shared bake matrix off the first selected tile that has one. */
  #readBake(): boolean {
    const bake = bakeOf(this.#primitive);
    if (bake === undefined) return this.#bake !== undefined;
    if (this.#bake !== undefined && sameMatrix(bake, this.#bake)) return true;
    const inverse = invertAffine(bake);
    if (inverse === undefined) return false;
    this.#bake = Array.from(bake);
    this.#matrix = this.#gpu.mat4(gridFromModel(this.meta, inverse));
    return true;
  }
}

/** Slot ranges a companion's part can tell apart: two to an `ivec4`. */
export const COMPANION_SLOT_VECS = 4;

/**
 * The slot ranges of `owners`' tiles in an incremental primitive (`_tileSlots`), merged where
 * they touch, as `[start, end)` pairs packed two to a vec4 (`COMPANION_SLOT_VECS` of them;
 * unused pairs empty). Tiles past what fits are left out, and `overflow` says how many.
 */
export function companionSlots(
  primitive: VisibilityPrimitive | undefined,
  owners: ReadonlySet<unknown>,
): { vecs: [number, number, number, number][]; overflow: number; count: number } {
  const ranges: [number, number][] = [];
  for (const [tile, slot] of primitive?._tileSlots ?? []) {
    if (!owners.has(tile.tileset)) continue;
    ranges.push([slot.start, slot.start + slot.count]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const range of ranges) {
    const last = merged[merged.length - 1];
    if (last && last[1] >= range[0]) last[1] = Math.max(last[1], range[1]);
    else merged.push([range[0], range[1]]);
  }
  const fits = COMPANION_SLOT_VECS * 2;
  const vecs: [number, number, number, number][] = [];
  for (let k = 0; k < COMPANION_SLOT_VECS; k += 1) {
    const a = merged[k * 2];
    const b = merged[k * 2 + 1];
    vecs.push([a?.[0] ?? 0, a?.[1] ?? 0, b?.[0] ?? 0, b?.[1] ?? 0]);
  }
  return {
    vecs,
    overflow: Math.max(0, merged.length - fits),
    count: Math.min(merged.length, fits),
  };
}

/** Whether `splatIndex` is in one of `vecs`' ranges (what the GLSL `inSlots` tests). */
export function inCompanionSlots(
  vecs: readonly (readonly [number, number, number, number])[],
  splatIndex: number,
): boolean {
  return vecs.some(
    ([s0, e0, s1, e1]) =>
      (splatIndex >= s0 && splatIndex < e0) || (splatIndex >= s1 && splatIndex < e1),
  );
}

/** GLSL: whether `splatIndex` is in the ranges of the `ivec4` array `slots`. */
export function companionSlotsGlsl(fn: string, slots: string): string {
  return `
bool ${fn}(uint splatIndex) {
    int i = int(splatIndex);
    for (int k = 0; k < ${String(COMPANION_SLOT_VECS)}; k++) {
        ivec4 r = ${slots}[k];
        if ((i >= r.x && i < r.y) || (i >= r.z && i < r.w)) {
            return true;
        }
    }
    return false;
}
`;
}

let companionSerial = 0;

/**
 * The view cones of a companion -- an inferred layer whose tiles the scan's primitive draws in
 * its own sort (`SplatPrimitive.companions`, `inferredLayers.ts`) -- as a part of the scan
 * primitive's visibility chain: the splats in the layer's slots are faded by the layer's own
 * grid, every other splat is left as it is. The same rule as `SplatViewCones`; its uniforms and
 * function are named for it, so several layers' parts sit on one primitive, beside the scan's
 * own cones. The grid is the layer's frame; the bake is read off its own tiles.
 */
export class CompanionViewCones implements SplatVisibilityPart {
  readonly visibilityFunction: string;
  readonly visibilityOrder = 11;
  readonly meta: ViewConesMeta;
  readonly #rows: { data: Uint8Array; height: number };
  readonly #gpu: ViewConeGpu;
  readonly #owners: ReadonlySet<unknown>;
  readonly #suffix: string;
  #texture: OwnedTexture | undefined;
  #primitive: VisibilityPrimitive | undefined;
  #bake: number[] | undefined;
  #matrix: unknown;
  #identity: unknown;
  #slotsOf: unknown;
  #slots: unknown[] = [];
  #active = false;

  /** `owner`: the layer's tileset, whose tiles' slots are faded. */
  constructor(meta: ViewConesMeta, texels: Uint8Array, gpu: ViewConeGpu, owner: unknown) {
    this.meta = meta;
    this.#rows = textureRows(texels);
    this.#gpu = gpu;
    this.#owners = new Set([owner]);
    this.#suffix = `c${String(++companionSerial)}`;
    this.visibilityFunction = `splatCompanionCones_${this.#suffix}`;
  }

  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void {
    if (this.#texture === undefined || this.#texture.isDestroyed()) {
      this.#texture = this.#gpu.createBytes(
        context,
        VIEW_CONES_TEXTURE_WIDTH,
        this.#rows.height,
        this.#rows.data,
      );
    }
    const s = this.#suffix;
    shaderBuilder.addVertexLines(companionViewConesGlsl(s));
    const [nx, ny, nz] = this.meta.dims;
    const dims = this.#gpu.vec4(nx, ny, nz, VIEW_CONES_TEXTURE_WIDTH);
    uniformMap[`u_cones_${s}`] = () => this.#texture;
    uniformMap[`u_coneFromModel_${s}`] = () => this.#matrixValue();
    uniformMap[`u_coneDims_${s}`] = () => dims;
    uniformMap[`u_coneFade_${s}`] = () => this.meta.fadeDeg;
    uniformMap[`u_coneActive_${s}`] = () => (this.#read() ? 1 : 0);
    uniformMap[`u_coneSlots_${s}`] = () => {
      this.#read();
      return this.#slots;
    };
  }

  install(primitive: VisibilityPrimitive): boolean {
    if (this.#primitive === primitive && hasVisibilityPart(primitive, this)) return true;
    if (this.#primitive && this.#primitive !== primitive) this.uninstall();
    if (!addVisibilityPart(primitive, this)) return false;
    this.#primitive = primitive;
    this.#slotsOf = undefined;
    return true;
  }

  get installed(): boolean {
    return hasVisibilityPart(this.#primitive, this);
  }

  uninstall(): void {
    if (this.#primitive) removeVisibilityPart(this.#primitive, this);
    this.#primitive = undefined;
  }

  destroy(): void {
    this.uninstall();
    if (this.#texture && !this.#texture.isDestroyed()) this.#texture.destroy();
    this.#texture = undefined;
  }

  /** Brings the slots and the bake up to the primitive's; whether the part acts. */
  #read(): boolean {
    const primitive = this.#primitive;
    const slots = primitive?._tileSlots;
    if (slots !== this.#slotsOf) {
      this.#slotsOf = slots;
      const { vecs, count } = companionSlots(primitive, this.#owners);
      this.#slots = vecs.map(([a, b, c, d]) => this.#gpu.vec4(a, b, c, d));
      this.#active = count > 0;
      let bake: Mat4 | undefined;
      for (const tile of slots?.keys() ?? []) {
        if (!this.#owners.has(tile.tileset)) continue;
        const candidate = tile.content?._lastSplatTransform;
        if (candidate?.length === 16) {
          bake = candidate;
          break;
        }
      }
      if (bake !== undefined && (this.#bake === undefined || !sameMatrix(bake, this.#bake))) {
        const inverse = invertAffine(bake);
        if (inverse !== undefined) {
          this.#bake = Array.from(bake);
          this.#matrix = this.#gpu.mat4(gridFromModel(this.meta, inverse));
        }
      }
    }
    return this.#active && this.#matrix !== undefined;
  }

  #matrixValue(): unknown {
    if (!this.#read() || this.#matrix === undefined) {
      this.#identity ??= this.#gpu.mat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
      return this.#identity;
    }
    return this.#matrix;
  }
}

/** The GLSL of a companion's view cones (`CompanionViewCones`), its names suffixed `s`. */
export function companionViewConesGlsl(s: string): string {
  return `
uniform highp sampler2D u_cones_${s};
uniform mat4 u_coneFromModel_${s};
uniform vec4 u_coneDims_${s};
uniform float u_coneFade_${s};
uniform float u_coneActive_${s};
uniform ivec4 u_coneSlots_${s}[${String(COMPANION_SLOT_VECS)}];
${companionSlotsGlsl(`splatCompanionSlot_${s}`, `u_coneSlots_${s}`)}
float splatCompanionCones_${s}(uint splatIndex, vec3 position) {
    if (u_coneActive_${s} < 0.5 || !splatCompanionSlot_${s}(splatIndex)) {
        return 1.0;
    }
    vec3 g = (u_coneFromModel_${s} * vec4(position, 1.0)).xyz;
    ivec3 dims = ivec3(u_coneDims_${s}.xyz);
    ivec3 c = clamp(ivec3(floor(g)), ivec3(0), dims - 1);
    int width = int(u_coneDims_${s}.w);
    int linear = (c.z * dims.y + c.y) * dims.x + c.x;
    vec4 t = texelFetch(u_cones_${s}, ivec2(linear - (linear / width) * width, linear / width), 0);
    if (t.b > 254.5 / 255.0) {
        return 1.0;
    }
    vec2 e = t.rg * 2.0 - 1.0;
    vec3 v = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    float k = max(-v.z, 0.0);
    v.x += v.x >= 0.0 ? -k : k;
    v.y += v.y >= 0.0 ? -k : k;
    vec3 axis = normalize(v);
    float halfAngle = t.b * 255.0 * 180.0 / 254.0;
    vec3 eye = (u_coneFromModel_${s} * czm_inverseModelView * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec3 view = normalize(g - eye);
    float angle = degrees(acos(clamp(dot(view, axis), -1.0, 1.0)));
    return clamp((halfAngle + u_coneFade_${s} - angle) / u_coneFade_${s}, 0.0, 1.0);
}
`;
}

function bakeOf(primitive: VisibilityPrimitive | undefined): Mat4 | undefined {
  if (!primitive) return undefined;
  const tiles: Iterable<SplatTile> =
    primitive._selectedTileSet ?? primitive._tileSlots?.keys() ?? [];
  for (const tile of tiles) {
    // A companion's tile (`CompanionViewCones`) is baked from its own frame, not the scan's.
    if (tile.tileset !== undefined && primitive._tileset !== undefined) {
      if (tile.tileset !== primitive._tileset) continue;
    }
    const bake = tile.content?._lastSplatTransform;
    if (bake?.length === 16) return bake;
  }
  return undefined;
}

function sameMatrix(a: Mat4, b: Mat4): boolean {
  for (let i = 0; i < 16; i += 1) if (a[i] !== b[i]) return false;
  return true;
}

interface Barrel {
  Texture?: new (options: Record<string, unknown>) => OwnedTexture;
  Sampler?: { NEAREST?: unknown };
  PixelFormat?: { RGBA?: number };
  PixelDatatype?: { UNSIGNED_BYTE?: number };
  ShaderDestination?: { VERTEX?: number };
  Matrix4?: { fromArray(values: readonly number[]): unknown };
  Cartesian4?: new (x: number, y: number, z: number, w: number) => unknown;
}

/** By name, not `import * as`: a namespace read by key keeps every engine export in the bundle. */
const CESIUM = {
  Texture,
  Sampler,
  PixelFormat,
  PixelDatatype,
  ShaderDestination,
  Matrix4,
  Cartesian4,
} as unknown as Barrel;

/** The real `ViewConeGpu`, or `undefined` when this CesiumJS build lacks what it needs. */
export function cesiumViewConeGpu(): ViewConeGpu | undefined {
  const barrel = CESIUM;
  const { Texture, Matrix4, Cartesian4 } = barrel;
  const nearest = barrel.Sampler?.NEAREST;
  const rgba = barrel.PixelFormat?.RGBA;
  const bytes = barrel.PixelDatatype?.UNSIGNED_BYTE;
  const vertex = barrel.ShaderDestination?.VERTEX;
  if (
    typeof Texture !== "function" ||
    !Matrix4 ||
    typeof Cartesian4 !== "function" ||
    nearest === undefined ||
    rgba === undefined ||
    bytes === undefined ||
    vertex === undefined
  ) {
    return undefined;
  }
  return {
    createBytes: (context, width, height, data) =>
      new Texture({
        context,
        source: { width, height, arrayBufferView: data },
        pixelFormat: rgba,
        pixelDatatype: bytes,
        preMultiplyAlpha: false,
        skipColorSpaceConversion: true,
        flipY: false,
        sampler: nearest,
      }),
    mat4: (values) => Matrix4.fromArray(values),
    vec4: (x, y, z, w) => new Cartesian4(x, y, z, w),
    vertexDestination: vertex,
  };
}

/**
 * Fades what an inferred layer's views never saw wherever its tiles are drawn: by its own
 * primitive (`SplatViewCones`), or by the scan's that draws them in its sort while `host()`
 * returns it (`CompanionViewCones`, inferredLayers.ts). One grid, fetched once; `sync` puts
 * each hook where it belongs now, and the layer's first tile does too.
 */
export function attachLayerViewCones(
  tileset: Cesium3DTileset,
  host: () => VisibilityPrimitive | undefined,
  gpu: ViewConeGpu | undefined = cesiumViewConeGpu(),
): { sync(): void; dispose(): void } {
  const meta = viewConesMetaOf((tileset.root as { extras?: unknown } | undefined)?.extras);
  const url = (tileset as unknown as { resource?: { url?: string } }).resource?.url;
  const none = { sync: () => undefined, dispose: () => undefined };
  if (!meta || !url || !gpu || !viewConesEnabled()) return none;
  let own: SplatViewCones | undefined;
  let companion: CompanionViewCones | undefined;
  let disposed = false;
  let asked = false;
  const sync = (): void => {
    // Fetched once CesiumJS shows the layer: while another renderer draws the scan, it draws
    // the layer, and its cones, itself (scanView/scanLayers.ts).
    if (!asked && tileset.show) {
      asked = true;
      load();
    }
    const primitive: VisibilityPrimitive | undefined =
      splatTilesetOf(tileset).gaussianSplatPrimitive;
    if (own && primitive) own.install(primitive);
    const target = host();
    if (!companion) return;
    if (target) companion.install(target);
    else if (companion.installed) companion.uninstall();
  };
  const load = (): void => {
    loadViewCones(url, meta)
      .then((texels) => {
        if (disposed) return;
        own = new SplatViewCones(meta, texels, gpu);
        companion = new CompanionViewCones(meta, texels, gpu, tileset);
        sync();
        log.info("inferred layer's view cones attached", { cells: texels.length / 4 });
      })
      .catch((error: unknown) => {
        log.warn("view cones did not load; drawing every splat", {
          message: error instanceof Error ? error.message : String(error),
        });
      });
  };
  const unsubscribe = tileset.tileLoad.addEventListener(sync);
  return {
    sync,
    dispose: () => {
      disposed = true;
      unsubscribe();
      own?.destroy();
      companion?.destroy();
      own = undefined;
      companion = undefined;
    },
  };
}

/**
 * Fades what `tileset`'s capture never saw, when its root declares a view-cone grid: the
 * grid is fetched once and the hook installed on the splat primitive as soon as there is one.
 * Returns the disposer. A tileset without a grid, an unpatched engine or `?viewCones=off`
 * costs nothing.
 */
export function attachViewCones(
  tileset: Cesium3DTileset,
  gpu: ViewConeGpu | undefined = cesiumViewConeGpu(),
): () => void {
  const meta = viewConesMetaOf((tileset.root as { extras?: unknown } | undefined)?.extras);
  const url = (tileset as unknown as { resource?: { url?: string } }).resource?.url;
  if (!meta || !url || !gpu || !viewConesEnabled()) return () => undefined;
  let hook: SplatViewCones | undefined;
  let disposed = false;
  const install = (): void => {
    const primitive: VisibilityPrimitive | undefined =
      splatTilesetOf(tileset).gaussianSplatPrimitive;
    if (hook && primitive) hook.install(primitive);
  };
  const unsubscribe = tileset.tileLoad.addEventListener(install);
  loadViewCones(url, meta)
    .then((texels) => {
      if (disposed) return;
      hook = new SplatViewCones(meta, texels, gpu);
      install();
      log.info("view cones attached", {
        cells: texels.length / 4,
        directionalShare: meta.directionalShare ?? null,
      });
    })
    .catch((error: unknown) => {
      log.warn("view cones did not load; drawing every splat", {
        message: error instanceof Error ? error.message : String(error),
      });
    });
  return () => {
    disposed = true;
    unsubscribe();
    hook?.destroy();
    hook = undefined;
  };
}
