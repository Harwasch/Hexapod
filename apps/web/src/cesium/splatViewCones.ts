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

function bakeOf(primitive: VisibilityPrimitive | undefined): Mat4 | undefined {
  if (!primitive) return undefined;
  const tiles: Iterable<SplatTile> =
    primitive._selectedTileSet ?? primitive._tileSlots?.keys() ?? [];
  for (const tile of tiles) {
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
