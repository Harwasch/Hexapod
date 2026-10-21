/**
 * Confidence and Data age on a splat scan: each splat tinted by the cell of the site lattice
 * it sits in (hexapod.cellgrid v1, `lib/cellGrid.ts`), in the CesiumJS splat primitive's
 * vertex shader, as one part of its colour chain (`splatColor.ts`).
 *
 * The look is `lib/cellLook.ts`'s, the one PlayCanvas and Spark draw too (`CELL_LOOK_GLSL`):
 *
 * - Confidence: `weak-v1` (design M4-6 C3). Weak cells (score below the grid's threshold, or
 *   unseen) red; the rest from amber to green by score; a cell with no evidence desaturated
 *   grey. Not colour alone: weak cells are hatched with dark diagonal bands and fair ones
 *   dotted, strong ones solid, a band pair or a dot every two cells of the level read
 *   (`CELL_PATTERN_CELLS`), so a reader who cannot tell red from green still finds the weak
 *   patches.
 * - Data age: the shown date's age, one colour per age bucket for every cell it observed
 *   (`lib/dataAge.ts`); light world-space stripes where a later date of the site has evidence,
 *   so "newer data here" can be seen without switching.
 *
 * The splat's own luminance shades the tint, so the scan's shapes still read.
 *
 * Which level a splat reads is picked per splat from its distance to the eye, in step with the
 * tiles' level of detail: the finest level whose cell covers at least `minPx` pixels
 * (`cellLevelFor`). Coarser levels hold the weakest of their children (C3's pyramid), so a far
 * view never hides a weak patch behind a strong average.
 *
 * The frame: a splat's position in the shader is in the draw command's model frame, the ENU
 * frame at the tileset's centre the engine baked every tile into (`primitive._rootTransform`,
 * `R`). The grid is in the tileset frame -- the root tile's frame, `root.computedTransform`
 * (`T`, which includes the site's placement). A model-frame point is `T⁻¹ · R · p` in the
 * grid's metres, whichever tile it came from: unlike un-baking by one tile's bake, this also
 * holds for tiles that carry their own origin (3D Tiles `tile.transform`, decision D1).
 *
 * Splats an inferred layer adds to the scan's sort (`primitive.companions`, inferredLayers.ts)
 * have no capture evidence: in either view they are grey, whatever the scan's cell says.
 *
 * Cost: nothing in Photo (the part is not in the chain, `cellViewSlot.ts`). In a view, one
 * texel fetch and a few dozen flops a splat; a mode switch or a new data-age texture changes
 * uniforms only, and the caller asks for a render.
 */

import { Cartesian4, Matrix4, PixelDatatype, PixelFormat, Sampler, Texture } from "cesium";

import {
  CELL_TEXTURE_WIDTH,
  cellTexelLayout,
  cellTextureRows,
  confidenceTexels,
  DEFAULT_UNSEEN_BIT,
  DEFAULT_WEAK_BELOW,
  MAX_CELL_LEVELS,
  type CellGrid,
} from "@/lib/cellGrid";

import {
  CELL_LOOK_GLSL,
  CELL_MIN_PX,
  CELL_MODE,
  NEWER_STRIPE_M,
  type CellViewMode,
} from "@/lib/cellLook";

import {
  addColorPart,
  colorChainOf,
  hasColorPart,
  removeColorPart,
  type ColorPrimitive,
  type SplatColorPart,
} from "./splatColor";
import { invertAffine, type Mat4 } from "./splatFrames";
import type { OwnedTexture } from "./splatGpuMotion";
import type { SplatPrimitive, SplatShaderBuilder } from "./splatInternals";
import { COMPANION_SLOT_VECS, companionSlots, companionSlotsGlsl } from "./splatSlots";

export type { CellViewMode };
export {
  CELL_MIN_PX,
  CELL_PATTERN_CELLS,
  cellTexelFor,
  cellViewColor,
  NEWER_STRIPE_M,
} from "@/lib/cellLook";
export { AGE_RAMP, CONFIDENCE_RAMP } from "@/lib/cellRamps";

/** No companions: one array, so its identity says nothing changed. */
const NO_FILLS: readonly unknown[] = [];

const MODE = CELL_MODE;

/** How the part makes its texture and uniform values. The real one wraps CesiumJS. */
export interface CellViewGpu {
  /** An `RG8` texture, read normalised, sampled nearest. */
  createRG(context: unknown, width: number, height: number, data: Uint8Array): OwnedTexture;
  mat4(values: readonly number[]): unknown;
  vec4(x: number, y: number, z: number, w: number): unknown;
}

/**
 * GLSL of the part: the shared look (`CELL_LOOK_GLSL`) and the per-splat level read. Uniforms:
 * the cell texture; the model-to-grid matrix; per level (i0, j0, ni, nj) and its first texel;
 * (mode, finest cell m, minimum px, levels); (weak threshold, unseen bit, stripe m, the model's
 * runtime scale); the companion slots drawn grey.
 */
export function cellViewGlsl(): string {
  return `
uniform highp sampler2D u_cellGrid;
uniform mat4 u_cellFromModel;
uniform ivec4 u_cellLevels[${String(MAX_CELL_LEVELS)}];
uniform ivec4 u_cellOffsets[${String(Math.ceil(MAX_CELL_LEVELS / 4))}];
uniform vec4 u_cellParams;
uniform vec4 u_cellRule;
uniform ivec4 u_cellFills[${String(COMPANION_SLOT_VECS)}];
${CELL_LOOK_GLSL}
${companionSlotsGlsl("splatCellFill", "u_cellFills")}
int splatCellOffset(int k) {
    ivec4 o = u_cellOffsets[k / 4];
    int c = k - (k / 4) * 4;
    return c == 0 ? o.x : c == 1 ? o.y : c == 2 ? o.z : o.w;
}

vec4 splatCellViewColor(uint splatIndex, vec3 position, vec4 color) {
    int mode = int(u_cellParams.x + 0.5);
    if (mode == 0) {
        return color;
    }
    if (splatCellFill(splatIndex)) {
        return hexapodCellGrey(color);
    }
    vec3 p = (u_cellFromModel * vec4(position, 1.0)).xyz;
    vec3 eye = (czm_inverseModelView * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    float dist = max(length(position - eye), 0.001);
    float pxPerRadian = 0.5 * czm_viewport.w * czm_projection[1][1];
    // The cell's size on the globe: grid metres times the model's runtime scale.
    float need = u_cellParams.z * dist / (pxPerRadian * u_cellParams.y * u_cellRule.w);
    int levels = int(u_cellParams.w + 0.5);
    int k = need <= 1.0 ? 0 : int(ceil(log2(need)));
    k = clamp(k, 0, levels - 1);
    float cell = u_cellParams.y * exp2(float(k));
    ivec2 t = hexapodCellTexel(u_cellGrid, p.xy, cell, vec4(u_cellLevels[k]), float(splatCellOffset(k)));
    return hexapodCellLook(mode, t, p.xy, cell, u_cellRule, color);
}
`;
}

/** Column-major `a · b`. */
function multiply(a: Mat4, b: Mat4): number[] {
  const out = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c += 1) {
    for (let r = 0; r < 4; r += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) sum += (a[k * 4 + r] ?? 0) * (b[c * 4 + k] ?? 0);
      out[c * 4 + r] = sum;
    }
  }
  return out;
}

/**
 * The matrix from the draw command's model frame to the grid's metres: `T⁻¹ · R`, with `R`
 * the primitive's ENU bake frame and `T` the root tile's computed transform. Undefined when
 * `T` cannot be inverted.
 */
export function cellFromModel(rootComputed: Mat4, modelFrame: Mat4): number[] | undefined {
  const inverse = invertAffine(rootComputed);
  return inverse ? multiply(inverse, modelFrame) : undefined;
}

function same(a: Mat4 | undefined, b: Mat4 | undefined): boolean {
  if (!a || !b) return a === b;
  for (let i = 0; i < 16; i += 1) if (a[i] !== b[i]) return false;
  return true;
}

/** The primitive fields the part reads, beyond the colour slot. */
export type CellViewPrimitive = ColorPrimitive &
  Pick<SplatPrimitive, "_rootTransform" | "_tileSlots" | "companions" | "isDestroyed">;

/** The cell-view part of one scan's primitive. */
export class SplatCellView implements SplatColorPart {
  readonly colorFunction = "splatCellViewColor";
  /** Below the inferred layer's Highlight (10), above the objects' highlight (0). */
  readonly colorOrder = 5;
  readonly grid: CellGrid;
  readonly #gpu: CellViewGpu;
  readonly #rootOf: () => Mat4 | undefined;
  readonly #levels: unknown[];
  readonly #offsets: unknown[];
  readonly #params: { cellM: number; levels: number };
  #rule: { scale: number; value: unknown };
  readonly #modeParams: Record<number, unknown>;
  readonly #rows: Partial<Record<CellViewMode, { data: Uint8Array; height: number }>> = {};
  readonly #textures = new Map<CellViewMode, OwnedTexture>();
  #context: unknown;
  #primitive: CellViewPrimitive | undefined;
  #frame: { root: number[]; model: number[]; matrix: unknown } | undefined;
  #identity: unknown;
  #fillsOf: unknown;
  #fills: unknown[];
  readonly #fillSet: (() => ReadonlySet<unknown>) | undefined;
  #fillsFrom: unknown;
  /** What is drawn; a mode without its texture yet draws the photo. */
  mode: CellViewMode = "confidence";

  /**
   * `rootOf`: the root tile's computed transform now (`T`). `fills`: the tilesets drawn grey,
   * as having no capture evidence -- by default every companion (an inferred layer); under a
   * view rule, whose companions are the site's other dates, its inferred layers alone.
   */
  constructor(
    grid: CellGrid,
    gpu: CellViewGpu,
    rootOf: () => Mat4 | undefined,
    fills?: () => ReadonlySet<unknown>,
  ) {
    this.#fillSet = fills;
    this.grid = grid;
    this.#gpu = gpu;
    this.#rootOf = rootOf;
    const levels = grid.meta.levels;
    const { offsets } = cellTexelLayout(grid.meta);
    this.#levels = Array.from({ length: MAX_CELL_LEVELS }, (_, k) => {
      const l = levels[k];
      return gpu.vec4(l?.i0 ?? 0, l?.j0 ?? 0, l?.ni ?? 0, l?.nj ?? 0);
    });
    this.#offsets = Array.from({ length: Math.ceil(MAX_CELL_LEVELS / 4) }, (_, v) =>
      gpu.vec4(
        offsets[v * 4] ?? 0,
        offsets[v * 4 + 1] ?? 0,
        offsets[v * 4 + 2] ?? 0,
        offsets[v * 4 + 3] ?? 0,
      ),
    );
    this.#params = { cellM: levels[0]?.cellM ?? 1, levels: levels.length };
    this.#modeParams = Object.fromEntries(
      Object.values(MODE).map((m) => [
        m,
        gpu.vec4(m, this.#params.cellM, CELL_MIN_PX, levels.length),
      ]),
    );
    this.#rule = { scale: Number.NaN, value: undefined };
    this.#rows.confidence = cellTextureRows(grid.meta, confidenceTexels(grid));
    this.#fills = companionSlots(undefined, new Set()).vecs.map(([a, b, c, d]) =>
      gpu.vec4(a, b, c, d),
    );
  }

  /** Data age's texture bytes (`lib/dataAge.ts` `dataAgeTexels`), replacing any before. */
  setDataAge(rg: Uint8Array): void {
    this.#textures.get("data-age")?.destroy();
    this.#textures.delete("data-age");
    this.#rows["data-age"] = cellTextureRows(this.grid.meta, rg);
  }

  /** Whether `mode` can be drawn now (its texture's bytes are here). */
  ready(mode: CellViewMode = this.mode): boolean {
    return this.#rows[mode] !== undefined;
  }

  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void {
    this.#context = context;
    shaderBuilder.addVertexLines(cellViewGlsl());
    uniformMap.u_cellGrid = () => this.#texture();
    uniformMap.u_cellFromModel = () => this.#frameMatrix();
    uniformMap.u_cellLevels = () => this.#levels;
    uniformMap.u_cellOffsets = () => this.#offsets;
    uniformMap.u_cellParams = () => this.#modeParams[this.#activeMode()];
    uniformMap.u_cellRule = () => this.#ruleValue();
    uniformMap.u_cellFills = () => this.#fillSlots();
  }

  install(primitive: CellViewPrimitive): boolean {
    if (this.#primitive === primitive && hasColorPart(primitive, this)) return true;
    if (this.#primitive && this.#primitive !== primitive) this.uninstall();
    // One cell view a primitive: two would declare the same uniforms (a date's own view and a
    // view rule's Data age on one primitive). The one there keeps it.
    if (colorChainOf(primitive)?.parts.some((p) => p.colorFunction === this.colorFunction))
      return false;
    if (!addColorPart(primitive, this)) return false;
    this.#primitive = primitive;
    this.#frame = undefined;
    this.#fillsOf = undefined;
    return true;
  }

  get installed(): boolean {
    return hasColorPart(this.#primitive, this);
  }

  uninstall(): void {
    if (this.#primitive) removeColorPart(this.#primitive, this);
    this.#primitive = undefined;
  }

  destroy(): void {
    this.uninstall();
    for (const texture of this.#textures.values()) if (!texture.isDestroyed()) texture.destroy();
    this.#textures.clear();
  }

  /** The shader's mode: off until the frame is known and the mode's bytes are here. */
  #activeMode(): number {
    if (!this.ready() || this.#readFrame() === undefined) return MODE.off;
    return MODE[this.mode];
  }

  #texture(): unknown {
    const mode = this.ready() ? this.mode : "confidence";
    let texture = this.#textures.get(mode);
    if (texture === undefined || texture.isDestroyed()) {
      const rows = this.#rows[mode] ?? this.#rows.confidence;
      if (!rows || this.#context === undefined) return undefined;
      texture = this.#gpu.createRG(this.#context, CELL_TEXTURE_WIDTH, rows.height, rows.data);
      this.#textures.set(mode, texture);
    }
    return texture;
  }

  /**
   * (weak threshold, unseen bit, stripe m, scale): the scale is how long a grid metre is on the
   * globe -- the root's computed transform's first column, `renderConfig.scale` for a model
   * drawn at one (tilesetScale.ts), 1 for a rigid placement -- so a splat picks its level by
   * the size its cell is drawn at, while it still indexes the grid in the grid's own metres.
   */
  #ruleValue(): unknown {
    const root = this.#rootOf();
    const length = root ? Math.hypot(root[0] ?? 0, root[1] ?? 0, root[2] ?? 0) : 1;
    const scale = Number.isFinite(length) && length > 0 ? length : 1;
    if (scale !== this.#rule.scale) {
      const meta = this.grid.meta;
      this.#rule = {
        scale,
        value: this.#gpu.vec4(
          meta.weakBelow ?? DEFAULT_WEAK_BELOW,
          meta.reasonBits?.unseen ?? DEFAULT_UNSEEN_BIT,
          NEWER_STRIPE_M,
          scale,
        ),
      };
    }
    return this.#rule.value;
  }

  /** `T⁻¹ · R`, rebuilt only when either changes. */
  #readFrame(): unknown {
    const root = this.#rootOf();
    const model = this.#primitive?._rootTransform;
    if (!root || !model) return undefined;
    if (this.#frame && same(this.#frame.root, root) && same(this.#frame.model, model)) {
      return this.#frame.matrix;
    }
    const values = cellFromModel(root, model);
    if (!values) return undefined;
    this.#frame = {
      root: Array.from({ length: 16 }, (_, i) => root[i] ?? 0),
      model: Array.from({ length: 16 }, (_, i) => model[i] ?? 0),
      matrix: this.#gpu.mat4(values),
    };
    return this.#frame.matrix;
  }

  #frameMatrix(): unknown {
    const matrix = this.#readFrame();
    if (matrix !== undefined) return matrix;
    this.#identity ??= this.#gpu.mat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    return this.#identity;
  }

  /** The slots of the layers the scan's primitive draws in its sort: grey, not the scan's. */
  #fillSlots(): unknown[] {
    const primitive = this.#primitive;
    const slots = primitive?._tileSlots;
    const fills = this.#fillSet?.() ?? primitive?.companions ?? NO_FILLS;
    const key = slots && [...fills].length > 0 ? slots : undefined;
    if (key !== this.#fillsOf || fills !== this.#fillsFrom) {
      this.#fillsOf = key;
      this.#fillsFrom = fills;
      const owners = new Set<unknown>(key ? fills : []);
      const slots = companionSlots(primitive, owners);
      // More layer ranges than the uniform holds: the last range is stretched over all those
      // past it. Fails closed -- an inferred splat is never given a measured colour; scan
      // splats between them go grey, which claims no evidence rather than false evidence.
      const last = slots.vecs[slots.vecs.length - 1];
      if (slots.overflow > 0 && last) {
        for (const [tile, slot] of primitive?._tileSlots ?? []) {
          if (owners.has(tile.tileset)) last[3] = Math.max(last[3], slot.start + slot.count);
        }
      }
      this.#fills = slots.vecs.map(([a, b, c, d]) => this.#gpu.vec4(a, b, c, d));
    }
    return this.#fills;
  }
}

interface Barrel {
  Texture?: new (options: Record<string, unknown>) => OwnedTexture;
  Sampler?: { NEAREST?: unknown };
  PixelFormat?: { RG?: number };
  PixelDatatype?: { UNSIGNED_BYTE?: number };
  Matrix4?: { fromArray(values: readonly number[]): unknown };
  Cartesian4?: new (x: number, y: number, z: number, w: number) => unknown;
}

/** By name, not `import * as`: a namespace read by key keeps every engine export in the bundle. */
const CESIUM = {
  Texture,
  Sampler,
  PixelFormat,
  PixelDatatype,
  Matrix4,
  Cartesian4,
} as unknown as Barrel;

/** The real `CellViewGpu`, or `undefined` when this CesiumJS build lacks what it needs. */
export function cesiumCellViewGpu(): CellViewGpu | undefined {
  const { Texture, Matrix4, Cartesian4 } = CESIUM;
  const nearest = CESIUM.Sampler?.NEAREST;
  const rg = CESIUM.PixelFormat?.RG;
  const bytes = CESIUM.PixelDatatype?.UNSIGNED_BYTE;
  if (
    typeof Texture !== "function" ||
    !Matrix4 ||
    typeof Cartesian4 !== "function" ||
    nearest === undefined ||
    rg === undefined ||
    bytes === undefined
  ) {
    return undefined;
  }
  return {
    createRG: (context, width, height, data) =>
      new Texture({
        context,
        source: { width, height, arrayBufferView: data },
        pixelFormat: rg,
        pixelDatatype: bytes,
        preMultiplyAlpha: false,
        skipColorSpaceConversion: true,
        flipY: false,
        sampler: nearest,
      }),
    mat4: (values) => Matrix4.fromArray(values),
    vec4: (x, y, z, w) => new Cartesian4(x, y, z, w),
  };
}
