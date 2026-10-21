/**
 * How Confidence and Data age colour one splat by its cell (hexapod.cellgrid v1, `cellGrid.ts`),
 * as one source for every renderer: CesiumJS's splat primitive (`cesium/splatCellView.ts`),
 * PlayCanvas's work-buffer modifier and Spark's world modifier (`cesium/scanView/cellModifier.ts`).
 * Each takes `CELL_LOOK_GLSL` as it is, so a weak cell is the same red, hatched the same way,
 * under all three; `cellViewColor` is its TypeScript twin, which the tests and the legend
 * reason with (e2e/confidenceViewScene.spec.ts runs both on the same inputs in a real GPU).
 *
 * - Confidence (`weak-v1`, design M4-6 C3): weak cells (under the grid's own `weakBelow`, or
 *   unseen) red and hatched with dark diagonal bands; the rest from amber to green by score, the
 *   fair ones dotted; a cell with no evidence (score 0, not unseen) the splat desaturated grey.
 * - Data age: one colour per age bucket for every cell the shown date observed (`dataAge.ts`),
 *   light world-space stripes where a later date of the site has evidence.
 *
 * The splat's luminance shades the tint, so the scan's shapes still read. The colours are the
 * legend's (`cellRamps.ts`), written into the GLSL from the same tables.
 *
 * What differs between renderers is only which level of the grid's pyramid a splat reads.
 * CesiumJS picks it per splat from the splat's distance to the eye (`cellTexelFor`). PlayCanvas
 * and Spark change a splat's colour once, as they copy or generate a tile, with no eye at hand:
 * they pick one level per tile with the same rule at the tile's centre (`cellLevelIndex`), and
 * copy the tile again only when that level steps. Coarser levels hold the weakest of their
 * children (C3), so a coarser read never hides a weak patch; it only draws bigger cells.
 */

import {
  CELL_TEXTURE_WIDTH,
  cellLevelFor,
  cellTexelLayout,
  DEFAULT_UNSEEN_BIT,
  DEFAULT_WEAK_BELOW,
  type CellGridMeta,
} from "./cellGrid";
import { AGE_RAMP, CONFIDENCE_RAMP, type Rgb } from "./cellRamps";

export type CellViewMode = "confidence" | "data-age";

/** The shaders' mode numbers. */
export const CELL_MODE: Record<CellViewMode | "off", number> = {
  off: 0,
  confidence: 1,
  "data-age": 2,
};

/** Smallest a cell may look on screen before a coarser level is read (pixels). */
export const CELL_MIN_PX = 6;
/** Data age's "newer data here" stripes: one light and one plain band per this many metres. */
export const NEWER_STRIPE_M = 0.6;
/** Confidence's hatching and dots repeat every this many cells of the level a splat reads. */
export const CELL_PATTERN_CELLS = 2;
/** How dark a hatch band or a dot is: the tint times this. */
export const CELL_PATTERN_DARK = 0.55;

/** A number as a GLSL float literal. */
function glslFloat(v: number): string {
  return Number.isInteger(v) ? `${String(v)}.0` : String(v);
}

function glslRamp(ramp: readonly Rgb[]): string {
  return ramp
    .map(([r, g, b]) => `vec3(${glslFloat(r)}, ${glslFloat(g)}, ${glslFloat(b)})`)
    .join(",\n    ");
}

/**
 * The look in GLSL ES 3.00, for every renderer: the two ramps (Confidence's six, then Data
 * age's six) and three functions.
 *
 * - `hexapodCellTexel(grid, p, cell, level, first)` (integers, as CesiumJS's uniforms are): the
 *   (R, G) bytes of the cell holding
 *   grid-frame point `p` (m) on one level -- its cells `cell` m wide, `level` its (i0, j0, ni,
 *   nj), `first` the texel its plane starts at in the cell texture -- or (0, 0) outside it.
 * - `hexapodCellLook(mode, t, p, cell, rule, color)`: what is drawn for a straight-alpha
 *   `color` whose cell reads `t`; `rule` is (weak threshold, unseen bit, stripe m, unused).
 * - `hexapodCellGrey(color)`: no evidence -- the splat desaturated.
 */
export const CELL_LOOK_GLSL = `
const vec3 HEXAPOD_CELL_RAMP[12] = vec3[12](
    ${glslRamp(CONFIDENCE_RAMP)},
    ${glslRamp(AGE_RAMP)});

ivec2 hexapodCellTexel(highp sampler2D grid, vec2 p, float cell, ivec4 level, int first) {
    int ci = int(floor(p.x / cell)) - level.x;
    int cj = int(floor(p.y / cell)) - level.y;
    if (ci < 0 || cj < 0 || ci >= level.z || cj >= level.w) {
        return ivec2(0);
    }
    int linear = first + cj * level.z + ci;
    int row = linear / ${String(CELL_TEXTURE_WIDTH)};
    vec2 texel = texelFetch(grid, ivec2(linear - row * ${String(CELL_TEXTURE_WIDTH)}, row), 0).rg;
    return ivec2(texel * 255.0 + 0.5);
}

vec4 hexapodCellGrey(vec4 color) {
    float lum = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    return vec4(vec3(lum * 0.85), color.a);
}

vec4 hexapodCellLook(int mode, ivec2 t, vec2 p, float cell, vec4 rule, vec4 color) {
    float lum = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    float shade = 0.45 + 0.55 * lum;
    vec3 rgb;
    if (mode == 1) {
        bool unseen = (t.y & int(rule.y + 0.5)) != 0;
        if (t.x == 0 && !unseen) {
            rgb = vec3(lum * 0.85);
        } else {
            float period = ${glslFloat(CELL_PATTERN_CELLS)} * cell;
            if (unseen || float(t.x) < rule.x) {
                rgb = HEXAPOD_CELL_RAMP[1] * shade;
                if (fract((p.x - p.y) / period) >= 0.5) {
                    rgb *= ${glslFloat(CELL_PATTERN_DARK)};
                }
            } else {
                int bin = 2 + clamp(int(4.0 * (float(t.x) - rule.x) / (256.0 - rule.x)), 0, 3);
                rgb = HEXAPOD_CELL_RAMP[bin] * shade;
                if (bin < 4 && fract(p.x / period) >= 0.6 && fract(p.y / period) >= 0.6) {
                    rgb *= ${glslFloat(CELL_PATTERN_DARK)};
                }
            }
        }
    } else {
        rgb = t.x == 0 ? vec3(lum * 0.85) : HEXAPOD_CELL_RAMP[6 + clamp(t.x - 1, 0, 5)] * shade;
        if (t.y > 0 && fract((p.x + p.y) / rule.z) < 0.5) {
            rgb = mix(rgb, vec3(1.0), 0.65);
        }
    }
    return vec4(rgb, color.a);
}
`;

/**
 * What `hexapodCellLook` does to one straight-alpha colour, for tests: `texel` the cell's
 * (R, G) bytes (or null outside the grid), `p` the splat in the grid's metres, `cell` the size
 * of the level's cells it reads (m); `fill` a splat with no capture evidence (an inferred
 * layer's), drawn grey whatever its cell says (`hexapodCellGrey`).
 */
export function cellViewColor(
  mode: CellViewMode,
  texel: readonly [number, number] | null,
  color: readonly [number, number, number, number],
  options: {
    weakBelow?: number;
    unseenBit?: number;
    p?: readonly [number, number];
    cell?: number;
    fill?: boolean;
  } = {},
): [number, number, number, number] {
  const lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2];
  const grey: [number, number, number, number] = [lum * 0.85, lum * 0.85, lum * 0.85, color[3]];
  if (options.fill) return grey;
  const [r, g] = texel ?? [0, 0];
  const shade = 0.45 + 0.55 * lum;
  const tint = (c: Rgb | undefined): [number, number, number, number] => [
    (c?.[0] ?? 0) * shade,
    (c?.[1] ?? 0) * shade,
    (c?.[2] ?? 0) * shade,
    color[3],
  ];
  if (mode === "confidence") {
    const weakBelow = options.weakBelow ?? DEFAULT_WEAK_BELOW;
    const unseen = (g & (options.unseenBit ?? DEFAULT_UNSEEN_BIT)) !== 0;
    if (r === 0 && !unseen) return grey;
    const period = CELL_PATTERN_CELLS * (options.cell ?? 1);
    const fract = (v: number) => v / period - Math.floor(v / period);
    const [x, y] = options.p ?? [0, 0];
    const dark = (c: [number, number, number, number]): [number, number, number, number] => [
      c[0] * CELL_PATTERN_DARK,
      c[1] * CELL_PATTERN_DARK,
      c[2] * CELL_PATTERN_DARK,
      c[3],
    ];
    if (unseen || r < weakBelow) {
      const weak = tint(CONFIDENCE_RAMP[1]);
      return fract(x - y) >= 0.5 ? dark(weak) : weak;
    }
    const bin = 2 + Math.min(3, Math.max(0, Math.floor((4 * (r - weakBelow)) / (256 - weakBelow))));
    const out = tint(CONFIDENCE_RAMP[bin]);
    return bin < 4 && fract(x) >= 0.6 && fract(y) >= 0.6 ? dark(out) : out;
  }
  const out = r === 0 ? grey : tint(AGE_RAMP[Math.min(5, Math.max(0, r - 1))]);
  const [x, y] = options.p ?? [0, 0];
  const band = (x + y) / NEWER_STRIPE_M - Math.floor((x + y) / NEWER_STRIPE_M);
  if (g > 0 && band < 0.5) {
    return [
      out[0] + (1 - out[0]) * 0.65,
      out[1] + (1 - out[1]) * 0.65,
      out[2] + (1 - out[2]) * 0.65,
      out[3],
    ];
  }
  return out;
}

/**
 * The level index (finest 0) whose cells cover at least `minPx` pixels at `distanceM`, seen at
 * `pxPerRadian`, within the grid's levels: CesiumJS's rule per splat, and the dedicated
 * renderers' per tile. `scale`: how long a grid metre is drawn, when `distanceM` is in metres on
 * the globe and the model is drawn at a runtime scale; 1 when `distanceM` is in the grid's own.
 */
export function cellLevelIndex(
  meta: Pick<CellGridMeta, "levels">,
  distanceM: number,
  pxPerRadian: number,
  minPx: number = CELL_MIN_PX,
  scale = 1,
): number {
  const levels = meta.levels;
  const cellM = (levels[0]?.cellM ?? 1) * scale;
  return Math.min(
    levels.length - 1,
    Math.max(0, cellLevelFor(distanceM, pxPerRadian, cellM, minPx)),
  );
}

/** The texel of the cell of level index `k` holding grid-frame `(x, y)`, or null outside it. */
export function cellTexelOn(
  meta: Pick<CellGridMeta, "levels">,
  x: number,
  y: number,
  k: number,
): number | null {
  const l = meta.levels[k];
  if (!l) return null;
  const cell = (meta.levels[0]?.cellM ?? 1) * 2 ** k;
  const ci = Math.floor(x / cell) - l.i0;
  const cj = Math.floor(y / cell) - l.j0;
  if (ci < 0 || cj < 0 || ci >= l.ni || cj >= l.nj) return null;
  return (cellTexelLayout(meta).offsets[k] ?? 0) + cj * l.ni + ci;
}

/**
 * The texel CesiumJS's shader reads for a splat at grid-frame `(x, y)` at `distanceM`, seen at
 * `pxPerRadian`: its level index (finest 0) and its cell's index in the cell texture, or null
 * outside that level's plane. The reference for the per-splat lookup in `cellViewGlsl`.
 */
export function cellTexelFor(
  meta: Pick<CellGridMeta, "levels">,
  x: number,
  y: number,
  distanceM: number,
  pxPerRadian: number,
  minPx: number = CELL_MIN_PX,
  scale = 1,
): { level: number; texel: number | null } {
  const k = cellLevelIndex(meta, distanceM, pxPerRadian, minPx, scale);
  return { level: k, texel: cellTexelOn(meta, x, y, k) };
}

/**
 * Device pixels one radian spans at the middle of a view `heightPx` device pixels tall with
 * vertical field of view `fovy`: CesiumJS's `0.5 * czm_viewport.w * czm_projection[1][1]`.
 */
export function pxPerRadianOf(heightPx: number, fovy: number): number {
  return (0.5 * heightPx) / Math.tan(fovy / 2);
}

/**
 * What a dedicated renderer is handed to draw a scan's cell view (`cesium/cellView.ts` makes
 * it, `cesium/scanView/scanCellView.ts` carries it): the mode, its texture (`cellTextureRows`:
 * RG8, `CELL_TEXTURE_WIDTH` cells a row), and the grid's levels and rule.
 */
export interface CellLookView {
  readonly mode: CellViewMode;
  readonly rows: { readonly data: Uint8Array; readonly height: number };
  readonly meta: Pick<CellGridMeta, "levels" | "weakBelow" | "reasonBits">;
}

/** `(weak threshold, unseen bit, stripe m, 0)`: the grid's own rule, as the shader reads it. */
export function cellRuleOf(meta: CellLookView["meta"]): Float32Array<ArrayBuffer> {
  return Float32Array.of(
    meta.weakBelow ?? DEFAULT_WEAK_BELOW,
    meta.reasonBits?.unseen ?? DEFAULT_UNSEEN_BIT,
    NEWER_STRIPE_M,
    0,
  );
}

/** One tile's level as its two uniforms (`cellTileUniforms`). */
export interface CellTileUniforms {
  /** The plane's (i0, j0, ni, nj). */
  level: Float32Array<ArrayBuffer>;
  /** (mode, cell size m, the plane's first texel, level index); mode 0 draws the photo. */
  tile: Float32Array<ArrayBuffer>;
}

/** Level index `k` of `view` as a tile's uniforms, into `out`; `view` null: the photo. */
export function cellTileUniforms(
  view: CellLookView | null,
  k: number,
  out: CellTileUniforms = { level: new Float32Array(4), tile: new Float32Array(4) },
): CellTileUniforms {
  const l = view?.meta.levels[k];
  if (!view || !l) {
    out.level.fill(0);
    out.tile.fill(0);
    return out;
  }
  out.level.set([l.i0, l.j0, l.ni, l.nj]);
  out.tile.set([
    CELL_MODE[view.mode],
    (view.meta.levels[0]?.cellM ?? 1) * 2 ** k,
    cellTexelLayout(view.meta).offsets[k] ?? 0,
    k,
  ]);
  return out;
}

/**
 * Of a tile's splats -- `positions` xyz, plus `offset`, in the grid's frame; every `step`-th --
 * how many there are and how many would read `tileLevel` by CesiumJS's per-splat rule: how far
 * a per-tile read agrees with CesiumJS's, for the developer readouts and the tests.
 */
export function cellLevelAgreement(
  meta: Pick<CellGridMeta, "levels">,
  positions: ArrayLike<number>,
  offset: readonly [number, number, number],
  eye: readonly [number, number, number],
  pxPerRadian: number,
  tileLevel: number,
  step = 1,
): { splats: number; agree: number } {
  let splats = 0;
  let agree = 0;
  const count = Math.floor(positions.length / 3);
  for (let i = 0; i < count; i += Math.max(1, step)) {
    const d = Math.hypot(
      (positions[i * 3] ?? 0) + offset[0] - eye[0],
      (positions[i * 3 + 1] ?? 0) + offset[1] - eye[1],
      (positions[i * 3 + 2] ?? 0) + offset[2] - eye[2],
    );
    splats += 1;
    if (cellLevelIndex(meta, d, pxPerRadian) === tileLevel) agree += 1;
  }
  return { splats, agree };
}
