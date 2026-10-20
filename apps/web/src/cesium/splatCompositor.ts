/**
 * The per-cell view rules on the GPU (ADR 0017 "Per-cell view rules"): one splat primitive --
 * the site frame's own date's -- draws every date of the site (and their inferred layers) in
 * one texture and one sort (`companions.ts`), and this part of its visibility chain
 * (`splatVisibility.ts`) keeps, in each cell of the site lattice, only the splats of the date
 * the rule chose for it (`lib/compositor.ts` `ownerPlane`).
 *
 * Per splat: its source (`splatSources.ts`: which tileset its slot holds), the cell it sits in
 * (its position through `u_compFromModel`, the model frame to the tileset frame, as the
 * confidence view does: `splatCellView.ts` `cellFromModel`), and that cell's owner (an R8
 * texture of the owner plane, one byte a cell). Weight 0 unless the two agree; else a feather
 * at a seam: `mix(0.5, 1, smoothstep(0, w, d))`, `d` the distance to the nearest cell edge
 * across which the owner differs, `w` the feather width (`featherWidth`, a UX constant). A
 * splat whose slot holds no listed source, or that sits outside the plane, is not drawn: the
 * rule fails closed, never by blending two dates.
 *
 * Its colour part (Highlight invented) tints the inferred owners' splats as the inferred
 * highlight does (`lib/inferred.ts` `INFERRED_HIGHLIGHT`), by the sources flagged in a mask.
 *
 * The two parts share their helpers (declared once, whichever the build reaches first) and
 * their textures. Every value is a uniform: a new plane, a new mask or new slots cost uploads,
 * never a shader.
 */

import { Cartesian4, Matrix4, PixelDatatype, PixelFormat, Sampler, Texture } from "cesium";

import { featherWidth, NO_OWNER, type OwnerPlane } from "@/lib/compositor";
import { INFERRED_HIGHLIGHT } from "@/lib/inferred";

import {
  addColorPart,
  hasColorPart,
  removeColorPart,
  type ColorPrimitive,
  type SplatColorPart,
} from "./splatColor";
import { cellFromModel } from "./splatCellView";
import type { Mat4 } from "./splatFrames";
import type { SplatPrimitive, SplatShaderBuilder } from "./splatInternals";
import {
  NO_SOURCE,
  SOURCE_TEXTURE_WIDTH,
  SplatSources,
  sourceGlsl,
  type SourceGpu,
  type SourceTexture,
} from "./splatSources";
import {
  addVisibilityPart,
  hasVisibilityPart,
  removeVisibilityPart,
  type SplatVisibilityPart,
  type VisibilityPrimitive,
} from "./splatVisibility";

/** Cells a row of the owner texture holds. */
export const OWNER_TEXTURE_WIDTH = SOURCE_TEXTURE_WIDTH;

/** How the parts make their textures and uniform values. The real one wraps CesiumJS. */
export interface CompositorGpu extends SourceGpu {
  mat4(values: readonly number[]): unknown;
  vec4(x: number, y: number, z: number, w: number): unknown;
}

/** The primitive fields the parts read, beyond the two chains. */
export type CompositorPrimitive = VisibilityPrimitive &
  ColorPrimitive &
  Pick<SplatPrimitive, "_rootTransform" | "_tileSlots" | "isDestroyed">;

const ownerShift = String(Math.log2(OWNER_TEXTURE_WIDTH));

/**
 * Uniforms and helpers both parts use. `u_compParams`: (active, cell m, feather m, slots the
 * source texture covers); `u_compPlane`: (i0, j0, ni, nj).
 */
export const COMPOSITOR_HELPERS_GLSL = `
#ifndef HEXAPOD_SPLAT_COMPOSITOR
#define HEXAPOD_SPLAT_COMPOSITOR
uniform highp sampler2D u_compSources;
uniform highp sampler2D u_compOwners;
uniform mat4 u_compFromModel;
uniform vec4 u_compPlane;
uniform vec4 u_compParams;
${sourceGlsl("splatCompSource", "u_compSources", "u_compParams.w")}
int splatCompOwner(ivec2 c) {
    int ni = int(u_compPlane.z + 0.5);
    if (c.x < 0 || c.y < 0 || c.x >= ni || c.y >= int(u_compPlane.w + 0.5)) {
        return ${String(NO_OWNER)};
    }
    int linear = c.y * ni + c.x;
    return int(texelFetch(u_compOwners, ivec2(linear & ${String(OWNER_TEXTURE_WIDTH - 1)}, linear >> ${ownerShift}), 0).r * 255.0 + 0.5);
}
#endif
`;

/** The visibility part: the owner's splats only, feathered at seams. */
export const COMPOSITOR_VISIBILITY_GLSL = `
float splatCompositorVisibility(uint splatIndex, vec3 position) {
    if (u_compParams.x < 0.5) {
        return 1.0;
    }
    int source = splatCompSource(splatIndex);
    if (source == ${String(NO_SOURCE)}) {
        return 0.0;
    }
    float cell = u_compParams.y;
    vec2 g = (u_compFromModel * vec4(position, 1.0)).xy / cell;
    vec2 f = floor(g);
    ivec2 c = ivec2(f) - ivec2(u_compPlane.xy);
    int owner = splatCompOwner(c);
    if (owner != source) {
        return 0.0;
    }
    vec2 t = (g - f) * cell;
    float d = 1.0e9;
    if (splatCompOwner(c - ivec2(1, 0)) != owner) {
        d = min(d, t.x);
    }
    if (splatCompOwner(c + ivec2(1, 0)) != owner) {
        d = min(d, cell - t.x);
    }
    if (splatCompOwner(c - ivec2(0, 1)) != owner) {
        d = min(d, t.y);
    }
    if (splatCompOwner(c + ivec2(0, 1)) != owner) {
        d = min(d, cell - t.y);
    }
    return mix(0.5, 1.0, smoothstep(0.0, u_compParams.z, d));
}
`;

/**
 * The colour part: an inferred owner's splats tinted and hatched as the inferred highlight
 * does. `u_compPattern`: (band m, dark bands' brightness, opacity factor, mask of the tinted
 * sources, 0 for off).
 */
export const COMPOSITOR_COLOR_GLSL = `
uniform vec4 u_compTint;
uniform vec4 u_compPattern;

vec4 splatCompositorColor(uint splatIndex, vec3 position, vec4 color) {
    int mask = int(u_compPattern.w + 0.5);
    if (u_compParams.x < 0.5 || mask == 0) {
        return color;
    }
    int source = splatCompSource(splatIndex);
    if (source >= 16 || ((mask >> source) & 1) == 0) {
        return color;
    }
    vec3 rgb = mix(color.rgb, u_compTint.rgb, u_compTint.a);
    float band = fract(dot(position, vec3(0.57735027)) / u_compPattern.x);
    rgb *= band < 0.5 ? 1.0 : u_compPattern.y;
    return vec4(rgb, color.a * u_compPattern.z);
}
`;

/** The owner plane's bytes as the owner texture holds them: whole rows, `NO_OWNER` after. */
export function ownerTexels(plane: Pick<OwnerPlane, "owners">): {
  data: Uint8Array;
  height: number;
} {
  const height = Math.max(1, Math.ceil(plane.owners.length / OWNER_TEXTURE_WIDTH));
  const data = new Uint8Array(height * OWNER_TEXTURE_WIDTH).fill(NO_OWNER);
  data.set(plane.owners);
  return { data, height };
}

function same(a: Mat4 | undefined, b: readonly number[] | undefined): boolean {
  if (!a || !b) return false;
  for (let i = 0; i < 16; i += 1) if (a[i] !== b[i]) return false;
  return true;
}

/** Both parts of the rules on one primitive, and the textures they read. */
export class SplatCompositor implements SplatVisibilityPart {
  readonly visibilityFunction = "splatCompositorVisibility";
  /** After a hidden instance (0, one fetch), before the view cones (10, two and an acos). */
  readonly visibilityOrder = 1;
  readonly sources: SplatSources;
  readonly #gpu: CompositorGpu;
  readonly #rootOf: () => Mat4 | undefined;
  #primitive: CompositorPrimitive | undefined;
  #context: unknown;
  #plane: OwnerPlane | undefined;
  #owners: SourceTexture | undefined;
  #ownersStale = true;
  #frame: { root: number[]; model: number[]; matrix: unknown } | undefined;
  #identity: unknown;
  #params: { key: string; value: unknown } = { key: "", value: undefined };
  #planeValue: unknown;
  #pattern: { mask: number; value: unknown } = { mask: -1, value: undefined };
  readonly #tint: unknown;
  /** The sources whose splats Highlight invented tints (bit k: source k); 0 for none. */
  tinted = 0;
  #building = false;

  /** `rootOf`: the host's root tile's computed transform now (the tileset frame). */
  constructor(gpu: CompositorGpu, rootOf: () => Mat4 | undefined) {
    this.#gpu = gpu;
    this.#rootOf = rootOf;
    this.sources = new SplatSources(gpu);
    this.#tint = gpu.vec4(...INFERRED_HIGHLIGHT.tint);
  }

  /** The plane drawn now; undefined while it is being worked out (nothing is drawn). */
  get plane(): OwnerPlane | undefined {
    return this.#plane;
  }

  setPlane(plane: OwnerPlane | undefined): void {
    if (plane === this.#plane) return;
    this.#plane = plane;
    this.#ownersStale = true;
    this.#planeValue = plane
      ? this.#gpu.vec4(plane.i0, plane.j0, plane.ni, plane.nj)
      : this.#gpu.vec4(0, 0, 0, 0);
  }

  /** Called by the patched engine through either chain. */
  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void {
    this.#context = context;
    shaderBuilder.addVertexLines(COMPOSITOR_HELPERS_GLSL);
    uniformMap.u_compSources = () => this.#sourceTexture();
    uniformMap.u_compOwners = () => this.#ownerTexture();
    uniformMap.u_compFromModel = () => this.#frameMatrix();
    uniformMap.u_compPlane = () => (this.#planeValue ??= this.#gpu.vec4(0, 0, 0, 0));
    uniformMap.u_compParams = () => this.#paramsValue();
    if (this.#building) {
      shaderBuilder.addVertexLines(COMPOSITOR_COLOR_GLSL);
      uniformMap.u_compTint = () => this.#tint;
      uniformMap.u_compPattern = () => this.#patternValue();
    } else {
      shaderBuilder.addVertexLines(COMPOSITOR_VISIBILITY_GLSL);
    }
  }

  /** The colour part (Highlight invented), sharing this object's helpers and textures. */
  readonly colorHook: SplatColorPart = {
    colorFunction: "splatCompositorColor",
    /** After the inferred layers' own highlight (10): the same look, by another rule. */
    colorOrder: 11,
    addToShader: (shaderBuilder, uniformMap, context) => {
      this.#building = true;
      try {
        this.addToShader(shaderBuilder, uniformMap, context);
      } finally {
        this.#building = false;
      }
    },
  };

  /** Joins both chains of `primitive`; false when the engine has no visibility hook. */
  install(primitive: CompositorPrimitive): boolean {
    if (this.#primitive && this.#primitive !== primitive) this.uninstall();
    if (this.#primitive !== primitive) {
      this.#frame = undefined;
      this.sources.setSources([]);
    }
    this.#primitive = primitive;
    const hides = hasVisibilityPart(primitive, this) || addVisibilityPart(primitive, this);
    if (hides && !hasColorPart(primitive, this.colorHook)) addColorPart(primitive, this.colorHook);
    return hides;
  }

  get installed(): boolean {
    return hasVisibilityPart(this.#primitive, this);
  }

  /** The primitive the parts are on. */
  get primitive(): CompositorPrimitive | undefined {
    return this.#primitive;
  }

  uninstall(): void {
    const primitive = this.#primitive;
    this.#primitive = undefined;
    if (!primitive || primitive.isDestroyed?.() === true) return;
    removeVisibilityPart(primitive, this);
    removeColorPart(primitive, this.colorHook);
  }

  /** Brings the source bytes up to the primitive's slots; whether anything changed. */
  sync(): boolean {
    return this.sources.sync(this.#primitive);
  }

  /** Whether the shader acts now: installed, with its frame known. */
  get active(): boolean {
    return this.installed && this.#readFrame() !== undefined;
  }

  destroy(): void {
    this.uninstall();
    this.sources.destroy();
    if (this.#owners && !this.#owners.isDestroyed()) this.#owners.destroy();
    this.#owners = undefined;
  }

  #sourceTexture(): unknown {
    // Read at draw time: a batch the engine published this frame is already in the slots.
    this.sources.sync(this.#primitive);
    return this.sources.texture(this.#context);
  }

  #ownerTexture(): unknown {
    if (this.#context === undefined) return this.#owners;
    if (this.#ownersStale || !this.#owners || this.#owners.isDestroyed()) {
      this.#owners?.destroy();
      const { data, height } = ownerTexels(this.#plane ?? { owners: new Uint8Array(0) });
      this.#owners = this.#gpu.createR8(this.#context, OWNER_TEXTURE_WIDTH, height, data);
      this.#ownersStale = false;
    }
    return this.#owners;
  }

  #paramsValue(): unknown {
    // Whichever uniform the draw reads first, the slots covered are this frame's.
    this.sources.sync(this.#primitive);
    const plane = this.#plane;
    const active = this.#readFrame() !== undefined;
    const cell = plane?.cellM ?? 1;
    const key = `${String(active)}:${String(cell)}:${String(this.sources.covered)}`;
    if (key !== this.#params.key) {
      this.#params = {
        key,
        value: this.#gpu.vec4(active ? 1 : 0, cell, featherWidth(cell), this.sources.covered),
      };
    }
    return this.#params.value;
  }

  #patternValue(): unknown {
    if (this.tinted !== this.#pattern.mask) {
      this.#pattern = {
        mask: this.tinted,
        value: this.#gpu.vec4(
          INFERRED_HIGHLIGHT.stripeM,
          INFERRED_HIGHLIGHT.stripeDark,
          INFERRED_HIGHLIGHT.opacity,
          this.tinted,
        ),
      };
    }
    return this.#pattern.value;
  }

  /** The model frame to the tileset frame (`cellFromModel`), rebuilt when either moves. */
  #readFrame(): unknown {
    const root = this.#rootOf();
    const model = this.#primitive?._rootTransform;
    if (!root || !model) return undefined;
    if (this.#frame && same(root, this.#frame.root) && same(model, this.#frame.model))
      return this.#frame.matrix;
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
}

interface Barrel {
  Texture?: new (options: Record<string, unknown>) => SourceTexture;
  Sampler?: { NEAREST?: unknown };
  PixelFormat?: { RED?: number };
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

/** The real `CompositorGpu`, or `undefined` when this CesiumJS build lacks what it needs. */
export function cesiumCompositorGpu(): CompositorGpu | undefined {
  const { Texture, Matrix4, Cartesian4 } = CESIUM;
  const nearest = CESIUM.Sampler?.NEAREST;
  const red = CESIUM.PixelFormat?.RED;
  const bytes = CESIUM.PixelDatatype?.UNSIGNED_BYTE;
  if (
    typeof Texture !== "function" ||
    !Matrix4 ||
    typeof Cartesian4 !== "function" ||
    nearest === undefined ||
    red === undefined ||
    bytes === undefined
  ) {
    return undefined;
  }
  return {
    createR8: (context, width, height, data) =>
      new Texture({
        context,
        source: { width, height, arrayBufferView: data },
        pixelFormat: red,
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
