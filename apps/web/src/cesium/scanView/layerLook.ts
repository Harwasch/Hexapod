/**
 * How a dedicated splat renderer draws the tiles of an inferred layer (scanLayers.ts), as
 * cesium/inferredLayers.ts draws them under CesiumJS: Highlight's purple and hatching
 * (lib/inferred.ts `INFERRED_HIGHLIGHT`, the rule of `INFERRED_COLOR_GLSL`), and the fade of the
 * layer's own view cones (lib/viewCones.ts, the rule of `VIEW_CONES_GLSL`), so a fill made for
 * the outside of a scan is not drawn from its inside under any renderer.
 *
 * Both are per splat, and the cones depend on where the eye is: the back-ends apply them where
 * they already change a splat -- PlayCanvas's work-buffer modifier, Spark's world modifier --
 * and run them again as the eye moves (`eyeMoved`: a layer is a few tiles of a few thousand
 * splats, and its fade spans 20 degrees). The splat's position reaches the shader in the scan's
 * frame (what both renderers draw in); `fromScan` takes it into the layer's own, where its
 * cones' grid and its hatching are.
 */

import { INFERRED_HIGHLIGHT, inferredHighlightColor } from "@/lib/inferred";
import {
  textureRows,
  visibility,
  VIEW_CONES_TEXTURE_WIDTH,
  type ViewConesMeta,
} from "@/lib/viewCones";

import type { Mat4 } from "../splatFrames";

/** A layer's view cones, as a back-end uploads them. */
export interface LayerCones {
  readonly meta: ViewConesMeta;
  /** The texels padded to whole rows of `VIEW_CONES_TEXTURE_WIDTH` (RGBA8). */
  readonly data: Uint8Array;
  readonly height: number;
}

/** How one layer's tiles are drawn now. */
export interface LayerLook {
  /** Highlight: pulled toward purple, hatched, a little see-through. */
  readonly highlight: boolean;
  /** The layer's view cones, or null when it has none (or `?viewCones=off`). */
  readonly cones: LayerCones | null;
  /** The layer's own frame from the scan's (column-major 4x4): `P⁻¹` for its placement `P`. */
  readonly fromScan: Mat4;
}

/** A layer's cones made ready for upload. */
export function layerCones(meta: ViewConesMeta, texels: Uint8Array): LayerCones {
  const { data, height } = textureRows(texels);
  return { meta, data, height };
}

/** The uniform values of a look, but the eye's (`coneEye`): one `vec4` or `mat4` each. */
export interface LayerUniforms {
  /** `fromScan`, column-major. */
  frame: Float32Array<ArrayBuffer>;
  /** Highlight's tint: rgb, and how far toward it. */
  tint: Float32Array<ArrayBuffer>;
  /** x: band width (m); y: the darker bands' brightness; z: opacity factor; w: highlight on. */
  pattern: Float32Array<ArrayBuffer>;
  /** The cones' grid: its origin (layer frame, m), and one over its cell size. */
  grid: Float32Array<ArrayBuffer>;
  /** Cells along x, y, z, and texels a row. */
  dims: Float32Array<ArrayBuffer>;
}

export function layerUniforms(look: LayerLook): LayerUniforms {
  const { stripeM, stripeDark, opacity, tint } = INFERRED_HIGHLIGHT;
  const meta = look.cones?.meta;
  return {
    frame: Float32Array.from(look.fromScan),
    tint: Float32Array.from(tint),
    pattern: Float32Array.of(stripeM, stripeDark, opacity, look.highlight ? 1 : 0),
    grid: meta
      ? Float32Array.of(meta.origin[0], meta.origin[1], meta.origin[2], 1 / meta.cell)
      : new Float32Array(4),
    dims: meta
      ? Float32Array.of(meta.dims[0], meta.dims[1], meta.dims[2], VIEW_CONES_TEXTURE_WIDTH)
      : Float32Array.of(1, 1, 1, VIEW_CONES_TEXTURE_WIDTH),
  };
}

/**
 * The eye uniform: where the eye is in the layer's frame, and the fade (degrees) in `w`, 0
 * when the layer has no cones (the shader then leaves every splat's opacity as it is).
 */
export function coneEye(
  look: LayerLook,
  eye: readonly number[],
  out: Float32Array<ArrayBuffer> = new Float32Array(4),
): Float32Array<ArrayBuffer> {
  const m = look.fromScan;
  const [x = 0, y = 0, z = 0] = eye;
  for (let r = 0; r < 3; r += 1) {
    out[r] = (m[r] ?? 0) * x + (m[4 + r] ?? 0) * y + (m[8 + r] ?? 0) * z + (m[12 + r] ?? 0);
  }
  out[3] = look.cones ? look.cones.meta.fadeDeg : 0;
  return out;
}

/**
 * Whether the eye moved far enough from where a layer's fade was last worked out to work it out
 * again: by more than a hundredth of its distance from the layer (half a degree of the 20 its
 * fade spans), or `minM`, whichever is more.
 */
export function eyeMoved(
  last: readonly number[] | null,
  eye: readonly number[],
  centre: readonly number[],
  minM = 0.02,
): boolean {
  if (last === null) return true;
  const moved = Math.hypot(
    (eye[0] ?? 0) - (last[0] ?? 0),
    (eye[1] ?? 0) - (last[1] ?? 0),
    (eye[2] ?? 0) - (last[2] ?? 0),
  );
  const away = Math.hypot(
    (eye[0] ?? 0) - (centre[0] ?? 0),
    (eye[1] ?? 0) - (centre[1] ?? 0),
    (eye[2] ?? 0) - (centre[2] ?? 0),
  );
  return moved > Math.max(minM, away * 0.01);
}

/**
 * The look as GLSL, for both WebGL back-ends: `hexapodLayerColor` takes a splat's position in
 * the scan's frame and its colour (straight alpha) and returns what is drawn. Its uniforms are
 * parameters, so each back-end passes its own by name (PlayCanvas's uniforms, Spark's dyno
 * inputs). The cones' rule is `VIEW_CONES_GLSL`'s; the highlight's `INFERRED_COLOR_GLSL`'s.
 *
 * In CesiumJS's order: the cones' weight on the opacity, then -- while one of the scan's
 * objects is highlighted -- the dim every splat outside it gets (`dimmed`: colour and opacity
 * factors, scanInstances.ts; a layer belongs to no object), then Highlight's purple.
 * `hexapodLayerDim` is that dim from the scan's instance uniforms.
 */
export const LAYER_LOOK_GLSL = `
vec3 hexapodLayerConeAxis(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 v = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-v.z, 0.0);
    v.x += v.x >= 0.0 ? -t : t;
    v.y += v.y >= 0.0 ? -t : t;
    return normalize(v);
}

float hexapodLayerCone(vec3 p, highp sampler2D cones, vec4 grid, vec4 dims, vec4 eye) {
    if (eye.w <= 0.0) {
        return 1.0;
    }
    vec3 g = (p - grid.xyz) * grid.w;
    ivec3 n = ivec3(dims.xyz);
    ivec3 c = clamp(ivec3(floor(g)), ivec3(0), n - 1);
    int width = int(dims.w);
    int linear = (c.z * n.y + c.y) * n.x + c.x;
    vec4 t = texelFetch(cones, ivec2(linear - (linear / width) * width, linear / width), 0);
    if (t.b > 254.5 / 255.0) {
        return 1.0;
    }
    vec3 axis = hexapodLayerConeAxis(t.rg);
    float halfAngle = t.b * 255.0 * 180.0 / 254.0;
    vec3 view = normalize(p - eye.xyz);
    float angle = degrees(acos(clamp(dot(view, axis), -1.0, 1.0)));
    return clamp((halfAngle + eye.w - angle) / eye.w, 0.0, 1.0);
}

vec2 hexapodLayerDim(vec4 instanceParams, vec4 instanceDim) {
    return instanceParams.x > 0.5 && instanceParams.z > 0.5 ? instanceDim.xy : vec2(1.0);
}

vec4 hexapodLayerColor(vec3 scanPosition, vec4 color, mat4 frame, vec4 tint, vec4 pattern,
                       highp sampler2D cones, vec4 grid, vec4 dims, vec4 eye, vec2 dimmed) {
    vec3 p = (frame * vec4(scanPosition, 1.0)).xyz;
    color.a *= hexapodLayerCone(p, cones, grid, dims, eye);
    color = vec4(color.rgb * dimmed.x, color.a * dimmed.y);
    if (pattern.w < 0.5) {
        return color;
    }
    vec3 rgb = mix(color.rgb, tint.rgb, tint.a);
    float band = fract(dot(p, vec3(0.57735027)) / pattern.x);
    rgb *= band < 0.5 ? 1.0 : pattern.y;
    return vec4(rgb, color.a * pattern.z);
}
`;

/**
 * What `LAYER_LOOK_GLSL` does to one splat (straight alpha), for tests: its position in the
 * scan's frame, its colour, the look and the eye (the scan's frame), and the objects' dim
 * while one is highlighted (`dimmed`, colour and opacity factors).
 */
export function evaluateLayerLook(
  position: readonly [number, number, number],
  color: readonly [number, number, number, number],
  look: LayerLook,
  eye: readonly [number, number, number],
  dimmed: readonly [number, number] = [1, 1],
): [number, number, number, number] {
  const m = look.fromScan;
  const p = [0, 1, 2].map(
    (r) =>
      (m[r] ?? 0) * position[0] +
      (m[4 + r] ?? 0) * position[1] +
      (m[8 + r] ?? 0) * position[2] +
      (m[12 + r] ?? 0),
  ) as [number, number, number];
  let weight = 1;
  const cones = look.cones;
  if (cones) {
    const { meta, data } = cones;
    const [nx, ny, nz] = meta.dims;
    const cell = (k: 0 | 1 | 2, n: number): number =>
      Math.min(n - 1, Math.max(0, Math.floor((p[k] - meta.origin[k]) / meta.cell)));
    const linear = (cell(2, nz) * ny + cell(1, ny)) * nx + cell(0, nx);
    const texel = [0, 1, 2, 3].map((k) => data[linear * 4 + k] ?? 0) as [
      number,
      number,
      number,
      number,
    ];
    const local = coneEye(look, [...eye]);
    const d = [p[0] - (local[0] ?? 0), p[1] - (local[1] ?? 0), p[2] - (local[2] ?? 0)];
    const length = Math.hypot(d[0] ?? 0, d[1] ?? 0, d[2] ?? 0) || 1;
    weight = visibility(
      texel,
      [(d[0] ?? 0) / length, (d[1] ?? 0) / length, (d[2] ?? 0) / length],
      meta.fadeDeg,
    );
  }
  return inferredHighlightColor(
    [
      color[0] * dimmed[0],
      color[1] * dimmed[0],
      color[2] * dimmed[0],
      color[3] * weight * dimmed[1],
    ],
    p,
    look.highlight,
  );
}
