/**
 * From where each part of a scan was seen (hexapod.viewcones v1, written by
 * `tools/captures/view_cones.py` next to `tileset.json` and declared on its root tile as
 * `extras.viewCones`), and the arithmetic the splat vertex shader applies with it.
 *
 * A capture walked inside a site looks right from where it was walked and like fog from the
 * globe's overview: the overview looks at the backs of trees only ever seen from inside, at
 * sky-coloured gaussians a canopy was given to explain the sky between its branches. The
 * packer gives every cell of a coarse grid a cone of directions it was seen from; a splat
 * seen from outside its cell's cone fades out over `fadeDeg`, and inside it is drawn exactly
 * as before. The view from where the capture was taken does not change.
 *
 * `viewcones.bin` is gzip; inside, one RGBA8 texel per cell, x fastest: `r`, `g` the cone's
 * axis, octahedrally encoded; `b` its half-angle (`deg * 254 / 180`), 255 when the cell is
 * seen from everywhere. Cell `(ix, iy, iz)` spans `origin + [i, i + 1) * cell` in the
 * tileset's local east/north/up metres; a point outside the grid takes the nearest edge cell.
 */

export type Vec3 = readonly [number, number, number];
/** Column-major 4x4, as CesiumJS's `Matrix4` packs it. */
export type Mat4 = readonly number[];

export interface ViewConesMeta {
  format: "hexapod.viewcones";
  version: 1;
  uri: string;
  origin: Vec3;
  cell: number;
  dims: Vec3;
  fadeDeg: number;
  directionalShare?: number;
}

/** A cell whose `b` is this is seen from everywhere. */
export const OMNI = 255;
/** Texels a row of the GPU texture holds. */
export const VIEW_CONES_TEXTURE_WIDTH = 4096;

/** Reads `extras.viewCones` off a tileset's root tile; null when absent or not this format. */
export function viewConesMetaOf(extras: unknown): ViewConesMeta | null {
  const meta = (extras as { viewCones?: Record<string, unknown> } | null | undefined)?.viewCones;
  if (meta?.format !== "hexapod.viewcones" || meta.version !== 1) return null;
  const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
  const triple = (v: unknown): boolean => Array.isArray(v) && v.length === 3 && v.every(finite);
  const dims = meta.dims as unknown[];
  if (
    typeof meta.uri !== "string" ||
    !finite(meta.cell) ||
    !(meta.cell > 0) ||
    !finite(meta.fadeDeg) ||
    !(meta.fadeDeg > 0) ||
    !triple(meta.origin) ||
    !triple(meta.dims) ||
    !dims.every((v) => Number.isInteger(v) && (v as number) > 0)
  ) {
    return null;
  }
  return meta as unknown as ViewConesMeta;
}

/** Cells in the grid. */
export function cellCount(meta: ViewConesMeta): number {
  return meta.dims[0] * meta.dims[1] * meta.dims[2];
}

async function gunzip(bytes: ArrayBuffer): Promise<Uint8Array> {
  const body = new Response(bytes).body;
  if (!body) throw new Error("view cones: empty response body");
  const stream = body.pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Fetches a scan's view-cone texels, `meta.uri` relative to its tileset; checks their size. */
export async function loadViewCones(tilesetUrl: string, meta: ViewConesMeta): Promise<Uint8Array> {
  const base = new URL(tilesetUrl, globalThis.location?.href);
  const response = await fetch(new URL(meta.uri, base).toString());
  if (!response.ok) throw new Error(`view cones answered ${String(response.status)}`);
  const texels = await gunzip(await response.arrayBuffer());
  if (texels.length !== cellCount(meta) * 4) {
    throw new Error(
      `view cones are ${String(texels.length)} bytes; a ${meta.dims.join("x")} grid is ` +
        String(cellCount(meta) * 4),
    );
  }
  return texels;
}

/** The texels padded to whole rows of `VIEW_CONES_TEXTURE_WIDTH`, and the row count. */
export function textureRows(texels: Uint8Array): { data: Uint8Array; height: number } {
  const cells = texels.length / 4;
  const height = Math.max(1, Math.ceil(cells / VIEW_CONES_TEXTURE_WIDTH));
  const data = new Uint8Array(VIEW_CONES_TEXTURE_WIDTH * height * 4);
  data.set(texels);
  return { data, height };
}

/**
 * The matrix from the draw command's model frame -- the baked frame the splat attribute
 * texture holds, `p_b = B · p_local` -- to grid cell coordinates: `(B⁻¹ · p_b − origin) / cell`.
 * Uniformly scaled, so directions in it are directions in the scan's frame.
 */
export function gridFromModel(meta: ViewConesMeta, inverseBake: Mat4): number[] {
  const s = 1 / meta.cell;
  const out = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c += 1) {
    for (let r = 0; r < 3; r += 1) {
      const v = inverseBake[c * 4 + r] ?? 0;
      out[c * 4 + r] = s * (c === 3 ? v - (meta.origin[r] ?? 0) * (inverseBake[15] ?? 1) : v);
    }
    out[c * 4 + 3] = inverseBake[c * 4 + 3] ?? (c === 3 ? 1 : 0);
  }
  return out;
}

/** The octahedral decode the shader does: `r`, `g` bytes to a unit vector. */
export function decodeAxis(r: number, g: number): Vec3 {
  let x = (r / 255) * 2 - 1;
  let y = (g / 255) * 2 - 1;
  const z = 1 - Math.abs(x) - Math.abs(y);
  const t = Math.max(-z, 0);
  x += x >= 0 ? -t : t;
  y += y >= 0 ? -t : t;
  const n = Math.hypot(x, y, z);
  return [x / n, y / n, z / n];
}

/**
 * The weight a viewer gives a splat in a cell with texel `(r, g, b)`, seen along the unit
 * direction `view` (viewer to splat): 1 inside the cone, 0 past its edge by `fadeDeg`.
 * The reference for `VIEW_CONES_GLSL`, and the same arithmetic as `view_cones.visibility`.
 */
export function visibility(
  texel: readonly [number, number, number, number],
  view: Vec3,
  fadeDeg: number,
): number {
  if (texel[2] === OMNI) return 1;
  const axis = decodeAxis(texel[0], texel[1]);
  const halfAngle = (texel[2] * 180) / 254;
  const cos = Math.min(1, Math.max(-1, axis[0] * view[0] + axis[1] * view[1] + axis[2] * view[2]));
  const angle = (Math.acos(cos) * 180) / Math.PI;
  return Math.min(1, Math.max(0, (halfAngle + fadeDeg - angle) / fadeDeg));
}

/**
 * The patched engine's `splatVertexVisibility` (`GaussianSplatPrimitive.vertexVisibility`):
 * the splat's cell, clamped to the grid; the camera in the same frame; the fade.
 */
export const VIEW_CONES_GLSL = `
vec3 viewConeAxis(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 v = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-v.z, 0.0);
    v.x += v.x >= 0.0 ? -t : t;
    v.y += v.y >= 0.0 ? -t : t;
    return normalize(v);
}

float splatVertexVisibility(uint splatIndex, vec3 position) {
    if (u_viewConeActive < 0.5) {
        return 1.0;
    }
    vec3 g = (u_viewConeFromModel * vec4(position, 1.0)).xyz;
    ivec3 dims = ivec3(u_viewConeDims.xyz);
    ivec3 c = clamp(ivec3(floor(g)), ivec3(0), dims - 1);
    int width = int(u_viewConeDims.w);
    int linear = (c.z * dims.y + c.y) * dims.x + c.x;
    vec4 t = texelFetch(u_viewCones, ivec2(linear - (linear / width) * width, linear / width), 0);
    if (t.b > 254.5 / 255.0) {
        return 1.0;
    }
    vec3 axis = viewConeAxis(t.rg);
    float halfAngle = t.b * 255.0 * 180.0 / 254.0;
    vec3 eye = (u_viewConeFromModel * czm_inverseModelView * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec3 view = normalize(g - eye);
    float angle = degrees(acos(clamp(dot(view, axis), -1.0, 1.0)));
    return clamp((halfAngle + u_viewConeFade - angle) / u_viewConeFade, 0.0, 1.0);
}
`;
