/**
 * A decoded SPZ cloud laid out as CesiumJS's glTF loader wants its attributes, so the loader
 * takes them as they are (engine patch, `GltfVertexBufferLoader` processSpz) instead of
 * looping over every splat on the main thread: colours as RGBA bytes, and each spherical
 * harmonic coefficient as its own xyz-per-splat array.
 *
 * The same arithmetic as the engine's own loops, so the result is identical: colour and
 * alpha times 255, clamped to [0, 255], truncated into a byte; coefficient `n` of degree `l`
 * at `base[l - 1] + 3 n` within a splat's `stride` (9, 24 or 45 for degree 1, 2, 3).
 */

export interface DecodedCloud {
  numPoints: number;
  shDegree: number;
  colors: Float32Array;
  alphas: Float32Array;
  sh: Float32Array;
}

const SH_STRIDE = [0, 9, 24, 45];
const SH_BASE = [0, 9, 24];

/** RGBA bytes, four per splat. */
export function rgbaOf(cloud: Pick<DecodedCloud, "numPoints" | "colors" | "alphas">): Uint8Array {
  const { numPoints: count, colors, alphas } = cloud;
  const rgba = new Uint8Array(count * 4);
  const byte = (v: number): number => Math.min(255, Math.max(0, v * 255));
  for (let i = 0; i < count; i++) {
    rgba[i * 4] = byte(colors[i * 3] ?? 0);
    rgba[i * 4 + 1] = byte(colors[i * 3 + 1] ?? 0);
    rgba[i * 4 + 2] = byte(colors[i * 3 + 2] ?? 0);
    rgba[i * 4 + 3] = byte(alphas[i] ?? 0);
  }
  return rgba;
}

/** Each coefficient's xyz per splat, keyed `"l:n"` (degree 1..shDegree, n in 0..2l). */
export function shCoefficientsOf(
  cloud: Pick<DecodedCloud, "numPoints" | "shDegree" | "sh">,
): Record<string, Float32Array> {
  const { numPoints: count, shDegree: degree, sh } = cloud;
  const stride = SH_STRIDE[degree] ?? 0;
  const out: Record<string, Float32Array> = {};
  for (let l = 1; l <= Math.min(degree, 3); l++) {
    for (let n = 0; n < 2 * l + 1; n++) {
      const values = new Float32Array(count * 3);
      const offset = (SH_BASE[l - 1] ?? 0) + n * 3;
      for (let i = 0; i < count; i++) {
        const at = i * stride + offset;
        values[i * 3] = sh[at] ?? 0;
        values[i * 3 + 1] = sh[at + 1] ?? 0;
        values[i * 3 + 2] = sh[at + 2] ?? 0;
      }
      out[`${String(l)}:${String(n)}`] = values;
    }
  }
  return out;
}

/** The degree-0 spherical-harmonic constant: colour = 0.5 + SH_C0 * dc. */
export const SH_C0 = 0.28209479177387814;
/** Coefficients per splat for each SH degree, all bands up to it. */
const SH_COEFFICIENTS = [0, 3, 8, 15];

export interface DecodedGeometry extends DecodedCloud {
  positions: Float32Array;
  scales: Float32Array;
  rotations: Float32Array;
}

/**
 * A decoded SPZ cloud as PlayCanvas's `GSplatData` takes it, in the glTF
 * KHR_gaussian_splatting convention PlayCanvas reads from glTF (`activated`: linear scale,
 * opacity after the sigmoid) -- the same values spz-loader gives CesiumJS. Colour goes back to
 * its degree-0 coefficient (`f_dc`), rotations from xyzw to PlayCanvas's w-first `rot_0..3`,
 * and higher bands to PLY's `f_rest` order: every coefficient's red, then green, then blue.
 * Bands above `maxShDegree` are left out (a phone keeps one: scanView/quality.ts).
 */
export function playcanvasProperties(
  cloud: DecodedGeometry,
  maxShDegree = 3,
): Record<string, Float32Array> {
  const n = cloud.numPoints;
  const out: Record<string, Float32Array> = {};
  const column = (source: Float32Array, stride: number, offset: number): Float32Array => {
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) values[i] = source[i * stride + offset] ?? 0;
    return values;
  };
  out.x = column(cloud.positions, 3, 0);
  out.y = column(cloud.positions, 3, 1);
  out.z = column(cloud.positions, 3, 2);
  out.rot_0 = column(cloud.rotations, 4, 3);
  out.rot_1 = column(cloud.rotations, 4, 0);
  out.rot_2 = column(cloud.rotations, 4, 1);
  out.rot_3 = column(cloud.rotations, 4, 2);
  out.scale_0 = column(cloud.scales, 3, 0);
  out.scale_1 = column(cloud.scales, 3, 1);
  out.scale_2 = column(cloud.scales, 3, 2);
  out.opacity = column(cloud.alphas, 1, 0);
  for (let c = 0; c < 3; c++) {
    const dc = new Float32Array(n);
    for (let i = 0; i < n; i++) dc[i] = ((cloud.colors[i * 3 + c] ?? 0.5) - 0.5) / SH_C0;
    out[`f_dc_${String(c)}`] = dc;
  }
  // The stride is the file's; only the first `kept` coefficients of each splat are read (the
  // bands are stored lowest first, so the first ones are the lower bands).
  const stride = (SH_COEFFICIENTS[Math.min(cloud.shDegree, 3)] ?? 0) * 3;
  const kept = SH_COEFFICIENTS[Math.max(0, Math.min(cloud.shDegree, maxShDegree, 3))] ?? 0;
  for (let c = 0; c < 3; c++) {
    for (let k = 0; k < kept; k++) {
      const values = new Float32Array(n);
      for (let i = 0; i < n; i++) values[i] = cloud.sh[i * stride + k * 3 + c] ?? 0;
      out[`f_rest_${String(c * kept + k)}`] = values;
    }
  }
  return out;
}

/** Morton code bits per axis, as PlayCanvas's `GSplatData.calcMortonOrder`. */
const MORTON_BITS = 10;
const MORTON_CELLS = 1 << MORTON_BITS;

/** Spreads the low 10 bits of `v` two apart (bit i to bit 3i). */
function spread(v: number): number {
  let x = v & (MORTON_CELLS - 1);
  x = (x ^ (x << 16)) & 0xff0000ff;
  x = (x ^ (x << 8)) & 0x0300f00f;
  x = (x ^ (x << 4)) & 0x030c30c3;
  x = (x ^ (x << 2)) & 0x09249249;
  return x;
}

/**
 * The order PlayCanvas's `GSplatData.calcMortonOrder` gives the splats at `x`, `y`, `z`:
 * along a Morton curve through a 1024³ grid over their extent, splats in the same cell in
 * their own order. PlayCanvas reorders a resource's splats this way so neighbours on screen
 * are neighbours in its textures; it used to run on the main thread, a `Map` of arrays per
 * cell, as each tile arrived (scanView/playcanvasBackend.ts). Here it is a stable two-pass
 * radix sort on the 30-bit codes, for a worker (playcanvasTile.worker.ts): the same order.
 * `order[i]` is the splat that goes to slot `i`.
 */
export function mortonOrder(x: Float32Array, y: Float32Array, z: Float32Array): Uint32Array {
  const n = x.length;
  const range = (axis: Float32Array): [number, number] => {
    let min = axis[0] ?? 0;
    let max = min;
    for (let i = 1; i < n; i++) {
      const v = axis[i] ?? 0;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return [min, max];
  };
  const [minX, maxX] = range(x);
  const [minY, maxY] = range(y);
  const [minZ, maxZ] = range(z);
  const sizeX = minX === maxX ? 0 : MORTON_CELLS / (maxX - minX);
  const sizeY = minY === maxY ? 0 : MORTON_CELLS / (maxY - minY);
  const sizeZ = minZ === maxZ ? 0 : MORTON_CELLS / (maxZ - minZ);
  const codes = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const ix = Math.min(MORTON_CELLS - 1, Math.floor(((x[i] ?? 0) - minX) * sizeX));
    const iy = Math.min(MORTON_CELLS - 1, Math.floor(((y[i] ?? 0) - minY) * sizeY));
    const iz = Math.min(MORTON_CELLS - 1, Math.floor(((z[i] ?? 0) - minZ) * sizeZ));
    codes[i] = (spread(iz) << 2) + (spread(iy) << 1) + spread(ix);
  }
  // Least significant 15 bits first, then the most: each pass stable, so equal codes keep
  // their splats' own order, as PlayCanvas's per-cell arrays do.
  const RADIX = 1 << 15;
  const counts = new Uint32Array(RADIX);
  let from = new Uint32Array(n);
  let to = new Uint32Array(n);
  for (let i = 0; i < n; i++) from[i] = i;
  for (const shift of [0, 15]) {
    counts.fill(0);
    for (let i = 0; i < n; i++) {
      const digit = ((codes[i] ?? 0) >>> shift) & (RADIX - 1);
      counts[digit] = (counts[digit] ?? 0) + 1;
    }
    let running = 0;
    for (let d = 0; d < RADIX; d++) {
      const c = counts[d] ?? 0;
      counts[d] = running;
      running += c;
    }
    for (let i = 0; i < n; i++) {
      const index = from[i] ?? 0;
      const digit = ((codes[index] ?? 0) >>> shift) & (RADIX - 1);
      const at = counts[digit] ?? 0;
      to[at] = index;
      counts[digit] = at + 1;
    }
    [from, to] = [to, from];
  }
  return from;
}

/** Every column put in `order` (as PlayCanvas's `GSplatData.reorder`): slot `i` gets `order[i]`. */
export function reorderColumns(
  columns: Record<string, Float32Array>,
  order: Uint32Array,
): Record<string, Float32Array> {
  const out: Record<string, Float32Array> = {};
  for (const [name, values] of Object.entries(columns)) {
    const moved = new Float32Array(order.length);
    for (let i = 0; i < order.length; i++) moved[i] = values[order[i] ?? 0] ?? 0;
    out[name] = moved;
  }
  return out;
}

/**
 * Moves the `x`, `y`, `z` columns so they are relative to the middle of their extent, and
 * returns that middle: positions near zero keep their precision on a GPU that stores them as
 * half floats (6 cm apart 100 m from the origin), and the caller places the tile there.
 */
export function centreColumns(columns: Record<string, Float32Array>): [number, number, number] {
  const centre = (axis: Float32Array | undefined): number => {
    if (!axis || axis.length === 0) return 0;
    let low = Infinity;
    let high = -Infinity;
    for (const value of axis) {
      if (value < low) low = value;
      if (value > high) high = value;
    }
    const middle = (low + high) / 2;
    for (let i = 0; i < axis.length; i++) axis[i] = (axis[i] ?? 0) - middle;
    return middle;
  };
  return [centre(columns.x), centre(columns.y), centre(columns.z)];
}
