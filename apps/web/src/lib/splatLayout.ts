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
