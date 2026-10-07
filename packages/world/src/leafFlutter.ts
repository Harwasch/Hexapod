/**
 * Leaf flutter as a spatially correlated field advected with the wind.
 *
 * The per-splat flutter in `flutter.ts` gives every splat a phase from a hash of its *index*, so
 * two splats a millimetre apart on one leaf move independently and the crown boils. Real foliage
 * moves in patches that a gust carries across the crown. This is Habel, Kusternig & Wimmer, EG
 * 2009, §7.2: leaves are samplers of a turbulent field; each point is projected onto planes,
 * offset by `−W·t`, and looked up in 2D motion textures, so nearby leaves share their motion
 * and the pattern travels downwind. Two departures, both stated:
 *
 * - **One oblique plane per displacement component** rather than three axis planes blended per
 *   component (Habel eq. 24: nine lookups). Each component projects the point onto its own plane
 *   through the downwind axis, tilted so that none of the three planes contains the crosswind or
 *   the vertical axis; a field read off a plane is constant along the plane's normal, and a
 *   normal along an axis would leave the crown in lockstep along that axis. Three lookups a splat
 *   instead of nine.
 * - The field is carried at `canopyAdvection · U`, a fraction of the mean speed, because wind
 *   inside a crown is much slower than above it; the fraction is an estimate.
 *
 * The texture's shortest wavelength is 4 leaf sizes (Habel: "the minimum wavelength represented
 * in the leaf motion textures is 4 times the maximum leaf size. This avoids too high
 * frequencies which could cause vertices of a single leaf to behave inconsistently") and its
 * longest 10, which sets the size of a coherent patch.
 *
 * Calm is exact: at `U = 0` the field reports `still`, and {@link applyAdvectedFlutter} does not
 * touch a coordinate.
 */

import { type FlutterField } from "./flutter";
import { SKIN_INFLUENCES, SKIN_WEIGHT_TOTAL, skinCount, type SplatSkin } from "./skin";
import { sampleTexture, trajectoryDirection, type MotionTexture } from "./spectral";

/** The flutter field of one frame: per-node amplitude plus what every splat needs to look up. */
export interface AdvectedFlutterField extends FlutterField {
  readonly kind: "advected";
  readonly texture: MotionTexture;
  /** Downwind unit vector, east and north components. */
  readonly downwind: readonly [number, number];
  /** Metres to texels: `1 / leafSizeM`. */
  readonly texelsPerMeter: number;
  /** How far the field has been carried downwind, texels. */
  readonly advectionTexels: number;
  /**
   * Three affine maps, one per displacement component (downwind, crosswind, up), from wind-frame
   * texel coordinates `(a, c, z)` to texture coordinates: eight numbers each,
   * `[m00, m01, m02, m10, m11, m12, ox, oy]`.
   */
  readonly lookups: Float64Array;
}

export function isAdvectedFlutter(field: FlutterField): field is AdvectedFlutterField {
  return (field as Partial<AdvectedFlutterField>).kind === "advected";
}

/**
 * Normals of the three lookup planes in the wind frame `(a, c, z)`. Mostly crosswind-and-up,
 * a little downwind, and no two alike — so no displacement component is constant along the
 * crosswind axis, the vertical, or the wind.
 */
export const FLUTTER_PLANE_NORMALS: readonly (readonly [number, number, number])[] = [
  [0.2, 0.62, 0.76],
  [-0.15, 0.79, -0.6],
  [0.25, -0.35, 0.9],
];

/** Offsets into the texture, in fractions of its size: far apart, so the lookups are unrelated. */
const LOOKUP_OFFSETS: readonly (readonly [number, number])[] = [
  [0.11, 0.37],
  [0.53, 0.83],
  [0.79, 0.19],
];

/**
 * The three lookup maps for a seed. Deterministic.
 *
 * Each maps the plane through the downwind axis `â` with normal `n` isometrically into the
 * texture, turned so that `â`'s image — the direction the pattern is advected along — is an
 * aperiodic trajectory for an hour's travel (`trajectoryDirection`).
 */
export function flutterLookups(
  seed: number,
  size: number,
  horizonTexels: number,
  wavelengthTexels: number,
): Float64Array {
  const out = new Float64Array(24);
  FLUTTER_PLANE_NORMALS.forEach((raw, j) => {
    const length = Math.hypot(raw[0], raw[1], raw[2]);
    const n = [raw[0] / length, raw[1] / length, raw[2] / length] as const;
    // u: the downwind axis projected into the plane; w = n × u.
    const ua = 1 - n[0] * n[0];
    const uc = -n[0] * n[1];
    const uz = -n[0] * n[2];
    const ul = Math.hypot(ua, uc, uz);
    const u = [ua / ul, uc / ul, uz / ul] as const;
    const w = [
      n[1] * u[2] - n[2] * u[1],
      n[2] * u[0] - n[0] * u[2],
      n[0] * u[1] - n[1] * u[0],
    ] as const;
    const [dx, dy] = trajectoryDirection(
      0x1eaf + j,
      seed,
      size,
      horizonTexels * ul,
      wavelengthTexels,
    );
    // Rotation taking (1, 0) to (dx, dy): [[dx, −dy], [dy, dx]], applied to (p·u, p·w).
    const o = j * 8;
    out[o] = dx * u[0] - dy * w[0];
    out[o + 1] = dx * u[1] - dy * w[1];
    out[o + 2] = dx * u[2] - dy * w[2];
    out[o + 3] = dy * u[0] + dx * w[0];
    out[o + 4] = dy * u[1] + dx * w[1];
    out[o + 5] = dy * u[2] + dx * w[2];
    out[o + 6] = (LOOKUP_OFFSETS[j]?.[0] ?? 0) * size;
    out[o + 7] = (LOOKUP_OFFSETS[j]?.[1] ?? 0) * size;
  });
  return out;
}

/**
 * The flutter offset at one canonical position, before the node's amplitude is applied, into
 * `out[0..2]` (east, north, up). The readable definition the batch loop below is tested against.
 */
export function advectedFlutterUnit(
  field: AdvectedFlutterField,
  x: number,
  y: number,
  z: number,
  out: Float64Array,
): void {
  const [ex, ey] = field.downwind;
  const k = field.texelsPerMeter;
  const a = (x * ex + y * ey) * k - field.advectionTexels;
  // Crosswind unit vector, 90° clockwise from downwind: (ey, −ex).
  const c = (x * ey - y * ex) * k;
  const h = z * k;
  const m = field.lookups;
  const value = (j: number): number => {
    const o = j * 8;
    return sampleTexture(
      field.texture,
      (m[o] ?? 0) * a + (m[o + 1] ?? 0) * c + (m[o + 2] ?? 0) * h + (m[o + 6] ?? 0),
      (m[o + 3] ?? 0) * a + (m[o + 4] ?? 0) * c + (m[o + 5] ?? 0) * h + (m[o + 7] ?? 0),
    );
  };
  const along = value(0);
  const across = value(1);
  out[0] = along * ex + across * ey;
  out[1] = along * ey - across * ex;
  out[2] = value(2);
}

/** `|advectedFlutterUnit|` can never exceed this: three components, each at most `maxAbs`. */
export function advectedFlutterUnitBound(texture: MotionTexture): number {
  return Math.sqrt(3) * texture.maxAbs;
}

/**
 * Adds every splat's advected flutter to `target`, in place. `positions` are the canonical
 * positions the offsets are looked up at — never the displaced ones, so there is no feedback.
 * Does nothing for a still field.
 */
export function applyAdvectedFlutter(
  target: Float32Array,
  positions: Float32Array,
  assignment: Uint16Array,
  field: AdvectedFlutterField,
  count: number,
  /** Blends each splat's amplitude over its skin's nodes; `assignment` is then not read. */
  skin?: SplatSkin,
): void {
  if (field.still) return;
  const amplitudes = field.amplitudeM;
  const limit = Math.min(
    count,
    skin === undefined ? assignment.length : skinCount(skin),
    Math.floor(target.length / 3),
  );
  // `advectedFlutterUnit`, written out for the per-splat loop: the advection is folded into
  // each map's offset once per frame and the bilinear sample is inlined. Same arithmetic up to
  // float association; `living.test.ts` pins the two together.
  const [ex, ey] = field.downwind;
  const k = field.texelsPerMeter;
  const m = field.lookups;
  const n = field.texture.size;
  const mask = n - 1;
  const data = field.texture.data;
  const adv = field.advectionTexels;
  const m00 = m[0] ?? 0,
    m01 = m[1] ?? 0,
    m02 = m[2] ?? 0,
    m03 = m[3] ?? 0,
    m04 = m[4] ?? 0,
    m05 = m[5] ?? 0;
  const m10 = m[8] ?? 0,
    m11 = m[9] ?? 0,
    m12 = m[10] ?? 0,
    m13 = m[11] ?? 0,
    m14 = m[12] ?? 0,
    m15 = m[13] ?? 0;
  const m20 = m[16] ?? 0,
    m21 = m[17] ?? 0,
    m22 = m[18] ?? 0,
    m23 = m[19] ?? 0,
    m24 = m[20] ?? 0,
    m25 = m[21] ?? 0;
  const o0x = (m[6] ?? 0) - m00 * adv,
    o0y = (m[7] ?? 0) - m03 * adv;
  const o1x = (m[14] ?? 0) - m10 * adv,
    o1y = (m[15] ?? 0) - m13 * adv;
  const o2x = (m[22] ?? 0) - m20 * adv,
    o2y = (m[23] ?? 0) - m23 * adv;
  const sample = (x: number, y: number): number => {
    const fx = x - Math.floor(x / n) * n;
    const fy = y - Math.floor(y / n) * n;
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const tx = fx - x0;
    const ty = fy - y0;
    const xa = x0 & mask;
    const xb = (x0 + 1) & mask;
    const ya = (y0 & mask) * n;
    const yb = ((y0 + 1) & mask) * n;
    const a = data[ya + xa] ?? 0;
    const b = data[ya + xb] ?? 0;
    const c = data[yb + xa] ?? 0;
    const d = data[yb + xb] ?? 0;
    return (a + (b - a) * tx) * (1 - ty) + (c + (d - c) * tx) * ty;
  };
  const skinNodes = skin?.nodes;
  const skinWeights = skin?.weights;
  const influences = SKIN_INFLUENCES;
  const scale = 1 / SKIN_WEIGHT_TOTAL;
  for (let i = 0; i < limit; i += 1) {
    let amplitude = 0;
    if (skinNodes === undefined || skinWeights === undefined) {
      amplitude = amplitudes[assignment[i] ?? 0] ?? 0;
    } else {
      // `blendedAmplitude`, inlined.
      const at = i * influences;
      for (let j = 0; j < influences; j += 1) {
        const q = skinWeights[at + j] ?? 0;
        if (q !== 0) amplitude += q * (amplitudes[skinNodes[at + j] ?? 0] ?? 0);
      }
      amplitude *= scale;
    }
    if (amplitude === 0) continue;
    const base = i * 3;
    const x = positions[base] ?? 0;
    const y = positions[base + 1] ?? 0;
    const a = (x * ex + y * ey) * k;
    const c = (x * ey - y * ex) * k;
    const h = (positions[base + 2] ?? 0) * k;
    const along = sample(m00 * a + m01 * c + m02 * h + o0x, m03 * a + m04 * c + m05 * h + o0y);
    const across = sample(m10 * a + m11 * c + m12 * h + o1x, m13 * a + m14 * c + m15 * h + o1y);
    const up = sample(m20 * a + m21 * c + m22 * h + o2x, m23 * a + m24 * c + m25 * h + o2y);
    target[base] = (target[base] ?? 0) + amplitude * (along * ex + across * ey);
    target[base + 1] = (target[base + 1] ?? 0) + amplitude * (along * ey - across * ex);
    target[base + 2] = (target[base + 2] ?? 0) + amplitude * up;
  }
}

/** `Σ_k (w_k/1023)·A[node_k]`: a splat's flutter amplitude under its skin. */
export function blendedAmplitude(skin: SplatSkin, i: number, amplitudes: Float64Array): number {
  const at = i * SKIN_INFLUENCES;
  let sum = 0;
  for (let k = 0; k < SKIN_INFLUENCES; k += 1) {
    const q = skin.weights[at + k] ?? 0;
    if (q === 0) continue;
    sum += q * (amplitudes[skin.nodes[at + k] ?? 0] ?? 0);
  }
  return sum / SKIN_WEIGHT_TOTAL;
}
