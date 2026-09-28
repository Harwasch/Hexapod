/**
 * Believability tests for Living Mode phase 1 (docs/DECISIONS/0008-living-mode.md), on the
 * synthetic tree. Each test is one of the phase's pass criteria, stated as a number:
 *
 * | criterion                                   | pass when                                   |
 * | ------------------------------------------- | ------------------------------------------- |
 * | each branch rings at its model frequency    | local-bend PSD peak within ±10 % of `f_b`   |
 * | the whole tree follows the height law       | trunk PSD peak within ±10 % of `2.4/√H`     |
 * | deflection is drag                          | RMS deflection ∝ `U^n`, `n = 2 ± 0.2`       |
 * | it never repeats                            | no autocorrelation > 0.2 from 10 s to 1 h   |
 * | the crown moves in patches                  | flutter corr. at 4ℓ < ½ of that at 0.5ℓ     |
 * | calm is the measurement                     | identity by value, flutter still            |
 * | same clock, same frame                      | bit-identical output for the same `t`       |
 *
 * The measured values are printed (`MEASURED …`) so a run can be quoted; the thresholds are the
 * phase-1 pass criteria, not tuned to the numbers. What none of this can settle is whether a
 * person finds the motion believable — that is the blind comparison in
 * `apps/web/e2e/livingCompare.spec.ts`.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  advectedFlutterUnit,
  applyTransform,
  assignSplatsToNodes,
  createLivingMotion,
  deform,
  deformPositions,
  deriveMotionSidecar,
  fft,
  flutterField,
  IDENTITY_TRANSFORM,
  livingFlutter,
  livingFrame,
  livingMaxDisplacement,
  livingTransforms,
  NO_GUSTS,
  parseMotionSidecar,
  parseRig,
  prepareLivingMotion,
  syntheticTreeRig,
  treeFrequencyHz,
  type LivingMotion,
  type LivingWind,
  type MotionRig,
  type NodeTransform,
  type Quat,
  type Vec3,
} from "./index";

/** The committed fixture: `data/tiles/synthetic-tree/source/`, rig and sidecar as shipped. */
function fixture(name: string): string {
  return readFileSync(
    fileURLToPath(new URL(`../../../data/tiles/synthetic-tree/source/${name}`, import.meta.url)),
    "utf8",
  );
}

const rig = parseRig(fixture("rig.json"));
const sidecar = parseMotionSidecar(fixture("motion.json"), rig);
const motion = createLivingMotion(rig, sidecar);
prepareLivingMotion(motion);

function windAt(speedMps: number, gusts = true, bearingDeg = 30): LivingWind {
  return {
    speedMps,
    bearingDeg,
    gust: gusts ? sidecar.wind.gust : NO_GUSTS,
    season: "summer",
  };
}

// ---------------------------------------------------------------------------------------------
// Signal helpers
// ---------------------------------------------------------------------------------------------

/** Welch PSD (Hann, 50 % overlap), one-sided, of a mean-removed signal. */
function welch(
  signal: Float64Array,
  segment: number,
  fs: number,
): { psd: Float64Array; df: number } {
  const out = new Float64Array(segment / 2);
  let mean = 0;
  for (const v of signal) mean += v;
  mean /= signal.length;
  let segments = 0;
  const re = new Float64Array(segment);
  const im = new Float64Array(segment);
  for (let start = 0; start + segment <= signal.length; start += segment / 2) {
    for (let i = 0; i < segment; i += 1) {
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / segment);
      re[i] = ((signal[start + i] ?? 0) - mean) * w;
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k < segment / 2; k += 1)
      out[k] = (out[k] ?? 0) + (re[k] ?? 0) ** 2 + (im[k] ?? 0) ** 2;
    segments += 1;
  }
  for (let k = 0; k < out.length; k += 1) out[k] = (out[k] ?? 0) / Math.max(1, segments);
  return { psd: out, df: fs / segment };
}

/**
 * Frequency of the largest PSD value after smoothing over ±`relative` in frequency — a
 * constant-Q smoother, so a broad, lightly damped peak is located by its body, not by the
 * tallest noise spike on its top.
 */
function peakHz(psd: Float64Array, df: number, lo: number, hi: number, relative = 0.06): number {
  let best = -1;
  let at = 0;
  for (
    let k = Math.max(1, Math.floor(lo / df));
    k <= Math.min(psd.length - 1, Math.ceil(hi / df));
    k += 1
  ) {
    const f = k * df;
    const a = Math.max(1, Math.floor((f * (1 - relative)) / df));
    const b = Math.min(psd.length - 1, Math.ceil((f * (1 + relative)) / df));
    let sum = 0;
    for (let j = a; j <= b; j += 1) sum += psd[j] ?? 0;
    const smoothed = sum / (b - a + 1);
    if (smoothed > best) {
      best = smoothed;
      at = f;
    }
  }
  return at;
}

/** Largest autocorrelation at lags in `[minLag, maxLag]` seconds, unbiased by overlap. */
function maxAutocorrelation(
  signal: Float64Array,
  fs: number,
  minLag: number,
  maxLag: number,
): { value: number; lagS: number } {
  const n = signal.length;
  let size = 1;
  while (size < 2 * n) size <<= 1;
  const re = new Float64Array(size);
  const im = new Float64Array(size);
  let mean = 0;
  for (const v of signal) mean += v;
  mean /= n;
  for (let i = 0; i < n; i += 1) re[i] = (signal[i] ?? 0) - mean;
  fft(re, im);
  for (let i = 0; i < size; i += 1) {
    re[i] = (re[i] ?? 0) ** 2 + (im[i] ?? 0) ** 2;
    im[i] = 0;
  }
  fft(re, im, true);
  const zero = (re[0] ?? 0) / n;
  let value = -1;
  let lagS = 0;
  for (
    let lag = Math.ceil(minLag * fs);
    lag <= Math.min(n - 1, Math.floor(maxLag * fs));
    lag += 1
  ) {
    const r = (re[lag] ?? 0) / (n - lag) / zero;
    if (r > value) {
      value = r;
      lagS = lag / fs;
    }
  }
  return { value, lagS };
}

/** The local rotation of node `i` (relative to its parent) as a rotation vector. */
function localRotationVector(
  transforms: readonly NodeTransform[],
  i: number,
  parent: number,
): Vec3 {
  const q = transforms[i]?.rotation ?? IDENTITY_TRANSFORM.rotation;
  const p =
    parent >= 0
      ? (transforms[parent]?.rotation ?? IDENTITY_TRANSFORM.rotation)
      : IDENTITY_TRANSFORM.rotation;
  // conj(p) ⊗ q
  const [px, py, pz, pw] = [-p[0], -p[1], -p[2], p[3]] as Quat;
  const [qx, qy, qz, qw] = q;
  const x = pw * qx + px * qw + py * qz - pz * qy;
  const y = pw * qy - px * qz + py * qw + pz * qx;
  const z = pw * qz + px * qy - py * qx + pz * qw;
  return [2 * x, 2 * y, 2 * z];
}

function displacementOf(
  transforms: readonly NodeTransform[],
  i: number,
  target: MotionRig = rig,
): Vec3 {
  const p = target.nodes[i]?.position ?? [0, 0, 0];
  const q = applyTransform(transforms[i] ?? IDENTITY_TRANSFORM, p);
  return [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
}

function highestNode(target: MotionRig): number {
  let best = 0;
  target.nodes.forEach((node, i) => {
    if (node.position[2] > (target.nodes[best]?.position[2] ?? 0)) best = i;
  });
  return best;
}

/** Branch bases (first node of each branch, excluding the root). */
const BRANCH_BASES = [...new Set(sidecar.nodes.branch.slice(1))];
const TREE_BASE = sidecar.nodes.branch.find((b, i) => i > 0 && sidecar.nodes.mode[i] === 0) ?? 1;

// ---------------------------------------------------------------------------------------------

describe("each branch rings at its model frequency", () => {
  it("puts every branch's local-bend spectral peak within ±10 % of f = 2.55·L^-0.59", () => {
    const fs = 16;
    const seconds = 900;
    const frames = fs * seconds;
    const wind = windAt(8, false);
    const bases = BRANCH_BASES.filter((b) => b !== TREE_BASE);
    const signals = bases.map(() => [
      new Float64Array(frames),
      new Float64Array(frames),
      new Float64Array(frames),
    ]);
    for (let k = 0; k < frames; k += 1) {
      const transforms = livingTransforms(motion, 500 + k / fs, wind);
      bases.forEach((b, j) => {
        const r = localRotationVector(transforms, b, rig.nodes[b]?.parent ?? -1);
        const s = signals[j];
        if (s === undefined) return;
        (s[0] as Float64Array)[k] = r[0];
        (s[1] as Float64Array)[k] = r[1];
        (s[2] as Float64Array)[k] = r[2];
      });
    }
    const errors: number[] = [];
    bases.forEach((b, j) => {
      const model = sidecar.nodes.frequencyHz[b] ?? 0;
      const parts = (signals[j] ?? []).map((s) => welch(s, 2048, fs));
      const df = parts[0]?.df ?? 1;
      const psd = new Float64Array(parts[0]?.psd.length ?? 0);
      for (const part of parts) part.psd.forEach((v, k) => (psd[k] = (psd[k] ?? 0) + v));
      const peak = peakHz(psd, df, 0.3, 7.5);
      errors.push(peak / model - 1);
    });
    const worst = errors.reduce((a, e) => (Math.abs(e) > Math.abs(a) ? e : a), 0);
    const mean = errors.reduce((a, e) => a + e, 0) / errors.length;
    console.info(
      `MEASURED branch PSD peak vs model: ${bases.length} branches, mean ${(mean * 100).toFixed(1)} %, worst ${(worst * 100).toFixed(1)} %`,
    );
    expect(Math.abs(worst)).toBeLessThanOrEqual(0.1);
  });
});

describe("the whole tree follows the pendulum law", () => {
  it.each([6, 15])("puts a %s m tree's trunk peak within ±10 % of 2.4/√H", (heightM) => {
    const tree = heightM === 6 ? rig : syntheticTreeRig({ heightM });
    const treeSidecar = heightM === 6 ? sidecar : deriveMotionSidecar(tree);
    const treeMotion: LivingMotion = heightM === 6 ? motion : createLivingMotion(tree, treeSidecar);
    // The top of the trunk chain moves with the whole-tree mode alone.
    let top = 1;
    tree.nodes.forEach((node, i) => {
      if (i > 0 && treeSidecar.nodes.mode[i] === 0 && node.band === "trunk") top = i;
    });
    const fs = 8;
    const frames = fs * 1200;
    const along = new Float64Array(frames);
    const across = new Float64Array(frames);
    for (let k = 0; k < frames; k += 1) {
      const d = displacementOf(livingTransforms(treeMotion, 200 + k / fs, windAt(8)), top, tree);
      along[k] = d[0];
      across[k] = d[1];
    }
    const a = welch(along, 4096, fs);
    const c = welch(across, 4096, fs);
    const psd = a.psd.map((v, k) => v + (c.psd[k] ?? 0));
    const law = treeFrequencyHz(treeSidecar.treeHeightM);
    const peak = peakHz(psd, a.df, 0.15, 3);
    console.info(
      `MEASURED whole-tree peak, H = ${treeSidecar.treeHeightM} m: ${peak.toFixed(3)} Hz vs 2.4/√H = ${law.toFixed(3)} Hz (${((peak / law - 1) * 100).toFixed(1)} %)`,
    );
    expect(Math.abs(peak / law - 1)).toBeLessThanOrEqual(0.1);
  });
});

describe("deflection is drag", () => {
  it("scales RMS deflection as U^(2 ± 0.2) from 2 to 10 m/s", () => {
    const speeds = [2, 4, 6, 8, 10];
    const tip = highestNode(rig);
    const fs = 10;
    const frames = fs * 300;
    const rms = speeds.map((u) => {
      let sum = 0;
      for (let k = 0; k < frames; k += 1) {
        const d = displacementOf(livingTransforms(motion, 3000 + k / fs, windAt(u)), tip);
        sum += d[0] ** 2 + d[1] ** 2 + d[2] ** 2;
      }
      return Math.sqrt(sum / frames);
    });
    const xs = speeds.map(Math.log);
    const ys = rms.map(Math.log);
    const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const my = ys.reduce((a, b) => a + b, 0) / ys.length;
    let num = 0;
    let den = 0;
    xs.forEach((x, i) => {
      num += (x - mx) * ((ys[i] ?? 0) - my);
      den += (x - mx) ** 2;
    });
    const exponent = num / den;
    console.info(
      `MEASURED RMS tip deflection ${speeds.map((u, i) => `${u} m/s ${((rms[i] ?? 0) * 100).toFixed(2)} cm`).join(", ")}; fitted exponent ${exponent.toFixed(3)}`,
    );
    expect(Math.abs(exponent - 2)).toBeLessThanOrEqual(0.2);
  });
});

describe("it never repeats", () => {
  it("has no autocorrelation peak above 0.2 at lags from 10 s to 1 h", () => {
    const fs = 2;
    const seconds = 7200;
    const frames = fs * seconds;
    const wind = windAt(8);
    const tip = highestNode(rig);
    // The fastest branch, where the aperiodicity budget is tightest.
    const fastest = BRANCH_BASES.reduce((a, b) =>
      (sidecar.nodes.frequencyHz[b] ?? 0) > (sidecar.nodes.frequencyHz[a] ?? 0) ? b : a,
    );
    let trunkTop = 1;
    rig.nodes.forEach((node, i) => {
      if (node.band === "trunk") trunkTop = i;
    });
    const tipTrack = new Float64Array(frames);
    const trunkTrack = new Float64Array(frames);
    const fastTrack = new Float64Array(frames);
    const tracks: [string, Float64Array][] = [
      ["tip deflection (east)", tipTrack],
      ["trunk top (north)", trunkTrack],
      [
        `fastest branch ${rig.nodes[fastest]?.id ?? "?"} (${sidecar.nodes.frequencyHz[fastest] ?? 0} Hz)`,
        fastTrack,
      ],
    ];
    for (let k = 0; k < frames; k += 1) {
      const transforms = livingTransforms(motion, 10_000 + k / fs, wind);
      tipTrack[k] = displacementOf(transforms, tip)[0];
      trunkTrack[k] = displacementOf(transforms, trunkTop)[1];
      fastTrack[k] = localRotationVector(transforms, fastest, rig.nodes[fastest]?.parent ?? -1)[0];
    }
    for (const [name, track] of tracks) {
      const { value, lagS } = maxAutocorrelation(track, fs, 10, 3600);
      console.info(
        `MEASURED max autocorrelation, ${name}: ${value.toFixed(3)} at ${lagS.toFixed(1)} s`,
      );
      expect({ name, peak: value < 0.2 }).toEqual({ name, peak: true });
    }
  });
});

/** Separations across the wind (at a 30° bearing), vertically, and along it. */
const SEPARATIONS: readonly Vec3[] = [
  [0.8660254, -0.5, 0],
  [0, 0, 1],
  [0.5, 0.8660254, 0],
];

describe("the crown moves in patches", () => {
  it("correlates flutter at half a leaf and decorrelates it by four leaves", () => {
    const leaf = sidecar.leafSizeM;
    const wind = windAt(6);
    const out = new Float64Array(3);
    const other = new Float64Array(3);
    const corr = (separation: number): number => {
      let cov = 0;
      let va = 0;
      let vb = 0;
      for (let b = 0; b < 24; b += 1) {
        const base: Vec3 = [
          ((b * 0.37) % 1) * 3 - 1.5,
          ((b * 0.61) % 1) * 3 - 1.5,
          3 + ((b * 0.29) % 1) * 2,
        ];
        const dir: Vec3 = SEPARATIONS[b % 3] ?? [0, 0, 1];
        for (let k = 0; k < 400; k += 1) {
          const field = livingFlutter(motion, 50 + k * 0.05, wind);
          advectedFlutterUnit(field, base[0], base[1], base[2], out);
          advectedFlutterUnit(
            field,
            base[0] + dir[0] * separation,
            base[1] + dir[1] * separation,
            base[2] + dir[2] * separation,
            other,
          );
          for (let c = 0; c < 3; c += 1) {
            cov += (out[c] ?? 0) * (other[c] ?? 0);
            va += (out[c] ?? 0) ** 2;
            vb += (other[c] ?? 0) ** 2;
          }
        }
      }
      return cov / Math.sqrt(va * vb);
    };
    const near = corr(0.5 * leaf);
    const far = corr(4 * leaf);
    console.info(
      `MEASURED flutter correlation: ${near.toFixed(3)} at 0.5 leaf, ${far.toFixed(3)} at 4 leaves`,
    );
    expect(far).toBeLessThan(0.5 * near);
    expect(near).toBeGreaterThan(0.9);
  });

  it("applies exactly the field it defines, per splat", () => {
    const positions = new Float32Array([0.3, -0.2, 4.1, 1.2, 0.8, 5.0, -1.1, 0.4, 3.3]);
    const assignment = new Uint16Array([
      rig.nodes.length - 1,
      rig.nodes.length - 2,
      rig.nodes.length - 3,
    ]);
    const field = livingFlutter(motion, 321.5, windAt(7));
    const target = new Float32Array(positions.length);
    const moved = deformPositions(
      positions,
      assignment,
      rig.nodes.map(() => IDENTITY_TRANSFORM),
      target,
      field,
    );
    const unit = new Float64Array(3);
    for (let i = 0; i < 3; i += 1) {
      const amplitude = field.amplitudeM[assignment[i] ?? 0] ?? 0;
      expect(amplitude).toBeGreaterThan(0);
      advectedFlutterUnit(
        field,
        positions[i * 3] ?? 0,
        positions[i * 3 + 1] ?? 0,
        positions[i * 3 + 2] ?? 0,
        unit,
      );
      for (let c = 0; c < 3; c += 1) {
        expect(moved[i * 3 + c]).toBeCloseTo(
          (positions[i * 3 + c] ?? 0) + amplitude * (unit[c] ?? 0),
          5,
        );
      }
    }
  });

  it("never repeats at a fixed point either", () => {
    const fs = 4;
    const frames = fs * 7200;
    const track = new Float64Array(frames);
    const out = new Float64Array(3);
    for (let k = 0; k < frames; k += 1) {
      advectedFlutterUnit(livingFlutter(motion, 10_000 + k / fs, windAt(8)), 0.4, -0.3, 4.2, out);
      track[k] = out[0] ?? 0;
    }
    const { value, lagS } = maxAutocorrelation(track, fs, 10, 3600);
    console.info(
      `MEASURED max autocorrelation, leaf flutter at a point: ${value.toFixed(3)} at ${lagS.toFixed(1)} s`,
    );
    expect(value).toBeLessThan(0.2);
  });

  it("carries the pattern downwind", () => {
    // A frozen field advected at canopyAdvection·U: a point downwind sees what an upwind point
    // saw `distance / speed` earlier.
    const wind = windAt(6, false, 90);
    const speed = wind.speedMps * sidecar.wind.canopyAdvection;
    const gap = 0.3;
    const up = new Float64Array(3);
    const down = new Float64Array(3);
    advectedFlutterUnit(livingFlutter(motion, 100, wind), 0, 1, 4, up);
    advectedFlutterUnit(livingFlutter(motion, 100 + gap / speed, wind), gap, 1, 4, down);
    for (let c = 0; c < 3; c += 1) expect(down[c]).toBeCloseTo(up[c] ?? 0, 9);
  });
});

describe("calm is the measurement", () => {
  it("returns the identity by value for every node and a still flutter field", () => {
    for (const t of [0, 1234.5, 2.3e7]) {
      const frame = livingFrame(motion, t, windAt(0));
      frame.transforms.forEach((transform) => {
        expect(
          transform.rotation.every((v, i) => Object.is(v, IDENTITY_TRANSFORM.rotation[i])),
        ).toBe(true);
        expect(transform.translation.every((v) => Object.is(v, 0))).toBe(true);
      });
      expect(frame.flutter.still).toBe(true);
      expect(frame.flutter.amplitudeM.every((v) => Object.is(v, 0))).toBe(true);
    }
  });

  it("restores the canonical positions exactly after any amount of wind", () => {
    const positions = new Float32Array(3000);
    for (let i = 0; i < positions.length; i += 3) {
      const node = rig.nodes[(i / 3) % rig.nodes.length]?.position ?? [0, 0, 0];
      positions[i] = node[0] + 0.01;
      positions[i + 1] = node[1] - 0.02;
      positions[i + 2] = node[2] + 0.03;
    }
    const assignment = assignSplatsToNodes(positions, rig);
    const windy = livingFrame(motion, 77, windAt(12));
    const moved = deformPositions(
      positions,
      assignment,
      windy.transforms,
      undefined,
      windy.flutter,
    );
    expect(moved).not.toEqual(positions);
    const calm = livingFrame(motion, 78, windAt(0));
    const restored = deformPositions(
      positions,
      assignment,
      calm.transforms,
      undefined,
      calm.flutter,
    );
    expect(Array.from(restored)).toEqual(Array.from(positions));
  });
});

describe("same clock, same frame", () => {
  it("is bit-identical for the same t, across independently built models", () => {
    const twin = createLivingMotion(rig, parseMotionSidecar(JSON.stringify(sidecar), rig));
    const positions = new Float32Array(600);
    for (let i = 0; i < positions.length; i += 1) positions[i] = ((i * 0.731) % 1) * 5;
    const assignment = assignSplatsToNodes(positions, rig);
    for (const t of [3.25, 86_400.125, 2.3e7 + 0.5]) {
      const a = livingFrame(motion, t, windAt(7));
      const b = livingFrame(twin, t, windAt(7));
      const pa = deformPositions(positions, assignment, a.transforms, undefined, a.flutter);
      const pb = deformPositions(positions, assignment, b.transforms, undefined, b.flutter);
      expect(Array.from(pa)).toEqual(Array.from(pb));
      a.transforms.forEach((transform, i) => {
        const other = b.transforms[i];
        expect(transform.rotation.every((v, k) => Object.is(v, other?.rotation[k]))).toBe(true);
        expect(transform.translation.every((v, k) => Object.is(v, other?.translation[k]))).toBe(
          true,
        );
      });
    }
  });

  it("gives a different wind for a different seed", () => {
    const other = createLivingMotion(rig, { ...sidecar, seed: 2 });
    const tip = highestNode(rig);
    const a = displacementOf(livingTransforms(motion, 40, windAt(7)), tip);
    const b = displacementOf(livingTransforms(other, 40, windAt(7)), tip);
    expect(a).not.toEqual(b);
  });
});

describe("seasons", () => {
  it("drops the leaves in winter: no flutter, stiffer branches, the tree's winter damping", () => {
    const winter: LivingWind = { ...windAt(8), season: "winter" };
    expect(livingFlutter(motion, 12, winter).still).toBe(true);
    // Branches ring faster leafless (Habel §5.4), so a twig's local bend changes faster.
    const twig = BRANCH_BASES.find((b) => sidecar.nodes.mode[b] === 1) ?? 1;
    const parent = rig.nodes[twig]?.parent ?? -1;
    const roughness = (w: LivingWind): number => {
      let sum = 0;
      let last = 0;
      for (let k = 0; k < 2000; k += 1) {
        const r = localRotationVector(livingTransforms(motion, 900 + k / 50, w), twig, parent)[0];
        if (k > 0) sum += (r - last) ** 2;
        last = r;
      }
      return sum;
    };
    const summerRate = roughness(windAt(8, false));
    const winterRate = roughness({ ...windAt(8, false), season: "winter" });
    console.info(
      `MEASURED winter/summer mean-square step of a branch's bend: ${(winterRate / summerRate).toFixed(2)}`,
    );
    expect(winterRate).toBeGreaterThan(summerRate);
    expect(sidecar.seasons.winter.dampingScale * 0.086).toBeCloseTo(0.039, 3);
  });
});

describe("bounds", () => {
  it("never exceeds its proven displacement bound over ten minutes", () => {
    const wind = windAt(10);
    const bound = livingMaxDisplacement(motion, wind);
    const tip = highestNode(rig);
    let worst = 0;
    for (let k = 0; k < 6000; k += 1) {
      const d = displacementOf(livingTransforms(motion, 700 + k / 10, wind), tip);
      worst = Math.max(worst, Math.hypot(d[0], d[1], d[2]));
    }
    console.info(
      `MEASURED 10 m/s: worst tip excursion ${(worst * 100).toFixed(1)} cm over 10 min, bound ${(bound * 100).toFixed(1)} cm`,
    );
    expect(worst).toBeLessThan(bound);
  });
});

describe("cost per frame (CPU)", () => {
  it("measures ms per frame on the synthetic tree's splats", () => {
    const bytes = Uint8Array.from(
      readFileSync(
        fileURLToPath(
          new URL("../../../data/tiles/synthetic-tree/source/positions.f32", import.meta.url),
        ),
      ),
    );
    const positions = new Float32Array(bytes.buffer);
    const assignment = assignSplatsToNodes(positions, rig);
    const out = new Float32Array(positions.length);
    const wind = windAt(6.3);
    const legacy = { strength: 0.1, bearingDeg: 30 };
    const time = (label: string, frame: (t: number) => void): number => {
      for (let k = 0; k < 30; k += 1) frame(k / 60);
      const frames = 300;
      const start = process.hrtime.bigint();
      for (let k = 0; k < frames; k += 1) frame(100 + k / 60);
      const ms = Number(process.hrtime.bigint() - start) / 1e6 / frames;
      console.info(
        `MEASURED ${label}: ${ms.toFixed(3)} ms/frame (${positions.length / 3} splats, ${rig.nodes.length} nodes)`,
      );
      return ms;
    };
    const living = time("living mode, transforms + advected flutter + deformPositions", (t) => {
      const f = livingFrame(motion, t, wind);
      deformPositions(positions, assignment, f.transforms, out, f.flutter);
    });
    time("living mode, transforms only", (t) => {
      livingTransforms(motion, t, wind);
    });
    time("legacy nine-sine model, deform + flutterField + deformPositions", (t) => {
      deformPositions(
        positions,
        assignment,
        deform(rig, t, legacy),
        out,
        flutterField(rig, t, legacy),
      );
    });
    // A real capture's worth: the same tree tiled twelve times over (144,000 splats).
    const big = new Float32Array(positions.length * 12);
    for (let copy = 0; copy < 12; copy += 1) big.set(positions, copy * positions.length);
    const bigAssignment = new Uint16Array(assignment.length * 12);
    for (let copy = 0; copy < 12; copy += 1)
      bigAssignment.set(assignment, copy * assignment.length);
    const bigOut = new Float32Array(big.length);
    const start = process.hrtime.bigint();
    for (let k = 0; k < 40; k += 1) {
      const f = livingFrame(motion, 100 + k / 60, wind);
      deformPositions(big, bigAssignment, f.transforms, bigOut, f.flutter);
    }
    console.info(
      `MEASURED living mode at ${big.length / 3} splats: ${(Number(process.hrtime.bigint() - start) / 1e6 / 40).toFixed(2)} ms/frame`,
    );
    // A generous ceiling: this is a regression tripwire, not a frame budget.
    expect(living).toBeLessThan(40);
  });
});
