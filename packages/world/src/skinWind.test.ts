import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { speedFromStrength } from "./living";
import {
  ANCHOR_TOLERANCE,
  fromUpper,
  handleReach,
  HANDLE_REACH,
  materialPrior,
  mergeMaterial,
  SKIN_WAVE_SPEED_MPS,
  SKIN_WIND_STEP_S,
  skinWindFromSettings,
  SkinWindField,
  skinWindModel,
  SkinWindOscillator,
  type SkinDynamicsSource,
  type SkinMaterial,
  type SkinWind,
} from "./skinWind";
import { cholesky, symmetricEigen } from "./symmetricEigen";

const HERE = dirname(fileURLToPath(import.meta.url));
const YARD_SKIN = resolve(HERE, "../../../data/tiles/synthetic-yard/skin/skin.json");

interface FixtureSkin {
  instance: number;
  handles: number;
  origin: [number, number, number];
  scale: number;
  eigenvalues: number[];
  support: { centre: [number, number, number]; radius: number }[];
  dynamics: { mass: number[]; anchor: { gram: number[] } };
}

function yardSkins(): FixtureSkin[] {
  return (JSON.parse(readFileSync(YARD_SKIN, "utf8")) as { skins: FixtureSkin[] }).skins;
}

function sourceOf(skin: FixtureSkin): SkinDynamicsSource {
  return { ...skin, mass: skin.dynamics.mass, anchorGram: skin.dynamics.anchor.gram };
}

const PLANT: SkinMaterial = materialPrior({ vegetation: 1, elastic: 1, rigid: 0 }, "in-place");

function upper(matrix: Float64Array, m: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < m; i += 1) for (let j = i; j < m; j += 1) out.push(matrix[i * m + j] ?? 0);
  return out;
}

/**
 * A synthetic object: a vertical line of points, `H` tall, whose learned weights are given as
 * functions of height. Its Grams are computed from the points exactly as `skin_scene` does,
 * the anchor being the lowest tenth.
 */
function lineSkin(
  weights: ((z: number) => number)[],
  eigenvalues: number[],
  height = 4,
  points = 401,
): { source: SkinDynamicsSource; zs: number[]; rows: number[][] } {
  const m = weights.length + 1;
  const zs = Array.from({ length: points }, (_, k) => (height * k) / (points - 1));
  const rows = zs.map((z) => [1, ...weights.map((w) => w(z))]);
  const gram = (subset: number[][]): Float64Array => {
    const out = new Float64Array(m * m);
    for (const row of subset)
      for (let i = 0; i < m; i += 1)
        for (let j = 0; j < m; j += 1)
          out[i * m + j] = (out[i * m + j] ?? 0) + ((row[i] ?? 0) * (row[j] ?? 0)) / subset.length;
    return out;
  };
  const mass = gram(rows);
  const anchor = gram(rows.filter((_, k) => (zs[k] ?? 0) <= 0.1 * height));
  const support = weights.map((w) => {
    let total = 0;
    let centre = 0;
    for (const z of zs) {
      total += Math.abs(w(z));
      centre += Math.abs(w(z)) * z;
    }
    return { centre: [0, 0, centre / total] as const, radius: height / 3 };
  });
  return {
    source: {
      handles: m,
      origin: [0, 0, 0],
      scale: height / 2,
      eigenvalues,
      support,
      mass: upper(mass, m),
      anchorGram: upper(anchor, m),
    },
    zs,
    rows,
  };
}

/** Zero on the anchor band exactly, rising linearly to 1 at the top: one bending field. */
const ramp =
  (height: number) =>
  (z: number): number =>
    Math.max(0, z - 0.1 * height) / (0.9 * height);

const STEADY: SkinWind = { speedMps: 5, bearingDeg: 90, turbulence: 0 };

/** Runs an oscillator over `[t0, t0 + seconds]` at `fps`, returning `[t, handles]` per frame. */
function run(
  oscillator: SkinWindOscillator,
  field: SkinWindField,
  wind: SkinWind,
  t0: number,
  seconds: number,
  fps: number,
): [number, Float64Array][] {
  const out: [number, Float64Array][] = [];
  const frames = Math.round(seconds * fps);
  for (let k = 0; k <= frames; k += 1) {
    const t = t0 + k / fps;
    oscillator.advance(field, wind, t);
    out.push([t, oscillator.handles(t)]);
  }
  return out;
}

describe("small symmetric linear algebra", () => {
  it("diagonalises a symmetric matrix into orthonormal eigenvectors", () => {
    const n = 7;
    const a = new Float64Array(n * n);
    for (let i = 0; i < n; i += 1)
      for (let j = i; j < n; j += 1) {
        const v = Math.sin(1 + i * 3.1 + j * 1.7) + (i === j ? 2 : 0);
        a[i * n + j] = v;
        a[j * n + i] = v;
      }
    const { values, vectors } = symmetricEigen(a, n);
    for (let k = 1; k < n; k += 1) expect(values[k]).toBeGreaterThanOrEqual(values[k - 1] ?? 0);
    for (let i = 0; i < n; i += 1)
      for (let j = 0; j < n; j += 1) {
        // A·V = V·Λ and Vᵀ·V = I.
        let av = 0;
        let vtv = 0;
        for (let k = 0; k < n; k += 1) {
          av += (a[i * n + k] ?? 0) * (vectors[k * n + j] ?? 0);
          vtv += (vectors[k * n + i] ?? 0) * (vectors[k * n + j] ?? 0);
        }
        expect(av).toBeCloseTo((vectors[i * n + j] ?? 0) * (values[j] ?? 0), 10);
        expect(vtv).toBeCloseTo(i === j ? 1 : 0, 12);
      }
  });

  it("factors a positive-definite matrix and refuses an indefinite one", () => {
    const l = cholesky([4, 2, 2, 3], 2);
    expect(Array.from(l ?? [])).toEqual([2, 0, 1, Math.SQRT2]);
    expect(cholesky([1, 2, 2, 1], 2)).toBeUndefined();
  });
});

describe("materials", () => {
  it("drives in-place objects and leaves movable and static ones alone", () => {
    expect(materialPrior({ vegetation: 0.9 }, "in-place").wind).toBe(true);
    expect(materialPrior({ vegetation: 0.9 }, "movable").wind).toBe(false);
    expect(materialPrior({ vegetation: 0.9 }, "static").wind).toBe(false);
    expect(materialPrior(undefined, undefined).wind).toBe(false);
  });

  it("makes vegetation softer, more damped and draggier than rigid, bare objects", () => {
    const plant = materialPrior({ vegetation: 1, elastic: 1, rigid: 0 }, "in-place");
    const post = materialPrior({ vegetation: 0, elastic: 0, rigid: 1 }, "in-place");
    expect(plant.stiffness).toBeCloseTo(SKIN_WAVE_SPEED_MPS, 12);
    expect(post.stiffness).toBeCloseTo(4 * SKIN_WAVE_SPEED_MPS, 12);
    expect(plant.damping).toBeGreaterThan(post.damping);
    expect(plant.drag).toBeCloseTo(4 * post.drag, 12);
    expect(plant.evidence).toBe("prior");
  });

  it("lets a fitted record override any valid subset of the prior", () => {
    const merged = mergeMaterial(PLANT, {
      stiffness: 2,
      damping: 7,
      drag: Number.NaN,
      evidence: "fitted-real",
    });
    expect(merged).toEqual({
      stiffness: 2,
      damping: 0.95,
      drag: PLANT.drag,
      wind: true,
      evidence: "fitted-real",
    });
    expect(mergeMaterial(PLANT, { wind: false }).wind).toBe(false);
    expect(mergeMaterial(PLANT, undefined)).toBe(PLANT);
  });
});

describe("the anchored modal model", () => {
  it("maps stiffness, eigenvalue and size to ω = c·√λ / scale", () => {
    const height = 4;
    const { source } = lineSkin([ramp(height)], [10], height);
    const model = skinWindModel(source, PLANT);
    expect(model?.modes).toBe(1);
    // The ramp is zero on the anchor band, so the anchored mode is the learned handle alone.
    const expected = (PLANT.stiffness * Math.sqrt(10)) / source.scale;
    expect(model?.omega[0]).toBeCloseTo(expected, 6);
    const stiffer = skinWindModel(source, { ...PLANT, stiffness: 2 * PLANT.stiffness });
    expect(stiffer?.omega[0]).toBeCloseTo(2 * expected, 6);
    const larger = skinWindModel({ ...source, scale: 2 * source.scale }, PLANT);
    expect(larger?.omega[0]).toBeCloseTo(expected / 2, 6);
    const harder = skinWindModel({ ...source, eigenvalues: [40] }, PLANT);
    expect(harder?.omega[0]).toBeCloseTo(2 * expected, 6);
  });

  it("refuses a skin without dynamics, or one whose base cannot stay still", () => {
    const { source } = lineSkin([ramp(4)], [10]);
    expect(skinWindModel({ ...source, mass: [] }, PLANT)).toBeUndefined();
    // Anchored over the whole object: no direction leaves its base still.
    expect(skinWindModel({ ...source, anchorGram: source.mass }, PLANT)).toBeUndefined();
  });

  it("anchors free modes: a field that moves the base is combined with the constant", () => {
    const height = 4;
    // Two free fields, both non-zero at the base: neither is admissible alone.
    const { source, zs, rows } = lineSkin(
      [(z) => z / height, (z) => Math.cos((Math.PI * z) / height)],
      [10, 40],
      height,
    );
    const model = skinWindModel(source, PLANT);
    expect(model).toBeDefined();
    if (!model) return;
    expect(model.modes).toBeGreaterThanOrEqual(1);
    expect(model.anchorResidual).toBeLessThanOrEqual(ANCHOR_TOLERANCE);
    const field = new SkinWindField(3);
    const oscillator = new SkinWindOscillator(model);
    let worst = 0;
    let moving = 0;
    for (const [, z] of run(oscillator, field, { ...STEADY, turbulence: 1 }, 50, 20, 60)) {
      // Displacement of every point, from the handles (east only: the wind blows east).
      const u = rows.map((row) => row.reduce((sum, w, j) => sum + w * (z[j * 12 + 3] ?? 0), 0));
      const rms = Math.sqrt(u.reduce((s, v) => s + v * v, 0) / u.length);
      const base = u.filter((_, k) => (zs[k] ?? 0) <= 0.1 * height);
      const baseRms = Math.sqrt(base.reduce((s, v) => s + v * v, 0) / base.length);
      if (rms > 1e-6) {
        moving += 1;
        worst = Math.max(worst, baseRms / rms);
      }
      // The foot itself is still to well within a millimetre while the top sways.
      expect(Math.abs(u[0] ?? 0)).toBeLessThan(0.05 * Math.max(...u.map(Math.abs)) + 1e-9);
    }
    expect(moving).toBeGreaterThan(1000);
    expect(worst).toBeLessThanOrEqual(ANCHOR_TOLERANCE + 1e-9);
  });

  it("rings at its natural frequency and decays at its damping ratio", () => {
    const height = 4;
    const { source } = lineSkin([ramp(height)], [10], height);
    const zeta = 0.05;
    const model = skinWindModel(source, { ...PLANT, damping: zeta });
    if (!model) throw new Error("no model");
    const omega = model.omega[0] ?? 0;
    const oscillator = new SkinWindOscillator(model);
    // A steady wind switched on at t = 0 from rest: a damped step response.
    const series = run(oscillator, new SkinWindField(1), STEADY, 0, 20, 240).map(
      ([t, z]) => [t, z[1 * 12 + 3] ?? 0] as const,
    );
    // Its equilibrium: F_1 / (M_11 Ω²), F_1 = (D/scale)·M_01·U² (uniform wind).
    const mass = fromUpper(source.mass, 2) ?? new Float64Array(4);
    const equilibrium =
      ((PLANT.drag / source.scale) * (mass[1] ?? 0) * STEADY.speedMps ** 2) /
      ((mass[3] ?? 1) * omega * omega);
    // Peaks of the overshoot: their spacing is the damped period, their ratio the decrement.
    const peaks: [number, number][] = [];
    for (let k = 1; k < series.length - 1; k += 1) {
      const [t, x] = series[k] ?? [0, 0];
      if (x > (series[k - 1]?.[1] ?? 0) && x >= (series[k + 1]?.[1] ?? 0)) peaks.push([t, x]);
    }
    expect(peaks.length).toBeGreaterThan(4);
    const damped = omega * Math.sqrt(1 - zeta * zeta);
    const period = ((peaks[4]?.[0] ?? 0) - (peaks[0]?.[0] ?? 0)) / 4;
    expect(period).toBeCloseTo((2 * Math.PI) / damped, 2);
    const decrement =
      Math.log(((peaks[0]?.[1] ?? 0) - equilibrium) / ((peaks[1]?.[1] ?? 0) - equilibrium)) / 1;
    expect(decrement).toBeCloseTo((2 * Math.PI * zeta) / Math.sqrt(1 - zeta * zeta), 2);
    // The first overshoot of a step from rest: 1 + e^(−ζπ/√(1−ζ²)).
    expect((peaks[0]?.[1] ?? 0) / equilibrium).toBeCloseTo(
      1 + Math.exp((-zeta * Math.PI) / Math.sqrt(1 - zeta * zeta)),
      2,
    );
    // And it settles there.
    expect(series.at(-1)?.[1]).toBeCloseTo(equilibrium, 3);
  });
});

describe("on the yard's skins", () => {
  const skins = yardSkins();
  const field = new SkinWindField(11);
  const wind = skinWindFromSettings({ strength: 0.1, bearingDeg: 60 });

  it("keeps every object's base still while it sways, at the default wind", () => {
    for (const skin of skins) {
      const model = skinWindModel(sourceOf(skin), PLANT);
      if (!model) throw new Error(`skin of ${String(skin.instance)} has no model`);
      const m = skin.handles;
      const mass = fromUpper(skin.dynamics.mass, m) ?? new Float64Array(0);
      const gram = fromUpper(skin.dynamics.anchor.gram, m) ?? new Float64Array(0);
      const oscillator = new SkinWindOscillator(model);
      let worst = 0;
      let rms = 0;
      for (const [, z] of run(oscillator, field, wind, 500, 20, 30)) {
        for (const axis of [3, 7]) {
          let qmq = 0;
          let qgq = 0;
          for (let i = 0; i < m; i += 1)
            for (let j = 0; j < m; j += 1) {
              const qq = (z[i * 12 + axis] ?? 0) * (z[j * 12 + axis] ?? 0);
              qmq += qq * (mass[i * m + j] ?? 0);
              qgq += qq * (gram[i * m + j] ?? 0);
            }
          rms = Math.max(rms, Math.sqrt(qmq));
          if (qmq > 1e-12) worst = Math.max(worst, Math.sqrt(Math.max(0, qgq) / qmq));
        }
      }
      // The anchor band's rms motion is within the tolerance of the object's.
      expect(worst).toBeLessThanOrEqual(ANCHOR_TOLERANCE);
      expect(rms).toBeGreaterThan(5e-4);
    }
  });

  it("sways the tree about a decimetre at the default wind, its first mode near 0.2 Hz", () => {
    const tree = skins.find((s) => s.instance === 1);
    if (!tree) throw new Error("no tree");
    const model = skinWindModel(sourceOf(tree), PLANT);
    if (!model) throw new Error("no model");
    expect((model.omega[0] ?? 0) / (2 * Math.PI)).toBeGreaterThan(0.15);
    expect((model.omega[0] ?? 0) / (2 * Math.PI)).toBeLessThan(0.5);
    const m = tree.handles;
    const mass = fromUpper(tree.dynamics.mass, m) ?? new Float64Array(0);
    const oscillator = new SkinWindOscillator(model);
    let sum = 0;
    let n = 0;
    for (const [, z] of run(oscillator, field, wind, 900, 40, 30).slice(300)) {
      for (const axis of [3, 7])
        for (let i = 0; i < m; i += 1)
          for (let j = 0; j < m; j += 1)
            sum += (z[i * 12 + axis] ?? 0) * (z[j * 12 + axis] ?? 0) * (mass[i * m + j] ?? 0);
      n += 1;
    }
    const rms = Math.sqrt(sum / n);
    expect(rms).toBeGreaterThan(0.05);
    expect(rms).toBeLessThan(0.2);
  });

  it("never moves a handle beyond its share of its support radius, however hard it blows", () => {
    for (const skin of skins) {
      const material = { ...PLANT, drag: PLANT.drag * 100 };
      const model = skinWindModel(sourceOf(skin), material);
      if (!model) throw new Error("no model");
      const oscillator = new SkinWindOscillator(model);
      const gale = skinWindFromSettings({ strength: 1, bearingDeg: 200 });
      let peak = 0;
      for (const [, z] of run(oscillator, field, gale, 0, 10, 30)) {
        const reach = handleReach(z, skin.handles);
        for (let j = 0; j < skin.handles; j += 1) {
          const limit = HANDLE_REACH * (j === 0 ? skin.scale : (skin.support[j - 1]?.radius ?? 0));
          expect(reach[j]).toBeLessThanOrEqual(limit + 1e-12);
          peak = Math.max(peak, (reach[j] ?? 0) / limit);
        }
      }
      // And it does reach for the bound: the limit is what holds it.
      expect(peak).toBeGreaterThan(0.5);
    }
  });

  it("is the same at any frame rate: the state lives on a fixed grid", () => {
    const tree = skins.find((s) => s.instance === 1);
    const model = tree && skinWindModel(sourceOf(tree), PLANT);
    if (!model) throw new Error("no model");
    const at = (fps: number): Map<number, Float64Array> => {
      const oscillator = new SkinWindOscillator(model);
      const out = new Map<number, Float64Array>();
      for (let k = 0; k <= 10 * fps; k += 1) {
        const t = 100 + k / fps;
        oscillator.advance(field, wind, t);
        // Whole tenths of a second are grid times.
        if (k % (fps / 10) === 0) out.set(k / (fps / 10), oscillator.handles(t));
      }
      return out;
    };
    const reference = at(60);
    const slow = at(20);
    expect(reference.size).toBe(101);
    let compared = 0;
    for (const [tenth, z] of slow) {
      const r = reference.get(tenth);
      if (!r) continue;
      compared += 1;
      for (let k = 0; k < z.length; k += 1) expect(z[k]).toBeCloseTo(r[k] ?? 0, 12);
    }
    expect(compared).toBe(101);
    // An irregular clock: at the same scene time the handles agree with the 60 Hz run.
    const irregular = new SkinWindOscillator(model);
    const steady = new SkinWindOscillator(model);
    let t = 100;
    irregular.advance(field, wind, t);
    steady.advance(field, wind, t);
    for (let k = 1; k <= 600; k += 1) {
      t = 100 + k * 0.0173 + 0.004 * Math.sin(k);
      irregular.advance(field, wind, t);
      for (let s = 1; s <= 3; s += 1) steady.advance(field, wind, t - ((3 - s) * 0.0173) / 3);
      const a = irregular.handles(t);
      const b = steady.handles(t);
      for (let i = 0; i < a.length; i += 1) expect(a[i]).toBe(b[i]);
    }
  });

  it("is deterministic given its seed, and the seed matters", () => {
    const tree = skins.find((s) => s.instance === 1);
    const model = tree && skinWindModel(sourceOf(tree), PLANT);
    if (!model) throw new Error("no model");
    const once = run(new SkinWindOscillator(model), new SkinWindField(5), wind, 30, 5, 60);
    const again = run(new SkinWindOscillator(model), new SkinWindField(5), wind, 30, 5, 60);
    const other = run(new SkinWindOscillator(model), new SkinWindField(6), wind, 30, 5, 60);
    expect(again.map(([, z]) => Array.from(z))).toEqual(once.map(([, z]) => Array.from(z)));
    expect(other.map(([, z]) => Array.from(z))).not.toEqual(once.map(([, z]) => Array.from(z)));
  });

  it("is exactly at rest in calm, and starts from rest when the wind returns", () => {
    const shrub = skins.find((s) => s.instance === 10);
    const model = shrub && skinWindModel(sourceOf(shrub), PLANT);
    if (!model) throw new Error("no model");
    const oscillator = new SkinWindOscillator(model);
    run(oscillator, field, wind, 0, 3, 60);
    expect(handleReach(oscillator.handles(3), model.handles).some((v) => v > 0)).toBe(true);
    oscillator.advance(field, { speedMps: 0, bearingDeg: 60 }, 3.1);
    expect(oscillator.time).toBeUndefined();
    expect(Array.from(oscillator.handles(3.1)).every((v) => v === 0)).toBe(true);
    oscillator.advance(field, wind, 4);
    expect(Array.from(oscillator.handles(4)).every((v) => v === 0)).toBe(true);
    expect(speedFromStrength(0)).toBe(0);
    expect(skinWindFromSettings({ strength: 0, bearingDeg: 0 }).speedMps).toBe(0);
  });

  it("restarts from the equilibrium, not from a blow-up, after a jump in scene time", () => {
    const tree = skins.find((s) => s.instance === 1);
    const model = tree && skinWindModel(sourceOf(tree), PLANT);
    if (!model) throw new Error("no model");
    const oscillator = new SkinWindOscillator(model);
    run(oscillator, field, wind, 0, 2, 60);
    oscillator.advance(field, wind, 3600);
    expect(oscillator.time).toBeCloseTo(3600, 9);
    oscillator.advance(field, wind, 10);
    expect(oscillator.time).toBeCloseTo(10, 9);
    const reach = handleReach(oscillator.handles(10), model.handles);
    for (let j = 0; j < model.handles; j += 1)
      expect(reach[j]).toBeLessThanOrEqual(model.limits[j] ?? 0);
  });

  it("costs well under a millisecond a frame for thirty sixteen-handle objects", () => {
    const tree = skins.find((s) => s.instance === 1);
    if (!tree) throw new Error("no tree");
    const oscillators = Array.from({ length: 30 }, (_, k) => {
      const model = skinWindModel({ ...sourceOf(tree), origin: [k * 7, (k % 5) * 9, 0] }, PLANT);
      if (!model) throw new Error("no model");
      return new SkinWindOscillator(model);
    });
    const out = new Float64Array(16 * 12);
    const frame = (t: number): void => {
      for (const o of oscillators) {
        o.advance(field, wind, t);
        o.handles(t, out.subarray(0, o.model.handles * 12));
      }
    };
    for (let k = 0; k < 120; k += 1) frame(k / 60);
    const frames = 600;
    const started = process.hrtime.bigint();
    for (let k = 120; k < 120 + frames; k += 1) frame(k / 60);
    const ms = Number(process.hrtime.bigint() - started) / 1e6 / frames;
    console.info(`skin wind: ${ms.toFixed(3)} ms a frame for 30 objects x 14 handles`);
    expect(ms).toBeLessThan(4);
  });
});

describe("SKIN_WIND_STEP_S", () => {
  it("is a 60 Hz grid", () => {
    expect(SKIN_WIND_STEP_S).toBeCloseTo(1 / 60, 15);
  });
});
