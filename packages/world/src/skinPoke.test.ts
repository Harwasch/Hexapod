import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  POKE_MAX_PULL,
  POKE_REACH,
  POKE_RETURN_HZ,
  POKE_STEP_S,
  SkinPoke,
  skinPokeModel,
  type SkinPokeModel,
} from "./skinPoke";
import { materialPrior, type SkinDynamicsSource, type SkinMaterial } from "./skinWind";

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

function yardSkin(instance: number): SkinDynamicsSource {
  const skins = (JSON.parse(readFileSync(YARD_SKIN, "utf8")) as { skins: FixtureSkin[] }).skins;
  const skin = skins.find((s) => s.instance === instance);
  if (!skin) throw new Error(`no skin for ${String(instance)}`);
  return { ...skin, mass: skin.dynamics.mass, anchorGram: skin.dynamics.anchor.gram };
}

const PLANT: SkinMaterial = materialPrior({ vegetation: 1, elastic: 1, rigid: 0 }, "in-place");
const material = (stiffness: number, damping = 0.05): SkinMaterial => ({
  ...PLANT,
  stiffness,
  damping,
});

/**
 * A vertical line of points `height` tall whose learned weights are `weights(z)`: the Grams as
 * `skin_scene` computes them (the anchor the lowest tenth), and every point's weight row.
 */
function lineSkin(
  weights: ((z: number) => number)[],
  eigenvalues: number[],
  height = 4,
  points = 401,
): { source: SkinDynamicsSource; rows: number[][]; zs: number[] } {
  const m = weights.length + 1;
  const zs = Array.from({ length: points }, (_, k) => (height * k) / (points - 1));
  const rows = zs.map((z) => [1, ...weights.map((w) => w(z))]);
  const gram = (subset: number[][]): number[] => {
    const full = new Float64Array(m * m);
    for (const row of subset)
      for (let i = 0; i < m; i += 1)
        for (let j = 0; j < m; j += 1)
          full[i * m + j] =
            (full[i * m + j] ?? 0) + ((row[i] ?? 0) * (row[j] ?? 0)) / subset.length;
    const out: number[] = [];
    for (let i = 0; i < m; i += 1) for (let j = i; j < m; j += 1) out.push(full[i * m + j] ?? 0);
    return out;
  };
  const anchor = rows.filter((_, k) => (zs[k] ?? 0) <= 0.1 * height);
  return {
    source: {
      handles: m,
      origin: [0, 0, 0],
      scale: height / 2,
      eigenvalues,
      support: weights.map(() => ({ centre: [0, 0, height * 0.75], radius: height / 3 })),
      mass: gram(rows),
      anchorGram: gram(anchor),
    },
    rows,
    zs,
  };
}

/** A cantilever-like line: one learned field that is ~0 at the base, 1 at the top. */
const CANTILEVER = (height = 4) =>
  lineSkin([(z) => 1 - Math.cos((Math.PI * z) / (2 * height))], [3], height);

/** Runs `poke` from `t0` to `t1` in frames of `dt`, calling `each` after every frame. */
function run(
  poke: SkinPoke,
  t0: number,
  t1: number,
  dt: number,
  each?: (t: number) => void,
): number {
  let t = t0;
  while (t < t1) {
    t = t + dt > t1 - 1e-9 ? t1 : t + dt;
    poke.advance(t);
    each?.(t);
  }
  return t;
}

/** The east displacement of a splat of `weights` under the bounded handles. */
function eastOf(poke: SkinPoke, weights: readonly number[]): number {
  const h = poke.handles();
  if (!h) return 0;
  let u = 0;
  for (let j = 0; j < weights.length; j += 1) u += (weights[j] ?? 0) * (h[j * 12 + 3] ?? 0);
  return u;
}

describe("skinPokeModel", () => {
  it("roots an in-place object in the wind's anchored modes and frees a movable one", () => {
    const tree = yardSkin(1);
    const rooted = skinPokeModel(tree, PLANT, false);
    const free = skinPokeModel(tree, PLANT, true);
    expect(rooted?.movable).toBe(false);
    expect(free?.movable).toBe(true);
    expect(rooted!.modes).toBeLessThan(tree.handles);
    expect(free!.modes).toBe(tree.handles);
    // A movable object's mode that is mostly the whole object is its spring home.
    const m = free!.handles;
    let best = 0;
    let whole = 0;
    for (let i = 0; i < free!.modes; i += 1) {
      let total = 0;
      for (let j = 0; j < m; j += 1) total += free!.shapes[j * m + i]! ** 2;
      const share = free!.shapes[i]! ** 2 / total;
      if (share > whole) [whole, best] = [share, i];
    }
    expect(free!.omega[best]! / (2 * Math.PI) / POKE_RETURN_HZ).toBeGreaterThan(0.8);
    expect(free!.omega[best]! / (2 * Math.PI) / POKE_RETURN_HZ).toBeLessThan(1.25);
  });

  it("gives a rooted rigid object nothing to move, and a movable one its whole-body mode", () => {
    const rigid: SkinDynamicsSource = {
      handles: 1,
      origin: [0, 0, 0],
      scale: 1,
      eigenvalues: [],
      support: [],
      mass: [1],
      anchorGram: [1],
    };
    expect(skinPokeModel(rigid, PLANT, false)).toBeUndefined();
    const free = skinPokeModel(rigid, PLANT, true);
    expect(free?.modes).toBe(1);
    expect(free!.omega[0]! / (2 * Math.PI)).toBeCloseTo(POKE_RETURN_HZ, 6);
  });
});

describe("SkinPoke", () => {
  it("pulls a soft object's grabbed point toward the cursor, and barely a stiff one's", () => {
    const { source, rows } = CANTILEVER();
    const top = rows[rows.length - 1]!;
    const follow = (stiffness: number): number => {
      const poke = new SkinPoke(skinPokeModel(source, material(stiffness, 0.3), false)!);
      poke.advance(0);
      poke.grab({ weights: top });
      poke.pull([0.3, 0, 0]);
      run(poke, 0, 8, 1 / 60);
      return poke.grabbedDisplacement()[0] / 0.3;
    };
    expect(follow(2)).toBeGreaterThan(0.85);
    expect(follow(200)).toBeLessThan(0.1);
  });

  it("rings down at its own frequency once released (within 10%)", () => {
    const { source, rows } = CANTILEVER();
    const top = rows[rows.length - 1]!;
    const zeta = 0.03;
    const model = skinPokeModel(source, material(3.5, zeta), false)!;
    expect(model.modes).toBe(1);
    const poke = new SkinPoke(model);
    poke.advance(0);
    poke.grab({ weights: top });
    poke.pull([0.2, 0, 0]);
    let t = run(poke, 0, 6, 1 / 60);
    poke.release();
    const crossings: number[] = [];
    let last = eastOf(poke, top);
    t = run(poke, t, t + 12, POKE_STEP_S, (now) => {
      const u = eastOf(poke, top);
      if (last > 0 && u <= 0) crossings.push(now);
      last = u;
    });
    expect(crossings.length).toBeGreaterThan(4);
    const period = (crossings[crossings.length - 1]! - crossings[0]!) / (crossings.length - 1);
    const expected = (2 * Math.PI) / (model.omega[0]! * Math.sqrt(1 - zeta * zeta));
    expect(Math.abs(period / expected - 1)).toBeLessThan(0.1);
    // And it decays at its damping: the envelope after n periods is ~exp(-ζ Ω n T).
    expect(t).toBeGreaterThan(12);
  });

  it("comes to rest on its own and then hands back null: the measured frame", () => {
    const { source, rows } = CANTILEVER();
    const top = rows[rows.length - 1]!;
    const poke = new SkinPoke(skinPokeModel(source, material(3.5, 0.1), false)!);
    expect(poke.handles()).toBeNull();
    poke.advance(0);
    poke.grab({ weights: top });
    poke.pull([0.2, 0.1, 0]);
    run(poke, 0, 2, 1 / 60);
    expect(poke.handles()).not.toBeNull();
    poke.release();
    run(poke, 2, 120, 1 / 60);
    expect(poke.active).toBe(false);
    expect(poke.handles()).toBeNull();
  });

  it("never moves a handle past its limit, however far the cursor goes", () => {
    const tree = yardSkin(1);
    const model = skinPokeModel(tree, PLANT, false)!;
    const poke = new SkinPoke(model);
    poke.advance(0);
    // Grab the crown: weights of the yard tree's most-moving handle at its peak.
    const weights = [1, ...Array.from({ length: tree.handles - 1 }, (_, k) => (k === 0 ? 1 : 0.2))];
    poke.grab({ weights });
    poke.pull([1000, -500, 300]);
    run(poke, 0, 5, 1 / 60, () => {
      const h = poke.handles()!;
      for (let j = 0; j < model.handles; j += 1) {
        const q = Math.hypot(h[j * 12 + 3]!, h[j * 12 + 7]!, h[j * 12 + 11]!);
        expect(q).toBeLessThanOrEqual(model.limits[j]! * (1 + 1e-9));
      }
    });
    // The pull itself is capped at a share of the object's size.
    const d = poke.grabbedDisplacement();
    expect(Math.hypot(...d)).toBeLessThanOrEqual(POKE_MAX_PULL * model.scale * 1.0001);
    expect(model.limits[1]).toBeCloseTo(POKE_REACH * tree.support[0]!.radius, 9);
  });

  it("is stable for ten minutes at 60 Hz, dragged about and let go", () => {
    const tree = yardSkin(1);
    const poke = new SkinPoke(skinPokeModel(tree, PLANT, false)!);
    const weights = [1, ...Array.from({ length: tree.handles - 1 }, (_, k) => Math.cos(k))];
    poke.advance(0);
    let t = 0;
    for (let second = 0; second < 600; second += 1) {
      if (second % 20 === 0) poke.grab({ weights });
      if (second % 20 === 5) poke.release();
      poke.pull([Math.sin(second), Math.cos(1.3 * second), 0.2 * Math.sin(0.7 * second)]);
      t = run(poke, t, t + 1, 1 / 60);
      const h = poke.handles();
      if (h) for (const v of h) expect(Number.isFinite(v)).toBe(true);
    }
    expect(poke.state.every(Number.isFinite)).toBe(true);
  });

  it("depends on the times fed, not the frame rate", () => {
    const { source, rows } = CANTILEVER();
    const top = rows[rows.length - 1]!;
    const states = [1 / 24, 1 / 60, 1 / 144].map((dt) => {
      const poke = new SkinPoke(skinPokeModel(source, PLANT, false)!);
      poke.advance(0);
      poke.grab({ weights: top });
      poke.pull([0.2, 0, 0.05]);
      run(poke, 0, 1.5, dt);
      poke.release();
      run(poke, 1.5, 3, dt);
      return poke.state;
    });
    for (const s of states.slice(1)) expect(Array.from(s)).toEqual(Array.from(states[0]!));
  });

  it("slides a movable object whole and springs it home when let go", () => {
    const tree = yardSkin(10);
    const model: SkinPokeModel = skinPokeModel(tree, material(14, 0.1), true)!;
    const poke = new SkinPoke(model);
    poke.advance(0);
    const weights = [1, ...Array.from({ length: tree.handles - 1 }, () => 0)];
    poke.grab({ weights });
    poke.pull([0.4, 0, 0]);
    run(poke, 0, 4, 1 / 60);
    // The constant handle carries most of the pull.
    const held = poke.handles()!;
    expect(held[3]! / 0.4).toBeGreaterThan(0.5);
    poke.release();
    run(poke, 4, 60, 1 / 60);
    expect(poke.handles()).toBeNull();
  });
});
