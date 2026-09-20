import { describe, expect, it } from "vitest";

import {
  DEG_TO_RAD,
  fbm1d,
  FBM_MAX_SLOPE,
  hash32,
  hashString,
  magnitude,
  maxWindMagnitude,
  noise1d,
  subtract,
  wind,
  WIND_CALM,
  WIND_MAX_SLOPE,
} from "./index";

/** Unit vectors for a bearing: downwind, and 90 degrees clockwise of it. */
function frame(bearingDeg: number): { down: [number, number]; across: [number, number] } {
  const b = bearingDeg * DEG_TO_RAD;
  return { down: [Math.sin(b), Math.cos(b)], across: [Math.cos(b), -Math.sin(b)] };
}

const BEARINGS = [0, 37, 90, 180, 217, 270, 359];

describe("noise", () => {
  it("is deterministic and seeded, never random", () => {
    for (let i = 0; i < 100; i += 1) {
      const x = i * 0.37;
      expect(noise1d(x, 7)).toBe(noise1d(x, 7));
      expect(fbm1d(x, 7)).toBe(fbm1d(x, 7));
    }
    expect(hash32(12, 3)).toBe(hash32(12, 3));
    expect(hashString("trunk-4")).toBe(hashString("trunk-4"));
    expect(hashString("trunk-4")).not.toBe(hashString("trunk-5"));
  });

  it("stays inside [-1, 1] without clipping", () => {
    for (let x = -200; x < 200; x += 0.013) {
      expect(Math.abs(noise1d(x, 3))).toBeLessThanOrEqual(1);
      expect(Math.abs(fbm1d(x, 3))).toBeLessThanOrEqual(1);
    }
  });

  it("is continuous across lattice boundaries", () => {
    const h = 1e-7;
    for (let i = -40; i <= 40; i += 1) {
      expect(Math.abs(fbm1d(i + h, 5) - fbm1d(i - h, 5)) / (2 * h)).toBeLessThan(FBM_MAX_SLOPE);
    }
  });

  it("respects its documented slope bound everywhere", () => {
    const h = 1e-6;
    let worst = 0;
    for (let x = -50; x < 50; x += 0.0007) {
      worst = Math.max(worst, Math.abs(fbm1d(x + h, 1) - fbm1d(x, 1)) / h);
    }
    expect(worst).toBeLessThan(FBM_MAX_SLOPE);
    expect(worst).toBeGreaterThan(0.5);
  });

  it("survives non-finite input rather than producing NaN", () => {
    expect(noise1d(Number.NaN)).toBe(0);
    expect(fbm1d(Number.POSITIVE_INFINITY)).toBe(0);
  });
});

describe("wind", () => {
  it("is bit-identical for the same inputs, across calls", () => {
    for (let k = 0; k < 500; k += 1) {
      const t = k * 0.017;
      expect(wind(0.42, 123, t)).toEqual(wind(0.42, 123, t));
    }
  });

  it("returns exactly zero at zero strength, for every bearing and time", () => {
    for (const bearingDeg of BEARINGS) {
      for (let k = 0; k < 200; k += 1) {
        expect(wind(0, bearingDeg, k * 0.31)).toEqual([0, 0, 0]);
      }
    }
    expect(wind(WIND_CALM.strength, WIND_CALM.bearingDeg, 12.5)).toEqual([0, 0, 0]);
  });

  it("never exceeds its closed-form magnitude bound", () => {
    for (const strength of [0.1, 0.5, 1]) {
      for (let k = 0; k <= 120 * 60; k += 7) {
        expect(magnitude(wind(strength, 42, k / 60))).toBeLessThanOrEqual(
          maxWindMagnitude(strength) + 1e-12,
        );
      }
    }
  });

  it("clamps strength into 0..1 and tolerates nonsense", () => {
    expect(wind(5, 0, 3)).toEqual(wind(1, 0, 3));
    expect(wind(-2, 0, 3)).toEqual([0, 0, 0]);
    expect(wind(Number.NaN, 0, 3)).toEqual([0, 0, 0]);
    expect(wind(1, Number.NaN, 3)).toEqual(wind(1, 0, 3));
    expect(wind(1, 0, Number.NaN)).toEqual(wind(1, 0, 0));
  });

  it("has no vertical component: this model has no updraft", () => {
    for (let k = 0; k < 100; k += 1) expect(wind(1, 77, k * 0.4)[2]).toBe(0);
  });

  it("blows towards the bearing on average, with zero-mean cross-wind", () => {
    for (const bearingDeg of BEARINGS) {
      const { down, across } = frame(bearingDeg);
      let along = 0;
      let cross = 0;
      const samples = 120 * 60;
      for (let k = 0; k < samples; k += 1) {
        const v = wind(1, bearingDeg, k / 60);
        along += v[0] * down[0] + v[1] * down[1];
        cross += v[0] * across[0] + v[1] * across[1];
      }
      expect(along / samples).toBeGreaterThan(0.5);
      expect(Math.abs(cross / samples)).toBeLessThan(0.01 * (along / samples));
    }
  });

  it("is always downwind-positive, never a reversal", () => {
    const { down } = frame(215);
    for (let k = 0; k <= 120 * 60; k += 3) {
      const v = wind(1, 215, k / 60);
      expect(v[0] * down[0] + v[1] * down[1]).toBeGreaterThan(0);
    }
  });

  it("changes no faster than its documented slope bound", () => {
    const h = 1e-6;
    let worst = 0;
    for (let t = 0; t < 200; t += 0.0013) {
      worst = Math.max(worst, magnitude(subtract(wind(1, 33, t + h), wind(1, 33, t))) / h);
    }
    expect(worst).toBeLessThan(WIND_MAX_SLOPE);
    expect(worst).toBeGreaterThan(0.01);
  });

  it("scales linearly with strength at any instant", () => {
    for (const t of [0, 3.7, 41.25, 119.99]) {
      const full = wind(1, 64, t);
      const half = wind(0.5, 64, t);
      expect(half[0]).toBeCloseTo(full[0] / 2, 12);
      expect(half[1]).toBeCloseTo(full[1] / 2, 12);
    }
  });
});
