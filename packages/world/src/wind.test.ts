import { describe, expect, it } from "vitest";

import {
  DEG_TO_RAD,
  fbm1d,
  FBM_MAX_SLOPE,
  gustDelaySeconds,
  hash32,
  hashString,
  magnitude,
  maxWindMagnitude,
  noise1d,
  subtract,
  wind,
  WIND_CALM,
  WIND_MAX_SLOPE,
  windAt,
  type Vec3,
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

describe("the gust is a field, not a vector", () => {
  /**
   * The property the whole term exists for: a gust **arrives**.
   *
   * One wind vector for the whole tree says a gust hits every limb at the same instant, which
   * is the one thing about wind everybody has watched not happen. The field here is the same
   * gust sampled at a retarded time — `t − (x·downwind)/speed`, tilted by a cross-wind term —
   * so the upwind side of a crown leads the downwind side by the travel time between them.
   *
   * It is measured as a lag, not asserted as a constant: the downwind sample's best match
   * against the upwind one is found by sliding one against the other, and the lag that wins
   * must be positive and near the travel time.
   */
  const DT = 1 / 60;

  /** The along-wind component of the gust at a point, sampled at 60 Hz. */
  function series(position: Vec3, frames: number, bearingDeg: number): number[] {
    const bearing = bearingDeg * (Math.PI / 180);
    const de = Math.sin(bearing);
    const dn = Math.cos(bearing);
    const out: number[] = [];
    for (let k = 0; k < frames; k += 1) {
      const g = windAt(0.8, bearingDeg, k * DT, position);
      out.push(g[0] * de + g[1] * dn);
    }
    return out;
  }

  /** The lag, in seconds, at which `later` best matches `earlier`. Positive means later lags. */
  function bestLagSeconds(earlier: readonly number[], later: readonly number[]): number {
    const maxLag = 180;
    let best = 0;
    let bestScore = -Infinity;
    for (let lag = -maxLag; lag <= maxLag; lag += 1) {
      let score = 0;
      let n = 0;
      for (let i = maxLag; i < earlier.length - maxLag; i += 1) {
        score += (earlier[i] ?? 0) * (later[i + lag] ?? 0);
        n += 1;
      }
      score /= Math.max(1, n);
      if (score > bestScore) {
        bestScore = score;
        best = lag;
      }
    }
    return best * DT;
  }

  it("delivers the same gust later the further downwind a point is", () => {
    for (const bearingDeg of [0, 90, 143, 250]) {
      const bearing = bearingDeg * (Math.PI / 180);
      const downwind: Vec3 = [Math.sin(bearing), Math.cos(bearing), 0];
      // Two points 4 m apart along the wind, on the axis so the cross-wind term is zero.
      const upwind: Vec3 = [-2 * downwind[0], -2 * downwind[1], 0];
      const behind: Vec3 = [2 * downwind[0], 2 * downwind[1], 0];
      const lag = bestLagSeconds(
        series(upwind, 3600, bearingDeg),
        series(behind, 3600, bearingDeg),
      );
      // 4 m at 3.5 m/s is 1.14 s. The estimate is a correlation peak on a noisy signal, so it
      // is checked for sign and order of magnitude rather than to three figures.
      expect(lag).toBeGreaterThan(0.5);
      expect(lag).toBeLessThan(2);
    }
  });

  it("puts points abreast of each other out of step as well", () => {
    // Without the cross-wind term the field is a plane wave and these two are identical.
    const bearingDeg = 250;
    const bearing = bearingDeg * (Math.PI / 180);
    const across: Vec3 = [Math.cos(bearing), -Math.sin(bearing), 0];
    const left: Vec3 = [-2 * across[0], -2 * across[1], 0];
    const right: Vec3 = [2 * across[0], 2 * across[1], 0];
    const a = series(left, 1800, bearingDeg);
    const b = series(right, 1800, bearingDeg);
    let differing = 0;
    for (let i = 0; i < a.length; i += 1) {
      if (Math.abs((a[i] ?? 0) - (b[i] ?? 0)) > 1e-3) differing += 1;
    }
    expect(differing).toBeGreaterThan(0.9 * a.length);
  });

  it("is exactly the uniform field at the origin, so every bound carries over", () => {
    for (const t of [0, 1.5, 37.25]) {
      for (const bearingDeg of [0, 143]) {
        expect(windAt(0.7, bearingDeg, t, [0, 0, 0])).toEqual(wind(0.7, bearingDeg, t));
      }
    }
  });

  it("stays inside the magnitude bound everywhere in the scene, not just at the origin", () => {
    // `maxWindMagnitude` is what `maxNodeAngle` and `maxFlutterAmplitude` are built on, and it
    // knows nothing about position. Shifting a bounded function's argument cannot leave its
    // range, and this is that argument checked rather than asserted.
    const limit = maxWindMagnitude(0.9);
    for (let k = 0; k < 400; k += 1) {
      const t = k * 0.11;
      for (const position of [
        [0, 0, 0],
        [3, -4, 6],
        [-9, 9, 1],
        [0.5, 0.25, 12],
      ] as Vec3[]) {
        const g = windAt(0.9, 143, t, position);
        expect(Math.hypot(g[0], g[1], g[2])).toBeLessThanOrEqual(limit + 1e-12);
      }
    }
  });

  it("is dead calm everywhere at zero strength", () => {
    for (const position of [
      [0, 0, 0],
      [5, 5, 5],
      [-3, 2, 9],
    ] as Vec3[]) {
      expect(windAt(0, 90, 12.5, position)).toEqual([0, 0, 0]);
    }
  });

  it("has a delay that is a pure function of bearing and position", () => {
    expect(gustDelaySeconds(0, [0, 0, 0])).toBe(0);
    // Downwind of the origin is later; upwind is earlier.
    expect(gustDelaySeconds(0, [0, 5, 0])).toBeGreaterThan(0);
    expect(gustDelaySeconds(0, [0, -5, 0])).toBeLessThan(0);
    // Height alone does not delay a gust: this field has no shear.
    expect(gustDelaySeconds(37, [0, 0, 9])).toBe(gustDelaySeconds(37, [0, 0, 0]));
    // And rotating the bearing rotates the field rigidly.
    expect(gustDelaySeconds(90, [5, 0, 0])).toBeCloseTo(gustDelaySeconds(0, [0, 5, 0]), 12);
  });
});
