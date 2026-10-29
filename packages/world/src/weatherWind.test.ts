import { describe, expect, it } from "vitest";
import {
  canopyShelter,
  canopyShelterAtHeight,
  plantWindRatio,
  prepareWeatherWind,
  sampleWeatherWind,
  weatherAngle,
  weatherBranch,
} from "./weatherWind";
const input = {
  meanMps: 8,
  gustMps: 16,
  directionDeg: 270,
  latitude: 44.944403,
  longitude: -93.425903,
};
describe("weather-driven simulated wind", () => {
  it("is deterministic and advects the entire two-axis field at the mean speed", () => {
    const f = prepareWeatherWind(input),
      t = 3.75,
      d = 64;
    expect(sampleWeatherWind(f, 0, 0, t)).toEqual(sampleWeatherWind(f, 0, 0, t));
    const a = sampleWeatherWind(f, 0, 0, t),
      b = sampleWeatherWind(f, d, 0, t + d / input.meanMps);
    expect(b.alongMps).toBeCloseTo(a.alongMps, 10);
    expect(b.crossMps).toBeCloseTo(a.crossMps, 10);
    expect(a.crossMps).not.toBe(0);
  });
  it("uses FROM bearings and supports gust factors", () => {
    const f = prepareWeatherWind({ ...input, directionDeg: 0, gustMps: undefined, gustFactor: 2 });
    expect(f.north).toBeCloseTo(-1);
    expect(f.gustMps).toBe(16);
  });
  it("calm is exact, trunk and ground remain exact at storm speed", () => {
    const calm = prepareWeatherWind({ ...input, meanMps: 0, gustMps: 0 });
    expect(weatherBranch(calm, 0, 0, 6, 2, 1, 0.4, 0)).toEqual([0, 0, 0]);
    const f = prepareWeatherWind(input);
    expect(weatherBranch(f, 0, 0, 6, 2, 1, 0.4, 0, true)).toEqual([0, 0, 0]);
    expect(weatherBranch(f, 0, 0, 0, 2, 1, 0.4, 0)).toEqual([0, 0, 0]);
  });
  it("preserves the measured quadratic response then bounds extrapolated storm forcing", () => {
    expect(weatherAngle(0)).toBe(0);
    expect(weatherAngle(6) / weatherAngle(3)).toBeCloseTo(4, 1);
    expect(weatherAngle(1e6)).toBeLessThanOrEqual(0.0115 * 625);
    expect(weatherAngle(23)).toBeGreaterThan(weatherAngle(12));
  });
  it("shelters interior and downwind copies while leaving windward edges exposed", () => {
    const f = prepareWeatherWind({ ...input, directionDeg: 270 });
    const ratios = canopyShelter(
      f,
      [
        { x: 0, y: 0 },
        { x: 3.5, y: 0 },
        { x: 7, y: 0 },
        { x: 7, y: 12 },
      ],
      3.89,
    );
    expect(ratios[0]).toBe(1);
    expect(ratios[1]).toBeLessThan(1);
    expect(ratios[2]).toBeLessThan(ratios[1]!);
    expect(ratios[3]).toBe(1);
    // Closed-canopy attenuation belongs below the canopy top, not above it.
    expect(canopyShelterAtHeight(Math.exp(-2.5), 0)).toBe(Math.exp(-2.5));
    expect(canopyShelterAtHeight(Math.exp(-2.5), 3)).toBeCloseTo(Math.exp(-1.25), 12);
    expect(canopyShelterAtHeight(Math.exp(-2.5), 6)).toBe(1);
    expect(canopyShelterAtHeight(Math.exp(-2.5), 8)).toBe(1);
    expect(canopyShelterAtHeight(1, 3)).toBe(1);
    const exposed = weatherBranch(f, 7, 0, 5, 3, 1, 0.4, 0, false, 6, 1);
    const sheltered = weatherBranch(
      f,
      7,
      0,
      5,
      3,
      1,
      0.4,
      0,
      false,
      6,
      canopyShelterAtHeight(ratios[2] ?? 1, 5),
    );
    expect(Math.hypot(...sheltered)).toBeLessThan(Math.hypot(...exposed));
  });
  it("reduces canopy wind, reaches 10m input, and pins the base", () => {
    expect(plantWindRatio(0)).toBe(0);
    expect(plantWindRatio(10)).toBeCloseTo(1);
    expect(plantWindRatio(3)).toBeLessThan(plantWindRatio(6));
  });
  it("changes length scales and forcing rates with speed and gustiness", () => {
    const a = prepareWeatherWind(input),
      b = prepareWeatherWind({ ...input, meanMps: 2, gustMps: 3 });
    expect(a.alongM).not.toBe(b.alongM);
    expect(a.meanMps / a.alongM).not.toBe(b.meanMps / b.alongM);
  });
});
