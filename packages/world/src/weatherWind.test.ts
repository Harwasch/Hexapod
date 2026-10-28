import { describe, expect, it } from "vitest";
import {
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
  it("grows continuously then streamlines and saturates without widened tolerances", () => {
    const v = [0, 0.3, 1, 3, 8, 15, 20, 100, 1000].map(weatherAngle);
    expect(v.every((x, i) => i === 0 || x > (v[i - 1] ?? 0))).toBe(true);
    expect(weatherAngle(0.2) / weatherAngle(0.1)).toBeGreaterThan(3.9);
    expect(weatherAngle(40) / weatherAngle(20)).toBeLessThan(1.5);
    expect(weatherAngle(1e6)).toBeLessThan(0.1);
    expect(weatherAngle(0.4)).toBeLessThan(weatherAngle(20) / 100);
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
