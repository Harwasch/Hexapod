import { describe, expect, it } from "vitest";
import { prepareWeatherWind } from "@twin/world";
import {
  articulationPixels,
  evaluateArticulation,
  prepareBreeze,
  type Articulation,
} from "./livingArticulation";
const wind = {
  meanMps: 11.49,
  gustMps: 22.9,
  directionDeg: 316,
  latitude: 44.944403,
  longitude: -93.425903,
};
function fixture(): Articulation {
  return {
    header: {
      format: "hexapod.living-articulation",
      version: 1,
      motionEvidence: "simulated",
      nodes: [
        { parent: -1, band: "trunk", position: [0, 0, 0] },
        { parent: 0, band: "trunk", position: [0, 0, 2] },
        { parent: 1, position: [1, 0, 5] },
      ],
      frequenciesHz: [0],
      referenceBearingDeg: 90,
      motionBinary: "",
      leafBinary: "",
      bindingBinary: "",
      leafCount: 1,
      tiles: {},
    },
    coefficients: new Float32Array(18),
    leaves: new Float32Array([1, 0, 5, 0, 1, 0, 3.7, 0.4, 0.08]),
    bindings: new Uint32Array([3, 2, 65535, 1]),
    weather: prepareWeatherWind(wind),
    tier: "B",
  };
}
describe("weather articulation adapter", () => {
  it("pins roots/trunks exactly, retains covariance rotation, and skips packet flutter at B", () => {
    const a = fixture();
    a.breeze = prepareBreeze(a, "windy");
    const p = articulationPixels(3),
      l = articulationPixels(1);
    l.fill(123);
    evaluateArticulation(a, 3, 1, 90, p, l);
    const first = p.slice();
    evaluateArticulation(a, 3, 1, 90, p, l);
    expect(p).toEqual(first);
    for (const i of [0, 1, 2])
      expect(Array.from(p.slice(i * 8, i * 8 + 8))).toEqual([0, 0, 0, 1, 0, 0, 0, 0]);
    expect(p[24]).not.toBe(0);
    expect(p[25]).not.toBe(0);
    expect(l[12]).toBe(123);
    evaluateArticulation(a, 3, 0, 90, p, l);
    for (let i = 0; i < 4; i++)
      expect(Array.from(p.slice(i * 8, i * 8 + 8))).toEqual([0, 0, 0, 1, 0, 0, 0, 0]);
  });
  it("allows per-leaf motion only on explicitly passing C, driven by weather speed", () => {
    const a = fixture();
    a.breeze = prepareBreeze(a, "windy");
    a.tier = "C";
    const p = articulationPixels(3),
      l = articulationPixels(1);
    evaluateArticulation(a, 3, 1, 90, p, l);
    expect(l[13]).not.toBe(0);
    a.weather = prepareWeatherWind({ ...wind, meanMps: 0, gustMps: 0 });
    evaluateArticulation(a, 3, 1, 90, p, l);
    expect(l[13]).toBe(0);
  });
});
