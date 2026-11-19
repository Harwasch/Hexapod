import { describe, expect, it } from "vitest";
import {
  articulationPixels,
  BREEZE_PRESETS,
  breezeAngle,
  evaluateArticulation,
  prepareBreeze,
  rotatePoint,
  multiplyQuaternion,
  type Articulation,
  type BreezePreset,
  type Quat,
} from "./livingArticulation";

function fixture(): Articulation {
  return {
    header: {
      format: "hexapod.living-articulation",
      version: 1,
      motionEvidence: "simulated",
      nodes: [
        { parent: -1, band: "trunk", position: [0, 0, 0] },
        { parent: 0, band: "trunk", position: [0, 0, 1] },
        { parent: 1, position: [1, 0, 2] },
        { parent: 2, position: [1.3, 0, 2.3] },
      ],
      frequenciesHz: [0],
      referenceBearingDeg: 90,
      motionBinary: "",
      leafBinary: "",
      bindingBinary: "",
      leafCount: 1,
      tiles: {},
    },
    coefficients: new Float32Array(24),
    leaves: new Float32Array([1, 0, 2, 0, 1, 0, 3.7, 0.4, 0.08]),
    bindings: new Uint32Array([3, 2, 65535, 1, 0, 0, 0, 0]),
  };
}
describe("simulated breeze", () => {
  for (const preset of Object.keys(BREEZE_PRESETS) as BreezePreset[]) {
    it(`${preset} is deterministic, bounded, continuous and pins trunk/root`, () => {
      const a = fixture(),
        model = prepareBreeze(a, preset);
      expect(prepareBreeze(a, preset)).toEqual(model);
      for (let t = -100; t < 100; t += 0.17) {
        expect(breezeAngle(model, 0, t)).toBe(0);
        expect(breezeAngle(model, 1, t)).toBe(0);
        for (const i of [2, 3]) {
          const angle = breezeAngle(model, i, t);
          expect(Math.abs(angle)).toBeLessThanOrEqual(BREEZE_PRESETS[preset].angle);
          expect(Math.abs(angle - breezeAngle(model, i, t + 1e-5))).toBeLessThan(1e-5);
        }
      }
      a.breeze = model;
      const p = articulationPixels(4),
        l = articulationPixels(1);
      evaluateArticulation(a, 2.3, 1, 60, p, l);
      expect(Array.from(p.slice(0, 24))).toEqual([
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      ]);
      const q = Array.from(p.slice(24, 28)) as Quat;
      expect(Math.hypot(...rotatePoint(q, [1, 0, 0]))).toBeCloseTo(1, 6);
      const pivot: [number, number, number] = [1, 0, 2];
      const half =
        0.5 *
        BREEZE_PRESETS[preset].leaf *
        (a.leaves[8] ?? 0) *
        Math.sin(2 * Math.PI * (a.leaves[6] ?? 0) * 2.3 + (a.leaves[7] ?? 0));
      const ca = Math.cos(Math.PI / 6),
        sa = Math.sin(Math.PI / 6);
      const leaf: Quat = [-sa * Math.sin(half), ca * Math.sin(half), 0, Math.cos(half)];
      const expectedQ = multiplyQuaternion(q, leaf);
      for (let c = 0; c < 4; c++) expect(l[12 + c]).toBeCloseTo(expectedQ[c] ?? 0, 6);
      const rest: [number, number, number] = [1.03, 0.02, 2.01];
      const relative = rotatePoint(leaf, [
        rest[0] - pivot[0],
        rest[1] - pivot[1],
        rest[2] - pivot[2],
      ]);
      const expected = rotatePoint(q, [
        relative[0] + pivot[0],
        relative[1] + pivot[1],
        relative[2] + pivot[2],
      ]);
      const combined = rotatePoint(Array.from(l.slice(12, 16)) as Quat, rest);
      for (let c = 0; c < 3; c++)
        expect((combined[c] ?? 0) + (l[8 + c] ?? 0)).toBeCloseTo(
          (expected[c] ?? 0) + (p[28 + c] ?? 0),
          5,
        );
      const before = p.slice();
      evaluateArticulation(a, 2.3, 1, 60, p, l);
      expect(p).toEqual(before);
      // Ground bindings stay zero, and have no packet to move or relight.
      expect(Array.from(a.bindings.slice(4))).toEqual([0, 0, 0, 0]);
      evaluateArticulation(a, 2.3, 0, 60, p, l);
      expect(Array.from(l.slice(8, 16))).toEqual([0, 0, 0, 0, 0, 0, 0, 1]);
    });
  }
  it("reduces local response with order and shorter branch length", () => {
    const m = prepareBreeze(fixture(), "calm");
    expect(m.branches[15]).toBeLessThan(m.branches[10] ?? 0);
  });
  it("bounds the fast quaternion against exact trigonometry, including out-of-range fallback", () => {
    const a = fixture();
    a.leaves[8] = 0.1;
    a.leaves[7] = Math.PI / 2;
    a.breeze = prepareBreeze(a, "windy");
    const p = articulationPixels(4),
      l = articulationPixels(1);
    for (const strength of [0, 1, 3, 5])
      for (let j = 0; j < 200; j++) {
        const t = j / 37;
        evaluateArticulation(a, t, strength, 90, p, l);
        const half =
          0.5 *
          strength *
          (a.leaves[8] ?? 0) *
          Math.sin(2 * Math.PI * (a.leaves[6] ?? 0) * t + (a.leaves[7] ?? 0));
        const parent = Array.from(p.slice(24, 28)) as Quat;
        const exact = multiplyQuaternion(parent, [0, Math.sin(half), 0, Math.cos(half)]);
        // Taylor cos remainder <= .15^6/720 < 1.6e-8; allow four Float32 product sums.
        for (let c = 0; c < 4; c++)
          expect(Math.abs((l[12 + c] ?? 0) - (exact[c] ?? 0))).toBeLessThan(1e-6);
      }
  });
});
