import { describe, expect, it } from "vitest";

import { rgbaOf, shCoefficientsOf } from "@/lib/splatLayout";

// The engine's own loops (GltfVertexBufferLoader.js processSpz), which the layout replaces.
function engineRgba(colors: Float32Array, alphas: Float32Array): Uint8Array {
  const clamp = (v: number): number => Math.min(255, Math.max(0, v));
  const out = new Uint8Array((colors.length / 3) * 4);
  for (let i = 0; i < colors.length / 3; i++) {
    out[i * 4] = clamp((colors[i * 3] ?? 0) * 255);
    out[i * 4 + 1] = clamp((colors[i * 3 + 1] ?? 0) * 255);
    out[i * 4 + 2] = clamp((colors[i * 3 + 2] ?? 0) * 255);
    out[i * 4 + 3] = clamp((alphas[i] ?? 0) * 255);
  }
  return out;
}

function engineSh(sh: Float32Array, count: number, degree: number, l: number, n: number) {
  const stride = [0, 9, 24, 45][degree] ?? 0;
  const base = [0, 9, 24];
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const idx = i * stride + (base[l - 1] ?? 0) + n * 3;
    out[i * 3] = sh[idx] ?? 0;
    out[i * 3 + 1] = sh[idx + 1] ?? 0;
    out[i * 3 + 2] = sh[idx + 2] ?? 0;
  }
  return out;
}

describe("a decoded cloud laid out for the glTF loader", () => {
  it("gives the same colour bytes as the engine, out-of-range values included", () => {
    const colors = new Float32Array([0, 0.5, 1, -0.2, 1.3, 0.999, 0.001, 0.25, 0.75]);
    const alphas = new Float32Array([1, 0.3, 2]);
    expect(rgbaOf({ numPoints: 3, colors, alphas })).toEqual(engineRgba(colors, alphas));
  });

  it("gives every coefficient the engine reads, for each degree", () => {
    for (const degree of [1, 2, 3]) {
      const count = 5;
      const stride = [0, 9, 24, 45][degree] ?? 0;
      const sh = Float32Array.from({ length: count * stride }, (_, i) => i * 0.01 - 1);
      const laidOut = shCoefficientsOf({ numPoints: count, shDegree: degree, sh });
      let keys = 0;
      for (let l = 1; l <= degree; l++) {
        for (let n = 0; n < 2 * l + 1; n++) {
          expect(laidOut[`${String(l)}:${String(n)}`]).toEqual(engineSh(sh, count, degree, l, n));
          keys++;
        }
      }
      expect(Object.keys(laidOut)).toHaveLength(keys);
    }
  });

  it("has no coefficients at degree 0", () => {
    expect(shCoefficientsOf({ numPoints: 4, shDegree: 0, sh: new Float32Array(0) })).toEqual({});
  });
});
