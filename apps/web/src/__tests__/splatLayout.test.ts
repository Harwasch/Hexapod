import { describe, expect, it } from "vitest";

import {
  SH_C0,
  centreColumns,
  playcanvasProperties,
  rgbaOf,
  shCoefficientsOf,
} from "@/lib/splatLayout";

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

describe("a decoded cloud laid out for PlayCanvas", () => {
  it("gives w-first rotations, the colour's degree-0 coefficient and PLY's f_rest order", () => {
    const cloud = {
      numPoints: 2,
      shDegree: 1,
      positions: new Float32Array([1, 2, 3, 4, 5, 6]),
      scales: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
      rotations: new Float32Array([0.1, 0.2, 0.3, 0.9, 0, 0, 0, 1]),
      alphas: new Float32Array([0.25, 1]),
      colors: new Float32Array([0.5, 0.5 + SH_C0, 0.5 - SH_C0, 1, 0, 0.5]),
      // Degree 1: three coefficients of rgb per splat.
      sh: Float32Array.from({ length: 18 }, (_, i) => i),
    };
    const p = playcanvasProperties(cloud);
    expect(Array.from(p.x ?? [])).toEqual([1, 4]);
    expect(Array.from(p.z ?? [])).toEqual([3, 6]);
    expect(Array.from(p.rot_0 ?? [])).toEqual([Math.fround(0.9), 1]);
    expect(Array.from(p.rot_1 ?? [])).toEqual([Math.fround(0.1), 0]);
    expect(Array.from(p.scale_2 ?? [])).toEqual([Math.fround(0.3), Math.fround(0.6)]);
    expect(Array.from(p.opacity ?? [])).toEqual([0.25, 1]);
    expect(p.f_dc_0?.[0]).toBeCloseTo(0, 6);
    expect(p.f_dc_1?.[0]).toBeCloseTo(1, 5);
    expect(p.f_dc_2?.[0]).toBeCloseTo(-1, 5);
    // f_rest: red of coefficients 0..2, then green, then blue.
    expect(Array.from(p.f_rest_0 ?? [])).toEqual([0, 9]);
    expect(Array.from(p.f_rest_1 ?? [])).toEqual([3, 12]);
    expect(Array.from(p.f_rest_3 ?? [])).toEqual([1, 10]);
    expect(Array.from(p.f_rest_8 ?? [])).toEqual([8, 17]);
    expect(p.f_rest_9).toBeUndefined();
  });
});

describe("centreColumns", () => {
  it("moves a tile's positions to its middle and says where that is", () => {
    const columns = {
      x: new Float32Array([100, 102]),
      y: new Float32Array([-50, -48]),
      z: new Float32Array([3, 3]),
    };
    expect(centreColumns(columns)).toEqual([101, -49, 3]);
    expect([...columns.x]).toEqual([-1, 1]);
    expect([...columns.y]).toEqual([-1, 1]);
    expect([...columns.z]).toEqual([0, 0]);
  });
});
