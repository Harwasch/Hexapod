import { describe, expect, it } from "vitest";

import {
  SH_C0,
  centreColumns,
  mortonOrder,
  playcanvasProperties,
  reorderColumns,
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

  it("keeps only the bands a device keeps: a phone's one of a degree-3 tile", () => {
    const count = 3;
    const cloud = {
      numPoints: count,
      shDegree: 3,
      positions: new Float32Array(count * 3),
      scales: new Float32Array(count * 3),
      rotations: new Float32Array(count * 4),
      alphas: new Float32Array(count),
      colors: new Float32Array(count * 3),
      // Degree 3: fifteen coefficients of rgb per splat.
      sh: Float32Array.from({ length: count * 45 }, (_, i) => i),
    };
    const full = playcanvasProperties(cloud);
    const phone = playcanvasProperties(cloud, 1);
    const rest = (p: Record<string, Float32Array>) =>
      Object.keys(p).filter((k) => k.startsWith("f_rest_")).length;
    expect(rest(full)).toBe(45);
    expect(rest(phone)).toBe(9);
    // Band 1's coefficients, read from the degree-3 stride: splat 1's first red is at 45.
    expect(Array.from(phone.f_rest_0 ?? [])).toEqual([0, 45, 90]);
    expect(Array.from(phone.f_rest_3 ?? [])).toEqual([1, 46, 91]);
    expect(Array.from(phone.f_rest_8 ?? [])).toEqual([8, 53, 98]);
    expect(rest(playcanvasProperties(cloud, 0))).toBe(0);
  });
});

/**
 * PlayCanvas's own `GSplatData.calcMortonOrder` (playcanvas 2.22, build/playcanvas.mjs), as
 * it is: the reference the worker's radix sort has to match.
 */
function playcanvasMortonOrder(x: Float32Array, y: Float32Array, z: Float32Array): Uint32Array {
  const minMax = (arr: Float32Array) => {
    let min = arr[0] ?? 0;
    let max = arr[0] ?? 0;
    for (let i = 1; i < arr.length; i++) {
      if ((arr[i] ?? 0) < min) min = arr[i] ?? 0;
      if ((arr[i] ?? 0) > max) max = arr[i] ?? 0;
    }
    return { min, max };
  };
  const part1By2 = (v: number): number => {
    v &= 1023;
    v = (v ^ (v << 16)) & 4278190335;
    v = (v ^ (v << 8)) & 50393103;
    v = (v ^ (v << 4)) & 51130563;
    v = (v ^ (v << 2)) & 153391689;
    return v;
  };
  const { min: minX, max: maxX } = minMax(x);
  const { min: minY, max: maxY } = minMax(y);
  const { min: minZ, max: maxZ } = minMax(z);
  const sizeX = minX === maxX ? 0 : 1024 / (maxX - minX);
  const sizeY = minY === maxY ? 0 : 1024 / (maxY - minY);
  const sizeZ = minZ === maxZ ? 0 : 1024 / (maxZ - minZ);
  const codes = new Map<number, number[]>();
  for (let i = 0; i < x.length; i++) {
    const ix = Math.min(1023, Math.floor(((x[i] ?? 0) - minX) * sizeX));
    const iy = Math.min(1023, Math.floor(((y[i] ?? 0) - minY) * sizeY));
    const iz = Math.min(1023, Math.floor(((z[i] ?? 0) - minZ) * sizeZ));
    const code = (part1By2(iz) << 2) + (part1By2(iy) << 1) + part1By2(ix);
    const list = codes.get(code);
    if (list) list.push(i);
    else codes.set(code, [i]);
  }
  const keys = Array.from(codes.keys()).sort((a, b) => a - b);
  const out = new Uint32Array(x.length);
  let at = 0;
  for (const key of keys) for (const i of codes.get(key) ?? []) out[at++] = i;
  return out;
}

describe("PlayCanvas's Morton order, off the main thread", () => {
  it("is PlayCanvas's own order, ties included", () => {
    let seed = 7;
    const random = (): number => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    for (const count of [1, 2, 17, 5000]) {
      // Clumped, so many splats share a cell and their order is the tie-break.
      const column = (scale: number) =>
        Float32Array.from({ length: count }, () => Math.round(random() * 40) * scale - 20);
      const x = column(0.5);
      const y = column(0.01);
      const z = column(3);
      expect(Array.from(mortonOrder(x, y, z))).toEqual(Array.from(playcanvasMortonOrder(x, y, z)));
    }
    // Every splat in one place: their own order.
    const flat = new Float32Array(4);
    expect(Array.from(mortonOrder(flat, flat, flat))).toEqual([0, 1, 2, 3]);
  });

  it("reorders every column the way GSplatData.reorder does", () => {
    const columns = {
      x: Float32Array.from([1, 2, 3]),
      opacity: Float32Array.from([0.1, 0.2, 0.3]),
    };
    const moved = reorderColumns(columns, Uint32Array.from([2, 0, 1]));
    expect(Array.from(moved.x ?? [])).toEqual([3, 1, 2]);
    expect(Array.from(moved.opacity ?? [])).toEqual([0.3, 0.1, 0.2].map(Math.fround));
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
