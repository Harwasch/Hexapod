/**
 * The living lab's pure parts (prototype): the hook wraps whatever hook is in, the ground grid
 * finds the ground under a crown, and As captured turns everything off.
 */

import { describe, expect, it } from "vitest";

import {
  groundGrid,
  livingHookVs,
  livingUniforms,
  packLiving,
  sunDirection,
  type GroundSample,
} from "@/cesium/scanView/livingHook";
import { SUN_HOOK_VS } from "@/cesium/scanView/sunHook";
import { useAsCaptured } from "@/state/asCaptured";
import { effectiveLivingLab, useLivingLab } from "@/state/livingLab";

function sample(points: [number, number, number][]): GroundSample {
  return {
    count: points.length,
    at: (i, out) => {
      const p = points[i] ?? [0, 0, 0];
      out[0] = p[0];
      out[1] = p[1];
      out[2] = p[2];
    },
  };
}

describe("livingHookVs", () => {
  it("wraps PlayCanvas's empty hook when none is in", () => {
    const vs = livingHookVs(undefined);
    expect(vs.match(/void modifySplatCenter\(/g)).toHaveLength(1);
    expect(vs).toContain("void hexapodLivingInnerCenter(");
    expect(vs).toContain("uniform vec3 view_position;");
  });

  it("renames the sun hook's functions and does not declare the eye twice", () => {
    const vs = livingHookVs(SUN_HOOK_VS);
    expect(vs).toContain("void hexapodLivingInnerColor(");
    expect(vs.match(/void modifySplatColor\(/g)).toHaveLength(1);
    expect(vs.match(/uniform vec3 view_position;/g)).toHaveLength(1);
  });

  it("uses no GLSL reserved words as names", () => {
    const vs = livingHookVs(undefined);
    expect(vs).not.toMatch(/\b(float|vec4)\s+(flat|half|input|output|filter|sample)\b/);
  });
});

describe("groundGrid", () => {
  it("puts the ground under a floating crown at the trunk's base", () => {
    const points: [number, number, number][] = [];
    // A trunk at the origin from 0 to 3 m, and a crown 3-6 m up, 4 m across.
    for (let z = 0; z <= 3; z += 0.05) points.push([0, 0, z]);
    for (let x = -2; x <= 2; x += 0.1)
      for (let y = -2; y <= 2; y += 0.1) points.push([x, y, 3 + ((((x * 7 + y * 3) % 3) + 3) % 3)]);
    const grid = groundGrid([sample(points)], 16, 1_000_000, 4);
    expect(grid).not.toBeNull();
    const z = grid?.z ?? new Float32Array();
    expect(Math.max(...z)).toBeLessThan(0.5);
  });

  it("follows a slope over a wide scan", () => {
    const points: [number, number, number][] = [];
    for (let x = 0; x <= 40; x += 0.25)
      for (let y = 0; y <= 40; y += 0.5) points.push([x, y, 0.1 * x]);
    const grid = groundGrid([sample(points)], 16, 1_000_000, 4);
    const z = grid?.z ?? new Float32Array();
    // West edge near 0 m, east edge near 4 m (less the 4 m reach's 0.4 m).
    expect(z[0]).toBeLessThan(0.5);
    expect(z[15]).toBeGreaterThan(3);
  });

  it("is null with no splats", () => {
    expect(groundGrid([])).toBeNull();
  });
});

describe("packLiving", () => {
  it("turns bearings clockwise from north into the scan frame's east and north", () => {
    const u = livingUniforms();
    packLiving(
      u,
      {
        t: 1,
        water: true,
        motion: "clumps",
        strength: 1,
        windBearingDeg: 90,
        gustSpeed: 4,
        greenness: 0.04,
        mask: false,
        waterLevel: 0,
        waterThickness: 0.2,
        waterBox: [0, 0, 1, 1],
        flowBearingDeg: 0,
        flowSpeed: 0.4,
        sunDir: sunDirection(180, 30),
        sunUp: 0.5,
        flatGround: 0,
      },
      null,
    );
    expect(u.t[2]).toBe(2);
    expect(u.wind[0]).toBeCloseTo(1);
    expect(u.wind[1]).toBeCloseTo(0);
    // Flow to the north: +y, 90° from +x.
    expect(u.water[3]).toBeCloseTo(Math.PI / 2);
    expect(u.sun[1]).toBeLessThan(0);
    expect(u.groundGrid[3]).toBe(0);
  });
});

describe("effectiveLivingLab", () => {
  it("draws nothing under As captured, and nothing with every arm off", () => {
    useLivingLab.getState().set({ water: true, motion: "gust", bench: false });
    expect(effectiveLivingLab()).not.toBeNull();
    useAsCaptured.setState({ on: true });
    expect(effectiveLivingLab()).toBeNull();
    useAsCaptured.setState({ on: false });
    useLivingLab.getState().set({ water: false, motion: "off" });
    expect(effectiveLivingLab()).toBeNull();
  });
});
