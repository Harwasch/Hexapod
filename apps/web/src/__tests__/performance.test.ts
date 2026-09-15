import { describe, expect, it } from "vitest";

import {
  buildLadder,
  decideScreenSpaceError,
  splatMinimumScreenSpaceError,
  type QualitySample,
} from "@/cesium/PerformanceManager";
import { QUALITY_SSE } from "@/state/settings";

const base: QualitySample = {
  bounds: QUALITY_SSE.balanced,
  current: 16,
  fps: 60,
  moving: false,
  loading: false,
  nearSite: true,
  altitude: 120,
  memoryRatio: 0.4,
};

describe("decideScreenSpaceError", () => {
  it("returns to the preset base when idle far from a site", () => {
    const d = decideScreenSpaceError({ ...base, fps: null, current: 24, altitude: 5000 });
    expect(d).toEqual({ screenSpaceError: 16, reason: "idle" });
  });

  it("refines step by step when idle close to a site, and holds while loading", () => {
    expect(decideScreenSpaceError({ ...base, fps: null, current: 24 }).screenSpaceError).toBe(22);
    expect(decideScreenSpaceError({ ...base, fps: null, current: 7 }).screenSpaceError).toBe(6);
    expect(
      decideScreenSpaceError({ ...base, fps: null, current: 24, loading: true }).screenSpaceError,
    ).toBe(24);
  });

  it("holds the tile selection while moving, whatever the frame rate", () => {
    expect(decideScreenSpaceError({ ...base, moving: true, fps: 20 })).toEqual({
      screenSpaceError: 16,
      reason: "moving (tiles held)",
    });
    expect(decideScreenSpaceError({ ...base, moving: true, fps: 60, current: 6 })).toEqual({
      screenSpaceError: 6,
      reason: "moving (tiles held)",
    });
  });

  it("never backs off on slow frames at rest (those are tiles arriving)", () => {
    expect(decideScreenSpaceError({ ...base, fps: 20, altitude: 5000 }).screenSpaceError).toBe(16);
    expect(decideScreenSpaceError({ ...base, fps: 20, altitude: 5000, current: 31 })).toEqual({
      screenSpaceError: 31,
      reason: "steady",
    });
  });

  it("follows bounds shifted by a ladder penalty at rest", () => {
    const bounds = { base: 22, min: 12, max: 32 };
    expect(decideScreenSpaceError({ ...base, bounds, fps: null, altitude: 5000 })).toEqual({
      screenSpaceError: 22,
      reason: "idle",
    });
    expect(
      decideScreenSpaceError({ ...base, bounds, fps: null, current: 13 }).screenSpaceError,
    ).toBe(12);
  });

  it("memory pressure wins over a good frame rate", () => {
    const d = decideScreenSpaceError({ ...base, fps: 60, memoryRatio: 1.5 });
    expect(d.screenSpaceError).toBe(20);
    expect(d.reason).toMatch(/memory pressure/);
  });

  it("refines only close to a site with measured headroom", () => {
    expect(decideScreenSpaceError({ ...base, fps: 58 }).screenSpaceError).toBe(14);
    expect(decideScreenSpaceError({ ...base, fps: 58, altitude: 900 }).screenSpaceError).toBe(16);
    expect(decideScreenSpaceError({ ...base, fps: 58, loading: true }).screenSpaceError).toBe(16);
  });
});

describe("splatMinimumScreenSpaceError", () => {
  it("lets ultra refine splats further than balanced, never below 4", () => {
    expect(splatMinimumScreenSpaceError("performance")).toBe(12);
    expect(splatMinimumScreenSpaceError("balanced")).toBe(8);
    expect(splatMinimumScreenSpaceError("ultra")).toBe(4);
  });
});

describe("buildLadder", () => {
  it("cuts anti-aliasing, then resolution to a half, then tiles, never past the preset maximum", () => {
    const steps = buildLadder("balanced");
    expect(steps.map((s) => s.label)).toEqual([
      "full",
      "MSAA off",
      "resolution 0.8",
      "resolution 0.65",
      "resolution 0.5",
      "tiles +3 SSE",
      "tiles +6 SSE",
      "tiles +9 SSE",
      "tiles +12 SSE",
      "tiles +15 SSE",
    ]);
    const last = steps[steps.length - 1];
    expect(QUALITY_SSE.balanced.base + (last?.ssePenalty ?? 0)).toBeLessThanOrEqual(
      QUALITY_SSE.balanced.max,
    );
  });

  it("has no anti-aliasing step for the performance preset, which starts without it", () => {
    expect(buildLadder("performance")[1]?.label).toBe("resolution 0.8");
  });
});
