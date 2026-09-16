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
  moving: false,
  loading: false,
  memoryRatio: 0.4,
};

describe("decideScreenSpaceError", () => {
  it("refines one step per tick at rest, at any height, down to the preset minimum", () => {
    expect(decideScreenSpaceError(base)).toEqual({
      screenSpaceError: 14,
      reason: "idle refinement",
    });
    expect(decideScreenSpaceError({ ...base, current: 3 }).screenSpaceError).toBe(2);
    expect(decideScreenSpaceError({ ...base, current: 2 })).toEqual({
      screenSpaceError: 2,
      reason: "at finest",
    });
  });

  it("holds while tiles are loading and while memory headroom is gone", () => {
    expect(decideScreenSpaceError({ ...base, loading: true })).toEqual({
      screenSpaceError: 16,
      reason: "loading",
    });
    expect(decideScreenSpaceError({ ...base, memoryRatio: 0.9 }).screenSpaceError).toBe(16);
    expect(decideScreenSpaceError({ ...base, memoryRatio: 0.9 }).reason).toMatch(/holding/);
  });

  it("holds the tile selection while moving, however fine it got", () => {
    expect(decideScreenSpaceError({ ...base, moving: true, current: 6 })).toEqual({
      screenSpaceError: 6,
      reason: "moving (tiles held)",
    });
  });

  it("never returns to the base on its own: only memory pressure coarsens", () => {
    expect(
      decideScreenSpaceError({ ...base, current: 8, memoryRatio: 0.75 }).screenSpaceError,
    ).toBe(8);
    const d = decideScreenSpaceError({ ...base, current: 8, memoryRatio: 1.5 });
    expect(d.screenSpaceError).toBe(12);
    expect(d.reason).toMatch(/memory pressure/);
    expect(
      decideScreenSpaceError({ ...base, current: 31, memoryRatio: 1.5 }).screenSpaceError,
    ).toBe(32);
  });

  it("follows bounds shifted by a ladder penalty", () => {
    const bounds = { base: 22, min: 12, max: 32 };
    expect(decideScreenSpaceError({ ...base, bounds, current: 13 }).screenSpaceError).toBe(12);
    expect(decideScreenSpaceError({ ...base, bounds, current: 12 }).reason).toBe("at finest");
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
