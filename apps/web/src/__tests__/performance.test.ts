import { describe, expect, it } from "vitest";

import {
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

  it("coarsens while moving, within the preset bounds", () => {
    const d = decideScreenSpaceError({ ...base, moving: true, fps: 20 });
    expect(d.reason).toBe("moving");
    expect(d.screenSpaceError).toBe(24);
    expect(d.screenSpaceError).toBeLessThanOrEqual(QUALITY_SSE.balanced.max);
  });

  it("backs off on low frame rate and stops at the preset maximum", () => {
    expect(decideScreenSpaceError({ ...base, fps: 20 }).screenSpaceError).toBe(19);
    expect(decideScreenSpaceError({ ...base, fps: 20, current: 31 }).screenSpaceError).toBe(32);
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
