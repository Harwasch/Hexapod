import { describe, expect, it } from "vitest";

import { decideScreenSpaceError, type QualitySample } from "@/cesium/PerformanceManager";
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
  it("never changes quality while the scene is idle", () => {
    const d = decideScreenSpaceError({ ...base, fps: null, current: 20 });
    expect(d).toEqual({ screenSpaceError: 20, reason: "idle" });
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
