import { describe, expect, it } from "vitest";

import { decideScreenSpaceError } from "@/cesium/PerformanceManager";
import { SPLAT_BYTES_ESTIMATE, SplatCount, splatMemory, tileGaussians } from "@/cesium/splatCount";

describe("the globe's splat budget, by count", () => {
  it("reads a tile's gaussians from the packer's extras, else from Cesium's own count", () => {
    expect(tileGaussians({ extras: { gaussians: 100_000 }, content: { pointsLength: 7 } })).toBe(
      100_000,
    );
    expect(tileGaussians({ extras: {}, content: { pointsLength: 42_000 } })).toBe(42_000);
    expect(tileGaussians({ extras: { gaussians: "many" }, content: undefined })).toBe(0);
    expect(tileGaussians({})).toBe(0);
  });

  it("counts each tile once, and takes back on unload what it added on load", () => {
    const count = new SplatCount();
    const parent = { extras: { gaussians: 12_000 } };
    const child = { extras: { gaussians: 90_000 }, content: { pointsLength: 90_000 } };
    count.load(parent);
    count.load(child);
    count.load(child);
    expect(count.total).toBe(102_000);
    // By unload Cesium may already have dropped the content; the count must not drift.
    child.content = { pointsLength: 0 };
    count.unload(child);
    count.unload(child);
    expect(count.total).toBe(12_000);
    count.unload({ extras: { gaussians: 5 } });
    expect(count.total).toBe(12_000);
  });

  it("puts the loaded splats against the Detail choice, so the ratio is a count ratio", () => {
    const memory = splatMemory(500_000, 400_000);
    expect(memory.bytes / memory.budget).toBeCloseTo(1.25);
    expect(memory.bytes).toBe(500_000 * SPLAT_BYTES_ESTIMATE);
  });

  it("makes the PerformanceManager coarsen past 125% of Detail, and hold from 70%", () => {
    const at = (loaded: number): string => {
      const { bytes, budget } = splatMemory(loaded, 400_000);
      return decideScreenSpaceError({
        bounds: { base: 16, min: 8, max: 64 },
        current: 16,
        moving: false,
        loading: false,
        memoryRatio: bytes / budget,
      }).reason;
    };
    expect(at(100_000)).toMatch(/^idle refinement/);
    expect(at(300_000)).toMatch(/^holding/);
    expect(at(520_000)).toMatch(/^memory pressure/);
  });
});
