import type { Cesium3DTileset } from "cesium";
import { describe, expect, it } from "vitest";

import {
  CONSTRAINED_WORLD_MIN_SSE_PX,
  devicePixelError,
  groupBounds,
} from "@/cesium/PerformanceManager";
import {
  constrainedDevice,
  deviceMemoryGb,
  shareTileCache,
  sharedTileCacheUsage,
  tileCacheBudget,
  tileCacheShare,
  totalTileCacheBudget,
} from "@/cesium/providers/tiles";
import { QUALITY_SSE } from "@/state/settings";

const MB = 1024 * 1024;
const desktop = (deviceMemory?: number) => ({ deviceMemory, handheld: false, saveData: false });
const phone = (deviceMemory?: number) => ({ deviceMemory, handheld: true, saveData: false });

describe("tile cache budgets, by device", () => {
  it("takes a browser that does not say (Safari) as a desktop's 8 GB, or a phone's smallest", () => {
    expect(deviceMemoryGb(desktop())).toBe(8);
    expect(deviceMemoryGb(phone())).toBe(2);
    expect(deviceMemoryGb(desktop(4))).toBe(4);
    expect(tileCacheBudget(desktop()).cacheBytes).toBe(384 * MB);
  });

  it("holds a phone, and anyone saving data, to the smaller budgets", () => {
    expect(constrainedDevice(phone(8))).toBe(true);
    expect(constrainedDevice({ ...desktop(16), saveData: true })).toBe(true);
    expect(constrainedDevice(desktop(16))).toBe(false);
    expect(tileCacheBudget(phone(8)).cacheBytes).toBe(192 * MB);
    expect(totalTileCacheBudget(phone(8)).cacheBytes).toBe(256 * MB);
    expect(totalTileCacheBudget(desktop(16)).cacheBytes).toBe(1024 * MB);
  });

  it("splits one device total between the tilesets, within each one's own budget", () => {
    // One tileset: its own budget (the total is larger).
    expect(tileCacheShare(1, desktop(8)).cacheBytes).toBe(384 * MB);
    // Three on a phone: 256 MB between them, overflow half of each share.
    const share = tileCacheShare(3, phone());
    expect(share.cacheBytes).toBeCloseTo((256 * MB) / 3, -3);
    expect(share.maximumCacheOverflowBytes).toBe(Math.round(share.cacheBytes / 2));
    // However many, never less than one view's tiles.
    expect(tileCacheShare(50, phone()).cacheBytes).toBe(64 * MB);
  });

  it("re-shares when a tileset joins or is destroyed, and counts what they hold", () => {
    const fake = (bytes: number) => {
      let destroyed = false;
      const tileset = {
        cacheBytes: 0,
        maximumCacheOverflowBytes: 0,
        totalMemoryUsageInBytes: bytes,
        isDestroyed: () => destroyed,
        destroy: () => {
          destroyed = true;
        },
      };
      return tileset as typeof tileset & Cesium3DTileset;
    };
    const before = sharedTileCacheUsage().bytes;
    const a = fake(100 * MB);
    shareTileCache(a);
    const alone = a.cacheBytes;
    const b = fake(50 * MB);
    shareTileCache(b);
    expect(a.cacheBytes).toBeLessThanOrEqual(alone);
    expect(a.cacheBytes).toBe(b.cacheBytes);
    expect(sharedTileCacheUsage().bytes - before).toBe(150 * MB);
    b.destroy();
    expect(a.cacheBytes).toBe(alone);
    expect(sharedTileCacheUsage().bytes - before).toBe(100 * MB);
    a.destroy();
  });
});

describe("the world's error on a constrained device", () => {
  const balanced = QUALITY_SSE.balanced;

  it("is never finer than the floor, at rest or moving, whatever the preset", () => {
    for (const pixelRatio of [1, 2, 3]) {
      const held = groupBounds(balanced, "world", { constrained: true, pixelRatio });
      // What CesiumJS is handed (the world's sink divides by the ratio).
      expect(devicePixelError(held.min, pixelRatio)).toBeGreaterThanOrEqual(
        CONSTRAINED_WORLD_MIN_SSE_PX,
      );
      expect(devicePixelError(held.base, pixelRatio)).toBeGreaterThanOrEqual(
        CONSTRAINED_WORLD_MIN_SSE_PX,
      );
      expect(held.max).toBeGreaterThanOrEqual(held.min);
    }
    const ultra = groupBounds(QUALITY_SSE.ultra, "world", { constrained: true, pixelRatio: 3 });
    expect(devicePixelError(ultra.min, 3)).toBe(CONSTRAINED_WORLD_MIN_SSE_PX);
  });

  it("leaves the sites, and every desktop, as the preset has them", () => {
    expect(groupBounds(balanced, "sites", { constrained: true, pixelRatio: 3 })).toBe(balanced);
    expect(groupBounds(balanced, "world", { constrained: false, pixelRatio: 3 })).toBe(balanced);
  });
});
