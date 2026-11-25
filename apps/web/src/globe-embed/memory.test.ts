import { afterEach, expect, it } from "vitest";
import {
  EmbedMemoryTier,
  heapNearLimit,
  rememberMemoryPressure,
  restoreMemoryTier,
} from "./memory";

afterEach(() => sessionStorage.clear());
it("halves caches and doubles error once, with the reduced tier used for future loads", () => {
  const tier = new EmbedMemoryTier();
  expect(tier.options).toEqual({
    cacheBytes: 335544320,
    maximumCacheOverflowBytes: 83886080,
    maximumScreenSpaceError: 8,
  });
  expect(tier.stepDown()).toBe(true);
  expect(tier.options).toEqual({
    cacheBytes: 167772160,
    maximumCacheOverflowBytes: 41943040,
    maximumScreenSpaceError: 16,
  });
  expect(tier.stepDown()).toBe(false);
  rememberMemoryPressure();
  expect(restoreMemoryTier().options).toEqual(tier.options);
});
it("only signals pressure at 85 percent of a valid heap limit", () => {
  expect(heapNearLimit()).toBe(false);
  expect(heapNearLimit({ usedJSHeapSize: 999, jsHeapSizeLimit: 0 })).toBe(false);
  expect(heapNearLimit({ usedJSHeapSize: 849, jsHeapSizeLimit: 1000 })).toBe(false);
  expect(heapNearLimit({ usedJSHeapSize: 850, jsHeapSizeLimit: 1000 })).toBe(true);
});
