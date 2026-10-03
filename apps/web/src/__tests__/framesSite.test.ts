import { describe, expect, it } from "vitest";

import { framesSite } from "@/cesium/SiteManager";

describe("framesSite (the site whose controls stay up)", () => {
  // The published camp: about 5 acres, a radius of some 80 m.
  const radius = 80;

  it("holds over the site and close by", () => {
    expect(framesSite(0, 300, radius)).toBe(true);
    expect(framesSite(400, 50, radius)).toBe(true);
  });

  it("holds for a pitched view zoomed out to a couple of kilometres", () => {
    // 2.3 km up at -35 degrees: the camera's ground position is ~3.3 km short of the site.
    expect(framesSite(3300, 2300, radius)).toBe(true);
  });

  it("lets go far out", () => {
    expect(framesSite(9000, 2000, radius)).toBe(false);
    expect(framesSite(0, 12_000, radius)).toBe(false);
  });
});
