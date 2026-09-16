import { describe, expect, it } from "vitest";

import { tighter } from "@/cesium/SiteManager";

import type { Footprint } from "@twin/contracts";

const rect = (w: number, s: number, e: number, n: number): Footprint => ({
  type: "Polygon",
  coordinates: [
    [
      [w, s],
      [e, s],
      [e, n],
      [w, n],
      [w, s],
    ],
  ],
});

describe("tighter", () => {
  const authored = rect(-122.45, 37.76, -122.36, 37.82);

  it("keeps a coverage that fits inside the authored footprint", () => {
    const coverage = rect(-122.44, 37.77, -122.37, 37.81);
    expect(tighter(coverage, authored)).toBe(coverage);
  });

  it("drops a coverage that reaches well past the authored footprint (coarse tiles)", () => {
    const coverage = rect(-122.6, 37.6, -122.2, 37.95);
    expect(tighter(coverage, authored)).toBeNull();
  });

  it("falls back cleanly without a coverage or without an authored footprint", () => {
    expect(tighter(null, authored)).toBeNull();
    const coverage = rect(-122.6, 37.6, -122.2, 37.95);
    expect(tighter(coverage, null)).toBe(coverage);
  });
});
