import { describe, expect, it } from "vitest";

import { calibrationFor, tighter } from "@/cesium/SiteManager";

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

describe("calibrationFor", () => {
  it("applies the asset's factor at kilometres and none up close", () => {
    expect(calibrationFor(0.25, 8.8)).toBe(0.25);
    expect(calibrationFor(0.25, 0.36)).toBe(1);
    expect(calibrationFor(0.25, 2)).toBeCloseTo(0.625, 2);
  });

  it("is the identity for uncalibrated assets and unknown scales", () => {
    expect(calibrationFor(1, 5)).toBe(1);
    expect(calibrationFor(0.25, Number.POSITIVE_INFINITY)).toBe(0.25);
  });
});
