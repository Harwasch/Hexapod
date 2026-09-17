import { describe, expect, it } from "vitest";

import { destination, haversineDistance } from "./coordinates";
import { coveragePath, preferredHeadingDeg } from "./coverage";

// A 200 m × 100 m rectangle (east-west long side) near Redmond.
const origin = { longitude: -122.139, latitude: 47.644 };
const east = destination(origin, 90, 200);
const north = destination(origin, 0, 100);
const rect = [
  [origin.longitude, origin.latitude],
  [east.longitude, east.latitude],
  [east.longitude, north.latitude],
  [origin.longitude, north.latitude],
  [origin.longitude, origin.latitude],
];

describe("coveragePath", () => {
  it("drives along the long axis with swath-spaced serpentine passes", () => {
    expect(preferredHeadingDeg(rect)).toBe(90);
    const path = coveragePath(rect, { swathM: 10 });
    expect(path.headingDeg).toBe(90);
    expect(path.passes).toBe(10);
    expect(path.swathM).toBe(10);
    expect(path.workingLengthM).toBeGreaterThan(1900);
    expect(path.workingLengthM).toBeLessThan(2100);
    expect(path.positions).toHaveLength(20);
    // Pass 1 heads east, pass 2 heads back west, joined at the east end.
    const [a0, a1, b0, b1] = path.positions;
    expect(a1!.longitude).toBeGreaterThan(a0!.longitude);
    expect(b1!.longitude).toBeLessThan(b0!.longitude);
    expect(haversineDistance(a1!, b0!)).toBeLessThan(11);
    // Every pass stays inside the rectangle.
    for (const p of path.positions) {
      expect(p.latitude).toBeGreaterThanOrEqual(origin.latitude - 1e-9);
      expect(p.latitude).toBeLessThanOrEqual(north.latitude + 1e-9);
    }
  });

  it("caps the drawn passes for a preview and reports the wider spacing", () => {
    const path = coveragePath(rect, { swathM: 2, maxPasses: 8 });
    expect(path.passes).toBe(8);
    expect(path.swathM).toBeCloseTo(12.5, 0);
  });

  it("splits a pass into spans across a concave polygon", () => {
    // A U shape: the middle passes cross the polygon twice.
    const e = (m: number) => destination(origin, 90, m).longitude;
    const n = (m: number) => destination(origin, 0, m).latitude;
    const u = [
      [e(0), n(0)],
      [e(300), n(0)],
      [e(300), n(200)],
      [e(200), n(200)],
      [e(200), n(45)],
      [e(100), n(45)],
      [e(100), n(200)],
      [e(0), n(200)],
      [e(0), n(0)],
    ];
    const path = coveragePath(u, { swathM: 20, headingDeg: 90 });
    expect(path.passes).toBe(10);
    // Passes above the notch (y > 45) have two spans of 100 m each; below it one of 300 m.
    expect(path.workingLengthM).toBeCloseTo(2 * 300 + 8 * 200, -2);
  });

  it("returns an empty path for degenerate input", () => {
    expect(
      coveragePath(
        [
          [0, 0],
          [1, 1],
        ],
        { swathM: 5 },
      ).positions,
    ).toEqual([]);
    expect(coveragePath(rect, { swathM: 0 }).passes).toBe(0);
  });
});
