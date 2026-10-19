import { describe, expect, it } from "vitest";

import fixtures from "../fixtures/coverage_cases.json";
import { destination, haversineDistance } from "./coordinates";
import {
  coverageLanes,
  coveragePath,
  footprintLanes,
  laneStrokes,
  preferredHeadingDeg,
} from "./coverage";

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

interface CoverageCase {
  name: string;
  geometry: { type: string; coordinates: unknown };
  spacingM: number;
  headingDeg?: number;
  maxPasses?: number;
  expect: {
    headingDeg?: number;
    passes?: number;
    spacingM?: number;
    lengthMin?: number;
    lengthMax?: number;
    lengthM?: number;
    lengthTolM?: number;
    /** Holes or several parts: coveragePath (outer ring only) cannot measure it. */
    outerRingOnlyFails?: boolean;
  };
}

// The same cases hold the API's planner (apps/api/tests/test_coverage.py, a shapely port that
// also handles holes and MultiPolygons) to the preview's numbers.
describe("shared coverage fixtures", () => {
  const cases = (fixtures as { cases: CoverageCase[] }).cases.filter(
    (c) => !c.expect.outerRingOnlyFails,
  );
  it.each(cases.map((c) => [c.name, c] as const))("%s", (_name, c) => {
    const ring = (c.geometry.coordinates as number[][][])[0] ?? [];
    const path = coveragePath(ring, {
      swathM: c.spacingM,
      ...(c.headingDeg !== undefined ? { headingDeg: c.headingDeg } : {}),
      ...(c.maxPasses !== undefined ? { maxPasses: c.maxPasses } : {}),
    });
    const e = c.expect;
    if (e.headingDeg !== undefined) expect(path.headingDeg).toBe(e.headingDeg);
    if (e.passes !== undefined) expect(path.passes).toBe(e.passes);
    if (e.spacingM !== undefined) expect(path.swathM).toBeCloseTo(e.spacingM, 1);
    if (e.lengthMin !== undefined) expect(path.workingLengthM).toBeGreaterThan(e.lengthMin);
    if (e.lengthMax !== undefined) expect(path.workingLengthM).toBeLessThan(e.lengthMax);
    if (e.lengthM !== undefined)
      expect(Math.abs(path.workingLengthM - e.lengthM)).toBeLessThan(e.lengthTolM ?? 1);
  });
});

// The canonical lanes: every fixture, holes and MultiPolygons included, matches the planner.
describe("footprintLanes on the shared fixtures", () => {
  const cases = (fixtures as { cases: CoverageCase[] }).cases;
  it.each(cases.map((c) => [c.name, c] as const))("%s", (_name, c) => {
    const lanes = footprintLanes(c.geometry as Parameters<typeof footprintLanes>[0], {
      swathM: c.spacingM,
      ...(c.headingDeg !== undefined ? { headingDeg: c.headingDeg } : {}),
      ...(c.maxPasses !== undefined ? { maxPasses: c.maxPasses } : {}),
    });
    const e = c.expect;
    if (e.headingDeg !== undefined) expect(lanes.headingDeg).toBe(e.headingDeg);
    if (e.passes !== undefined) expect(lanes.passes).toBe(e.passes);
    if (e.spacingM !== undefined) expect(lanes.swathM).toBeCloseTo(e.spacingM, 1);
    if (e.lengthMin !== undefined) expect(lanes.workingLengthM).toBeGreaterThan(e.lengthMin);
    if (e.lengthMax !== undefined) expect(lanes.workingLengthM).toBeLessThan(e.lengthMax);
    if (e.lengthM !== undefined)
      expect(Math.abs(lanes.workingLengthM - e.lengthM)).toBeLessThan(e.lengthTolM ?? 1);
  });
});

describe("coverageLanes", () => {
  const holed = (fixtures as { cases: CoverageCase[] }).cases.find((c) => c.name.includes("hole"))!;
  const polygon = holed.geometry.coordinates as number[][][];

  it("splits the passes a hole crosses and never lays a lane over it", () => {
    const result = coverageLanes(polygon, { swathM: 10 });
    expect(result.passes).toBe(10);
    // Five of ten passes cross the 50 m hole: two lanes each.
    expect(result.lanes).toHaveLength(15);
    const [west, south, east, north] = [-122.138001827, 47.64426982, -122.137336379, 47.64471952];
    for (const lane of result.lanes) {
      const mid = {
        longitude: (lane.a.longitude + lane.b.longitude) / 2,
        latitude: (lane.a.latitude + lane.b.latitude) / 2,
      };
      const insideHole =
        mid.longitude > west &&
        mid.longitude < east &&
        mid.latitude > south &&
        mid.latitude < north;
      expect(insideHole).toBe(false);
      expect(haversineDistance(lane.a, lane.b)).toBeCloseTo(lane.lengthM, -1);
    }
    // Same as the outer ring alone, less the hole.
    const whole = coverageLanes([polygon[0]!], { swathM: 10 });
    expect(whole.workingLengthM - result.workingLengthM).toBeCloseTo(250, -1);
  });

  it("drives a serpentine: each pass reverses, and a split pass keeps its direction", () => {
    const { lanes } = coverageLanes(polygon, { swathM: 10 });
    const east = (l: (typeof lanes)[number]) => l.b.longitude > l.a.longitude;
    const byPass = new Map<number, boolean[]>();
    for (const lane of lanes) byPass.set(lane.pass, [...(byPass.get(lane.pass) ?? []), east(lane)]);
    const directions = [...byPass.values()];
    for (const dirs of directions) expect(new Set(dirs).size).toBe(1);
    for (let i = 1; i < directions.length; i++)
      expect(directions[i]![0]).toBe(!directions[i - 1]![0]);
  });

  it("strokes join consecutive passes but never bridge a hole", () => {
    const { lanes } = coverageLanes(polygon, { swathM: 10 });
    const strokes = laneStrokes(lanes);
    expect(strokes.flat()).toHaveLength(lanes.length * 2);
    // Without holes the whole serpentine is one stroke.
    expect(laneStrokes(coverageLanes([polygon[0]!], { swathM: 10 }).lanes)).toHaveLength(1);
    expect(strokes.length).toBeGreaterThan(1);
  });

  it("returns nothing for degenerate input", () => {
    expect(
      coverageLanes(
        [
          [
            [0, 0],
            [1, 1],
          ],
        ],
        { swathM: 5 },
      ).lanes,
    ).toEqual([]);
    expect(coverageLanes(polygon, { swathM: 0 }).passes).toBe(0);
  });
});

describe("footprintLanes over many parts", () => {
  // Two 200 m x 100 m rectangles (the long side east-west) 900 m apart north to south: a
  // footprint 1.1 km across its passes, like workable ground left in pieces.
  const southWest = origin;
  const farNorth = destination(origin, 0, 1000);
  const rectangleAt = (corner: { longitude: number; latitude: number }) => {
    const e = destination(corner, 90, 200);
    const n = destination(corner, 0, 100);
    return [
      [
        [corner.longitude, corner.latitude],
        [e.longitude, corner.latitude],
        [e.longitude, n.latitude],
        [corner.longitude, n.latitude],
        [corner.longitude, corner.latitude],
      ],
    ];
  };
  const parts = {
    type: "MultiPolygon" as const,
    coordinates: [rectangleAt(southWest), rectangleAt(farNorth)],
  };

  it("caps the passes of the whole footprint, at one spacing for every part", () => {
    const capped = footprintLanes(parts, { swathM: 6, maxPasses: 28, headingDeg: 90 });
    // 1.1 km / 28 passes: about 39 m apart, so each 100 m part takes three passes, not the
    // seventeen a 6 m spacing would give it alone.
    expect(capped.swathM).toBeCloseTo(1100 / 28, 0);
    const perPart = new Map<boolean, number>();
    for (const lane of capped.lanes) {
      const north = lane.a.latitude > (southWest.latitude + farNorth.latitude) / 2;
      perPart.set(north, (perPart.get(north) ?? 0) + 1);
    }
    expect([...perPart.values()]).toEqual([3, 3]);
    expect(capped.passes).toBe(6);
    // Uncapped, each part is swept at the spacing asked for, as before.
    const fine = footprintLanes(parts, { swathM: 10, headingDeg: 90 });
    expect(fine.passes).toBe(20);
    expect(fine.swathM).toBe(10);
  });

  it("sweeps one polygon exactly as before", () => {
    const single = { type: "Polygon" as const, coordinates: rectangleAt(southWest) };
    const lanes = footprintLanes(single, { swathM: 2, maxPasses: 8 });
    expect(lanes.passes).toBe(8);
    expect(lanes.swathM).toBeCloseTo(12.5, 1);
  });
});
