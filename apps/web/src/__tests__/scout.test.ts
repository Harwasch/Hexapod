import type { Footprint } from "@twin/contracts";
import { destination, footprintContains, haversineDistance } from "@twin/geo";
import { describe, expect, it } from "vitest";

import { MACHINE_CLASSES } from "@/missions/machineClasses";
import {
  DEFAULT_SCOUT,
  RELIEF_MAX_SAMPLES,
  reliefFrom,
  reliefPoints,
  routeStats,
  scoutBudgetM2,
  scoutGeometry,
  scoutLanes,
  scoutOf,
  scoutZonesInOrder,
  sharedScout,
  USABLE_BATTERY_MIN,
} from "@/missions/scout";
import type { Zone } from "@/missions/types";

const origin = { longitude: -120.04, latitude: 34.698 };

function rectangle(widthM: number, heightM: number): Footprint {
  const east = destination(origin, 90, widthM).longitude;
  const north = destination(origin, 0, heightM).latitude;
  return {
    type: "Polygon",
    coordinates: [
      [
        [origin.longitude, origin.latitude],
        [east, origin.latitude],
        [east, north],
        [origin.longitude, north],
        [origin.longitude, origin.latitude],
      ],
    ],
  };
}

describe("scout geometry", () => {
  it("flies the drone class: the one table the planner reads", () => {
    const drone = MACHINE_CLASSES.drone;
    expect(DEFAULT_SCOUT.altitudeM).toBe(drone.altitudeAglM);
    expect(DEFAULT_SCOUT.speedMs).toBe(drone.speedMps);
    expect(DEFAULT_SCOUT.sideOverlap).toBe(drone.sidelapPct / 100);
    expect(DEFAULT_SCOUT.forwardOverlap).toBe(drone.frontlapPct / 100);
    expect(drone.label).toContain("Mavic 3 Enterprise");
  });

  it("derives GSD, footprints and spacings for the M3E wide camera at 60 m", () => {
    const g = scoutGeometry(DEFAULT_SCOUT);
    // 73.7° across: 89.9 m of ground across at 60 m, over 5,280 px.
    expect(g.footprintAcrossM).toBeCloseTo(89.93, 1);
    expect(g.gsdM).toBeCloseTo(0.01703, 4);
    expect(g.footprintAlongM).toBeCloseTo(67.45, 1);
    expect(g.lineSpacingM).toBeCloseTo(26.98, 1);
    expect(g.photoSpacingM).toBeCloseTo(16.86, 1);
    expect(g.maxSpeedMs).toBeGreaterThan(8);
    expect(g.speedMs).toBe(8);
    expect(g.warnings).toEqual([]);
    // About 2 cm per pixel at 70 m, the planner's 2 cm scan height.
    expect(scoutGeometry({ ...DEFAULT_SCOUT, altitudeM: 70 }).gsdM).toBeCloseTo(0.0199, 3);
  });

  it("caps the height at 120 m with a Part 107 warning, and the speed at the camera's pace", () => {
    const high = scoutGeometry({ ...DEFAULT_SCOUT, altitudeM: 150 });
    expect(high.altitudeM).toBe(120);
    expect(high.warnings.join(" ")).toMatch(/Part 107/);
    const fast = scoutGeometry({ ...DEFAULT_SCOUT, altitudeM: 20, speedMs: 15 });
    expect(fast.speedMs).toBeLessThan(15);
    expect(fast.warnings.join(" ")).toMatch(/camera fires/);
  });

  it("budgets one battery of lanes: about 64 acres at the class's flight", () => {
    const budget = scoutBudgetM2();
    const g = scoutGeometry(DEFAULT_SCOUT);
    expect(budget).toBeCloseTo(g.lineSpacingM * 8 * USABLE_BATTERY_MIN * 60 * 0.8, 3);
    expect(budget).toBeGreaterThan(250_000);
    expect(budget).toBeLessThan(270_000);
  });

  it("fills a scout from a stored capture, keeping the class's values for what is missing", () => {
    expect(scoutOf(null)).toEqual(DEFAULT_SCOUT);
    const stored = scoutOf({ altitudeM: 90, forwardOverlap: 0.8, sideOverlap: 0.75, speedMs: 6 });
    expect(stored.altitudeM).toBe(90);
    expect(stored.camera).toBe("dji-m3e-wide");
    expect(stored.headingDeg).toBeNull();
    expect([stored.purpose, stored.pattern, stored.gimbalPitchDeg]).toEqual([
      "lift-knowledge-fog",
      "nadir",
      -90,
    ]);
  });

  it("keeps a re-scan's purpose, pattern and camera pitch", () => {
    const rescanOf = {
      siteId: "00000000-0000-4000-8000-000000000001",
      epochId: "00000000-0000-4000-8000-000000000002",
      policy: "weak-v1" as const,
      cells: 40,
      reasons: ["views" as const],
    };
    const stored = scoutOf({
      purpose: "re-scan-weak",
      pattern: "orbit",
      gimbalPitchDeg: -30,
      rescanOf,
      altitudeM: 30,
      forwardOverlap: 0.8,
      sideOverlap: 0.7,
      speedMs: 5,
    });
    expect([stored.purpose, stored.pattern, stored.gimbalPitchDeg]).toEqual([
      "re-scan-weak",
      "orbit",
      -30,
    ]);
    expect(stored.rescanOf).toEqual(rescanOf);
  });
});

describe("scout lanes and route", () => {
  it("lays lanes one line spacing apart along the long side, and counts the route", () => {
    const footprint = rectangle(600, 300);
    const lanes = scoutLanes(footprint, DEFAULT_SCOUT);
    const spacing = scoutGeometry(DEFAULT_SCOUT).lineSpacingM;
    // A pass every spacing, centred in its strip: the last strip's centre may fall outside.
    expect(lanes.length).toBeGreaterThanOrEqual(Math.floor(300 / spacing));
    expect(lanes.length).toBeLessThanOrEqual(Math.ceil(300 / spacing));
    for (const lane of lanes) expect(lane.lengthM).toBeCloseTo(600, -1);
    const stats = routeStats([lanes], DEFAULT_SCOUT);
    expect(stats.workingM).toBeCloseTo(600 * lanes.length, -2);
    expect(stats.transitM).toBeGreaterThan(0);
    expect(stats.minutes).toBeCloseTo((stats.workingM + stats.transitM) / 8 / 60, 5);
    expect(stats.batteries).toBe(Math.ceil(stats.minutes / USABLE_BATTERY_MIN));
    expect(stats.photos).toBeGreaterThan(lanes.length * 30);
  });

  it("honours a heading the area carries", () => {
    const lanes = scoutLanes(rectangle(600, 300), { ...DEFAULT_SCOUT, headingDeg: 0 });
    for (const lane of lanes) expect(lane.lengthM).toBeCloseTo(300, -1);
  });
});

describe("one route, one capture", () => {
  const area = (id: string, scout: Zone["scout"]) =>
    ({ id, footprint: rectangle(300, 200), scout }) as Zone;

  it("flies the areas in the overlay's order, scout areas only", () => {
    const zones = [
      area("A-01", DEFAULT_SCOUT),
      area("A-02", DEFAULT_SCOUT),
      area("Z-14", undefined),
    ];
    const overlay = {
      zones: [
        { zoneId: "A-02", machineIds: [] },
        { zoneId: "Z-14", machineIds: [] },
        { zoneId: "A-01", machineIds: [] },
      ],
    };
    expect(scoutZonesInOrder(zones, overlay).map((z) => z.id)).toEqual(["A-02", "A-01"]);
  });

  it("shares a capture only when height, speed and overlaps match; heading may differ", () => {
    expect(
      sharedScout([area("A", DEFAULT_SCOUT), area("B", { ...DEFAULT_SCOUT, headingDeg: 30 })]),
    ).toEqual(DEFAULT_SCOUT);
    expect(
      sharedScout([area("A", DEFAULT_SCOUT), area("B", { ...DEFAULT_SCOUT, altitudeM: 90 })]),
    ).toBeNull();
    expect(sharedScout([])).toBeNull();
  });
});

describe("relief samples", () => {
  it("are fixed by the outlines, bounded, on the outline and inside it", () => {
    const small = rectangle(300, 200);
    const points = reliefPoints([small]);
    // 1,000 m of outline every 30 m at most, and a grid inside: well under the cap.
    expect(points.length).toBeGreaterThan(30);
    expect(points.length).toBeLessThanOrEqual(RELIEF_MAX_SAMPLES);
    expect(points[0]?.longitude).toBeCloseTo(origin.longitude, 9);
    expect(points[0]?.latitude).toBeCloseTo(origin.latitude, 9);
    const inside = points.filter((p) => footprintContains(small, p));
    expect(inside.length).toBeGreaterThan(10);
    // The same outline gives the same points, whatever the height flown over it.
    expect(reliefPoints([small])).toEqual(points);
    // A big patch stays within the cap.
    expect(reliefPoints([rectangle(3000, 2000), small]).length).toBeLessThanOrEqual(
      RELIEF_MAX_SAMPLES,
    );
  });

  it("takes the ground at the sample nearest the first waypoint, and the range over all", () => {
    const points = reliefPoints([rectangle(300, 200)]);
    const heights = points.map((_, i) => 100 + i);
    const far = destination(origin, 90, 300);
    const nearest = points.reduce(
      (best, p, i) =>
        haversineDistance(p, far) < haversineDistance(points[best] ?? p, far) ? i : best,
      0,
    );
    const relief = reliefFrom(points, heights, far, "surface");
    expect(relief).toEqual({
      start: 100 + nearest,
      min: 100,
      max: 100 + points.length - 1,
      source: "surface",
    });
    expect(
      reliefFrom(
        points,
        points.map(() => undefined),
        far,
        "terrain",
      ),
    ).toBeNull();
  });
});
