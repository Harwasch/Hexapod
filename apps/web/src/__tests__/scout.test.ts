import type { Footprint } from "@twin/contracts";
import { destination } from "@twin/geo";
import { describe, expect, it } from "vitest";

import { MACHINE_CLASSES } from "@/missions/machineClasses";
import {
  DEFAULT_SCOUT,
  routeStats,
  scoutBudgetM2,
  scoutGeometry,
  scoutLanes,
  scoutOf,
  USABLE_BATTERY_MIN,
} from "@/missions/scout";

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
