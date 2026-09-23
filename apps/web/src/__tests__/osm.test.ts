import { describe, expect, it } from "vitest";

import { rectangleFootprint, viewAreaZone } from "@/missions/areas";
import { kindForGoal, overpassQuery, parseOverpass, zoneFromCandidate } from "@/missions/osm";

const lake = [
  { lat: 36.6, lon: -119.9 },
  { lat: 36.6, lon: -119.89 },
  { lat: 36.61, lon: -119.89 },
  { lat: 36.61, lon: -119.9 },
  { lat: 36.6, lon: -119.9 },
];

describe("OpenStreetMap areas", () => {
  it("maps the goal to a kind of ground", () => {
    expect(kindForGoal("Detailed 3d survey of the lake coastline")).toBe("water");
    expect(kindForGoal("Mow the hay fields this week")).toBe("farmland");
    expect(kindForGoal("Clear the thistle")).toBe("wood");
    expect(kindForGoal("Mow Z-14")).toBeNull();
  });

  it("builds a bounded query for the kind", () => {
    const q = overpassQuery({ south: 36.5, west: -120, north: 36.7, east: -119.8 }, "water");
    expect(q).toContain('way["natural"="water"](36.5,-120,36.7,-119.8);');
    expect(q).toContain('relation["natural"="water"]');
    expect(q).toContain("out geom");
  });

  it("parses closed ways and assembles relation outer rings, largest first", () => {
    const half1 = lake.slice(0, 3); // (-119.9,36.6) → (-119.89,36.6) → (-119.89,36.61)
    const half2 = lake.slice(2); // continues back to the start
    const candidates = parseOverpass(
      {
        elements: [
          {
            type: "way",
            id: 1,
            tags: { name: "Small Pond", natural: "water" },
            geometry: lake
              .map((g) => ({ lat: g.lat, lon: g.lon + 0.05 }))
              .map((g, i) => (i === 2 ? { lat: 36.602, lon: g.lon } : g)),
          },
          { type: "way", id: 2, tags: {}, geometry: lake.slice(0, 3) }, // open: dropped
          {
            type: "relation",
            id: 9,
            tags: { name: "Test Lake", type: "multipolygon" },
            members: [
              { type: "way", role: "outer", geometry: half1 },
              { type: "way", role: "outer", geometry: [...half2].reverse() },
              { type: "way", role: "inner", geometry: lake },
            ],
          },
          { type: "node" },
        ],
      },
      "water",
    );
    expect(candidates.map((c) => c.name)).toEqual(["Test Lake", "Small Pond"]);
    const lakeArea = candidates[0]!;
    expect(lakeArea.footprint.type).toBe("Polygon");
    expect(lakeArea.acres).toBeGreaterThan(200);
    const zone = zoneFromCandidate("A-03", lakeArea);
    expect(zone.id).toBe("A-03");
    expect(zone.attribution).toContain("OpenStreetMap");
    expect(zone.acres).toBe(Math.round(lakeArea.acres));
  });
});

describe("view areas", () => {
  it("scale and move with their origin", () => {
    const origin = {
      center: { longitude: -119.9, latitude: 36.6 },
      metersPerPixel: 2,
      width: 1000,
      height: 500,
      fraction: 0.5,
      dx: 0,
      dy: 0,
    };
    const base = viewAreaZone("A-01", "View area 01", origin);
    const bigger = viewAreaZone("A-01", "View area 01", { ...origin, fraction: 1 });
    expect(bigger.acres).toBeGreaterThan(base.acres * 3.5);
    const moved = viewAreaZone("A-01", "View area 01", { ...origin, dx: 0.25, dy: 0 });
    expect(moved.anchor.longitude).toBeGreaterThan(base.anchor.longitude);
    expect(moved.anchor.latitude).toBeCloseTo(base.anchor.latitude, 6);
    expect(moved.view?.dx).toBe(0.25);
    expect(rectangleFootprint(origin.center, 10, 10).type).toBe("Polygon");
  });
});
