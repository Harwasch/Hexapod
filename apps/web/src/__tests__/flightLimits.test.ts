/**
 * What the DJI export refuses before it offers a download (missions/flightLimits.ts): a route
 * with an area it could not plan, whole or in part, and a route that would leave the site's
 * boundary or enter one of its holes (holes are no-fly).
 */
import type { Footprint } from "@twin/contracts";
import { destination } from "@twin/geo";
import { describe, expect, it } from "vitest";

import { flightRefusal, routeRefusal } from "@/missions/flightLimits";
import { DEFAULT_SCOUT, type Scout } from "@/missions/scout";

const origin = { longitude: -120.04, latitude: 34.698 };

/** A `widthM` x `heightM` rectangle `dx`, `dy` metres east and north of the origin. */
function rectangle(widthM: number, heightM: number, dx = 0, dy = 0): Footprint {
  const sw = destination(destination(origin, 90, dx), 0, dy);
  const east = destination(sw, 90, widthM).longitude;
  const north = destination(sw, 0, heightM).latitude;
  return {
    type: "Polygon",
    coordinates: [
      [
        [sw.longitude, sw.latitude],
        [east, sw.latitude],
        [east, north],
        [sw.longitude, north],
        [sw.longitude, sw.latitude],
      ],
    ],
  };
}

const oblique: Scout = {
  ...DEFAULT_SCOUT,
  purpose: "re-scan-weak",
  pattern: "oblique",
  gimbalPitchDeg: -45,
};

describe("an empty or partial route", () => {
  it("is refused when an area is too small for a single lane, naming it", () => {
    const refusal = routeRefusal([
      { name: "Fog patch 1", footprint: rectangle(300, 300), scout: DEFAULT_SCOUT },
      { name: "Weak area 2", footprint: rectangle(5, 5, 400), scout: DEFAULT_SCOUT },
    ]);
    expect(refusal).toContain("Weak area 2: no lane fits");
    expect(refusal).not.toContain("Fog patch 1");
  });

  it("is refused when an oblique fits neither crossing pass", () => {
    const refusal = routeRefusal([
      { name: "Weak area 1", footprint: rectangle(5, 5), scout: oblique },
    ]);
    expect(refusal).toContain("Weak area 1: no lane fits");
  });

  it("is refused when an oblique fits only one of its two crossing passes", () => {
    const refusal = routeRefusal([
      { name: "Weak area 1", footprint: rectangle(200, 5), scout: oblique },
    ]);
    expect(refusal).toContain("Weak area 1: only one of its two crossing passes fits");
  });

  it("is refused when there is nothing to fly at all", () => {
    expect(routeRefusal([])).toContain("Nothing to fly");
  });

  it("is not refused when every area's passes are planned", () => {
    expect(
      routeRefusal([
        { name: "Fog patch 1", footprint: rectangle(300, 300), scout: DEFAULT_SCOUT },
        { name: "Weak area 2", footprint: rectangle(200, 200, 400), scout: oblique },
        {
          name: "Weak area 3",
          footprint: rectangle(10, 10, 800),
          scout: { ...oblique, pattern: "orbit", gimbalPitchDeg: -30 },
        },
      ]),
    ).toBeNull();
  });
});

/** `outer` with `holes` cut out of it: each a [west, south, east, north] box in metres. */
function holed(outer: Footprint, ...holes: [number, number, number, number][]): Footprint {
  const ring = ([w, s, e, n]: [number, number, number, number]) =>
    (rectangle(e - w, n - s, w, s).coordinates[0] ?? []).slice().reverse();
  return {
    type: "Polygon",
    coordinates: [...(outer.coordinates as number[][][]), ...holes.map(ring)],
  } as Footprint;
}

const orbit: Scout = { ...oblique, pattern: "orbit", gimbalPitchDeg: -30 };

describe("the site's boundary and holes", () => {
  it("refuse a route when there is no boundary to keep it inside", () => {
    const areas = [{ name: "Fog patch 1", footprint: rectangle(300, 300), scout: DEFAULT_SCOUT }];
    expect(flightRefusal(areas, null)).toContain("No site boundary");
  });

  it("let a route fly lanes that end on the boundary itself", () => {
    const site = rectangle(300, 300);
    const areas = [{ name: "Fog patch 1", footprint: site, scout: DEFAULT_SCOUT }];
    expect(flightRefusal(areas, site)).toBeNull();
  });

  it("refuse an orbit that would leave the boundary, and keep one inside it", () => {
    // At 60 m and 30° down the ring flies about 104 m out from the area.
    const nearEdge = [{ name: "Weak area 1", footprint: rectangle(10, 10, 20, 145), scout: orbit }];
    const refusal = flightRefusal(nearEdge, rectangle(300, 300));
    expect(refusal).toContain("Weak area 1: its orbit");
    expect(refusal).toContain("would leave the site boundary");
    const middle = [{ name: "Weak area 1", footprint: rectangle(10, 10, 295, 295), scout: orbit }];
    expect(flightRefusal(middle, rectangle(600, 600))).toBeNull();
  });

  it("refuse an area whose legs between lanes would cross a hole", () => {
    const site = holed(rectangle(300, 300), [100, 100, 200, 200]);
    const areas = [{ name: "Fog patch 1", footprint: site, scout: DEFAULT_SCOUT }];
    expect(flightRefusal(areas, site)).toContain(
      "Fog patch 1: a leg between its passes would cross a no-fly hole",
    );
  });

  it("refuse a leg from one area to the next across a hole", () => {
    const site = holed(rectangle(700, 300), [300, 20, 400, 280]);
    const areas = [
      { name: "Fog patch 1", footprint: rectangle(250, 100, 0, 100), scout: DEFAULT_SCOUT },
      { name: "Fog patch 2", footprint: rectangle(250, 100, 450, 100), scout: DEFAULT_SCOUT },
    ];
    expect(flightRefusal(areas, site)).toContain(
      "The leg from Fog patch 1 to Fog patch 2 would cross a no-fly hole",
    );
    // Without the hole the same route is flown.
    expect(flightRefusal(areas, rectangle(700, 300))).toBeNull();
  });

  it("treat a hole in an area as no-fly too", () => {
    const patch = holed(rectangle(300, 300), [100, 100, 200, 200]);
    const areas = [{ name: "Fog patch 1", footprint: patch, scout: DEFAULT_SCOUT }];
    expect(flightRefusal(areas, rectangle(300, 300))).toContain("no-fly hole");
  });
});
