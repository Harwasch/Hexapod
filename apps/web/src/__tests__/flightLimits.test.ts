/**
 * What the DJI export refuses before it offers a download (missions/flightLimits.ts): a route
 * with an area it could not plan, whole or in part.
 */
import type { Footprint } from "@twin/contracts";
import { destination } from "@twin/geo";
import { describe, expect, it } from "vitest";

import { routeRefusal } from "@/missions/flightLimits";
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
