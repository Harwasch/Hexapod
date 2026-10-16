import { describe, expect, it } from "vitest";

import type { LandArea } from "@twin/contracts";
import { destination } from "@twin/geo";

import { anywhereProject } from "@/missions/anywhere";
import { DemoMissionProvider } from "@/missions/demo";
import {
  boundaryZoneOf,
  groundModeFor,
  isPrivateZone,
  landIdOfProject,
  landProject,
  landZoneAt,
  landZones,
  zonesNamedInGoal,
} from "@/missions/landGround";
import { siteProject } from "@/missions/siteProject";

import { ground, land, LAND_ID, origin } from "./landGroundFixture";

describe("land ground", () => {
  it("makes private zones that remember the land, the revision and the line", () => {
    const zones = landZones(ground);
    expect(zones.map((z) => z.id)).toEqual(["L-0f1e2d3c", "L-aaaa0001", "L-aaaa0002"]);
    const [boundary, fence, pond] = zones;
    expect(boundary?.tone).toBe("teal");
    expect(boundary?.acres).toBe(247.1);
    expect(fence?.acres).toBe(2.5);
    expect(pond?.acres).toBe(0.4);
    expect(zones.every((z) => z.private && isPrivateZone(z))).toBe(true);
    expect(fence?.land).toMatchObject({
      kind: "feature",
      landId: LAND_ID,
      featureId: "aaaa0001-0000-4000-8000-000000000001",
      revision: 3,
      boundaryRevision: 2,
      lengthM: 1000,
      bufferM: 10,
    });
    expect(fence?.land?.line?.coordinates).toHaveLength(2);
    expect(fence?.attribution).toContain("Walked with a GPS");
    expect(boundaryZoneOf(zones)?.id).toBe("L-0f1e2d3c");
  });

  it("gives a land of its own a project with no fleet and no numbers in its meta", () => {
    const project = landProject(land, ground);
    expect(project.id).toBe(`land:${LAND_ID}`);
    expect(landIdOfProject(project.id)).toBe(LAND_ID);
    expect(landIdOfProject("demo-blackrock")).toBeNull();
    expect(project.simulated).toBe(false);
    expect(project.machines).toEqual([]);
    expect(project.zones).toHaveLength(3);
    expect(project.meta).not.toMatch(/\d/);
  });

  it("layers onto a project only when its own ground is on the land", () => {
    expect(groundModeFor(null, land)).toBe("project");
    expect(groundModeFor(anywhereProject(), land)).toBe("project");
    expect(groundModeFor(landProject(land, ground), land)).toBe("project");
    const far = siteProject({
      id: "site-far",
      slug: "far",
      name: "Far away",
      boundary: {
        type: "Polygon",
        coordinates: [
          [
            [-122.14, 47.64],
            [-122.13, 47.64],
            [-122.13, 47.65],
            [-122.14, 47.65],
            [-122.14, 47.64],
          ],
        ],
      },
      areaM2: 800_000,
      centroid: { longitude: -122.135, latitude: 47.645 },
    } as unknown as Parameters<typeof siteProject>[0]);
    expect(groundModeFor(far, land)).toBe("project");
    const demo = new DemoMissionProvider().projectForSite("builtin-demo", null)!;
    const onDemo: LandArea = { ...land, boundary: demo.zones[0]!.footprint };
    expect(groundModeFor(demo, onDemo)).toBe("layer");
  });

  it("finds the zone a goal names by its name or by what its category holds", () => {
    const zones = landZones(ground);
    expect(zonesNamedInGoal("monitor the fence every week", zones)).toEqual(["L-aaaa0001"]);
    expect(zonesNamedInGoal("check the troughs", zones)).toEqual(["L-aaaa0002"]);
    expect(zonesNamedInGoal("check the stock pond", zones)).toEqual(["L-aaaa0002"]);
    // Work verbs and time words name no ground; neither does the land's generic name.
    expect(zonesNamedInGoal("monitor every week", zones)).toEqual([]);
    expect(zonesNamedInGoal("survey the land", zones)).toEqual([]);
    expect(zonesNamedInGoal("mow the orchard", zones)).toEqual([]);
  });

  it("takes the smallest land zone under a click", () => {
    const zones = landZones(ground);
    const pond = destination(destination(origin, 90, 420), 0, 420);
    expect(landZoneAt(pond, zones)?.id).toBe("L-aaaa0002");
    const middle = destination(destination(origin, 90, 200), 0, 200);
    expect(landZoneAt(middle, zones)?.id).toBe("L-0f1e2d3c");
    expect(landZoneAt({ longitude: 0, latitude: 0 }, zones)).toBeNull();
  });
});
