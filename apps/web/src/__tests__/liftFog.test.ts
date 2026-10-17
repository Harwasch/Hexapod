import { describe, expect, it } from "vitest";

import {
  FOGGY_THRESHOLD,
  scoutGoal,
  scoutZones,
  SCOUT_ATTRIBUTION,
} from "@/features/knowledge/liftFog";
import { isPrivateZone } from "@/missions/landGround";
import { DEFAULT_SCOUT } from "@/missions/scout";

const square = (west: number) => ({
  type: "Polygon" as const,
  coordinates: [
    [
      [west, 34.69],
      [west + 0.004, 34.69],
      [west + 0.004, 34.694],
      [west, 34.694],
      [west, 34.69],
    ],
  ],
});

describe("Lift the fog", () => {
  it("asks for a survey, never a treatment", () => {
    const goal = scoutGoal();
    expect(goal).toMatch(/^Survey the foggy ground from the air at 1\.7 cm per pixel/);
    // The planner reads "clear" as a treatment.
    expect(goal.toLowerCase()).not.toContain("clear");
    expect(FOGGY_THRESHOLD).toBe(0.5);
  });

  it("makes private scout areas derived from the land, after the ids in use", () => {
    const zones = scoutZones(
      [
        { footprint: square(-120.05), reason: "Foggiest 40 ac within one battery" },
        { footprint: square(-120.04), reason: "Next" },
      ],
      { id: "land-1", revision: 3 },
      [{ id: "A-01" }, { id: "Z-14" }],
    );
    expect(zones.map((z) => z.id)).toEqual(["A-02", "A-03"]);
    expect(zones.map((z) => z.name)).toEqual(["Fog patch 1", "Fog patch 2"]);
    for (const zone of zones) {
      expect(zone.scout).toEqual(DEFAULT_SCOUT);
      expect(zone.task).toBe("Scout");
      expect(zone.attribution).toBe(SCOUT_ATTRIBUTION);
      expect(zone.land).toEqual({
        kind: "derived",
        landId: "land-1",
        featureId: null,
        boundaryRevision: 3,
      });
      expect(isPrivateZone(zone)).toBe(true);
    }
    expect(zones[0]?.note).toBe("Foggiest 40 ac within one battery");
  });
});
