import { beforeEach, describe, expect, it, vi } from "vitest";

import type { KnowledgeScoutRead } from "@twin/contracts";

import type * as KnowledgeApi from "@/features/knowledge/api";
import type * as PlanDrafting from "@/features/mission/planDrafting";
import { isPrivateZone, landProject } from "@/missions/landGround";
import {
  DEFAULT_SCOUT,
  goalAtDetail,
  scoutGeometry,
  scoutGoal,
  scoutOf,
  scoutResolution,
} from "@/missions/scout";
import { useMission } from "@/state/mission";

import { ground, land } from "./landGroundFixture";

const startPlanDraft = vi.fn((..._args: unknown[]) => Promise.resolve());
const requestScoutAreas = vi.fn((..._args: unknown[]): Promise<KnowledgeScoutRead> => {
  throw new Error("set per test");
});

vi.mock("@/features/mission/planDrafting", async (importOriginal) => ({
  ...(await importOriginal<typeof PlanDrafting>()),
  startPlanDraft: (...args: unknown[]) => startPlanDraft(...args),
}));
vi.mock("@/features/knowledge/api", async (importOriginal) => ({
  ...(await importOriginal<typeof KnowledgeApi>()),
  requestScoutAreas: (...args: unknown[]) => requestScoutAreas(...args),
}));

const { FOGGY_THRESHOLD, liftFog, scoutZones, SCOUT_ATTRIBUTION } =
  await import("@/features/knowledge/liftFog");

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

  it("restates the detail for a new height, and answers the planner with it", () => {
    const at120 = scoutGeometry(scoutOf({ ...DEFAULT_SCOUT, altitudeM: 120 })).gsdM;
    expect(goalAtDetail(scoutGoal(), at120)).toBe(
      "Survey the foggy ground from the air at 3.4 cm per pixel so the fog can lift",
    );
    // A goal the operator rewrote without a detail is theirs: kept.
    expect(goalAtDetail("Survey the north paddock", at120)).toBe("Survey the north paddock");
    // The finest choice the flight meets: 1.7 cm meets "2 cm", 3.4 cm meets "5 cm".
    expect(scoutResolution(scoutGeometry(DEFAULT_SCOUT).gsdM)).toBe("2");
    expect(scoutResolution(at120)).toBe("5");
    expect(scoutResolution(0.08)).toBe("10");
  });
});

describe("Lift the fog from the map", () => {
  beforeEach(() => {
    startPlanDraft.mockClear();
    useMission.setState({ composer: null, landGround: null });
    useMission.getState().setProject(landProject(land, ground));
  });

  it("starts a fresh plan, never inside an open revision", async () => {
    // Revising an approved mow plan: its machines, answers and the plan it replaces.
    useMission.getState().openComposer({
      goal: "Mow the west field",
      zoneIds: ["L-0f1e2d3c"],
      machineIds: ["MOW-01"],
      replacePlanId: "plan-mow",
    });
    useMission.getState().updateComposer({ answers: { passes: "two" } });
    requestScoutAreas.mockResolvedValueOnce({
      areas: [
        {
          footprint: square(-120.05),
          areaM2: 160_000,
          cells: 400,
          meanDensity: 0.95,
          bestClassCounts: { none: 400 },
          reason: "Foggiest ground within one battery",
        },
      ],
      threshold: FOGGY_THRESHOLD,
      budgetM2: 500_000,
      truncatedByBudget: false,
      forecast: "Forecast, not evidence.",
      policyVersion: "fog-v1",
    });
    await liftFog(null, land);
    const composer = useMission.getState().composer;
    expect(composer?.replacePlanId).toBeNull();
    expect(composer?.machineIds).toEqual([]);
    expect(composer?.answers).toEqual({});
    expect(composer?.goal).toBe(scoutGoal());
    // Only the patch: the revision's boundary zone did not come along.
    const [patch] = composer?.zoneIds ?? [];
    expect(composer?.zoneIds).toHaveLength(1);
    expect(useMission.getState().project?.zones.find((z) => z.id === patch)?.scout).toEqual(
      DEFAULT_SCOUT,
    );
    expect(startPlanDraft).toHaveBeenCalledWith(scoutGoal(), {
      zoneIds: [patch],
      answers: { resolution: "2" },
    });
  });
});
