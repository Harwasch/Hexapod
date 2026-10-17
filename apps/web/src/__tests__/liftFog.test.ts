import { beforeEach, describe, expect, it, vi } from "vitest";

import type { KnowledgeScoutRead } from "@twin/contracts";

import type * as Queries from "@/api/queries";
import type * as KnowledgeApi from "@/features/knowledge/api";
import type * as PlanDrafting from "@/features/mission/planDrafting";
import type * as LandIdentity from "@/state/landIdentity";
import { isPrivateZone, landProject, landZones } from "@/missions/landGround";
import { siteProject } from "@/missions/siteProject";
import {
  DEFAULT_SCOUT,
  goalAtDetail,
  scoutGeometry,
  scoutGoal,
  scoutOf,
  scoutResolution,
} from "@/missions/scout";
import { useLand } from "@/state/land";
import { missionLandId, useMission } from "@/state/mission";
import { useToasts } from "@/state/toasts";

import { ground, land, LAND_ID } from "./landGroundFixture";

const startPlanDraft = vi.fn((..._args: unknown[]) => Promise.resolve());
const requestScoutAreas = vi.fn((..._args: unknown[]): Promise<KnowledgeScoutRead> => {
  throw new Error("set per test");
});

vi.mock("@/features/mission/planDrafting", async (importOriginal) => ({
  ...(await importOriginal<typeof PlanDrafting>()),
  startPlanDraft: (...args: unknown[]) => startPlanDraft(...args),
}));
const fetchLandGround = vi.fn((..._args: unknown[]) => Promise.resolve(ground));
vi.mock("@/api/queries", async (importOriginal) => ({
  ...(await importOriginal<typeof Queries>()),
  fetchLandGround: (...args: unknown[]) => fetchLandGround(...args),
}));
vi.mock("@/features/knowledge/api", async (importOriginal) => ({
  ...(await importOriginal<typeof KnowledgeApi>()),
  requestScoutAreas: (...args: unknown[]) => requestScoutAreas(...args),
}));

// Whose session it is: a test signs someone else in by setting `scope.current`.
const scope = vi.hoisted(() => ({ current: null as string | null }));
vi.mock("@/state/landIdentity", async (importOriginal) => {
  const actual = await importOriginal<typeof LandIdentity>();
  return { ...actual, landScope: () => scope.current ?? actual.landScope() };
});

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
    scope.current = null;
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

  const foggy: KnowledgeScoutRead = {
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
  };

  it("puts the patches on the land's own project when another site is current", async () => {
    // A site switch keeps the land selected, and the fog with it.
    const far = siteProject({
      id: "far",
      slug: "far",
      name: "Redmond scan",
      boundary: {
        type: "Polygon",
        coordinates: [
          [
            [-122.14, 47.64],
            [-122.13, 47.64],
            [-122.13, 47.65],
            [-122.14, 47.64],
          ],
        ],
      },
      areaM2: 800_000,
      centroid: { longitude: -122.135, latitude: 47.645 },
    } as unknown as Parameters<typeof siteProject>[0]);
    useMission.getState().setProject(far);
    useLand.getState().select(land);
    requestScoutAreas.mockResolvedValueOnce(foggy);
    await liftFog(null, land);
    const state = useMission.getState();
    expect(state.project?.id).toBe(`land:${LAND_ID}`);
    expect(missionLandId()).toBe(LAND_ID);
    const [patch] = state.composer?.zoneIds ?? [];
    expect(state.project?.zones.find((z) => z.id === patch)?.land?.landId).toBe(LAND_ID);
    expect(state.areas[far.id] ?? []).toEqual([]);
    useLand.getState().clear();
  });

  it("adopts nothing when the patches arrive after a sign-out", async () => {
    let answer: (read: KnowledgeScoutRead) => void = () => undefined;
    requestScoutAreas.mockImplementationOnce(
      () => new Promise<KnowledgeScoutRead>((resolve) => (answer = resolve)),
    );
    useToasts.setState({ items: [] });
    const lifting = liftFog(null, land);
    // Signed out (or another workspace chosen) while the API answers.
    scope.current = "someone-else/workspace-b";
    useLand.getState().clear();
    useMission.getState().dropPrivate();
    answer(foggy);
    await lifting;
    const state = useMission.getState();
    expect(state.composer).toBeNull();
    expect(state.project?.zones.some((z) => z.task === "Scout")).toBe(false);
    expect(JSON.stringify(state.areas)).not.toContain(LAND_ID);
    expect(startPlanDraft).not.toHaveBeenCalled();
    // Nor does the next session hear about the last one's land.
    expect(JSON.stringify(useToasts.getState().items)).not.toContain(land.name);
  });

  it("closes a card on another land's project and plans on this land's", async () => {
    const otherId = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d";
    const other = { ...land, id: otherId, name: "Creek Field" };
    const otherGround = { ...ground, landId: otherId, name: "Creek Field" };
    useMission
      .getState()
      .engageLand(
        { landId: otherId, zones: landZones(otherGround) },
        landProject(other, otherGround),
      );
    useMission.getState().openComposer({ goal: "monitor the fence", zoneIds: ["L-aaaa0001"] });
    useLand.getState().select(land);
    requestScoutAreas.mockResolvedValueOnce(foggy);
    await liftFog(null, land);
    const state = useMission.getState();
    // Filed under one land: the project and the patches' land agree, as the server requires.
    expect(state.project?.id).toBe(`land:${LAND_ID}`);
    expect(state.composer?.goal).toBe(scoutGoal());
    expect(state.areas[`land:${otherId}`] ?? []).toEqual([]);
    useLand.getState().clear();
  });
});
