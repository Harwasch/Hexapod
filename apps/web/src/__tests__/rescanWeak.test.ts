/**
 * "Plan a re-scan of weak areas" (features/rescan/rescanWeak.ts): an epoch's weak cells, as
 * the API outlines them (`POST …/epochs/{epoch}/weak-areas`, weak-v1), become re-scan zones
 * on the site's project, each carrying the scout capture its pattern asks for, and a fresh
 * plan composer drafts a survey over them. The API is mocked; the zones, the composer and
 * the toasts are the module's own.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Site, WeakArea, WeakAreas } from "@twin/contracts";

import type * as Queries from "@/api/queries";
import { ApiError } from "@/api/error";
import type * as PlanDrafting from "@/features/mission/planDrafting";
import { isPrivateZone } from "@/missions/landGround";
import {
  DEFAULT_SCOUT,
  MAX_ALTITUDE_M,
  MIN_ALTITUDE_M,
  scoutBudgetM2,
  scoutGeometry,
  scoutOf,
} from "@/missions/scout";
import { siteProject, siteProjectId } from "@/missions/siteProject";
import { useMission } from "@/state/mission";
import { useToasts } from "@/state/toasts";

const startPlanDraft = vi.fn((..._args: unknown[]) => Promise.resolve());
const requestWeakAreas = vi.fn((..._args: unknown[]): Promise<WeakAreas> => {
  throw new Error("set per test");
});

vi.mock("@/features/mission/planDrafting", async (importOriginal) => ({
  ...(await importOriginal<typeof PlanDrafting>()),
  startPlanDraft: (...args: unknown[]) => startPlanDraft(...args),
}));
vi.mock("@/api/queries", async (importOriginal) => ({
  ...(await importOriginal<typeof Queries>()),
  requestWeakAreas: (...args: unknown[]) => requestWeakAreas(...args),
}));

const { altitudeForGsd, rescanGoal, rescanWeak, rescanZones, RESCAN_ATTRIBUTION } =
  await import("@/features/rescan/rescanWeak");

const SITE_ID = "99999999-9999-4999-8999-999999999999";
const EPOCH_ID = "99999999-9999-4999-8999-000000000001";

const square = (west: number) => ({
  type: "Polygon" as const,
  coordinates: [
    [
      [west, 46.13],
      [west + 0.0002, 46.13],
      [west + 0.0002, 46.1302],
      [west, 46.1302],
      [west, 46.13],
    ],
  ],
});

function area(patch: Partial<WeakArea> = {}): WeakArea {
  return {
    footprint: square(-123.88),
    areaM2: 240,
    cells: 240,
    meanScore: 60,
    reasons: { views: 200, spread: 40, gsd: 0 },
    reliefM: 1.5,
    diameterM: 22,
    pattern: "oblique",
    targetGsdM: null,
    gimbalPitchDeg: -45,
    hint: "Seen from too few views: fly crossing obliques",
    ...patch,
  };
}

function read(areas: WeakArea[]): WeakAreas {
  return {
    epochId: EPOCH_ID,
    assetId: "99999999-9999-4999-8999-0000000000a1",
    policy: "weak-v1",
    cellM: 1,
    level: 2,
    weakCells: areas.reduce((sum, a) => sum + a.cells, 0),
    totalCells: 1280,
    truncatedByBudget: false,
    heuristic: "Patterns are UX rules of thumb, not measurements",
    areas,
  };
}

const site = {
  id: SITE_ID,
  slug: "yard",
  name: "Yard",
  boundary: square(-123.881),
  areaM2: 6_400,
  centroid: { longitude: -123.8805, latitude: 46.1301, height: null },
  assets: [],
} as unknown as Site;

describe("re-scan zones", () => {
  it("fly a target detail at the height that gives it, within 20 to 120 m", () => {
    for (const gsdM of [0.008, 0.0123, 0.02]) {
      const altitudeM = altitudeForGsd(gsdM);
      const flown = scoutGeometry(scoutOf({ ...DEFAULT_SCOUT, altitudeM })).gsdM;
      // Whole metres, never coarser than asked: a metre higher would be.
      expect(Number.isInteger(altitudeM)).toBe(true);
      expect(flown).toBeLessThanOrEqual(gsdM);
      expect(
        scoutGeometry(scoutOf({ ...DEFAULT_SCOUT, altitudeM: altitudeM + 1 })).gsdM,
      ).toBeGreaterThan(gsdM);
    }
    expect(altitudeForGsd(0.0001)).toBe(MIN_ALTITUDE_M);
    expect(altitudeForGsd(1)).toBe(MAX_ALTITUDE_M);
    expect(altitudeForGsd(null)).toBe(DEFAULT_SCOUT.altitudeM);
  });

  it("are public areas on the site, each a re-scan of the epoch's weak cells", () => {
    const zones = rescanZones(
      read([
        area(),
        area({ pattern: "nadir", gimbalPitchDeg: -90, targetGsdM: 0.01, reasons: { gsd: 30 } }),
      ]),
      SITE_ID,
      [{ id: "SITE" }, { id: "A-03" }, { id: "A-L07" }],
    );
    // Numbered with drawn areas (A-…): public ground, unlike a Land's derived A-L… patches.
    expect(zones.map((z) => z.id)).toEqual(["A-04", "A-05"]);
    expect(zones.map((z) => z.name)).toEqual(["Weak area 1", "Weak area 2"]);
    const [oblique, nadir] = zones;
    for (const zone of zones) {
      expect(zone.task).toBe("Re-scan");
      expect(zone.attribution).toBe(RESCAN_ATTRIBUTION);
      expect(isPrivateZone(zone)).toBe(false);
      expect(zone.land).toBeUndefined();
      expect(zone.private).toBeUndefined();
    }
    expect(oblique?.note).toBe("Seen from too few views: fly crossing obliques");
    expect(oblique?.scout).toEqual({
      ...DEFAULT_SCOUT,
      purpose: "re-scan-weak",
      pattern: "oblique",
      gimbalPitchDeg: -45,
      altitudeM: DEFAULT_SCOUT.altitudeM,
      rescanOf: {
        siteId: SITE_ID,
        epochId: EPOCH_ID,
        policy: "weak-v1",
        cells: 240,
        // The reasons the area's weak cells carry, never a zero count.
        reasons: ["views", "spread"],
      },
    });
    expect(nadir?.scout?.pattern).toBe("nadir");
    expect(nadir?.scout?.gimbalPitchDeg).toBe(-90);
    expect(nadir?.scout?.altitudeM).toBe(altitudeForGsd(0.01));
    expect(nadir?.scout?.rescanOf?.reasons).toEqual(["gsd"]);
  });

  it("ask for a survey at the finest detail flown, never a treatment", () => {
    const goal = rescanGoal(0.0172);
    expect(goal).toMatch(/at 1\.7 cm per pixel/);
    for (const word of ["mow", "cut", "clear", "spray", "treat", "drill", "seed", "till"])
      expect(goal.toLowerCase()).not.toContain(word);
  });
});

describe("Plan a re-scan of weak areas", () => {
  beforeEach(() => {
    startPlanDraft.mockClear();
    requestWeakAreas.mockReset();
    useToasts.setState({ items: [] });
    useMission.setState({ composer: null, landGround: null });
    useMission.getState().setProject(siteProject(site));
  });

  const toasts = () => useToasts.getState().items.map((t) => t.title);

  it("opens a fresh composer with the weak areas and drafts a survey over them", async () => {
    useMission.getState().openComposer({ goal: "Mow the yard", machineIds: ["MOW-01"] });
    requestWeakAreas.mockResolvedValueOnce(
      read([
        area(),
        area({
          footprint: square(-123.879),
          pattern: "nadir",
          gimbalPitchDeg: -90,
          targetGsdM: 0.01,
        }),
      ]),
    );
    await rescanWeak(null, site, EPOCH_ID);
    expect(requestWeakAreas).toHaveBeenCalledWith(SITE_ID, EPOCH_ID, {
      budgetM2: scoutBudgetM2(DEFAULT_SCOUT),
      focus: null,
      maxAreas: 3,
    });
    const state = useMission.getState();
    const composer = state.composer;
    expect(composer?.machineIds).toEqual([]);
    expect(composer?.zoneIds).toHaveLength(2);
    expect(composer?.ground?.source).toBe("rescan");
    const zones = composer?.zoneIds.map((id) => state.project?.zones.find((z) => z.id === id));
    expect(zones?.map((z) => z?.scout?.purpose)).toEqual(["re-scan-weak", "re-scan-weak"]);
    // The finest detail asked for names the goal and answers the planner.
    const finest = scoutGeometry(scoutOf({ ...DEFAULT_SCOUT, altitudeM: altitudeForGsd(0.01) }));
    expect(startPlanDraft).toHaveBeenCalledWith(rescanGoal(finest.gsdM), {
      zoneIds: composer?.zoneIds,
      answers: { resolution: "2" },
    });
    expect(toasts()).toEqual([]);
  });

  it("plans on the site's project when another project is current", async () => {
    useMission
      .getState()
      .setProject(siteProject({ ...site, id: "other", slug: "other", name: "Other" }));
    requestWeakAreas.mockResolvedValueOnce(read([area()]));
    await rescanWeak(null, site, EPOCH_ID);
    const state = useMission.getState();
    expect(state.project?.id).toBe(siteProjectId(SITE_ID));
    expect(state.composer?.zoneIds).toHaveLength(1);
    expect(state.areas[siteProjectId("other")] ?? []).toEqual([]);
  });

  it("says when nothing is weak, and plans nothing", async () => {
    requestWeakAreas.mockResolvedValueOnce(read([]));
    await rescanWeak(null, site, EPOCH_ID);
    expect(toasts()).toEqual(["Nothing here is weak (weak-v1)"]);
    expect(useMission.getState().composer).toBeNull();
    expect(startPlanDraft).not.toHaveBeenCalled();
  });

  it.each([
    ["confidence-not-measured", "Not measured"],
    ["no-rescan-yet", "No re-scan yet"],
    ["confidence-unreadable", "Could not read this date's confidence map"],
    ["no-site-frame", "This site has no frame to place weak cells by"],
  ])("says what a 409 %s means", async (code, title) => {
    requestWeakAreas.mockRejectedValueOnce(
      new ApiError(409, { status: 409, title: "Conflict", detail: "nope", code }, "409"),
    );
    await rescanWeak(null, site, EPOCH_ID);
    expect(toasts()).toEqual([title]);
    expect(useMission.getState().composer).toBeNull();
    expect(startPlanDraft).not.toHaveBeenCalled();
  });

  it("says when the API failed otherwise", async () => {
    requestWeakAreas.mockRejectedValueOnce(new Error("offline"));
    await rescanWeak(null, site, EPOCH_ID);
    expect(toasts()).toEqual(["Could not find the weak areas"]);
  });

  it("adopts nothing when the view moved to another project while the API answered", async () => {
    let answer: (value: WeakAreas) => void = () => undefined;
    requestWeakAreas.mockImplementationOnce(
      () => new Promise<WeakAreas>((resolve) => (answer = resolve)),
    );
    const planning = rescanWeak(null, site, EPOCH_ID);
    const other = siteProject({ ...site, id: "other", slug: "other", name: "Other" });
    useMission.getState().setProject(other);
    answer(read([area()]));
    await planning;
    expect(useMission.getState().project?.id).toBe(other.id);
    expect(useMission.getState().composer).toBeNull();
    expect(startPlanDraft).not.toHaveBeenCalled();
  });
});
