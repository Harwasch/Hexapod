/**
 * "Make this the job": the lit ground (protected patches left out) becomes private areas
 * derived from the Land, in a fresh plan composer, and a typed goal keeps them as its ground.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import type { SignatureSimilarity, SignatureZones } from "@twin/contracts";

import { api } from "@/api/client";
import type * as PlanDrafting from "@/features/mission/planDrafting";
import { isPrivateZone, landProject } from "@/missions/landGround";
import { useMission } from "@/state/mission";
import { useShowOnce } from "@/state/showOnce";

import { ground, land } from "./landGroundFixture";

const startPlanDraft = vi.fn((..._args: unknown[]) => Promise.resolve());
vi.mock("@/features/mission/planDrafting", async (importOriginal) => ({
  ...(await importOriginal<typeof PlanDrafting>()),
  startPlanDraft: (...args: unknown[]) => startPlanDraft(...args),
}));

const { LIKE_HERE_ATTRIBUTION, isLikeHereZone, likeHereZones, makeJob } =
  await import("@/features/showOnce/makeJob");
const { planFromText } = await import("@/features/mission/planFlow");

const square = (west: number, size = 0.002) => ({
  type: "Polygon" as const,
  coordinates: [
    [
      [west, 34.7],
      [west + size, 34.7],
      [west + size, 34.7 + size],
      [west, 34.7 + size],
      [west, 34.7],
    ],
  ],
});

const scenes = ["2025-01-15", "2025-04-05", "2025-07-24", "2025-10-17"].map((day, window) => ({
  window,
  period: { startDate: `${day.slice(0, 7)}-01`, endDate: `${day.slice(0, 7)}-30` },
  sceneId: `S2_${window}`,
  acquiredAt: `${day}T18:55:00Z`,
}));
const seed = {
  longitude: -120.0402,
  latitude: 34.6981,
  cellLongitude: -120.04022,
  cellLatitude: 34.69806,
  windows: [],
};

function zonesRead(): SignatureZones {
  return {
    rasterId: "raster-1",
    landId: land.id,
    boundaryRevision: land.revision,
    stale: false,
    threshold: 0.8,
    seed,
    zones: [
      { footprint: square(-120.04), areaM2: 25_200, cells: 240, meanSimilarity: 0.9, patches: 1 },
      {
        footprint: square(-120.03, 0.001),
        areaM2: 6_100,
        cells: 58,
        meanSimilarity: 0.86,
        patches: 1,
      },
    ],
    remainder: {
      footprint: { type: "MultiPolygon", coordinates: [square(-120.02, 0.0008).coordinates] },
      areaM2: 9_000,
      cells: 88,
      meanSimilarity: 0.84,
      patches: 21,
    },
    dropped: { patches: 262, areaM2: 99_530 },
    protectedCells: 31,
    matchedCells: 3356,
    cellAreaM2: 100,
    scenes,
    method: "cosine",
    interpretation: "Similar seasonal satellite signature at 10 m. Not a species ID.",
    attribution: "Contains modified Copernicus Sentinel data; Element 84 Earth Search",
  };
}

describe("like-here zones", () => {
  it("are private, derived from the land, named by size and cited", () => {
    const zones = likeHereZones(zonesRead(), land, [{ id: "A-01" }, { id: "Z-14" }]);
    expect(zones.map((z) => z.id)).toEqual(["A-02", "A-03", "A-04"]);
    expect(zones.map((z) => z.name)).toEqual([
      "Like-here patch 1 (6.2 ac)",
      "Like-here patch 2 (1.5 ac)",
      "Like-here: 21 smaller patches (2.2 ac)",
    ]);
    for (const zone of zones) {
      expect(zone.tone).toBe("amber");
      expect(isPrivateZone(zone)).toBe(true);
      expect(zone.land).toEqual({
        kind: "derived",
        landId: land.id,
        featureId: null,
        boundaryRevision: land.revision,
      });
      expect(zone.attribution?.startsWith(LIKE_HERE_ATTRIBUTION)).toBe(true);
      expect(zone.attribution?.length).toBeLessThanOrEqual(300);
      expect(zone.attribution).toContain("34.69806, -120.04022");
      expect(zone.attribution).toContain("2025-01-15, 2025-04-05, 2025-07-24, 2025-10-17");
      expect(zone.note).toContain("Not a species ID");
      expect(isLikeHereZone(zone, land.id)).toBe(true);
    }
    expect(zones[2]?.footprint.type).toBe("MultiPolygon");
  });
});

describe("Make this the job", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    startPlanDraft.mockClear();
    useMission.setState({ composer: null, landGround: null, log: [], areas: {} });
    useMission.getState().setProject(landProject(land, ground));
    const grid = {
      west: -120.05,
      south: 34.69,
      east: -120.046,
      north: 34.692,
      columns: 4,
      rows: 2,
      cellM: 10,
      cellAreaM2: 100,
      scores: Uint8Array.from([250, 240, 0, 0, 0, 230, 0, 0]),
    };
    useShowOnce.getState().stop();
    useShowOnce
      .getState()
      .start({ landId: land.id, revision: land.revision, rasterId: "raster-1" });
    useShowOnce.getState().setResult({
      read: { seed, scenes } as unknown as SignatureSimilarity,
      grid,
    });
    useShowOnce.getState().setGuarded(Uint8Array.from([0, 0, 0, 0, 0, 1, 0, 0]));
  });

  it("outlines the lit ground without the protected cells and opens a plan on it", async () => {
    const post = vi
      .spyOn(api, "POST")
      .mockResolvedValue({ data: zonesRead(), response: new Response() });
    const bar = vi.fn();
    window.addEventListener("twin:bar", bar);
    const summary = await makeJob(land, null);
    window.removeEventListener("twin:bar", bar);
    expect(post).toHaveBeenCalledWith("/api/v1/land/rasters/{raster_id}/similar/zones", {
      params: { path: { raster_id: "raster-1" } },
      body: {
        longitude: seed.longitude,
        latitude: seed.latitude,
        threshold: 0.8,
        // The one protected cell, row 1 column 1: its lon/lat rectangle.
        protected: [[-120.049, 34.69, -120.048, 34.691]].map((r) =>
          r.map((v): unknown => expect.closeTo(v, 9)),
        ),
      },
    });
    expect(summary.areas).toBe(3);
    expect(summary.acres).toBeCloseTo(40_300 / 4046.8564224, 6);
    const { composer, project, log } = useMission.getState();
    const added = project?.zones.filter((z) => isLikeHereZone(z, land.id)) ?? [];
    expect(added.map((z) => z.id)).toEqual(["A-01", "A-02", "A-03"]);
    expect(composer?.zoneIds).toEqual(["A-01", "A-02", "A-03"]);
    expect(composer?.goal).toBe("");
    expect(composer?.ground?.source).toBe("signature");
    expect(composer?.ground?.note).toContain("at least 80% alike");
    expect(composer?.ground?.note).toContain("262 specks under half an acre");
    expect(log.at(-1)?.text).toBe(
      "I outlined 3 areas (10.0 ac) that look like the spot you tapped. What should happen there?",
    );
    expect(bar).toHaveBeenCalledTimes(1);
    expect(startPlanDraft).not.toHaveBeenCalled();

    // The sentence in the bar keeps the like-here areas as the ground, never the boundary.
    await planFromText("mow these patches", null);
    expect(startPlanDraft).toHaveBeenCalledWith("mow these patches", {
      zoneIds: ["A-01", "A-02", "A-03"],
    });

    // Made again: the last job's areas, which no plan uses, are replaced.
    await makeJob(land, null);
    const again = useMission.getState().project?.zones.filter((z) => isLikeHereZone(z, land.id));
    expect(again?.map((z) => z.id)).toEqual(["A-01", "A-02", "A-03"]);
  });

  it("says when nothing at this setting is big enough, and changes nothing", async () => {
    vi.spyOn(api, "POST").mockResolvedValue({
      data: { ...zonesRead(), zones: [], remainder: null },
      response: new Response(),
    });
    const summary = await makeJob(land, null);
    expect(summary.areas).toBe(0);
    expect(useMission.getState().composer).toBeNull();
    expect(useMission.getState().project?.zones.some((z) => isLikeHereZone(z, land.id))).toBe(
      false,
    );
  });
});
