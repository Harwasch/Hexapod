import { beforeEach, describe, expect, it, vi } from "vitest";

import { destination } from "@twin/geo";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import type * as PlanDrafting from "@/features/mission/planDrafting";
import { zoneFromFootprint } from "@/missions/areas";
import { demoRanchProject } from "@/missions/demo";
import { landProject, landZones } from "@/missions/landGround";
import type * as Osm from "@/missions/osm";
import type { Zone } from "@/missions/types";
import { useMission } from "@/state/mission";

import { ground, land, LAND_ID, origin } from "./landGroundFixture";

const startPlanDraft = vi.fn((..._args: unknown[]) => Promise.resolve());
const fetchOsmAreas = vi.fn(() => Promise.resolve([]));
const fetchOsmContaining = vi.fn(() => Promise.resolve(null));

vi.mock("@/features/mission/planDrafting", async (importOriginal) => ({
  ...(await importOriginal<typeof PlanDrafting>()),
  startPlanDraft: (...args: unknown[]) => startPlanDraft(...args),
}));
vi.mock("@/missions/osm", async (importOriginal) => ({
  ...(await importOriginal<typeof Osm>()),
  fetchOsmAreas: () => fetchOsmAreas(),
  fetchOsmContaining: () => fetchOsmContaining(),
}));

const { outlineAt, planFromText } = await import("@/features/mission/planFlow");

/** A square `size` m across, `east` and `north` m from the fixture's origin. */
function squareAt(east: number, north: number, size: number): Zone["footprint"] {
  const at = (e: number, n: number): [number, number] => {
    const p = destination(destination(origin, 90, e), 0, n);
    return [p.longitude, p.latitude];
  };
  return {
    type: "Polygon",
    coordinates: [
      [
        at(east, north),
        at(east + size, north),
        at(east + size, north + size),
        at(east, north + size),
        at(east, north),
      ],
    ],
  };
}

/** Statuses the composer passed through, to prove it never waited for a click. */
function watchStatuses(): string[] {
  const seen: string[] = [];
  useMission.subscribe((s) => {
    if (s.composer) seen.push(s.composer.status);
  });
  return seen;
}

describe("planning on a saved land", () => {
  beforeEach(() => {
    startPlanDraft.mockClear();
    fetchOsmAreas.mockClear();
    fetchOsmContaining.mockClear();
    useMission.setState({ composer: null, landGround: null });
    useMission.getState().setProject(landProject(land, ground));
  });

  it("takes the feature the goal names, never a guess", async () => {
    const statuses = watchStatuses();
    const reply = await planFromText("monitor the fence every week", null);
    const composer = useMission.getState().composer;
    expect(composer?.zoneIds).toEqual(["L-aaaa0001"]);
    expect(composer?.ground?.source).toBe("land");
    expect(composer?.ground?.note).toContain("North fence");
    expect(composer?.ground?.note).toContain("1.0 km line");
    expect(reply).toMatch(/^On Oak Bench:/);
    expect(startPlanDraft).toHaveBeenCalledWith("monitor the fence every week", {
      zoneIds: ["L-aaaa0001"],
    });
    expect(statuses).not.toContain("awaiting-ground");
    expect(fetchOsmAreas).not.toHaveBeenCalled();
  });

  it("takes the land's boundary when the goal names none of its features", async () => {
    await planFromText("graze it lightly", null);
    const composer = useMission.getState().composer;
    expect(composer?.zoneIds).toEqual(["L-0f1e2d3c"]);
    expect(composer?.ground?.note).toBe("Oak Bench, 247.1 ac, saved boundary revision 2.");
    expect(fetchOsmAreas).not.toHaveBeenCalled();
  });

  it("swaps a card seeded with the boundary for the feature the goal names", async () => {
    useMission.getState().openComposer({ zoneIds: ["L-0f1e2d3c"] });
    await planFromText("check the stock pond", null);
    expect(useMission.getState().composer?.zoneIds).toEqual(["L-aaaa0002"]);
    useMission.getState().closeComposer();
    useMission.getState().openComposer({ zoneIds: ["L-0f1e2d3c"] });
    await planFromText("monitor every week", null);
    expect(useMission.getState().composer?.zoneIds).toEqual(["L-0f1e2d3c"]);
  });

  it("takes the land zone under a click instead of outlining", async () => {
    useMission.getState().openComposer();
    const pond = destination(destination(origin, 90, 420), 0, 420);
    await outlineAt({ ...pond, x: 0.5, y: 0.5 }, "check the water", {} as CesiumSceneManager);
    const composer = useMission.getState().composer;
    expect(composer?.zoneIds).toEqual(["L-aaaa0002"]);
    expect(composer?.ground?.source).toBe("land");
    expect(fetchOsmContaining).not.toHaveBeenCalled();
    expect(startPlanDraft).toHaveBeenCalledWith("check the water", { zoneIds: ["L-aaaa0002"] });
  });

  /** An area derived from the Land, as a workable redraft leaves it after its card closes. */
  function leftover(name: string, from: Zone["footprint"]): Zone {
    return {
      ...zoneFromFootprint("A-01", name, from),
      private: true,
      attribution: "Workable ground for a robotic brush mower: slope ≤ 25 % (assumed)",
      land: { kind: "derived", landId: LAND_ID, featureId: null, boundaryRevision: 2 },
    };
  }

  it("never takes an area derived from the land as one of its saved records", async () => {
    // Smaller than the pond it was cut from, so it would win a smallest-zone pick.
    const project = useMission.getState().project!;
    useMission
      .getState()
      .addArea(project.id, leftover("Stock pond, workable", squareAt(410, 410, 20)));
    const reply = await planFromText("survey the stock pond every month", null);
    let composer = useMission.getState().composer;
    expect(composer?.zoneIds).toEqual(["L-aaaa0002"]);
    expect(composer?.ground?.note).not.toContain("workable");
    expect(reply).not.toContain("workable");
    // A click inside the leftover area takes the Land's record under it.
    useMission.getState().closeComposer();
    useMission.getState().openComposer();
    const inside = destination(destination(origin, 90, 420), 0, 420);
    await outlineAt({ ...inside, x: 0.5, y: 0.5 }, "check the water", {} as CesiumSceneManager);
    composer = useMission.getState().composer;
    expect(composer?.zoneIds).toEqual(["L-aaaa0002"]);
    expect(composer?.ground?.note).not.toContain("workable");
  });

  it("keeps a fleet's zone a goal names over a leftover area derived from the land", async () => {
    const demo = demoRanchProject();
    useMission.getState().setProject(demo);
    useMission.getState().engageLand({ landId: LAND_ID, zones: landZones(ground) });
    const west = demo.zones.find((z) => z.id === "Z-14")!;
    useMission.getState().addArea(demo.id, leftover("West grassland, workable", west.footprint));
    const reply = await planFromText("Survey the west grassland every month", null);
    const composer = useMission.getState().composer;
    expect(composer?.zoneIds).toEqual(["Z-14"]);
    expect(composer?.ground?.source).not.toBe("land");
    expect(reply).not.toMatch(/inventory feature/);
  });
});
