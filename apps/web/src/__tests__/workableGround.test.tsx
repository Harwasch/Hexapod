/**
 * "Work only the ground it can" (`features/mission/workableGround.ts`): which zones it reworks
 * and to whose limits, the words and the zone it makes, and the hatch while the card is open.
 */

import { render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { WorkableGround } from "@twin/contracts";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { SceneContext, SceneRegistry } from "@/cesium/SceneContext";
import {
  formatAcres,
  leftOutGround,
  redraftOnWorkable,
  useLeftOutHatch,
  workableAttribution,
  workableOfferText,
  workableSourcesText,
  workableTargets,
  workableTotals,
  workableZone,
} from "@/features/mission/workableGround";
import type * as PlanDraftingModule from "@/features/mission/planDrafting";
import { setAsideZones, startPlanDraft } from "@/features/mission/planDrafting";
import { capabilityOf, type Rehearsal } from "@/missions/rehearsal";
import type { Project, Zone } from "@/missions/types";
import { useMission } from "@/state/mission";
import { useWorkable } from "@/state/workable";

vi.mock("@/features/mission/planDrafting", async (original) => ({
  ...(await original<typeof PlanDraftingModule>()),
  startPlanDraft: vi.fn(),
}));

const ACRE = 4046.8564224;
const square = {
  type: "Polygon" as const,
  coordinates: [
    [
      [-120.05, 34.69],
      [-120.04, 34.69],
      [-120.04, 34.7],
      [-120.05, 34.7],
      [-120.05, 34.69],
    ],
  ],
};

function zone(id: string, extra: Partial<Zone> = {}): Zone {
  return {
    id,
    name: "West grassland",
    short: id,
    footprint: square,
    tone: "amber",
    treated: false,
    acres: 400,
    task: "Mow",
    machines: "",
    progressPct: 0,
    note: "",
    anchor: { longitude: -120.045, latitude: 34.695 },
    ...extra,
  };
}

function answer(extra: Partial<WorkableGround> = {}): WorkableGround {
  return {
    landId: "land-1",
    boundaryRevision: 2,
    workable: { type: "MultiPolygon", coordinates: [square.coordinates] },
    footprintM2: 414 * ACRE,
    workableM2: 225.2 * ACRE,
    removed: {
      steepM2: 136.6 * ACRE,
      steepEdgeM2: 37.6 * ACRE,
      creekBufferM2: 13.9 * ACRE,
      sliversM2: 1.19 * ACRE,
      slivers: 49,
      noDataM2: 0,
    },
    leftOut: [],
    limits: {
      machineClass: "mower",
      label: "Robotic brush mower (assumed)",
      maxSlopePct: 25,
      creekBufferM: 15,
    },
    elevation: {
      rasterId: "1108dcf3-68ff-557a-8d9d-9d99c0d4676d",
      dataset: "usgs-3dep-13",
      label: "USGS 3DEP 1/3″ bare-earth DEM, 10 m",
      published: "2025-08-27",
      resolutionM: 10,
      edgeM: 5,
      stale: false,
      fallback: false,
    },
    creeks: {
      status: "available",
      label: "USGS National Hydrography Dataset (NHD) flowlines, 1:24,000",
      counted: 22,
      ephemeralIncluded: false,
      message: "",
    },
    sources: [],
    assumptions: [],
    warnings: [],
    ...extra,
  };
}

describe("which zones are reworked, and to whose limits", () => {
  const hand = (zoneId: string, zoneLabel: string) => ({
    id: `${zoneId}:0:cross-slope`,
    zoneId,
    zoneLabel,
  });
  const rehearsal = {
    hands: [hand("Z-14", "Z-14"), hand("Z-14", "Z-14"), hand("L-fence", "Perimeter fence")],
    lanes: [
      { zoneId: "Z-14", machineId: "TR-04", removable: true },
      { zoneId: "Z-14", machineId: "SD-01", removable: true },
      { zoneId: "L-fence", machineId: "TR-04", removable: false },
    ],
    machines: [
      { id: "TR-04", capability: capabilityOf("mower", "its model") },
      { id: "SD-01", capability: capabilityOf("seeder", "its model") },
    ],
  } as unknown as Rehearsal;

  it("takes the strictest class on a zone, and leaves a followed line's hands", () => {
    const { targets, kept } = workableTargets(rehearsal, [zone("Z-14"), zone("L-fence")]);
    // The seed drill's assumed 20 % is stricter than the mower's 25 %.
    expect(targets.map((t) => [t.zoneId, t.machineClass, t.hands])).toEqual([
      ["Z-14", "seeder", 2],
    ]);
    expect(kept).toEqual([{ zoneId: "L-fence", label: "Perimeter fence", hands: 1, why: "line" }]);
  });

  it("counts the hands of lanes left out of the preview, which the plan still drives", () => {
    const leftOut = [
      { zoneId: "Z-14", label: "Z-14", hands: 3 },
      // Z-08's only hands are on a lane left out; its other lanes still say who drives it.
      { zoneId: "Z-08", label: "Z-08", hands: 2 },
      // Every lane of Z-30 is left out: no lane says which class drives it.
      { zoneId: "Z-30", label: "Z-30", hands: 4 },
    ];
    const withZ08 = {
      ...rehearsal,
      lanes: [...rehearsal.lanes, { zoneId: "Z-08", machineId: "TR-04", removable: true }],
    } as unknown as Rehearsal;
    const { targets, kept } = workableTargets(
      withZ08,
      [zone("Z-14"), zone("L-fence"), zone("Z-08"), zone("Z-30")],
      leftOut,
    );
    expect(targets.map((t) => [t.zoneId, t.machineClass, t.hands])).toEqual([
      ["Z-14", "seeder", 5],
      ["Z-08", "mower", 2],
    ]);
    expect(kept).toEqual([
      { zoneId: "L-fence", label: "Perimeter fence", hands: 1, why: "line" },
      { zoneId: "Z-30", label: "Z-30", hands: 4, why: "left-out" },
    ]);
  });

  it("does not rework what only flies", () => {
    const flown = {
      ...rehearsal,
      lanes: [{ zoneId: "Z-14", machineId: "DR-01", removable: true }],
      machines: [{ id: "DR-01", capability: capabilityOf("drone", "its model") }],
      hands: [hand("Z-14", "Z-14")],
    } as unknown as Rehearsal;
    expect(workableTargets(flown, [zone("Z-14")]).targets).toEqual([]);
  });
});

describe("the words and the zone", () => {
  it("says what it cuts and what is left, every part that rounds to something", () => {
    const totals = workableTotals([answer()]);
    expect(workableOfferText(totals)).toBe(
      "−137 ac too steep, −38 ac steep edge, −14 ac creek buffer, −1.2 ac too small to work → 225 ac",
    );
    expect(formatAcres(4.24 * ACRE)).toBe("4.2 ac");
    expect(formatAcres(1263.4 * ACRE)).toBe("1,263 ac");
    const none = workableTotals([
      answer({
        removed: {
          steepM2: 0,
          steepEdgeM2: 0,
          creekBufferM2: 0,
          sliversM2: 0,
          slivers: 0,
          noDataM2: 0,
        },
      }),
    ]);
    expect(workableOfferText(none)).toBe("Nothing to leave out: all 225 ac");
  });

  it("attributes every source and assumption, within a zone's 300 characters", () => {
    const line = workableAttribution(answer(), "West grassland (Z-14)");
    expect(line).toBe(
      "Workable ground for a robotic brush mower: slope ≤ 25 % (assumed), 5 m clear of steeper cells, on USGS 3DEP 1/3″ DEM, published 2025-08-27, 15 m creek buffer (assumed) on USGS NHD, from West grassland (Z-14)",
    );
    const surface = workableAttribution(
      answer({
        elevation: { ...answer().elevation, dataset: "cop-dem-glo-30", published: null },
        creeks: { ...answer().creeks, status: "unavailable" },
      }),
      "x".repeat(400),
    );
    expect(surface).toContain("on Copernicus GLO-30 surface model (incl. canopy)");
    expect(surface).toContain("creek buffer not applied (USGS NHD not read)");
    expect(surface.length).toBe(300);
  });

  it("names the creeks as a source only when NHD was read for every answer", () => {
    expect(
      workableSourcesText([answer(), answer({ creeks: { ...answer().creeks, status: "empty" } })]),
    ).toBe(
      "Derived from USGS 3DEP 1/3″ bare-earth DEM, 10 m and USGS NHD creeks; the limits are assumed.",
    );
    const unread = workableSourcesText([
      answer(),
      answer({ creeks: { ...answer().creeks, status: "unavailable" } }),
    ]);
    expect(unread).toBe(
      "Derived from USGS 3DEP 1/3″ bare-earth DEM, 10 m; creek buffer not applied (USGS NHD not read). The limits are assumed.",
    );
    expect(unread).not.toContain("NHD creeks");
  });

  it("hatches every left-out reason but slivers, which are counted, not drawn", () => {
    const piece = (reason: WorkableGround["leftOut"][number]["reason"]) => ({
      reason,
      label: `Left out: ${reason}`,
      areaM2: 100,
      geometry: { type: "MultiPolygon" as const, coordinates: [square.coordinates] },
    });
    const pieces = leftOutGround(
      answer({
        leftOut: [piece("steep"), piece("steep-edge"), piece("sliver"), piece("no-data")],
      }),
    );
    expect(pieces.map((p) => p.reason)).toEqual(["steep", "steep-edge", "no-data"]);
  });

  it("makes a private, derived area of what is left, or nothing when nothing is", () => {
    const made = workableZone(answer(), { zone: zone("Z-14"), label: "Z-14" }, "A-03");
    expect(made).toMatchObject({
      id: "A-03",
      name: "West grassland, workable",
      short: "derived",
      tone: "teal",
      task: "Mow",
      private: true,
      land: { kind: "derived", landId: "land-1", featureId: null, boundaryRevision: 2 },
    });
    expect(made?.note).toBe(
      "Derived: the part of West grassland a robotic brush mower is assumed to work, 174 ac over 25 % slope or within 5 m of it and 14 ac within 15 m of creeks left out. Not measured on the ground.",
    );
    expect(
      workableZone(answer({ workable: null }), { zone: zone("Z-14"), label: "Z-14" }, "A-03"),
    ).toBeNull();
  });
});

describe("the redraft", () => {
  afterEach(() => {
    useMission.setState({ project: null, composer: null });
    useWorkable.getState().clear();
    vi.mocked(startPlanDraft).mockReset();
  });

  it("says when the creeks were not read, and keeps a zone with nothing workable whole", () => {
    const draw = zone("Z-08", { name: "Creek draw", footprint: square });
    const project = {
      id: "demo",
      zones: [zone("Z-14"), draw],
      machines: [],
      plans: [],
    } as unknown as Project;
    useMission.setState({
      project,
      composer: {
        goal: "Mow the grassland and the creek draw",
        zoneIds: ["Z-14", "Z-08"],
        machineIds: [],
        replacePlanId: null,
        status: "ready",
        ground: null,
        draft: null,
        previousDraft: null,
        answers: {},
        asked: {},
        error: null,
      },
    });
    const target = (z: Zone, label: string, hands: number) => ({
      zoneId: z.id,
      label,
      zone: z,
      machineClass: "mower" as const,
      hands,
    });
    const failure = redraftOnWorkable({
      draft: { zoneIds: ["Z-14", "Z-08"], machineIds: ["TR-04"] },
      answers: [
        {
          target: target(zone("Z-14"), "Z-14", 130),
          result: answer({ creeks: { ...answer().creeks, status: "unavailable" } }),
        },
        { target: target(draw, "Z-08", 4), result: answer({ workable: null, workableM2: 0 }) },
      ],
      handsBefore: 147,
      heightAt: () => 0,
    });
    expect(failure).toBeNull();
    const state = useMission.getState();
    // Z-14's place goes to its workable area; Z-08 has none, so it stays as it was.
    expect(state.composer?.zoneIds).toEqual(["A-L01", "Z-08"]);
    expect(state.project?.zones.map((z) => z.id)).toEqual(["Z-14", "Z-08", "A-L01"]);
    expect(state.composer?.setAside).toEqual({ "Z-14": "A-L01" });
    const note = state.composer?.ground?.note ?? "";
    expect(note).toContain("creek buffer not applied (USGS NHD not read)");
    expect(note).not.toContain("NHD creeks");
    expect(note).toContain(
      "Creek draw (Z-08) has no ground its machines can work within the assumed limits, so it stays as it was, with its 4 hands.",
    );
    expect(state.log.at(-1)?.text).toBe(note);
    // "Down from" counts the reworked zone's hands; the total is what Approve counted.
    expect(useWorkable.getState().rerun).toEqual({
      zoneIds: ["A-L01"],
      handsBefore: 147,
      reworkedBefore: 130,
      creeksRead: false,
      labels: ["Z-14"],
    });
    expect(startPlanDraft).toHaveBeenCalledWith("Mow the grassland and the creek draw", {
      zoneIds: ["A-L01", "Z-08"],
      machineIds: ["TR-04"],
      reshaped: true,
    });
  });
});

describe("zones set aside for their derived ground", () => {
  it("stays out while its replacement, or the replacement's own, is in scope", () => {
    // The fix applied twice: Z-14 became A-01, and A-01 became A-02.
    const setAside = { "Z-14": "A-01", "A-01": "A-02" };
    expect(setAsideZones(setAside, ["A-02", "Z-21"])).toEqual(["Z-14", "A-01"]);
    expect(setAsideZones(setAside, ["A-01"])).toEqual(["Z-14"]);
    // Named in the scope itself, a zone is back.
    expect(setAsideZones(setAside, ["Z-14", "A-02"])).toEqual(["A-01"]);
    expect(setAsideZones(setAside, ["Z-21"])).toEqual([]);
    // A loop ends the walk.
    expect(setAsideZones({ "A-01": "A-02", "A-02": "A-01" }, ["Z-21"])).toEqual([]);
  });
});

describe("the hatch while the card is open", () => {
  afterEach(() => {
    useMission.setState({ composer: null });
    useWorkable.getState().clear();
  });

  function Hatch({ project }: { project: Project }) {
    useLeftOutHatch(project);
    return null;
  }

  it("hatches the open card's workable areas, and forgets them when the card closes", () => {
    const showLeftOut = vi.fn();
    const registry = new SceneRegistry();
    registry.set({ rehearsal: { showLeftOut } } as unknown as CesiumSceneManager);
    const project = { zones: [zone("A-01")] } as unknown as Project;
    const piece = { reason: "steep" as const, label: "Left out: too steep", footprint: square };
    const heightAt = () => 250;
    useWorkable
      .getState()
      .remember([{ zoneId: "A-01", sourceLabel: "Z-14", leftOut: [piece], heightAt }], {
        zoneIds: ["A-01"],
        handsBefore: 147,
        reworkedBefore: 147,
        creeksRead: true,
        labels: ["Z-14"],
      });
    useMission.setState({
      composer: { zoneIds: ["A-01"] } as never,
    });
    const view = render(
      <SceneContext.Provider value={registry}>
        <Hatch project={project} />
      </SceneContext.Provider>,
    );
    expect(showLeftOut).toHaveBeenLastCalledWith([piece], heightAt);
    // The card closes: the hatch goes, and so does what it remembered.
    useMission.setState({ composer: null });
    view.unmount();
    expect(showLeftOut).toHaveBeenLastCalledWith([]);
    expect(useWorkable.getState().records).toEqual({});
  });
});
