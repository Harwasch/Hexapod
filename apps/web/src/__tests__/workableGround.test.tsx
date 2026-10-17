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
  useLeftOutHatch,
  workableAttribution,
  workableOfferText,
  workableTargets,
  workableTotals,
  workableZone,
} from "@/features/mission/workableGround";
import { capabilityOf, type Rehearsal } from "@/missions/rehearsal";
import type { Project, Zone } from "@/missions/types";
import { useMission } from "@/state/mission";
import { useWorkable } from "@/state/workable";

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
    workableM2: 262.8 * ACRE,
    removed: {
      steepM2: 136.6 * ACRE,
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
    expect(kept).toEqual([{ label: "Perimeter fence", hands: 1 }]);
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
      "−137 ac too steep, −14 ac creek buffer, −1.2 ac too small to work → 263 ac",
    );
    expect(formatAcres(4.24 * ACRE)).toBe("4.2 ac");
    expect(formatAcres(1263.4 * ACRE)).toBe("1,263 ac");
    const none = workableTotals([
      answer({
        removed: { steepM2: 0, creekBufferM2: 0, sliversM2: 0, slivers: 0, noDataM2: 0 },
      }),
    ]);
    expect(workableOfferText(none)).toBe("Nothing to leave out: all 263 ac");
  });

  it("attributes every source and assumption, within a zone's 300 characters", () => {
    const line = workableAttribution(answer(), "West grassland (Z-14)");
    expect(line).toBe(
      "Workable ground for a robotic brush mower: slope ≤ 25 % (assumed) on USGS 3DEP 1/3″ DEM, published 2025-08-27, 15 m creek buffer (assumed) on USGS NHD, from West grassland (Z-14)",
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
      "Derived: the part of West grassland a robotic brush mower is assumed to work, 137 ac over 25 % slope and 14 ac within 15 m of creeks left out. Not measured on the ground.",
    );
    expect(
      workableZone(answer({ workable: null }), { zone: zone("Z-14"), label: "Z-14" }, "A-03"),
    ).toBeNull();
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
