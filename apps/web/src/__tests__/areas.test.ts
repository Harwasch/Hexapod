import { describe, expect, it } from "vitest";

import { footprintAreaM2 } from "@twin/geo";

import { anywhereProject } from "@/missions/anywhere";
import {
  areaFromMeasurement,
  isAreaZone,
  nextAreaId,
  rectangleFootprint,
  viewAreaZone,
  viewFootprint,
  zoneFromFootprint,
} from "@/missions/areas";
import { DemoMissionProvider } from "@/missions/demo";
import { recordBodyFromDraft, zonesFromRecord } from "@/missions/planDraft";
import { useMission } from "@/state/mission";

const center = { longitude: -122.139, latitude: 47.644 };

describe("drawn areas", () => {
  it("numbers areas past the highest in use and sizes them from the footprint", () => {
    expect(nextAreaId([])).toBe("A-01");
    expect(nextAreaId([{ id: "A-03" }, { id: "Z-14" }])).toBe("A-04");
    const rect = rectangleFootprint(center, 200, 100);
    expect(footprintAreaM2(rect)).toBeCloseTo(400 * 200, -3);
    const zone = zoneFromFootprint("A-01", "Test", rect);
    expect(zone.acres).toBe(Math.round((400 * 200) / 4046.86));
    expect(isAreaZone(zone)).toBe(true);
    expect(zone.anchor.longitude).toBeCloseTo(center.longitude, 4);
  });

  it("takes half the viewport around the view centre, clamped", () => {
    const view = viewFootprint(center, 2, { width: 1440, height: 900 });
    expect(footprintAreaM2(view)).toBeCloseTo(1440 * 900, -4);
    const tiny = viewFootprint(center, 0.01, { width: 1440, height: 900 });
    expect(footprintAreaM2(tiny)).toBeCloseTo(120 * 120, -2);
  });

  it("turns a finished area measurement into a zone", () => {
    const zone = areaFromMeasurement(
      {
        id: "m1",
        mode: "area",
        complete: true,
        createdAt: 1,
        points: [
          { longitude: -122.14, latitude: 47.64, height: 0 },
          { longitude: -122.13, latitude: 47.64, height: 0 },
          { longitude: -122.13, latitude: 47.65, height: 0 },
        ],
      },
      "A-02",
      "Drawn",
    );
    expect(zone?.footprint.type).toBe("Polygon");
    expect(zone?.acres).toBeGreaterThan(0);
    // A double-click repeats the last corner; it is dropped rather than making a bad ring.
    const doubled = areaFromMeasurement(
      {
        id: "m3",
        mode: "area",
        complete: true,
        createdAt: 1,
        points: [
          { longitude: -122.14, latitude: 47.64, height: 0 },
          { longitude: -122.13, latitude: 47.64, height: 0 },
          { longitude: -122.13, latitude: 47.65, height: 0 },
          { longitude: -122.13, latitude: 47.65, height: 0 },
        ],
      },
      "A-04",
      "d",
    );
    expect(doubled?.footprint.type === "Polygon" && doubled.footprint.coordinates[0]?.length).toBe(
      4,
    );
    expect(
      areaFromMeasurement(
        {
          id: "m4",
          mode: "area",
          complete: true,
          createdAt: 1,
          points: [
            { longitude: -122.14, latitude: 47.64, height: 0 },
            { longitude: -122.13, latitude: 47.64, height: 0 },
            { longitude: -122.13, latitude: 47.64, height: 0 },
          ],
        },
        "A-05",
        "e",
      ),
    ).toBeNull();
    expect(
      areaFromMeasurement(
        { id: "m2", mode: "distance", complete: true, createdAt: 1, points: [] },
        "A-03",
        "x",
      ),
    ).toBeNull();
  });

  it("composes drawn areas into the project and round-trips them through plan records", () => {
    const store = useMission.getState();
    store.setProject(anywhereProject());
    const zone = zoneFromFootprint("A-01", "View area 01", rectangleFootprint(center, 100, 100));
    store.addArea("anywhere", zone);
    expect(useMission.getState().project?.zones.map((z) => z.id)).toEqual(["A-01"]);
    const body = recordBodyFromDraft(
      {
        title: "t",
        objective: "o",
        zoneIds: ["A-01"],
        machineIds: [],
        cadence: "once",
        startDate: "2026-09-17",
        endDate: null,
        estimates: { acres: 5, machineHours: 3, calendarDays: 1 },
        steps: [],
        assumptions: [],
        risks: [],
        questions: [],
        clarifications: [],
        source: "rules",
        model: null,
        note: "",
      },
      "goal",
      "anywhere",
      null,
      useMission.getState().project?.zones ?? [],
    );
    expect(body.areas?.map((a) => a.id)).toEqual(["A-01"]);
    const record = {
      ...body,
      id: "p",
      siteId: null,
      status: "scheduled" as const,
      revision: 1,
      revisions: [],
      createdAt: "",
      updatedAt: "",
      areas: body.areas ?? [],
      zoneIds: body.zoneIds ?? [],
      machineIds: body.machineIds ?? [],
      steps: body.steps ?? [],
      assumptions: body.assumptions ?? [],
      risks: body.risks ?? [],
      questions: body.questions ?? [],
      cadence: body.cadence ?? ("once" as const),
      source: body.source ?? ("rules" as const),
      model: body.model ?? null,
      endDate: body.endDate ?? null,
    };
    expect(zonesFromRecord(record).map((z) => z.name)).toEqual(["View area 01"]);
    store.removeArea("anywhere", "A-01");
    expect(useMission.getState().project?.zones).toEqual([]);
    // Demo zones are kept in front of drawn areas.
    const demo = new DemoMissionProvider().projectForSite("builtin-demo", null)!;
    store.setProject(demo);
    store.addArea(demo.id, zone);
    const ids = useMission.getState().project?.zones.map((z) => z.id) ?? [];
    expect(ids[0]).toBe("Z-14");
    expect(ids.at(-1)).toBe("A-01");
    store.removeArea(demo.id, "A-01");
  });

  it("keeps a locally shaped view area when the stored copy comes back from the plans", () => {
    const store = useMission.getState();
    const project = anywhereProject();
    store.setProject(project);
    const origin = {
      center: { longitude: -119.9, latitude: 36.6 },
      metersPerPixel: 1,
      width: 1000,
      height: 600,
      fraction: 0.5,
      dx: 0,
      dy: 0,
    };
    store.addArea(project.id, viewAreaZone("A-01", "View area 01", origin));
    const stored = zoneFromFootprint(
      "A-01",
      "View area 01",
      rectangleFootprint(origin.center, 20, 20),
    );
    const other = zoneFromFootprint(
      "A-02",
      "View area 02",
      rectangleFootprint(origin.center, 20, 20),
    );
    store.mergeAreas(project.id, [stored, other]);
    const zones = useMission.getState().project?.zones ?? [];
    expect(zones.find((z) => z.id === "A-01")?.view?.fraction).toBe(0.5);
    expect(zones.map((z) => z.id)).toEqual(["A-01", "A-02"]);
    store.removeArea(project.id, "A-01");
    store.removeArea(project.id, "A-02");
  });
});
