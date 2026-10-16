import { describe, expect, it } from "vitest";

import type { Footprint } from "@twin/geo";
import { footprintAreaM2, footprintContains, haversineDistance, toAcres } from "@twin/geo";

import {
  BUILTIN_SCAN_SITE_ID,
  builtinRanchSite,
  builtinSites,
  SCAN_SHOWCASE_SLUG,
} from "@/api/fallback";
import {
  assumedAcresPerHour,
  DEMO_PROJECT,
  DemoMissionProvider,
  demoRanchArrival,
  demoRanchFraming,
  demoRanchProject,
} from "@/missions/demo";
import {
  BUILTIN_RANCH_ID,
  DEMO_RANCH,
  DEMO_RANCH_SLUG,
  readDemoRanch,
} from "@/missions/demoRanchData";
import {
  eyeOf,
  frameGround,
  framingContains,
  HUD_INSETS,
  projectGroundPoints,
} from "@/missions/framing";
import { parseHours } from "@/missions/rates";
import type { LonLat } from "@/missions/types";

import raw from "../../../api/app/seed/demo_ranch.json";

const zone = (id: string) => DEMO_PROJECT.zones.find((z) => z.id === id)!;
const machine = (id: string) => DEMO_PROJECT.machines.find((m) => m.id === id)!;
const plan = (id: string) => DEMO_PROJECT.plans.find((p) => p.id === id)!;
const acres = (id: "Z-14" | "Z-21" | "Z-08") =>
  Math.round(toAcres(DEMO_RANCH.derived.zones[id].areaM2));

/** Inside the outer ring and outside every hole: where a machine can stand. */
function onGround(footprint: Footprint, point: LonLat): boolean {
  if (!footprintContains(footprint, point)) return false;
  if (footprint.type !== "Polygon") return true;
  return footprint.coordinates
    .slice(1)
    .every((hole) => !footprintContains({ type: "Polygon", coordinates: [hole] }, point));
}

describe("the demo ranch file", () => {
  it("names the field it cannot read", () => {
    expect(readDemoRanch(raw).name).toBe(DEMO_RANCH.name);
    expect(() => readDemoRanch({ ...raw, slug: 3 })).toThrow(/slug/);
    expect(() => readDemoRanch({ ...raw, camera: { ...raw.camera, target: [1] } })).toThrow(
      /camera\.target/,
    );
    expect(() => readDemoRanch({ ...raw, derived: { ...raw.derived, zones: {} } })).toThrow(
      /derived\.zones\.Z-14/,
    );
  });

  it("keeps every zone inside the outline", () => {
    for (const z of DEMO_PROJECT.zones) {
      const ring = z.footprint.type === "Polygon" ? z.footprint.coordinates[0]! : [];
      expect(ring.length).toBeGreaterThan(3);
      for (const [longitude, latitude] of ring) {
        // The derived outlines are clipped to the boundary and rounded to 1e-7° (a centimetre):
        // a vertex on the edge may read as a hair outside, so ask a few decimetres inwards.
        const { centroid } = DEMO_RANCH.derived;
        const towards = {
          longitude: longitude! + (centroid.longitude - longitude!) * 2e-4,
          latitude: latitude! + (centroid.latitude - latitude!) * 2e-4,
        };
        expect(footprintContains(DEMO_RANCH.boundary.geometry, towards), `${z.id} vertex`).toBe(
          true,
        );
      }
    }
  });
});

describe("the demo project's numbers", () => {
  it("are the measured footprints' acres", () => {
    for (const id of ["Z-14", "Z-21", "Z-08"] as const) {
      const derived = DEMO_RANCH.derived.zones[id];
      expect(zone(id).acres).toBe(Math.round(toAcres(derived.areaM2)));
      // A stale `derived` (the outline edited, the build not run) is caught here.
      expect(Math.abs(footprintAreaM2(zone(id).footprint) / derived.areaM2 - 1)).toBeLessThan(
        0.005,
      );
    }
    const site = Math.round(toAcres(DEMO_RANCH.derived.siteAreaM2));
    expect(DEMO_PROJECT.zones.reduce((sum, z) => sum + z.acres, 0)).toBeLessThanOrEqual(site);
  });

  it("make each plan's acres the sum of its zones'", () => {
    const thistle = acres("Z-14") + acres("Z-21");
    expect(plan("thistle").meta).toContain(`${thistle.toLocaleString()} acres`);
    expect(plan("seed").meta).toContain(`${thistle.toLocaleString()} acres`);
    expect(plan("seed").stats[0]?.value).toBe(String(thistle));
    const site = Math.round(toAcres(DEMO_RANCH.derived.siteAreaM2));
    expect(plan("survey3d").meta).toContain(`${site.toLocaleString()} acres`);
    // The project's own words carry no number: the badge adds the site's area.
    expect(DEMO_PROJECT.meta).not.toMatch(/\d/);
  });

  it("speak of this place: no lake, no pipeline, no highway", () => {
    const words = JSON.stringify({
      plans: DEMO_PROJECT.plans,
      agent: DEMO_PROJECT.agent,
      zones: DEMO_PROJECT.zones.map((z) => [z.name, z.note, z.task]),
      machines: DEMO_PROJECT.machines.map((m) => m.zoneLabel),
      fleetNote: DEMO_PROJECT.fleetNote,
      log: DEMO_PROJECT.workLog.map((r) => r.zone),
    });
    expect(words).not.toMatch(/lake|pipeline|highway|Z-03|bench|raptor/i);
    expect(plan("creek").title).toMatch(/creek/i);
    expect(plan("creek").meta).toContain("Figueroa Creek");
    expect(DEMO_PROJECT.plans.map((p) => p.id)).toEqual([
      "thistle",
      "creek",
      "survey3d",
      "seed",
      "fuelbreak",
    ]);
  });

  it("stands every machine on the ranch, the working ones in their zones", () => {
    for (const m of DEMO_PROJECT.machines) {
      expect(footprintContains(DEMO_RANCH.boundary.geometry, m.position), m.id).toBe(true);
    }
    expect(onGround(zone("Z-14").footprint, machine("TR-04").position)).toBe(true);
    expect(onGround(zone("Z-21").footprint, machine("TR-12").position)).toBe(true);
    expect(onGround(zone("Z-08").footprint, machine("TR-18").position)).toBe(true);
    // TR-04's track ends where it stands.
    expect(machine("TR-04").track?.at(-1)).toEqual(machine("TR-04").position);
    // Two markers at Base 1 would stack: TR-07 holds 60 m off it.
    expect(haversineDistance(machine("TR-07").position, machine("TR-15").position)).toBeCloseTo(
      60,
      0,
    );
  });

  it("logs shifts at the assumed machine rates, never more than a zone holds", () => {
    const mow = assumedAcresPerHour("mow");
    // A 2 m mower at 1.4 m/s, cutting 80% of the time: about two acres an hour.
    expect(mow).toBeGreaterThan(1.9);
    expect(mow).toBeLessThan(2.1);
    for (const row of DEMO_PROJECT.workLog) {
      const hours = parseHours(row.hours)!;
      if (row.task.startsWith("Mow")) expect(Math.abs(row.acres! - hours * mow)).toBeLessThan(1);
    }
    for (const id of ["Z-14", "Z-21", "Z-08"] as const) {
      const logged = DEMO_PROJECT.workLog
        .filter((row) => row.zone.startsWith(id) && /pass 1|Transect/.test(row.task))
        .reduce((sum, row) => sum + (row.acres ?? 0), 0);
      expect(logged).toBeLessThanOrEqual(acres(id));
    }
    // Z-21's rows are all of its 12%; the transect is Z-08's 34%.
    const z21 = DEMO_PROJECT.workLog.filter((row) => row.zone.startsWith("Z-21"));
    expect(z21.reduce((sum, row) => sum + (row.acres ?? 0), 0)).toBe(
      Math.round(0.12 * acres("Z-21")),
    );
    expect(DEMO_PROJECT.workLog.find((row) => row.task === "Transect survey")?.acres).toBe(
      Math.round(0.34 * acres("Z-08")),
    );
  });

  it("counts the fleet stats from the fleet and the log", () => {
    const stat = (label: string) => DEMO_PROJECT.fleetStats.find((s) => s.label === label)?.value;
    const last = DEMO_PROJECT.workLog
      .map((r) => r.date)
      .sort()
      .at(-1);
    const lastShift = DEMO_PROJECT.workLog
      .filter((r) => r.date === last)
      .reduce((sum, r) => sum + (r.acres ?? 0), 0);
    expect(stat("Treated, last shift")).toBe(`${lastShift} ac`);
    expect(stat("Working now")).toBe(
      String(DEMO_PROJECT.machines.filter((m) => m.status !== "idle").length),
    );
    expect(stat("Need attention")).toBe(
      String(DEMO_PROJECT.machines.filter((m) => m.status === "attention").length),
    );
  });

  it("credits every zone outline's source", () => {
    expect(zone("Z-14").attribution).toContain("ESA WorldCover");
    expect(zone("Z-08").attribution).toContain("Figueroa Creek");
    expect(zone("Z-21").attribution).toContain("demo outline");
  });
});

describe("DemoMissionProvider", () => {
  const provider = new DemoMissionProvider();

  it("attaches the fleet to the ranch and nothing else", () => {
    expect(provider.projectForSite("some-uuid", DEMO_RANCH_SLUG)?.siteId).toBe("some-uuid");
    expect(provider.projectForSite(BUILTIN_RANCH_ID, null)?.id).toBe("demo-ranch");
    expect(provider.projectForSite("x", SCAN_SHOWCASE_SLUG)).toBeNull();
    expect(provider.projectForSite(BUILTIN_SCAN_SITE_ID, null)).toBeNull();
    expect(provider.projectForSite("builtin-x", null)).toBeNull();
    expect(demoRanchProject().siteId).toBe(BUILTIN_RANCH_ID);
  });
});

describe("the built-in ranch", () => {
  it("is the seeded site's twin: same area, same Overview", () => {
    const site = builtinRanchSite();
    expect(site.areaM2).toBe(DEMO_RANCH.derived.siteAreaM2);
    expect(site.assets).toEqual([]);
    const [bookmark] = site.cameraBookmarks;
    const eye = eyeOf(demoRanchFraming(), DEMO_RANCH.camera.targetHeightM);
    expect(haversineDistance(bookmark!, eye)).toBeLessThan(1);
    expect(Math.abs(bookmark!.height - eye.height)).toBeLessThan(1);
    expect(site.attribution.map((a) => a.text)).toContain(DEMO_RANCH.sources.worldcover!.text);
    expect(builtinSites().map((s) => s.id)).toEqual([BUILTIN_RANCH_ID, BUILTIN_SCAN_SITE_ID]);
  });
});

describe("the ranch arrival", () => {
  const points = projectGroundPoints(DEMO_PROJECT);

  it.each([
    [1440, 900],
    [1280, 720],
  ])("shows every zone, chip and machine clear of the HUD at %i×%i", (width, height) => {
    const options = { aspect: width / height, insets: HUD_INSETS };
    const fit = framingContains(demoRanchFraming(), points, options);
    if (!fit.inside) {
      // A framing for both windows: fitted to each, the farther one, which holds in both.
      const { headingDeg, pitchDeg } = DEMO_RANCH.camera;
      const fresh = [1440 / 900, 1280 / 720]
        .map((aspect) =>
          frameGround(points, { aspect, insets: HUD_INSETS, headingDeg, pitchDeg, margin: 1.12 }),
        )
        .sort((a, b) => b.rangeM - a.rangeM)[0]!;
      throw new Error(
        `The Overview no longer frames the ranch at ${width}×${height} (worst point ${String(fit.worst.index)} is ${fit.worst.excess.toFixed(3)} outside). ` +
          `Paste into demo_ranch.json's camera, then reseed: ${JSON.stringify({
            target: [
              Number(fresh.target.longitude.toFixed(7)),
              Number(fresh.target.latitude.toFixed(7)),
            ],
            headingDeg,
            pitchDeg,
            rangeM: Math.ceil(fresh.rangeM / 10) * 10,
          })}`,
      );
    }
    // A desktop window keeps the authored Overview.
    expect(demoRanchArrival({ width, height, fovDeg: 60 })).toBeNull();
  });

  it("refits for a phone held upright, which the Overview would crop", () => {
    const phone = { width: 390, height: 844, fovDeg: 60 };
    const arrival = demoRanchArrival(phone);
    expect(arrival).not.toBeNull();
    // Looking the same way up the valley, more steeply down, so the ranch fills the height.
    expect(arrival!.heading).toBe(DEMO_RANCH.camera.headingDeg);
    expect(arrival!.pitch).toBeLessThan(DEMO_RANCH.camera.pitchDeg);
  });
});
