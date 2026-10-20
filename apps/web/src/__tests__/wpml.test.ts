import type { Footprint } from "@twin/contracts";
import {
  bearing,
  boundsOf,
  centerOf,
  destination,
  haversineDistance,
  laneStrokes,
} from "@twin/geo";
import { describe, expect, it } from "vitest";

import {
  DEFAULT_SCOUT,
  MAX_ALTITUDE_M,
  ORBIT_MIN_STANDOFF_M,
  ORBIT_STEP_DEG,
  ORBIT_RINGS,
  orbitRing,
  orbitStandoffM,
  reliefOutlines,
  routeStats,
  SCOUT_ALTITUDES_M,
  scoutGeometry,
  scoutLanes,
  scoutPasses,
  scoutWaylines,
  waylineStats,
  type Scout,
  type ScoutWayline,
} from "@/missions/scout";
import { buildScoutKmz, escapeXml, templateKml, waylinesWpml, WPML_NS } from "@/missions/wpml";

const origin = { longitude: -120.04, latitude: 34.698 };
const e = (m: number) => destination(origin, 90, m).longitude;
const n = (m: number) => destination(origin, 0, m).latitude;
const holed: Footprint = {
  type: "Polygon",
  coordinates: [
    [
      [e(0), n(0)],
      [e(500), n(0)],
      [e(500), n(240)],
      [e(0), n(240)],
      [e(0), n(0)],
    ],
    [
      [e(200), n(80)],
      [e(300), n(80)],
      [e(300), n(160)],
      [e(200), n(160)],
      [e(200), n(80)],
    ],
  ],
};
const second: Footprint = {
  type: "Polygon",
  coordinates: [
    [
      [e(700), n(0)],
      [e(900), n(0)],
      [e(900), n(100)],
      [e(700), n(100)],
      [e(700), n(0)],
    ],
  ],
};

function parse(xml: string): Document {
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  expect(doc.getElementsByTagName("parsererror")).toHaveLength(0);
  return doc;
}

const wpml = (node: Element | Document, name: string) =>
  Array.from(node.getElementsByTagNameNS(WPML_NS, name));
const text = (node: Element | Document, name: string) => wpml(node, name)[0]?.textContent ?? "";

describe("DJI WPML scout route", () => {
  const routes = scoutWaylines([
    { footprint: holed, scout: DEFAULT_SCOUT },
    { footprint: second, scout: DEFAULT_SCOUT },
  ]);
  const lanes = [scoutLanes(holed, DEFAULT_SCOUT), scoutLanes(second, DEFAULT_SCOUT)].flat();

  it("flies a fog scout's areas as one wayline: the lanes the map draws, in order", () => {
    expect(routes).toHaveLength(1);
    expect(routes[0]?.waypoints).toEqual(lanes.flatMap((lane) => [lane.a, lane.b]));
    expect(routes[0]?.gimbalPitchDeg).toBe(-90);
    expect(routes[0]?.poi).toBeNull();
  });
  const waylines = parse(waylinesWpml(routes, DEFAULT_SCOUT));
  const template = parse(templateKml(routes, DEFAULT_SCOUT, new Date(2026, 9, 16)));

  it("uses the KML and WPML 1.0.2 namespaces, for the M3E", () => {
    expect(waylines.documentElement.namespaceURI).toBe("http://www.opengis.net/kml/2.2");
    expect(waylines.documentElement.lookupNamespaceURI("wpml")).toBe(
      "http://www.dji.com/wpmz/1.0.2",
    );
    expect(text(waylines, "droneEnumValue")).toBe("77");
    expect(text(waylines, "payloadEnumValue")).toBe("66");
    expect(text(waylines, "executeHeightMode")).toBe("relativeToStartPoint");
    expect(text(template, "heightMode")).toBe("relativeToStartPoint");
    expect(text(template, "author")).toBe("Land workspace (fog-v1 scout)");
    expect(Number(text(waylines, "globalRTHHeight"))).toBeGreaterThanOrEqual(60);
  });

  it("never climbs over the 400 ft cap to fly home, at any height the export offers", () => {
    for (const altitudeM of SCOUT_ALTITUDES_M) {
      const scout = { ...DEFAULT_SCOUT, altitudeM };
      const flown = scoutWaylines([{ footprint: holed, scout }]);
      const heights = [
        Number(text(parse(waylinesWpml(flown, scout)), "globalRTHHeight")),
        Number(text(parse(templateKml(flown, scout, new Date(2026, 9, 16))), "globalRTHHeight")),
      ];
      // 20 m over the route, at least 60 m, never over the cap (120 m is 394 ft).
      const expected = Math.min(Math.max(altitudeM + 20, 60), MAX_ALTITUDE_M);
      expect(heights).toEqual([expected, expected]);
      expect(scoutGeometry(scout).returnHeightM).toBe(expected);
    }
    expect(scoutGeometry({ ...DEFAULT_SCOUT, altitudeM: 120 }).returnHeightM).toBe(120);
  });

  it("puts one waypoint at each end of every lane, indexed from 0 without gaps", () => {
    const placemarks = Array.from(waylines.getElementsByTagName("Placemark"));
    expect(placemarks).toHaveLength(lanes.length * 2);
    placemarks.forEach((placemark, i) => expect(text(placemark, "index")).toBe(String(i)));
    expect(template.getElementsByTagName("Placemark")).toHaveLength(placemarks.length);
    const bounds = [boundsOf(holed), boundsOf(second)];
    for (const placemark of placemarks) {
      const coordinates = placemark.getElementsByTagName("coordinates")[0]?.textContent ?? "";
      const [lon, lat] = coordinates.split(",").map(Number) as [number, number];
      expect(coordinates).toMatch(/^-?\d+\.\d{7},-?\d+\.\d{7}$/);
      const within = bounds.some(
        (b) =>
          lon >= b.west - 1e-6 &&
          lon <= b.east + 1e-6 &&
          lat >= b.south - 1e-6 &&
          lat <= b.north + 1e-6,
      );
      expect(within).toBe(true);
      expect(Number(text(placemark, "executeHeight"))).toBe(60);
    }
  });

  it("photographs every lane at the photo spacing and nothing on the transit legs", () => {
    const spacing = scoutGeometry(DEFAULT_SCOUT).photoSpacingM;
    const groups = wpml(waylines, "actionGroup");
    const photos = groups.filter((g) => text(g, "actionTriggerType") === "multipleDistance");
    expect(photos).toHaveLength(lanes.length);
    photos.forEach((group, k) => {
      expect(text(group, "actionGroupStartIndex")).toBe(String(2 * k));
      expect(text(group, "actionGroupEndIndex")).toBe(String(2 * k + 1));
      expect(Number(text(group, "actionTriggerParam"))).toBeCloseTo(spacing, 2);
      expect(text(group, "actionActuatorFunc")).toBe("takePhoto");
      expect(text(group, "payloadLensIndex")).toBe("wide");
    });
    const gimbal = groups.filter((g) => text(g, "actionActuatorFunc") === "gimbalRotate");
    expect(gimbal).toHaveLength(1);
    expect(text(gimbal[0]!, "gimbalPitchRotateAngle")).toBe("-90");
    const ids = groups.map((g) => Number(text(g, "actionGroupId")));
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("says it is a draft, and escapes text", () => {
    expect(waylinesWpml(routes, DEFAULT_SCOUT)).toContain("not flight-tested");
    expect(escapeXml(`<a & "b">`)).toBe("&lt;a &amp; &quot;b&quot;&gt;");
  });

  it("zips template and waylines under wpmz/", () => {
    const kmz = buildScoutKmz(routes, DEFAULT_SCOUT, new Date(2026, 9, 16));
    const names = new TextDecoder().decode(kmz);
    expect(names).toContain("wpmz/template.kml");
    expect(names).toContain("wpmz/waylines.wpml");
    expect(new DataView(kmz.buffer).getUint32(kmz.length - 22, true)).toBe(0x06054b50);
    expect(new DataView(kmz.buffer).getUint16(kmz.length - 12, true)).toBe(3);
  });
});

describe("DJI WPML re-scan patterns", () => {
  const rescan = (pattern: "nadir" | "oblique" | "orbit", gimbalPitchDeg: number): Scout => ({
    ...DEFAULT_SCOUT,
    purpose: "re-scan-weak",
    pattern,
    gimbalPitchDeg,
  });
  const folders = (doc: Document) => Array.from(doc.getElementsByTagName("Folder"));
  const placemarks = (folder: Element) => Array.from(folder.getElementsByTagName("Placemark"));

  it("flies crossing obliques as two waylines, the second at right angles, camera pitched", () => {
    const scout = rescan("oblique", -45);
    const waylines = scoutWaylines([{ footprint: second, scout }]);
    expect(waylines).toHaveLength(2);
    const [along, across] = waylines;
    // The first set is the lanes the map draws; the second crosses them.
    expect(along?.waypoints).toEqual(scoutLanes(second, scout).flatMap((lane) => [lane.a, lane.b]));
    const heading = (w: ScoutWayline | undefined) =>
      w?.waypoints[0] && w.waypoints[1] ? bearing(w.waypoints[0], w.waypoints[1]) : NaN;
    const turn = Math.abs(((heading(across) - heading(along) + 540) % 360) - 180);
    expect(Math.abs(turn - 90)).toBeLessThan(1);

    const flown = parse(waylinesWpml(waylines, scout));
    const template = parse(templateKml(waylines, scout, new Date(2026, 9, 16)));
    // One executable wayline per template, ids from 0, waypoint indices from 0 in each.
    expect(folders(flown).map((f) => [text(f, "templateId"), text(f, "waylineId")])).toEqual([
      ["0", "0"],
      ["1", "1"],
    ]);
    expect(folders(template).map((f) => text(f, "templateId"))).toEqual(["0", "1"]);
    folders(flown).forEach((folder, k) => {
      const points = placemarks(folder);
      expect(points).toHaveLength(waylines[k]?.waypoints.length ?? -1);
      points.forEach((p, i) => expect(text(p, "index")).toBe(String(i)));
      expect(text(folder, "waypointHeadingMode")).toBe("followWayline");
      const gimbal = wpml(folder, "actionGroup").filter(
        (g) => text(g, "actionActuatorFunc") === "gimbalRotate",
      );
      expect(gimbal.map((g) => text(g, "gimbalPitchRotateAngle"))).toEqual(["-45"]);
      // Photos on every lane, none on the legs between.
      const photos = wpml(folder, "actionGroup").filter(
        (g) => text(g, "actionTriggerType") === "multipleDistance",
      );
      expect(photos).toHaveLength(points.length / 2);
    });
    // The pitch at every waypoint of the template (gimbalPitchMode usePointSetting).
    for (const folder of folders(template)) {
      expect(text(folder, "gimbalPitchMode")).toBe("usePointSetting");
      expect(new Set(placemarks(folder).map((p) => text(p, "gimbalPitchAngle")))).toEqual(
        new Set(["-45"]),
      );
    }
  });

  it("flies an orbit round the outline, the nose on its centre", () => {
    const scout = rescan("orbit", -30);
    const [orbit, ...rest] = scoutWaylines([{ footprint: second, scout }]);
    expect(rest).toEqual([]);
    if (!orbit?.poi) throw new Error("no orbit");
    expect(orbit.poi).toEqual(centerOf(second));
    // Two closed rings round the outline at the stand-off where the pitched camera meets the
    // ground, as the planner sizes it (its convex hull grown by the stand-off).
    const standoff = scoutGeometry(scout).altitudeM / Math.tan(Math.PI / 6);
    expect(orbitStandoffM(scout)).toBeCloseTo(standoff, 6);
    const ring = orbitRing(second, scout);
    expect(orbit.waypoints).toEqual([...ring, ...ring, ring[0]]);
    expect(ORBIT_RINGS).toBe(2);
    for (const p of ring) expect(outlineDistanceM(second, p)).toBeCloseTo(standoff, 0);
    // Straight down, an orbit still keeps its distance (the planner's 5 m floor).
    expect(orbitStandoffM(rescan("orbit", -90))).toBe(ORBIT_MIN_STANDOFF_M);

    const flown = parse(waylinesWpml([orbit], scout));
    const poi = `${orbit.poi.latitude.toFixed(7)},${orbit.poi.longitude.toFixed(7)},0.000000`;
    const points = placemarks(folders(flown)[0]!);
    expect(points).toHaveLength(orbit.waypoints.length);
    for (const p of points) {
      expect(text(p, "waypointHeadingMode")).toBe("towardPOI");
      // WPML's order: latitude, longitude, height (0: it takes none).
      expect(text(p, "waypointPoiPoint")).toBe(poi);
    }
    const template = parse(templateKml([orbit], scout, new Date(2026, 9, 16)));
    expect(text(template, "waypointHeadingMode")).toBe("towardPOI");
    expect(text(template, "waypointPoiPoint")).toBe(poi);
    // Photos all the way round, from the first waypoint to the last.
    const photos = wpml(flown, "actionGroup").filter(
      (g) => text(g, "actionTriggerType") === "multipleDistance",
    );
    expect(
      photos.map((g) => [text(g, "actionGroupStartIndex"), text(g, "actionGroupEndIndex")]),
    ).toEqual([["0", String(orbit.waypoints.length - 1)]]);
    expect(wpml(flown, "gimbalPitchRotateAngle").map((e) => e.textContent)).toEqual(["-30"]);
  });

  it("keeps consecutive nadir areas on one wayline and gives each pattern its own", () => {
    const nadir = rescan("nadir", -90);
    const waylines = scoutWaylines([
      { footprint: holed, scout: DEFAULT_SCOUT },
      { footprint: second, scout: nadir },
      { footprint: second, scout: rescan("orbit", -45) },
      { footprint: holed, scout: DEFAULT_SCOUT },
    ]);
    expect(waylines.map((w) => (w.poi ? "orbit" : w.gimbalPitchDeg))).toEqual([-90, "orbit", -90]);
    expect(waylines[0]?.waypoints).toHaveLength(
      (scoutLanes(holed, DEFAULT_SCOUT).length + scoutLanes(second, nadir).length) * 2,
    );
    const kmz = buildScoutKmz(waylines, DEFAULT_SCOUT, new Date(2026, 9, 16));
    expect(kmz.length).toBeGreaterThan(0);
  });

  it("counts an oblique's crossing set and an orbit's rings in the route's length", () => {
    const oblique = rescan("oblique", -45);
    const nadirStats = waylineStats(
      scoutWaylines([{ footprint: second, scout: DEFAULT_SCOUT }]),
      DEFAULT_SCOUT,
    );
    const obliqueStats = waylineStats(
      scoutWaylines([{ footprint: second, scout: oblique }]),
      oblique,
    );
    expect(obliqueStats.workingM).toBeGreaterThan(1.5 * nadirStats.workingM);
    expect(obliqueStats.photos).toBeGreaterThan(nadirStats.photos);
    const orbit = rescan("orbit", -30);
    const ring = scoutWaylines([{ footprint: second, scout: orbit }]);
    const stats = waylineStats(ring, orbit);
    // Two rings, all of it photographed, nothing in transit: the planner's ring length (the
    // outline's convex hull grown by the stand-off: its perimeter plus a circle) twice.
    const perimeter = 2 * (200 + 100) + 2 * Math.PI * orbitStandoffM(orbit);
    expect(Math.abs(stats.workingM / (ORBIT_RINGS * perimeter) - 1)).toBeLessThan(0.01);
    expect(stats.transitM).toBe(0);
    expect(stats.photos).toBe(Math.floor(stats.workingM / scoutGeometry(orbit).photoSpacingM) + 1);
    // A fog scout's numbers are routeStats' over the same lanes.
    const lanes = [scoutLanes(holed, DEFAULT_SCOUT), scoutLanes(second, DEFAULT_SCOUT)];
    const old = routeStats(lanes, DEFAULT_SCOUT);
    const now = waylineStats(
      scoutWaylines([
        { footprint: holed, scout: DEFAULT_SCOUT },
        { footprint: second, scout: DEFAULT_SCOUT },
      ]),
      DEFAULT_SCOUT,
    );
    expect(now.photos).toBe(old.photos);
    expect(now.workingM).toBeCloseTo(old.workingM, -1);
    expect(now.transitM).toBeCloseTo(old.transitM, 3);
  });

  it("numbers action groups once across the KMZ, as WPML asks (unique within a file)", () => {
    const waylines = scoutWaylines([
      { footprint: holed, scout: DEFAULT_SCOUT },
      { footprint: second, scout: rescan("oblique", -45) },
      { footprint: second, scout: rescan("orbit", -30) },
    ]);
    expect(waylines).toHaveLength(4);
    for (const xml of [
      waylinesWpml(waylines, DEFAULT_SCOUT),
      templateKml(waylines, DEFAULT_SCOUT, new Date(2026, 9, 16)),
    ]) {
      const ids = wpml(parse(xml), "actionGroupId").map((e) => Number(e.textContent));
      // Monotonic and continuous from 0, as the reference recommends.
      expect(ids).toEqual(ids.map((_, i) => i));
    }
  });

  it("orbits a long strip along its sides, not round a circle past its ends", () => {
    // 200 m by 20 m: a circle round its corners would aim the camera ~90 m off the long sides.
    const strip: Footprint = {
      type: "Polygon",
      coordinates: [
        [
          [e(0), n(0)],
          [e(200), n(0)],
          [e(200), n(20)],
          [e(0), n(20)],
          [e(0), n(0)],
        ],
      ],
    };
    const scout = rescan("orbit", -30);
    const ring = orbitRing(strip, scout);
    const standoff = orbitStandoffM(scout);
    for (const p of ring) expect(Math.abs(outlineDistanceM(strip, p) - standoff)).toBeLessThan(1);
    const length = ring.reduce(
      (sum, p, i) => sum + haversineDistance(p, ring[(i + 1) % ring.length] ?? p),
      0,
    );
    const planner = 2 * (200 + 20) + 2 * Math.PI * standoff;
    expect(Math.abs(length / planner - 1)).toBeLessThan(0.01);
    // A single point still gets a ring round it.
    const dot: Footprint = {
      type: "Polygon",
      coordinates: [
        [
          [e(5), n(5)],
          [e(5), n(5)],
          [e(5), n(5)],
          [e(5), n(5)],
        ],
      ],
    };
    const round = orbitRing(dot, rescan("orbit", -90));
    expect(round.length).toBe(360 / ORBIT_STEP_DEG);
    for (const p of round)
      expect(haversineDistance({ longitude: e(5), latitude: n(5) }, p)).toBeCloseTo(
        ORBIT_MIN_STANDOFF_M,
        1,
      );
  });

  it("draws on the map what the export flies", () => {
    expect(scoutPasses(second, DEFAULT_SCOUT)).toEqual(scoutLanes(second, DEFAULT_SCOUT));
    const oblique = rescan("oblique", -45);
    const [along, across] = scoutWaylines([{ footprint: second, scout: oblique }]);
    const passes = scoutPasses(second, oblique);
    expect(passes.flatMap((l) => [l.a, l.b])).toEqual([
      ...(along?.waypoints ?? []),
      ...(across?.waypoints ?? []),
    ]);
    // The two sets are drawn apart: no stroke joins the last lane of one to the next set.
    expect(laneStrokes(passes).length).toBeGreaterThanOrEqual(2);
    const orbit = rescan("orbit", -30);
    const ring = orbitRing(second, orbit);
    const drawn = scoutPasses(second, orbit);
    expect(drawn).toHaveLength(ring.length);
    expect(laneStrokes(drawn)).toEqual([
      [...ring.flatMap((p, i) => [p, ring[(i + 1) % ring.length]])],
    ]);
  });

  it("samples the ground an orbit flies over, not only inside the outline", () => {
    const orbit = rescan("orbit", -30);
    const outlines = reliefOutlines([
      { footprint: holed, scout: DEFAULT_SCOUT },
      { footprint: second, scout: orbit },
    ]);
    expect(outlines.slice(0, 2)).toEqual([holed, second]);
    const ring = orbitRing(second, orbit);
    expect(outlines[2]).toEqual({
      type: "Polygon",
      coordinates: [[...ring, ring[0]].map((p) => [p?.longitude ?? 0, p?.latitude ?? 0])],
    });
    expect(outlines).toHaveLength(3);
  });
});

/** Metres from `p` to the nearest edge of `footprint`'s outer ring (flat, about `p`). */
function outlineDistanceM(footprint: Footprint, p: { longitude: number; latitude: number }) {
  const kx = 111_320 * Math.cos((p.latitude * Math.PI) / 180);
  const ring = (footprint.type === "Polygon" ? footprint.coordinates[0] : []) ?? [];
  const xy = ring.map(([lon, lat]) => [
    ((lon ?? 0) - p.longitude) * kx,
    ((lat ?? 0) - p.latitude) * 111_320,
  ]);
  let best = Infinity;
  for (let i = 1; i < xy.length; i++) {
    const [ax = 0, ay = 0] = xy[i - 1] ?? [];
    const [bx = 0, by = 0] = xy[i] ?? [];
    const dx = bx - ax;
    const dy = by - ay;
    const t = dx || dy ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / (dx * dx + dy * dy))) : 0;
    best = Math.min(best, Math.hypot(ax + t * dx, ay + t * dy));
  }
  return best;
}
