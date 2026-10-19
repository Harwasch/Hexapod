import type { Footprint } from "@twin/contracts";
import { boundsOf, destination } from "@twin/geo";
import { describe, expect, it } from "vitest";

import {
  DEFAULT_SCOUT,
  MAX_ALTITUDE_M,
  SCOUT_ALTITUDES_M,
  scoutGeometry,
  scoutLanes,
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
  const routes = [scoutLanes(holed, DEFAULT_SCOUT), scoutLanes(second, DEFAULT_SCOUT)];
  const lanes = routes.flat();
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
      const flown = [scoutLanes(holed, scout)];
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
