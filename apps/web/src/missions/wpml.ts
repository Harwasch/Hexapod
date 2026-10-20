/**
 * The scout route as a DJI WPML KMZ: `wpmz/template.kml` and `wpmz/waylines.wpml` (WPML
 * 1.0.2), zipped, for DJI Pilot 2 and FlightHub 2 with a Mavic 3 Enterprise class aircraft.
 *
 * It is an explicit waypoint route, not a mapping template the app regenerates. Each wayline
 * (`missions/scout.ts` `scoutWaylines`, by the areas' patterns) is one `Folder` in each file:
 * a waypoint template (`templateId` k) and the executable route made from it (`templateId` k,
 * `waylineId` k), indices from 0 within each, as the WPML reference asks. Heights are relative
 * to the take-off point.
 *
 * - Nadir lanes (a fog scout, a nadir re-scan): one waypoint at each end of every lane the map
 *   draws, in the same order, holes honoured; along each lane an action group fires the wide
 *   camera every photo spacing (multipleDistance + takePhoto) and transit legs take none.
 * - Crossing obliques: the same, twice, the second set of lanes at right angles; the camera at
 *   the capture's pitch (`gimbalPitchAngle` at every template waypoint, `gimbalRotate` at the
 *   first of each wayline), the nose along each lane.
 * - An orbit: waypoints round a circle, the nose on its centre (`waypointHeadingMode`
 *   `towardPOI`, `waypointPoiPoint` "latitude,longitude,0" -- WPML takes no height for it),
 *   photos all the way round.
 *
 * The WPML elements are DJI's (Cloud API, DJI WPML: template.kml, waylines.wpml, common
 * elements); none is written that the reference does not define for a waypoint route.
 *
 * Draft, not flight-tested: nobody has imported this into DJI Pilot 2 or FlightHub 2 yet, so
 * the files say so, and so does the console. Terrain following is not attempted.
 *
 * A stub until there is a drone: offered only with `VITE_ENABLE_DRONE_EXPORT` on (off by
 * default). docs/DRONES.md "Drone missions: before first flight" lists what must be checked
 * first: the WPML fields DJI's spec does not show, the required ones missing, the drone and
 * app, and the test import into DJI Pilot 2.
 */

import { haversineDistance, type LonLat } from "@twin/geo";

import { zipStore } from "@/lib/zipStore";

import { SCOUT_CAMERAS, scoutGeometry, type Scout, type ScoutWayline } from "./scout";

export const KML_NS = "http://www.opengis.net/kml/2.2";
export const WPML_NS = "http://www.dji.com/wpmz/1.0.2";
export const AUTHOR = "Land workspace (fog-v1 scout)";
const NOTICE =
  "Draft scout route, not flight-tested: review it in DJI Pilot 2 or FlightHub 2 before " +
  "flying. Heights are relative to the take-off point and do not follow terrain. Check " +
  "airspace (B4UFLY or LAANC) and fly under Part 107.";

/** XML-escapes text content and attribute values. */
export function escapeXml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/** A number as WPML writes it: no exponent, trailing zeros trimmed. */
function num(value: number, digits = 2): string {
  return String(Number(value.toFixed(digits)));
}

function el(name: string, value: string | number): string {
  return `<wpml:${name}>${typeof value === "number" ? num(value) : escapeXml(value)}</wpml:${name}>`;
}

/** `waypointPoiPoint`: latitude, longitude, height (unused by WPML, so 0); none is all zeros. */
function poiPoint(poi: LonLat | null): string {
  return poi
    ? `${poi.latitude.toFixed(7)},${poi.longitude.toFixed(7)},0.000000`
    : "0.000000,0.000000,0.000000";
}

function headingMode(wayline: ScoutWayline): string {
  return wayline.poi ? "towardPOI" : "followWayline";
}

function missionConfig(scout: Scout): string {
  const camera = SCOUT_CAMERAS[scout.camera];
  const geometry = scoutGeometry(scout);
  return [
    "<wpml:missionConfig>",
    el("flyToWaylineMode", "safely"),
    el("finishAction", "goHome"),
    el("exitOnRCLost", "executeLostAction"),
    el("executeRCLostAction", "goBack"),
    el("takeOffSecurityHeight", 20),
    el("globalTransitionalSpeed", 10),
    el("globalRTHHeight", geometry.returnHeightM),
    "<wpml:droneInfo>",
    el("droneEnumValue", camera.droneEnum),
    el("droneSubEnumValue", camera.droneSubEnum),
    "</wpml:droneInfo>",
    "<wpml:payloadInfo>",
    el("payloadEnumValue", camera.payloadEnum),
    el("payloadSubEnumValue", camera.payloadSubEnum),
    el("payloadPositionIndex", 0),
    "</wpml:payloadInfo>",
    "</wpml:missionConfig>",
  ].join("\n");
}

/** The camera to `pitchDeg` at the wayline's first waypoint. */
function gimbalTo(groupId: number, pitchDeg: number): string {
  return [
    "<wpml:actionGroup>",
    el("actionGroupId", groupId),
    el("actionGroupStartIndex", 0),
    el("actionGroupEndIndex", 0),
    el("actionGroupMode", "sequence"),
    "<wpml:actionTrigger>",
    el("actionTriggerType", "reachPoint"),
    "</wpml:actionTrigger>",
    "<wpml:action>",
    el("actionId", 0),
    el("actionActuatorFunc", "gimbalRotate"),
    "<wpml:actionActuatorFuncParam>",
    el("gimbalRotateMode", "absoluteAngle"),
    el("gimbalPitchRotateEnable", 1),
    el("gimbalPitchRotateAngle", pitchDeg),
    el("gimbalRollRotateEnable", 0),
    el("gimbalRollRotateAngle", 0),
    el("gimbalYawRotateEnable", 0),
    el("gimbalYawRotateAngle", 0),
    el("gimbalRotateTimeEnable", 0),
    el("gimbalRotateTime", 0),
    el("payloadPositionIndex", 0),
    "</wpml:actionActuatorFuncParam>",
    "</wpml:action>",
    "</wpml:actionGroup>",
  ].join("\n");
}

/** Photos every `spacing` metres from waypoint `start` to waypoint `end`. */
function photosAlong(groupId: number, start: number, end: number, spacingM: number): string {
  return [
    "<wpml:actionGroup>",
    el("actionGroupId", groupId),
    el("actionGroupStartIndex", start),
    el("actionGroupEndIndex", end),
    el("actionGroupMode", "sequence"),
    "<wpml:actionTrigger>",
    el("actionTriggerType", "multipleDistance"),
    el("actionTriggerParam", spacingM),
    "</wpml:actionTrigger>",
    "<wpml:action>",
    el("actionId", 0),
    el("actionActuatorFunc", "takePhoto"),
    "<wpml:actionActuatorFuncParam>",
    el("fileSuffix", "fog"),
    el("payloadPositionIndex", 0),
    el("payloadLensIndex", "wide"),
    el("useGlobalPayloadLensIndex", 0),
    "</wpml:actionActuatorFuncParam>",
    "</wpml:action>",
    "</wpml:actionGroup>",
  ].join("\n");
}

/**
 * The action groups that start at waypoint `index` of `wayline`; `ids` numbers them once
 * across the whole file (WPML: an actionGroupId is unique within a KMZ).
 */
function groupsAt(
  wayline: ScoutWayline,
  index: number,
  spacingM: number,
  ids: { next: number },
): string {
  const groups: string[] = [];
  if (index === 0) groups.push(gimbalTo(ids.next++, wayline.gimbalPitchDeg));
  if (wayline.photos === "along") {
    // Round the whole orbit, from its first waypoint to its last.
    if (index === 0)
      groups.push(photosAlong(ids.next++, 0, wayline.waypoints.length - 1, spacingM));
  } else if (index % 2 === 0) {
    // Every lane starts at an even index: its photos run to the next waypoint, the lane's end.
    groups.push(photosAlong(ids.next++, index, index + 1, spacingM));
  }
  return groups.join("\n");
}

function point(waypoint: LonLat): string {
  return `<Point><coordinates>${waypoint.longitude.toFixed(7)},${waypoint.latitude.toFixed(7)}</coordinates></Point>`;
}

function document(body: string[]): string {
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    `<!-- ${escapeXml(NOTICE)} -->`,
    `<kml xmlns="${KML_NS}" xmlns:wpml="${WPML_NS}">`,
    "<Document>",
    ...body,
    "</Document>",
    "</kml>",
    "",
  ].join("\n");
}

function lengthM(waypoints: readonly LonLat[]): number {
  let total = 0;
  for (let i = 1; i < waypoints.length; i++) {
    const a = waypoints[i - 1];
    const b = waypoints[i];
    if (a && b) total += haversineDistance(a, b);
  }
  return total;
}

/** `wpmz/waylines.wpml`: what the aircraft executes, one `Folder` a wayline. */
export function waylinesWpml(waylines: readonly ScoutWayline[], scout: Scout): string {
  const geometry = scoutGeometry(scout);
  const ids = { next: 0 };
  const folders = waylines.map((wayline, id) => {
    const placemarks = wayline.waypoints.map((waypoint, index) =>
      [
        "<Placemark>",
        point(waypoint),
        el("index", index),
        el("executeHeight", geometry.altitudeM),
        el("waypointSpeed", geometry.speedMs),
        "<wpml:waypointHeadingParam>",
        el("waypointHeadingMode", headingMode(wayline)),
        el("waypointHeadingAngle", 0),
        el("waypointPoiPoint", poiPoint(wayline.poi)),
        el("waypointHeadingAngleEnable", 0),
        el("waypointHeadingPathMode", "followBadArc"),
        "</wpml:waypointHeadingParam>",
        "<wpml:waypointTurnParam>",
        el("waypointTurnMode", "toPointAndStopWithDiscontinuityCurvature"),
        el("waypointTurnDampingDist", 0),
        "</wpml:waypointTurnParam>",
        el("useStraightLine", 1),
        groupsAt(wayline, index, geometry.photoSpacingM, ids),
        "</Placemark>",
      ]
        .filter(Boolean)
        .join("\n"),
    );
    const distanceM = lengthM(wayline.waypoints);
    return [
      "<Folder>",
      el("templateId", id),
      el("executeHeightMode", "relativeToStartPoint"),
      el("waylineId", id),
      el("distance", distanceM),
      el("duration", distanceM / geometry.speedMs),
      el("autoFlightSpeed", geometry.speedMs),
      ...placemarks,
      "</Folder>",
    ].join("\n");
  });
  return document([missionConfig(scout), ...folders]);
}

/** `wpmz/template.kml`: the route as the planning apps show and edit it, a template a wayline. */
export function templateKml(
  waylines: readonly ScoutWayline[],
  scout: Scout,
  createdAt: Date = new Date(),
): string {
  const geometry = scoutGeometry(scout);
  const ids = { next: 0 };
  const folders = waylines.map((wayline, id) => {
    const placemarks = wayline.waypoints.map((waypoint, index) =>
      [
        "<Placemark>",
        point(waypoint),
        el("index", index),
        el("ellipsoidHeight", geometry.altitudeM),
        el("height", geometry.altitudeM),
        el("useGlobalHeight", 1),
        el("useGlobalSpeed", 1),
        el("useGlobalHeadingParam", 1),
        el("useGlobalTurnParam", 1),
        el("gimbalPitchAngle", wayline.gimbalPitchDeg),
        groupsAt(wayline, index, geometry.photoSpacingM, ids),
        "</Placemark>",
      ]
        .filter(Boolean)
        .join("\n"),
    );
    return [
      "<Folder>",
      el("templateType", "waypoint"),
      el("templateId", id),
      "<wpml:waylineCoordinateSysParam>",
      el("coordinateMode", "WGS84"),
      el("heightMode", "relativeToStartPoint"),
      "</wpml:waylineCoordinateSysParam>",
      el("autoFlightSpeed", geometry.speedMs),
      el("globalHeight", geometry.altitudeM),
      el("caliFlightEnable", 0),
      el("gimbalPitchMode", "usePointSetting"),
      "<wpml:globalWaypointHeadingParam>",
      el("waypointHeadingMode", headingMode(wayline)),
      el("waypointHeadingAngle", 0),
      el("waypointPoiPoint", poiPoint(wayline.poi)),
      el("waypointHeadingPoiIndex", 0),
      "</wpml:globalWaypointHeadingParam>",
      el("globalWaypointTurnMode", "toPointAndStopWithDiscontinuityCurvature"),
      el("globalUseStraightLine", 0),
      ...placemarks,
      "<wpml:payloadParam>",
      el("payloadPositionIndex", 0),
      el("imageFormat", "wide"),
      "</wpml:payloadParam>",
      "</Folder>",
    ].join("\n");
  });
  const time = String(createdAt.getTime());
  return document([
    el("author", AUTHOR),
    el("createTime", time),
    el("updateTime", time),
    missionConfig(scout),
    ...folders,
  ]);
}

/** The KMZ: `wpmz/` with the template and the waylines, store-only. */
export function buildScoutKmz(
  waylines: readonly ScoutWayline[],
  scout: Scout,
  createdAt: Date = new Date(),
): Uint8Array {
  const encoder = new TextEncoder();
  return zipStore(
    [
      { name: "wpmz/", data: null },
      { name: "wpmz/template.kml", data: encoder.encode(templateKml(waylines, scout, createdAt)) },
      { name: "wpmz/waylines.wpml", data: encoder.encode(waylinesWpml(waylines, scout)) },
    ],
    createdAt,
  );
}
