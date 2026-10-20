/**
 * The scout route as a DJI WPML KMZ: `wpmz/template.kml` and `wpmz/waylines.wpml` (WPML
 * 1.0.2), zipped, for DJI Pilot 2 and FlightHub 2 with a Mavic 3 Enterprise class aircraft.
 *
 * It is an explicit waypoint route, not a mapping template the app regenerates: one waypoint
 * at each end of every lane the map draws (`missions/scout.ts`, holes honoured), in the same
 * order, heights relative to the take-off point. Along each lane an action group fires the
 * wide camera every photo spacing (multipleDistance + takePhoto); transit legs between lanes
 * take no photos. The gimbal points straight down at the first waypoint.
 *
 * Draft, not flight-tested: nobody has imported this into DJI Pilot 2 or FlightHub 2 yet, so
 * the files say so, and so does the console. Terrain following is not attempted.
 *
 * A stub until there is a drone: offered only with `VITE_ENABLE_DRONE_EXPORT` on (off by
 * default). docs/DRONES.md "Drone missions: before first flight" lists what must be checked
 * first: the WPML fields DJI's spec does not show, the required ones missing, the drone and
 * app, and the test import into DJI Pilot 2.
 */

import type { CoverageLane } from "@twin/geo";

import { zipStore } from "@/lib/zipStore";

import { SCOUT_CAMERAS, routeStats, scoutGeometry, type Scout } from "./scout";

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

interface Waypoint {
  longitude: number;
  latitude: number;
}

/** The waypoints in flying order: the two ends of every lane, zone after zone. */
export function waypointsOf(routes: readonly CoverageLane[][]): Waypoint[] {
  return routes.flatMap((lanes) => lanes.flatMap((lane) => [lane.a, lane.b]));
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

function gimbalDown(): string {
  return [
    "<wpml:actionGroup>",
    el("actionGroupId", 0),
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
    el("gimbalPitchRotateAngle", -90),
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

/** Photos every `spacing` metres from waypoint `start` to `start + 1` (one lane). */
function photosAlong(groupId: number, start: number, spacingM: number): string {
  return [
    "<wpml:actionGroup>",
    el("actionGroupId", groupId),
    el("actionGroupStartIndex", start),
    el("actionGroupEndIndex", start + 1),
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

/** The action groups that start at waypoint `index`. */
function groupsAt(index: number, spacingM: number): string {
  const groups: string[] = [];
  if (index === 0) groups.push(gimbalDown());
  // Every lane starts at an even index: its photos run to the next waypoint, the lane's end.
  if (index % 2 === 0) groups.push(photosAlong(index / 2 + 1, index, spacingM));
  return groups.join("\n");
}

function point(waypoint: Waypoint): string {
  return `<Point><coordinates>${waypoint.longitude.toFixed(7)},${waypoint.latitude.toFixed(7)}</coordinates></Point>`;
}

function document(scout: Scout, body: string[]): string {
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

/** `wpmz/waylines.wpml`: what the aircraft executes. */
export function waylinesWpml(routes: readonly CoverageLane[][], scout: Scout): string {
  const geometry = scoutGeometry(scout);
  const stats = routeStats(routes, scout);
  const placemarks = waypointsOf(routes).map((waypoint, index) =>
    [
      "<Placemark>",
      point(waypoint),
      el("index", index),
      el("executeHeight", geometry.altitudeM),
      el("waypointSpeed", geometry.speedMs),
      "<wpml:waypointHeadingParam>",
      el("waypointHeadingMode", "followWayline"),
      el("waypointHeadingAngle", 0),
      el("waypointPoiPoint", "0.000000,0.000000,0.000000"),
      el("waypointHeadingAngleEnable", 0),
      el("waypointHeadingPathMode", "followBadArc"),
      "</wpml:waypointHeadingParam>",
      "<wpml:waypointTurnParam>",
      el("waypointTurnMode", "toPointAndStopWithDiscontinuityCurvature"),
      el("waypointTurnDampingDist", 0),
      "</wpml:waypointTurnParam>",
      el("useStraightLine", 1),
      groupsAt(index, geometry.photoSpacingM),
      "</Placemark>",
    ]
      .filter(Boolean)
      .join("\n"),
  );
  return document(scout, [
    missionConfig(scout),
    "<Folder>",
    el("templateId", 0),
    el("executeHeightMode", "relativeToStartPoint"),
    el("waylineId", 0),
    el("distance", stats.workingM + stats.transitM),
    el("duration", stats.minutes * 60),
    el("autoFlightSpeed", geometry.speedMs),
    ...placemarks,
    "</Folder>",
  ]);
}

/** `wpmz/template.kml`: the route as the planning apps show and edit it. */
export function templateKml(
  routes: readonly CoverageLane[][],
  scout: Scout,
  createdAt: Date = new Date(),
): string {
  const geometry = scoutGeometry(scout);
  const placemarks = waypointsOf(routes).map((waypoint, index) =>
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
      el("gimbalPitchAngle", -90),
      groupsAt(index, geometry.photoSpacingM),
      "</Placemark>",
    ]
      .filter(Boolean)
      .join("\n"),
  );
  const time = String(createdAt.getTime());
  return document(scout, [
    el("author", AUTHOR),
    el("createTime", time),
    el("updateTime", time),
    missionConfig(scout),
    "<Folder>",
    el("templateType", "waypoint"),
    el("templateId", 0),
    "<wpml:waylineCoordinateSysParam>",
    el("coordinateMode", "WGS84"),
    el("heightMode", "relativeToStartPoint"),
    "</wpml:waylineCoordinateSysParam>",
    el("autoFlightSpeed", geometry.speedMs),
    el("globalHeight", geometry.altitudeM),
    el("caliFlightEnable", 0),
    el("gimbalPitchMode", "usePointSetting"),
    "<wpml:globalWaypointHeadingParam>",
    el("waypointHeadingMode", "followWayline"),
    el("waypointHeadingAngle", 0),
    el("waypointPoiPoint", "0.000000,0.000000,0.000000"),
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
  ]);
}

/**
 * Why a capture's pattern cannot be exported yet: the waylines here fly nadir lanes, and a
 * re-scan's crossing obliques or orbit would be flown as nadir lines without it (the planner
 * still sizes them; the export follows).
 */
export const PATTERN_NOT_EXPORTED =
  "This re-scan asks for obliques/an orbit; the DJI export flies nadir only so far (W4c)";

/** The KMZ: `wpmz/` with the template and the waylines, store-only. */
export function buildScoutKmz(
  routes: readonly CoverageLane[][],
  scout: Scout,
  createdAt: Date = new Date(),
): Uint8Array {
  if (scout.pattern !== "nadir") throw new Error(PATTERN_NOT_EXPORTED);
  const encoder = new TextEncoder();
  return zipStore(
    [
      { name: "wpmz/", data: null },
      { name: "wpmz/template.kml", data: encoder.encode(templateKml(routes, scout, createdAt)) },
      { name: "wpmz/waylines.wpml", data: encoder.encode(waylinesWpml(routes, scout)) },
    ],
    createdAt,
  );
}
