/**
 * Scout flights that would lift the Fog of Knowing: the camera geometry, the lanes and the
 * route numbers for a mapping drone flying a fog patch in parallel lines, nadir photos at an
 * even spacing.
 *
 * The flight values (height above take-off, speed, side and forward overlap, the lens's
 * field of view across and the image aspect) are the drone class's in
 * `apps/api/app/services/machine_classes.json`, the one table the planner sizes the same
 * flying with; a scout area may carry its own values (`Zone.scout`, a `ScoutCapture`). Only
 * the camera's own facts the export needs (pixel count, how fast it can fire, DJI's enum
 * values) live here. Every number is an assumption, said so where it is shown.
 *
 * The lanes are `footprintLanes` (holes honoured): exactly the dashed passes the map draws
 * and the waylines the KMZ carries (`missions/wpml.ts`).
 */

import type { Footprint, ScoutCapture } from "@twin/contracts";
import {
  bearing,
  boundsOf,
  centerOf,
  destination,
  footprintAreaM2,
  footprintContains,
  footprintLanes,
  haversineDistance,
  outerRings,
  type CoverageLane,
  type LonLat,
} from "@twin/geo";

import { MACHINE_CLASSES } from "./machineClasses";
import type { PlanOverlay, Zone } from "./types";

export interface ScoutCamera {
  label: string;
  imageWidthPx: number;
  imageHeightPx: number;
  /** Shortest interval between two photos the camera sustains, seconds. */
  minIntervalS: number;
  /** DJI WPML enum values for the aircraft and its payload. */
  droneEnum: number;
  droneSubEnum: number;
  payloadEnum: number;
  payloadSubEnum: number;
  source: string;
}

export const SCOUT_CAMERAS: Record<ScoutCapture["camera"] & string, ScoutCamera> = {
  "dji-m3e-wide": {
    label: "DJI Mavic 3 Enterprise wide camera (assumed)",
    imageWidthPx: 5280,
    imageHeightPx: 3956,
    minIntervalS: 0.7,
    droneEnum: 77,
    droneSubEnum: 0,
    payloadEnum: 66,
    payloadSubEnum: 0,
    source: "DJI Mavic 3 Enterprise specifications; DJI WPML 1.0.2 enum values",
  },
};

/** A scout area's capture, every field set. */
export type Scout = Required<Omit<ScoutCapture, "headingDeg">> & { headingDeg: number | null };

const drone = MACHINE_CLASSES.drone;

/** The drone class's flight, as a scout capture: what "Lift the fog" proposes. */
export const DEFAULT_SCOUT: Scout = {
  purpose: "lift-knowledge-fog",
  // Nadir lines, camera straight down; a re-scan of weak areas (M4) may ask for crossing
  // obliques or an orbit, which the planner sizes and the DJI export does not fly yet.
  pattern: "nadir",
  gimbalPitchDeg: -90,
  rescanOf: null,
  camera: "dji-m3e-wide",
  altitudeM: drone.altitudeAglM,
  forwardOverlap: drone.frontlapPct / 100,
  sideOverlap: drone.sidelapPct / 100,
  speedMs: drone.speedMps,
  headingDeg: null,
};

/** What a scout area's chip on the map says: its passes are a forecast, not evidence. */
export const SCOUT_SHORT = "would lift fog";
/** What a re-scan area's chip on the map says: its passes are a forecast, not evidence. */
export const RESCAN_SHORT = "would re-scan";

function cmPerPixel(gsdM: number): number {
  return Math.round(gsdM * 1000) / 10;
}

const GOAL_DETAIL = /\bat \d+(?:\.\d+)? cm per pixel\b/;

/**
 * The scout goal: a survey at the detail flown (it must never say "clear", which the planner
 * reads as treat).
 */
export function scoutGoal(gsdM = scoutGeometry(DEFAULT_SCOUT).gsdM): string {
  return `Survey the foggy ground from the air at ${cmPerPixel(gsdM)} cm per pixel so the fog can lift`;
}

/**
 * The goal restated for a new height: the detail it names becomes the one flown. A goal that
 * names no detail (the operator rewrote it) is kept as it is.
 */
export function goalAtDetail(goal: string, gsdM: number): string {
  return goal.replace(GOAL_DETAIL, `at ${cmPerPixel(gsdM)} cm per pixel`);
}

/**
 * The planner's answer to "How much detail?" (2, 5 or 10 cm per pixel) that a flight at this
 * detail meets: the finest choice it is no coarser than.
 */
export function scoutResolution(gsdM: number): "2" | "5" | "10" {
  const cm = gsdM * 100;
  return cm <= 2 ? "2" : cm <= 5 ? "5" : "10";
}

/** Heights the export offers, metres above the take-off point. */
export const SCOUT_ALTITUDES_M = [45, 60, 90, 120] as const;
/** US Part 107: 400 ft above ground. Heights here are above take-off, so this is the cap. */
export const MAX_ALTITUDE_M = 120;
export const MIN_ALTITUDE_M = 20;
/**
 * How far above the route the drone climbs to fly home, and the least it climbs to: clear of
 * trees and structures on the way back. Never above MAX_ALTITUDE_M, so the return stays under
 * the same 400 ft cap as the route.
 */
export const RETURN_ABOVE_ROUTE_M = 20;
export const MIN_RETURN_HEIGHT_M = 60;
/** Flying time one battery gives with a reserve left: an assumption, labelled as one. */
export const USABLE_BATTERY_MIN = 25;
/** Share of a battery spent on lanes rather than turns and transit, for the budget. */
export const TURN_EFFICIENCY = 0.8;

export function scoutOf(capture: ScoutCapture | null | undefined): Scout {
  return {
    ...DEFAULT_SCOUT,
    ...(capture ?? {}),
    // The capture's own purpose: a re-scan stays a re-scan (the API refuses an oblique or
    // orbit "lift the fog", so forcing it would make the zone's capture invalid).
    purpose: capture?.purpose ?? DEFAULT_SCOUT.purpose,
    pattern: capture?.pattern ?? DEFAULT_SCOUT.pattern,
    gimbalPitchDeg: capture?.gimbalPitchDeg ?? DEFAULT_SCOUT.gimbalPitchDeg,
    rescanOf: capture?.rescanOf ?? null,
    camera: capture?.camera ?? DEFAULT_SCOUT.camera,
    headingDeg: capture?.headingDeg ?? null,
  };
}

export interface ScoutGeometry {
  /** The height flown: the one asked for, within 20 to 120 m. */
  altitudeM: number;
  /** Ground sample distance at that height over flat ground, metres per pixel. */
  gsdM: number;
  footprintAcrossM: number;
  footprintAlongM: number;
  lineSpacingM: number;
  photoSpacingM: number;
  /** Fastest the camera keeps up with the photo spacing, m/s. */
  maxSpeedMs: number;
  /** The speed flown: the one asked for, never faster than the camera keeps up with. */
  speedMs: number;
  /**
   * The height above take-off the drone climbs to when it flies home at the end of the route
   * or when it loses the controller: 20 m over the route, at least 60 m, at most 120 m.
   */
  returnHeightM: number;
  warnings: string[];
}

export function scoutGeometry(scout: Scout): ScoutGeometry {
  const camera = SCOUT_CAMERAS[scout.camera] ?? SCOUT_CAMERAS["dji-m3e-wide"];
  const warnings: string[] = [];
  let altitudeM = scout.altitudeM;
  if (altitudeM > MAX_ALTITUDE_M) {
    altitudeM = MAX_ALTITUDE_M;
    warnings.push(
      `Height lowered to ${MAX_ALTITUDE_M} m: Part 107 limits flights to 400 ft above the ground.`,
    );
  } else if (altitudeM < MIN_ALTITUDE_M) {
    altitudeM = MIN_ALTITUDE_M;
    warnings.push(`Height raised to ${MIN_ALTITUDE_M} m to stay clear of trees and structures.`);
  }
  const footprintAcrossM = 2 * altitudeM * Math.tan((drone.cameraHfovDeg * Math.PI) / 180 / 2);
  const footprintAlongM = footprintAcrossM / drone.imageAspect;
  const lineSpacingM = footprintAcrossM * (1 - scout.sideOverlap);
  const photoSpacingM = footprintAlongM * (1 - scout.forwardOverlap);
  const maxSpeedMs = photoSpacingM / camera.minIntervalS;
  let speedMs = scout.speedMs;
  if (speedMs > maxSpeedMs) {
    speedMs = Math.floor(maxSpeedMs * 10) / 10;
    warnings.push(
      `Speed lowered to ${speedMs} m/s: the camera fires at most every ${camera.minIntervalS} s.`,
    );
  }
  return {
    altitudeM,
    gsdM: footprintAcrossM / camera.imageWidthPx,
    footprintAcrossM,
    footprintAlongM,
    lineSpacingM,
    photoSpacingM,
    maxSpeedMs,
    speedMs,
    returnHeightM: Math.min(
      Math.max(altitudeM + RETURN_ABOVE_ROUTE_M, MIN_RETURN_HEIGHT_M),
      MAX_ALTITUDE_M,
    ),
    warnings,
  };
}

/**
 * A plan's scout areas in the order its overlay draws them (`overlayFor`: by step, then the
 * rest of the scope): the order the map routes them and the export flies them.
 */
export function scoutZonesInOrder(zones: readonly Zone[], overlay: PlanOverlay): Zone[] {
  return overlay.zones.flatMap(({ zoneId }) => {
    const zone = zones.find((z) => z.id === zoneId);
    return zone?.scout ? [zone] : [];
  });
}

/**
 * The one capture a set of scout areas shares, or null when they differ: a route is flown at
 * one height, speed and overlap. The heading, pattern and camera pitch may differ: the export
 * flies each area's own (`scoutWaylines`).
 */
export function sharedScout(zones: readonly Pick<Zone, "scout">[]): Scout | null {
  const scouts = zones.map((zone) => scoutOf(zone.scout));
  const first = scouts[0];
  if (!first) return null;
  const same = (other: Scout) =>
    other.camera === first.camera &&
    other.altitudeM === first.altitudeM &&
    other.speedMs === first.speedMs &&
    other.forwardOverlap === first.forwardOverlap &&
    other.sideOverlap === first.sideOverlap;
  return scouts.every(same) ? first : null;
}

/**
 * Why a set of scout areas cannot be written as a DJI route, or null when it can. The KMZ
 * (`buildScoutKmz`, `scoutWaylines`) flies obliques and orbits at their capture's pitch, but
 * nadir lanes only straight down (consecutive nadir areas share one wayline at -90°), so a
 * nadir capture at another pitch would export a different flight than the plan holds: refused.
 */
export function kmzRefusal(zones: readonly Pick<Zone, "scout">[]): string | null {
  const odd = zones
    .map((zone) => scoutOf(zone.scout))
    .find((scout) => scout.pattern === "nadir" && scout.gimbalPitchDeg !== -90);
  if (!odd) return null;
  const pitch = `${Math.round(odd.gimbalPitchDeg)}`.replace("-", "−");
  return `This route includes a nadir capture pitched ${pitch}°. The DJI export writes nadir lanes only straight-down, so it would fly something other than the plan.`;
}

/** The lanes a scout area is flown in: one image footprint apart, less the side overlap. */
export function scoutLanes(footprint: Footprint, scout: Scout): CoverageLane[] {
  const geometry = scoutGeometry(scout);
  return footprintLanes(footprint, {
    swathM: geometry.lineSpacingM,
    ...(scout.headingDeg !== null ? { headingDeg: scout.headingDeg } : {}),
  }).lanes;
}

export interface RouteStats {
  workingM: number;
  transitM: number;
  minutes: number;
  photos: number;
  batteries: number;
}

/** Length, time, photos and batteries for lanes flown in order (zone after zone). */
export function routeStats(routes: readonly CoverageLane[][], scout: Scout): RouteStats {
  const geometry = scoutGeometry(scout);
  let workingM = 0;
  let transitM = 0;
  let photos = 0;
  let previous: CoverageLane | null = null;
  for (const lanes of routes) {
    for (const lane of lanes) {
      workingM += lane.lengthM;
      photos += Math.floor(lane.lengthM / geometry.photoSpacingM) + 1;
      if (previous) transitM += haversineDistance(previous.b, lane.a);
      previous = lane;
    }
  }
  const minutes = (workingM + transitM) / geometry.speedMs / 60;
  return {
    workingM,
    transitM,
    minutes,
    photos,
    batteries: minutes > 0 ? Math.ceil(minutes / USABLE_BATTERY_MIN) : 0,
  };
}

/** Rings an orbit re-scan flies, as the planner sizes it (`planner.ORBIT_RINGS`). */
export const ORBIT_RINGS = 2;
/** Nearer than this an orbit is not flown (`planner.ORBIT_MIN_STANDOFF_M`), metres. */
export const ORBIT_MIN_STANDOFF_M = 5;
/** Most an orbit's heading turns between two waypoints round a corner: 15°. */
export const ORBIT_STEP_DEG = 15;

/**
 * One executable route of the export (a WPML wayline): waypoints in flying order, the camera
 * pitch it is flown at, and what the nose faces -- along the route, or an orbit's centre.
 * `photos` says where the camera fires: along each lane (waypoint pairs; the legs between
 * lanes take none) or all the way round.
 */
export interface ScoutWayline {
  waypoints: LonLat[];
  gimbalPitchDeg: number;
  poi: LonLat | null;
  photos: "lanes" | "along";
}

/**
 * How far out an orbit flies from the outline: where a camera pitched `gimbalPitchDeg` down
 * from the capture's height meets the ground, never nearer than 5 m (`planner._orbit_standoff_m`).
 */
export function orbitStandoffM(scout: Scout): number {
  const pitch = (-scout.gimbalPitchDeg * Math.PI) / 180;
  const reach = pitch < Math.PI / 2 ? scoutGeometry(scout).altitudeM / Math.tan(pitch) : 0;
  return Math.max(ORBIT_MIN_STANDOFF_M, reach);
}

const ends = (lanes: readonly CoverageLane[]): LonLat[] => lanes.flatMap((l) => [l.a, l.b]);

type Xy = [number, number];
const cross = (o: Xy, a: Xy, b: Xy) =>
  (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

/** The convex hull of `points`, counter-clockwise, no repeats (monotone chain). */
function convexHull(points: readonly Xy[]): Xy[] {
  const sorted = [...points]
    .sort((a, b) => a[0] - b[0] || a[1] - b[1])
    .filter((p, i, all) => i === 0 || p[0] !== all[i - 1]?.[0] || p[1] !== all[i - 1]?.[1]);
  if (sorted.length < 3) return sorted;
  const half = (list: readonly Xy[]) => {
    const out: Xy[] = [];
    for (const p of list) {
      while (out.length >= 2 && cross(out[out.length - 2] ?? p, out[out.length - 1] ?? p, p) <= 0)
        out.pop();
      out.push(p);
    }
    return out.slice(0, -1);
  };
  return [...half(sorted), ...half([...sorted].reverse())];
}

/**
 * One orbit ring: the outline's convex hull grown by `orbitStandoffM` (the ring the planner
 * sizes, `coverage.orbit_length_m`), its straight sides parallel to the hull's and its corners
 * arcs in steps of at most `ORBIT_STEP_DEG`. Worked flat about the outline's bounds centre (a
 * field is small enough), counter-clockwise, not closed. Empty for an empty outline.
 */
export function orbitRing(footprint: Footprint, scout: Scout): LonLat[] {
  const centre = centerOf(footprint);
  const ky = 111_320;
  const kx = ky * Math.cos((centre.latitude * Math.PI) / 180);
  const points = outerRings(footprint).flatMap((ring) =>
    ring.map(([lon = 0, lat = 0]): Xy => [
      (lon - centre.longitude) * kx,
      (lat - centre.latitude) * ky,
    ]),
  );
  if (points.length === 0) return [];
  const hull = convexHull(points);
  const r = orbitStandoffM(scout);
  const step = (ORBIT_STEP_DEG * Math.PI) / 180;
  const ring: LonLat[] = [];
  const at = ([x, y]: Xy, a: number) =>
    ring.push({
      longitude: centre.longitude + (x + r * Math.cos(a)) / kx,
      latitude: centre.latitude + (y + r * Math.sin(a)) / ky,
    });
  hull.forEach((p, i) => {
    const n = hull.length;
    const prev = hull[(i + n - 1) % n] ?? p;
    const next = hull[(i + 1) % n] ?? p;
    // Outward normals of the sides in and out of this corner (a point: all the way round).
    const from = n > 1 ? Math.atan2(p[1] - prev[1], p[0] - prev[0]) - Math.PI / 2 : 0;
    let to = n > 1 ? Math.atan2(next[1] - p[1], next[0] - p[0]) - Math.PI / 2 : 2 * Math.PI;
    while (to < from - 1e-12) to += 2 * Math.PI;
    for (let a = from; a < to - 1e-9; a += step) at(p, a);
    if (n > 1) at(p, to);
  });
  return ring;
}

/**
 * The passes of a scout area as the map draws them, which are what the export flies: nadir
 * lanes; an oblique's lanes then its crossing set (a pass number apart, so the two are drawn
 * as separate strokes); one orbit ring as consecutive legs (one stroke round).
 */
export function scoutPasses(footprint: Footprint, scout: Scout): CoverageLane[] {
  if (scout.pattern === "orbit") {
    const ring = orbitRing(footprint, scout);
    return ring.map((a, pass) => {
      const b = ring[(pass + 1) % ring.length] ?? a;
      return { pass, a, b, lengthM: haversineDistance(a, b) };
    });
  }
  const [first, crossing] = laneSets(footprint, scout);
  const gap = (first.at(-1)?.pass ?? 0) + 2;
  return [...first, ...crossing.map((lane) => ({ ...lane, pass: lane.pass + gap }))];
}

/** An area's lane sets: the lanes the map draws, and for an oblique a crossing set as well. */
export function laneSets(footprint: Footprint, scout: Scout): [CoverageLane[], CoverageLane[]] {
  const swathM = scoutGeometry(scout).lineSpacingM;
  const first = footprintLanes(footprint, {
    swathM,
    ...(scout.headingDeg !== null ? { headingDeg: scout.headingDeg } : {}),
  });
  if (scout.pattern !== "oblique") return [first.lanes, []];
  const crossing = footprintLanes(footprint, { swathM, headingDeg: (first.headingDeg + 90) % 360 });
  return [first.lanes, crossing.lanes];
}

/**
 * The outlines whose ground the relief check samples: every area's, and each orbit's ring,
 * which flies outside its area.
 */
export function reliefOutlines(
  areas: readonly { footprint: Footprint; scout: Scout }[],
): Footprint[] {
  const rings = areas.flatMap(({ footprint, scout }) => {
    const ring = scout.pattern === "orbit" ? orbitRing(footprint, scout) : [];
    const first = ring[0];
    return first
      ? [
          {
            type: "Polygon" as const,
            coordinates: [[...ring, first].map((p) => [p.longitude, p.latitude])],
          },
        ]
      : [];
  });
  return [...areas.map((a) => a.footprint), ...rings];
}

/**
 * The waylines that fly scout areas in order, each as its capture's pattern asks:
 * - nadir: the lanes the map draws, camera straight down; consecutive nadir areas share one
 *   wayline, so a fog scout is the one route it always was;
 * - oblique: those lanes and a crossing set at right angles, two waylines, the camera at the
 *   capture's pitch facing along each lane (four view directions in all);
 * - orbit: `ORBIT_RINGS` laps of its `orbitRing`, closed, the nose on the outline's bounds
 *   centre.
 */
export function scoutWaylines(
  areas: readonly { footprint: Footprint; scout: Scout }[],
): ScoutWayline[] {
  const waylines: ScoutWayline[] = [];
  let nadir: ScoutWayline | null = null;
  for (const { footprint, scout } of areas) {
    const { pattern, gimbalPitchDeg } = scout;
    if (pattern === "orbit") {
      nadir = null;
      const ring = orbitRing(footprint, scout);
      const laps = Array.from({ length: ORBIT_RINGS }, () => ring).flat();
      const poi = centerOf(footprint);
      waylines.push({
        waypoints: [...laps, ...ring.slice(0, 1)],
        gimbalPitchDeg,
        poi,
        photos: "along",
      });
      continue;
    }
    const [first, crossing] = laneSets(footprint, scout);
    if (pattern === "oblique") {
      nadir = null;
      for (const lanes of [first, crossing])
        waylines.push({ waypoints: ends(lanes), gimbalPitchDeg, poi: null, photos: "lanes" });
      continue;
    }
    if (!nadir) {
      nadir = { waypoints: [], gimbalPitchDeg: -90, poi: null, photos: "lanes" };
      waylines.push(nadir);
    }
    nadir.waypoints.push(...ends(first));
  }
  return waylines.filter((w) => w.waypoints.length >= 2);
}

/** Length, time, photos and batteries for waylines flown one after the other. */
export function waylineStats(waylines: readonly ScoutWayline[], scout: Scout): RouteStats {
  const geometry = scoutGeometry(scout);
  let workingM = 0;
  let transitM = 0;
  let photos = 0;
  let previous: LonLat | null = null;
  for (const { waypoints, photos: shoot } of waylines) {
    let along = 0;
    waypoints.forEach((point, i) => {
      const leg = previous ? haversineDistance(previous, point) : 0;
      previous = point;
      // A lane runs from an even waypoint to the next; every other leg is transit.
      if (shoot === "along" ? i > 0 : i % 2 === 1) {
        workingM += leg;
        along += leg;
        if (shoot === "lanes") {
          photos += Math.floor(leg / geometry.photoSpacingM) + 1;
          along = 0;
        }
      } else transitM += leg;
    });
    if (shoot === "along") photos += Math.floor(along / geometry.photoSpacingM) + 1;
  }
  const minutes = (workingM + transitM) / geometry.speedMs / 60;
  return {
    workingM,
    transitM,
    minutes,
    photos,
    batteries: minutes > 0 ? Math.ceil(minutes / USABLE_BATTERY_MIN) : 0,
  };
}

/** Ground one battery covers at this capture, m² (the lanes' share of a battery's time). */
export function scoutBudgetM2(scout: Scout = DEFAULT_SCOUT): number {
  const geometry = scoutGeometry(scout);
  return geometry.lineSpacingM * geometry.speedMs * USABLE_BATTERY_MIN * 60 * TURN_EFFICIENCY;
}

/** Ground samples for the relief check: at most this many for all the areas together. */
export const RELIEF_MAX_SAMPLES = 150;
/** Closest two samples along an outline or across the grid inside it, metres. */
const RELIEF_MIN_STEP_M = 30;

/**
 * Where the ground under scout areas is sampled for the relief check: each area's outline
 * walked at an even step (lane ends, and so the first waypoint, lie on it) and a grid inside
 * it, at most `max` points for all the areas. Fixed by the outlines alone, not by the height
 * or the lanes, so a new height reuses the heights already sampled.
 */
export function reliefPoints(footprints: readonly Footprint[], max = RELIEF_MAX_SAMPLES): LonLat[] {
  const rings = footprints.flatMap((footprint) => outerRings(footprint));
  const at = (p: number[]): LonLat => ({ longitude: p[0] ?? 0, latitude: p[1] ?? 0 });
  let perimeterM = 0;
  for (const ring of rings)
    for (let i = 1; i < ring.length; i++)
      perimeterM += haversineDistance(at(ring[i - 1] ?? []), at(ring[i] ?? []));
  if (perimeterM <= 0) return [];
  const areaM2 = footprints.reduce((sum, footprint) => sum + footprintAreaM2(footprint), 0);
  const edgeStepM = Math.max(RELIEF_MIN_STEP_M, perimeterM / (max / 2));
  const gridStepM = Math.max(RELIEF_MIN_STEP_M, Math.sqrt(areaM2 / (max / 2)));
  const points: LonLat[] = [];
  for (const ring of rings) {
    // Carried from one edge to the next, so the step stays even round a corner.
    let next = 0;
    for (let i = 1; i < ring.length; i++) {
      const a = at(ring[i - 1] ?? []);
      const b = at(ring[i] ?? []);
      const length = haversineDistance(a, b);
      const heading = bearing(a, b);
      for (; next < length; next += edgeStepM) points.push(destination(a, heading, next));
      next -= length;
    }
  }
  for (const footprint of footprints) {
    const { west, south, east, north } = boundsOf(footprint);
    const dlat = gridStepM / 111_320;
    const dlon = gridStepM / (111_320 * Math.cos((((south + north) / 2) * Math.PI) / 180));
    for (let latitude = south + dlat / 2; latitude < north; latitude += dlat)
      for (let longitude = west + dlon / 2; longitude < east; longitude += dlon)
        if (footprintContains(footprint, { longitude, latitude }))
          points.push({ longitude, latitude });
  }
  if (points.length <= max) return points;
  const stride = points.length / max;
  return points.filter((_, i) => Math.floor(i / stride) !== Math.floor((i - 1) / stride));
}

export interface Relief {
  /** Ground at the sample nearest the first waypoint, where the take-off is assumed. */
  start: number;
  min: number;
  max: number;
  /** What was sampled: the drawn 3D surface (trees and roofs) or the bare terrain model. */
  source: "surface" | "terrain";
}

/**
 * The relief under a route from the heights sampled at `reliefPoints`: the lowest and highest
 * ground over its areas and the ground nearest its first waypoint. Null with fewer than two
 * heights.
 */
export function reliefFrom(
  points: readonly LonLat[],
  heights: readonly (number | undefined)[],
  first: LonLat,
  source: Relief["source"],
): Relief | null {
  let start: number | null = null;
  let nearest = Infinity;
  let min = Infinity;
  let max = -Infinity;
  let count = 0;
  points.forEach((point, i) => {
    const height = heights[i];
    if (typeof height !== "number" || !Number.isFinite(height)) return;
    count++;
    min = Math.min(min, height);
    max = Math.max(max, height);
    const distance = haversineDistance(point, first);
    if (distance < nearest) {
      nearest = distance;
      start = height;
    }
  });
  if (count < 2 || start === null) return null;
  return { start, min, max, source };
}
