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
import { footprintLanes, haversineDistance, type CoverageLane } from "@twin/geo";

import { MACHINE_CLASSES } from "./machineClasses";

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
  camera: "dji-m3e-wide",
  altitudeM: drone.altitudeAglM,
  forwardOverlap: drone.frontlapPct / 100,
  sideOverlap: drone.sidelapPct / 100,
  speedMs: drone.speedMps,
  headingDeg: null,
};

/** What a scout area's chip on the map says: its passes are a forecast, not evidence. */
export const SCOUT_SHORT = "would lift fog";

/** Heights the export offers, metres above the take-off point. */
export const SCOUT_ALTITUDES_M = [45, 60, 90, 120] as const;
/** US Part 107: 400 ft above ground. Heights here are above take-off, so this is the cap. */
export const MAX_ALTITUDE_M = 120;
export const MIN_ALTITUDE_M = 20;
/** Flying time one battery gives with a reserve left: an assumption, labelled as one. */
export const USABLE_BATTERY_MIN = 25;
/** Share of a battery spent on lanes rather than turns and transit, for the budget. */
export const TURN_EFFICIENCY = 0.8;

export function scoutOf(capture: ScoutCapture | null | undefined): Scout {
  return {
    ...DEFAULT_SCOUT,
    ...(capture ?? {}),
    purpose: "lift-knowledge-fog",
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
    warnings,
  };
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

/** Ground one battery covers at this capture, m² (the lanes' share of a battery's time). */
export function scoutBudgetM2(scout: Scout = DEFAULT_SCOUT): number {
  const geometry = scoutGeometry(scout);
  return geometry.lineSpacingM * geometry.speedMs * USABLE_BATTERY_MIN * 60 * TURN_EFFICIENCY;
}
