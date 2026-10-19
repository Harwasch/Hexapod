/**
 * The assumed machine classes, read from the one file the planner sizes work with.
 *
 * `apps/api/app/services/machine_classes.json` holds every speed, swath and camera number an
 * outline-sized estimate rests on. The API's planner turns them into machine-hours; the console
 * names them next to those hours (the plan card's PATH list), so the number shown and the
 * assumption it came from are always the same one. None of it is measured: every label says
 * "(assumed)". The visual preview spacing on the map (`PREVIEW_SWATH_M`) is separate and is
 * labelled "preview spacing" where it is drawn.
 *
 * The import is relative on purpose: apps/api is outside this package, and Vite (whose fs
 * allow-list is the workspace root), vitest and tsc all resolve it.
 */

import type { MachineClassId } from "@twin/contracts";

import raw from "../../../api/app/services/machine_classes.json";

export interface GroundMachineClass {
  label: string;
  swathM: number;
  speedMps: number;
  turnS: number;
  /** Steepest slope along or across a lane the class is assumed to work (the rehearsal's). */
  maxSlopePct: number;
  /** How close to a mapped creek a lane is assumed to be allowed, metres (the rehearsal's). */
  creekBufferM: number;
}

export interface DroneMachineClass {
  label: string;
  altitudeAglM: number;
  speedMps: number;
  sidelapPct: number;
  frontlapPct: number;
  cameraHfovDeg: number;
  imageAspect: number;
  maxWindMps: number;
}

export interface MachineClasses {
  mower: GroundMachineClass;
  seeder: GroundMachineClass;
  survey: GroundMachineClass;
  drone: DroneMachineClass;
}

export const MACHINE_CLASSES: MachineClasses = raw;

/** Ground width one drone image covers across the flight line, metres. */
export function droneFootprintWidthM(altitudeM = MACHINE_CLASSES.drone.altitudeAglM): number {
  const drone = MACHINE_CLASSES.drone;
  return 2 * altitudeM * Math.tan((drone.cameraHfovDeg * Math.PI) / 180 / 2);
}

/** Distance between neighbouring flight lines: the footprint less the side overlap, metres. */
export function droneLineSpacingM(altitudeM = MACHINE_CLASSES.drone.altitudeAglM): number {
  return droneFootprintWidthM(altitudeM) * (1 - MACHINE_CLASSES.drone.sidelapPct / 100);
}

export function machineClassLabel(id: MachineClassId): string {
  return MACHINE_CLASSES[id].label;
}

/**
 * A drone height the planner sized a zone's flight at, and what it is measured from: a scout
 * area's capture is flown above its take-off point (its exported route is relative to it), so
 * on sloping ground the clearance differs from the height; every other flight is planned above
 * the ground.
 */
export interface FlightHeight {
  altitudeM: number;
  above: "ground" | "take-off";
}

/**
 * One line on what a class is assumed to do, e.g. "Robotic brush mower (assumed): 2 m swath at
 * 1.4 m/s, 8 s per turn". A drone's flight height depends on the detail asked for (a scan at
 * 2 cm flies at 70 m), so pass the heights the planner used (`ZoneCoverage.altitudeM`) with
 * what each is above; the line then names those, and the spacing they give, rather than the
 * class default.
 */
export function machineClassSummary(
  id: MachineClassId,
  flights: readonly FlightHeight[] = [],
): string {
  if (id === "drone") {
    const drone = MACHINE_CLASSES.drone;
    const listed = flights.length
      ? flights
      : [{ altitudeM: drone.altitudeAglM, above: "ground" as const }];
    const heights = [
      ...new Map(listed.map((f) => [`${f.altitudeM} ${f.above}`, f] as const)).values(),
    ].sort((a, b) => a.altitudeM - b.altitudeM || a.above.localeCompare(b.above));
    const flown = heights
      .map(
        (f) =>
          `${f.altitudeM} m above ${f.above} (lines ${Math.round(droneLineSpacingM(f.altitudeM))} m apart)`,
      )
      .join(" or ");
    return `${drone.label}: ${flown} at ${drone.speedMps} m/s, ${drone.sidelapPct} % side overlap`;
  }
  const machine = MACHINE_CLASSES[id];
  return `${machine.label}: ${machine.swathM} m swath at ${machine.speedMps} m/s, ${machine.turnS} s per turn`;
}
