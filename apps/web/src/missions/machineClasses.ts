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
  maxSlopePct: number;
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
 * One line on what a class is assumed to do, e.g. "Robotic brush mower (assumed): 2 m swath at
 * 1.4 m/s, 8 s per turn". A drone's flight height depends on the detail asked for (a scan at
 * 2 cm flies at 70 m), so pass the heights the planner used (`ZoneCoverage.altitudeM`); the
 * line then names those, and the spacing they give, rather than the class default.
 */
export function machineClassSummary(
  id: MachineClassId,
  altitudesM: readonly number[] = [],
): string {
  if (id === "drone") {
    const drone = MACHINE_CLASSES.drone;
    const heights = [...new Set(altitudesM.length ? altitudesM : [drone.altitudeAglM])].sort(
      (a, b) => a - b,
    );
    const flown = heights
      .map((h) => `${h} m above ground (lines ${Math.round(droneLineSpacingM(h))} m apart)`)
      .join(" or ");
    return `${drone.label}: ${flown} at ${drone.speedMps} m/s, ${drone.sidelapPct} % side overlap`;
  }
  const machine = MACHINE_CLASSES[id];
  return `${machine.label}: ${machine.swathM} m swath at ${machine.speedMps} m/s, ${machine.turnS} s per turn`;
}
