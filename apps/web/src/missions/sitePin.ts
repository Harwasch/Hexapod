/**
 * The site pin: what the map shows of a project from far away.
 *
 * Zone chips and machine markers are DOM overlays pinned to the ground. From 400 km up a
 * whole site is a few pixels across, and its six markers and three chips land on top of each
 * other in one illegible knot. Above `GLOBE_SCALE_ALTITUDE_M` they collapse into one pin with
 * the site's name and how many things it stands for; below it they are drawn as before.
 */

import type { LonLat, Project } from "./types";

/** Camera altitude above which a project's overlays collapse into its site pin. */
export const GLOBE_SCALE_ALTITUDE_M = 50_000;

/** True when the camera is high enough that the project is drawn as one pin. */
export function collapsesToPin(altitudeM: number): boolean {
  return Number.isFinite(altitudeM) && altitudeM > GLOBE_SCALE_ALTITUDE_M;
}

/**
 * Where the pin stands: the middle of what it replaces — the zones' label anchors, else the
 * machines. Null for a project with neither (the "Anywhere" project before an area is drawn).
 */
export function siteAnchor(project: Pick<Project, "zones" | "machines">): LonLat | null {
  const points: LonLat[] =
    project.zones.length > 0
      ? project.zones.map((zone) => zone.anchor)
      : project.machines.map((machine) => machine.position);
  if (points.length === 0) return null;
  const sum = points.reduce(
    (acc, p) => ({ longitude: acc.longitude + p.longitude, latitude: acc.latitude + p.latitude }),
    { longitude: 0, latitude: 0 },
  );
  return { longitude: sum.longitude / points.length, latitude: sum.latitude / points.length };
}

/** What the pin stands for, counted the way the overlays it replaces would be drawn. */
export function pinCounts(
  project: Pick<Project, "zones" | "machines">,
  zonesShown: boolean,
): { machines: number; zones: number; total: number } {
  const machines = project.machines.length;
  const zones = zonesShown ? project.zones.length : 0;
  return { machines, zones, total: machines + zones };
}

/** The pin's spoken name: "Blackrock Mesa: 6 machines, 3 zones". */
export function pinLabel(name: string, counts: { machines: number; zones: number }): string {
  const parts = [
    counts.machines > 0
      ? `${counts.machines} ${counts.machines === 1 ? "machine" : "machines"}`
      : "",
    counts.zones > 0 ? `${counts.zones} ${counts.zones === 1 ? "zone" : "zones"}` : "",
  ].filter(Boolean);
  return parts.length ? `${name}: ${parts.join(", ")}` : name;
}
