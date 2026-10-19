/**
 * A saved Land as mission ground.
 *
 * The API's `GET /land/{id}/ground` turns a Land's boundary and inventory features into
 * plannable items; this module makes them zones (private, labelled with where they came
 * from), decides whether they join the current mission project or become a project of their
 * own, and finds the zone a goal or a click means. Pure functions: the store and the flows
 * (`state/mission.ts`, `features/mission/landPlanning.ts`) do the rest.
 */

import type { LandArea, LandGround } from "@twin/contracts";
import {
  bearing,
  boundsOf,
  centerOf,
  destination,
  footprintAreaM2,
  footprintContains,
  haversineDistance,
  toAcres,
  type LonLat,
} from "@twin/geo";

import { WORK_VERBS } from "@/lib/intents";

import { ANYWHERE_PROJECT_ID } from "./anywhere";
import { isAreaZone } from "./areas";
import type { Project, Zone } from "./types";

export const LAND_ZONE_PREFIX = "L-";
export const LAND_PROJECT_PREFIX = "land:";

export function landProjectId(landId: string): string {
  return `${LAND_PROJECT_PREFIX}${landId}`;
}

/** The land a `land:<id>` project is for, or null for any other project. */
export function landIdOfProject(projectId: string | null | undefined): string | null {
  return projectId?.startsWith(LAND_PROJECT_PREFIX)
    ? projectId.slice(LAND_PROJECT_PREFIX.length)
    : null;
}

export function isLandZone(zone: Pick<Zone, "id" | "land">): boolean {
  return zone.land !== undefined || zone.id.startsWith(LAND_ZONE_PREFIX);
}

/** Ground that must stay in its workspace: a Land's zones and anything derived from them. */
export function isPrivateZone(zone: Pick<Zone, "private" | "land">): boolean {
  return Boolean(zone.private) || zone.land !== undefined;
}

/**
 * The point halfway along a line, by distance. A line feature's chip sits there, on the line:
 * the middle of its bounds can be far off it (a fence round three sides of a ranch).
 */
export function halfwayAlong(coordinates: readonly (readonly number[])[]): LonLat {
  const points = coordinates.map(([longitude = 0, latitude = 0]) => ({ longitude, latitude }));
  const segments = points.slice(1).map((to, i) => {
    const from = points[i] ?? to;
    return { from, to, length: haversineDistance(from, to) };
  });
  let remaining = segments.reduce((sum, segment) => sum + segment.length, 0) / 2;
  for (const { from, to, length } of segments) {
    if (remaining <= length) {
      return length > 0 ? destination(from, bearing(from, to), remaining) : from;
    }
    remaining -= length;
  }
  return points.at(-1) ?? { longitude: 0, latitude: 0 };
}

/** The ground's items as zones: the boundary in teal, features in amber, all private. */
export function landZones(ground: LandGround): Zone[] {
  return ground.items.map((item) => ({
    id: item.zoneId,
    name: item.name,
    short: item.kind === "boundary" ? "land" : (item.category ?? "feature"),
    footprint: item.footprint,
    tone: item.kind === "boundary" ? "teal" : "amber",
    treated: false,
    // Tenths, not zoneFromFootprint's whole acres: a 10 m corridor along a fence is small.
    acres: Math.round(toAcres(item.areaM2) * 10) / 10,
    task: "",
    machines: "unassigned",
    progressPct: 0,
    note: item.source,
    attribution: item.source,
    anchor: item.line ? halfwayAlong(item.line.coordinates) : centerOf(item.footprint),
    land: {
      kind: item.kind,
      landId: ground.landId,
      featureId: item.featureId ?? null,
      boundaryRevision: ground.boundaryRevision,
      revision: item.revision,
      line: item.line ?? null,
      lengthM: item.lengthM ?? null,
      bufferM: item.bufferM ?? null,
    },
    private: true,
  }));
}

/** The boundary zone of a Land's ground, if it is among `zones`. */
export function boundaryZoneOf(zones: readonly Zone[], landId?: string): Zone | null {
  return (
    zones.find((z) => z.land?.kind === "boundary" && (!landId || z.land.landId === landId)) ?? null
  );
}

/**
 * A mission project for a saved Land with no fleet of its own, modelled on `siteProject`:
 * the Land's zones, no machines, plans from the workspace. No numbers in `meta`; the badge
 * shows the Land's area in the viewer's units (`ProjectCard`).
 */
export function landProject(land: LandArea, ground: LandGround): Project {
  const zones = landZones(ground);
  return {
    id: landProjectId(land.id),
    siteId: null,
    name: land.name,
    meta: "saved land · no fleet registered",
    simulated: false,
    machines: [],
    zones,
    plans: [],
    agent: {
      headline: "Agent idle",
      summary: "Nothing running",
      footer: "Plans on this land stay in your workspace. Tell me what to do there.",
      actions: [],
    },
    fleetStats: [
      { value: "0", label: "Machines" },
      { value: String(zones.length), label: zones.length === 1 ? "Zone" : "Zones" },
    ],
    fleetNote:
      "No machines are registered for this land; plans draft without assignments until a robot bridge registers them.",
    workLog: [],
    feeds: [],
  };
}

function overlaps(
  a: { west: number; south: number; east: number; north: number },
  b: { west: number; south: number; east: number; north: number },
): boolean {
  return a.west <= b.east && b.west <= a.east && a.south <= b.north && b.south <= a.north;
}

/**
 * Whether a Land joins the current project ("layer": its fleet and zones stay, the Land's
 * zones are added) or becomes the project ("project"). It layers only onto a project whose
 * own ground or machines are on the Land.
 */
export function groundModeFor(project: Project | null, land: LandArea): "layer" | "project" {
  if (!project || project.id === ANYWHERE_PROJECT_ID || landIdOfProject(project.id) !== null)
    return "project";
  const bounds = boundsOf(land.boundary);
  const zoneOverlaps = project.zones.some(
    (z) => !isAreaZone(z) && !isLandZone(z) && overlaps(boundsOf(z.footprint), bounds),
  );
  const machineInside = project.machines.some(
    (m) =>
      m.position.longitude >= bounds.west &&
      m.position.longitude <= bounds.east &&
      m.position.latitude >= bounds.south &&
      m.position.latitude <= bounds.north,
  );
  return zoneOverlaps || machineInside ? "layer" : "project";
}

/** Words that say nothing about which ground: articles, time, and linking words. */
const STOP_WORDS = new Set(
  (
    "the a an every each week weekly day daily month monthly this that my our along around near " +
    "with before after by all and or of on in at to for from"
  ).split(" "),
);
/** Words a zone name shares with too much to pick it by. */
const GENERIC_WORDS = new Set(["area", "zone", "land", "untitled", "boundary"]);
/** Goal words that name a kind of thing an inventory category holds. */
const CATEGORY_WORDS: Record<string, string[]> = {
  water: ["creek", "pond", "trough", "tank", "stream", "spring"],
  transport: ["road", "track", "gate", "trail"],
  power: ["solar", "panel", "pole", "power"],
  building: ["barn", "house", "shed"],
  vegetation: ["tree", "oak", "hedge", "orchard"],
};
const WORK_VERB = new RegExp(`^(?:${WORK_VERBS})$`);

/** "troughs" and "trough" are one word here. */
function stem(word: string): string {
  return word.length > 3 && word.endsWith("s") && !word.endsWith("ss") ? word.slice(0, -1) : word;
}

function words(text: string): string[] {
  return (text.toLowerCase().match(/[a-z]+/g) ?? []).filter(
    (w) => w.length >= 3 && !STOP_WORDS.has(w),
  );
}

/**
 * The zones a goal names ("monitor the fence every week" → the North fence), by the words of
 * their names or the kind of thing their category holds ("check the troughs" → a water
 * feature). Name matches come first; at most five.
 */
export function zonesNamedInGoal(goal: string, zones: readonly Zone[]): string[] {
  const asked = new Set(
    words(goal)
      .filter((w) => !WORK_VERB.test(w))
      .map(stem),
  );
  if (asked.size === 0) return [];
  const byName: string[] = [];
  const byKind: string[] = [];
  for (const zone of zones) {
    const named = words(zone.name)
      .filter((w) => !GENERIC_WORDS.has(w))
      .map(stem);
    if (named.some((w) => asked.has(w))) {
      byName.push(zone.id);
      continue;
    }
    const kinds = CATEGORY_WORDS[zone.short] ?? [];
    if (kinds.some((w) => asked.has(w))) byKind.push(zone.id);
  }
  return [...byName, ...byKind].slice(0, 5);
}

/**
 * A zone the Land itself records: its boundary or an inventory feature. Not an area derived
 * from it (a fog patch, a like-here patch, workable ground), which stays on the project after
 * its card closes and is no saved record of the Land.
 */
function isLandRecordZone(zone: Pick<Zone, "id" | "land">): boolean {
  return isLandZone(zone) && zone.land?.kind !== "derived";
}

/**
 * The smallest zone the Land records under a point (a feature inside the boundary wins), or
 * null. Areas derived from the Land are never taken as its records.
 */
export function landZoneAt(point: LonLat, zones: readonly Zone[]): Zone | null {
  let best: Zone | null = null;
  let bestArea = Infinity;
  for (const zone of zones) {
    if (!isLandRecordZone(zone) || !footprintContains(zone.footprint, point)) continue;
    const area = footprintAreaM2(zone.footprint);
    if (area < bestArea) {
      best = zone;
      bestArea = area;
    }
  }
  return best;
}

/**
 * Zones the mission layer leaves to the Land layer: the selected Land's boundary, which
 * `LandBridge` already draws (one outline, and a click on it stays the Land's).
 */
export function hiddenMissionZoneIds(
  project: Pick<Project, "zones"> | null,
  activeLandId: string | null,
): ReadonlySet<string> {
  if (!project || !activeLandId) return new Set();
  return new Set(
    project.zones
      .filter((z) => z.land?.kind === "boundary" && z.land.landId === activeLandId)
      .map((z) => z.id),
  );
}

/**
 * What the plan card says about a land zone it took as ground. A derived area says it was
 * derived and quotes how; a revision is named only when the zone carries one.
 */
export function landGroundNote(zone: Zone, landName: string): string {
  const land = zone.land;
  if (land?.kind === "derived") {
    const how = zone.attribution ? ` ${zone.attribution}.` : "";
    return `${zone.name}, ${zone.acres.toLocaleString()} ac derived from ${landName}, not a saved record.${how}`;
  }
  if (land?.kind === "boundary") {
    return `${landName}, ${zone.acres.toLocaleString()} ac, saved boundary revision ${land.revision ?? land.boundaryRevision ?? 1}.`;
  }
  const feature = `inventory feature${land?.revision ? ` (revision ${land.revision})` : ""}`;
  if (land?.line && land.lengthM) {
    return `${zone.name}, ${feature}, ${(land.lengthM / 1000).toFixed(1)} km line from your saved land.`;
  }
  return `${zone.name}, ${feature}, ${zone.acres.toLocaleString()} ac from your saved land.`;
}
