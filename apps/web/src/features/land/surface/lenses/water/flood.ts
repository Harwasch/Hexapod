import type { Footprint } from "@twin/contracts";
import { boundsOf, polygonsOf } from "@twin/geo";

import type { LandPoint } from "@/state/land";

import { anchorOf } from "../../geometryAnchor";
import { areaLabel } from "../../landFlow";
import { sideOfLand, type FloodArea, type SourceState } from "../overviewBundle";

/**
 * FEMA flood zones in plain words. Zones A* are the 1% yearly ("100-year") flood zones, V*
 * the same on the coast with waves; together they are FEMA's special flood hazard area.
 * Shaded X (and old B) is moderate risk; unshaded X (and old C) minimal; D not studied.
 */
export type FloodRisk = "high" | "coastal" | "moderate" | "minimal" | "undetermined" | "other";

export function floodRisk(
  area: Pick<FloodArea, "zone" | "subtype" | "specialFloodHazard">,
): FloodRisk {
  const zone = area.zone.trim().toUpperCase();
  const subtype = area.subtype.toUpperCase();
  if (/^V[A-Z0-9]{0,3}$/.test(zone)) return "coastal";
  if (/^A[A-Z0-9]{0,3}$/.test(zone)) return "high";
  if (area.specialFloodHazard === true) return "high";
  if (zone === "D") return "undetermined";
  if (zone === "B") return "moderate";
  if (zone.startsWith("X"))
    return /0\.2|LEVEE|REDUCED|1 PCT|SHADED/.test(subtype) ? "moderate" : "minimal";
  if (zone === "C") return "minimal";
  return "other";
}

/** The risks drawn on the map and listed as facts, strongest first. */
export const SHOWN_RISKS = ["coastal", "high", "moderate", "undetermined"] as const;
export type ShownRisk = (typeof SHOWN_RISKS)[number];

export const RISK_COLORS: Record<ShownRisk, string> = {
  coastal: "#2f6fe0",
  high: "#4f9cf9",
  moderate: "#a8d1ff",
  undetermined: "#c6ced8",
};

export const RISK_LABELS: Record<ShownRisk, string> = {
  coastal: "Coastal flood zone",
  high: "Mapped flood zone",
  moderate: "Moderate flood risk",
  undetermined: "Flood risk not studied",
};

type Span = [number, number];

/** Where a horizontal line at `latitude` is inside the polygon (even-odd over its rings). */
function spansAt(polygon: number[][][], latitude: number): Span[] {
  const crossings: number[] = [];
  for (const ring of polygon)
    for (let i = 0; i + 1 < ring.length; i++) {
      const [x1 = 0, y1 = 0] = ring[i] ?? [];
      const [x2 = 0, y2 = 0] = ring[i + 1] ?? [];
      if (y1 > latitude === y2 > latitude) continue;
      crossings.push(x1 + ((latitude - y1) / (y2 - y1)) * (x2 - x1));
    }
  crossings.sort((a, b) => a - b);
  const spans: Span[] = [];
  for (let i = 0; i + 1 < crossings.length; i += 2)
    spans.push([crossings[i] ?? 0, crossings[i + 1] ?? 0]);
  return spans;
}

function union(spans: Span[]): Span[] {
  const sorted = [...spans].sort((a, b) => a[0] - b[0]);
  const merged: Span[] = [];
  for (const [start, end] of sorted) {
    const last = merged.at(-1);
    if (last && start <= last[1]) last[1] = Math.max(last[1], end);
    else merged.push([start, end]);
  }
  return merged;
}

function intersect(a: Span[], b: Span[]): Span[] {
  const out: Span[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    const [a0, a1] = a[i] ?? [0, 0];
    const [b0, b1] = b[j] ?? [0, 0];
    const start = Math.max(a0, b0);
    const end = Math.min(a1, b1);
    if (end > start) out.push([start, end]);
    if (a1 < b1) i++;
    else j++;
  }
  return out;
}

/**
 * The share of the land the shapes cover together, counting overlaps once, and the centre of
 * what they cover. Scans the land in thin east–west strips (weighted by the cosine of the
 * latitude), so a shape drawn twice, or two zones that overlap, are never added up.
 */
export function coverage(
  land: Footprint,
  shapes: readonly Footprint[],
  rows = 360,
): { fraction: number; center: LandPoint | null } {
  const bounds = boundsOf(land);
  const height = bounds.north - bounds.south;
  if (!(height > 0) || !shapes.length) return { fraction: 0, center: null };
  const landPolygons = polygonsOf(land);
  const shapePolygons = shapes.flatMap((shape) =>
    polygonsOf(shape).map((polygon) => {
      const b = boundsOf({ type: "Polygon", coordinates: polygon });
      return { polygon, south: b.south, north: b.north };
    }),
  );
  const step = height / rows;
  let landTotal = 0;
  let covered = 0;
  let sumX = 0;
  let sumY = 0;
  for (let row = 0; row < rows; row++) {
    const latitude = bounds.south + (row + 0.5) * step;
    const weight = Math.cos((latitude * Math.PI) / 180);
    const inside = union(landPolygons.flatMap((polygon) => spansAt(polygon, latitude)));
    if (!inside.length) continue;
    for (const [start, end] of inside) landTotal += (end - start) * weight;
    const crossing = shapePolygons.filter(
      (shape) => shape.south <= latitude && shape.north >= latitude,
    );
    if (!crossing.length) continue;
    const shaded = intersect(
      inside,
      union(crossing.flatMap((shape) => spansAt(shape.polygon, latitude))),
    );
    for (const [start, end] of shaded) {
      const length = (end - start) * weight;
      covered += length;
      sumX += ((start + end) / 2) * length;
      sumY += latitude * length;
    }
  }
  return {
    fraction: landTotal > 0 ? Math.min(1, covered / landTotal) : 0,
    center: covered > 0 ? [sumX / covered, sumY / covered] : null,
  };
}

export interface FloodGroup {
  /** One area estimate for all the pieces, overlaps counted once. */
  areaM2: number;
  /** Share of the land, 0–1. */
  share: number;
  /** Where the pieces sit, in words ("the southwest"), or null when they cover most of it. */
  side: string | null;
  geometry: Footprint;
  anchor: LandPoint;
  areas: FloodArea[];
}

/**
 * Measures a set of flood pieces against the land. The strip scan gives the share; the
 * result is held between the largest single piece (the union can't be smaller) and the sum
 * of the pieces (overlaps only shrink it), using FEMA's own geodesic areas for both.
 */
export function measureGroup(
  areas: readonly FloodArea[],
  land: { boundary: Footprint; areaM2: number },
): FloodGroup | null {
  if (!areas.length) return null;
  const largest = Math.max(...areas.map((area) => area.areaM2));
  const total = areas.reduce((sum, area) => sum + area.areaM2, 0);
  const { fraction, center } = coverage(
    land.boundary,
    areas.map((area) => area.geometry),
  );
  const estimate = land.areaM2 > 0 ? fraction * land.areaM2 : largest;
  const areaM2 = Math.min(
    Math.max(estimate, largest),
    total,
    land.areaM2 > 0 ? land.areaM2 : total,
  );
  const share = land.areaM2 > 0 ? Math.min(1, areaM2 / land.areaM2) : 0;
  const geometry: Footprint = {
    type: "MultiPolygon",
    coordinates: areas.flatMap((area) => polygonsOf(area.geometry)),
  };
  const biggest = [...areas].sort((a, b) => b.areaM2 - a.areaM2)[0];
  const side = center && share < 0.6 ? sideOfLand(center, land.boundary) : null;
  return {
    areaM2,
    share,
    side: side === "the middle" ? null : side,
    geometry,
    anchor: anchorOf(biggest?.geometry ?? geometry),
    areas: [...areas],
  };
}

/** One distinct FEMA zone on the land, for a pin ("Zone AE · 3.1 ac"). */
export interface FloodZonePin {
  key: string;
  zone: string;
  risk: ShownRisk;
  areaM2: number;
  anchor: LandPoint;
  geometry: Footprint;
}

export interface FloodSummary {
  /** High and coastal together: FEMA's special flood hazard area. */
  hazard: FloodGroup | null;
  groups: Partial<Record<ShownRisk, FloodGroup>>;
  zones: FloodZonePin[];
  /** Only minimal-hazard (or unclassified) areas came back. */
  minimalOnly: boolean;
  /** Pieces include a floodway, where building is restricted. */
  floodway: boolean;
}

export function summarizeFlood(
  areas: readonly FloodArea[],
  land: { boundary: Footprint; areaM2: number },
): FloodSummary {
  const byRisk = new Map<ShownRisk, FloodArea[]>();
  for (const area of areas) {
    const risk = floodRisk(area);
    if (!(SHOWN_RISKS as readonly string[]).includes(risk)) continue;
    const shown = risk as ShownRisk;
    byRisk.set(shown, [...(byRisk.get(shown) ?? []), area]);
  }
  const groups: Partial<Record<ShownRisk, FloodGroup>> = {};
  for (const risk of SHOWN_RISKS) {
    const group = measureGroup(byRisk.get(risk) ?? [], land);
    if (group) groups[risk] = group;
  }
  const hazard = measureGroup(
    [...(byRisk.get("coastal") ?? []), ...(byRisk.get("high") ?? [])],
    land,
  );
  const zones: FloodZonePin[] = [];
  for (const risk of SHOWN_RISKS) {
    const byZone = new Map<string, FloodArea[]>();
    for (const area of byRisk.get(risk) ?? [])
      byZone.set(area.zone, [...(byZone.get(area.zone) ?? []), area]);
    for (const [zone, pieces] of byZone) {
      const group = measureGroup(pieces, land);
      if (group)
        zones.push({
          key: `${risk}:${zone}`,
          zone,
          risk,
          areaM2: group.areaM2,
          anchor: group.anchor,
          geometry: group.geometry,
        });
    }
  }
  zones.sort((a, b) => b.areaM2 - a.areaM2);
  return {
    hazard,
    groups,
    zones: zones.slice(0, 8),
    minimalOnly: areas.length > 0 && byRisk.size === 0,
    floodway: (byRisk.get("high") ?? []).some((area) => /FLOODWAY/i.test(area.subtype)),
  };
}

/** "3.1 acres", with "1 acre" read properly. */
export function areaWords(squareMetres: number): string {
  return areaLabel(squareMetres).replace(/^1 acres$/, "1 acre");
}

/** A rounded size for a sentence that already says "about": "103 acres", "3.1 acres". */
export function aboutArea(squareMetres: number): string {
  const acres = squareMetres / 4046.8564224;
  if (acres < 10) return areaWords(squareMetres);
  return `${Math.round(acres).toLocaleString()} acres`;
}

/** The short form for a pin: "3.1 ac". */
export function areaShort(squareMetres: number): string {
  return areaLabel(squareMetres).replace(/ acres$/, " ac");
}

function verb(label: string, plural: string, singular: string): string {
  return label.endsWith("acres") ? plural : singular;
}

/** What one group of zones means, in a few words for its fact. */
export function riskMeaning(risk: ShownRisk, group: FloodGroup): string {
  if (risk === "coastal") return "Coastal wave risk";
  if (risk === "high")
    return group.areas.every((area) => /^A[OH]$/i.test(area.zone.trim()))
      ? "1% yearly chance of shallow flooding"
      : "1% yearly flood chance";
  if (risk === "moderate")
    return group.areas.every((area) => /LEVEE/i.test(area.subtype))
      ? "Lower risk behind a levee"
      : "0.2% yearly flood chance";
  return "Possible flood risk";
}

/**
 * The Water lens's sentence about flooding, from what FEMA returned. "Empty" means FEMA maps
 * no flood zone on the land; "uncovered" and "unavailable" mean it could not be checked.
 */
export function floodHeadline(
  source: SourceState,
  summary: FloodSummary | null,
  finished: boolean,
): string | null {
  if (source.status === "uncovered")
    return /smaller/i.test(source.message ?? "")
      ? "This land is too large for a flood-zone check."
      : "Flood maps don't cover this land.";
  if (source.status === "unavailable") return "Flood zones couldn't be checked.";
  if (source.status === "empty") return "No mapped flood zone on this land.";
  if (!summary) return finished ? "Flood zones couldn't be checked." : null;
  const { hazard, groups } = summary;
  if (hazard) {
    if (hazard.share >= 0.95) return "Nearly all of this land is in a mapped flood zone.";
    const size = aboutArea(hazard.areaM2);
    if (hazard.share >= 0.6) return `Most of this land, about ${size}, is in a mapped flood zone.`;
    return `About ${size}${hazard.side ? ` in ${hazard.side}` : ""} ${verb(size, "are", "is")} in a mapped flood zone.`;
  }
  const moderate = groups.moderate;
  if (moderate) {
    if (moderate.share >= 0.95) return "This land has moderate flood risk throughout.";
    const size = aboutArea(moderate.areaM2);
    return `About ${size}${moderate.side ? ` in ${moderate.side}` : ""} ${verb(size, "have", "has")} moderate flood risk.`;
  }
  if (groups.undetermined) return "Flood risk on this land hasn't been studied.";
  return "No mapped flood zone on this land.";
}

/** A FEMA subtype ("0.2 PCT ANNUAL CHANCE FLOOD HAZARD") in sentence case. */
export function subtypeWords(subtype: string): string {
  const words = subtype
    .trim()
    .toLowerCase()
    .replace(/\bpct\b/g, "percent");
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : "";
}
