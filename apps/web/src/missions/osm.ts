/**
 * Candidate areas from OpenStreetMap for the ground in view: water bodies, fields, woodland,
 * parks. The agent picks the kind from the goal; the operator picks the feature. Outlines are
 * © OpenStreetMap contributors (ODbL) and say so on the zone.
 */

import type { Footprint } from "@twin/contracts";
import { footprintAreaM2, toAcres } from "@twin/geo";

import { zoneFromFootprint } from "./areas";
import type { Zone } from "./types";

export type OsmAreaKind = "water" | "farmland" | "wood" | "park";

export const OSM_KINDS: Record<OsmAreaKind, { label: string; selectors: string[] }> = {
  water: {
    label: "Water and shoreline",
    selectors: ['["natural"="water"]', '["natural"="wetland"]'],
  },
  farmland: {
    label: "Fields",
    selectors: ['["landuse"="farmland"]', '["landuse"="orchard"]', '["landuse"="meadow"]'],
  },
  wood: {
    label: "Woodland and scrub",
    selectors: ['["natural"="wood"]', '["landuse"="forest"]', '["natural"="scrub"]'],
  },
  park: {
    label: "Parks and grass",
    selectors: ['["leisure"="park"]', '["landuse"="grass"]', '["landuse"="recreation_ground"]'],
  },
};

export const OSM_ATTRIBUTION = "Outline © OpenStreetMap contributors (ODbL)";
export const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

/** The kind of ground a goal talks about, or null when it does not say. */
export function kindForGoal(goal: string): OsmAreaKind | null {
  const g = goal.toLowerCase();
  if (
    /\b(lake|pond|reservoir|shore|shoreline|coast|coastline|river|creek|water|wetland|marsh)\b/.test(
      g,
    )
  )
    return "water";
  if (/\b(field|fields|farm|farmland|crop|crops|orchard|meadow|pasture|hay)\b/.test(g))
    return "farmland";
  if (/\b(forest|wood|woodland|trees|brush|scrub|chaparral|thistle|weeds|vegetation)\b/.test(g))
    return "wood";
  if (/\b(park|lawn|grass|turf|green)\b/.test(g)) return "park";
  return null;
}

export interface Bbox {
  south: number;
  west: number;
  north: number;
  east: number;
}

export function overpassQuery(bbox: Bbox, kind: OsmAreaKind): string {
  const b = `(${bbox.south},${bbox.west},${bbox.north},${bbox.east})`;
  const parts = OSM_KINDS[kind].selectors.flatMap((sel) => [
    `way${sel}${b};`,
    `relation${sel}${b};`,
  ]);
  return `[out:json][timeout:25];(${parts.join("")});out geom 40;`;
}

interface OverpassGeom {
  lat: number;
  lon: number;
}
interface OverpassWay {
  type: "way";
  id: number;
  tags?: Record<string, string>;
  geometry?: OverpassGeom[];
}
interface OverpassRelation {
  type: "relation";
  id: number;
  tags?: Record<string, string>;
  members?: { type: string; role: string; geometry?: OverpassGeom[] }[];
}
export interface OverpassResponse {
  elements: (OverpassWay | OverpassRelation | { type: string })[];
}

export interface OsmCandidate {
  id: string;
  name: string;
  footprint: Footprint;
  acres: number;
}

function closed(ring: [number, number][]): boolean {
  const a = ring[0];
  const b = ring[ring.length - 1];
  return Boolean(a && b && ring.length >= 4 && a[0] === b[0] && a[1] === b[1]);
}

/** Chains a relation's outer ways into closed rings by matching end points. */
function assembleRings(ways: [number, number][][]): [number, number][][] {
  const pool = ways.map((w) => [...w]);
  const rings: [number, number][][] = [];
  while (pool.length) {
    const ring = pool.shift();
    if (!ring) break;
    let extended = true;
    while (!closed(ring) && extended) {
      extended = false;
      const tail = ring[ring.length - 1];
      if (!tail) break;
      for (let i = 0; i < pool.length; i++) {
        const next = pool[i];
        const head = next?.[0];
        const end = next?.[next.length - 1];
        if (!next || !head || !end) continue;
        if (head[0] === tail[0] && head[1] === tail[1]) {
          ring.push(...next.slice(1));
        } else if (end[0] === tail[0] && end[1] === tail[1]) {
          ring.push(...next.slice(0, -1).reverse());
        } else continue;
        pool.splice(i, 1);
        extended = true;
        break;
      }
    }
    if (closed(ring)) rings.push(ring);
  }
  return rings;
}

function label(tags: Record<string, string> | undefined, fallback: string): string {
  return tags?.name ?? tags?.["name:en"] ?? fallback;
}

/** Candidate polygons from an Overpass reply, largest first, tiny slivers dropped. */
export function parseOverpass(response: OverpassResponse, kind: OsmAreaKind): OsmCandidate[] {
  const out: OsmCandidate[] = [];
  for (const el of response.elements) {
    if (el.type === "way") {
      const way = el as OverpassWay;
      const ring = (way.geometry ?? []).map((g) => [g.lon, g.lat] as [number, number]);
      if (!closed(ring)) continue;
      const footprint: Footprint = { type: "Polygon", coordinates: [ring] };
      out.push({
        id: `osm-w${way.id}`,
        name: label(way.tags, OSM_KINDS[kind].label),
        footprint,
        acres: toAcres(footprintAreaM2(footprint)),
      });
    } else if (el.type === "relation") {
      const rel = el as OverpassRelation;
      const outers = (rel.members ?? [])
        .filter(
          (m) => m.type === "way" && (m.role === "outer" || m.role === "") && m.geometry?.length,
        )
        .map((m) => (m.geometry ?? []).map((g) => [g.lon, g.lat] as [number, number]));
      const rings = assembleRings(outers);
      if (rings.length === 0) continue;
      const footprint: Footprint =
        rings.length === 1
          ? { type: "Polygon", coordinates: [rings[0] ?? []] }
          : { type: "MultiPolygon", coordinates: rings.map((r) => [r]) };
      out.push({
        id: `osm-r${rel.id}`,
        name: label(rel.tags, OSM_KINDS[kind].label),
        footprint,
        acres: toAcres(footprintAreaM2(footprint)),
      });
    }
  }
  return out
    .filter((c) => c.acres >= 0.5)
    .sort((a, b) => b.acres - a.acres)
    .slice(0, 12);
}

/** Fetches candidates for the view; throws on network failure so the UI can say so. */
export async function fetchOsmAreas(
  bbox: Bbox,
  kind: OsmAreaKind,
  signal?: AbortSignal,
): Promise<OsmCandidate[]> {
  const response = await fetch(OVERPASS_URL, {
    method: "POST",
    body: new URLSearchParams({ data: overpassQuery(bbox, kind) }),
    signal: signal ?? null,
  });
  if (!response.ok) throw new Error(`OpenStreetMap lookup failed (${response.status})`);
  return parseOverpass((await response.json()) as OverpassResponse, kind);
}

export function zoneFromCandidate(id: string, candidate: OsmCandidate): Zone {
  return {
    ...zoneFromFootprint(id, candidate.name, candidate.footprint),
    attribution: OSM_ATTRIBUTION,
    note: OSM_ATTRIBUTION,
  };
}
