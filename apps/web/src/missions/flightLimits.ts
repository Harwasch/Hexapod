/**
 * What the DJI export refuses before it offers a download. Drone missions are a stub until
 * there is a drone (`VITE_ENABLE_DRONE_EXPORT`, off; docs/DRONES.md), so these keep a future
 * enablement from inheriting a dangerous default: a route is offered whole or not at all.
 *
 * Loaded only with the drone card (`DroneScoutExport`), never with the globe.
 */

import type { Footprint } from "@twin/contracts";
import type { LonLat } from "@twin/geo";

import {
  laneSets,
  orbitRing,
  orbitStandoffM,
  scoutGeometry,
  scoutWaylines,
  type Scout,
} from "./scout";

/** A scout area as the export flies it, with the name the card calls it by. */
export interface FlownArea {
  name: string;
  footprint: Footprint;
  scout: Scout;
}

/**
 * Why a route cannot be offered because a pass of it could not be planned, naming each area
 * and what is missing; null when every area's passes are planned. A route with an area left
 * out, or an oblique with one of its crossing passes, would fly less than the plan holds.
 */
export function routeRefusal(areas: readonly FlownArea[]): string | null {
  if (areas.length === 0) return "Nothing to fly: there are no scout areas in this route.";
  const gaps = areas.flatMap(({ name, footprint, scout }): string[] => {
    if (scout.pattern === "orbit")
      return orbitRing(footprint, scout).length < 2
        ? [`${name}: no orbit can be drawn round it`]
        : [];
    const [first, crossing] = laneSets(footprint, scout);
    const sets = scout.pattern === "oblique" ? [first, crossing] : [first];
    const planned = sets.filter((lanes) => lanes.length > 0).length;
    if (planned === sets.length) return [];
    if (planned === 0) {
      const { lineSpacingM, altitudeM } = scoutGeometry(scout);
      return [
        `${name}: no lane fits (lines ${Math.round(lineSpacingM)} m apart at ${Math.round(altitudeM)} m), so nothing would be flown over it`,
      ];
    }
    return [`${name}: only one of its two crossing passes fits; the other would have no lane`];
  });
  return gaps.length > 0
    ? `Not every pass could be planned, so no route is offered. ${gaps.join(". ")}.`
    : null;
}

type Xy = [number, number];

/** How far a sample may sit outside the boundary or inside a hole, metres: lane ends lie on
 * outlines, so a point on one (to rounding) is on the right side of it. */
const EDGE_TOLERANCE_M = 1;
/** Closest two samples along a leg, metres, and the most samples a whole route takes. */
const SAMPLE_STEP_M = 1;
const MAX_SAMPLES = 200_000;

/** The polygons' rings: each polygon's outer ring, then its holes. */
function polygonsOf(footprint: Footprint): number[][][][] {
  return footprint.type === "Polygon" ? [footprint.coordinates] : footprint.coordinates;
}

function insideRing([x, y]: Xy, ring: readonly Xy[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i] ?? [0, 0];
    const [xj, yj] = ring[j] ?? [0, 0];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function nearRing([x, y]: Xy, ring: readonly Xy[], within: number): boolean {
  for (let i = 1; i < ring.length; i++) {
    const [ax, ay] = ring[i - 1] ?? [0, 0];
    const [bx, by] = ring[i] ?? [0, 0];
    const dx = bx - ax;
    const dy = by - ay;
    const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy || 1)));
    if (Math.hypot(x - ax - t * dx, y - ay - t * dy) <= within) return true;
  }
  return false;
}

/**
 * Why a route would fly where it must not, or null when every leg stays inside `limits` (the
 * site's or Land's boundary) and out of its holes and the areas' own holes: holes are no-fly.
 * Each area's waylines (`scoutWaylines`, what the KMZ writes) are walked leg by leg, and the
 * legs between them, at most a metre between samples; a sample within a metre of an outline
 * counts as on its right side. With no boundary to check against, nothing is offered.
 */
export function flightRefusal(
  areas: readonly FlownArea[],
  limits: Footprint | null,
): string | null {
  if (!limits) return "No site boundary to keep the route inside, so no route is offered.";
  const first = polygonsOf(limits)[0]?.[0]?.[0] ?? [0, 0];
  const ky = 111_320;
  const kx = ky * Math.cos(((first[1] ?? 0) * Math.PI) / 180);
  const xy = ([lon = 0, lat = 0]: readonly number[]): Xy => [
    (lon - (first[0] ?? 0)) * kx,
    (lat - (first[1] ?? 0)) * ky,
  ];
  const outers: Xy[][] = [];
  const holes: Xy[][] = [];
  for (const footprint of [limits, ...areas.map((a) => a.footprint)])
    for (const [outer, ...inner] of polygonsOf(footprint)) {
      if (footprint === limits && outer) outers.push(outer.map(xy));
      holes.push(...inner.map((ring) => ring.map(xy)));
    }
  const fault = (p: Xy): "outside" | "hole" | null => {
    if (
      !outers.some((r) => insideRing(p, r)) &&
      !outers.some((r) => nearRing(p, r, EDGE_TOLERANCE_M))
    )
      return "outside";
    if (holes.some((r) => insideRing(p, r) && !nearRing(p, r, EDGE_TOLERANCE_M))) return "hole";
    return null;
  };
  const legs: { a: LonLat; b: LonLat; area: FlownArea; from: FlownArea | null }[] = [];
  let last: { point: LonLat; area: FlownArea } | null = null;
  for (const area of areas)
    for (const { waypoints } of scoutWaylines([area])) {
      waypoints.forEach((point, i) => {
        const before = i > 0 ? waypoints[i - 1] : last?.point;
        if (before)
          legs.push({ a: before, b: point, area, from: i > 0 ? null : (last?.area ?? null) });
      });
      const end = waypoints.at(-1);
      if (end) last = { point: end, area };
    }
  const lengths = legs.map(({ a, b }) => {
    const [ax, ay] = xy([a.longitude, a.latitude]);
    const [bx, by] = xy([b.longitude, b.latitude]);
    return Math.hypot(bx - ax, by - ay);
  });
  const step = Math.max(SAMPLE_STEP_M, lengths.reduce((sum, l) => sum + l, 0) / MAX_SAMPLES);
  for (const [i, { a, b, area, from }] of legs.entries()) {
    const pa = xy([a.longitude, a.latitude]);
    const pb = xy([b.longitude, b.latitude]);
    const n = Math.max(1, Math.ceil((lengths[i] ?? 0) / step));
    for (let k = 0; k <= n; k++) {
      const found = fault([pa[0] + ((pb[0] - pa[0]) * k) / n, pa[1] + ((pb[1] - pa[1]) * k) / n]);
      if (!found) continue;
      const what = found === "hole" ? "cross a no-fly hole" : "leave the site boundary";
      const why =
        from && from !== area
          ? `The leg from ${from.name} to ${area.name} would ${what}`
          : area.scout.pattern === "orbit"
            ? `${area.name}: its orbit flies ${Math.round(orbitStandoffM(area.scout))} m out from the area and would ${what}`
            : found === "hole"
              ? `${area.name}: a leg between its passes would ${what}`
              : `${area.name}: its route would ${what}`;
      return `${why}. The drone must stay inside the boundary and out of its holes, so no route is offered.`;
    }
  }
  return null;
}
