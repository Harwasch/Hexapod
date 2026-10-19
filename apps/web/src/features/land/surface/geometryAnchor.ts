import type { Footprint, LandMapGeometry } from "@twin/contracts";
import { footprintContains, polygonsOf, ringAreaM2 } from "@twin/geo";

import type { LandPoint } from "@/state/land";

type Ring = number[][];

function ringCentroid(ring: Ring): LandPoint {
  let x = 0;
  let y = 0;
  let a = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    const [x1 = 0, y1 = 0] = ring[i] ?? [];
    const [x2 = 0, y2 = 0] = ring[i + 1] ?? [];
    const cross = x1 * y2 - x2 * y1;
    a += cross;
    x += (x1 + x2) * cross;
    y += (y1 + y2) * cross;
  }
  if (Math.abs(a) < 1e-12) {
    const first = ring[0] ?? [0, 0];
    return [first[0] ?? 0, first[1] ?? 0];
  }
  return [x / (3 * a), y / (3 * a)];
}

/** The midpoint of the widest span of the polygon along a parallel: always inside it. */
function interiorOnParallel(polygon: Ring[], latitude: number): LandPoint | null {
  const crossings: number[] = [];
  for (const ring of polygon)
    for (let i = 0; i < ring.length - 1; i++) {
      const [x1 = 0, y1 = 0] = ring[i] ?? [];
      const [x2 = 0, y2 = 0] = ring[i + 1] ?? [];
      if (y1 > latitude === y2 > latitude) continue;
      crossings.push(x1 + ((latitude - y1) / (y2 - y1)) * (x2 - x1));
    }
  crossings.sort((a, b) => a - b);
  let best: LandPoint | null = null;
  let width = -1;
  for (let i = 0; i + 1 < crossings.length; i += 2) {
    const left = crossings[i] ?? 0;
    const right = crossings[i + 1] ?? 0;
    if (right - left > width) {
      width = right - left;
      best = [(left + right) / 2, latitude];
    }
  }
  return best;
}

/**
 * Where a card's leader line ends for a shape: a point's own position, a line's middle vertex,
 * or a spot inside an area (its centroid when that falls inside, else inside its widest span).
 */
export function anchorOf(geometry: LandMapGeometry | Footprint): LandPoint {
  if (geometry.type === "Point")
    return [geometry.coordinates[0] ?? 0, geometry.coordinates[1] ?? 0];
  if (geometry.type === "LineString") {
    const middle = geometry.coordinates[Math.floor(geometry.coordinates.length / 2)] ?? [0, 0];
    return [middle[0] ?? 0, middle[1] ?? 0];
  }
  const polygons = polygonsOf(geometry);
  let largest = polygons[0];
  let area = -1;
  for (const polygon of polygons) {
    const outer = polygon[0];
    if (!outer) continue;
    const size = ringAreaM2(outer);
    if (size > area) {
      area = size;
      largest = polygon;
    }
  }
  const outer = largest?.[0];
  if (!largest || !outer) return [0, 0];
  const centroid = ringCentroid(outer);
  const single: Footprint = { type: "Polygon", coordinates: largest };
  if (footprintContains(single, { longitude: centroid[0], latitude: centroid[1] })) return centroid;
  return interiorOnParallel(largest, centroid[1]) ?? centroid;
}
