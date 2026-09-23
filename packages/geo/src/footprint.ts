/** GeoJSON polygon utilities (RFC 7946). */

import type { MultiPolygon, Polygon, Position } from "geojson";

import { destination, type LonLat, DEG_TO_RAD, EARTH_RADIUS_M } from "./coordinates";

export type Footprint = Polygon | MultiPolygon;

export interface Bounds {
  west: number;
  south: number;
  east: number;
  north: number;
}

export function polygonsOf(footprint: Footprint): Position[][][] {
  return footprint.type === "Polygon" ? [footprint.coordinates] : footprint.coordinates;
}

export function outerRings(footprint: Footprint): Position[][] {
  return polygonsOf(footprint)
    .map((polygon) => polygon[0])
    .filter((ring): ring is Position[] => Array.isArray(ring));
}

export function boundsOf(footprint: Footprint): Bounds {
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  for (const ring of outerRings(footprint)) {
    for (const [lon, lat] of ring) {
      if (lon === undefined || lat === undefined) continue;
      west = Math.min(west, lon);
      east = Math.max(east, lon);
      south = Math.min(south, lat);
      north = Math.max(north, lat);
    }
  }
  return { west, south, east, north };
}

export function centerOf(footprint: Footprint): LonLat {
  const { west, south, east, north } = boundsOf(footprint);
  return { longitude: (west + east) / 2, latitude: (south + north) / 2 };
}

/** Radius (metres) of the smallest circle around the bounds centre that covers every vertex. */
export function boundingRadiusM(footprint: Footprint): number {
  const center = centerOf(footprint);
  let radius = 0;
  for (const ring of outerRings(footprint)) {
    for (const [lon, lat] of ring) {
      if (lon === undefined || lat === undefined) continue;
      const dx = (lon - center.longitude) * DEG_TO_RAD * Math.cos(center.latitude * DEG_TO_RAD);
      const dy = (lat - center.latitude) * DEG_TO_RAD;
      radius = Math.max(radius, Math.sqrt(dx * dx + dy * dy) * EARTH_RADIUS_M);
    }
  }
  return radius;
}

/** Signed spherical area of a ring in m² (positive = counter-clockwise), after Chamberlain & Duquette. */
export function ringAreaM2(ring: Position[]): number {
  if (ring.length < 3) return 0;
  let total = 0;
  const n = ring.length;
  for (let i = 0; i < n; i++) {
    const p1 = ring[i];
    const p2 = ring[(i + 1) % n];
    const p3 = ring[(i + 2) % n];
    const lon1 = p1?.[0];
    const lat2 = p2?.[1];
    const lon3 = p3?.[0];
    if (lon1 === undefined || lat2 === undefined || lon3 === undefined) continue;
    total += (lon3 - lon1) * DEG_TO_RAD * Math.sin(lat2 * DEG_TO_RAD);
  }
  return (total * EARTH_RADIUS_M * EARTH_RADIUS_M) / 2;
}

export function footprintAreaM2(footprint: Footprint): number {
  let area = 0;
  for (const polygon of polygonsOf(footprint)) {
    polygon.forEach((ring, index) => {
      const ringArea = Math.abs(ringAreaM2(ring));
      area += index === 0 ? ringArea : -ringArea;
    });
  }
  return Math.max(0, area);
}

/** Validates that a value is a Polygon or MultiPolygon with closed rings of ≥4 positions. */
export function validateFootprint(
  value: unknown,
): { ok: true; footprint: Footprint } | { ok: false; error: string } {
  if (typeof value !== "object" || value === null)
    return { ok: false, error: "Footprint must be a GeoJSON object" };
  const candidate = value as {
    type?: unknown;
    coordinates?: unknown;
    features?: unknown;
    geometry?: unknown;
  };
  if (candidate.type === "Feature" && candidate.geometry)
    return validateFootprint(candidate.geometry);
  if (
    candidate.type === "FeatureCollection" &&
    Array.isArray(candidate.features) &&
    candidate.features.length > 0
  ) {
    return validateFootprint(candidate.features[0]);
  }
  if (candidate.type !== "Polygon" && candidate.type !== "MultiPolygon") {
    return { ok: false, error: "Footprint must be a Polygon or MultiPolygon" };
  }
  if (!Array.isArray(candidate.coordinates)) return { ok: false, error: "Missing coordinates" };
  const polygons = candidate.type === "Polygon" ? [candidate.coordinates] : candidate.coordinates;
  for (const polygon of polygons) {
    if (!Array.isArray(polygon) || polygon.length === 0)
      return { ok: false, error: "Polygon has no rings" };
    for (const ring of polygon) {
      if (!Array.isArray(ring) || ring.length < 4)
        return { ok: false, error: "Each ring needs at least 4 positions" };
      for (const position of ring) {
        if (
          !Array.isArray(position) ||
          position.length < 2 ||
          typeof position[0] !== "number" ||
          typeof position[1] !== "number"
        ) {
          return { ok: false, error: "Positions must be [longitude, latitude] numbers" };
        }
        const [lon, lat] = position as [number, number];
        if (lon < -180 || lon > 180 || lat < -90 || lat > 90)
          return { ok: false, error: "Coordinates out of range" };
      }
      const first = ring[0] as [number, number];
      const last = ring[ring.length - 1] as [number, number];
      if (first[0] !== last[0] || first[1] !== last[1])
        return { ok: false, error: "Rings must be closed" };
    }
  }
  return { ok: true, footprint: candidate as Footprint };
}

/** Approximate circle polygon around a centre (counter-clockwise, closed). */
export function circleFootprint(center: LonLat, radiusM: number, segments = 32): Polygon {
  const ring: Position[] = [];
  for (let i = 0; i < segments; i++) {
    const point = destination(center, (360 * i) / segments, radiusM);
    ring.push([point.longitude, point.latitude]);
  }
  const first = ring[0];
  if (first) ring.push([first[0] ?? 0, first[1] ?? 0]);
  return { type: "Polygon", coordinates: [ring] };
}

/** Flattened [lon, lat, lon, lat, ...] of the first outer ring — what Cesium's fromDegreesArray wants. */
export function flattenRing(ring: Position[]): number[] {
  const out: number[] = [];
  for (const [lon, lat] of ring) {
    if (lon !== undefined && lat !== undefined) out.push(lon, lat);
  }
  return out;
}

/** True when the point lies inside the outer ring of any polygon (even-odd rule). */
export function footprintContains(footprint: Footprint, point: LonLat): boolean {
  for (const ring of outerRings(footprint)) {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const a = ring[i];
      const b = ring[j];
      if (!a || !b) continue;
      const [ax, ay] = a;
      const [bx, by] = b;
      if (ax === undefined || ay === undefined || bx === undefined || by === undefined) continue;
      const crosses =
        ay > point.latitude !== by > point.latitude &&
        point.longitude < ((bx - ax) * (point.latitude - ay)) / (by - ay) + ax;
      if (crosses) inside = !inside;
    }
    if (inside) return true;
  }
  return false;
}
