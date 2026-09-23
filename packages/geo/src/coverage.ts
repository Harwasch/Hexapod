/**
 * Coverage paths: the boustrophedon (back-and-forth) passes a machine drives to treat a
 * polygon. Used to preview a plan on the map and to size its work; the geometry is a
 * planning preview, not a field-ready route (no headland turns, obstacles or slope).
 */

import type { Position } from "geojson";

import { DEG_TO_RAD, EARTH_RADIUS_M, type LonLat } from "./coordinates";
import { boundsOf } from "./footprint";

export interface CoverageOptions {
  /** Working width of the implement, metres. */
  swathM: number;
  /** Direction of travel of the passes, degrees clockwise from north. Default: the long axis. */
  headingDeg?: number;
  /** Cap on drawn passes; above it the spacing widens (a preview, flagged by `swathM`). */
  maxPasses?: number;
}

export interface CoveragePath {
  /** Serpentine path: pass, turn, pass … in lon/lat. Empty for a degenerate ring. */
  positions: LonLat[];
  /** Number of passes drawn. */
  passes: number;
  /** Spacing actually used between drawn passes (equals the swath unless capped). */
  swathM: number;
  /** Sum of the pass lengths inside the polygon (turns excluded), metres. */
  workingLengthM: number;
  headingDeg: number;
}

interface XY {
  x: number;
  y: number;
}

function project(ring: Position[], origin: LonLat): XY[] {
  const kx = Math.cos(origin.latitude * DEG_TO_RAD) * DEG_TO_RAD * EARTH_RADIUS_M;
  const ky = DEG_TO_RAD * EARTH_RADIUS_M;
  const out: XY[] = [];
  for (const [lon, lat] of ring) {
    if (lon === undefined || lat === undefined) continue;
    out.push({ x: (lon - origin.longitude) * kx, y: (lat - origin.latitude) * ky });
  }
  return out;
}

function unproject(point: XY, origin: LonLat): LonLat {
  const kx = Math.cos(origin.latitude * DEG_TO_RAD) * DEG_TO_RAD * EARTH_RADIUS_M;
  const ky = DEG_TO_RAD * EARTH_RADIUS_M;
  return { longitude: origin.longitude + point.x / kx, latitude: origin.latitude + point.y / ky };
}

function rotate(point: XY, angleRad: number): XY {
  const c = Math.cos(angleRad);
  const s = Math.sin(angleRad);
  return { x: point.x * c - point.y * s, y: point.x * s + point.y * c };
}

/** Travel direction along the polygon's longer bounding side (east-west when wider than tall). */
export function preferredHeadingDeg(ring: Position[]): number {
  const bounds = boundsOf({ type: "Polygon", coordinates: [ring] });
  const midLat = ((bounds.south + bounds.north) / 2) * DEG_TO_RAD;
  const width = (bounds.east - bounds.west) * Math.cos(midLat);
  const height = bounds.north - bounds.south;
  return width >= height ? 90 : 0;
}

/** Sorted x-coordinates where the horizontal line at `y` crosses the polygon boundary. */
function crossings(polygon: XY[], y: number): number[] {
  const xs: number[] = [];
  const n = polygon.length;
  for (let i = 0; i < n; i++) {
    const p = polygon[i];
    const q = polygon[(i + 1) % n];
    if (!p || !q) continue;
    // Half-open rule so a vertex exactly on the line counts once.
    if (p.y <= y === q.y <= y) continue;
    xs.push(p.x + ((y - p.y) * (q.x - p.x)) / (q.y - p.y));
  }
  return xs.sort((a, b) => a - b);
}

export function coveragePath(ring: Position[], options: CoverageOptions): CoveragePath {
  const closed =
    ring.length > 1 &&
    ring[0]?.[0] === ring[ring.length - 1]?.[0] &&
    ring[0]?.[1] === ring[ring.length - 1]?.[1];
  const vertices = closed ? ring.slice(0, -1) : ring;
  const headingDeg = options.headingDeg ?? preferredHeadingDeg(ring);
  const empty: CoveragePath = {
    positions: [],
    passes: 0,
    swathM: options.swathM,
    workingLengthM: 0,
    headingDeg,
  };
  if (vertices.length < 3 || !(options.swathM > 0)) return empty;
  const bounds = boundsOf({ type: "Polygon", coordinates: [ring] });
  const origin = {
    longitude: (bounds.west + bounds.east) / 2,
    latitude: (bounds.south + bounds.north) / 2,
  };
  // Rotate so the travel direction lies along +x; heading is clockwise from north, and
  // north is +y, so travel along heading h is the +x axis after rotating by (h - 90°).
  const angle = (headingDeg - 90) * DEG_TO_RAD;
  const polygon = project(vertices, origin).map((p) => rotate(p, angle));
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of polygon) {
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const extent = maxY - minY;
  if (!(extent > 0)) return empty;
  let swath = options.swathM;
  const wanted = Math.max(1, Math.ceil(extent / swath));
  const maxPasses = options.maxPasses ?? Infinity;
  if (wanted > maxPasses) swath = extent / maxPasses;
  const passes = Math.max(1, Math.ceil(extent / swath));
  const path: XY[] = [];
  let workingLengthM = 0;
  let drawn = 0;
  for (let k = 0; k < passes; k++) {
    const y = minY + swath * (k + 0.5);
    if (y > maxY) break;
    const xs = crossings(polygon, y);
    if (xs.length < 2) continue;
    const spans: [number, number][] = [];
    for (let i = 0; i + 1 < xs.length; i += 2) {
      const a = xs[i];
      const b = xs[i + 1];
      if (a === undefined || b === undefined) continue;
      spans.push([a, b]);
      workingLengthM += b - a;
    }
    const forward = drawn % 2 === 0;
    const ordered = forward ? spans : [...spans].reverse();
    for (const [a, b] of ordered) {
      path.push({ x: forward ? a : b, y }, { x: forward ? b : a, y });
    }
    drawn++;
  }
  const inverse = -angle;
  return {
    positions: path.map((p) => unproject(rotate(p, inverse), origin)),
    passes: drawn,
    swathM: swath,
    workingLengthM,
    headingDeg,
  };
}
