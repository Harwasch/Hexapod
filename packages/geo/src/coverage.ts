/**
 * Coverage paths: the boustrophedon (back-and-forth) passes a machine drives to treat a
 * polygon. Used to preview a plan on the map and to size its work; the geometry is a
 * planning preview, not a field-ready route (no headland turns, obstacles or slope).
 *
 * `coverageLanes` and `footprintLanes` are the canonical lanes: every ring counts (even-odd),
 * so a hole or a notch splits a pass instead of being worked over, and every part of a
 * MultiPolygon is covered. The map's preview, the scout route export (DJI WPML) and the
 * rehearsal run these; `coveragePath` (outer ring only) stays for its existing callers.
 * The API's planner measures the same lanes with shapely (`apps/api/app/services/coverage.py`)
 * against the shared fixtures in `packages/geo/fixtures/coverage_cases.json`.
 */

import type { Position } from "geojson";

import { DEG_TO_RAD, EARTH_RADIUS_M, type LonLat } from "./coordinates";
import { boundsOf, polygonsOf, type Footprint } from "./footprint";

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

/** One straight run of work inside the polygon: from `a` to `b` in the direction driven. */
export interface CoverageLane {
  /** Pass number, from 0 in sweep order; a pass split by a hole has several lanes. */
  pass: number;
  a: LonLat;
  b: LonLat;
  lengthM: number;
}

export interface CoverageLanes {
  /** Lanes in driving order: serpentine across passes, along the pass within one. */
  lanes: CoverageLane[];
  headingDeg: number;
  /** Spacing actually used between passes (equals the swath unless capped). */
  swathM: number;
  /** Passes with at least one lane. */
  passes: number;
  /** Sum of the lane lengths, metres. */
  workingLengthM: number;
}

function openRing(ring: Position[]): Position[] {
  const first = ring[0];
  const last = ring[ring.length - 1];
  const closed =
    ring.length > 1 &&
    first !== undefined &&
    last !== undefined &&
    first[0] === last[0] &&
    first[1] === last[1];
  return closed ? ring.slice(0, -1) : ring;
}

/** Sorted x-coordinates where the line at `y` crosses any of the rings (even-odd). */
function crossingsOf(rings: XY[][], y: number): number[] {
  const xs: number[] = [];
  for (const ring of rings) {
    const n = ring.length;
    for (let i = 0; i < n; i++) {
      const p = ring[i];
      const q = ring[(i + 1) % n];
      if (!p || !q) continue;
      if (p.y <= y === q.y <= y) continue;
      xs.push(p.x + ((y - p.y) * (q.x - p.x)) / (q.y - p.y));
    }
  }
  return xs.sort((a, b) => a - b);
}

interface LaneSweep {
  lanes: CoverageLane[];
  passes: number;
  swathM: number;
  workingLengthM: number;
}

/** The lanes of one polygon (outer ring and holes), passes numbered from `firstPass`. */
function sweep(
  polygon: Position[][],
  headingDeg: number,
  options: CoverageOptions,
  firstPass: number,
  forwardFirst: boolean,
): LaneSweep {
  const outer = polygon[0] ? openRing(polygon[0]) : [];
  const empty: LaneSweep = { lanes: [], passes: 0, swathM: options.swathM, workingLengthM: 0 };
  if (outer.length < 3 || !(options.swathM > 0)) return empty;
  const bounds = boundsOf({ type: "Polygon", coordinates: [polygon[0] ?? []] });
  const origin = {
    longitude: (bounds.west + bounds.east) / 2,
    latitude: (bounds.south + bounds.north) / 2,
  };
  const angle = (headingDeg - 90) * DEG_TO_RAD;
  const rings = polygon
    .map(openRing)
    .filter((ring) => ring.length >= 3)
    .map((ring) => project(ring, origin).map((p) => rotate(p, angle)));
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of rings[0] ?? []) {
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const extent = maxY - minY;
  if (!(extent > 0)) return empty;
  let swath = options.swathM;
  const maxPasses = options.maxPasses ?? Infinity;
  if (Math.ceil(extent / swath) > maxPasses) swath = extent / maxPasses;
  const count = Math.max(1, Math.ceil(extent / swath));
  const inverse = -angle;
  const lanes: CoverageLane[] = [];
  let workingLengthM = 0;
  let drawn = 0;
  for (let k = 0; k < count; k++) {
    const y = minY + swath * (k + 0.5);
    if (y > maxY) break;
    const xs = crossingsOf(rings, y);
    const spans: [number, number][] = [];
    for (let i = 0; i + 1 < xs.length; i += 2) {
      const a = xs[i];
      const b = xs[i + 1];
      if (a === undefined || b === undefined || !(b - a > 1e-6)) continue;
      spans.push([a, b]);
    }
    if (spans.length === 0) continue;
    const forward = (drawn % 2 === 0) === forwardFirst;
    const ordered = forward ? spans : [...spans].reverse();
    for (const [a, b] of ordered) {
      const start = unproject(rotate({ x: forward ? a : b, y }, inverse), origin);
      const end = unproject(rotate({ x: forward ? b : a, y }, inverse), origin);
      lanes.push({ pass: firstPass + drawn, a: start, b: end, lengthM: b - a });
      workingLengthM += b - a;
    }
    drawn++;
  }
  return { lanes, passes: drawn, swathM: swath, workingLengthM };
}

/**
 * Lanes over a polygon given as rings (outer first, then holes): passes `swathM` apart along
 * `headingDeg` (default: the long axis), each split wherever a ring crosses it.
 */
export function coverageLanes(polygon: Position[][], options: CoverageOptions): CoverageLanes {
  const headingDeg = options.headingDeg ?? preferredHeadingDeg(polygon[0] ?? []);
  const result = sweep(polygon, headingDeg, options, 0, true);
  return { ...result, headingDeg };
}

/**
 * The spacing a capped sweep of the whole footprint uses: `swathM`, widened so its passes
 * across every part together fit `maxPasses` (the API's `coverage.measure` caps the same way).
 */
function footprintSwath(
  footprint: Footprint,
  headingDeg: number,
  options: CoverageOptions,
): number {
  const maxPasses = options.maxPasses;
  if (maxPasses === undefined || !(options.swathM > 0)) return options.swathM;
  const bounds = boundsOf(footprint);
  const origin = {
    longitude: (bounds.west + bounds.east) / 2,
    latitude: (bounds.south + bounds.north) / 2,
  };
  const angle = (headingDeg - 90) * DEG_TO_RAD;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const polygon of polygonsOf(footprint))
    for (const p of project(openRing(polygon[0] ?? []), origin)) {
      const y = rotate(p, angle).y;
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  const extent = maxY - minY;
  return extent > 0 && Math.ceil(extent / options.swathM) > maxPasses
    ? extent / maxPasses
    : options.swathM;
}

/**
 * Lanes over every polygon of a footprint, in one heading (the whole footprint's long axis
 * unless given); passes are numbered on across the parts and the serpentine carries on. A cap
 * on passes is the whole footprint's: a footprint of many parts (workable ground, a remainder
 * of patches) is drawn at one spacing, as wide as the cap needs across all of them, not at the
 * fine spacing each small part would allow on its own. One polygon sweeps as before.
 */
export function footprintLanes(footprint: Footprint, options: CoverageOptions): CoverageLanes {
  const bounds = boundsOf(footprint);
  const headingDeg =
    options.headingDeg ??
    preferredHeadingDeg([
      [bounds.west, bounds.south],
      [bounds.east, bounds.south],
      [bounds.east, bounds.north],
      [bounds.west, bounds.north],
    ]);
  const polygons = polygonsOf(footprint);
  const shared =
    polygons.length > 1
      ? { swathM: footprintSwath(footprint, headingDeg, options), headingDeg }
      : options;
  const lanes: CoverageLane[] = [];
  let passes = 0;
  let workingLengthM = 0;
  let swathM = options.swathM;
  for (const polygon of polygons) {
    const part = sweep(polygon, headingDeg, shared, passes, passes % 2 === 0);
    lanes.push(...part.lanes);
    passes += part.passes;
    workingLengthM += part.workingLengthM;
    if (part.passes) swathM = Math.max(swathM, part.swathM);
  }
  return { lanes, headingDeg, swathM, passes, workingLengthM };
}

/**
 * Lanes as lines to draw: consecutive passes join at their ends (the turn), while lanes of
 * the same pass stay apart, so nothing is drawn across a hole or a notch.
 */
export function laneStrokes(lanes: readonly CoverageLane[]): LonLat[][] {
  const strokes: LonLat[][] = [];
  let current: LonLat[] = [];
  let previous: CoverageLane | null = null;
  for (const lane of lanes) {
    const turn = previous !== null && lane.pass === previous.pass + 1;
    if (!turn && current.length) {
      strokes.push(current);
      current = [];
    }
    current.push(lane.a, lane.b);
    previous = lane;
  }
  if (current.length) strokes.push(current);
  return strokes;
}
