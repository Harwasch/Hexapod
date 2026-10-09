/**
 * Where a scan's tall things stand, so the photorealistic world's own copies of them are cut
 * away under the scan (ClippingManager, SiteManager.applyClip).
 *
 * A splat scan clips the world tileset with the footprint the catalog has for it: the asset's
 * own, else its site's boundary. For a scan of open ground that is the whole capture, and the
 * world's ground beside it meets the scan's ground. But a tall object is in the world twice:
 * the Minnetonka tree's footprint is a 2.7 m circle round its trunk, and Google's own tree,
 * whose crown reaches further, stood beside the scanned one -- measured on production, its
 * crown 3 to 6 m up at 1 to 3 m outside the circle, to the south and south-west.
 *
 * So the clip also covers the scan's tall parts: its splats more than `TALL_M` above its own
 * ground, from the root tile (the whole scan, merged, a few thousand splats), gathered on a
 * grid into separate things (a tree, a tent, a row of hedges), each outlined by its convex hull
 * and widened by a margin that grows with its height -- the world's copy is another capture,
 * registered on its own, with a crown its mesh smears outwards. Low scans give nothing
 * (`canopyOutlines` is empty), so a scan on open ground keeps the clip it had: no larger hole
 * in the world. Floaters -- a few faint splats in the sky or far out -- are too light to count.
 *
 * Pure: points in the scan's own east/north/up metres in, outlines in the same frame out, and
 * on the globe through the scan's placed transform (`outlinesOnGlobe`, `withOutlines`).
 */

import { Cartesian3, Cartographic, Math as CesiumMath, Matrix4 } from "cesium";

import type { Footprint } from "@twin/contracts";
import { polygonsOf } from "@twin/geo";

/** The scan's splats as a tile decodes them (lib/spzPositions.ts `spzPickData`). */
export interface ScanPoints {
  /** x, y, z per splat: the scan's east, north and up metres. */
  readonly positions: Float32Array;
  /** Opacity after the sigmoid, 0 to 1. */
  readonly opacity: Float32Array;
  /** Each splat's largest axis (m); absent counts as none. */
  readonly radii?: Float32Array;
}

/** A splat this opaque or more is part of what is seen. */
export const MIN_OPACITY = 0.3;
/**
 * Without a ground from the placement, the scan's ground is this low percentile of its opaque
 * splats' heights. A placed scan has one (SiteManager: the terrain it rests on): the root tile
 * of a tree has no ground in it at all, its lowest opaque splats are the crown's.
 */
export const GROUND_PERCENTILE = 0.05;
/** Higher than this above the ground (m), a thing stands out of the world's ground. */
export const TALL_M = 1.5;
/**
 * A thing must rise from near the ground: its lowest tall splat within this of `TALL_M` (m).
 * What hangs higher with nothing under it is a floater in the sky, not a crown.
 */
export const ROOTED_M = 1.5;
/** The height a margin grows with is counted up to this (m): a forest's 40 m pines widen
 *  their outline no more than a 10 m tree does. */
export const MARGIN_HEIGHT_CAP_M = 10;
/** The smallest grid cell things are gathered on (m). */
export const MIN_CELL_M = 0.5;
/** A thing whose splats' opacities sum to less than this is a floater, not a thing. */
export const MIN_THING_OPACITY = 3;
/** Every outline is widened by at least this (m)... */
export const MIN_MARGIN_M = 0.5;
/** ...plus this share of its thing's height above the ground. */
export const MARGIN_PER_HEIGHT = 0.2;
/** Most outlines handed to the clip: the largest things (polygons share one texture). */
export const MAX_OUTLINES = 24;
/** A merged splat counts as no wider than this (m) when it is spread over the grid. */
const MAX_SPREAD_M = 3;
/** Directions the margin is swept in: the outline is the hull of each corner's circle. */
const MARGIN_DIRECTIONS = 16;

/** One thing's outline: a closed ring (first point repeated), counter-clockwise, metres. */
export type Outline = [number, number][];

/**
 * The outlines of a scan's tall things, in its own frame, at least `TALL_M` above its ground
 * (`groundZ`, the height in the scan's frame it rests on, when the placement knows it); empty
 * for a scan with none (open ground), or with too little to tell.
 */
export function canopyOutlines(points: ScanPoints, groundZ: number | null = null): Outline[] {
  const { positions, opacity, radii } = points;
  const count = Math.min(Math.floor(positions.length / 3), opacity.length);
  let opaque = 0;
  for (let i = 0; i < count; i++) if ((opacity[i] ?? 0) >= MIN_OPACITY) opaque += 1;
  if (opaque < 10) return [];
  const ground = groundZ ?? groundOf(points) ?? 0;

  // The tall splats, and how wide each is spread on the ground.
  const tall: { x: number; y: number; z: number; r: number; o: number }[] = [];
  for (let i = 0; i < count; i++) {
    const o = opacity[i] ?? 0;
    const z = positions[i * 3 + 2] ?? 0;
    if (o < MIN_OPACITY || z - ground < TALL_M) continue;
    const x = positions[i * 3] ?? 0;
    const y = positions[i * 3 + 1] ?? 0;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    tall.push({ x, y, z, r: Math.min(MAX_SPREAD_M, Math.max(0, radii?.[i] ?? 0)), o });
  }
  if (tall.length === 0) return [];

  // A grid as coarse as the splats are wide (the root tile's merged splats are wider for a
  // larger scan), never finer than MIN_CELL_M, each splat marking every cell it covers.
  const spreads = tall.map((s) => s.r).sort((a, b) => a - b);
  const cell = Math.max(MIN_CELL_M, spreads[Math.floor(spreads.length / 2)] ?? 0);
  const key = (i: number, j: number): string => `${String(i)},${String(j)}`;
  const cells = new Map<string, { i: number; j: number; mass: number; members: number[] }>();
  tall.forEach((s, n) => {
    const reach = Math.ceil(s.r / cell);
    const ci = Math.floor(s.x / cell);
    const cj = Math.floor(s.y / cell);
    for (let i = ci - reach; i <= ci + reach; i++)
      for (let j = cj - reach; j <= cj + reach; j++) {
        const k = key(i, j);
        let c = cells.get(k);
        if (!c) {
          c = { i, j, mass: 0, members: [] };
          cells.set(k, c);
        }
        if (i === ci && j === cj) c.mass += s.o;
        c.members.push(n);
      }
  });

  // Things: cells joined to their eight neighbours.
  const seen = new Set<string>();
  const things: { members: Set<number>; mass: number }[] = [];
  for (const [start, first] of cells) {
    if (seen.has(start)) continue;
    seen.add(start);
    const members = new Set<number>();
    let mass = 0;
    const queue = [first];
    while (queue.length > 0) {
      const c = queue.pop();
      if (!c) break;
      mass += c.mass;
      for (const m of c.members) members.add(m);
      for (let di = -1; di <= 1; di++)
        for (let dj = -1; dj <= 1; dj++) {
          const k = key(c.i + di, c.j + dj);
          const next = cells.get(k);
          if (next && !seen.has(k)) {
            seen.add(k);
            queue.push(next);
          }
        }
    }
    if (mass >= MIN_THING_OPACITY) things.push({ members, mass });
  }

  return things
    .sort((a, b) => b.mass - a.mass)
    .slice(0, MAX_OUTLINES)
    .map(({ members }) => {
      const corners: [number, number][] = [];
      let top = ground;
      let bottom = Number.POSITIVE_INFINITY;
      for (const n of members) {
        const s = tall[n];
        if (!s) continue;
        top = Math.max(top, s.z);
        bottom = Math.min(bottom, s.z);
        corners.push(
          [s.x - s.r, s.y - s.r],
          [s.x + s.r, s.y - s.r],
          [s.x + s.r, s.y + s.r],
          [s.x - s.r, s.y + s.r],
        );
      }
      // Hanging in the sky with nothing under it: a floater.
      if (bottom - ground > TALL_M + ROOTED_M) return [];
      const height = Math.min(top - ground, MARGIN_HEIGHT_CAP_M);
      return widen(convexHull(corners), MIN_MARGIN_M + MARGIN_PER_HEIGHT * height);
    })
    .filter((ring) => ring.length >= 4);
}

/** The convex hull of `points`, counter-clockwise, not closed (Andrew's monotone chain). */
export function convexHull(points: readonly [number, number][]): [number, number][] {
  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (sorted.length < 3) return sorted;
  const cross = (o: [number, number], a: [number, number], b: [number, number]): number =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: [number, number][] = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower.at(-2) ?? p, lower.at(-1) ?? p, p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper: [number, number][] = [];
  for (let k = sorted.length - 1; k >= 0; k--) {
    const p = sorted[k];
    if (!p) continue;
    while (upper.length >= 2 && cross(upper.at(-2) ?? p, upper.at(-1) ?? p, p) <= 0) upper.pop();
    upper.push(p);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

/** `hull` grown outwards by `margin` (the hull of a circle round each corner), closed. */
function widen(hull: [number, number][], margin: number): Outline {
  const swept: [number, number][] = [];
  for (const [x, y] of hull)
    for (let k = 0; k < MARGIN_DIRECTIONS; k++) {
      const a = (2 * Math.PI * k) / MARGIN_DIRECTIONS;
      // Out to the circle's tangent: the polygon round a circle, not inside it.
      const r = margin / Math.cos(Math.PI / MARGIN_DIRECTIONS);
      swept.push([x + r * Math.cos(a), y + r * Math.sin(a)]);
    }
  const ring = convexHull(swept);
  const first = ring[0];
  return first ? [...ring, first] : [];
}

/**
 * The scan's ground from its own splats, when nothing else says where it is: the low
 * percentile of its opaque splats' heights (`GROUND_PERCENTILE`).
 */
export function groundOf(points: ScanPoints): number | null {
  const { positions, opacity } = points;
  const heights: number[] = [];
  const count = Math.min(Math.floor(positions.length / 3), opacity.length);
  for (let i = 0; i < count; i++)
    if ((opacity[i] ?? 0) >= MIN_OPACITY) heights.push(positions[i * 3 + 2] ?? 0);
  if (heights.length === 0) return null;
  heights.sort((a, b) => a - b);
  return heights[Math.floor((heights.length - 1) * GROUND_PERCENTILE)] ?? null;
}

/**
 * `outlines` (the scan's own frame) on the globe: each point at `groundZ` in that frame, through
 * `toWorld` (the tileset root's computed transform: its placement, lift and runtime scale), as
 * longitude and latitude in degrees.
 */
export function outlinesOnGlobe(
  outlines: readonly Outline[],
  toWorld: Matrix4,
  groundZ: number,
): [number, number][][] {
  const local = new Cartesian3();
  const world = new Cartesian3();
  const carto = new Cartographic();
  const rings: [number, number][][] = [];
  for (const outline of outlines) {
    const ring: [number, number][] = [];
    for (const [x, y] of outline) {
      Matrix4.multiplyByPoint(toWorld, Cartesian3.fromElements(x, y, groundZ, local), world);
      const at = Cartographic.fromCartesian(world, undefined, carto) as Cartographic | undefined;
      if (!at) continue;
      ring.push([CesiumMath.toDegrees(at.longitude), CesiumMath.toDegrees(at.latitude)]);
    }
    if (ring.length >= 4) rings.push(ring);
  }
  return rings;
}

/**
 * The clip under a scan: the catalog's footprint and each of its tall things' outlines (rings
 * of longitude and latitude, closed), as one footprint. The footprint as it is when there are
 * none -- a scan of open ground clips what it always clipped.
 */
export function withOutlines(
  footprint: Footprint | null,
  rings: readonly [number, number][][],
): Footprint | null {
  if (rings.length === 0) return footprint;
  const polygons = footprint ? polygonsOf(footprint) : [];
  return {
    type: "MultiPolygon",
    coordinates: [...polygons, ...rings.map((ring) => [ring])],
  };
}
