/**
 * The clip under a splat scan covers its tall things (scanFootprint.ts): the world's own copy
 * of a scanned tree stood beside it, on production, wherever the tree's crown reached past the
 * catalog's footprint (a 2.7 m circle round the trunk). A scan of open ground keeps the clip it
 * had, so no larger hole is cut in the world for it.
 */
import { Cartesian3, Math as CesiumMath, Matrix4, Transforms } from "cesium";
import { describe, expect, it } from "vitest";

import type { Footprint } from "@twin/contracts";

import {
  canopyOutlines,
  convexHull,
  MIN_MARGIN_M,
  outlinesOnGlobe,
  withOutlines,
  type Outline,
  type ScanPoints,
} from "@/cesium/scanFootprint";

/** A seeded random, so the clouds below are the same every run. */
function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

interface Splat {
  x: number;
  y: number;
  z: number;
  opacity?: number;
  radius?: number;
}

function points(splats: Splat[]): ScanPoints {
  return {
    positions: Float32Array.from(splats.flatMap((s) => [s.x, s.y, s.z])),
    opacity: Float32Array.from(splats.map((s) => s.opacity ?? 0.9)),
    radii: Float32Array.from(splats.map((s) => s.radius ?? 0.06)),
  };
}

/** A disc of ground splats `radius` m across, at height 0, around (cx, cy). */
function ground(rand: () => number, radius: number, n: number, cx = 0, cy = 0): Splat[] {
  return Array.from({ length: n }, () => {
    const r = radius * Math.sqrt(rand());
    const a = 2 * Math.PI * rand();
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a), z: 0.02 * rand() };
  });
}

/** A tree: a trunk, and a crown `crown` m in radius whose middle is 4 m up, at (cx, cy). */
function tree(rand: () => number, crown: number, cx = 0, cy = 0): Splat[] {
  const trunk = Array.from({ length: 200 }, () => ({
    x: cx + 0.15 * (rand() - 0.5),
    y: cy + 0.15 * (rand() - 0.5),
    z: 3 * rand(),
  }));
  const leaves = Array.from({ length: 3000 }, () => {
    const r = crown * Math.cbrt(rand());
    const a = 2 * Math.PI * rand();
    const b = Math.acos(2 * rand() - 1);
    return {
      x: cx + r * Math.sin(b) * Math.cos(a),
      y: cy + r * Math.sin(b) * Math.sin(a),
      z: 4 + r * Math.cos(b) * 0.8,
    };
  });
  return [...trunk, ...leaves];
}

/** Whether (x, y) is inside a closed counter-clockwise convex ring. */
function inside(ring: Outline, x: number, y: number): boolean {
  for (let i = 0; i + 1 < ring.length; i++) {
    const [ax, ay] = ring[i] ?? [0, 0];
    const [bx, by] = ring[i + 1] ?? [0, 0];
    if ((bx - ax) * (y - ay) - (by - ay) * (x - ax) < 0) return false;
  }
  return true;
}

describe("canopyOutlines", () => {
  it("outlines a tree's whole crown, with a margin, where the catalog had only its trunk", () => {
    const rand = random(1);
    const scan = points([...ground(rand, 2, 400), ...tree(rand, 2.6)]);
    const outlines = canopyOutlines(scan, 0);
    expect(outlines).toHaveLength(1);
    const [outline] = outlines;
    if (!outline) throw new Error("no outline");
    // Every way round, past the crown's 2.6 m by at least the smallest margin; the world's tree
    // stood 3 to 3.5 m out from the trunk on production.
    for (let k = 0; k < 16; k++) {
      const a = (2 * Math.PI * k) / 16;
      const reach = 2.6 + MIN_MARGIN_M;
      expect(inside(outline, reach * Math.cos(a), reach * Math.sin(a))).toBe(true);
      expect(inside(outline, 3.5 * Math.cos(a), 3.5 * Math.sin(a))).toBe(true);
    }
    // But not the whole neighbourhood: a 10 m tree widens by a couple of metres, no more.
    expect(inside(outline, 7, 0)).toBe(false);
    expect(inside(outline, 0, -7)).toBe(false);
  });

  it("gives nothing for a scan of open ground: its clip stays the catalog's footprint", () => {
    const rand = random(2);
    // The Pumpkin: a patch of ground and pumpkins half a metre tall.
    const pumpkins = Array.from({ length: 2000 }, () => {
      const r = 0.4 * Math.sqrt(rand());
      const a = 2 * Math.PI * rand();
      return { x: r * Math.cos(a), y: r * Math.sin(a), z: 0.5 * rand() };
    });
    const outlines = canopyOutlines(points([...ground(rand, 4, 3000), ...pumpkins]), 0);
    expect(outlines).toHaveLength(0);
    const authored: Footprint = {
      type: "Polygon",
      coordinates: [
        [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 1],
          [0, 0],
        ],
      ],
    };
    expect(withOutlines(authored, [])).toBe(authored);
  });

  it("outlines two trees apart as two things, not the ground between them", () => {
    const rand = random(3);
    const scan = points([
      ...ground(rand, 20, 3000),
      ...tree(rand, 2, -12, 0),
      ...tree(rand, 2, 12, 0),
    ]);
    const outlines = canopyOutlines(scan, 0);
    expect(outlines).toHaveLength(2);
    // Each tree is in one outline, and the open ground between them in neither.
    expect(outlines.some((o) => inside(o, -12, 0))).toBe(true);
    expect(outlines.some((o) => inside(o, 12, 0))).toBe(true);
    expect(outlines.some((o) => inside(o, 0, 0))).toBe(false);
  });

  it("leaves out floaters: faint splats, and a cloud in the sky with nothing under it", () => {
    const rand = random(4);
    const faint = Array.from({ length: 50 }, () => ({
      x: 30 + rand(),
      y: rand(),
      z: 3 + rand(),
      opacity: 0.1,
    }));
    const sky = Array.from({ length: 40 }, () => ({
      x: -25 + rand(),
      y: 25 + rand(),
      z: 18 + rand(),
    }));
    const scan = points([...ground(rand, 3, 400), ...tree(rand, 2.6), ...faint, ...sky]);
    const outlines = canopyOutlines(scan, 0);
    expect(outlines).toHaveLength(1);
    const [outline] = outlines;
    if (!outline) throw new Error("no outline");
    expect(inside(outline, 30, 0)).toBe(false);
    expect(inside(outline, -25, 25)).toBe(false);
  });

  it("finds the tree's ground from the placement, when the root tile holds only its crown", () => {
    const rand = random(5);
    // The Minnetonka tree's root tile: no ground and no trunk below 1.8 m, only its crown.
    const crown = tree(rand, 2.6).filter((s) => s.z > 1.8);
    // Taken from its own lowest splats, the ground would be the crown's underside, and only
    // the crown's top would count as tall.
    const unplaced = canopyOutlines(points(crown));
    const placed = canopyOutlines(points(crown), 0);
    expect(placed).toHaveLength(1);
    const [outline] = placed;
    if (!outline) throw new Error("no outline");
    for (let k = 0; k < 8; k++) {
      const a = (2 * Math.PI * k) / 8;
      expect(inside(outline, 2.9 * Math.cos(a), 2.9 * Math.sin(a))).toBe(true);
    }
    const area = (ring: Outline) => {
      let s = 0;
      for (let i = 0; i + 1 < ring.length; i++)
        s +=
          (ring[i]?.[0] ?? 0) * (ring[i + 1]?.[1] ?? 0) -
          (ring[i + 1]?.[0] ?? 0) * (ring[i]?.[1] ?? 0);
      return Math.abs(s) / 2;
    };
    expect(area(outline)).toBeGreaterThan(area(unplaced[0] ?? []));
  });
});

describe("the clip on the globe", () => {
  it("puts the outlines where the scan is placed, and joins them to the catalog's footprint", () => {
    // The scan's frame: east/north/up at the Minnetonka tree, lifted as its clamp lifts it.
    const origin = Cartesian3.fromDegrees(-93.425903, 44.944565, 254.6);
    const toWorld = Transforms.eastNorthUpToFixedFrame(origin);
    const outline: Outline = [
      [-4, -4],
      [4, -4],
      [4, 4],
      [-4, 4],
      [-4, -4],
    ];
    const [ring] = outlinesOnGlobe([outline], toWorld, 0);
    if (!ring) throw new Error("no ring");
    const metresPerDegree = 111_320 * Math.cos(CesiumMath.toRadians(44.944565));
    const [west, south] = ring[0] ?? [0, 0];
    expect((west + 93.425903) * metresPerDegree).toBeCloseTo(-4, 1);
    expect((south - 44.944565) * 111_132).toBeCloseTo(-4, 1);
    // A runtime scale draws the scan, and its outline, twice the size.
    const scaled = Matrix4.multiplyByUniformScale(toWorld, 2, new Matrix4());
    const [big] = outlinesOnGlobe([outline], scaled, 0);
    expect(((big?.[0]?.[0] ?? 0) + 93.425903) * metresPerDegree).toBeCloseTo(-8, 1);

    const circle: Footprint = {
      type: "MultiPolygon",
      coordinates: [
        [
          [
            [-93.4259, 44.9446],
            [-93.4258, 44.9446],
            [-93.4258, 44.9445],
            [-93.4259, 44.9446],
          ],
        ],
      ],
    };
    const joined = withOutlines(circle, [ring]);
    expect(joined?.type).toBe("MultiPolygon");
    expect(joined?.coordinates).toHaveLength(2);
    expect(withOutlines(null, [ring])?.coordinates).toHaveLength(1);
  });

  it("hulls points counter-clockwise without the ones inside", () => {
    const hull = convexHull([
      [0, 0],
      [2, 0],
      [1, 1],
      [2, 2],
      [0, 2],
    ]);
    expect(hull).toEqual([
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2],
    ]);
  });
});
