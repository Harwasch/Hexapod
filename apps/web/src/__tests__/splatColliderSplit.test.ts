/**
 * Two dates of a site compared either side of the divider (SiteManager.setCompareDates): a pick
 * at a point on screen answers from the date drawn there only (SplatCollider). Without that a
 * click left of the divider measured, selected or pivoted on the right date's invisible solids.
 * Queries with no point on screen (the camera's collision) still count both dates.
 */
import { BoundingSphere, Cartesian3, Event, Matrix4, SplitDirection, Transforms } from "cesium";
import { describe, expect, it, vi } from "vitest";

import { parseCollision, type CollisionMeta } from "@/lib/collision";
import type * as Collision from "@/lib/collision";
import { SplatCollider } from "@/cesium/SplatCollider";

const ORIGIN = Cartesian3.fromDegrees(-82.6966, 28.0389, 12);
const ROOT = Transforms.eastNorthUpToFixedFrame(ORIGIN);

/** A packaged collision payload: one brick record per 8³ block, as the packager writes it. */
function payload(cells: [number, number, number][]): { raw: Uint8Array; bricks: number } {
  const bricks = new Map<string, { b: [number, number, number]; mask: Uint8Array }>();
  for (const [ix, iy, iz] of cells) {
    const b: [number, number, number] = [ix >> 3, iy >> 3, iz >> 3];
    const key = b.join(",");
    const entry = bricks.get(key) ?? { b, mask: new Uint8Array(64) };
    const n = (ix & 7) + 8 * (iy & 7) + 64 * (iz & 7);
    entry.mask[n >> 3] = (entry.mask[n >> 3] ?? 0) | (1 << (n & 7));
    bricks.set(key, entry);
  }
  const raw = new Uint8Array(bricks.size * 76);
  const view = new DataView(raw.buffer);
  [...bricks.values()].forEach(({ b, mask }, r) => {
    view.setInt32(r * 76, b[0], true);
    view.setInt32(r * 76 + 4, b[1], true);
    view.setInt32(r * 76 + 8, b[2], true);
    raw.set(mask, r * 76 + 12);
  });
  return { raw, bricks: bricks.size };
}

/** A wall 5 m east of the origin in the scan's frame, 4 m wide and high, 10 cm cells. */
const wall = vi.hoisted((): { meta: CollisionMeta | undefined; raw: Uint8Array } => ({
  meta: undefined,
  raw: new Uint8Array(),
}));
vi.mock("@/lib/collision", async (importOriginal) => {
  const actual = await importOriginal<typeof Collision>();
  return {
    ...actual,
    loadCollision: vi.fn(() =>
      wall.meta
        ? Promise.resolve(actual.parseCollision(wall.raw, wall.meta))
        : Promise.reject(new Error("no wall made")),
    ),
  };
});

function makeWall(): CollisionMeta {
  const cells: [number, number, number][] = [];
  for (let y = 0; y < 40; y++) for (let z = 0; z < 40; z++) cells.push([50, y, z]);
  const { raw, bricks } = payload(cells);
  const meta = {
    format: "hexapod.collision",
    version: 1,
    uri: "collision.bin",
    cell: 0.1,
    origin: [0, 0, 0],
    brick: 8,
    bricks,
  } as CollisionMeta;
  wall.raw = raw;
  wall.meta = meta;
  // Parsed once here too, so a bad payload fails at the source rather than in the collider.
  expect(parseCollision(raw, meta).solidCells).toBe(1600);
  return meta;
}

/** A scan whose wall stands `eastM` east of the origin (its frame moved west by 5 − eastM). */
function scan(eastM: number, splitDirection: SplitDirection, meta: CollisionMeta) {
  const toWorld = Matrix4.multiply(
    ROOT,
    Matrix4.fromTranslation(new Cartesian3(eastM - 5, 0, 0)),
    new Matrix4(),
  );
  return {
    gaussianSplatPrimitive: {},
    show: true,
    splitDirection,
    modelMatrix: Matrix4.clone(Matrix4.IDENTITY),
    boundingSphere: BoundingSphere.transform(
      new BoundingSphere(new Cartesian3(5, 2, 2), 6),
      toWorld,
    ),
    root: { extras: { collision: meta }, computedTransform: toWorld },
    resource: { url: `https://example.invalid/scan-${String(eastM)}/tileset.json` },
  };
}

/** March (LEFT, wall 5 m east) and June (RIGHT, wall 3 m east: nearer), the divider at 40 %. */
async function compared() {
  const meta = makeWall();
  const march = scan(5, SplitDirection.LEFT, meta);
  const june = scan(3, SplitDirection.RIGHT, meta);
  const tilesets = [march, june];
  const preUpdate = new Event();
  const scene = {
    preUpdate,
    primitives: { length: 2, get: (i: number) => tilesets[i] },
    requestRender: vi.fn(),
    splitPosition: 0.4,
    canvas: { clientWidth: 1000 },
  };
  const collider = new SplatCollider(scene as never, () => false);
  preUpdate.raiseEvent();
  await vi.waitFor(() => expect(collider.describe()).toHaveLength(2));
  preUpdate.raiseEvent();
  const origin = Matrix4.multiplyByPoint(ROOT, new Cartesian3(0, 2, 2), new Cartesian3());
  const direction = Cartesian3.normalize(
    Matrix4.multiplyByPointAsVector(ROOT, Cartesian3.UNIT_X, new Cartesian3()),
    new Cartesian3(),
  );
  return { collider, march, june, scene, ray: { origin, direction } };
}

describe("picks with two dates either side of the divider", () => {
  it("a click left of the divider never answers from the right date's solids", async () => {
    const { collider, march, ray } = await compared();
    const hit = collider.raycast(ray, undefined, { x: 250 });
    expect(hit?.tileset).toBe(march);
    expect(hit?.distance).toBeCloseTo(5, 0);
  });

  it("a click right of it answers from the right date's", async () => {
    const { collider, june, ray } = await compared();
    const hit = collider.raycast(ray, undefined, { x: 750 });
    expect(hit?.tileset).toBe(june);
    expect(hit?.distance).toBeCloseTo(3, 0);
  });

  it("follows the divider as it moves", async () => {
    const { collider, march, june, scene, ray } = await compared();
    expect(collider.raycast(ray as never, undefined, { x: 500 })?.tileset).toBe(june);
    scene.splitPosition = 0.6;
    expect(collider.raycast(ray as never, undefined, { x: 500 })?.tileset).toBe(march);
  });

  it("a query with no point on screen counts both dates", async () => {
    const { collider, june, ray } = await compared();
    expect(collider.raycast(ray as never)?.tileset).toBe(june);
    const near = Cartesian3.add(
      ray.origin,
      Cartesian3.multiplyByScalar(ray.direction, 2.8, new Cartesian3()),
      new Cartesian3(),
    );
    expect(collider.distanceToSurface(near, 1)).not.toBeNull();
  });
});

describe("picks with two dates another renderer draws, both hidden from CesiumJS", () => {
  it("answer on each side from the date drawn there, and with no point from both", async () => {
    const meta = makeWall();
    // Neither has a splat primitive: CesiumJS never loaded a tile of either (PlayCanvas draws).
    const hidden = (eastM: number, side: SplitDirection) => {
      const { gaussianSplatPrimitive: _none, ...tileset } = scan(eastM, side, meta);
      return { ...tileset, show: false, isDestroyed: () => false };
    };
    const march = hidden(5, SplitDirection.LEFT);
    const june = hidden(3, SplitDirection.RIGHT);
    const preUpdate = new Event();
    const scene = {
      preUpdate,
      primitives: { length: 0, get: () => undefined },
      requestRender: vi.fn(),
      splitPosition: 0.4,
      canvas: { clientWidth: 1000 },
    };
    const collider = new SplatCollider(scene as never, () => false);
    collider.setSolidWhileHidden([march, june]);
    preUpdate.raiseEvent();
    await vi.waitFor(() => expect(collider.describe()).toHaveLength(2));
    preUpdate.raiseEvent();
    const origin = Matrix4.multiplyByPoint(ROOT, new Cartesian3(0, 2, 2), new Cartesian3());
    const direction = Cartesian3.normalize(
      Matrix4.multiplyByPointAsVector(ROOT, Cartesian3.UNIT_X, new Cartesian3()),
      new Cartesian3(),
    );
    const ray = { origin, direction };
    expect(collider.raycast(ray as never, undefined, { x: 250 })?.tileset).toBe(march);
    expect(collider.raycast(ray as never, undefined, { x: 750 })?.tileset).toBe(june);
    expect(collider.raycast(ray as never)?.tileset).toBe(june);
    // One no longer drawn elsewhere is not walked on.
    collider.setSolidWhileHidden([june]);
    expect(collider.raycast(ray as never, undefined, { x: 250 })).toBeNull();
  });
});
