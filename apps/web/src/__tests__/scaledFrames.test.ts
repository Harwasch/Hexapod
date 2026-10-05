/**
 * The code that works in a scan's own frame, under a runtime scale (`renderConfig.scale`).
 *
 * The root's computed transform carries the scale, so a unit of the scan's frame is `scale`
 * metres on the globe. Collision (`SplatCollider`), the other renderers' camera
 * (`scanView/pose.ts`) and CesiumJS's pick source (`cesiumPickSource`) each cross between that
 * frame and the globe; these pin that they cross with the true inverse and convert lengths, so
 * a scan drawn at half size is also walked into, rendered and picked at half size.
 */
import { BoundingSphere, Cartesian3, Event, Matrix4, Transforms } from "cesium";
import { afterEach, describe, expect, it, vi } from "vitest";

import { parseCollision, type CollisionMeta } from "@/lib/collision";
import type * as Collision from "@/lib/collision";
import { scaledModelMatrix, uniformScale } from "@/cesium/placement";
import { scanPose } from "@/cesium/scanView/pose";
import { largestAxes, localRadii } from "@/cesium/sceneSelect/cesiumPickSource";
import { SplatCollider } from "@/cesium/SplatCollider";
import { invertAffine } from "@/cesium/splatFrames";
import { attachInstances, instanceSphere } from "@/cesium/splatInstances";
import { objectModelMatrix } from "@/cesium/splitObjects";
import { inverseScaledTransformation } from "@/cesium/tilesetScale";
import { parseInstances } from "@/lib/instances";
import { yawQuat, type Vec3 } from "@/lib/sceneObjects";

const ORIGIN = Cartesian3.fromDegrees(-82.6966, 28.0389, 12);
const ROOT = Transforms.eastNorthUpToFixedFrame(ORIGIN);

/** The scan's frame on the globe at `scale` about its origin: `M · R`. */
function toWorldAt(scale: number): Matrix4 {
  const model = Matrix4.fromArray(scaledModelMatrix([ORIGIN.x, ORIGIN.y, ORIGIN.z], scale));
  return Matrix4.multiply(model, ROOT, new Matrix4());
}

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

/** A collider over one packaged scan drawn at `scale`, once its collision file is in. */
async function colliderAt(scale: number): Promise<{ collider: SplatCollider; toWorld: Matrix4 }> {
  const meta = makeWall();
  const toWorld = toWorldAt(scale);
  const tileset = {
    gaussianSplatPrimitive: {},
    show: true,
    modelMatrix: Matrix4.fromArray(scaledModelMatrix([ORIGIN.x, ORIGIN.y, ORIGIN.z], scale)),
    boundingSphere: BoundingSphere.transform(
      new BoundingSphere(new Cartesian3(5, 2, 2), 6),
      toWorld,
    ),
    root: { extras: { collision: meta }, computedTransform: toWorld },
    resource: { url: "https://example.invalid/scan/tileset.json" },
  };
  const preUpdate = new Event();
  const scene = {
    preUpdate,
    primitives: { length: 1, get: () => tileset },
    requestRender: vi.fn(),
  };
  const collider = new SplatCollider(scene as never, () => false);
  preUpdate.raiseEvent();
  await vi.waitFor(() => expect(collider.active).toBe(true));
  preUpdate.raiseEvent();
  return { collider, toWorld };
}

/** A world ray from the scan-frame point `from`, along the scan's east. */
function eastRay(toWorld: Matrix4, from: [number, number, number]) {
  const origin = Matrix4.multiplyByPoint(toWorld, Cartesian3.fromArray(from), new Cartesian3());
  const direction = Cartesian3.normalize(
    Matrix4.multiplyByPointAsVector(toWorld, Cartesian3.UNIT_X, new Cartesian3()),
    new Cartesian3(),
  );
  return { origin, direction };
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("collision under a runtime scale", () => {
  it("meets the wall half as far away when the scan is drawn at half size", async () => {
    const full = await colliderAt(1);
    const half = await colliderAt(0.5);
    const atFull = full.collider.raycast(eastRay(full.toWorld, [0, 2, 2]));
    const atHalf = half.collider.raycast(eastRay(half.toWorld, [0, 2, 2]));
    expect(atFull?.distance).toBeCloseTo(5, 3);
    expect(atHalf?.distance).toBeCloseTo(2.5, 3);
    // And the point is on the wall as drawn: 5 m in the scan's frame from where the ray left.
    const toLocal = inverseScaledTransformation(half.toWorld, new Matrix4());
    const local = Matrix4.multiplyByPoint(
      toLocal,
      atHalf?.point ?? Cartesian3.ZERO,
      new Cartesian3(),
    );
    expect(local.x).toBeCloseTo(5, 3);
    expect(local.y).toBeCloseTo(2, 3);
  });

  it("measures distances and clearance in metres as drawn", async () => {
    const full = await colliderAt(1);
    const half = await colliderAt(0.5);
    const near = (toWorld: Matrix4) =>
      Matrix4.multiplyByPoint(toWorld, new Cartesian3(4, 2, 2), new Cartesian3());
    const dFull = full.collider.distanceToSurface(near(full.toWorld), 3);
    const dHalf = half.collider.distanceToSurface(near(half.toWorld), 3);
    expect(dFull).not.toBeNull();
    expect(dHalf ?? 0).toBeCloseTo((dFull ?? 0) / 2, 6);
    // A search radius in metres: 0.75 m reaches the half-size wall 0.5 m away, not the full.
    expect(full.collider.distanceToSurface(near(full.toWorld), 0.75)).toBeNull();
    expect(half.collider.distanceToSurface(near(half.toWorld), 0.75)).not.toBeNull();
    expect(half.collider.clearance(near(half.toWorld))).toBeCloseTo(
      full.collider.clearance(near(full.toWorld)) / 2,
      9,
    );
  });

  it("stops a move at the wall as drawn", async () => {
    const half = await colliderAt(0.5);
    const from = Matrix4.multiplyByPoint(half.toWorld, new Cartesian3(3, 2, 2), new Cartesian3());
    const to = Matrix4.multiplyByPoint(half.toWorld, new Cartesian3(7, 2, 2), new Cartesian3());
    const moved = half.collider.resolve(from, to);
    expect(moved.blocked).toBe(true);
    const toLocal = inverseScaledTransformation(half.toWorld, new Matrix4());
    const stopped = Matrix4.multiplyByPoint(toLocal, moved.position, new Cartesian3());
    expect(stopped.x).toBeLessThan(5);
  });
});

describe("a dedicated renderer's camera under a runtime scale", () => {
  const camera = (toWorld: Matrix4) =>
    ({
      positionWC: Matrix4.multiplyByPoint(toWorld, new Cartesian3(0, -10, 2), new Cartesian3()),
      directionWC: Matrix4.multiplyByPointAsVector(ROOT, Cartesian3.UNIT_Y, new Cartesian3()),
      upWC: Matrix4.multiplyByPointAsVector(ROOT, Cartesian3.UNIT_Z, new Cartesian3()),
      frustum: { fovy: 1, near: 0.2, far: 1000 },
    }) as never;
  const size = { width: 800, height: 600, pixelRatio: 1 };

  it("stands where it is among the splats, looks along unit vectors, clips as Cesium does", () => {
    const toWorld = toWorldAt(0.5);
    const pose = scanPose(
      camera(toWorld),
      inverseScaledTransformation(toWorld, new Matrix4()),
      size,
    );
    expect(pose.eye[0]).toBeCloseTo(0, 6);
    expect(pose.eye[1]).toBeCloseTo(-10, 6);
    expect(pose.eye[2]).toBeCloseTo(2, 6);
    expect(pose.direction[1]).toBeCloseTo(1, 9);
    expect(Math.hypot(...pose.direction)).toBeCloseTo(1, 12);
    expect(Math.hypot(...pose.up)).toBeCloseTo(1, 12);
    // 0.2 m and 1 km on the globe are 0.4 and 2000 units of a half-size scan's frame.
    expect(pose.near).toBeCloseTo(0.4, 9);
    expect(pose.far).toBeCloseTo(2000, 6);
  });

  it("is what it always was for a rigid frame", () => {
    const pose = scanPose(camera(ROOT), Matrix4.inverseTransformation(ROOT, new Matrix4()), size);
    expect(pose.near).toBe(0.2);
    expect(pose.far).toBe(1000);
    expect(pose.eye[1]).toBeCloseTo(-10, 6);
  });
});

describe("CesiumJS's pick source under a runtime scale", () => {
  it("takes the engine's baked radii back to the scan's frame", () => {
    // The engine bakes a splat's scales by the bake's scale: 0.05 m drawn at half size is a
    // 0.1 m splat in the scan's frame, where its position is un-baked to.
    const baked = new Float32Array([0.05, 0.02, 0.01, 0.004, 0.006, 0.002]);
    const bake = Matrix4.toArray(toWorldAt(0.5));
    expect(uniformScale(bake)).toBeCloseTo(0.5, 12);
    const radii = localRadii(baked, 2, bake);
    expect(radii[0]).toBeCloseTo(0.1, 6);
    expect(radii[1]).toBeCloseTo(0.012, 6);
    // A rigid bake: the radii as they always were.
    expect(localRadii(baked, 2, Matrix4.toArray(ROOT))).toEqual(largestAxes(baked, 2));
  });
});

describe("what follows the root's transform under a runtime scale", () => {
  it("puts an object's sphere where the scaled scan draws it, half the size", async () => {
    const raw = {
      format: "hexapod.instances",
      version: 1,
      instances: [
        {
          id: 3,
          parent: null,
          level: 0,
          splats: 10,
          bounds: { min: [1, 0, 0], max: [3, 2, 1] },
          centroid: [2, 1, 0.5],
          tags: [],
          properties: {},
          behaviour: "movable",
          views: 1,
        },
      ],
      tiles: {},
    };
    const parsed = parseInstances(raw);
    if (!parsed) throw new Error("fixture did not parse");
    const gpu = {
      createUintQuads: () => ({
        copyFrom: () => undefined,
        isDestroyed: () => false,
        destroy: () => undefined,
      }),
      createBytes: () => ({
        copyFrom: () => undefined,
        isDestroyed: () => false,
        destroy: () => undefined,
      }),
      vec4: () => ({}),
    };
    const scene = {
      preUpdate: { addEventListener: () => () => undefined },
      requestRender: () => undefined,
    };
    const sphereAt = async (scale: number, assetId: string) => {
      const tileset = {
        root: {
          extras: { instances: { uri: "instances.json", count: 1 } },
          computedTransform: toWorldAt(scale),
        },
        resource: { url: "https://x.test/scan/tileset.json" },
      };
      const dispose = attachInstances(tileset as never, scene as never, assetId, gpu, () =>
        Promise.resolve(parsed),
      );
      await new Promise((r) => setTimeout(r, 0));
      const sphere = instanceSphere(assetId, 3);
      dispose();
      return sphere;
    };
    const full = await sphereAt(1, "scaled-full");
    const half = await sphereAt(0.5, "scaled-half");
    expect(half?.radius).toBeCloseTo((full?.radius ?? 0) / 2, 9);
    // Its centre half as far from the origin: the box's centre, (2, 1, 0.5), at half size.
    const centre = Matrix4.multiplyByPoint(
      toWorldAt(0.5),
      new Cartesian3(2, 1, 0.5),
      new Cartesian3(),
    );
    expect(Cartesian3.distance(half?.center ?? Cartesian3.ZERO, centre)).toBeLessThan(1e-6);
    expect(Cartesian3.distance(centre, ORIGIN)).toBeCloseTo(Math.hypot(2, 1, 0.5) / 2, 6);
  });

  it("moves a split object by its pose in the scan's frame, scaled with the scan", () => {
    const model = Matrix4.fromArray(scaledModelMatrix([ORIGIN.x, ORIGIN.y, ORIGIN.z], 0.5));
    const pose = { translation: [1, 0, 0] as Vec3, rotation: yawQuat(Math.PI / 2) };
    const object = objectModelMatrix(model, ROOT, { origin: [2, 1, 0] as Vec3 }, pose);
    // The object's origin, (2, 1, 0) in the scan's frame, moved a metre east by its pose: (3, 1,
    // 0), drawn at half size about the scan's origin.
    const at = Matrix4.multiplyByPoint(
      Matrix4.multiply(object, ROOT, new Matrix4()),
      new Cartesian3(2, 1, 0),
      new Cartesian3(),
    );
    const want = Matrix4.multiplyByPoint(toWorldAt(0.5), new Cartesian3(3, 1, 0), new Cartesian3());
    expect(Cartesian3.distance(at, want)).toBeLessThan(1e-6);
  });

  it("un-bakes through a scaled bake: the view cones' and the deformer's inverse is a true one", () => {
    const bake = Matrix4.toArray(toWorldAt(0.5));
    const inverse = invertAffine(bake);
    expect(inverse).toBeDefined();
    const local = new Cartesian3(3, -2, 1);
    const baked = Matrix4.multiplyByPoint(toWorldAt(0.5), local, new Cartesian3());
    const back = Matrix4.multiplyByPoint(Matrix4.fromArray(inverse ?? []), baked, new Cartesian3());
    expect(Cartesian3.distance(back, local)).toBeLessThan(1e-6);
  });
});
