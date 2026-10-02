import { Cartesian3, Event, Matrix4 } from "cesium";
import type { Cesium3DTileset, Scene } from "cesium";
import { beforeEach, describe, expect, it } from "vitest";

import { attachInstances, instanceSphere, setInstanceOffset } from "@/cesium/splatInstances";
import { attachSplitObjects, objectModelMatrix } from "@/cesium/splitObjects";
import {
  applyLocal,
  isRest,
  poseLocalMatrix,
  poseOf,
  REST_POSE,
  splitObjectsOf,
  yawQuat,
  type Vec3,
} from "@/lib/sceneObjects";
import { useSceneObjects } from "@/state/sceneObjects";
import { parseInstances } from "@/lib/instances";

const ENTRY = {
  uri: "objects/3/tileset.json",
  instance: 3,
  origin: [2.0828, 1.8201, -0.0776],
  pose: { translation: [0, 0, 0], rotation: [0, 0, 0, 1] },
  splats: 27319,
  fill: "fills/3/tileset.json",
};

function sphereCentre(assetId: string, id: number): number[] {
  const sphere = instanceSphere(assetId, id);
  if (!sphere) throw new Error(`no sphere for ${String(id)}`);
  return [sphere.center.x, sphere.center.y, sphere.center.z];
}

function close(a: readonly number[], b: readonly number[], eps = 1e-9): void {
  expect(a.length).toBe(b.length);
  a.forEach((v, i) => expect(Math.abs(v - (b[i] ?? Number.NaN))).toBeLessThan(eps));
}

describe("split objects", () => {
  beforeEach(() => useSceneObjects.setState({ objects: {}, poses: {} }));

  it("reads what the scan's root declares and skips malformed entries", () => {
    const refs = splitObjectsOf({
      objects: [
        ENTRY,
        { ...ENTRY, instance: 3 }, // listed twice
        { ...ENTRY, instance: 0 },
        { ...ENTRY, instance: 9, uri: "" },
        { ...ENTRY, instance: 8, origin: [1, 2] },
        { ...ENTRY, instance: 7, pose: { translation: [1, 2, 3], rotation: [0, 0, 0, 2] } },
        "nonsense",
      ],
    });
    expect(refs.map((r) => r.instance)).toEqual([3, 7]);
    expect(refs[0]).toMatchObject({ uri: ENTRY.uri, fill: ENTRY.fill, splats: 27319 });
    expect(refs[1]?.pose).toEqual({ translation: [1, 2, 3], rotation: [0, 0, 0, 1] });
    expect(splitObjectsOf({})).toEqual([]);
    expect(splitObjectsOf(undefined)).toEqual([]);
    expect(poseOf(null)).toEqual(REST_POSE);
    expect(isRest(REST_POSE)).toBe(true);
    expect(isRest({ translation: [0, 0, 0], rotation: yawQuat(0.1) })).toBe(false);
  });

  it("moves a point about the object's origin, in the scan's frame", () => {
    const origin: Vec3 = [2, 1, 0];
    const pose = { translation: [1, 0, 0] as Vec3, rotation: yawQuat(Math.PI / 2) };
    const m = poseLocalMatrix(origin, pose);
    // The origin goes by the translation; a point a metre east of it turns to the north.
    close(applyLocal(m, origin), [3, 1, 0]);
    close(applyLocal(m, [3, 1, 0.5]), [3, 2, 0.5]);
    close(poseLocalMatrix(origin, REST_POSE), Matrix4.toArray(Matrix4.IDENTITY));
  });

  it("draws an object where its pose puts it under the scan's root and model matrix", () => {
    const sceneRoot = Matrix4.fromTranslation(new Cartesian3(1000, 2000, 3000));
    const parent = Matrix4.fromTranslation(new Cartesian3(0, 0, 5));
    const ref = { origin: [2, 1, 0] as Vec3 };
    const objectRoot = Matrix4.multiply(
      sceneRoot,
      Matrix4.fromTranslation(new Cartesian3(2, 1, 0)),
      new Matrix4(),
    );
    const pose = { translation: [1, 0, 0] as Vec3, rotation: yawQuat(Math.PI / 2) };
    const model = objectModelMatrix(parent, sceneRoot, ref, pose);
    // A point a metre east of the object's origin, in the object's own frame.
    const world = Matrix4.multiplyByPoint(
      Matrix4.multiply(model, objectRoot, new Matrix4()),
      new Cartesian3(1, 0, 0.5),
      new Cartesian3(),
    );
    // ...lands at origin + t + R(1, 0, 0.5) = (3, 2, 0.5) in the scan, under its root and +5 up.
    close([world.x, world.y, world.z], [1003, 2002, 3005.5], 1e-6);
    expect(Matrix4.equals(objectModelMatrix(parent, sceneRoot, ref, REST_POSE), parent)).toBe(true);
  });

  it("loads the objects beside the scan, follows its show and the store's poses", async () => {
    const sceneRoot = Matrix4.fromTranslation(new Cartesian3(10, 0, 0));
    const parent = {
      root: { extras: { objects: [ENTRY] }, transform: sceneRoot },
      resource: { url: "https://tiles.example/site/split/tileset.json?sig=1" },
      show: true,
      modelMatrix: Matrix4.clone(Matrix4.IDENTITY),
      maximumScreenSpaceError: 16,
    } as unknown as Cesium3DTileset;
    const child = {
      root: { extras: { object: { instance: 3, ids: [3, 14, 16] } } },
      show: false,
      modelMatrix: Matrix4.clone(Matrix4.IDENTITY),
      destroy: () => undefined,
    } as unknown as Cesium3DTileset;
    const added: unknown[] = [];
    const removed: unknown[] = [];
    const preUpdate = new Event();
    let renders = 0;
    const scene = {
      primitives: { add: (p: unknown) => added.push(p), remove: (p: unknown) => removed.push(p) },
      preUpdate,
      requestRender: () => {
        renders += 1;
      },
    } as unknown as Scene;
    const urls: string[] = [];
    const followed: string[] = [];
    const dispose = attachSplitObjects(
      parent,
      scene,
      "pumpkin",
      (url) => {
        urls.push(url);
        return Promise.resolve(child);
      },
      (_tileset, assetId) => {
        followed.push(assetId);
        return () => followed.push(`${assetId} off`);
      },
    );
    await new Promise((r) => setTimeout(r, 0));
    expect(urls).toEqual(["https://tiles.example/site/split/objects/3/tileset.json?sig=1"]);
    expect(added).toEqual([child]);
    expect(followed).toEqual(["pumpkin"]);
    expect(useSceneObjects.getState().objects.pumpkin?.map((o) => o.instance)).toEqual([3]);
    preUpdate.raiseEvent();
    expect(child.show).toBe(true);
    expect(Matrix4.equals(child.modelMatrix, parent.modelMatrix)).toBe(true);

    const pose = { translation: [1, 0, 0] as Vec3, rotation: yawQuat(Math.PI / 2) };
    const before = renders;
    useSceneObjects.getState().setPose("pumpkin", 3, pose);
    expect(renders).toBe(before + 1);
    preUpdate.raiseEvent();
    close(
      Matrix4.toArray(child.modelMatrix),
      Matrix4.toArray(objectModelMatrix(parent.modelMatrix, sceneRoot, ENTRY as never, pose)),
    );
    (parent as { show: boolean }).show = false;
    preUpdate.raiseEvent();
    expect(child.show).toBe(false);
    useSceneObjects.getState().setPose("pumpkin", 3, null);
    preUpdate.raiseEvent();
    expect(Matrix4.equals(child.modelMatrix, parent.modelMatrix)).toBe(true);
    dispose();
    expect(removed).toEqual([child]);
    expect(followed).toEqual(["pumpkin", "pumpkin off"]);
    expect(useSceneObjects.getState().objects.pumpkin).toBeUndefined();
    // Without a declared table nothing is flown to; the offsets were cleared either way.
    expect(instanceSphere("pumpkin", 3)).toBeUndefined();
  });

  it("flies to a moved object where it is drawn", async () => {
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
    const tileset = {
      root: {
        extras: { instances: { uri: "instances.json", count: 1 } },
        computedTransform: Matrix4.clone(Matrix4.IDENTITY),
      },
      resource: { url: "https://x.test/scan/tileset.json" },
    } as unknown as Cesium3DTileset;
    const scene = {
      preUpdate: { addEventListener: () => () => undefined },
      requestRender: () => undefined,
    } as unknown as Scene;
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
    const parsed = parseInstances(raw);
    if (!parsed) throw new Error("fixture did not parse");
    const dispose = attachInstances(tileset, scene, "moved", gpu, () => Promise.resolve(parsed));
    await new Promise((r) => setTimeout(r, 0));
    close(sphereCentre("moved", 3), [2, 1, 0.5]);
    const pose = { translation: [5, 0, 0] as Vec3, rotation: yawQuat(0) };
    setInstanceOffset("moved", [3], Matrix4.fromArray(poseLocalMatrix([2, 1, 0], pose)));
    close(sphereCentre("moved", 3), [7, 1, 0.5]);
    setInstanceOffset("moved", [3], undefined);
    close(sphereCentre("moved", 3), [2, 1, 0.5]);
    dispose();
  });

  it("costs nothing without a declaration", () => {
    const parent = { root: { extras: {} }, resource: { url: "x" } } as unknown as Cesium3DTileset;
    expect(attachSplitObjects(parent, {} as Scene, "x")()).toBeUndefined();
  });
});
