/**
 * Scene objects moving under the dedicated splat renderers (cesium/scanView/scanMotion.ts,
 * scanObjects.ts): the tables a back-end draws from give, splat for splat on the committed
 * yard, the displacement the skin format defines and the rigid motion telemetry sets; the
 * covariance is redrawn through the motion's Jacobian exactly (`J·Σ·Jᵀ`), a turn by the short
 * way; a sorter's centres follow a rigid motion; the link hands a back-end a motion only when
 * a driver changed something, says which skins and instances changed, and reports a gap where
 * nothing can be bound; a split object's tile is placed where its pose puts it.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { checksumPositions, type RigidMotion } from "@twin/world";
import { beforeEach, describe, expect, it } from "vitest";

import {
  buildScanMotion,
  covarianceThrough,
  evaluateScanMotion,
  matrixOfQuat,
  motionGap,
  movedCenters,
  packRigid,
  SCAN_MOTION_GLSL,
  ScanMotionLink,
  TEXELS_PER_SLOT,
  type MotionSources,
  type ScanMotion,
} from "@/cesium/scanView/scanMotion";
import { multiply4, objectPlacement, ScanObjects } from "@/cesium/scanView/scanObjects";
import { playcanvasModifierGlsl } from "@/cesium/scanView/playcanvasBackend";
import { parseInstances, type InstancesDoc } from "@/lib/instances";
import { poseLocalMatrix, REST_POSE } from "@/lib/sceneObjects";
import {
  HANDLE_FLOATS,
  parseSkin,
  rigidHandle,
  rowWeight,
  skinDisplacement,
  tileSkin,
  type SkinDoc,
} from "@/lib/skin";
import { useInstances } from "@/state/instances";
import { useSceneObjects } from "@/state/sceneObjects";
import type { TileNode } from "@/view/tiles";

import { yardTiles } from "./splatYardFixture";

const YARD = resolve(process.cwd(), "../../data/tiles/synthetic-yard");

function yardSkin(): SkinDoc {
  const raw: unknown = JSON.parse(readFileSync(resolve(YARD, "skin/skin.json"), "utf8"));
  const doc = parseSkin(raw, Uint8Array.from(readFileSync(resolve(YARD, "skin/skin.bin"))).buffer);
  if (!doc) throw new Error("the fixture is not a skin");
  return doc;
}

function yardInstances(): InstancesDoc {
  const doc = parseInstances(
    JSON.parse(readFileSync(resolve(YARD, "instances/instances.json"), "utf8")) as unknown,
  );
  if (!doc) throw new Error("the fixture is not an instances document");
  return doc;
}

/** A deterministic pseudo-random stream. */
function random(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

function quat(axis: [number, number, number], angle: number): [number, number, number, number] {
  const n = Math.hypot(...axis);
  const s = Math.sin(angle / 2) / n;
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
}

type Mat3 = number[][];
const mul = (a: Mat3, b: Mat3): Mat3 =>
  [0, 1, 2].map((r) =>
    [0, 1, 2].map((c) => [0, 1, 2].reduce((s, k) => s + (a[r]?.[k] ?? 0) * (b[k]?.[c] ?? 0), 0)),
  );
const tr = (a: Mat3): Mat3 => [0, 1, 2].map((r) => [0, 1, 2].map((c) => a[c]?.[r] ?? 0));
const diag = (s: readonly number[]): Mat3 => [
  [s[0] ?? 0, 0, 0],
  [0, s[1] ?? 0, 0],
  [0, 0, s[2] ?? 0],
];
/** `R·S²·Rᵀ` of a rotation and scales. */
const covariance = (q: readonly number[], s: readonly number[]): Mat3 => {
  const m = mul(matrixOfQuat(q), diag(s));
  return mul(m, tr(m));
};
const maxDiff = (a: Mat3, b: Mat3): number =>
  Math.max(...a.flatMap((row, r) => row.map((v, c) => Math.abs(v - (b[r]?.[c] ?? 0)))));

describe("the skin handle table", () => {
  const doc = yardSkin();
  it("gives every skinned splat of the yard the format's displacement and linear part", () => {
    const next = random(7);
    const driven = new Map<number, Float64Array>();
    for (const skin of doc.skins) {
      const handles = new Float64Array(skin.handles * HANDLE_FLOATS);
      for (let i = 0; i < handles.length; i += 1) handles[i] = (next() - 0.5) * 0.2;
      driven.set(skin.id, handles);
    }
    const motion = buildScanMotion(
      null,
      { doc, drivenSkins: driven, covariance: true },
      null,
      new Set(),
      new Set(),
    );
    expect(motion.params[0]).toBe(1);
    expect(motion.params[1]).toBe(doc.maxId);
    let checked = 0;
    let worst = 0;
    for (const tile of yardTiles.values()) {
      const found = tileSkin(doc, checksumPositions(tile.local));
      if (!found) continue;
      for (let i = 0; i < found.skins.length; i += 1) {
        const skinId = found.skins[i] ?? 0;
        const x: [number, number, number] = [
          tile.local[i * 3] ?? 0,
          tile.local[i * 3 + 1] ?? 0,
          tile.local[i * 3 + 2] ?? 0,
        ];
        const words = found.words.subarray(i * 4, i * 4 + 4);
        const got = evaluateScanMotion(motion, { skin: skinId, words, id: 0 }, x);
        const skin = doc.byId.get(skinId);
        if (!skin) {
          expect(got.delta).toEqual([0, 0, 0]);
          continue;
        }
        const learned = Array.from({ length: skin.handles - 1 }, (_, k) =>
          rowWeight(words, 0, k, doc.scale),
        );
        const want = skinDisplacement(
          driven.get(skinId) ?? [],
          skin.handles,
          skin.origin,
          learned,
          x,
        );
        for (let r = 0; r < 3; r += 1) {
          // Float32 tables, positions tens of metres out: a few micrometres.
          worst = Math.max(worst, Math.abs((got.delta[r] ?? 0) - (want[r] ?? 0)));
        }
        // The linear part is Σ w_j A_j.
        const handles = driven.get(skinId) ?? new Float64Array();
        for (let r = 0; r < 3; r += 1)
          for (let c = 0; c < 3; c += 1) {
            let a = 0;
            for (let j = 0; j < skin.handles; j += 1)
              a += (j === 0 ? 1 : (learned[j - 1] ?? 0)) * (handles[j * 12 + r * 4 + c] ?? 0);
            expect(got.linear[r * 3 + c]).toBeCloseTo(a, 5);
          }
        checked += 1;
      }
    }
    expect(checked).toBeGreaterThan(10_000);
    expect(worst).toBeLessThan(5e-5);
  });

  it("leaves skins at rest, and everything when nothing moves", () => {
    const one = doc.skins[0];
    if (!one) throw new Error("no skin");
    const motion = buildScanMotion(
      null,
      {
        doc,
        drivenSkins: new Map([[one.id, new Float64Array(one.handles * 12).fill(0.1)]]),
        covariance: true,
      },
      null,
      new Set(),
      new Set(),
    );
    const other = doc.skins[1];
    if (!other) throw new Error("one skin only");
    const words = new Uint32Array([0x7f7f7f7f, 0x7f7f7f7f, 0x7f7f7f7f, 0x7f7f7f7f]);
    expect(evaluateScanMotion(motion, { skin: other.id, words, id: 0 }, [1, 2, 3]).delta).toEqual([
      0, 0, 0,
    ]);
    const still = buildScanMotion(
      null,
      { doc, drivenSkins: new Map(), covariance: true },
      null,
      new Set(),
      new Set(),
    );
    expect(still.params[0]).toBe(0);
    expect(evaluateScanMotion(still, { skin: one.id, words, id: 0 }, [1, 2, 3]).delta).toEqual([
      0, 0, 0,
    ]);
  });
});

describe("the rigid tables", () => {
  const box = { min: [0, 0, 0], max: [1, 1, 1] };
  const doc = parseInstances({
    format: "hexapod.instances",
    version: 1,
    instances: [
      { id: 1, bounds: box, tags: [], behaviour: "movable" },
      { id: 2, parent: 1, level: 1, bounds: box, tags: [], behaviour: "movable" },
      { id: 3, parent: 2, level: 2, bounds: box, tags: [], behaviour: "movable" },
      { id: 4, bounds: box, tags: [], behaviour: "static" },
    ],
  });
  if (!doc) throw new Error("not a doc");

  it("moves a driven instance and everything below it by R x + t, a deeper one by its own", () => {
    const turn: RigidMotion = { rotation: quat([0, 0, 1], 0.7), translation: [2, -1, 0.5] };
    const lift: RigidMotion = { rotation: quat([1, 0, 0], -0.2), translation: [0, 0, 3] };
    const packed = packRigid(
      doc,
      new Map([
        [2, lift],
        [1, turn],
      ]),
    );
    expect([1, 2, 3, 4].map((id) => packed.slots[id])).toEqual([1, 2, 2, 0]);
    const motion = buildScanMotion(
      doc,
      null,
      new Map([
        [1, turn],
        [2, lift],
      ]),
      new Set(),
      new Set(),
    );
    expect(motion.params[2]).toBe(1);
    const x: [number, number, number] = [3.5, -2.25, 1.5];
    for (const [id, m] of [
      [1, turn],
      [2, lift],
      [3, lift],
    ] as const) {
      const r = matrixOfQuat(m.rotation);
      const want = [0, 1, 2].map(
        (k) =>
          (r[k]?.[0] ?? 0) * x[0] +
          (r[k]?.[1] ?? 0) * x[1] +
          (r[k]?.[2] ?? 0) * x[2] +
          m.translation[k]! -
          x[k]!,
      );
      const got = evaluateScanMotion(motion, { skin: 0, words: [], id }, x);
      for (let k = 0; k < 3; k += 1) expect(got.delta[k]).toBeCloseTo(want[k] ?? 0, 5);
      // J = I + A is the rotation.
      for (let a = 0; a < 3; a += 1)
        for (let b = 0; b < 3; b += 1)
          expect((got.linear[a * 3 + b] ?? 0) + (a === b ? 1 : 0)).toBeCloseTo(r[a]?.[b] ?? 0, 6);
    }
    expect(evaluateScanMotion(motion, { skin: 0, words: [], id: 4 }, x).delta).toEqual([0, 0, 0]);
    // The sorter's view: every leaf's full motion [R | t].
    expect(motion.rigidByLeaf.get(3)).toBe(motion.rigidByLeaf.get(2));
    expect(motion.poses.length / 4).toBeGreaterThanOrEqual(3 * TEXELS_PER_SLOT);
  });

  it("moves a sorter's centres with the objects they belong to, and only those", () => {
    const turn: RigidMotion = { rotation: quat([0, 0, 1], Math.PI / 2), translation: [1, 0, 0] };
    const { byLeaf } = packRigid(doc, new Map([[1, turn]]));
    const origin: [number, number, number] = [10, 20, 0];
    const rest = new Float32Array([1, 0, 0, 0, 1, 0, 5, 5, 5]);
    const ids = new Uint32Array([3, 0, 4]);
    const out = new Float32Array(9);
    expect(movedCenters(rest, origin, ids, byLeaf, out)).toBe(true);
    // (11, 20, 0) turned a quarter about z is (-20, 11, 0), then +1 in x, then back off origin.
    expect(Array.from(out.subarray(0, 3)).map((v) => Math.round(v * 1e4) / 1e4)).toEqual([
      -29, -9, 0,
    ]);
    expect(Array.from(out.subarray(3))).toEqual([0, 1, 0, 5, 5, 5]);
    expect(movedCenters(rest, origin, ids, new Map(), out)).toBe(false);
    expect(Array.from(out)).toEqual(Array.from(rest));
  });
});

describe("the covariance through the motion's Jacobian", () => {
  it("redraws J·Σ·Jᵀ exactly as a rotation and three scales", () => {
    const next = random(11);
    for (let trial = 0; trial < 200; trial += 1) {
      const q = quat([next() - 0.5, next() - 0.5, next() - 0.5], next() * 6);
      const s = [0.01 + next(), 0.01 + next() * 0.3, 0.001 + next() * 0.05];
      const j: Mat3 = [0, 1, 2].map((r) =>
        [0, 1, 2].map((c) => (r === c ? 1 : 0) + (next() - 0.5) * 0.8),
      );
      const got = covarianceThrough(j, q, s);
      const want = mul(mul(j, covariance(q, s)), tr(j));
      const scale = Math.max(...want.flat().map(Math.abs));
      expect(maxDiff(covariance(got.rotation, got.scale), want) / scale).toBeLessThan(1e-9);
      expect(Math.hypot(...got.rotation)).toBeCloseTo(1, 12);
    }
  });

  it("turns a splat with a rotation and keeps its scales; leaves it alone at rest", () => {
    const q = quat([1, 2, 3], 0.4);
    const s = [0.3, 0.1, 0.02];
    const r = matrixOfQuat(quat([0, 0, 1], 1.1));
    const got = covarianceThrough(r, q, s);
    expect(got.scale).toEqual(s);
    expect(
      maxDiff(covariance(got.rotation, got.scale), mul(mul(r, covariance(q, s)), tr(r))),
    ).toBeLessThan(1e-12);
    const identity = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    expect(covarianceThrough(identity, q, s)).toEqual({ rotation: q, scale: s });
  });

  it("grows a splat scaled up with its object (the harness's 2.5× shrub)", () => {
    const q = quat([0, 1, 0], 0.3);
    const got = covarianceThrough(diag([2.5, 2.5, 2.5]), q, [0.2, 0.1, 0.05]);
    expect([...got.scale].sort((a, b) => b - a).map((v) => Math.round(v * 1e9) / 1e9)).toEqual([
      0.5, 0.25, 0.125,
    ]);
  });

  it("is the shader both back-ends share", () => {
    for (const name of [
      "hexapodSkinMotion",
      "hexapodRigidMotion",
      "hexapodCovariance",
      "hexapodJacobi",
      "hexapodQuatOfMatrix",
    ]) {
      expect(SCAN_MOTION_GLSL).toContain(name);
    }
    // PlayCanvas reads the skin streams only on tiles that carry them.
    expect(playcanvasModifierGlsl(true, false)).not.toContain("loadSplatSkin");
    expect(playcanvasModifierGlsl(true, true)).toContain("loadSplatWeights()");
    expect(playcanvasModifierGlsl(false, true)).not.toContain("loadSplatInstance");
    expect(playcanvasModifierGlsl(true, true)).toContain("modifySplatRotationScale");
  });
});

/** Shared parts a driver writes into, as the link reads them. */
function fakeSources(skin: SkinDoc | undefined, instances: InstancesDoc | undefined) {
  const state = {
    drivenSkins: new Map<number, Float64Array>(),
    covariance: true,
    skinVersion: 0,
    motions: new Map<number, RigidMotion>(),
    rigidVersion: 0,
  };
  const sources: MotionSources = {
    skin: () =>
      skin
        ? {
            doc: skin,
            drivenSkins: state.drivenSkins,
            covariance: state.covariance,
            motionVersion: state.skinVersion,
          }
        : undefined,
    rigid: () => ({ instanceMotions: state.motions, motionVersion: state.rigidVersion }),
    instances: () => instances,
  };
  return { state, sources };
}

describe("the link from the shared drivers to a back-end", () => {
  beforeEach(() => {
    useInstances.setState({ motionGaps: {} });
  });
  const skin = yardSkin();
  const instances = yardInstances();
  const extras = { skin: { uri: "skin.json", count: 4 } };

  it("hands a motion only when a driver changed something, and says what changed", () => {
    const handed: (ScanMotion | null)[] = [];
    const backend = {
      name: "playcanvas" as const,
      setMotion: (m: ScanMotion | null) => handed.push(m),
    };
    const { state, sources } = fakeSources(skin, instances);
    const link = new ScanMotionLink("yard", backend, { native: false, extras, sources });
    expect(link.update()).toBe(true);
    expect(link.update()).toBe(false);
    const tree = skin.byInstance.get(1);
    if (!tree) throw new Error("no tree skin");
    state.drivenSkins.set(
      tree.id,
      rigidHandle([0, 0, 0, 1], [1, 0, 0], new Float64Array(tree.handles * 12)),
    );
    state.skinVersion += 1;
    expect(link.update()).toBe(true);
    expect([...(handed.at(-1)?.changedSkins ?? [])]).toEqual([tree.id]);
    expect(handed.at(-1)?.params[0]).toBe(1);
    state.motions.set(8, { rotation: [0, 0, 0, 1], translation: [0, 2, 0] });
    state.rigidVersion += 1;
    expect(link.update()).toBe(true);
    expect(handed.at(-1)?.changedSkins.size).toBe(0);
    expect(handed.at(-1)?.changedIds.has(8)).toBe(true);
    // Back at rest: the skin is listed as changed so its tiles redraw.
    state.drivenSkins.clear();
    state.skinVersion += 1;
    expect(link.update()).toBe(true);
    expect([...(handed.at(-1)?.changedSkins ?? [])]).toEqual([tree.id]);
    expect(handed.at(-1)?.params[0]).toBe(0);
    // Covariances switched: a new motion with extra.y off.
    state.covariance = false;
    expect(link.update()).toBe(true);
    expect(handed.at(-1)?.extra[1]).toBe(0);
    link.dispose();
    expect(handed.at(-1)).toBeNull();
    expect(useInstances.getState().motionGaps).toEqual({});
  });

  it("reports a gap where nothing can be bound, and only for a scan that moves anything", () => {
    const backend = { name: "playcanvas" as const, setMotion: () => undefined };
    const { sources } = fakeSources(skin, instances);
    const native = new ScanMotionLink("yard", backend, { native: true, extras, sources });
    expect(native.update()).toBe(false);
    expect(useInstances.getState().motionGaps.yard?.reason).toMatch(/checksums/);
    native.dispose();
    expect(useInstances.getState().motionGaps.yard).toBeUndefined();
    const still = new ScanMotionLink("yard", backend, { native: true, extras: {}, sources });
    expect(useInstances.getState().motionGaps.yard).toBeUndefined();
    still.dispose();
    expect(motionGap({ name: "spark" }, false)).toMatch(/cannot move/);
    expect(motionGap(backend, false)).toBeNull();
  });
});

describe("split objects under a dedicated renderer", () => {
  const scanRoot = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 100, 200, 300, 1];
  const origin: [number, number, number] = [2, 3, 0];
  const objectRoot = multiply4(scanRoot, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...origin, 1]);

  it("draws an object at rest where it was measured, and at its pose otherwise", () => {
    const rest = objectPlacement(scanRoot, objectRoot, { origin }, REST_POSE);
    expect(rest.map((v) => Math.round(v * 1e9) / 1e9)).toEqual([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 3, 0, 1,
    ]);
    const pose = { translation: [1, 0, 0.5] as const, rotation: quat([0, 0, 1], 0.5) };
    const moved = objectPlacement(scanRoot, objectRoot, { origin }, pose);
    const want = multiply4(poseLocalMatrix(origin, pose), [
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      ...origin,
      1,
    ]);
    for (let i = 0; i < 16; i += 1) expect(moved[i]).toBeCloseTo(want[i] ?? 0, 9);
  });

  it("loads each declared object's tile, places it, and follows the store's pose", async () => {
    const placed: (readonly number[] | null)[] = [];
    const added: string[] = [];
    const backend = {
      load: (url: string, tile: TileNode) => Promise.resolve(`${url}#${tile.uri}`),
      add: (mesh: string) => added.push(mesh),
      remove: () => undefined,
      dispose: () => undefined,
      place: (_mesh: string, matrix: readonly number[] | null) => placed.push(matrix),
    };
    const objects = new ScanObjects<string>(backend, "yard");
    const extras = {
      objects: [{ uri: "objects/3/tileset.json", instance: 3, origin, splats: 10 }],
    };
    await objects.load("https://scan.example/tiles/tileset.json", extras, scanRoot, () =>
      Promise.resolve({
        asset: { version: "1.1" },
        geometricError: 0,
        root: {
          transform: objectRoot,
          boundingVolume: { sphere: [0, 0, 0, 1] },
          geometricError: 0,
          content: { uri: "object.glb" },
        },
      }),
    );
    expect(added).toEqual(["https://scan.example/tiles/objects/3/tileset.json#object.glb"]);
    expect(objects.count).toBe(1);
    expect(
      placed
        .at(-1)
        ?.slice(12, 15)
        .map((v) => Math.round(v * 1e9) / 1e9),
    ).toEqual([2, 3, 0]);
    expect(objects.tick()).toBe(false);
    useSceneObjects
      .getState()
      .setPose("yard", 3, { translation: [0, 0, 4], rotation: [0, 0, 0, 1] });
    expect(objects.tick()).toBe(true);
    expect(
      placed
        .at(-1)
        ?.slice(12, 15)
        .map((v) => Math.round(v * 1e9) / 1e9),
    ).toEqual([2, 3, 4]);
    useSceneObjects.getState().setPose("yard", 3, null);
    expect(objects.tick()).toBe(true);
    objects.stop();
    expect(objects.count).toBe(0);
  });
});
