/**
 * `SplatDeformer` on a level-of-detail tileset: the committed REPLACE fixture, tile by tile.
 *
 * The properties M5 has to hold, each against the real tiles and the real rig:
 *
 * - a gaussian's displacement depends on its canonical position and the rig, never on which
 *   tile carries it or where that tile lands in the aggregate;
 * - merged parents move too, bound by the same nearest-node rule as leaves;
 * - calm restores every tile's exact engine bytes;
 * - tiles loading, unloading and REPLACE swapping parents for children re-derive, reusing the
 *   bindings of tiles that stayed;
 * - nothing canonical is ever written, per tile;
 * - and the GPU path's texture packing and shader arithmetic agree with the CPU path.
 */

import { beforeEach, describe, expect, it } from "vitest";

import {
  assignSplatsToNodes,
  checksumPositions,
  deform,
  deformPositions,
  flutterField,
  flutterHash,
  maxDisplacement,
  maxFlutterAmplitude,
  positionKeys,
  skinSplatsToNodes,
  type FlutterField,
  type NodeTransform,
  type WindSettings,
} from "@twin/world";

import { SplatDeformer } from "@/cesium/SplatDeformer";
import {
  clearSplatCaptures,
  digestSplatPositions,
  recordSplatCapture,
} from "@/cesium/splatCaptureRegistry";
import { transformPositions } from "@/cesium/splatFrames";
import { evaluateSplatMotion } from "@/cesium/splatGpuMotion";
import { hasMotionPart } from "@/cesium/splatMotionChain";
import { bitsToFloat32, positionWordOffset, splatTextureLayout } from "@/cesium/splatTexels";

import { canonicalPositions, fixtureRig } from "./splatFixture";
import { buildDrawCommand, fakeFactory, FakeHookedPrimitive } from "./splatGpuFixture";
import {
  childrenOf,
  FakeTile,
  FakeTiledPrimitive,
  FakeTiledTileset,
  lodBake,
  lodLeaves,
  lodRig,
  lodTiles,
} from "./splatTilesFixture";

const GALE: WindSettings = { strength: 1, bearingDeg: 250 };
const STILL: WindSettings = { strength: 0, bearingDeg: 250 };

function frameAt(t: number, wind: WindSettings = GALE): [NodeTransform[], FlutterField] {
  return [deform(lodRig, t, wind), flutterField(lodRig, t, wind)];
}

/** One `FakeTile` per uri, made once, so a tile that stays selected is the same object. */
function tileSet(): Map<string, FakeTile> {
  const tiles = new Map<string, FakeTile>();
  for (const [uri, tile] of lodTiles) tiles.set(uri, new FakeTile(uri, tile.local));
  return tiles;
}

function pick(tiles: Map<string, FakeTile>, uris: readonly string[]): FakeTile[] {
  return uris.map((uri) => {
    const tile = tiles.get(uri);
    if (tile === undefined) throw new Error(`no tile ${uri}`);
    return tile;
  });
}

/** Commits a selection and files its capture, as the engine and the interception would. */
function commit(primitive: FakeTiledPrimitive, tiles: readonly FakeTile[]): void {
  const packed = primitive.commit(tiles);
  recordSplatCapture({
    count: primitive._numSplats,
    digest: digestSplatPositions(primitive._positions, primitive._numSplats),
    data: packed,
    at: 0,
  });
}

/** The baked position a frame puts splat `i` at, read back out of the last upload. */
function uploaded(primitive: FakeTiledPrimitive, i: number): [number, number, number] {
  const layout = splatTextureLayout(
    primitive._numSplats,
    primitive._splatRowMask,
    primitive._splatRowShift,
  );
  const upload = primitive.gaussianSplatTexture.uploads.at(-1);
  if (upload === undefined) throw new Error("nothing uploaded");
  // Uploads are row bands; the band's first row is `yOffset`.
  const word = positionWordOffset(i, layout) - upload.yOffset * layout.width * 4;
  return [0, 1, 2].map((k) => bitsToFloat32(upload.words[word + k] ?? 0)) as [
    number,
    number,
    number,
  ];
}

/**
 * Where one gaussian should be, from its canonical position alone: the rig's motion applied
 * to a one-gaussian array, then baked. The reference every tiled result is held to.
 */
function reference(
  local: Float32Array,
  index: number,
  transforms: NodeTransform[],
  field: FlutterField,
): [number, number, number] {
  const p = local.slice(index * 3, index * 3 + 3);
  const moved = deformPositions(
    p,
    assignSplatsToNodes(p, lodRig),
    transforms,
    undefined,
    field,
    positionKeys(p),
    skinSplatsToNodes(p, lodRig),
  );
  const baked = transformPositions(moved, lodBake, new Float32Array(3));
  return [baked[0] ?? 0, baked[1] ?? 0, baked[2] ?? 0];
}

const ROOT = "splat.glb";
const MIDDLE = childrenOf(ROOT);

beforeEach(() => {
  clearSplatCaptures();
});

describe("the committed LOD fixture and its rig", () => {
  it("lists every tile's digest, and the leaves hold the single-tile tree exactly", () => {
    expect(lodRig.tileChecksums?.length).toBe(lodTiles.size);
    for (const tile of lodTiles.values()) {
      expect(lodRig.tileChecksums).toContain(checksumPositions(tile.local));
    }
    const leafCount = lodLeaves.reduce((n, uri) => n + (lodTiles.get(uri)?.local.length ?? 0), 0);
    expect(leafCount).toBe(canonicalPositions.length);
    // Same nodes as the single-tile tree's rig: a tiling does not change the tree.
    expect(lodRig.nodes).toEqual(fixtureRig.nodes);
  });
});

describe("attaching to a multi-tile snapshot", () => {
  it("binds every selected tile and reports an exact re-bake", () => {
    const primitive = new FakeTiledPrimitive();
    const tiles = tileSet();
    commit(primitive, pick(tiles, MIDDLE));
    const deformer = new SplatDeformer({ tileset: new FakeTiledTileset(primitive), rig: lodRig });
    const status = deformer.apply(...frameAt(1));
    expect(status.phase).toBe("ready");
    expect(status.tiles).toBe(MIDDLE.length);
    expect(status.tileBindings).toBe(MIDDLE.length);
    expect(status.numSplats).toBe(primitive._numSplats);
    expect(status.bakeResidualM).toBe(0);
    expect(status.motion).toBe("cpu");
    expect(status.displaced).toBe(true);
  });

  it("moves merged parents as well as leaves", () => {
    const primitive = new FakeTiledPrimitive();
    commit(primitive, pick(tileSet(), [ROOT]));
    const deformer = new SplatDeformer({ tileset: new FakeTiledTileset(primitive), rig: lodRig });
    deformer.apply(...frameAt(4));
    const baked = primitive._positions;
    let moved = 0;
    for (let i = 0; i < primitive._numSplats; i += 1) {
      const p = uploaded(primitive, i);
      if (p.some((v, k) => v !== baked[i * 3 + k])) moved += 1;
    }
    // The 114 merged root gaussians: most sit in the crown and ride it.
    expect(moved).toBeGreaterThan(primitive._numSplats / 2);
  });
});

describe("a gaussian moves by where it stands, not by its tile", () => {
  // A generous timeout on top: thousands of one-gaussian references are slow on a loaded runner.
  it(
    "matches the single-gaussian reference in every selection, whatever its index",
    { timeout: 60_000 },
    () => {
      const tiles = tileSet();
      const [transforms, field] = frameAt(6.5);
      // Far (merged root), middle (merged octants and a packed leaf), near (every leaf), and
      // the near view in a different order: the same gaussians at different indices.
      const selections = [[ROOT], MIDDLE, [...lodLeaves], [...lodLeaves].reverse()];
      const bound = maxDisplacement(lodRig, GALE) + maxFlutterAmplitude(lodRig, GALE);
      // The reference depends on a gaussian's canonical position alone, so each is computed once
      // and held against every selection that draws it (the near views draw the same leaves).
      const references = new Map<string, [number, number, number][]>();
      const referenceOf = (
        uri: string,
        local: Float32Array,
        j: number,
      ): [number, number, number] => {
        let list = references.get(uri);
        if (!list) {
          list = [];
          references.set(uri, list);
        }
        const want = list[j] ?? reference(local, j, transforms, field);
        list[j] = want;
        return want;
      };
      // Every gaussian is checked; failures are collected and asserted once. One `expect` per
      // gaussian made this the suite's slowest test, which timed out in loaded full runs.
      const mismatches: {
        selection: number;
        uri: string;
        j: number;
        got: number[];
        want: number[];
      }[] = [];
      const tooFar: { selection: number; uri: string; j: number; moved: number }[] = [];
      let checked = 0;
      for (const [s, selection] of selections.entries()) {
        const primitive = new FakeTiledPrimitive();
        commit(primitive, pick(tiles, selection));
        const deformer = new SplatDeformer({
          tileset: new FakeTiledTileset(primitive),
          rig: lodRig,
        });
        deformer.apply(transforms, field);
        let start = 0;
        for (const uri of selection) {
          const local = lodTiles.get(uri)?.local ?? new Float32Array(0);
          for (let j = 0; j < local.length / 3; j += 1) {
            const got = uploaded(primitive, start + j);
            const want = referenceOf(uri, local, j);
            checked += 1;
            // Bit for bit (Object.is, as toEqual compares numbers): the aggregate is only
            // concatenation, so nothing may differ.
            if (!got.every((v, k) => Object.is(v, want[k]))) {
              mismatches.push({ selection: s, uri, j, got, want });
            }
            const rest = primitive._positions.subarray((start + j) * 3, (start + j) * 3 + 3);
            const moved = Math.hypot(...got.map((v, k) => v - (rest[k] ?? 0)));
            // A splat sits away from its node, so it can swing a little past the node bound.
            if (!(moved < bound * 2 + 0.1)) tooFar.push({ selection: s, uri, j, moved });
          }
          start += local.length / 3;
        }
      }
      expect(mismatches.slice(0, 5)).toEqual([]);
      expect(mismatches.length).toBe(0);
      expect(tooFar.slice(0, 5)).toEqual([]);
      // Every gaussian of every selection was held to the reference.
      expect(checked).toBe(
        selections.reduce(
          (n, selection) =>
            n + selection.reduce((m, uri) => m + (lodTiles.get(uri)?.local.length ?? 0) / 3, 0),
          0,
        ),
      );
      expect(checked).toBeGreaterThan(0);
    },
  );

  it("gives a parent gaussian and a leaf gaussian at one position the same motion", () => {
    // The merge puts parents at cell centroids, so no committed parent coincides with a leaf;
    // make one that does. A "parent" tile holding copies of a few leaf gaussians, stamped into
    // the rig like any tile: those gaussians must move exactly as the leaf's own do.
    const tiles = tileSet();
    const leaf = lodLeaves[0] ?? "";
    const leafLocal = lodTiles.get(leaf)?.local ?? new Float32Array(0);
    const copies = leafLocal.slice(0, 30);
    const parent = new FakeTile("parent-copy.glb", copies);
    const rig = {
      ...lodRig,
      tileChecksums: [...(lodRig.tileChecksums ?? []), checksumPositions(copies)],
    };
    const [transforms, field] = frameAt(3.25);

    const run = (selection: FakeTile[]): FakeTiledPrimitive => {
      const primitive = new FakeTiledPrimitive();
      commit(primitive, selection);
      new SplatDeformer({ tileset: new FakeTiledTileset(primitive), rig }).apply(transforms, field);
      return primitive;
    };
    const withLeaf = run(pick(tiles, [leaf]));
    const withParent = run([...pick(tiles, MIDDLE.slice(0, 2)), parent]);
    const offset = withParent._numSplats - 10;
    for (let j = 0; j < 10; j += 1) {
      expect(uploaded(withParent, offset + j)).toEqual(uploaded(withLeaf, j));
    }
  });
});

describe("tiles load, unload and replace", () => {
  it("re-derives on every selection change and binds each tile once", () => {
    const primitive = new FakeTiledPrimitive();
    const tiles = tileSet();
    commit(primitive, pick(tiles, [ROOT]));
    const deformer = new SplatDeformer({ tileset: new FakeTiledTileset(primitive), rig: lodRig });
    expect(deformer.apply(...frameAt(1)).tileBindings).toBe(1);

    // REPLACE: the root gives way to its children.
    commit(primitive, pick(tiles, MIDDLE));
    let status = deformer.apply(...frameAt(1.1));
    expect(status.phase).toBe("ready");
    expect(status.rederivations).toBe(1);
    expect(status.tiles).toBe(MIDDLE.length);
    expect(status.tileBindings).toBe(1 + MIDDLE.length);

    // One octant refines into its children; the rest stay selected and are not re-bound.
    const refined = MIDDLE[0] ?? "";
    const next = [...childrenOf(refined), ...MIDDLE.slice(1)];
    commit(primitive, pick(tiles, next));
    status = deformer.apply(...frameAt(1.2));
    expect(status.rederivations).toBe(2);
    expect(status.tileBindings).toBe(1 + MIDDLE.length + childrenOf(refined).length);

    // And back out to the root, which is still loaded: nothing new to bind.
    commit(primitive, pick(tiles, [ROOT]));
    status = deformer.apply(...frameAt(1.3));
    expect(status.rederivations).toBe(3);
    expect(status.tileBindings).toBe(1 + MIDDLE.length + childrenOf(refined).length);

    // A tile the engine unloaded and loaded again is a new content object: bound afresh.
    const reloaded = new FakeTile(ROOT, lodTiles.get(ROOT)?.local ?? new Float32Array(0));
    commit(primitive, [reloaded]);
    status = deformer.apply(...frameAt(1.4));
    expect(status.tileBindings).toBe(2 + MIDDLE.length + childrenOf(refined).length);
    expect(status.phase).toBe("ready");
  });

  it("waits while a rebuild is in flight, and when the tile list does not add up", () => {
    const primitive = new FakeTiledPrimitive();
    const tiles = tileSet();
    commit(primitive, pick(tiles, MIDDLE));
    // `_selectedTileSet` already names the pending rebuild's tiles, not the committed ones.
    primitive._pendingSnapshot = {};
    const deformer = new SplatDeformer({ tileset: new FakeTiledTileset(primitive), rig: lodRig });
    expect(deformer.apply(...frameAt(1)).reason).toBe("tiles");
    expect(primitive.gaussianSplatTexture.copyFrom).not.toHaveBeenCalled();

    // A list in another order than the snapshot: sampled positions disagree.
    primitive._pendingSnapshot = undefined;
    primitive._selectedTileSet = new Set([...primitive._selectedTileSet].reverse());
    expect(deformer.apply(...frameAt(1)).reason).toBe("tiles");
    expect(primitive.gaussianSplatTexture.copyFrom).not.toHaveBeenCalled();

    // The engine's own order: attaches.
    primitive._selectedTileSet = new Set(pick(tiles, MIDDLE));
    expect(deformer.apply(...frameAt(1)).phase).toBe("ready");
  });

  it("restores every tile's exact bytes at calm, across a selection change", () => {
    const primitive = new FakeTiledPrimitive();
    const tiles = tileSet();
    commit(primitive, pick(tiles, MIDDLE));
    const deformer = new SplatDeformer({ tileset: new FakeTiledTileset(primitive), rig: lodRig });
    deformer.apply(...frameAt(2));
    commit(primitive, pick(tiles, [...lodLeaves]));
    deformer.apply(...frameAt(2.5));
    expect(deformer.status.displaced).toBe(true);
    deformer.apply(...frameAt(3, STILL));
    expect(deformer.status.displaced).toBe(false);
    for (let i = 0; i < primitive._numSplats; i += 1) {
      const got = uploaded(primitive, i);
      for (let k = 0; k < 3; k += 1) {
        expect(Object.is(got[k], primitive._positions[i * 3 + k])).toBe(true);
      }
    }
    const uploads = primitive.gaussianSplatTexture.uploads.length;
    deformer.apply(...frameAt(4, STILL));
    expect(primitive.gaussianSplatTexture.uploads.length).toBe(uploads);
  });

  // Two hundred and forty CPU frames: a generous timeout, so a loaded runner cannot fail it.
  it(
    "never writes a tile's canonical positions, across hundreds of frames and selections",
    { timeout: 60_000 },
    () => {
      const primitive = new FakeTiledPrimitive();
      const tiles = tileSet();
      const snapshot = (): Map<string, Uint8Array> =>
        new Map(
          [...tiles].map(([uri, tile]) => [
            uri,
            Uint8Array.from(new Uint8Array(tile.content.positions.buffer.slice(0))),
          ]),
        );
      const before = snapshot();
      const localBefore = new Map(
        [...lodTiles].map(([uri, tile]) => [uri, Float32Array.from(tile.local)]),
      );
      const deformer = new SplatDeformer({ tileset: new FakeTiledTileset(primitive), rig: lodRig });
      const selections = [[ROOT], MIDDLE, [...lodLeaves]];
      const sameBytes = (a: Uint8Array, b: Uint8Array): boolean =>
        a.length === b.length && a.every((value, i) => value === b[i]);
      let last: readonly string[] = [];
      for (let frame = 0; frame < 240; frame += 1) {
        if (frame % 40 === 0) {
          last = selections[(frame / 40) % 3] ?? [];
          commit(primitive, pick(tiles, last));
        }
        const engine = Uint8Array.from(new Uint8Array(primitive._positions.buffer.slice(0)));
        const t = frame / 60;
        const wind = { strength: 0.3 + 0.7 * Math.abs(Math.sin(t)), bearingDeg: 30 + t * 20 };
        deformer.apply(deform(lodRig, t, wind), flutterField(lodRig, t, wind));
        expect(sameBytes(new Uint8Array(primitive._positions.buffer), engine)).toBe(true);
      }
      const after = snapshot();
      for (const [uri, bytes] of before)
        expect(sameBytes(after.get(uri) ?? new Uint8Array(), bytes)).toBe(true);
      for (const [uri, tile] of lodTiles) {
        expect(checksumPositions(tile.local)).toBe(
          checksumPositions(localBefore.get(uri) ?? tile.local),
        );
      }
      // And the deformer's own canonical copy is the un-baked tiles, not something drifting.
      const canonical = deformer.canonicalPositions ?? new Float32Array(0);
      let offset = 0;
      for (const uri of last) {
        const local = lodTiles.get(uri)?.local ?? new Float32Array(0);
        expect(checksumPositions(canonical.subarray(offset, offset + local.length))).toBe(
          checksumPositions(local),
        );
        offset += local.length;
      }
    },
  );
});

describe("refusing rather than misleading, per tile", () => {
  it("refuses a tile the rig does not list", () => {
    const primitive = new FakeTiledPrimitive();
    const tiles = tileSet();
    const stranger = Float32Array.from(lodTiles.get(ROOT)?.local ?? []);
    stranger[4] = (stranger[4] ?? 0) + 1 / 4096;
    commit(primitive, [...pick(tiles, MIDDLE.slice(0, 2)), new FakeTile("x.glb", stranger)]);
    const deformer = new SplatDeformer({ tileset: new FakeTiledTileset(primitive), rig: lodRig });
    const status = deformer.apply(...frameAt(1));
    expect(status.phase).toBe("refused");
    expect(status.reason).toBe("checksum");
    expect(status.observedChecksum).toBe(checksumPositions(stranger));
    expect(primitive.gaussianSplatTexture.copyFrom).not.toHaveBeenCalled();
  });

  it("refuses a level-of-detail tileset under a single-tile rig", () => {
    // The single-tile rig names one digest, the whole tree's; no tile of the LOD tiling is it.
    const primitive = new FakeTiledPrimitive();
    commit(primitive, pick(tileSet(), MIDDLE));
    const deformer = new SplatDeformer({
      tileset: new FakeTiledTileset(primitive),
      rig: fixtureRig,
    });
    expect(deformer.apply(...frameAt(1)).reason).toBe("checksum");
  });
});

// -----------------------------------------------------------------------------------------------

describe("the GPU path", () => {
  it("installs the hook, never writes the attribute texture, and matches the CPU path", () => {
    const factory = fakeFactory();
    const primitive = new FakeHookedPrimitive();
    const tiles = tileSet();
    primitive.commit(pick(tiles, MIDDLE)); // no capture filed: the GPU path does not need one
    const deformer = new SplatDeformer({
      tileset: new FakeTiledTileset(primitive),
      rig: lodRig,
      gpu: factory,
    });
    let status = deformer.apply(...frameAt(0, STILL));
    expect(status.phase).toBe("ready");
    expect(status.motion).toBe("gpu");
    expect(hasMotionPart(primitive, deformer.gpuMotion)).toBe(true);

    // Nothing drawn displaced until the engine has built a command with the hook in it.
    status = deformer.apply(...frameAt(5));
    expect(status.displaced).toBe(false);
    const uniforms = buildDrawCommand(primitive);
    const [transforms, field] = frameAt(5);
    status = deformer.apply(transforms, field);
    expect(status.displaced).toBe(true);
    expect(uniforms.u_splatMotionActive?.()).toBe(1);
    // One motion row for 214 nodes: 16 KB, not a splat-count's worth of texture.
    expect(status.lastUploadWords).toBeLessThanOrEqual(1024 * 4 * 2);
    expect(primitive.gaussianSplatTexture.copyFrom).not.toHaveBeenCalled();

    // What the shader computes, from the textures as packed, against the CPU reference.
    const binding = factory.made.find((t) => t.initial instanceof Uint32Array && t.width > 1);
    expect(binding).toBeDefined();
    const motion = deformer.gpuMotion?.motionData ?? new Float32Array(0);
    const skin = deformer.skin ?? { nodes: new Uint16Array(0), weights: new Uint16Array(0) };
    const keys = deformer.flutterKeys ?? new Uint32Array(0);
    let start = 0;
    let worst = 0;
    for (const uri of MIDDLE) {
      const local = lodTiles.get(uri)?.local ?? new Float32Array(0);
      for (let j = 0; j < local.length / 3; j += 1) {
        const i = start + j;
        const rest = primitive._positions.subarray(i * 3, i * 3 + 3);
        const gpu = evaluateSplatMotion(
          motion,
          skin.nodes.subarray(i * 4, i * 4 + 4),
          skin.weights.subarray(i * 4, i * 4 + 4),
          flutterHash(keys[i] ?? 0),
          [rest[0] ?? 0, rest[1] ?? 0, rest[2] ?? 0],
        );
        const cpu = reference(local, j, transforms, field);
        worst = Math.max(worst, ...gpu.map((v, k) => Math.abs(v - (cpu[k] ?? 0))));
      }
      start += local.length / 3;
    }
    // float32 textures against the CPU path's float32 output: micrometres, not millimetres.
    expect(worst).toBeLessThan(2e-5);

    // Calm: the shader is told nothing moves, and returns every fetched position untouched.
    status = deformer.apply(...frameAt(6, STILL));
    expect(status.displaced).toBe(false);
    expect(uniforms.u_splatMotionActive?.()).toBe(0);
    expect(primitive.gaussianSplatTexture.copyFrom).not.toHaveBeenCalled();
  });

  it("draws nothing displaced for a snapshot it has not bound", () => {
    const factory = fakeFactory();
    const primitive = new FakeHookedPrimitive();
    const tiles = tileSet();
    primitive.commit(pick(tiles, MIDDLE));
    const deformer = new SplatDeformer({
      tileset: new FakeTiledTileset(primitive),
      rig: lodRig,
      gpu: factory,
    });
    deformer.apply(...frameAt(0, STILL));
    const uniforms = buildDrawCommand(primitive);
    deformer.apply(...frameAt(5));
    expect(uniforms.u_splatMotionActive?.()).toBe(1);
    // A snapshot committed between our write and the draw: its bindings are not ours yet.
    primitive.commit(pick(tiles, [...lodLeaves]));
    expect(uniforms.u_splatMotionActive?.()).toBe(0);
    // The next tick binds it, and motion resumes.
    deformer.apply(...frameAt(5.1));
    expect(uniforms.u_splatMotionActive?.()).toBe(1);
    expect(deformer.status.rederivations).toBe(1);
  });

  it("uninstalls the hook and releases its textures on destroy", () => {
    const factory = fakeFactory();
    const primitive = new FakeHookedPrimitive();
    primitive.commit(pick(tileSet(), MIDDLE));
    const deformer = new SplatDeformer({
      tileset: new FakeTiledTileset(primitive),
      rig: lodRig,
      gpu: factory,
    });
    deformer.apply(...frameAt(0, STILL));
    buildDrawCommand(primitive);
    deformer.apply(...frameAt(5));
    deformer.destroy();
    expect(primitive.vertexMotion).toBeUndefined();
    expect(factory.made.every((texture) => texture.destroyed)).toBe(true);
  });

  it("stays on the CPU path for an engine without the hook", () => {
    const primitive = new FakeTiledPrimitive();
    commit(primitive, pick(tileSet(), MIDDLE));
    const deformer = new SplatDeformer({
      tileset: new FakeTiledTileset(primitive),
      rig: lodRig,
      gpu: fakeFactory(),
    });
    const status = deformer.apply(...frameAt(1));
    expect(status.motion).toBe("cpu");
    expect(status.cpuReason).toBe("no-hook");
    expect(primitive.gaussianSplatTexture.copyFrom).toHaveBeenCalled();
  });

  it("falls back to the CPU path, and says why, for tiles that do not share a bake", () => {
    const primitive = new FakeHookedPrimitive();
    const [first, second] = MIDDLE;
    // The same tree, but the second tile placed by a matrix a quarter metre east: each tile is
    // self-consistent, so it binds, but no one matrix un-bakes both, which the shader needs.
    const shifted = [...lodBake];
    shifted[12] = (shifted[12] ?? 0) + 0.25;
    commit(primitive, [
      new FakeTile(first ?? "", lodTiles.get(first ?? "")?.local ?? new Float32Array(0)),
      new FakeTile(second ?? "", lodTiles.get(second ?? "")?.local ?? new Float32Array(0), shifted),
    ]);
    const deformer = new SplatDeformer({
      tileset: new FakeTiledTileset(primitive),
      rig: lodRig,
      gpu: fakeFactory(),
    });
    const status = deformer.apply(...frameAt(1));
    expect(status.phase).toBe("ready");
    expect(status.motion).toBe("cpu");
    expect(status.cpuReason).toBe("mixed-bake");
    expect(primitive.vertexMotion).toBeUndefined();
    expect(primitive.gaussianSplatTexture.copyFrom).toHaveBeenCalled();
  });
});
