/**
 * The CesiumJS bindings on the committed 10.4 km site (`data/tiles/synthetic-far`), whose far
 * tiles sit in frames of their own: the snapshot's tiles (aggregated and incremental), their
 * bindings and the view cones all read a tile's bake from the site frame (`siteBakeOf`), so a
 * moved tile un-bakes to the digest `expected.json` lists, and every tile of the tileset shares
 * one bake array whatever its frame.
 *
 * The tiles are built the way `Cesium3DTile` composes them: `computedTransform` is the
 * parent's times the tile's own `transform`, in doubles from the root's ECEF placement, so
 * each tile's `B` carries the rounding the engine's does, and `B · T(-offset)` is the same bake
 * as the root's only to the last bits.
 *
 * Baked positions are float32, which un-bake exactly only within 2048 m of the primitive's
 * frame (`splatFrames.test.ts`, the recorded limit). Every moved tile of this fixture is more
 * than 2 km from the tileset's centre (that is why it was moved), so the binding tests put the
 * primitive's frame `E` at a site point among moved tiles instead: the algebra under test is
 * the same whichever point `E` is at.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";

import { describe, expect, it } from "vitest";

import { checksumPositions } from "@twin/world";

import { invertAffine, transformPoint, unbakePositions, type Mat4 } from "@/cesium/splatFrames";
import type {
  SplatPrimitive,
  SplatShaderBuilder,
  SplatTile,
  SplatTileContent,
} from "@/cesium/splatInternals";
import {
  aggregateBindings,
  bindTile,
  siteBakeOf,
  snapshotTiles,
  type SnapshotTile,
  type TileBinding,
} from "@/cesium/splatTiles";
import {
  CompanionViewCones,
  SplatViewCones,
  type SplatVertexVisibility,
  type ViewConeGpu,
} from "@/cesium/splatViewCones";
import { spzPickDataOf } from "@/lib/spzPositions";
import { cellCount, gridFromModel, type ViewConesMeta } from "@/lib/viewCones";
import { spzFromGlb } from "@/view/glb";

import { bakeFixture, eastNorthUpToFixedFrame, fixtureRig, multiplyMat4 } from "./splatFixture";

const FAR = resolve(process.cwd(), "../../data/tiles/synthetic-far");
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

interface RawTile {
  content: { uri: string };
  transform?: number[];
  boundingVolume: { box: number[] };
  children?: RawTile[];
}

const tilesetJson = JSON.parse(readFileSync(resolve(FAR, "splat/tileset.json"), "utf8")) as {
  root: RawTile & { transform: number[] };
};
const expected = (
  JSON.parse(readFileSync(resolve(FAR, "source/expected.json"), "utf8")) as {
    tiles: Record<string, { offset: [number, number, number]; checksum: string }>;
  }
).tiles;

function localOf(uri: string): Float32Array {
  const glb = readFileSync(resolve(FAR, "splat", uri));
  const spz = spzFromGlb(glb.buffer.slice(glb.byteOffset, glb.byteOffset + glb.byteLength));
  const positions = spzPickDataOf(gunzipSync(spz))?.positions;
  if (!positions) throw new Error(`${uri} has no positions`);
  return positions;
}

const placement = tilesetJson.root.transform;
const box = tilesetJson.root.boundingVolume.box;
/** inverse(E) for the primitive's frame `E`: ENU at `site` (site-frame metres) on the globe. */
function fromEcefAt(site: readonly [number, number, number]): number[] {
  const at = transformPoint(placement, site[0], site[1], site[2], [0, 0, 0]);
  return invertAffine(eastNorthUpToFixedFrame(at[0], at[1], at[2])) ?? [];
}
/** The engine's `E`: at the root bounding volume's centre. */
const AT_CENTRE: [number, number, number] = [box[0] ?? 0, box[1] ?? 0, box[2] ?? 0];
/** Among moved tiles: within 2 km of splat_4-0, splat_4-0-46 and splat_4-4-0-02 (all moved)
 *  and of splat_4-0-0 (not moved). */
const AMONG_MOVED: [number, number, number] = [1240, 380, 0];

type BakedTile = SplatTile & { content: SplatTileContent & { _lastSplatTransform: Mat4 } };

interface FarTile {
  readonly uri: string;
  readonly tile: BakedTile;
  readonly baked: Float32Array;
  /** Whether the tile sits in a frame of its own (a non-zero offset). */
  readonly moved: boolean;
  /** The largest baked coordinate: within 2048 m the float32 bake un-bakes exactly. */
  readonly reach: number;
}

/** The tileset as the engine holds it, every tile baked by its own `B`, owned by `owner`. */
function buildTiles(owner: object, at: readonly [number, number, number] = AT_CENTRE): FarTile[] {
  const fromEcef = fromEcefAt(at);
  const out: FarTile[] = [];
  const walk = (raw: RawTile, parent: SplatTile | undefined, parentComputed: Mat4): void => {
    const transform = raw.transform;
    const computed =
      transform === undefined
        ? Array.from(parentComputed)
        : multiplyMat4(parentComputed, transform);
    const bake = multiplyMat4(fromEcef, computed);
    const baked = bakeFixture(localOf(raw.content.uri), bake);
    let reach = 0;
    for (const value of baked) reach = Math.max(reach, Math.abs(value));
    const tile: BakedTile = {
      ...(transform === undefined ? {} : { transform }),
      ...(parent === undefined ? {} : { parent }),
      tileset: owner,
      content: { _lastSplatTransform: bake, positions: baked, pointsLength: baked.length / 3 },
    };
    const moved = (expected[raw.content.uri]?.offset ?? [0, 0, 0]).some((value) => value !== 0);
    out.push({ uri: raw.content.uri, tile, baked, moved, reach });
    for (const child of raw.children ?? []) walk(child, tile, computed);
  };
  walk(tilesetJson.root, undefined, IDENTITY);
  return out;
}

function concat(list: readonly FarTile[]): Float32Array {
  const out = new Float32Array(list.reduce((n, t) => n + t.baked.length, 0));
  let at = 0;
  for (const t of list) {
    out.set(t.baked, at);
    at += t.baked.length;
  }
  return out;
}

/** What the snapshot says each tile is: its un-bake under the snapshot's own bake. */
function checkSnapshot(list: readonly FarTile[], snapshot: readonly SnapshotTile[]): void {
  expect(snapshot.map((t) => t.content)).toEqual(list.map((t) => t.tile.content));
  const first = snapshot[0]?.bake;
  let nearMoved = 0;
  let nearStill = 0;
  for (const [index, entry] of snapshot.entries()) {
    const far = list[index];
    if (far === undefined) throw new Error("tile out of range");
    // One array for the whole tileset, so the GPU path's exact `sharedBake` holds.
    expect(entry.bake, far.uri).toBe(first);
    if (far.reach >= 2048) continue;
    const site = unbakePositions(far.baked, invertAffine(entry.bake) ?? []);
    expect(checksumPositions(site), far.uri).toBe(expected[far.uri]?.checksum);
    if (!far.moved) nearStill += 1;
    else {
      nearMoved += 1;
      // The engine's own `B` would digest a moved tile as its local positions.
      const own = invertAffine(far.tile.content._lastSplatTransform) ?? [];
      expect(checksumPositions(unbakePositions(far.baked, own)), far.uri).not.toBe(
        expected[far.uri]?.checksum,
      );
    }
  }
  expect(nearMoved).toBeGreaterThanOrEqual(3);
  expect(nearStill).toBeGreaterThanOrEqual(1);
}

describe("CesiumJS bindings of a scan in tile frames of its own", () => {
  it("the aggregated snapshot un-bakes every near tile to its site-frame digest, under one bake", () => {
    const owner = {};
    const list = buildTiles(owner, AMONG_MOVED);
    const positions = concat(list);
    const primitive = {
      _selectedTileSet: new Set(list.map((t) => t.tile)),
      _positions: positions,
    } as unknown as SplatPrimitive;
    const listed = snapshotTiles(
      { root: list[0]?.tile },
      primitive,
      positions,
      positions.length / 3,
    );
    if (listed.kind !== "tiles") throw new Error(`snapshot waits: ${listed.detail}`);
    checkSnapshot(list, listed.tiles);

    // Bound as the deformer binds them, the near tiles keep one `commonBake`.
    const accepted = new Set(Object.values(expected).map((t) => t.checksum));
    const near = listed.tiles.filter((_, i) => (list[i]?.reach ?? Infinity) < 2048);
    let start = 0;
    const packed: SnapshotTile[] = [];
    const bindings: TileBinding[] = [];
    for (const entry of near) {
      const result = bindTile(
        positions.subarray(entry.start * 3, (entry.start + entry.count) * 3),
        entry.bake,
        fixtureRig,
        accepted,
        1e-3,
      );
      if (result.kind !== "bound") throw new Error(result.detail);
      bindings.push(result.binding);
      packed.push({ ...entry, start });
      start += entry.count;
    }
    expect(aggregateBindings(packed, bindings, start).commonBake).toBe(near[0]?.bake);
  });

  it("the incremental snapshot does the same from each tile's slots", () => {
    const owner = {};
    const list = buildTiles(owner, AMONG_MOVED);
    const positions = concat(list);
    const slots = new Map<SplatTile, { start: number; count: number }>();
    let start = 0;
    for (const t of list) {
      slots.set(t.tile, { start, count: t.baked.length / 3 });
      start += t.baked.length / 3;
    }
    const primitive = { _tileSlots: slots, _positions: positions } as unknown as SplatPrimitive;
    const listed = snapshotTiles(owner, primitive, positions, start);
    if (listed.kind !== "tiles") throw new Error(`snapshot waits: ${listed.detail}`);
    checkSnapshot(list, listed.tiles);
  });

  it("a tile's site bake is the same array whichever tile of the tileset is read first", () => {
    const list = buildTiles({});
    // The bakes are equal only in exact maths: some differ in their last bits.
    const raw = list.map((t) => {
      const offset = expected[t.uri]?.offset ?? [0, 0, 0];
      const m = t.tile.content._lastSplatTransform;
      return transformPoint(m, -offset[0], -offset[1], -offset[2], [0, 0, 0]);
    });
    expect(raw.some((p) => p.some((v, k) => v !== raw[0]?.[k]))).toBe(true);
    const bakes = [...list].reverse().map((t) => siteBakeOf(t.tile));
    for (const bake of bakes) expect(bake).toBe(bakes[0]);
    // Not the engine's matrix itself, which it re-bakes into in place.
    expect(bakes[0]).not.toBe(list[0]?.tile.content._lastSplatTransform);
    // Another tileset keeps its own.
    const other = buildTiles({})[0]?.tile;
    expect(other && siteBakeOf(other)).not.toBe(bakes[0]);
    // A real re-bake (the model moved a millimetre) is a new bake, not the old one.
    const root = list[0]?.tile;
    if (root === undefined) throw new Error("no root");
    const rebake = Array.from(root.content._lastSplatTransform);
    rebake[14] = (rebake[14] ?? 0) + 1e-3;
    const rebaked = siteBakeOf({ ...root, content: { _lastSplatTransform: rebake } });
    expect(rebaked).not.toBe(bakes[0]);
    expect(rebaked?.[14]).toBe(rebake[14]);
  });
});

describe("view cones on a scan in tile frames of its own", () => {
  const META: ViewConesMeta = {
    format: "hexapod.viewcones",
    version: 1,
    uri: "viewcones.bin",
    origin: [-4000, -100, -10],
    cell: 50,
    dims: [160, 40, 2],
    fadeDeg: 20,
  };
  let mat4Calls = 0;
  const gpu: ViewConeGpu = {
    createBytes: () => ({
      copyFrom: () => undefined,
      isDestroyed: () => false,
      destroy: () => undefined,
    }),
    mat4: (values) => {
      mat4Calls += 1;
      return { mat4: [...values] };
    },
    vec4: (x, y, z, w) => ({ vec4: [x, y, z, w] }),
    vertexDestination: 1,
  };
  const build = (hook: SplatViewCones | CompanionViewCones): Record<string, () => unknown> => {
    const uniforms: Record<string, () => unknown> = {};
    const builder: SplatShaderBuilder = {
      addUniform: () => undefined,
      addVertexLines: () => undefined,
    };
    hook.addToShader(builder, uniforms, {});
    return uniforms;
  };
  // The bake of the unmoved root, composed here rather than by `siteBakeOf`; a moved tile's own
  // `B` is kilometres from it.
  const rootBake = multiplyMat4(fromEcefAt(AT_CENTRE), placement);
  const expectGrid = (value: unknown): void => {
    const want = gridFromModel(META, invertAffine(rootBake) ?? []);
    const got = (value as { mat4: number[] }).mat4;
    got.forEach((v, i) => expect(v).toBeCloseTo(want[i] ?? Number.NaN, 6));
  };

  it("fades through the site bake when a moved tile is read first, and keeps it as the selection turns", () => {
    const owner = {};
    const list = buildTiles(owner);
    const hook = new SplatViewCones(META, new Uint8Array(cellCount(META) * 4), gpu);
    const uniforms = build(hook);
    const primitive = {
      vertexVisibility: undefined as SplatVertexVisibility | undefined,
      _tileset: owner,
      _selectedTileSet: new Set(list.filter((t) => t.moved).map((t) => t.tile)),
      isDestroyed: () => false,
    };
    expect(hook.install(primitive)).toBe(true);
    expect(uniforms.u_viewConeActive?.()).toBe(1);
    expectGrid(uniforms.u_viewConeFromModel?.());
    // The root first now: the same bake, so the grid is not rebuilt.
    const calls = mat4Calls;
    primitive._selectedTileSet = new Set(list.map((t) => t.tile));
    expectGrid(uniforms.u_viewConeFromModel?.());
    expect(mat4Calls).toBe(calls);
    hook.destroy();
  });

  it("a companion layer in tile frames of its own fades through its site bake", () => {
    const scan = {};
    const layer = {};
    const scanRoot = buildTiles(scan)[0];
    const layerTiles = buildTiles(layer).filter((t) => t.moved);
    const slots = new Map<SplatTile, { start: number; count: number }>();
    let start = 0;
    for (const t of [scanRoot, ...layerTiles]) {
      if (t === undefined) continue;
      slots.set(t.tile, { start, count: t.baked.length / 3 });
      start += t.baked.length / 3;
    }
    const hook = new CompanionViewCones(META, new Uint8Array(cellCount(META) * 4), gpu, layer);
    const uniforms = build(hook);
    const primitive = {
      vertexVisibility: undefined as SplatVertexVisibility | undefined,
      _tileSlots: slots,
      isDestroyed: () => false,
    };
    expect(hook.install(primitive)).toBe(true);
    const key = (name: string): string =>
      Object.keys(uniforms).find((k) => k.startsWith(name)) ?? name;
    expect(uniforms[key("u_coneActive_")]?.()).toBe(1);
    expectGrid(uniforms[key("u_coneFromModel_")]?.());
    hook.destroy();
  });
});
