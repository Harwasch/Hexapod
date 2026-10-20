/**
 * Site bakes on the committed 10.4 km site (`data/tiles/synthetic-far`), whose far tiles sit
 * in frames of their own: every tile of the tileset gets one bake array from the site frame
 * (`siteBakeOf`, `canonicalSiteBake`), whatever its frame.
 *
 * The tiles are built the way `Cesium3DTile` composes them: `computedTransform` is the
 * parent's times the tile's own `transform`, in doubles from the root's ECEF placement, so
 * each tile's `B` carries the rounding the engine's does, and `B · T(-offset)` is the same bake
 * as the root's only to the last bits.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";

import { describe, expect, it } from "vitest";

import { invertAffine, transformPoint, type Mat4 } from "@/cesium/splatFrames";
import type { SplatTile, SplatTileContent } from "@/cesium/splatInternals";
import { siteBakeOf } from "@/cesium/splatTiles";
import { spzPickDataOf } from "@/lib/spzPositions";
import { spzFromGlb } from "@/view/glb";

import { bakeFixture, eastNorthUpToFixedFrame, multiplyMat4 } from "./splatFixture";

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

describe("site bakes of a scan in tile frames of its own", () => {
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
