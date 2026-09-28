/**
 * The committed level-of-detail fixture (`data/tiles/synthetic-tree-lod`), decoded tile by
 * tile, and a fake primitive that aggregates any selection of its tiles the way
 * `GaussianSplatPrimitive.update` does.
 *
 * The tiles are read straight out of their GLBs — the SPZ block, gunzipped, positions decoded
 * from 24-bit fixed point exactly as the engine's decoder produces them — so the rig's
 * `tileChecksums` are checked against the same bytes the browser un-bakes.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";

import { parseRig, type MotionRig } from "@twin/world";

import { invertAffine, transformPoint, type Mat4 } from "@/cesium/splatFrames";
import type { SplatTile, SplatTileContent, SplatTilesetLike } from "@/cesium/splatInternals";

import {
  bakeFixture,
  eastNorthUpToFixedFrame,
  FakeSplatTexture,
  multiplyMat4,
  packedBufferFor,
} from "./splatFixture";

/** `data/tiles/synthetic-tree-lod/<name>`. */
export function lodPath(name: string): string {
  return resolve(process.cwd(), "../../data/tiles/synthetic-tree-lod", name);
}

interface TileJson {
  content: { uri: string };
  geometricError: number;
  children?: TileJson[];
  transform?: number[];
  boundingVolume: { box: number[] };
}

const lodTileset = JSON.parse(readFileSync(lodPath("tileset.json"), "utf8")) as { root: TileJson };

export const lodRig: MotionRig = parseRig(readFileSync(lodPath("rig.json"), "utf8"));

/** The SPZ block of a GLB `splat_tiles.build_glb` wrote, decoded to float32 positions. */
export function decodeSplatGlb(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const jsonLength = view.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + jsonLength))) as {
    bufferViews: { byteOffset?: number; byteLength: number }[];
  };
  const binStart = 20 + jsonLength + 8;
  const spzView = json.bufferViews[0];
  if (spzView === undefined) throw new Error("no SPZ buffer view");
  const start = binStart + (spzView.byteOffset ?? 0);
  const raw = gunzipSync(bytes.subarray(start, start + spzView.byteLength));
  const header = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const count = header.getUint32(8, true);
  const fractionalBits = header.getUint8(13);
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count * 3; i += 1) {
    const o = 16 + i * 3;
    let fixed = (raw[o] ?? 0) | ((raw[o + 1] ?? 0) << 8) | ((raw[o + 2] ?? 0) << 16);
    if (fixed & 0x800000) fixed -= 0x1000000;
    const value = fixed / (1 << fractionalBits);
    out[i] = value === 0 ? 0 : value;
  }
  return out;
}

/** One tile of the fixture: its uri, its canonical (local) positions, and its place in the tree. */
export interface LodTile {
  readonly uri: string;
  readonly local: Float32Array;
  readonly parent: string | undefined;
  readonly leaf: boolean;
}

function readTiles(): Map<string, LodTile> {
  const tiles = new Map<string, LodTile>();
  const walk = (tile: TileJson, parent: string | undefined): void => {
    const uri = tile.content.uri;
    tiles.set(uri, {
      uri,
      local: decodeSplatGlb(Uint8Array.from(readFileSync(lodPath(uri)))),
      parent,
      leaf: (tile.children?.length ?? 0) === 0,
    });
    for (const child of tile.children ?? []) walk(child, uri);
  };
  walk(lodTileset.root, undefined);
  return tiles;
}

export const lodTiles: ReadonlyMap<string, LodTile> = readTiles();

/** The uris of `uri`'s children. */
export function childrenOf(uri: string): string[] {
  return [...lodTiles.values()].filter((tile) => tile.parent === uri).map((tile) => tile.uri);
}

/** Every leaf, depth first. */
export const lodLeaves: readonly string[] = [...lodTiles.values()]
  .filter((tile) => tile.leaf)
  .map((tile) => tile.uri);

const placement = lodTileset.root.transform ?? [];
const box = lodTileset.root.boundingVolume.box;
const centre = transformPoint(placement, box[0] ?? 0, box[1] ?? 0, box[2] ?? 0, [0, 0, 0]);

/** `primitive._rootTransform` for this tileset. */
export const lodRootTransform: readonly number[] = eastNorthUpToFixedFrame(
  centre[0],
  centre[1],
  centre[2],
);

/** Every tile's `B`: all share the root's transform and the same GLB node matrix. */
export const lodBake: readonly number[] = multiplyMat4(
  invertAffine(lodRootTransform) ?? [],
  placement,
);

/** A tile as the engine holds it: loaded content with its own baked positions. */
export class FakeTile implements SplatTile {
  readonly children: SplatTile[] = [];
  content: SplatTileContent & {
    _lastSplatTransform: Mat4;
    positions: Float32Array;
    pointsLength: number;
  };

  constructor(
    readonly uri: string,
    local: Float32Array,
    bake: Mat4 = lodBake,
  ) {
    this.content = {
      _lastSplatTransform: Array.from(bake),
      positions: bakeFixture(local, bake),
      pointsLength: local.length / 3,
    };
  }
}

/** Aggregates like the engine: `concat(content.positions)` over the selection, in order. */
export class FakeTiledPrimitive {
  _positions: Float32Array = new Float32Array(0);
  _numSplats = 0;
  _splatRowMask: number;
  _splatRowShift: number;
  _snapshot = { generation: 0 };
  _pendingSnapshot: unknown = undefined;
  _rootTransform: readonly number[] = lodRootTransform;
  _selectedTileSet = new Set<FakeTile>();
  gaussianSplatTexture: FakeSplatTexture = new FakeSplatTexture();

  constructor(rowShift = 12) {
    this._splatRowShift = rowShift;
    this._splatRowMask = (1 << rowShift) - 1;
  }

  /** Commits a snapshot of `tiles`, as `commitSnapshot` would; returns the packed buffer. */
  commit(tiles: readonly FakeTile[]): Uint32Array {
    const total = tiles.reduce((sum, tile) => sum + tile.content.pointsLength, 0);
    const positions = new Float32Array(total * 3);
    let offset = 0;
    for (const tile of tiles) {
      positions.set(tile.content.positions, offset);
      offset += tile.content.positions.length;
    }
    this._positions = positions;
    this._numSplats = total;
    this._selectedTileSet = new Set(tiles);
    this._pendingSnapshot = undefined;
    this._snapshot = { generation: this._snapshot.generation + 1 };
    // A new snapshot gets a new texture holding the measured positions.
    this.gaussianSplatTexture = new FakeSplatTexture();
    return packedBufferFor(positions);
  }
}

export class FakeTiledTileset implements SplatTilesetLike {
  readonly root: SplatTile = { children: [{}, {}] };
  constructor(readonly gaussianSplatPrimitive: FakeTiledPrimitive) {}
}
