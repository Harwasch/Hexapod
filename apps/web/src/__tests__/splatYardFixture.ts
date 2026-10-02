/**
 * The committed synthetic yard (`data/tiles/synthetic-yard/splat`): a level-of-detail tileset of
 * three leafy trees, five shrubs, two snags, a building and a lawn, with the forest rig, motion
 * sidecar and plant binding `tools/captures/scene_plants.py` wrote beside its tiles — decoded
 * tile by tile, the way `splatTilesFixture.ts` decodes the single tree's.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import {
  loadLivingMotion,
  parsePlantBinding,
  parseRig,
  type LivingMotion,
  type MotionRig,
  type PlantBinding,
} from "@twin/world";

import { invertAffine, transformPoint } from "@/cesium/splatFrames";

import { eastNorthUpToFixedFrame, multiplyMat4 } from "./splatFixture";
import { decodeSplatGlb, type LodTile } from "./splatTilesFixture";

/** `data/tiles/synthetic-yard/splat/<name>`. */
export function yardPath(name: string): string {
  return resolve(process.cwd(), "../../data/tiles/synthetic-yard/splat", name);
}

interface TileJson {
  content: { uri: string };
  children?: TileJson[];
  transform?: number[];
  boundingVolume: { box: number[] };
}

const tileset = JSON.parse(readFileSync(yardPath("tileset.json"), "utf8")) as { root: TileJson };

export const yardRig: MotionRig = parseRig(readFileSync(yardPath("rig.json"), "utf8"));
export const yardBinding: PlantBinding = parsePlantBinding(
  readFileSync(yardPath("plants.json"), "utf8"),
  yardRig,
);
export const yardMotion: LivingMotion = loadLivingMotion(
  yardRig,
  readFileSync(yardPath("motion.json"), "utf8"),
);

function readTiles(): Map<string, LodTile> {
  const tiles = new Map<string, LodTile>();
  const walk = (tile: TileJson, parent: string | undefined): void => {
    const uri = tile.content.uri;
    tiles.set(uri, {
      uri,
      local: decodeSplatGlb(Uint8Array.from(readFileSync(yardPath(uri)))),
      parent,
      leaf: (tile.children?.length ?? 0) === 0,
    });
    for (const child of tile.children ?? []) walk(child, uri);
  };
  walk(tileset.root, undefined);
  return tiles;
}

export const yardTiles: ReadonlyMap<string, LodTile> = readTiles();

export function yardChildrenOf(uri: string): string[] {
  return [...yardTiles.values()].filter((tile) => tile.parent === uri).map((tile) => tile.uri);
}

export const yardLeaves: readonly string[] = [...yardTiles.values()]
  .filter((tile) => tile.leaf)
  .map((tile) => tile.uri);

const placement = tileset.root.transform ?? [];
const box = tileset.root.boundingVolume.box;
const centre = transformPoint(placement, box[0] ?? 0, box[1] ?? 0, box[2] ?? 0, [0, 0, 0]);

export const yardRootTransform: readonly number[] = eastNorthUpToFixedFrame(
  centre[0],
  centre[1],
  centre[2],
);

export const yardBake: readonly number[] = multiplyMat4(
  invertAffine(yardRootTransform) ?? [],
  placement,
);
