/**
 * The committed 10.4 km site (`data/tiles/synthetic-far`, `tools/captures/synthetic_far.py`)
 * read the way the browser reads it: a scan too wide for SPZ's +/-2048 m, whose far tiles
 * are written in frames of their own (translation-only `transform`s, nested) and whose widest
 * tiles keep fewer fractional bits. `source/expected.json` is what the generator says each
 * tile is; the web readers must agree with it.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";

import { describe, expect, it } from "vitest";

import { spzFromGlb } from "@/view/glb";
import { parseTileset, type TileNode } from "@/view/tiles";

const FAR = resolve(process.cwd(), "../../data/tiles/synthetic-far");

interface Expected {
  tiles: Record<
    string,
    {
      offset: [number, number, number];
      fractionalBits: number;
      gaussians: number;
      checksum: string;
    }
  >;
}

const expected = JSON.parse(readFileSync(resolve(FAR, "source/expected.json"), "utf8")) as Expected;
const tree = parseTileset(JSON.parse(readFileSync(resolve(FAR, "splat/tileset.json"), "utf8")));

function everyTile(root: TileNode): TileNode[] {
  return [root, ...root.children.flatMap(everyTile)];
}

/** The tile's SPZ, unzipped, as Spark and spz-loader are handed it. */
function rawSpz(uri: string): Uint8Array {
  const glb = readFileSync(resolve(FAR, "splat", uri));
  const bytes = glb.buffer.slice(glb.byteOffset, glb.byteOffset + glb.byteLength);
  return gunzipSync(spzFromGlb(bytes));
}

describe("the far fixture's tile frames", () => {
  const tiles = everyTile(tree.root);

  it("names every tile the generator wrote", () => {
    expect(tiles.map((tile) => tile.uri).sort()).toEqual(Object.keys(expected.tiles).sort());
  });

  it("parses each tile's offset as expected.json composes it", () => {
    for (const tile of tiles) {
      const want = expected.tiles[tile.uri]?.offset ?? [Number.NaN, 0, 0];
      expect(tile.offset ?? [0, 0, 0], tile.uri).toEqual(want);
    }
    // Some are moved, and one is moved inside a moved parent.
    expect(tiles.filter((tile) => tile.offset).length).toBeGreaterThanOrEqual(4);
  });

  it("reads each tile's fractional bits from its SPZ header", () => {
    for (const tile of tiles) {
      const raw = rawSpz(tile.uri);
      expect(raw[13], tile.uri).toBe(expected.tiles[tile.uri]?.fractionalBits);
    }
    expect(Math.min(...tiles.map((tile) => rawSpz(tile.uri)[13] ?? 12))).toBeLessThanOrEqual(11);
  });
});
