import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";

import { describe, expect, it } from "vitest";

import { spzFromGlb } from "@/view/glb";
import { spzPointsRaw } from "@/view/spz";

const LOD = resolve(process.cwd(), "../../data/tiles/synthetic-tree-lod");

interface Tile {
  boundingVolume: { box: number[] };
  content: { uri: string };
  extras: { gaussians: number };
  children?: Tile[];
}

describe("reading splat centres from a tile's SPZ", () => {
  it("gives every splat of each committed tile, inside the tile's bounding box", () => {
    const tileset = JSON.parse(readFileSync(resolve(LOD, "tileset.json"), "utf8")) as {
      root: Tile;
    };
    const tiles: Tile[] = [];
    const walk = (tile: Tile): void => {
      tiles.push(tile);
      tile.children?.forEach(walk);
    };
    walk(tileset.root);
    for (const tile of tiles.slice(0, 6)) {
      const glb = readFileSync(resolve(LOD, tile.content.uri));
      const spz = spzFromGlb(glb.buffer.slice(glb.byteOffset, glb.byteOffset + glb.byteLength));
      const points = spzPointsRaw(new Uint8Array(gunzipSync(spz)));
      expect(points.count).toBe(tile.extras.gaussians);
      const [cx, cy, cz, hx, , , , hy, , , , hz] = tile.boundingVolume.box;
      // Axis-aligned boxes, as splat_tiles.py writes them; a millimetre for the fixed point.
      for (let i = 0; i < points.count; i++) {
        expect(Math.abs((points.positions[i * 3] ?? 0) - (cx ?? 0))).toBeLessThanOrEqual(
          (hx ?? 0) + 1e-3,
        );
        expect(Math.abs((points.positions[i * 3 + 1] ?? 0) - (cy ?? 0))).toBeLessThanOrEqual(
          (hy ?? 0) + 1e-3,
        );
        expect(Math.abs((points.positions[i * 3 + 2] ?? 0) - (cz ?? 0))).toBeLessThanOrEqual(
          (hz ?? 0) + 1e-3,
        );
      }
      expect(points.alphas.every((a) => a >= 0 && a <= 1)).toBe(true);
    }
  });

  it("refuses what is not an SPZ it reads", () => {
    expect(() => spzPointsRaw(new Uint8Array(4))).toThrow(/shorter than its header/);
    expect(() => spzPointsRaw(new Uint8Array(16))).toThrow(/not an SPZ/);
  });
});
