import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  DEFAULT_SPLAT_BUDGET,
  OPTIONS_STORAGE,
  detailScreenSpaceScale,
  splatBudget,
} from "@/lib/detail";
import { chooseTiles, countTiles, parseTileset, type TileNode } from "@/view/tiles";

function tile(
  uri: string,
  gaussians: number,
  geometricError: number,
  children: TileNode[] = [],
): TileNode {
  return { uri, gaussians, geometricError, children };
}

/**
 * The shape splat_tiles.py writes: a root subset, octants with their own subsets, and leaves
 * (error 0) holding the rest. 100k a tile, 780k in all.
 */
const scan = tile("splat.glb", 100_000, 0.4, [
  tile("splat_1.glb", 100_000, 0.1, [
    tile("splat_1-7.glb", 90_000, 0),
    tile("splat_1-0123.glb", 80_000, 0),
  ]),
  tile("splat_3.glb", 100_000, 0.2, [
    tile("splat_3-5.glb", 100_000, 0.05, [tile("splat_3-5-0.glb", 60_000, 0)]),
    tile("splat_3-012.glb", 70_000, 0),
  ]),
  tile("splat_02.glb", 80_000, 0),
]);

const uris = (tiles: TileNode[]): string[] => tiles.map((chosen) => chosen.uri);

describe("the viewer's tile budget", () => {
  it("loads the whole scan when the budget allows it", () => {
    const chosen = chooseTiles(scan, 1_000_000);
    expect(chosen).toHaveLength(countTiles(scan).tiles);
    expect(countTiles(scan).gaussians).toBe(780_000);
  });

  it("refines where the error is largest first, and never a child before its parent", () => {
    const chosen = uris(chooseTiles(scan, 400_000));
    // Root (always), then the root's children (their region's error is the root's 0.4):
    // 3 and 1 and the packed leaf 02 -- 380k -- and nothing else fits.
    expect(chosen).toEqual(["splat.glb", "splat_1.glb", "splat_3.glb", "splat_02.glb"]);
    const order = chooseTiles(scan, 1_000_000).map((chosen) => chosen.uri);
    expect(order.indexOf("splat_3.glb")).toBeLessThan(order.indexOf("splat_3-5.glb"));
    expect(order.indexOf("splat_3-5.glb")).toBeLessThan(order.indexOf("splat_3-5-0.glb"));
    // Under splat_3 (error 0.2) comes before under splat_1 (0.1).
    expect(order.indexOf("splat_3-5.glb")).toBeLessThan(order.indexOf("splat_1-7.glb"));
  });

  it("never exceeds the budget, and fills what a skipped tile leaves", () => {
    for (const budget of [150_000, 200_000, 250_000, 400_000, 800_000]) {
      const chosen = chooseTiles(scan, budget);
      const spent = chosen.reduce((sum, chosenTile) => sum + chosenTile.gaussians, 0);
      expect(spent).toBeLessThanOrEqual(Math.max(budget, 100_000));
    }
    // 200k: the root, then splat_1 (100k) fits and splat_3 would not.
    expect(uris(chooseTiles(scan, 200_000))).toEqual(["splat.glb", "splat_1.glb"]);
    // 290k: splat_1 and splat_3 do not both fit, but the 80k packed leaf does.
    expect(uris(chooseTiles(scan, 290_000))).toEqual(["splat.glb", "splat_1.glb", "splat_02.glb"]);
  });

  it("always loads the root, even a pre-hierarchy single tile with no count", () => {
    const old = parseTileset({ root: { content: { uri: "splat.glb" }, geometricError: 0.5 } });
    expect(uris(chooseTiles(old, 200_000))).toEqual(["splat.glb"]);
    expect(() => parseTileset({ root: { geometricError: 1 } })).toThrow(/names no content/);
    expect(() => parseTileset({})).toThrow(/no root/);
  });

  it("reads the committed tree's tileset as one tile of 12,000", () => {
    const path = resolve(process.cwd(), "../../data/tiles/synthetic-tree/splat/tileset.json");
    const root = parseTileset(JSON.parse(readFileSync(path, "utf8")) as unknown);
    expect(root).toMatchObject({ uri: "splat.glb", gaussians: 12_000, geometricError: 0 });
    expect(chooseTiles(root, DEFAULT_SPLAT_BUDGET)).toEqual([root]);
  });
});

describe("the Detail choice", () => {
  afterEach(() => localStorage.clear());

  it("is read from the phone page's saved options", () => {
    expect(splatBudget()).toBe(DEFAULT_SPLAT_BUDGET);
    localStorage.setItem(OPTIONS_STORAGE, JSON.stringify({ detail: "800000" }));
    expect(splatBudget()).toBe(800_000);
    localStorage.setItem(OPTIONS_STORAGE, JSON.stringify({ detail: "12" }));
    expect(splatBudget()).toBe(DEFAULT_SPLAT_BUDGET);
  });

  it("scales the globe's screen-space error by the square root of the budget", () => {
    expect(detailScreenSpaceScale(DEFAULT_SPLAT_BUDGET)).toBe(1);
    expect(detailScreenSpaceScale(200_000)).toBeCloseTo(Math.SQRT2);
    expect(detailScreenSpaceScale(800_000)).toBeCloseTo(Math.SQRT1_2);
  });
});
