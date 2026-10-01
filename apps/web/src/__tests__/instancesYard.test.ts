/**
 * The committed e2e fixture for scene instances (`data/tiles/synthetic-yard/instances/`, the
 * yard segmented against its own labels; e2e/instances.spec.ts) still fits the committed yard
 * tiles: every tile's checksum is listed, with one id per gaussian. If the yard is re-packed,
 * this fails here rather than as an unexplained pixel count in the browser.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { checksumPositions } from "@twin/world";
import { describe, expect, it } from "vitest";

import { parseInstances, tileInstanceIds } from "@/lib/instances";

import { yardTiles } from "./splatYardFixture";

const doc = parseInstances(
  JSON.parse(
    readFileSync(
      resolve(process.cwd(), "../../data/tiles/synthetic-yard/instances/instances.json"),
      "utf8",
    ),
  ),
);

describe("the yard's instances fixture", () => {
  it("reads cleanly", () => {
    expect(doc).not.toBeNull();
    expect(doc?.issues).toEqual([]);
    expect(doc?.instances.length).toBeGreaterThan(10);
  });

  it("lists every yard tile, one id per gaussian, most of them an instance's", () => {
    if (!doc) throw new Error("no document");
    expect(doc.tiles.size).toBe(yardTiles.size);
    let claimed = 0;
    let total = 0;
    for (const tile of yardTiles.values()) {
      const ids = tileInstanceIds(doc, checksumPositions(tile.local));
      expect(ids?.length, tile.uri).toBe(tile.local.length / 3);
      if (!tile.leaf || !ids) continue;
      total += ids.length;
      for (const id of ids) if (id !== 0) claimed += 1;
    }
    expect(claimed / total).toBeGreaterThan(0.95);
  });
});
