import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";

import { describe, expect, it } from "vitest";

import {
  PrecomputedSolids,
  collisionMetaOf,
  parseCollision,
  type CollisionMeta,
} from "@/lib/collision";

/** Payload for solid cells (cell indices relative to the origin), as the packager writes it. */
function payload(cells: [number, number, number][]): { raw: Uint8Array; bricks: number } {
  const bricks = new Map<string, { b: [number, number, number]; mask: Uint8Array }>();
  for (const [ix, iy, iz] of cells) {
    const b: [number, number, number] = [ix >> 3, iy >> 3, iz >> 3];
    const key = b.join(",");
    const entry = bricks.get(key) ?? { b, mask: new Uint8Array(64) };
    const n = (ix & 7) + 8 * (iy & 7) + 64 * (iz & 7);
    entry.mask[n >> 3] = (entry.mask[n >> 3] ?? 0) | (1 << (n & 7));
    bricks.set(key, entry);
  }
  const raw = new Uint8Array(bricks.size * 76);
  const view = new DataView(raw.buffer);
  [...bricks.values()].forEach(({ b, mask }, r) => {
    view.setInt32(r * 76, b[0], true);
    view.setInt32(r * 76 + 4, b[1], true);
    view.setInt32(r * 76 + 8, b[2], true);
    raw.set(mask, r * 76 + 12);
  });
  return { raw, bricks: bricks.size };
}

const meta = (bricks: number, cell = 0.1, origin: [number, number, number] = [0, 0, 0]) =>
  ({
    format: "hexapod.collision",
    version: 1,
    uri: "collision.bin",
    cell,
    origin,
    brick: 8,
    bricks,
  }) as CollisionMeta;

describe("packaged collision", () => {
  it("reads the tileset's declaration, and nothing that is not one", () => {
    expect(collisionMetaOf({ collision: meta(3) })).toMatchObject({ cell: 0.1, bricks: 3 });
    expect(collisionMetaOf({ collision: { ...meta(3), version: 2 } })).toBeNull();
    expect(collisionMetaOf({})).toBeNull();
    expect(collisionMetaOf(undefined)).toBeNull();
  });

  it("puts each bit at the cell the format says", () => {
    const cells: [number, number, number][] = [
      [0, 0, 0],
      [7, 7, 7],
      [8, 0, 0],
      [3, 12, 21],
    ];
    const { raw, bricks } = payload(cells);
    const grid = parseCollision(raw, meta(bricks));
    for (const [x, y, z] of cells) expect(grid.solidAt(x, y, z)).toBe(true);
    expect(grid.solidAt(1, 0, 0)).toBe(false);
    expect(grid.solidAt(3, 12, 20)).toBe(false);
    expect(grid.solidCells).toBe(4);
  });

  it("answers rays and moves in world metres, from its origin", () => {
    // A wall of cells at x index 50 (5.0-5.1 m past the origin), 4 m square.
    const cells: [number, number, number][] = [];
    for (let y = 0; y < 40; y++) for (let z = 0; z < 40; z++) cells.push([50, y, z]);
    const { raw, bricks } = payload(cells);
    const solids = new PrecomputedSolids(parseCollision(raw, meta(bricks, 0.1, [100, 200, 0])));
    expect(solids.raycast([100, 202, 2], [1, 0, 0], 50)).toBeCloseTo(5, 5);
    // From far away the ray is clipped to the wall's box, not walked cell by cell.
    expect(solids.raycast([-5000, 202, 2], [1, 0, 0], 10_000)).toBeCloseTo(5105, 3);
    expect(solids.raycast([100, 202, 2], [-1, 0, 0], 50)).toBeNull();
    expect(solids.raycast([100, 202, 9], [1, 0, 0], 50)).toBeNull();
    const moved = solids.sweep([100, 202, 2], [110, 202, 2]);
    expect(moved.blocked).toBe(true);
    expect(moved.position[0]).toBeLessThan(105);
    expect(moved.position[0]).toBeGreaterThan(105 - solids.clearance - 0.2);
  });

  it("refuses a payload that does not match its declaration", () => {
    expect(() => parseCollision(new Uint8Array(70), meta(1))).toThrow(/bytes for 1 bricks/);
  });
});

describe("the packager's own file", () => {
  it("parses the fixture tree's collision.bin to the counts its tileset declares", () => {
    const folder = resolve(__dirname, "../../../../data/tiles/synthetic-tree/splat");
    const tileset = JSON.parse(readFileSync(resolve(folder, "tileset.json"), "utf8")) as {
      root: { extras?: unknown };
    };
    const meta = collisionMetaOf(tileset.root.extras);
    if (!meta) throw new Error("the fixture declares no collision grid");
    const grid = parseCollision(gunzipSync(readFileSync(resolve(folder, meta.uri))), meta);
    expect(grid.solidCells).toBe(meta.solidCells);
    expect(grid.empty).toBe(false);
    // A ray straight down onto a solid cell meets it, from above its column's top solid.
    const bounds = grid.solidBounds();
    if (!bounds) throw new Error("no bounds");
    let solid: [number, number, number] | null = null;
    for (let ix = bounds.lo[0]; ix <= bounds.hi[0] && !solid; ix++)
      for (let iy = bounds.lo[1]; iy <= bounds.hi[1] && !solid; iy++)
        for (let iz = bounds.hi[2]; iz >= bounds.lo[2] && !solid; iz--)
          if (grid.solidAt(ix, iy, iz)) solid = [ix, iy, iz];
    if (!solid) throw new Error("no solid cell");
    const [ox, oy, oz] = meta.origin;
    const x = ox + (solid[0] + 0.5) * meta.cell;
    const y = oy + (solid[1] + 0.5) * meta.cell;
    const top = oz + (bounds.hi[2] + 2) * meta.cell;
    const hit = grid.raycast([x, y, top], [0, 0, -1], 1000);
    expect(hit).not.toBeNull();
    expect(top - (hit ?? 0)).toBeCloseTo(oz + (solid[2] + 1) * meta.cell, 6);
  });
});
