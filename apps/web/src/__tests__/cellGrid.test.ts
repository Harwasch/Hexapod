/**
 * hexapod.cellgrid v1 (lib/cellGrid.ts): the descriptor read off a root tile's extras, the
 * planes fetched, checked and addressed, the level a distance asks for, and what `weak-v1`
 * calls a cell -- against a grid built by the TypeScript fixture encoder from the container's
 * specification (src/dev/cellGridFixture.ts).
 */
import { gzipSync } from "node:zlib";

import { afterEach, describe, expect, it, vi } from "vitest";

import { encodeConfidenceGrid, REASON_BITS, type FixtureCell } from "@/dev/cellGridFixture";
import {
  cellAt,
  cellAtIndex,
  cellGridMeta,
  cellLevelFor,
  cellVerdict,
  decodeCellGrid,
  evidenceAt,
  loadCellGrid,
  MAX_LATTICE_LEVEL,
  type CellGridMeta,
} from "@/lib/cellGrid";

/** 1 m cells (level 2) over x 0-8 m, y -2-4 m: weak west of x = 4, strong east. */
const cell = (i: number, j: number): FixtureCell =>
  i < 4
    ? { score: 40 + (j + 2), reasons: REASON_BITS.views, relief: 2 }
    : { score: 250 - i, reasons: 0, relief: i };

const fixture = { level: 2, i0: 0, j0: -2, ni: 8, nj: 6, cell, coarser: 3 };

afterEach(() => vi.unstubAllGlobals());

describe("the descriptor", () => {
  it("reads extras.confidence and refuses what it cannot address", async () => {
    const { meta } = await encodeConfidenceGrid(fixture);
    const read = cellGridMeta({ confidence: meta, viewCones: {} });
    expect(read?.levels.map((l) => l.level)).toEqual([2, 3, 4, 5]);
    expect(cellGridMeta({ viewCones: {} })).toBeNull();
    expect(cellGridMeta({ confidence: { ...meta, version: 2 } })).toBeNull();
    expect(cellGridMeta({ confidence: { ...meta, frame: "ecef" } })).toBeNull();
    const [first, ...rest] = meta.levels;
    if (!first) throw new Error("no levels");
    // A plane whose size disagrees with its cells and channels.
    const short = { ...first, bytes: first.bytes - 1 };
    expect(cellGridMeta({ confidence: { ...meta, levels: [short, ...rest] } })).toBeNull();
    // A level that skips a step of the lattice.
    expect(cellGridMeta({ confidence: { ...meta, levels: [first, ...rest.slice(1)] } })).toBeNull();
    // A cell size that is not the lattice's.
    const wrong = { ...first, cellM: 1.5 };
    expect(cellGridMeta({ confidence: { ...meta, levels: [wrong, ...rest] } })).toBeNull();
    // Listed coarsest first: sorted, not refused.
    const reversed = cellGridMeta({ confidence: { ...meta, levels: [...meta.levels].reverse() } });
    expect(reversed?.levels[0]?.level).toBe(2);
  });

  it("refuses a level or a plane size no site lattice has", async () => {
    const { meta } = await encodeConfidenceGrid({ ...fixture, coarser: 0 });
    const [first] = meta.levels;
    if (!first) throw new Error("no levels");
    const at = (level: number, extra: Record<string, unknown> = {}) => ({
      confidence: {
        ...meta,
        lattice: { ...meta.lattice, level },
        levels: [{ ...first, level, cellM: 0.25 * 2 ** level, ...extra }],
      },
    });
    // Level 1024 with a finite cell size: 0.25 m x 2^1024 is Infinity, and |1 - Inf| is not
    // more than Inf x 1e-9, so it used to pass, and comparing grids then walked forever.
    expect(cellGridMeta(at(1024, { cellM: 1 }))).toBeNull();
    expect(cellGridMeta(at(MAX_LATTICE_LEVEL + 1))).toBeNull();
    expect(cellGridMeta(at(-1))).toBeNull();
    expect(cellGridMeta(at(MAX_LATTICE_LEVEL))).not.toBeNull();
    // More cells in a plane than C3 lets a level hold (2^22).
    const big = { ni: 4096, nj: 1025, bytes: 4096 * 1025 * 3 };
    expect(cellGridMeta(at(2, big))).toBeNull();
  });

  it("is the same bytes every time it is built", async () => {
    const a = await encodeConfidenceGrid(fixture);
    const b = await encodeConfidenceGrid(fixture);
    expect(Buffer.from(a.planes).equals(Buffer.from(b.planes))).toBe(true);
    expect(a.meta).toEqual(b.meta);
    // Level 9, no timestamp: the container's gzip is reproducible too.
    expect(gzipSync(a.planes, { level: 9 }).equals(gzipSync(b.planes, { level: 9 }))).toBe(true);
    expect(a.meta.summary).toMatchObject({ cells: 48, scored: 48, weak: 24, weakShare: 0.5 });
  });
});

describe("the planes", () => {
  async function served(meta: CellGridMeta, planes: Uint8Array) {
    const fetched: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        fetched.push(url);
        return Promise.resolve(new Response(gzipSync(planes, { level: 9 })));
      }),
    );
    const grid = await loadCellGrid("/fixture-tiles/yard/tileset.json", meta);
    return { grid, fetched };
  }

  it("are fetched beside the tileset and addressed by tileset-frame metres", async () => {
    const { meta, planes } = await encodeConfidenceGrid(fixture);
    const { grid, fetched } = await served(meta, planes);
    expect(fetched).toEqual([`${location.origin}/fixture-tiles/yard/confidence.bin`]);
    expect(cellAt(grid, 0.5, -1.5, 2)).toEqual({ score: 40, reasons: 1, relief: 2 });
    expect(cellAt(grid, 6.99, 3.2, 2)).toEqual({ score: 244, reasons: 0, relief: 6 });
    // Outside the grid, and below its origin: no cell rather than a neighbour's.
    expect(cellAt(grid, 8.01, 0, 2)).toBeNull();
    expect(cellAt(grid, 0, -2.01, 2)).toBeNull();
    expect(cellAt(grid, -0.01, 0, 2)).toBeNull();
  });

  it("carry C3's pyramid: the weakest scored child, every reason, the highest relief", async () => {
    const { meta, planes } = await encodeConfidenceGrid(fixture);
    const { grid } = await served(meta, planes);
    // Level 3 (2 m) cell (1, -1) covers level-2 cells i 2-3, j -2..-1: all weak.
    expect(cellAtIndex(grid, 3, 1, -1)).toEqual({ score: 40, reasons: 1, relief: 2 });
    // Level 4 (4 m) cell (0, -1) covers i 0-3: weak; (1, 0) covers i 4-7, j 0-3: strong.
    expect(cellAtIndex(grid, 4, 1, 0)).toEqual({ score: 243, reasons: 0, relief: 7 });
    // Level 5 (8 m) cell (0, -1) holds both halves (j -2..-1): the weak one wins.
    expect(cellAtIndex(grid, 5, 0, -1)).toEqual({ score: 40, reasons: 1, relief: 7 });
  });

  it("are refused when short or when they do not match their checksum", async () => {
    const { meta, planes } = await encodeConfidenceGrid(fixture);
    await expect(decodeCellGrid(meta, planes.subarray(0, planes.length - 1))).rejects.toThrow(
      /bytes; its levels need/,
    );
    const tampered = new Uint8Array(planes);
    tampered[0] = (tampered[0] ?? 0) ^ 1;
    await expect(decodeCellGrid(meta, tampered)).rejects.toThrow(/sha256/);
    vi.stubGlobal("fetch", () => Promise.resolve(new Response("", { status: 404 })));
    await expect(loadCellGrid("/t/tileset.json", meta)).rejects.toThrow(/404/);
  });
});

describe("which level a splat reads", () => {
  it("is the finest whose cell covers the minimum pixels at that distance", () => {
    // 1 m cells, 1000 px a radian, at least 4 px: fine to 250 m, then one level per doubling.
    expect(cellLevelFor(10, 1000, 1, 4)).toBe(0);
    expect(cellLevelFor(250, 1000, 1, 4)).toBe(0);
    expect(cellLevelFor(251, 1000, 1, 4)).toBe(1);
    expect(cellLevelFor(500, 1000, 1, 4)).toBe(1);
    expect(cellLevelFor(2000, 1000, 1, 4)).toBe(3);
    expect(cellLevelFor(0, 1000, 1, 4)).toBe(0);
  });
});

describe("weak-v1", () => {
  it("is weak below 128 or unseen, no evidence at 0, strong in the upper half", async () => {
    const { meta } = await encodeConfidenceGrid(fixture);
    expect(cellVerdict(0, 0, meta)).toBe("none");
    expect(cellVerdict(0, REASON_BITS.unseen, meta)).toBe("weak");
    expect(cellVerdict(127, 0, meta)).toBe("weak");
    expect(cellVerdict(128, 0, meta)).toBe("fair");
    expect(cellVerdict(191, 0, meta)).toBe("fair");
    expect(cellVerdict(192, 0, meta)).toBe("strong");
    expect(cellVerdict(250, REASON_BITS.unseen, meta)).toBe("weak");
  });
});

describe("evidence across two grids' levels", () => {
  it("reads a containing cell for a finer level and any covered cell for a coarser", async () => {
    // A newer epoch scored only x 4-6, y 0-2 (1 m cells).
    const newer = await encodeConfidenceGrid({
      level: 2,
      i0: 0,
      j0: 0,
      ni: 8,
      nj: 4,
      cell: (i, j) => ({ score: i >= 4 && i < 6 && j < 2 ? 200 : 0 }),
      coarser: 2,
    });
    const grid = await decodeCellGrid(newer.meta, newer.planes);
    expect(evidenceAt(grid, 2, 4, 0)).toBe(true);
    expect(evidenceAt(grid, 2, 3, 0)).toBe(false);
    // Finer than the grid (0.5 m, level 1): the 1 m cell holding it.
    expect(evidenceAt(grid, 1, 9, 1)).toBe(true);
    expect(evidenceAt(grid, 1, 7, 1)).toBe(false);
    // Coarser than the grid's coarsest (level 4, 4 m): any 4 m cell under it.
    expect(evidenceAt(grid, 5, 0, 0)).toBe(true);
    expect(evidenceAt(grid, 5, 1, 0)).toBe(false);
    // Far coarser (2^36 of the grid's coarsest cells a side): only the cells the grid has are
    // read, never every cell the request covers.
    expect(evidenceAt(grid, 40, 0, 0)).toBe(true);
    expect(evidenceAt(grid, 40, -1, 0)).toBe(false);
  });
});
