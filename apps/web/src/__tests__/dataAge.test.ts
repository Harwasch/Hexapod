/**
 * Data age (lib/dataAge.ts): one date's age bucket, the later dates of its site, which of them
 * can honestly be compared, and the texture that marks "newer data here".
 */
import { gzipSync } from "node:zlib";

import { afterEach, describe, expect, it, vi } from "vitest";

import { encodeConfidenceGrid } from "@/dev/cellGridFixture";
import { cellTexelLayout, decodeCellGrid } from "@/lib/cellGrid";
import {
  AGE_BUCKETS,
  ageBucket,
  dataAgeTexels,
  laterDates,
  loadLaterGrids,
  sameFrame,
} from "@/lib/dataAge";

import { A1, A2, E1, E2, epoch, splat } from "./epochFixtures";

const DAY = 86_400_000;
const now = Date.parse("2026-06-01T15:00:00Z");

afterEach(() => vi.unstubAllGlobals());

describe("age buckets", () => {
  it("are under a month, 1-3, 3-6, 6-12 months, 1-2 years, older", () => {
    const at = (days: number) => new Date(now - days * DAY).toISOString();
    expect(AGE_BUCKETS).toHaveLength(6);
    expect(ageBucket(at(0), now)).toBe(0);
    expect(ageBucket(at(29.9), now)).toBe(0);
    expect(ageBucket(at(30), now)).toBe(1);
    expect(ageBucket(at(91), now)).toBe(2);
    expect(ageBucket(at(200), now)).toBe(3);
    expect(ageBucket(at(400), now)).toBe(4);
    expect(ageBucket(at(5000), now)).toBe(5);
    // A clock behind the capture's: fresh, not negative.
    expect(ageBucket(at(-3), now)).toBe(0);
    expect(ageBucket("not a date", now)).toBeNull();
    expect(ageBucket(null, now)).toBeNull();
  });
});

describe("the later dates of a site", () => {
  it("are the published ones observed after the shown model's, with their current models", () => {
    const E3 = epoch("e3", {
      ordinal: 3,
      observedAt: "2026-09-01T00:00:00Z",
      currentAssetId: "a3",
      status: "rejected",
    });
    const A3 = splat("a3", { epochId: "e3" });
    const context = { epochs: [E1, E2, E3], assets: [A1, A2, A3] };
    const fromMarch = laterDates(context, A1.id, "gaussian-splat");
    expect(fromMarch.shown?.id).toBe(E1.id);
    expect(fromMarch.later.map((o) => o.epoch.id)).toEqual([E2.id]);
    expect(laterDates(context, A2.id, "gaussian-splat").later).toEqual([]);
    // A model of no date has no later dates, and no age.
    expect(laterDates(context, "elsewhere", "gaussian-splat")).toEqual({ shown: null, later: [] });
  });
});

const ROOT = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -2468208.19, -3675852.33, 4575543.04, 1];

describe("which later dates are compared", () => {
  it("only one in the same frame with a grid; the rest are named, not guessed at", async () => {
    const newer = await encodeConfidenceGrid({
      level: 2,
      i0: 0,
      j0: 0,
      ni: 4,
      nj: 4,
      cell: () => ({ score: 200 }),
    });
    const tilesets: Record<string, unknown> = {
      "https://tiles.example/same/tileset.json": {
        root: {
          transform: ROOT.map((v, i) => (i === 12 ? v + 0.0004 : v)),
          extras: { confidence: newer.meta },
        },
      },
      "https://tiles.example/moved/tileset.json": {
        root: {
          transform: ROOT.map((v, i) => (i === 13 ? v + 5 : v)),
          extras: { confidence: newer.meta },
        },
      },
      "https://tiles.example/bare/tileset.json": { root: { transform: ROOT, extras: {} } },
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response(gzipSync(newer.planes, { level: 9 })))),
    );
    const later = ["same", "moved", "bare", "gone"].map((name, k) => ({
      epoch: epoch(`l${String(k)}`, { observedAt: `2026-0${String(k + 6)}-01T00:00:00Z` }),
      asset: splat(name, {
        source: { type: "3d-tiles-url", url: `https://tiles.example/${name}/tileset.json` },
      }),
    }));
    const ion = {
      epoch: epoch("ion", { observedAt: "2026-11-01T00:00:00Z" }),
      asset: splat("ion", { source: { type: "cesium-ion", assetId: 1 } } as never),
    };
    const grids = await loadLaterGrids([...later, ion], ROOT, (url) =>
      url in tilesets ? Promise.resolve(tilesets[url]) : Promise.reject(new Error("answered 404")),
    );
    expect(grids.map((g) => ("grid" in g ? "compared" : g.skipped))).toEqual([
      "compared",
      "Jul 1, 2026: placed in another frame",
      "Aug 1, 2026: not measured",
      "Sep 1, 2026: could not be read (answered 404)",
      "Nov 1, 2026: not a tileset this page can read",
    ]);
  });

  it("calls two root transforms one frame to a millimetre", () => {
    expect(
      sameFrame(
        ROOT,
        ROOT.map((v, i) => (i === 14 ? v + 0.0009 : v)),
      ),
    ).toBe(true);
    expect(
      sameFrame(
        ROOT,
        ROOT.map((v, i) => (i === 14 ? v + 0.002 : v)),
      ),
    ).toBe(false);
    expect(
      sameFrame(
        ROOT,
        ROOT.map((v, i) => (i === 1 ? v + 1e-6 : v)),
      ),
    ).toBe(false);
    expect(sameFrame(undefined, undefined)).toBe(true);
    expect(sameFrame(ROOT, undefined)).toBe(false);
  });
});

describe("the Data age texture", () => {
  it("is the shown date's bucket where it scored, striped where a later date has evidence", async () => {
    // Shown: 1 m cells over 8 x 4 m, scored everywhere but cell (0, 0).
    const shown = await encodeConfidenceGrid({
      level: 2,
      i0: 0,
      j0: 0,
      ni: 8,
      nj: 4,
      cell: (i, j) => ({ score: i === 0 && j === 0 ? 0 : 150 }),
      coarser: 3,
    });
    // Later: 2 m cells (level 3), evidence over x 4-8 only.
    const newer = await encodeConfidenceGrid({
      level: 3,
      i0: 0,
      j0: 0,
      ni: 4,
      nj: 2,
      cell: (i) => ({ score: i >= 2 ? 220 : 0 }),
    });
    const grid = await decodeCellGrid(shown.meta, shown.planes);
    const later = await decodeCellGrid(newer.meta, newer.planes);
    const { rg, newerCells } = dataAgeTexels(grid, 2, [later]);
    expect(rg).toHaveLength(cellTexelLayout(grid.meta).cells * 2);
    const at = (i: number, j: number) => [rg[(j * 8 + i) * 2], rg[(j * 8 + i) * 2 + 1]];
    expect(at(0, 0)).toEqual([0, 0]);
    expect(at(1, 0)).toEqual([3, 0]);
    expect(at(4, 0)).toEqual([3, 255]);
    expect(at(7, 3)).toEqual([3, 255]);
    expect(newerCells).toBe(16);
    // Level 3 (2 m, 4 x 2), right after the finest's 32 cells: east half striped.
    const level3 = (i: number, j: number) => rg[(32 + j * 4 + i) * 2 + 1];
    expect([level3(0, 0), level3(1, 1), level3(2, 0), level3(3, 1)]).toEqual([0, 0, 255, 255]);
    // No date: nothing gets an age, the stripes still say where newer data is.
    const undated = dataAgeTexels(grid, null, [later]);
    expect(undated.rg[4 * 2]).toBe(0);
    expect(undated.rg[4 * 2 + 1]).toBe(255);
    // No later dates: no stripes.
    expect(dataAgeTexels(grid, 0, []).newerCells).toBe(0);
  });
});
