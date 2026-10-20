/**
 * Per-cell view rules (lib/compositor.ts): which date owns each cell of the site lattice.
 *
 * The invariant first: at every point at most one source has a non-zero weight -- a cell never
 * blends two dates. Then each rule's choice: the newest observed date that saw the cell, older
 * ones where it did not, inferred layers only where nothing observed did and the rule allows
 * them, nothing observed after "As of"; a date without a grid covers its leaf tiles' boxes
 * and says so; the plane is at the coarsest finest level, coarser while too big; the seam
 * feather.
 */
import { describe, expect, it } from "vitest";

import { encodeConfidenceGrid } from "@/dev/cellGridFixture";
import { decodeCellGrid, type CellGrid } from "@/lib/cellGrid";
import {
  compositorWeight,
  featherWidth,
  leafTileBoxes,
  MAX_PLANE_CELLS,
  NO_OWNER,
  ownerAgeGrid,
  ownerAt,
  ownerPlane,
  planeLevel,
  type CompositorSource,
  type ViewRule,
} from "@/lib/compositor";
import { VIEW_RULES } from "@/features/views/ruleText";

/** A 1 m grid (level 2) over x in [0, 20), y in [0, 10): scored where `seen(i, j)`. */
async function grid(seen: (i: number, j: number) => boolean, level = 2): Promise<CellGrid> {
  const per = 2 ** (level - 2);
  const { meta, planes } = await encodeConfidenceGrid({
    level,
    i0: 0,
    j0: 0,
    ni: Math.ceil(20 / per),
    nj: Math.ceil(10 / per),
    cell: (i, j) => ({ score: seen(i * per, j * per) ? 200 : 0 }),
    coarser: 2,
  });
  return decodeCellGrid(meta, planes);
}

const MAR = "2026-03-01T15:00:00Z";
const JUN = "2026-06-01T15:00:00Z";
const SEP = "2026-09-01T15:00:00Z";

async function site(): Promise<CompositorSource[]> {
  return [
    // The frame's own date saw everything.
    { source: 0, epochId: "mar", observedAt: MAR, inferred: false, grid: await grid(() => true) },
    // June saw the east half only.
    {
      source: 1,
      epochId: "jun",
      observedAt: JUN,
      inferred: false,
      grid: await grid((i) => i >= 10),
    },
    // June's inferred layer fills a strip in the west, where June never looked.
    {
      source: 2,
      epochId: "jun",
      observedAt: JUN,
      inferred: true,
      boxes: [{ minX: 2, minY: 2, maxX: 4, maxY: 4 }],
    },
  ];
}

const ALL_RULES: ViewRule[] = ["latest", "ground-truth", "highlight-invented", "data-age"];

describe("the owner plane", () => {
  it("never gives a point two drawn sources, under any rule", async () => {
    const sources = await site();
    // A source outside the plane's grid too: nothing owns there.
    for (const rule of ALL_RULES) {
      const plane = ownerPlane(rule, sources, null);
      for (let k = 0; k < 4000; k += 1) {
        const x = -2 + ((k * 7919) % 2400) / 100;
        const y = -2 + ((k * 104729) % 1400) / 100;
        const drawn = [0, 1, 2, NO_OWNER].filter((s) => compositorWeight(plane, s, x, y) > 0);
        expect(drawn.length).toBeLessThanOrEqual(1);
        // And the one drawn is the cell's owner.
        if (drawn.length === 1) expect(drawn[0]).toBe(ownerAt(plane, x, y));
      }
    }
  });

  it("latest: the newest date that saw the cell, older ones elsewhere", async () => {
    const plane = ownerPlane("latest", await site(), null);
    expect(plane.cellM).toBe(1);
    expect(ownerAt(plane, 15.5, 5.5)).toBe(1);
    expect(ownerAt(plane, 5.5, 5.5)).toBe(0);
    // March saw the strip: an observed date wins over an inferred layer, however new.
    expect(ownerAt(plane, 3, 3)).toBe(0);
    // Off the site: nobody.
    expect(ownerAt(plane, 25, 5)).toBe(NO_OWNER);
    expect(plane.sources).toEqual([
      { source: 0, epochId: "mar", coverage: "grid", cells: 100 },
      { source: 1, epochId: "jun", coverage: "grid", cells: 100 },
      { source: 2, epochId: "jun", coverage: "tile-bounds", cells: 0 },
    ]);
  });

  it("an inferred layer owns only where nothing observed saw, and never under ground truth", async () => {
    const [mar, jun, fill] = await site();
    if (!mar || !jun || !fill) throw new Error("fixture");
    // March saw nothing in the west strip either, now.
    const blind = { ...mar, grid: await grid((i, j) => !(i >= 2 && i < 4 && j >= 2 && j < 4)) };
    const latest = ownerPlane("latest", [blind, jun, fill], null);
    expect(ownerAt(latest, 3, 3)).toBe(2);
    expect(ownerPlane("highlight-invented", [blind, jun, fill], null).owners).toEqual(
      latest.owners,
    );
    const truth = ownerPlane("ground-truth", [blind, jun, fill], null);
    expect(ownerAt(truth, 3, 3)).toBe(NO_OWNER);
    // Inferred layers hidden by the viewer (Inferred style: Hide) take nothing either.
    expect(
      ownerAt(ownerPlane("latest", [blind, jun, fill], null, { allowInferred: false }), 3, 3),
    ).toBe(NO_OWNER);
    for (const plane of [truth]) expect(plane.owners.includes(2)).toBe(false);
  });

  it("as of a date, leaves out everything observed after it", async () => {
    const sources = await site();
    const asOfMarch = ownerPlane("latest", sources, MAR);
    expect(ownerAt(asOfMarch, 15.5, 5.5)).toBe(0);
    expect(asOfMarch.owners.includes(1)).toBe(false);
    expect(asOfMarch.sources.map((s) => s.epochId)).toEqual(["mar"]);
    // A September date seen nowhere changes nothing as of June.
    const sep = {
      source: 3,
      epochId: "sep",
      observedAt: SEP,
      inferred: false,
      grid: await grid(() => true),
    };
    expect(ownerPlane("latest", [...sources, sep], JUN).owners).toEqual(
      ownerPlane("latest", sources, JUN).owners,
    );
    expect(ownerAt(ownerPlane("latest", [...sources, sep], null), 5.5, 5.5)).toBe(3);
  });

  it("a date without a grid covers the cells under its leaf tiles' boxes, and says so", async () => {
    const [mar] = await site();
    if (!mar) throw new Error("fixture");
    const upload: CompositorSource = {
      source: 1,
      epochId: "upload",
      observedAt: JUN,
      inferred: false,
      boxes: [{ minX: 10.2, minY: 0, maxX: 12.5, maxY: 1 }],
    };
    const plane = ownerPlane("latest", [mar, upload], null);
    // Every cell the box overlaps, even in part.
    expect(ownerAt(plane, 10.1, 0.5)).toBe(1);
    expect(ownerAt(plane, 12.9, 0.5)).toBe(1);
    expect(ownerAt(plane, 13.1, 0.5)).toBe(0);
    expect(ownerAt(plane, 11, 1.5)).toBe(0);
    expect(plane.sources.find((s) => s.source === 1)).toEqual({
      source: 1,
      epochId: "upload",
      coverage: "tile-bounds",
      cells: 3,
    });
    // No grid at all: 1 m cells.
    expect(ownerPlane("latest", [upload], null).cellM).toBe(1);
  });

  it("is drawn at the coarsest finest level, and coarser while it would be too big", async () => {
    const fine = {
      source: 0,
      epochId: "a",
      observedAt: MAR,
      inferred: false,
      grid: await grid(() => true, 2),
    };
    const coarse = {
      source: 1,
      epochId: "b",
      observedAt: JUN,
      inferred: false,
      grid: await grid(() => true, 3),
    };
    expect(planeLevel([fine, coarse])).toBe(3);
    expect(ownerPlane("latest", [fine, coarse], null).cellM).toBe(2);
    const huge: CompositorSource = {
      source: 0,
      epochId: "huge",
      observedAt: MAR,
      inferred: false,
      boxes: [{ minX: 0, minY: 0, maxX: 4000, maxY: 4000 }],
    };
    const plane = ownerPlane("latest", [huge], null);
    expect(plane.ni * plane.nj).toBeLessThanOrEqual(MAX_PLANE_CELLS);
    expect(plane.cellM).toBe(2);
  });

  it("feathers a splat near a seam to half its opacity, and leaves the inside alone", async () => {
    const plane = ownerPlane("latest", await site(), null);
    const w = featherWidth(plane.cellM);
    expect(w).toBe(0.25);
    expect(featherWidth(4)).toBe(0.5);
    // June's west edge is at x = 10: right on it, half; a feather's width in, whole.
    expect(compositorWeight(plane, 1, 10.0001, 5.5)).toBeCloseTo(0.5, 2);
    expect(compositorWeight(plane, 1, 10 + w, 5.5)).toBe(1);
    expect(compositorWeight(plane, 0, 9.9999, 5.5)).toBeCloseTo(0.5, 2);
    // An edge between two cells of one owner is no seam.
    expect(compositorWeight(plane, 1, 15.0001, 5.5)).toBe(1);
    // The site's outer edge is a seam with nobody.
    expect(compositorWeight(plane, 1, 19.9999, 5.5)).toBeCloseTo(0.5, 2);
  });

  it("colours each cell by its owner's age under data age, striped where a later date saw", async () => {
    const sources = await site();
    const plane = ownerPlane("data-age", sources, MAR);
    const now = Date.parse("2026-10-20T00:00:00Z");
    const { grid: g, rg } = ownerAgeGrid(plane, sources, MAR, now);
    expect(g.meta.levels).toHaveLength(1);
    const at = (i: number, j: number) => (j - plane.j0) * plane.ni + (i - plane.i0);
    // March, seven months old: bucket 3 (6-12 months), so R = 4; June saw the east: striped.
    expect([rg[at(15, 5) * 2], rg[at(15, 5) * 2 + 1]]).toEqual([4, 255]);
    expect([rg[at(5, 5) * 2], rg[at(5, 5) * 2 + 1]]).toEqual([4, 0]);
    expect(g.planes[at(5, 5)]).toBe(1);
    // A cell nobody owns has no age, though a later date (given as NO_OWNER, as Data age's
    // "newer" dates are) saw it: only its stripe.
    const [mar] = sources;
    if (!mar) throw new Error("fixture");
    const blind = { ...mar, grid: await grid((i, j) => !(i >= 2 && i < 4 && j >= 2 && j < 4)) };
    const later: CompositorSource = {
      source: NO_OWNER,
      epochId: "sep",
      observedAt: SEP,
      inferred: false,
      grid: await grid(() => true),
    };
    const holed = ownerPlane("data-age", [blind], MAR);
    const { rg: holedRg } = ownerAgeGrid(holed, [blind, later], MAR, now);
    const hole = (3 - holed.j0) * holed.ni + (3 - holed.i0);
    expect(holed.owners[hole]).toBe(NO_OWNER);
    expect([holedRg[hole * 2], holedRg[hole * 2 + 1]]).toEqual([0, 255]);
  });

  it("names its four rules", () => {
    expect(VIEW_RULES.map((r) => r.value)).toEqual(ALL_RULES);
  });
});

describe("a tileset's leaf boxes", () => {
  it("composes child transforms down from the root, which is the frame, and skips empty leaves", () => {
    const boxes = leafTileBoxes({
      transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5000, 5000, 0, 1],
      boundingVolume: { box: [0, 0, 0, 100, 0, 0, 0, 100, 0, 0, 0, 10] },
      content: { uri: "root.glb" },
      children: [
        {
          boundingVolume: { box: [2, 3, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1] },
          content: { uri: "a.glb" },
        },
        {
          // A tile with its own origin (decision D1): translation from its parent.
          transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 100, 0, 0, 1],
          boundingVolume: { box: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1] },
          children: [
            { boundingVolume: { sphere: [1, 1, 0, 0.5] }, content: { uri: "b.glb" } },
            { boundingVolume: { box: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1] } },
          ],
        },
      ],
    });
    expect(boxes).toEqual([
      { minX: 1, minY: 1, maxX: 3, maxY: 5 },
      { minX: 100.5, minY: 0.5, maxX: 101.5, maxY: 1.5 },
    ]);
    expect(leafTileBoxes(undefined)).toEqual([]);
  });
});
