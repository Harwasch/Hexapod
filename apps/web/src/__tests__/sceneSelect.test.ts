import { describe, expect, it } from "vitest";

import {
  customId,
  decodeRanges,
  encodeRanges,
  isCustomId,
  parseCustomSets,
  rangesLength,
  serializeCustomSets,
  setFromInstances,
  withCustomSets,
  type CustomSet,
} from "@/lib/customSets";
import { parseInstances, tileInstanceIds, withDescendants } from "@/lib/instances";
import {
  bestByIoU,
  bestByIoUIndexed,
  bestSet,
  bestSetByIoU,
  bestSetByIoUIndexed,
  buildCandidates,
  chainOf,
  chipText,
  combinationLabel,
  commonChain,
  cycleIndex,
  drillIndex,
  hiddenForShowOnly,
  levelText,
  PAINT_PARENT_SLACK,
  PAINT_SET_GAIN,
  paintedCount,
  paintIndex,
  paintSums,
  paintSumsIndexed,
  selectionLabel,
  setIoU,
  splatShares,
  steadySet,
  type PaintSample,
  type PaintSetMatch,
  type PaintSums,
} from "@/lib/sceneSelect";
import {
  BrushMask,
  paintedSplats,
  projectTiles,
  visibleSplats,
  type ScreenSplats,
} from "@/lib/splatPaint";
import { buildTileIndex, castRay, hitWeights, labelsNear, type PickTile } from "@/lib/splatPick";
import { selectedId, selectedIds, useSceneSelect } from "@/state/sceneSelect";

const CHECKSUM = "fnv1a32:6:0000abcd";

/** A tree (1) with a crown (3) and a trunk (4), a shrub (2), and a tile of six splats. */
function doc() {
  const parsed = parseInstances({
    format: "hexapod.instances",
    version: 1,
    instances: [
      {
        id: 1,
        parent: null,
        level: 0,
        splats: 4,
        bounds: { min: [0, 0, 0], max: [2, 2, 6] },
        tags: [{ label: "tree", score: 0.9 }],
      },
      { id: 2, parent: null, level: 0, splats: 2, bounds: { min: [5, 0, 0], max: [6, 1, 1] } },
      { id: 3, parent: 1, level: 1, splats: 2, bounds: { min: [0, 0, 3], max: [2, 2, 6] } },
      {
        id: 4,
        parent: 1,
        level: 1,
        splats: 2,
        bounds: { min: [0.8, 0.8, 0], max: [1.2, 1.2, 3] },
        category: "trunk_wood",
      },
    ],
    // Splats: 0,1 crown; 2,3 trunk; 4,5 shrub.
    tiles: { [CHECKSUM]: [3, 2, 4, 2, 2, 2] },
  });
  if (!parsed) throw new Error("fixture did not parse");
  return parsed;
}

function tile(positions: number[], radius = 0.1, opacity = 0.9): PickTile {
  const count = positions.length / 3;
  return {
    checksum: CHECKSUM,
    count,
    positions: Float32Array.from(positions),
    radii: new Float32Array(count).fill(radius),
    opacity: new Float32Array(count).fill(opacity),
  };
}

describe("candidates", () => {
  it("walks a chain from the leaf to the top level", () => {
    expect(chainOf(doc(), 3)).toEqual([3, 1]);
    expect(chainOf(doc(), 2)).toEqual([2]);
    expect(chainOf(doc(), 99)).toEqual([]);
  });

  it("offers the main hit's chain, then other instances near the front", () => {
    const weights = new Map([
      [3, { weight: 0.6, t: 10 }],
      [4, { weight: 0.3, t: 10.4 }],
      [2, { weight: 0.2, t: 30 }],
      [0, { weight: 0.9, t: 9 }],
    ]);
    expect(buildCandidates(doc(), weights)).toEqual({ ids: [3, 1, 4], chain: 2 });
  });

  it("drops a faint or far hit, and offers nothing without an instance", () => {
    const faint = new Map([
      [3, { weight: 0.9, t: 10 }],
      [2, { weight: 0.05, t: 10 }],
    ]);
    expect(buildCandidates(doc(), faint).ids).toEqual([3, 1]);
    expect(buildCandidates(doc(), new Map([[0, { weight: 1, t: 1 }]]))).toEqual({
      ids: [],
      chain: 0,
    });
  });

  it("chooses the whole object first, then one level finer a click", () => {
    // A spool (1) of planks (5) cut in pieces (9), and a pallet (7) beside it.
    const candidates = { ids: [9, 5, 1, 7], chain: 3 };
    // Nothing selected: the top of the chain.
    expect(drillIndex(candidates, null)).toBe(2);
    // Inside the chain: one level deeper toward the hit, and at the leaf it stays.
    expect(drillIndex(candidates, 1)).toBe(1);
    expect(drillIndex(candidates, 5)).toBe(0);
    expect(drillIndex(candidates, 9)).toBe(0);
    // Another object, or the one beside it: the whole object again.
    expect(drillIndex(candidates, 42)).toBe(2);
    expect(drillIndex(candidates, 7)).toBe(2);
    expect(drillIndex({ ids: [], chain: 0 }, null)).toBe(-1);
  });

  it("starts below a top that is most of the scan", () => {
    const candidates = { ids: [9, 5, 1], chain: 3 };
    // The top is 80% of the scan: its child on the hit's chain instead.
    const share = (id: number): number => (id === 1 ? 0.8 : 0.1);
    expect(drillIndex(candidates, null, share)).toBe(1);
    // Two levels that big: down past both; a lone level is all there is.
    expect(drillIndex(candidates, null, (id) => (id === 9 ? 0.1 : 0.9))).toBe(0);
    expect(drillIndex({ ids: [1], chain: 1 }, null, () => 1)).toBe(0);
    // Drilling from the big top (cycled to) still goes one level deeper.
    expect(drillIndex(candidates, 1, share)).toBe(1);
  });

  it("measures an instance's share of the scan with everything below it", () => {
    const parsed = parseInstances({
      format: "hexapod.instances",
      version: 1,
      instances: [
        { id: 1, parent: null, level: 0, splats: 10, bounds: { min: [0, 0, 0], max: [1, 1, 1] } },
        { id: 2, parent: 1, level: 1, splats: 50, bounds: { min: [0, 0, 0], max: [1, 1, 1] } },
        { id: 3, parent: null, level: 0, splats: 40, bounds: { min: [0, 0, 0], max: [1, 1, 1] } },
      ],
      tiles: {},
    });
    if (!parsed) throw new Error("fixture did not parse");
    const shares = splatShares(parsed);
    expect(shares.get(1)).toBeCloseTo(0.6);
    expect(shares.get(2)).toBeCloseTo(0.5);
    expect(shares.get(3)).toBeCloseTo(0.4);
    expect(splatShares(parsed)).toBe(shares);
  });

  it("cycles both ways around the candidates", () => {
    expect(cycleIndex(0, 3, 1)).toBe(1);
    expect(cycleIndex(2, 3, 1)).toBe(0);
    expect(cycleIndex(0, 3, -1)).toBe(2);
    expect(cycleIndex(0, 0, 1)).toBe(-1);
  });

  it("names a candidate by tag, then category, never by id", () => {
    const d = doc();
    expect(selectionLabel(d.byId.get(1), 1)).toBe("Tree");
    // A category, as newer files carry (the objects panel's `Instance.category`).
    const trunk = { ...d.byId.get(4), category: "trunk_wood" } as Parameters<
      typeof selectionLabel
    >[0];
    expect(selectionLabel(trunk, 4)).toBe("Trunk wood");
    // Never an id: the category the objects panel files it under.
    expect(selectionLabel(d.byId.get(2), 2, "trees")).toBe("Trees");
    expect(selectionLabel(d.byId.get(2), 2)).toBe("Unnamed object");
    expect(chipText("Tree", 2, 3, 3)).toBe("Tree · 1 of 3");
    expect(chipText("Tree", 0, 1, 1)).toBe("Tree");
  });

  it("counts a chain's levels from the whole object, then the instances nearby", () => {
    // Leaf → top of the chain first in the list, so index 2 of a chain of 3 is the top.
    expect(levelText(2, 3, 3)).toBe("1 of 3");
    expect(levelText(0, 3, 3)).toBe("3 of 3");
    expect(levelText(2, 3, 5)).toBe("1 of 3 · +2 nearby");
    expect(levelText(3, 3, 5)).toBe("Nearby 1 of 2");
    expect(levelText(4, 3, 5)).toBe("Nearby 2 of 2");
    // A chain of one with others beside it, and a lone candidate.
    expect(levelText(0, 1, 3)).toBe("+2 nearby");
    expect(levelText(0, 1, 1)).toBe("");
  });

  it("hides everything but a selection and its ancestors for Show only", () => {
    const d = doc();
    expect(hiddenForShowOnly(d, withDescendants(d, [3]))).toEqual([2, 4]);
    expect(hiddenForShowOnly(d, withDescendants(d, [1]))).toEqual([2]);
  });
});

describe("picking", () => {
  it("composites the hits front to back: what is behind a solid splat counts for little", () => {
    // Three splats along +x at 5, 6 and 20 m, the first nearly opaque.
    const t = tile([5, 0, 0, 6, 0, 0, 20, 0, 0], 0.1, 0.95);
    const hits = castRay([t], { origin: [0, 0, 0], direction: [1, 0, 0] });
    expect(hits.map((h) => h.index)).toEqual([0, 1]);
    expect(hits[0]?.weight).toBeCloseTo(0.95, 2);
    expect(hits[1]?.weight ?? 0).toBeLessThan(0.06);
  });

  it("misses splats away from the ray, and skips what is not drawn", () => {
    const t = tile([5, 3, 0, 8, 0, 0], 0.1);
    const hits = castRay([t], { origin: [0, 0, 0], direction: [2, 0, 0] });
    expect(hits.map((h) => h.index)).toEqual([1]);
    expect(
      castRay([t], { origin: [0, 0, 0], direction: [1, 0, 0] }, { include: () => false }),
    ).toEqual([]);
  });

  it("finds the same hits through the block index as by brute force", () => {
    const positions: number[] = [];
    for (let i = 0; i < 500; i++) positions.push((i * 37) % 50, ((i * 11) % 23) / 10, 0);
    const t = tile(positions, 0.08, 0.5);
    const index = buildTileIndex(t);
    expect([...index.order].sort((a, b) => a - b)).toEqual([...Array(500).keys()]);
    const hits = castRay([t], { origin: [-1, 0, 0], direction: [1, 0, 0] });
    for (const hit of hits) expect(Math.abs(positions[hit.index * 3 + 1] ?? 9)).toBeLessThan(0.3);
    expect(hits.length).toBeGreaterThan(0);
  });

  it("sums each instance's share of the pixel", () => {
    const d = doc();
    const t = tile([0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10], 0.1, 0.3);
    const ids = tileInstanceIds(d, CHECKSUM);
    const hits = castRay([t], { origin: [0, 0, 0], direction: [0, 0, 1] });
    const weights = hitWeights(hits, (h) => ids?.[h.index] ?? 0);
    expect([...weights.keys()]).toEqual([3, 4, 2]);
    expect(weights.get(3)?.weight ?? 0).toBeGreaterThan(weights.get(4)?.weight ?? 0);
  });

  it("labels an unlabelled spot by the labelled splats drawn around it", () => {
    // Splat 0 unlabelled where the ray meets the scan; 7 and 9 labelled beside it, 9 nearer.
    const t = tile([0, 0, 5, 0.3, 0, 5, -0.1, 0, 5, 0, 0.1, 5, 4, 0, 5], 0.1, 0.9);
    const ids = [0, 7, 9, 9, 3];
    const near = labelsNear([t], [0, 0, 5], 0.5, (_tile, index) => ids[index] ?? 0);
    expect([...near.keys()].sort()).toEqual([7, 9]);
    expect(near.get(9)?.weight ?? 0).toBeGreaterThan(near.get(7)?.weight ?? 0);
    // What is hidden does not label anything.
    const shown = labelsNear(
      [t],
      [0, 0, 5],
      0.5,
      (_tile, index) => ids[index] ?? 0,
      (_tile, index) => ids[index] !== 9,
    );
    expect([...shown.keys()]).toEqual([7]);
  });
});

describe("painting", () => {
  /** Orthographic-ish: x, y in [-1, 1] straight to the screen, w = 10 - z (depth). */
  const viewProj = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 10].map((v, i) =>
    i === 0 || i === 5 ? v * 10 : v,
  );

  it("keeps the front-most splats of each cell, and paints those under the brush", () => {
    // Two splats at the screen's centre, one behind the other; one at the left edge.
    const t = tile([0, 0, 5, 0, 0, 0, -0.4, 0, 5], 0.1, 0.9);
    const screen = projectTiles([t], viewProj, 100, 100, 4);
    expect(screen.count).toBe(3);
    const visible = visibleSplats(screen);
    expect([...visible]).toEqual([1, 0, 1]);
    const mask = new BrushMask(screen.cols, screen.rows, 4);
    mask.stamp(50, 50, 6, 1);
    expect([...paintedSplats(screen, visible, mask)]).toEqual([1, 0, 0]);
    mask.line(0, 50, 100, 50, 6, 1);
    expect([...paintedSplats(screen, visible, mask)]).toEqual([1, 0, 1]);
    mask.stamp(50, 50, 6, 0);
    expect([...paintedSplats(screen, visible, mask)]).toEqual([0, 0, 1]);
  });

  it("chooses the instance, at any level, with the best intersection over union", () => {
    const d = doc();
    // Visible splats: crown 0,1, trunk 2,3, shrub 4,5.
    const ids = [3, 3, 4, 4, 2, 2];
    const weights = [1, 1, 1, 1, 1, 1];
    // The whole tree painted: the tree wins over its crown or trunk.
    expect(bestByIoU(d, { ids, weights, painted: [1, 1, 1, 1, 0, 0] })).toEqual({ id: 1, iou: 1 });
    // The crown only.
    expect(bestByIoU(d, { ids, weights, painted: [1, 1, 0, 0, 0, 0] })).toEqual({ id: 3, iou: 1 });
    // Half the shrub and the crown: no good match.
    const loose = bestByIoU(d, { ids, weights, painted: [1, 0, 0, 0, 1, 0] });
    expect(loose?.iou ?? 1).toBeLessThan(0.5);
    expect(bestByIoU(d, { ids: [0, 0], weights: [1, 1], painted: [1, 1] })).toBeNull();
  });

  it("matches through the view's index exactly as over every splat", () => {
    // A small fixed-seed generator (mulberry32): the same cases every run.
    let seed = 0x5eed;
    const random = (): number => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const pick = (n: number): number => Math.floor(random() * n);
    for (let round = 0; round < 40; round++) {
      // Up to four levels; a few ids the file does not list, and id 0.
      const count = 5 + pick(40);
      const box = { min: [0, 0, 0], max: [1, 1, 1] };
      const parsed = parseInstances({
        format: "hexapod.instances",
        version: 1,
        instances: Array.from({ length: count }, (_, k) => ({
          id: k + 1,
          parent: k > 2 && random() < 0.8 ? 1 + pick(k) : null,
          level: 0,
          splats: 1 + pick(50),
          bounds: box,
        })),
        tiles: {},
      });
      if (!parsed) throw new Error("fixture did not parse");
      const cols = 1 + pick(12);
      const rows = 1 + pick(9);
      const n = pick(900);
      const screen: ScreenSplats = {
        cols,
        rows,
        cellPx: 3,
        tile: new Uint32Array(n),
        index: new Uint32Array(n),
        cell: Int32Array.from({ length: n }, () => pick(cols * rows)),
        depth: new Float32Array(n),
        opacity: Float32Array.from({ length: n }, () => 0.05 + 0.95 * random()),
        count: n,
        front: new Float32Array(cols * rows),
      };
      const visible = Uint8Array.from({ length: n }, () => (random() < 0.7 ? 1 : 0));
      const ids = Uint32Array.from({ length: n }, () =>
        random() < 0.1 ? 0 : random() < 0.05 ? count + 1 + pick(3) : 1 + pick(count),
      );
      const index = paintIndex(parsed, screen, visible, ids);
      const weights = Float32Array.from({ length: n }, (_, k) =>
        visible[k] ? (screen.opacity[k] ?? 0) : 0,
      );
      for (let stroke = 0; stroke < 5; stroke++) {
        const mask = new BrushMask(cols, rows, 3);
        const share = random();
        for (let c = 0; c < mask.data.length; c++) mask.data[c] = random() < share ? 1 : 0;
        const painted = paintedSplats(screen, visible, mask);
        const label = `round ${String(round)}, stroke ${String(stroke)}`;
        expect(bestByIoUIndexed(index, mask), label).toEqual(
          bestByIoU(parsed, { ids, weights, painted }),
        );
        // And the best set: the same sums, the same members in the same order.
        expect(bestSetByIoUIndexed(index, mask), label).toEqual(
          bestSetByIoU(parsed, { ids, weights, painted }),
        );
        expect(paintedCount(index, mask), label).toBe(painted.reduce((a, b) => a + b, 0));
      }
    }
  });
});

describe("painting a combination", () => {
  /**
   * A cable spool (1): its bottom flange (2), top flange and drum (3), planks (4); and the
   * ground beside it (9). `own` splats of the spool itself are ground fused into it.
   */
  function spool() {
    const parsed = parseInstances({
      format: "hexapod.instances",
      version: 1,
      instances: [
        { id: 1, parent: null, level: 0, splats: 8, bounds: box, tags: tag("spool") },
        { id: 2, parent: 1, level: 1, splats: 10, bounds: box, tags: tag("bottom flange") },
        { id: 3, parent: 1, level: 1, splats: 12, bounds: box, tags: tag("top flange + drum") },
        { id: 4, parent: 1, level: 1, splats: 3, bounds: box, tags: tag("planks") },
        { id: 9, parent: null, level: 0, splats: 100, bounds: box, tags: tag("ground") },
      ],
      tiles: {},
    });
    if (!parsed) throw new Error("fixture did not parse");
    return parsed;
  }
  const box = { min: [0, 0, 0], max: [1, 1, 1] };
  function tag(label: string) {
    return [{ label, score: 0.9 }];
  }

  /** Visible splats of weight 1: per leaf id, how many are on screen and how many painted. */
  function sample(
    rows: readonly (readonly [number, number, number])[],
  ): PaintSample & { ids: number[]; weights: number[]; painted: number[] } {
    const ids: number[] = [];
    const painted: number[] = [];
    for (const [id, visible, brushed] of rows) {
      for (let k = 0; k < visible; k++) {
        ids.push(id);
        painted.push(k < brushed ? 1 : 0);
      }
    }
    return { ids, weights: ids.map(() => 1), painted };
  }

  it("selects the spool when its parts together are no better than it", () => {
    const d = spool();
    // Both flanges and the drum painted; the planks are behind, and the spool's own splat
    // (half a splat's weight of ground) costs it little.
    const s = sample([
      [2, 10, 10],
      [3, 12, 12],
      [9, 100, 0],
    ]);
    s.ids.push(1);
    s.weights.push(0.5);
    s.painted.push(0);
    const set = bestSetByIoU(d, s);
    expect(set?.ids).toEqual([1]);
    expect(set?.iou).toBeCloseTo(22 / 22.5);
    // Exactly, the parts are the best set; within `PAINT_PARENT_SLACK` the spool is chosen.
    expect(bestSet(paintSums(d, s), { gain: 0, slack: 0 })?.ids).toEqual([3, 2]);
    expect(1 - 22 / 22.5).toBeLessThan(PAINT_PARENT_SLACK);
  });

  it("selects both flanges and the drum when the spool carries ground nobody painted", () => {
    const d = spool();
    const s = sample([
      [1, 8, 0],
      [2, 10, 10],
      [3, 12, 12],
      [4, 3, 0],
      [9, 100, 0],
    ]);
    // One instance at a time, the spool is the best (and the flanges toggle under it): a
    // greedy search from there can never trade it for its parts.
    expect(bestByIoU(d, s)).toEqual({ id: 1, iou: 22 / 33 });
    // The combination is: the larger part first.
    expect(bestSetByIoU(d, s)).toEqual({ ids: [3, 2], iou: 1 });
    expect(commonChain(d, [3, 2])).toEqual([1]);
  });

  it("follows a stroke across the spool: one flange, then both, never toggling", () => {
    const d = spool();
    const stroke = [
      // The bottom flange first, then the drum and top flange as the brush moves up.
      [8, 0],
      [10, 3],
      [10, 7],
      [10, 12],
    ] as const;
    const shown: number[][] = [];
    let previous: readonly number[] | null = null;
    for (const [bottom, top] of stroke) {
      const sums = paintSums(
        d,
        sample([
          [1, 8, 0],
          [2, 10, bottom],
          [3, 12, top],
          [4, 3, 0],
          [9, 100, 0],
        ]),
      );
      const match = steadySet(sums, previous);
      previous = match?.ids ?? null;
      shown.push([...(match?.ids ?? [])]);
    }
    expect(shown[0]).toEqual([2]);
    expect(shown.at(-1)).toEqual([3, 2]);
    // Once both flanges are lit, they stay lit; the top flange alone is never shown.
    expect(shown.some((ids) => ids.length === 1 && ids[0] === 3)).toBe(false);
  });

  it("drops a sliver of a neighbour that adds little, and keeps what is half of the area", () => {
    const d = spool();
    // The spool painted whole, and one splat of the hundred-splat ground under the brush's edge.
    const s = sample([
      [1, 8, 8],
      [2, 10, 10],
      [3, 12, 12],
      [4, 3, 3],
      [9, 100, 1],
    ]);
    expect(bestSetByIoU(d, s)?.ids).toEqual([1]);
    // A tenth of each flange's area brushed: loose, but the two flanges are still what was
    // painted (1/10 together against 1/11 for one) -- a share of the IoU decides, not a margin.
    const loose = sample([
      [1, 8, 0],
      [2, 10, 1],
      [3, 10, 1],
      [4, 3, 0],
      [9, 100, 0],
    ]);
    expect(bestSetByIoU(d, loose)).toEqual({ ids: [2, 3], iou: 0.1 });
    // A small object painted whole beside it is part of what was painted.
    const rock = sample([
      [1, 8, 8],
      [2, 10, 10],
      [3, 12, 12],
      [4, 3, 3],
      [9, 100, 0],
      [7, 6, 6],
    ]);
    const withRock = parseInstances({
      format: "hexapod.instances",
      version: 1,
      instances: [
        ...d.instances.map((i) => ({ ...i, centroid: undefined })),
        { id: 7, parent: null, level: 0, splats: 6, bounds: box, tags: tag("rock") },
      ],
      tiles: {},
    });
    if (!withRock) throw new Error("fixture did not parse");
    expect(bestSetByIoU(withRock, rock)).toEqual({ ids: [1, 7], iou: 1 });
  });

  it("holds the shown match between near-equal answers, and moves on when one is better", () => {
    // A shrub (1) whose crown is 2 and whose stem is its own splat; a bed (3) beside it.
    const parsed = parseInstances({
      format: "hexapod.instances",
      version: 1,
      instances: [
        { id: 1, parent: null, level: 0, splats: 1, bounds: box },
        { id: 2, parent: 1, level: 1, splats: 40, bounds: box },
        { id: 3, parent: null, level: 0, splats: 20, bounds: box },
      ],
      tiles: {},
    });
    if (!parsed) throw new Error("fixture did not parse");
    const sums = (stem: number, crown: number) =>
      paintSums(
        parsed,
        sample([
          [1, 1, stem],
          [2, 40, crown],
          [3, 20, 0],
        ]),
      );
    // The stem painted too: the shrub (41/41) over its crown (40/41).
    expect(steadySet(sums(1, 40), null)?.ids).toEqual([1]);
    // Then not: the crown (40/40) is the best, but the shrub (40/41) is as good as makes no
    // difference, and stays lit.
    expect(bestSet(sums(0, 40))?.ids).toEqual([2]);
    expect(steadySet(sums(0, 40), [1])).toEqual({ ids: [1], iou: 40 / 41 });
    expect(steadySet(sums(0, 40), [1], 0)?.ids).toEqual([2]);
    // Shown nothing yet, or something far worse (the bed): the best.
    expect(steadySet(sums(0, 40), null)?.ids).toEqual([2]);
    expect(setIoU(sums(0, 40), [3])).toBe(0);
    expect(steadySet(sums(0, 40), [3])).toEqual({ ids: [2], iou: 1 });
  });

  /** Every antichain of the instances met, the best IoU of all: what `bestSet` must find. */
  function bruteForce(sums: PaintSums): number {
    const n = sums.ids.length;
    const above = (k: number): Set<number> => {
      const out = new Set<number>();
      for (let up = sums.parent[k] ?? -1; up >= 0; up = sums.parent[up] ?? -1) out.add(up);
      return out;
    };
    const ancestors = Array.from({ length: n }, (_, k) => above(k));
    let best = 0;
    for (let mask = 1; mask < 1 << n; mask++) {
      const members: number[] = [];
      for (let k = 0; k < n; k++) if (mask & (1 << k)) members.push(k);
      if (members.some((k) => members.some((m) => ancestors[k]?.has(m)))) continue;
      let i = 0;
      let v = 0;
      for (const k of members) {
        i += sums.inter[k] ?? 0;
        v += sums.visible[k] ?? 0;
      }
      const union = sums.painted + v - i;
      best = Math.max(best, union > 0 ? i / union : 0);
    }
    return best;
  }

  it("finds the best set exactly, disjoint, the same however the splats are listed", () => {
    let seed = 0xc0ffee;
    const random = (): number => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const pick = (n: number): number => Math.floor(random() * n);
    let combinations = 0;
    for (let round = 0; round < 150; round++) {
      const count = 3 + pick(10);
      const parsed = parseInstances({
        format: "hexapod.instances",
        version: 1,
        instances: Array.from({ length: count }, (_, k) => ({
          id: k + 1,
          parent: k > 0 && random() < 0.75 ? 1 + pick(k) : null,
          level: 0,
          splats: 1,
          bounds: box,
        })),
        tiles: {},
      });
      if (!parsed) throw new Error("fixture did not parse");
      const n = 20 + pick(200);
      const ids = Array.from({ length: n }, () => 1 + pick(count));
      // Opacities are float32: their sums are exact in doubles, however ordered.
      const weights = ids.map(() => Math.fround(0.05 + 0.95 * random()));
      const share = random();
      const painted = ids.map(() => (random() < share ? 1 : 0));
      const s = { ids, weights, painted };
      const sums = paintSums(parsed, s);
      const label = `round ${String(round)}`;
      if (sums.ids.length > 14) continue;
      const exact = bestSet(sums, { gain: 0, slack: 0 });
      const simple = bestSetByIoU(parsed, s);
      if (sums.ids.length === 0) {
        expect(simple, label).toBeNull();
        continue;
      }
      const optimum = bruteForce(sums);
      expect(exact?.iou ?? 0, label).toBeCloseTo(optimum, 9);
      // Simpler at a small cost, never worse than one instance by more than that.
      const single = bestByIoU(parsed, s)?.iou ?? 0;
      const cost = (1 - PAINT_SET_GAIN) * (1 - PAINT_PARENT_SLACK);
      expect(simple?.iou ?? 0, label).toBeGreaterThanOrEqual(optimum * cost - 1e-9);
      expect(simple?.iou ?? 0, label).toBeGreaterThanOrEqual(single * cost - 1e-9);
      for (const set of [exact, simple]) {
        const members = set?.ids ?? [];
        expect(new Set(members).size, label).toBe(members.length);
        // No member holds another.
        for (const id of members)
          for (const up of chainOf(parsed, id).slice(1)) expect(members, label).not.toContain(up);
        expect(set?.iou ?? 0, label).toBeCloseTo(setIoU(sums, members), 12);
      }
      if ((simple?.ids.length ?? 0) > 1) combinations++;
      // The same splats in another order: the same set.
      const order = [...ids.keys()].sort(() => random() - 0.5);
      const shuffled = {
        ids: order.map((k) => ids[k] ?? 0),
        weights: order.map((k) => weights[k] ?? 0),
        painted: order.map((k) => painted[k] ?? 0),
      };
      expect(bestSetByIoU(parsed, shuffled), label).toEqual(simple);
    }
    // The rounds hold combinations, not only single instances.
    expect(combinations).toBeGreaterThan(10);
  });

  it("names a combination by its members and what holds them", () => {
    expect(combinationLabel(["Top flange + drum", "Bottom flange"], "Spool")).toBe(
      "Top flange + drum + Bottom flange (of Spool)",
    );
    expect(combinationLabel(["Crown", "Trunk"], null)).toBe("Crown + Trunk");
    expect(combinationLabel(["Plank", "Plank", "Flange"], "Spool")).toBe("3 parts of Spool");
    expect(combinationLabel(["A", "B", "C", "D"], null)).toBe("4 objects");
    expect(
      combinationLabel(["A rather long name", "And another long name", "And a third"], "X"),
    ).toBe("3 parts of X");
    const d = spool();
    expect(commonChain(d, [2, 3])).toEqual([1]);
    expect(commonChain(d, [2, 9])).toEqual([]);
  });

  it("is selected as one: its members, then what holds them as the coarser candidate", () => {
    const store = useSceneSelect.getState();
    store.selectSet("scan", { ids: [3, 2], iou: 0.9 }, [1], null);
    let state = useSceneSelect.getState();
    expect(state.candidates).toEqual([3, 1]);
    expect(state.chain).toBe(2);
    expect(selectedIds(state)).toEqual([3, 2]);
    expect(selectedId(state)).toBe(3);
    // Up to the spool and back to the combination.
    state.cycle(1);
    expect(selectedIds(useSceneSelect.getState())).toEqual([1]);
    useSceneSelect.getState().cycle(-1);
    expect(selectedIds(useSceneSelect.getState())).toEqual([3, 2]);
    // A click's selection is one instance again.
    useSceneSelect.getState().select("scan", [2, 1], 2, 1, null);
    state = useSceneSelect.getState();
    expect(state.combination).toBeNull();
    expect(selectedIds(state)).toEqual([1]);
    useSceneSelect.getState().clear();
    expect(selectedIds(useSceneSelect.getState())).toEqual([]);
  });

  it("matches a camp-sized view within a preview's budget", () => {
    // 1440 × 900 CSS px in 3 px cells; 4,000 objects of 5 parts, three of them in 2 pieces
    // (48,000 instances); 2.2 M visible splats, each object's in its own block of cells.
    const cols = 480;
    const rows = 300;
    const blocksX = 80;
    const blocksY = 50;
    const blockW = cols / blocksX;
    const blockH = rows / blocksY;
    let seed = 0xca4b;
    const random = (): number => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const instances: { id: number; parent: number | null; level: number }[] = [];
    const leavesOf: number[][] = [];
    let next = 1;
    for (let o = 0; o < blocksX * blocksY; o++) {
      const root = next++;
      instances.push({ id: root, parent: null, level: 0 });
      const leaves = [root];
      for (let p = 0; p < 5; p++) {
        const part = next++;
        instances.push({ id: part, parent: root, level: 1 });
        if (p % 2 === 0) {
          for (let q = 0; q < 2; q++) {
            const piece = next++;
            instances.push({ id: piece, parent: part, level: 2 });
            leaves.push(piece);
          }
        } else leaves.push(part);
      }
      leavesOf.push(leaves);
    }
    const parsed = parseInstances({
      format: "hexapod.instances",
      version: 1,
      instances: instances.map((i) => ({ ...i, splats: 1, bounds: box })),
      tiles: {},
    });
    if (!parsed) throw new Error("fixture did not parse");
    const perCell = 15;
    const count = cols * rows * perCell;
    const cell = new Int32Array(count);
    const opacity = new Float32Array(count);
    const ids = new Uint32Array(count);
    for (let c = 0, k = 0; c < cols * rows; c++) {
      const x = c % cols;
      const y = Math.floor(c / cols);
      const leaves = leavesOf[Math.floor(y / blockH) * blocksX + Math.floor(x / blockW)] ?? [];
      // A part a cell, mostly: its splats in a run, as a tile's are.
      const main = leaves[Math.floor(((x % blockW) * blockH + (y % blockH)) / 4) % leaves.length];
      for (let s = 0; s < perCell; s++, k++) {
        cell[k] = c;
        opacity[k] = 0.05 + 0.95 * random();
        ids[k] = random() < 0.8 ? (main ?? 0) : (leaves[Math.floor(random() * leaves.length)] ?? 0);
      }
    }
    const visible = new Uint8Array(count).fill(1);
    const built = performance.now();
    const index = paintIndex(parsed, { cols, rows, cell, opacity, count }, visible, ids);
    const indexMs = performance.now() - built;
    const strokes: { name: string; mask: BrushMask }[] = [];
    // A stroke of the default brush (18 px) across a third of the screen.
    const typical = new BrushMask(cols, rows, 3);
    typical.line(300, 420, 780, 470, 18, 1);
    strokes.push({ name: "18 px stroke", mask: typical });
    // The largest brush (120 px) scrubbed across half the screen.
    const large = new BrushMask(cols, rows, 3);
    for (let y = 200; y <= 700; y += 120) large.line(100, y, 820, y, 120, 1);
    strokes.push({ name: "120 px scrub", mask: large });
    const rowsOut: string[] = [];
    const median = (list: number[]): number =>
      [...list].sort((a, b) => a - b)[Math.floor(list.length / 2)] ?? 0;
    for (const { name, mask } of strokes) {
      const single: number[] = [];
      const set: number[] = [];
      let match: PaintSetMatch | null = null;
      let met = 0;
      // As a stroke's previews run, many times over: the first few warm up.
      for (let run = 0; run < 25; run++) {
        let t = performance.now();
        bestByIoUIndexed(index, mask);
        const one = performance.now() - t;
        t = performance.now();
        const sums = paintSumsIndexed(index, mask);
        match = steadySet(sums, match?.ids ?? null);
        const many = performance.now() - t;
        met = sums.ids.length;
        if (run < 5) continue;
        single.push(one);
        set.push(many);
      }
      const cells = mask.data.reduce((a, b) => a + b, 0);
      rowsOut.push(
        `${name}: ${String(cells)} cells, ${String(met)} instances met, one instance ` +
          `${median(single).toFixed(2)} ms, best set ${median(set).toFixed(2)} ms ` +
          `(${String(match?.ids.length ?? 0)} members, IoU ${(match?.iou ?? 0).toFixed(3)})`,
      );
      // Well within the preview's 100 ms, on a shared machine; the report says what it took.
      expect(median(set)).toBeLessThan(60);
      expect(match).not.toBeNull();
    }
    console.info(
      `camp-sized paint match: ${String(count)} visible splats, ${String(parsed.instances.length)} ` +
        `instances, index ${indexMs.toFixed(0)} ms\n  ${rowsOut.join("\n  ")}`,
    );
  });
});

describe("painted objects", () => {
  const set: CustomSet = {
    key: "a",
    name: "Bench",
    tiles: { [CHECKSUM]: encodeRanges([5, 1, 0, 1]) },
    splats: 3,
    bounds: { min: [0, 0, 0], max: [1, 1, 1] },
    created: 1,
  };

  it("encodes splat indices as ranges and back", () => {
    expect(encodeRanges([7, 3, 4, 5, 9, 4])).toEqual([3, 3, 7, 1, 9, 1]);
    expect(decodeRanges([3, 3, 7, 1, 9, 1])).toEqual([3, 4, 5, 7, 9]);
    expect(rangesLength([3, 3, 7, 1])).toBe(4);
    expect(encodeRanges([])).toEqual([]);
  });

  it("round-trips through storage and drops what is malformed", () => {
    const text = serializeCustomSets([set]);
    expect(parseCustomSets(text)).toEqual([set]);
    expect(parseCustomSets("not json")).toEqual([]);
    expect(parseCustomSets(JSON.stringify({ format: "other", version: 1, sets: [set] }))).toEqual(
      [],
    );
    const bad = JSON.parse(text) as { sets: Record<string, unknown>[] };
    bad.sets.push({ ...set, key: "b", tiles: { [CHECKSUM]: [3, 1, 2, 1] } });
    bad.sets.push({ ...set, key: "c", bounds: { min: [0, 0], max: [1, 1, 1] } });
    expect(parseCustomSets(JSON.stringify(bad)).map((s) => s.key)).toEqual(["a"]);
  });

  it("keeps a combination as the splats its instances carry in every tile", () => {
    const d = doc();
    // The trunk (4) and the shrub (2): splats 2, 3 and 4, 5 of the tile, one range.
    const kept = setFromInstances(d, [4, 2], "Trunk + Shrub", "k", 7);
    expect(kept).toEqual({
      key: "k",
      name: "Trunk + Shrub",
      tiles: { [CHECKSUM]: [2, 4] },
      splats: 4,
      bounds: { min: [0.8, 0, 0], max: [6, 1.2, 3] },
      created: 7,
    });
    // The tree with what is below it; and nothing for an instance without splats.
    expect(setFromInstances(d, [1], "Tree", "t", 0)?.tiles).toEqual({ [CHECKSUM]: [0, 4] });
    expect(setFromInstances(d, [99], "None", "n", 0)).toBeNull();
    // Kept and read back as any painted object.
    if (kept) expect(parseCustomSets(serializeCustomSets([kept]))).toEqual([kept]);
  });

  it("draws a set as an instance of its own past the file's ids", () => {
    const d = doc();
    const merged = withCustomSets(d, [set]);
    const id = customId(d, 0);
    expect(id).toBe(5);
    expect(isCustomId(d, id)).toBe(true);
    expect(isCustomId(d, 4)).toBe(false);
    expect(merged.maxId).toBe(5);
    expect(merged.byId.get(5)?.tags[0]?.label).toBe("Bench");
    expect([...(tileInstanceIds(merged, CHECKSUM) ?? [])]).toEqual([5, 5, 4, 4, 2, 5]);
    // The file's document is untouched, and the same sets give the same document.
    expect([...(tileInstanceIds(d, CHECKSUM) ?? [])]).toEqual([3, 3, 4, 4, 2, 2]);
    const sets = [set];
    expect(withCustomSets(d, sets)).toBe(withCustomSets(d, sets));
    expect(withCustomSets(d, [])).toBe(d);
  });
});
