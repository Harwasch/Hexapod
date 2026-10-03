import { describe, expect, it } from "vitest";

import {
  customId,
  decodeRanges,
  encodeRanges,
  isCustomId,
  parseCustomSets,
  rangesLength,
  serializeCustomSets,
  withCustomSets,
  type CustomSet,
} from "@/lib/customSets";
import { parseInstances, tileInstanceIds, withDescendants } from "@/lib/instances";
import {
  bestByIoU,
  buildCandidates,
  chainOf,
  chipText,
  cycleIndex,
  defaultIndex,
  hiddenForShowOnly,
  selectionLabel,
} from "@/lib/sceneSelect";
import { BrushMask, paintedSplats, projectTiles, visibleSplats } from "@/lib/splatPaint";
import { buildTileIndex, castRay, hitWeights, labelsNear, type PickTile } from "@/lib/splatPick";

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

  it("chooses the smallest of the chain that is big enough on screen", () => {
    const candidates = { ids: [3, 1, 4], chain: 2 };
    expect(defaultIndex(candidates, (id) => (id === 3 ? 60 : 200))).toBe(0);
    expect(defaultIndex(candidates, (id) => (id === 3 ? 10 : 200))).toBe(1);
    // Nothing big enough: the top of the chain.
    expect(defaultIndex(candidates, () => 5)).toBe(1);
    expect(defaultIndex({ ids: [], chain: 0 }, () => 100)).toBe(-1);
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
    expect(chipText("Tree", 1, 4)).toBe("Tree · 2 of 4");
    expect(chipText("Tree", 0, 1)).toBe("Tree");
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
