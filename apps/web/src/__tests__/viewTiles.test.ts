import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  DEFAULT_SPLAT_BUDGET,
  OPTIONS_STORAGE,
  detailScreenSpaceScale,
  splatBudget,
} from "@/lib/detail";
import {
  LOAD_FACTOR,
  countTiles,
  parseTileset,
  planLoads,
  plannedGaussians,
  type LoadStep,
  type TileNode,
  type TileTree,
} from "@/view/tiles";

function tile(
  uri: string,
  gaussians: number,
  geometricError: number,
  children: TileNode[] = [],
): TileNode {
  return { uri, gaussians, geometricError, children };
}

/**
 * The shape splat_tiles.py writes now: merged parents (an eighth or less of what is under
 * them), leaves (error 0) holding every gaussian once: 780k in the leaves.
 */
const replaceScan: TileTree = {
  refine: "REPLACE",
  root: tile("splat.glb", 40_000, 0.8, [
    tile("splat_1.glb", 20_000, 0.2, [
      tile("splat_1-7.glb", 90_000, 0),
      tile("splat_1-0123.glb", 80_000, 0),
    ]),
    tile("splat_3.glb", 30_000, 0.4, [
      tile("splat_3-5.glb", 20_000, 0.1, [
        tile("splat_3-5-0.glb", 100_000, 0),
        tile("splat_3-5-1.glb", 60_000, 0),
      ]),
      tile("splat_3-012.glb", 70_000, 0),
    ]),
    tile("splat_02.glb", 80_000, 0),
    tile("splat_4.glb", 25_000, 0.3, [
      tile("splat_4-0.glb", 100_000, 0),
      tile("splat_4-1.glb", 100_000, 0),
      tile("splat_4-2.glb", 100_000, 0),
    ]),
  ]),
};

/** What tilesets packed before merged parents look like: ADD, parents a thinned subset. */
const addScan: TileTree = {
  refine: "ADD",
  root: tile("splat.glb", 100_000, 0.4, [
    tile("splat_1.glb", 100_000, 0.1, [
      tile("splat_1-7.glb", 90_000, 0),
      tile("splat_1-0123.glb", 80_000, 0),
    ]),
    tile("splat_3.glb", 100_000, 0.2, [
      tile("splat_3-5.glb", 100_000, 0.05, [tile("splat_3-5-0.glb", 60_000, 0)]),
      tile("splat_3-012.glb", 70_000, 0),
    ]),
    tile("splat_02.glb", 80_000, 0),
  ]),
};

const uris = (tiles: TileNode[]): string[] => tiles.map((chosen) => chosen.uri);

/** What is loaded once the steps have run, in the order it arrived. */
function loaded(steps: LoadStep[]): string[] {
  const shown: TileNode[] = [];
  for (const step of steps) {
    shown.push(...step.add);
    for (const gone of step.remove) shown.splice(shown.indexOf(gone), 1);
  }
  return uris(shown);
}

/** Every root-to-leaf path meets the loaded set exactly once: a hole-free, double-free cut. */
function isCut(root: TileNode, shown: Set<string>): boolean {
  const visit = (node: TileNode, covered: boolean): boolean => {
    const here = shown.has(node.uri);
    if (here && covered) return false;
    if (node.children.length === 0) return covered || here;
    return node.children.every((child) => visit(child, covered || here));
  };
  return visit(root, false);
}

describe("what the scan viewer downloads (REPLACE, merged parents)", () => {
  it("downloads every leaf when the whole scan fits, and nothing else is left loaded", () => {
    const steps = planLoads(replaceScan, 1_000_000);
    expect(countTiles(replaceScan).gaussians).toBe(780_000);
    expect(loaded(steps).sort()).toEqual(
      [
        "splat_1-7.glb",
        "splat_1-0123.glb",
        "splat_3-5-0.glb",
        "splat_3-5-1.glb",
        "splat_3-012.glb",
        "splat_02.glb",
        "splat_4-0.glb",
        "splat_4-1.glb",
        "splat_4-2.glb",
      ].sort(),
    );
    expect(plannedGaussians(steps)).toBe(780_000);
  });

  it("shows the root first, then swaps a tile for all its children at once", () => {
    const steps = planLoads(replaceScan, 1_000_000);
    expect(steps[0]).toEqual({ add: [replaceScan.root], remove: [] });
    for (const step of steps.slice(1)) {
      expect(step.remove).toHaveLength(1);
      expect(step.add).toEqual(step.remove[0]?.children);
    }
    // Largest error first: the root (0.8), then splat_3 (0.4), splat_4 (0.3), splat_1 (0.2).
    expect(steps.slice(1).map((step) => step.remove[0]?.uri)).toEqual([
      "splat.glb",
      "splat_3.glb",
      "splat_4.glb",
      "splat_1.glb",
      "splat_3-5.glb",
    ]);
  });

  it("stops at the finest cut that fits, and every step is a cut with no holes", () => {
    for (const budget of [50_000, 150_000, 300_000, 420_000, 600_000, 1_000_000]) {
      const steps = planLoads(replaceScan, budget);
      const shown = new Set<string>();
      for (const step of steps) {
        step.add.forEach((added) => shown.add(added.uri));
        step.remove.forEach((gone) => shown.delete(gone.uri));
        expect(isCut(replaceScan.root, shown)).toBe(true);
      }
      expect(plannedGaussians(steps)).toBeLessThanOrEqual(Math.max(budget, 40_000));
    }
    // 420k: the root's children (155k), then splat_3's (215k). splat_4's three leaves would
    // make it 490k, so that swap is skipped -- and splat_1's smaller one still fits (365k).
    const steps = planLoads(replaceScan, 420_000);
    expect(loaded(steps).sort()).toEqual(
      [
        "splat_1-7.glb",
        "splat_1-0123.glb",
        "splat_3-5.glb",
        "splat_3-012.glb",
        "splat_02.glb",
        "splat_4.glb",
      ].sort(),
    );
    expect(plannedGaussians(steps)).toBe(365_000);
  });

  it("loads the root alone when not even one swap fits, and never a tile with no count", () => {
    expect(loaded(planLoads(replaceScan, 10_000))).toEqual(["splat.glb"]);
    const unknown: TileTree = {
      refine: "REPLACE",
      root: tile("splat.glb", 1_000, 1, [tile("a.glb", Number.POSITIVE_INFINITY, 0)]),
    };
    expect(loaded(planLoads(unknown, 1e9))).toEqual(["splat.glb"]);
  });

  it("downloads LOAD_FACTOR times what Spark draws", () => {
    expect(LOAD_FACTOR).toBeGreaterThan(1);
    // Standard, on this scan: the budget to download is 1.6M, so all of it.
    expect(plannedGaussians(planLoads(replaceScan, DEFAULT_SPLAT_BUDGET * LOAD_FACTOR))).toBe(
      countTiles(replaceScan).gaussians,
    );
  });
});

describe("what the scan viewer downloads (ADD, before merged parents)", () => {
  it("only ever adds, root first, coarsest region first, within the budget", () => {
    const steps = planLoads(addScan, 400_000);
    expect(steps.every((step) => step.remove.length === 0 && step.add.length === 1)).toBe(true);
    expect(loaded(steps)).toEqual(["splat.glb", "splat_1.glb", "splat_3.glb", "splat_02.glb"]);
    expect(countTiles(addScan).gaussians).toBe(780_000);
    expect(loaded(planLoads(addScan, 200_000))).toEqual(["splat.glb", "splat_1.glb"]);
    // 290k: splat_1 and splat_3 do not both fit, but the 80k packed leaf does.
    expect(loaded(planLoads(addScan, 290_000))).toEqual([
      "splat.glb",
      "splat_1.glb",
      "splat_02.glb",
    ]);
    const all = loaded(planLoads(addScan, 1_000_000));
    expect(all).toHaveLength(countTiles(addScan).tiles);
    expect(all.indexOf("splat_3.glb")).toBeLessThan(all.indexOf("splat_3-5.glb"));
  });
});

describe("reading a tileset", () => {
  it("reads refinement from the root, REPLACE when it does not say", () => {
    const tree = (refine?: string): TileTree =>
      parseTileset({ root: { refine, content: { uri: "splat.glb" }, geometricError: 0.5 } });
    expect(tree("ADD").refine).toBe("ADD");
    expect(tree("REPLACE").refine).toBe("REPLACE");
    expect(tree().refine).toBe("REPLACE");
  });

  it("always loads the root, even a pre-hierarchy single tile with no count", () => {
    const old = parseTileset({
      root: { refine: "ADD", content: { uri: "splat.glb" }, geometricError: 0.5 },
    });
    expect(loaded(planLoads(old, 200_000))).toEqual(["splat.glb"]);
    expect(() => parseTileset({ root: { geometricError: 1 } })).toThrow(/names no content/);
    expect(() => parseTileset({})).toThrow(/no root/);
  });

  it("reads the committed tree's tileset as one REPLACE tile of 12,000", () => {
    const path = resolve(process.cwd(), "../../data/tiles/synthetic-tree/splat/tileset.json");
    const tree = parseTileset(JSON.parse(readFileSync(path, "utf8")) as unknown);
    expect(tree.refine).toBe("REPLACE");
    expect(tree.root).toMatchObject({ uri: "splat.glb", gaussians: 12_000, geometricError: 0 });
    expect(planLoads(tree, DEFAULT_SPLAT_BUDGET)).toEqual([{ add: [tree.root], remove: [] }]);
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
