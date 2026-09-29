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
  planAdditive,
  type LoadStep,
  type Sphere,
  type TileNode,
  type TileTree,
} from "@/view/tiles";
import { TileStreamer, chooseCut, type StreamHost, type View } from "@/view/stream";

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

/** A view from `eye`, with every tile in view unless `hidden` says otherwise. */
function viewFrom(
  eye: [number, number, number],
  projection = 1000,
  hidden: (bounds: Sphere) => boolean = () => false,
): View {
  return { eye, projection, visible: (bounds) => !hidden(bounds) };
}

const cutOf = (tree: TileTree, view: View, budget: number): string[] =>
  [...chooseCut(tree, view, budget).tiles].map((chosen) => chosen.uri).sort();

/** Two regions 100 m apart, each a merged parent over two leaves. */
function at(x: number, radius: number): Sphere {
  return { center: [x, 0, 0], radius };
}
const site: TileTree = {
  refine: "REPLACE",
  root: {
    ...tile("root.glb", 10_000, 4, [
      {
        ...tile("west.glb", 20_000, 1, [
          { ...tile("west-a.glb", 100_000, 0), bounds: at(-60, 10) },
          { ...tile("west-b.glb", 100_000, 0), bounds: at(-40, 10) },
        ]),
        bounds: at(-50, 25),
      },
      {
        ...tile("east.glb", 20_000, 1, [
          { ...tile("east-a.glb", 100_000, 0), bounds: at(40, 10) },
          { ...tile("east-b.glb", 100_000, 0), bounds: at(60, 10) },
        ]),
        bounds: at(50, 25),
      },
    ]),
    bounds: at(0, 80),
  },
};

describe("the cut the scan viewer draws (REPLACE, merged parents)", () => {
  it("is the root alone from far away, and every leaf close up with room for them", () => {
    expect(cutOf(site, viewFrom([0, 0, 100_000]), 1e9)).toEqual(["root.glb"]);
    expect(cutOf(site, viewFrom([0, 0, 0]), 1e9)).toEqual(
      ["east-a.glb", "east-b.glb", "west-a.glb", "west-b.glb"].sort(),
    );
    // Every leaf of the octree fixture too: 780k.
    const all = chooseCut(replaceScan, viewFrom([0, 0, 0]), DEFAULT_SPLAT_BUDGET * LOAD_FACTOR);
    expect(all.gaussians).toBe(countTiles(replaceScan).gaussians);
  });

  it("spends the budget where the camera is: the near region fine, the far one merged", () => {
    // Beside the west region, with room for one region's leaves.
    expect(cutOf(site, viewFrom([-50, 0, 30]), 250_000)).toEqual(
      ["east.glb", "west-a.glb", "west-b.glb"].sort(),
    );
    expect(cutOf(site, viewFrom([50, 0, 30]), 250_000)).toEqual(
      ["east-a.glb", "east-b.glb", "west.glb"].sort(),
    );
  });

  it("keeps what is out of view in the cut, coarse, and refines what is in view first", () => {
    // Midway, the east region behind the camera: the west one is refined, the east one stays
    // merged rather than being dropped, so turning round shows it at once.
    const lookingWest = viewFrom([0, 0, 30], 1000, (bounds) => bounds.center[0] > 0);
    expect(cutOf(site, lookingWest, 250_000)).toEqual(
      ["east.glb", "west-a.glb", "west-b.glb"].sort(),
    );
  });

  it("is always a cut with no holes, and within the budget once past the root", () => {
    for (const budget of [5_000, 50_000, 150_000, 300_000, 420_000, 1_000_000]) {
      for (const eye of [
        [0, 0, 0],
        [0, 0, 5],
        [30, -20, 2],
      ] as [number, number, number][]) {
        const { tiles, gaussians } = chooseCut(replaceScan, viewFrom(eye), budget);
        expect(isCut(replaceScan.root, new Set([...tiles].map((chosen) => chosen.uri)))).toBe(true);
        expect(gaussians).toBeLessThanOrEqual(Math.max(budget, 40_000));
      }
    }
  });

  it("never swaps in a tile with no count", () => {
    const unknown: TileTree = {
      refine: "REPLACE",
      root: tile("splat.glb", 1_000, 1, [tile("a.glb", Number.POSITIVE_INFINITY, 0)]),
    };
    expect(cutOf(unknown, viewFrom([0, 0, 0]), 1e9)).toEqual(["splat.glb"]);
  });
});

/** A host whose fetches resolve when the test says, recording what is on screen. */
function fakeHost() {
  const pending = new Map<string, () => void>();
  const onScreen = new Set<string>();
  const disposed: string[] = [];
  const host: StreamHost<string> = {
    load: (chosen) =>
      new Promise<string>((done) => pending.set(chosen.uri, () => done(chosen.uri))),
    show: (chosen) => {
      onScreen.add(chosen.uri);
    },
    hide: (chosen) => {
      onScreen.delete(chosen.uri);
    },
    dispose: (mesh) => disposed.push(mesh),
  };
  /** Lets every fetch asked for so far arrive, in the order given (or all). */
  const arrive = async (...uris: string[]): Promise<void> => {
    for (const uri of uris.length ? uris : [...pending.keys()]) {
      pending.get(uri)?.();
      pending.delete(uri);
    }
    await Promise.resolve();
    await Promise.resolve();
  };
  return { host, pending, onScreen, disposed, arrive };
}

describe("streaming towards the cut", () => {
  it("swaps a parent out only once all its children are here: never a hole", async () => {
    const { host, pending, onScreen, arrive } = fakeHost();
    const streamer = new TileStreamer(site, host, {
      budget: 1e9,
      cacheBudget: 1e9,
      concurrency: 8,
    });
    streamer.adopt(site.root, "root.glb");
    const close = viewFrom([-50, 0, 5]);
    streamer.update(close);
    expect([...pending.keys()].sort()).toEqual(["east.glb", "west.glb"]);
    await arrive("west.glb");
    streamer.update(close);
    // One child is not a cut: the root stays until both are here.
    expect([...onScreen]).toEqual([]);
    expect(streamer.drawn.map((drawn) => drawn.uri)).toEqual(["root.glb"]);
    await arrive("east.glb");
    streamer.update(close);
    expect(streamer.drawn.map((drawn) => drawn.uri).sort()).toEqual(["east.glb", "west.glb"]);
    // The nearest region's leaves are asked for first.
    expect([...pending.keys()].slice(0, 2).sort()).toEqual(["west-a.glb", "west-b.glb"]);
    await arrive();
    streamer.update(close);
    expect(isCut(site.root, new Set(streamer.drawn.map((drawn) => drawn.uri)))).toBe(true);
    expect(streamer.drawn.map((drawn) => drawn.uri).sort()).toEqual(
      ["east-a.glb", "east-b.glb", "west-a.glb", "west-b.glb"].sort(),
    );
  });

  it("keeps the ancestors of what is drawn, so leaving goes back to one at once", async () => {
    const { host, onScreen, disposed, arrive } = fakeHost();
    const streamer = new TileStreamer(site, host, {
      budget: 1e9,
      cacheBudget: 60_000,
      concurrency: 8,
    });
    streamer.adopt(site.root, "root.glb");
    const close = viewFrom([-50, 0, 5]);
    for (let round = 0; round < 3; round++) {
      streamer.update(close);
      await arrive();
    }
    streamer.update(close);
    expect(streamer.drawn).toHaveLength(4);
    // The cache (60k) is far smaller than the 450k loaded, but only the leaves' own spare
    // copies could go, and they are all drawn: the merged parents above them stay.
    expect(disposed).toEqual([]);
    // Far away the root is enough; it is still loaded, so there is no wait, and the leaves
    // it replaces are disposed to fit the cache.
    streamer.update(viewFrom([0, 0, 100_000]));
    expect(streamer.drawn.map((drawn) => drawn.uri)).toEqual(["root.glb"]);
    expect([...onScreen]).toEqual(["root.glb"]);
    expect(disposed.sort()).toEqual(
      ["east-a.glb", "east-b.glb", "west-a.glb", "west-b.glb"].sort(),
    );
  });

  it("fetches no more than its concurrency at once, and does not retry a broken tile", async () => {
    const failing: StreamHost<string> = {
      load: (chosen) =>
        chosen.uri === "west.glb" ? Promise.reject(new Error("404")) : Promise.resolve(chosen.uri),
      show: () => undefined,
      hide: () => undefined,
      dispose: () => undefined,
    };
    let loads = 0;
    const counting: StreamHost<string> = {
      ...failing,
      load: (chosen) => {
        loads += 1;
        return failing.load(chosen);
      },
    };
    const streamer = new TileStreamer(site, counting, {
      budget: 1e9,
      cacheBudget: 1e9,
      concurrency: 1,
    });
    streamer.adopt(site.root, "root.glb");
    streamer.update(viewFrom([-50, 0, 5]));
    expect(loads).toBe(1);
    for (let round = 0; round < 4; round++) {
      await Promise.resolve();
      await Promise.resolve();
      streamer.update(viewFrom([-50, 0, 5]));
    }
    // west.glb failed once and was not asked for again; the root covers it still.
    expect(loads).toBe(2);
    expect(streamer.drawn.map((drawn) => drawn.uri)).toEqual(["root.glb"]);
  });
});

describe("what the scan viewer downloads (ADD, before merged parents)", () => {
  it("only ever adds, root first, coarsest region first, within the budget", () => {
    const steps = planAdditive(addScan, 400_000);
    expect(steps.every((step) => step.remove.length === 0 && step.add.length === 1)).toBe(true);
    expect(loaded(steps)).toEqual(["splat.glb", "splat_1.glb", "splat_3.glb", "splat_02.glb"]);
    expect(countTiles(addScan).gaussians).toBe(780_000);
    expect(loaded(planAdditive(addScan, 200_000))).toEqual(["splat.glb", "splat_1.glb"]);
    // 290k: splat_1 and splat_3 do not both fit, but the 80k packed leaf does.
    expect(loaded(planAdditive(addScan, 290_000))).toEqual([
      "splat.glb",
      "splat_1.glb",
      "splat_02.glb",
    ]);
    const all = loaded(planAdditive(addScan, 1_000_000));
    expect(all).toHaveLength(countTiles(addScan).tiles);
    expect(all.indexOf("splat_3.glb")).toBeLessThan(all.indexOf("splat_3-5.glb"));
  });
});

describe("reading a tileset", () => {
  it("reads each tile's bounds as a sphere: boxes through their corners", () => {
    const tree = parseTileset({
      root: {
        content: { uri: "splat.glb" },
        geometricError: 1,
        boundingVolume: { box: [1, 2, 3, 3, 0, 0, 0, 4, 0, 0, 0, 12] },
        children: [
          {
            content: { uri: "a.glb" },
            geometricError: 0,
            boundingVolume: { sphere: [0, 0, 0, 2] },
          },
          {
            content: { uri: "b.glb" },
            geometricError: 0,
            boundingVolume: { region: [0, 0, 1, 1, 0, 1] },
          },
        ],
      },
    });
    expect(tree.root.bounds).toEqual({ center: [1, 2, 3], radius: 13 });
    expect(tree.root.children[0]?.bounds).toEqual({ center: [0, 0, 0], radius: 2 });
    expect(tree.root.children[1]?.bounds).toBeNull();
  });

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
    expect(loaded(planAdditive(old, 200_000))).toEqual(["splat.glb"]);
    expect(() => parseTileset({ root: { geometricError: 1 } })).toThrow(/names no content/);
    expect(() => parseTileset({})).toThrow(/no root/);
  });

  it("reads the committed tree's tileset as one REPLACE tile of 12,000", () => {
    const path = resolve(process.cwd(), "../../data/tiles/synthetic-tree/splat/tileset.json");
    const tree = parseTileset(JSON.parse(readFileSync(path, "utf8")) as unknown);
    expect(tree.refine).toBe("REPLACE");
    expect(tree.root).toMatchObject({ uri: "splat.glb", gaussians: 12_000, geometricError: 0 });
    expect(cutOf(tree, viewFrom([0, 0, 0]), DEFAULT_SPLAT_BUDGET)).toEqual(["splat.glb"]);
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
