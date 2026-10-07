import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

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
import {
  RETRY_FAILED_MS,
  TileStreamer,
  centreWeight,
  chooseCut,
  type StreamHost,
  type View,
} from "@/view/stream";

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

  it("measures a tile's distance to its box, so a tile behind the camera does not win", () => {
    // Standing between two tiles: inside the sphere around the one behind (its box ends 8 m
    // away), outside the one ahead (10 m). Room for one refinement. By the spheres, the tile behind
    // was "touching the lens" and took the budget; by the boxes the one in view does.
    const boxed = (uri: string, centre: number, half: [number, number, number]): TileNode => {
      const box = [centre, 0, 0, half[0], 0, 0, 0, half[1], 0, 0, 0, half[2]];
      return {
        uri,
        gaussians: 10_000,
        geometricError: 0.5,
        bounds: { center: [centre, 0, 0], radius: Math.hypot(...half) },
        box,
        children: [tile(`${uri}-a`, 50_000, 0), tile(`${uri}-b`, 50_000, 0)],
      };
    };
    const behind = boxed("behind", -11.5, [3.5, 8, 8]);
    const ahead = boxed("ahead", 12, [2, 2, 2]);
    const tree: TileTree = {
      refine: "REPLACE",
      root: { ...tile("root", 5_000, 2, [behind, ahead]), bounds: null },
    };
    const lookingAhead = viewFrom([0, 0, 0], 1000, (bounds) => bounds.center[0] < 0);
    expect(cutOf(tree, lookingAhead, 115_000)).toEqual(["ahead-a", "ahead-b", "behind"]);
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

  it("refines what is in the middle of the view before the same at its edge", () => {
    // Two equal regions, the same distance from the camera: one straight ahead, one off to
    // the side near the frame's corner. Room for one refinement.
    const region = (uri: string, x: number, y: number): TileNode => ({
      ...tile(uri, 10_000, 1, [tile(`${uri}-a`, 50_000, 0), tile(`${uri}-b`, 50_000, 0)]),
      bounds: { center: [x, y, 0], radius: 2 },
    });
    const ahead = region("ahead", 0, 40);
    const aside = region("aside", 40 * Math.sin(0.55), 40 * Math.cos(0.55));
    const tree: TileTree = {
      refine: "REPLACE",
      root: { ...tile("root", 1_000, 4, [aside, ahead]), bounds: null },
    };
    const view: View = {
      ...viewFrom([0, 0, 0]),
      centre: { forward: [0, 1, 0], halfDiagonal: 0.6 },
    };
    expect(cutOf(tree, view, 115_000)).toEqual(["ahead-a", "ahead-b", "aside"]);
    expect(centreWeight({ center: [0, 40, 0], radius: 2 }, view)).toBe(1);
    expect(
      centreWeight({ center: aside.bounds?.center ?? [0, 0, 0], radius: 2 }, view),
    ).toBeLessThan(1);
    // Without a middle every tile in view counts alike.
    expect(centreWeight({ center: [40, 0, 0], radius: 2 }, viewFrom([0, 0, 0]))).toBe(1);
    // And the cut says the budget held it back.
    expect(chooseCut(tree, view, 115_000).limited).toBe(true);
    expect(chooseCut(tree, view, 1e9).limited).toBe(false);
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

  it("turning round in place re-fetches nothing near: the cache keeps what surrounds you", async () => {
    // Eight regions in a ring round the camera, near (6 m) and far (25 m) in turn, each a
    // merged parent over two leaves; room to refine about half of them at once.
    const region = (index: number): TileNode => {
      const angle = (index * Math.PI) / 4;
      const range = index % 2 === 0 ? 6 : 25;
      const centre: [number, number, number] = [
        Math.cos(angle) * range,
        Math.sin(angle) * range,
        0,
      ];
      const leaf = (name: string): TileNode => ({
        ...tile(`r${String(index)}-${name}.glb`, 50_000, 0),
        bounds: { center: centre, radius: 2 },
      });
      return {
        ...tile(`r${String(index)}.glb`, 10_000, 0.3, [leaf("a"), leaf("b")]),
        bounds: { center: centre, radius: 2 },
        box: [...centre, 1.5, 0, 0, 0, 1.5, 0, 0, 0, 1.5],
      };
    };
    const ring: TileTree = {
      refine: "REPLACE",
      root: tile("ring.glb", 20_000, 3, [0, 1, 2, 3, 4, 5, 6, 7].map(region)),
    };
    const facing = (headingDeg: number): View =>
      viewFrom([0, 0, 0], 1000, (bounds) => {
        const bearing = (Math.atan2(bounds.center[1], bounds.center[0]) * 180) / Math.PI;
        const off = Math.abs(((bearing - headingDeg + 540) % 360) - 180);
        return off > 50;
      });
    const { host, pending, arrive } = fakeHost();
    const fetched: string[] = [];
    const counted: StreamHost<string> = {
      ...host,
      load: (chosen) => {
        fetched.push(chosen.uri);
        return host.load(chosen);
      },
    };
    const budget = 440_000;
    const streamer = new TileStreamer(ring, counted, {
      budget,
      cacheBudget: budget * 1.5,
      concurrency: 16,
    });
    streamer.adopt(ring.root, "ring.glb");
    const lap = async (): Promise<void> => {
      for (let heading = 0; heading < 360; heading += 45) {
        for (let step = 0; step < 4 && (step === 0 || pending.size > 0); step++) {
          streamer.update(facing(heading));
          await arrive();
        }
        streamer.update(facing(heading));
      }
    };
    await lap();
    fetched.length = 0;
    await lap();
    const nearLeaves = fetched.filter((uri) => /^r[0246]-/.test(uri));
    expect(nearLeaves).toEqual([]);
  });

  it("aborts a fetch the view stops wanting, and tries a failed tile again later", async () => {
    const signals = new Map<string, AbortSignal>();
    const failing = new Set<string>(["east.glb"]);
    let now = 0;
    const clock = vi.spyOn(performance, "now").mockImplementation(() => now);
    const host: StreamHost<string> = {
      load: (chosen, signal) => {
        if (signal) signals.set(chosen.uri, signal);
        if (failing.has(chosen.uri)) return Promise.reject(new Error("503"));
        return new Promise<string>(() => undefined);
      },
      show: () => undefined,
      hide: () => undefined,
      dispose: () => undefined,
    };
    const streamer = new TileStreamer(site, host, {
      budget: 1e9,
      cacheBudget: 1e9,
      concurrency: 8,
    });
    streamer.adopt(site.root, "root.glb");
    const close = viewFrom([-50, 0, 5]);
    streamer.update(close);
    await Promise.resolve();
    expect(signals.get("west.glb")?.aborted).toBe(false);
    // Far away the root alone is wanted: two updates later the west fetch is dropped.
    const far = viewFrom([0, 0, 100_000]);
    streamer.update(far);
    expect(signals.get("west.glb")?.aborted).toBe(false);
    streamer.update(far);
    expect(signals.get("west.glb")?.aborted).toBe(true);
    // The east tile failed; back close, it is not asked for again at once, but is later.
    signals.delete("east.glb");
    streamer.update(close);
    expect(signals.has("east.glb")).toBe(false);
    now += 6000;
    failing.clear();
    streamer.update(close);
    expect(signals.has("east.glb")).toBe(true);
    clock.mockRestore();
  });

  it("puts at most the per-update allowance on screen, and asks for the rest at once", async () => {
    const { host, arrive } = fakeHost();
    const streamer = new TileStreamer(site, host, {
      budget: 1e9,
      cacheBudget: 1e9,
      concurrency: 8,
      maxShownPerUpdate: 150_000,
    });
    let asked = 0;
    streamer.onArrival = () => (asked += 1);
    streamer.adopt(site.root, "root.glb");
    const close = viewFrom([-50, 0, 5]);
    streamer.update(close);
    await arrive();
    streamer.update(close);
    await arrive();
    asked = 0;
    // Both regions' leaves are here: 200k each, over a 150k allowance. One region goes (a
    // swap cannot be split), the other waits for the next update, asked for at once.
    streamer.update(close);
    const uris = () => streamer.drawn.map((drawn) => drawn.uri).sort();
    expect(uris().filter((uri) => uri.endsWith("-a.glb") || uri.endsWith("-b.glb"))).toHaveLength(
      2,
    );
    await new Promise((done) => setTimeout(done, 0));
    expect(asked).toBeGreaterThan(0);
    streamer.update(close);
    expect(uris()).toEqual(["east-a.glb", "east-b.glb", "west-a.glb", "west-b.glb"]);
  });

  it("fetches the next level near the camera ahead, once the view is served", async () => {
    const { host, pending, arrive } = fakeHost();
    const streamer = new TileStreamer(site, host, {
      budget: 1e9,
      cacheBudget: 1e9,
      concurrency: 8,
      prefetchRadiusM: 45,
    });
    streamer.adopt(site.root, "root.glb");
    // The view wants the two regions; the west one is within the ring, the east one not.
    const view = viewFrom([-50, 0, 40], 1);
    streamer.update(view);
    expect([...pending.keys()].sort()).toEqual(
      ["east.glb", "west-a.glb", "west-b.glb", "west.glb"].sort(),
    );
    await arrive();
    streamer.update(view);
    // Fetched ahead, not shown.
    expect(streamer.drawn.map((drawn) => drawn.uri).sort()).toEqual(["east.glb", "west.glb"]);
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

  it("says when a failed tile is due again, so a page that updates only when told retries", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "performance"] });
    try {
      let fail = true;
      let loads = 0;
      const host: StreamHost<string> = {
        load: (chosen) => {
          loads += 1;
          return fail ? Promise.reject(new Error("503")) : Promise.resolve(chosen.uri);
        },
        show: () => undefined,
        hide: () => undefined,
        dispose: () => undefined,
      };
      const streamer = new TileStreamer(site, host, {
        budget: 1e9,
        cacheBudget: 1e9,
        concurrency: 1,
      });
      let told = 0;
      streamer.onArrival = () => (told += 1);
      streamer.adopt(site.root, "root.glb");
      const close = viewFrom([-50, 0, 5]);
      streamer.update(close);
      await vi.advanceTimersByTimeAsync(0);
      // Told at once (the failure), and nothing else until it is due.
      expect(told).toBe(1);
      await vi.advanceTimersByTimeAsync(RETRY_FAILED_MS - 100);
      expect(told).toBe(1);
      await vi.advanceTimersByTimeAsync(200);
      expect(told).toBe(2);
      fail = false;
      const before = loads;
      streamer.update(close);
      expect(loads).toBeGreaterThan(before);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("a flight's destination, fetched ahead", () => {
  const close = viewFrom([-50, 0, 5]);
  const far = viewFrom([0, 0, 100_000]);

  it("fetches the cut there and every tile above it, shallowest first, before the way there", async () => {
    const { host, pending, arrive } = fakeHost();
    const streamer = new TileStreamer(site, host, {
      budget: 1e9,
      cacheBudget: 1e9,
      concurrency: 3,
    });
    streamer.adopt(site.root, "root.glb");
    streamer.update(far);
    expect([...pending.keys()]).toEqual([]);
    streamer.prefetchView(close);
    expect(streamer.prefetchingDestination).toBe(true);
    // Two of three slots (one stays the view's), the level under the root first: a tile swaps
    // for its children only once all of them are in.
    expect([...pending.keys()].sort()).toEqual(["east.glb", "west.glb"]);
    for (let round = 0; round < 4; round++) {
      await arrive();
      // Still far away: the views on the way want the root alone, and abandon nothing ahead.
      streamer.update(far);
    }
    expect(pending.size).toBe(0);
    // Arrived: the destination's cut is drawn in one update, nothing left to fetch.
    streamer.update(close);
    expect(streamer.drawn.map((drawn) => drawn.uri).sort()).toEqual(
      ["east-a.glb", "east-b.glb", "west-a.glb", "west-b.glb"].sort(),
    );
    expect(pending.size).toBe(0);
    expect(streamer.prefetchingDestination).toBe(false);
  });

  it("keeps what it fetched from the cache's eviction until it is reached or forgotten", async () => {
    const { host, disposed, arrive } = fakeHost();
    const streamer = new TileStreamer(site, host, {
      budget: 1e9,
      cacheBudget: 60_000,
      concurrency: 3,
    });
    streamer.adopt(site.root, "root.glb");
    streamer.update(far);
    streamer.prefetchView(close);
    for (let round = 0; round < 4; round++) {
      await arrive();
      streamer.update(far);
    }
    expect(disposed).toEqual([]);
    // The flight was cancelled: what only it wanted goes to fit the cache.
    streamer.prefetchView(null);
    streamer.update(far);
    expect(disposed.length).toBeGreaterThan(0);
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
