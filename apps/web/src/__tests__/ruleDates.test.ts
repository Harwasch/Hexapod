/**
 * The other dates a per-cell view rule draws under PlayCanvas (scanView/ruleDates.ts): each
 * streamed from the renderer's own pose within the budget it draws to, its tiles placed where
 * the date is in the scan's frame, its inferred layers with it; dropped when the rule no longer
 * names it, and everything let go on stop. And the pose's view: what is in front and inside
 * the frustum is visible, what is behind or off to a side is not.
 */
import { afterEach, describe, expect, it } from "vitest";

import {
  allotments,
  poseView,
  roomForRoots,
  RuleDates,
  type DatesBackend,
} from "@/cesium/scanView/ruleDates";
import { RULE_KIT } from "@/cesium/scanView/ruleKit";
import { namedKit, type RuleDate } from "@/cesium/scanView/scanRule";
import type { LayerLook } from "@/cesium/scanView/layerLook";
import type { ScanPose } from "@/cesium/scanView/types";
import { useSettings } from "@/state/settings";
import type { TileNode } from "@/view/tiles";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
/** The globe's streaming classes, as the rule's session lends them (ruleKit.ts). */
const KIT = namedKit(RULE_KIT);
const SHIFTED = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1];

const pose: ScanPose = {
  eye: [0, -20, 5],
  direction: [0, 1, 0],
  up: [0, 0, 1],
  fovy: Math.PI / 3,
  near: 0.1,
  far: 1000,
  width: 960,
  height: 600,
  pixelRatio: 1,
};

interface Mesh {
  uri: string;
  kind: "date" | "layer";
}

function fakeBackend() {
  const calls: string[] = [];
  const onScreen = new Set<Mesh>();
  const placed = new Map<Mesh, readonly number[] | null>();
  const looks = new Map<Mesh, LayerLook>();
  const backend: DatesBackend<Mesh> = {
    loadDate: (url, tile: TileNode) => {
      calls.push(`date ${new URL(tile.uri, url).pathname}`);
      return Promise.resolve({ uri: tile.uri, kind: "date" });
    },
    loadLayer: (url, tile: TileNode) => {
      calls.push(`layer ${new URL(tile.uri, url).pathname}`);
      return Promise.resolve({ uri: tile.uri, kind: "layer" });
    },
    add: (mesh) => onScreen.add(mesh),
    remove: (mesh) => onScreen.delete(mesh),
    dispose: (mesh) => calls.push(`dispose ${mesh.uri}`),
    place: (mesh, matrix) => placed.set(mesh, matrix),
    setLayerLook: (mesh, look) => looks.set(mesh, look),
  };
  return { backend, calls, onScreen, placed, looks };
}

const leaf = (uri: string) => ({
  root: {
    geometricError: 0,
    boundingVolume: { sphere: [0, 0, 0, 5] },
    content: { uri },
    extras: { gaussians: 2 },
  },
});

async function flush(): Promise<void> {
  for (let i = 0; i < 6; i += 1) await new Promise((r) => setTimeout(r, 0));
}

const JUNE: RuleDate = {
  url: "https://tiles.example/june/tileset.json",
  assetId: "june",
  // Its own source, 1, and its inferred layer's, 2.
  owns: [1, 2],
  extras: {
    inferredLayers: [
      {
        uri: "fill/tileset.json",
        evidence: {
          kind: "inferred",
          filler: "fixture",
          views: 3,
          gaussians: 10,
          meanConfidence: 0.5,
        },
      },
    ],
  },
  placement: SHIFTED,
  root: IDENTITY,
};

/**
 * A rule's plane over 64 x 24 m of 1 m cells: the scan PlayCanvas draws (source 0) owns the
 * cells west of `seam`, June (1) those east of it.
 */
function owners(seam = 32, dates: RuleDate[] = [JUNE]) {
  const ni = 64;
  const nj = 24;
  const data = new Uint8Array(4096).fill(255);
  for (let j = 0; j < nj; j += 1)
    for (let i = 0; i < ni; i += 1) data[j * ni + i] = i < seam ? 0 : 1;
  return {
    dates,
    plane: [0, 0, ni, nj] as const,
    cellM: 1,
    owners: { data, height: 1 },
    scanOwns: [0],
  };
}

/** Looking north at the whole plane from above its south edge. */
const WHOLE: ScanPose = {
  ...pose,
  eye: [32, -30, 30],
  direction: [0, 0.6, -0.8],
  up: [0, 0.8, 0.6],
};
/** Looking north at its west end, close and low: nothing east of x = 32 in view. */
const WEST: ScanPose = { ...pose, eye: [6, -3, 4], direction: [0, 0.8, -0.6], up: [0, 0.6, 0.8] };

/** The lent kit with a streamer that records the budgets it is handed. */
function spyKit() {
  const budgets: number[] = [];
  class Spy<M> extends KIT.TileStreamer<M> {
    override setBudget(budget: number, cacheBudget: number): void {
      budgets.push(budget);
      super.setBudget(budget, cacheBudget);
    }
  }
  return { kit: { ...KIT, TileStreamer: Spy }, budgets };
}

describe("the dates a rule draws under PlayCanvas", () => {
  afterEach(() => useSettings.getState().set({ inferredStyle: "hide" }));

  it("streams each from the pose, placed where it is, its inferred layer with it", async () => {
    useSettings.getState().set({ inferredStyle: "show" });
    const { backend, calls, onScreen, placed } = fakeBackend();
    let arrivals = 0;
    const dates = new RuleDates(
      backend,
      {
        arrived: () => {
          arrivals += 1;
        },
        changed: () => undefined,
        budget: () => 1e6,
        fetchJson: (url) => Promise.resolve(leaf(url.includes("fill") ? "fill.glb" : "june.glb")),
      },
      KIT,
    );
    dates.setRule(owners());
    await flush();
    expect(arrivals).toBeGreaterThan(0);
    dates.plan(pose, 1_000);
    await flush();
    dates.plan(pose, 2_000);
    expect(calls).toContain("date /june/june.glb");
    expect(calls).toContain("layer /june/fill/fill.glb");
    const june = [...onScreen].find((m) => m.kind === "date");
    expect(june).toBeDefined();
    // Its tiles go under its placement in the scan's frame.
    expect(placed.get(june!)).toEqual(SHIFTED);
    expect(dates.status()).toMatchObject({ dates: 1, tiles: 2 });

    // The rule no longer names it: off screen and let go.
    dates.setRule(owners(32, []));
    expect(onScreen.size).toBe(0);
    expect(calls).toContain("dispose june.glb");
    expect(dates.status()).toMatchObject({ dates: 0, tiles: 0 });
    dates.stop();
  });

  it("plans at most every replan interval, and says when the held-back plan is due", async () => {
    const { backend } = fakeBackend();
    const dates = new RuleDates(
      backend,
      {
        arrived: () => undefined,
        changed: () => undefined,
        budget: () => 1e6,
        fetchJson: () => Promise.resolve(leaf("june.glb")),
      },
      KIT,
    );
    dates.setRule(owners(32, [{ ...JUNE, extras: {} }]));
    await flush();
    dates.plan(pose, 1_000);
    expect(dates.dueBy()).toBeNull();
    dates.plan({ ...pose, eye: [1, -20, 5] }, 1_050);
    expect(dates.dueBy()).toBe(1_150);
    dates.plan({ ...pose, eye: [1, -20, 5] }, 1_200);
    expect(dates.dueBy()).toBeNull();
    // Nothing moved and nothing came: nothing to plan.
    dates.plan({ ...pose, eye: [1, -20, 5] }, 1_400);
    expect(dates.dueBy()).toBeNull();
    dates.stop();
  });
});

describe("one budget for every date a rule draws", () => {
  it("is split by each date's share of the visible cells it owns", async () => {
    const { kit, budgets } = spyKit();
    const { backend } = fakeBackend();
    const dates = new RuleDates(
      backend,
      {
        arrived: () => undefined,
        changed: () => undefined,
        budget: () => 1e6,
        fetchJson: () => Promise.resolve(leaf("june.glb")),
      },
      kit,
    );
    dates.setRule(owners(32, [{ ...JUNE, extras: {} }]));
    await flush();
    // The whole plane in view, half of it each: half the budget each. The scan PlayCanvas
    // draws takes its half through the renderer (`scanShare`); June's streamer the other.
    dates.plan(WHOLE, 1_000);
    expect(dates.scanShare()).toBeCloseTo(0.5, 1);
    expect(budgets.at(-1)).toBeCloseTo(0.5e6, -5);
    // Three quarters June's: three quarters of the budget June's.
    dates.setRule(owners(16, [{ ...JUNE, extras: {} }]));
    dates.plan(WHOLE, 2_000);
    expect(dates.scanShare()).toBeCloseTo(0.25, 1);
    expect(budgets.at(-1)).toBeCloseTo(0.75e6, -5);
    dates.stop();
  });

  it("streams nothing for a date that owns no cell in view", async () => {
    const { backend, calls, onScreen } = fakeBackend();
    const dates = new RuleDates(
      backend,
      {
        arrived: () => undefined,
        changed: () => undefined,
        budget: () => 1e6,
        fetchJson: () => Promise.resolve(leaf("june.glb")),
      },
      KIT,
    );
    dates.setRule(owners(32, [{ ...JUNE, extras: {} }]));
    await flush();
    dates.plan(WHOLE, 1_000);
    await flush();
    dates.plan(WHOLE, 2_000);
    expect([...onScreen].some((m) => m.kind === "date")).toBe(true);
    // Looking at the west end: June owns nothing in view, so it holds nothing.
    dates.plan(WEST, 3_000);
    expect(dates.scanShare()).toBe(1);
    expect([...onScreen].some((m) => m.kind === "date")).toBe(false);
    expect(calls).toContain("dispose june.glb");
    expect(dates.status()).toMatchObject({ dates: 1, tiles: 0, gaussians: 0 });
    // Back: June streams again.
    dates.plan(WHOLE, 4_000);
    await flush();
    dates.plan(WHOLE, 5_000);
    expect([...onScreen].some((m) => m.kind === "date")).toBe(true);
    dates.stop();
  });

  it("reserves each date's root first, the camera still as the budget drops below them", async () => {
    // A streamer keeps its root whatever its budget: each date is handed its root, then its
    // share of what is left. The scan's root (100k) and June's (100k) fit a 1M budget.
    const { kit, budgets } = spyKit();
    const { backend } = fakeBackend();
    let budget = 1e6;
    const dates = new RuleDates(
      backend,
      {
        arrived: () => undefined,
        changed: () => undefined,
        budget: () => budget,
        fetchJson: () => Promise.resolve(leaf("june.glb")),
      },
      kit,
    );
    dates.setRule({ ...owners(32, [{ ...JUNE, extras: {}, rootGaussians: 1e5 }]), scanRoot: 1e5 });
    await flush();
    dates.plan(WHOLE, 1_000);
    // Half each of the 800k left: June 100k + 400k; the scan the same, through the host.
    expect(budgets.at(-1)).toBeCloseTo(5e5, -4);
    expect(dates.scanShare()).toBeCloseTo(0.5, 2);
    // The frames slowed, the camera still: 300k leaves 100k to share.
    budget = 3e5;
    dates.plan(WHOLE, 2_000);
    expect(budgets.at(-1)).toBeCloseTo(1.5e5, -4);
    expect(dates.scanShare() * budget).toBeCloseTo(1.5e5, -4);
    // Below the roots (they fit the device's budget, or the rule would have been refused):
    // each holds its root and nothing more.
    budget = 1.5e5;
    dates.plan(WHOLE, 3_000);
    expect(budgets.at(-1)).toBe(1e5);
    expect(dates.scanShare() * budget).toBeCloseTo(1e5, 0);
    dates.stop();
  });

  it("reaches every date when the budget drops with the camera still", async () => {
    const { kit, budgets } = spyKit();
    const { backend } = fakeBackend();
    let budget = 1e6;
    const dates = new RuleDates(
      backend,
      {
        arrived: () => undefined,
        changed: () => undefined,
        budget: () => budget,
        fetchJson: () => Promise.resolve(leaf("june.glb")),
      },
      kit,
    );
    dates.setRule(owners(32, [{ ...JUNE, extras: {} }]));
    await flush();
    dates.plan(WHOLE, 1_000);
    await flush();
    dates.plan(WHOLE, 2_000);
    // Nothing moves and nothing comes; the frames slowed, and the host's budget dropped.
    budget = 2e5;
    dates.plan(WHOLE, 3_000);
    expect(budgets.at(-1)).toBeCloseTo(1e5, -4);
    dates.stop();
  });
});

describe("each date's budget, its root first", () => {
  it("reserves every root and splits what is left by share, within the budget", () => {
    // 400k: the scan's root 300k, June's 50k, half each of the 50k left -- 400k in all, where a
    // plain split (200k each) would have drawn the scan's 300k root and June's 200k: 500k.
    const both = allotments(400_000, { root: 300_000, share: 0.5 }, [{ root: 50_000, share: 0.5 }]);
    expect(both).toEqual({ scan: 325_000, dates: [75_000] });
    expect(both.scan + (both.dates[0] ?? 0)).toBe(400_000);
    // A date with no share holds nothing, its root neither; the scan's root is always held
    // (the host keeps it, its share or none).
    expect(allotments(400_000, { root: 300_000, share: 0 }, [{ root: 50_000, share: 1 }])).toEqual({
      scan: 300_000,
      dates: [100_000],
    });
  });

  it("says when the roots alone do not fit a device's budget, and holds the roots past it", () => {
    // Two 300k roots in 400k: no split can draw both within it.
    expect(roomForRoots(400_000, [300_000, 300_000])).toBe(false);
    expect(roomForRoots(400_000, [200_000, 200_000])).toBe(true);
    // A budget that drops below roots that fit the device's: each its root, no more.
    expect(
      allotments(400_000, { root: 300_000, share: 0.5 }, [{ root: 300_000, share: 0.5 }]),
    ).toEqual({ scan: 300_000, dates: [300_000] });
  });
});

describe("the view from a renderer's pose", () => {
  it("sees what is ahead within the frustum, not what is behind or off to a side", () => {
    const view = poseView(pose);
    expect(view.visible({ center: [0, 0, 0], radius: 1 })).toBe(true);
    expect(view.visible({ center: [0, -40, 5], radius: 1 })).toBe(false);
    expect(view.visible({ center: [100, -10, 5], radius: 1 })).toBe(false);
    // A sphere reaching into the frustum from the side counts.
    expect(view.visible({ center: [20, -5, 5], radius: 10 })).toBe(true);
    expect(view.projection).toBeCloseTo(600 / (2 * Math.tan(Math.PI / 6)), 9);
    expect(view.centre?.forward).toEqual([0, 1, 0]);
  });
});
