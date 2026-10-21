/**
 * The other dates a per-cell view rule draws under PlayCanvas (scanView/ruleDates.ts): each
 * streamed from the renderer's own pose within the budget it draws to, its tiles placed where
 * the date is in the scan's frame, its inferred layers with it; dropped when the rule no longer
 * names it, and everything let go on stop. And the pose's view: what is in front and inside
 * the frustum is visible, what is behind or off to a side is not.
 */
import { afterEach, describe, expect, it } from "vitest";

import { poseView, RuleDates, type DatesBackend } from "@/cesium/scanView/ruleDates";
import type { RuleDate } from "@/cesium/scanView/scanRule";
import type { LayerLook } from "@/cesium/scanView/layerLook";
import type { ScanPose } from "@/cesium/scanView/types";
import { useSettings } from "@/state/settings";
import type { TileNode } from "@/view/tiles";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
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

describe("the dates a rule draws under PlayCanvas", () => {
  afterEach(() => useSettings.getState().set({ inferredStyle: "hide" }));

  it("streams each from the pose, placed where it is, its inferred layer with it", async () => {
    useSettings.getState().set({ inferredStyle: "show" });
    const { backend, calls, onScreen, placed } = fakeBackend();
    let arrivals = 0;
    const dates = new RuleDates(backend, {
      arrived: () => {
        arrivals += 1;
      },
      changed: () => undefined,
      budget: () => 1e6,
      fetchJson: (url) => Promise.resolve(leaf(url.includes("fill") ? "fill.glb" : "june.glb")),
    });
    dates.setDates([JUNE]);
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
    dates.setDates([]);
    expect(onScreen.size).toBe(0);
    expect(calls).toContain("dispose june.glb");
    expect(dates.status()).toMatchObject({ dates: 0, tiles: 0 });
    dates.stop();
  });

  it("plans at most every replan interval, and says when the held-back plan is due", async () => {
    const { backend } = fakeBackend();
    const dates = new RuleDates(backend, {
      arrived: () => undefined,
      changed: () => undefined,
      budget: () => 1e6,
      fetchJson: () => Promise.resolve(leaf("june.glb")),
    });
    dates.setDates([{ ...JUNE, extras: {} }]);
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
