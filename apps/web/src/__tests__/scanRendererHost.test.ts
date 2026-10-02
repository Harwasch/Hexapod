/**
 * ScanRendererHost with a fake renderer and fake timers: what wakes the overlay, and that
 * nothing else does. Every frame the overlay draws is a `render` call here.
 */

import { Cartesian3, Event, Matrix4, PerspectiveFrustum, type Cesium3DTileset } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ScanRendererHost,
  prefetchScanDestination,
  type BackendModule,
} from "@/cesium/scanView/ScanRendererHost";
import type { BackendHooks, ScanBackend, ScanPose } from "@/cesium/scanView/types";
import { RETRY_FAILED_MS, TileStreamer } from "@/view/stream";

/** A scan at the origin: a root over two regions 40 m apart, each a merged parent of leaves. */
const TILESET = {
  asset: { version: "1.1" },
  geometricError: 20,
  root: {
    refine: "REPLACE",
    geometricError: 10,
    boundingVolume: { box: [0, 0, 0, 40, 0, 0, 0, 40, 0, 0, 0, 10] },
    content: { uri: "root.glb" },
    extras: { gaussians: 1000 },
    children: [-20, 20].map((x) => ({
      geometricError: 2,
      boundingVolume: { box: [x, 0, 0, 20, 0, 0, 0, 40, 0, 0, 0, 10] },
      content: { uri: `r${String(x)}.glb` },
      extras: { gaussians: 1000 },
      children: [-10, 10].map((y) => ({
        geometricError: 0,
        boundingVolume: { box: [x, y, 0, 20, 0, 0, 0, 20, 0, 0, 0, 10] },
        content: { uri: `r${String(x)}_${String(y)}.glb` },
        extras: { gaussians: 1000 },
      })),
    })),
  },
};

interface Rig {
  host: ScanRendererHost;
  renders: ScanPose[];
  loads: string[];
  /** Fetches asked for and not answered yet, by tile uri. */
  pending: Map<string, { resolve: () => void; reject: (error: Error) => void }>;
  hooks: () => BackendHooks;
  budgets: number[];
  camera: {
    positionWC: Cartesian3;
    directionWC: Cartesian3;
    upWC: Cartesian3;
    frustum: PerspectiveFrustum;
  };
  viewer: { resolutionScale: number; useBrowserRecommendedResolution: boolean };
  /** The globe renders one frame (its `postRender`). */
  globe: () => void;
  /** `ms` of display frames (16 ms each), with the globe rendering in each when asked. */
  run: (ms: number, globe?: boolean) => Promise<void>;
}

/** The tileset the next rig's fetch answers with. */
let served: unknown = TILESET;

async function rig(
  options: { tileset?: unknown; position?: Cartesian3; direction?: Cartesian3 } = {},
): Promise<Rig> {
  if (options.tileset) served = options.tileset;
  const renders: ScanPose[] = [];
  const loads: string[] = [];
  const pending: Rig["pending"] = new Map();
  const budgets: number[] = [];
  let hooks: BackendHooks | null = null;
  const backend: ScanBackend<string> = {
    name: "playcanvas",
    loadFactor: 1,
    load: (_url, tile) => {
      loads.push(tile.uri);
      if (tile.uri === "root.glb") return Promise.resolve(tile.uri);
      return new Promise<string>((resolve, reject) =>
        pending.set(tile.uri, { resolve: () => resolve(tile.uri), reject }),
      );
    },
    add: () => undefined,
    remove: () => undefined,
    dispose: () => undefined,
    render: (pose) => void renders.push(pose),
    isDrawn: () => true,
    setBudget: (drawn) => void budgets.push(drawn),
    destroy: () => undefined,
  };
  const module: BackendModule = {
    createBackend: (_canvas, _budget, given) => {
      hooks = given;
      return Promise.resolve(backend as ScanBackend<unknown>);
    },
  };
  const canvas = document.createElement("canvas");
  Object.defineProperty(canvas, "clientWidth", { value: 960 });
  Object.defineProperty(canvas, "clientHeight", { value: 600 });
  document.body.appendChild(canvas);
  const postRender = new Event();
  const camera = {
    positionWC: options.position ?? new Cartesian3(-20, -60, 10),
    directionWC: options.direction ?? new Cartesian3(0, 1, 0),
    upWC: new Cartesian3(0, 0, 1),
    frustum: new PerspectiveFrustum({ fov: 1, aspectRatio: 1.6, near: 0.1, far: 1e4 }),
  };
  const viewer = {
    camera,
    canvas,
    scene: { postRender },
    resolutionScale: 1,
    useBrowserRecommendedResolution: true,
  };
  const host = new ScanRendererHost(viewer as never, () => Promise.resolve(module));
  const tileset = {
    resource: { url: "https://scan.test/tileset.json" },
    root: { computedTransform: Matrix4.IDENTITY.clone(), extras: { nativeLod: false } },
    isDestroyed: () => false,
  } as unknown as Cesium3DTileset;
  host.setRenderer("playcanvas");
  host.setTarget({ key: "scan", tileset });
  const globe = (): void => {
    postRender.raiseEvent();
  };
  const run = async (ms: number, withGlobe = false): Promise<void> => {
    for (let t = 0; t < ms; t += 16) {
      if (withGlobe) globe();
      await vi.advanceTimersByTimeAsync(16);
    }
  };
  // The session starts (tileset, root) and draws its first frames.
  await run(400);
  return {
    host,
    renders,
    loads,
    pending,
    hooks: () => {
      if (!hooks) throw new Error("no backend yet");
      return hooks;
    },
    budgets,
    camera,
    viewer,
    globe,
    run,
  };
}

/** Lets every fetch asked for so far arrive. */
async function arriveAll(r: Rig): Promise<void> {
  for (const [uri, fetch] of [...r.pending]) {
    r.pending.delete(uri);
    fetch.resolve();
  }
  await vi.advanceTimersByTimeAsync(0);
}

/** Loads everything the view wants and lets the overlay settle. */
async function settle(r: Rig): Promise<void> {
  for (let round = 0; round < 6 && r.pending.size > 0; round++) {
    await arriveAll(r);
    await r.run(400);
  }
  await r.run(1000);
}

describe("what wakes the splat overlay", () => {
  beforeEach(() => {
    vi.useFakeTimers({
      toFake: [
        "setTimeout",
        "clearTimeout",
        "requestAnimationFrame",
        "cancelAnimationFrame",
        "performance",
      ],
    });
    served = TILESET;
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response(JSON.stringify(served), { status: 200 }))),
    );
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    document.body.innerHTML = "";
  });

  it("draws while the scan arrives, and nothing at all once it is in and still", async () => {
    const r = await rig();
    expect(r.renders.length).toBeGreaterThan(0);
    expect(r.loads).toContain("r-20.glb");
    await settle(r);
    expect(r.host.status().tiles).toBeGreaterThan(1);
    const before = r.renders.length;
    await r.run(5000);
    expect(r.renders.length).toBe(before);
    r.host.destroy();
  });

  it("a tile arriving at rest is drawn", async () => {
    const r = await rig();
    const before = r.renders.length;
    await arriveAll(r);
    await r.run(100);
    expect(r.renders.length).toBeGreaterThan(before);
    r.host.destroy();
  });

  it("the globe rendering for anything else draws nothing", async () => {
    const r = await rig();
    await settle(r);
    const before = r.renders.length;
    // Wind on another site: the globe renders every frame for five seconds.
    await r.run(5000, true);
    expect(r.renders.length).toBe(before);
    r.host.destroy();
  });

  it("a moved camera draws with the globe, cut, then the full-resolution frame at rest", async () => {
    const r = await rig();
    await settle(r);
    r.viewer.useBrowserRecommendedResolution = false;
    r.viewer.resolutionScale = 1;
    vi.stubGlobal("devicePixelRatio", 2);
    const before = r.renders.length;
    r.camera.positionWC = new Cartesian3(-20, -58, 10);
    r.globe();
    expect(r.renders.length).toBe(before + 1);
    expect(r.renders.at(-1)?.pixelRatio).toBeCloseTo(1.2);
    // 200 ms later, with nothing else happening: the still frame at the full ratio.
    await r.run(400);
    expect(r.renders.at(-1)?.pixelRatio).toBe(2);
    const settled = r.renders.length;
    await r.run(3000);
    expect(r.renders.length).toBe(settled);
    r.host.destroy();
  });

  it("the globe's resolution changing (a ladder step, the sharpened still) is drawn", async () => {
    const r = await rig();
    await settle(r);
    r.viewer.useBrowserRecommendedResolution = false;
    r.viewer.resolutionScale = 0.65;
    vi.stubGlobal("devicePixelRatio", 2);
    r.globe();
    expect(r.renders.at(-1)?.pixelRatio).toBeCloseTo(1.3);
    r.host.destroy();
  });

  it("a re-plan the throttle held back is done once it allows", async () => {
    const r = await rig();
    await settle(r);
    const plans = vi.spyOn(TileStreamer.prototype, "update");
    // Two moves 30 ms apart: the first re-plans, the second is inside the 150 ms window.
    r.camera.positionWC = new Cartesian3(-20, -59, 10);
    r.globe();
    expect(plans).toHaveBeenCalledTimes(1);
    await r.run(32);
    r.camera.positionWC = new Cartesian3(-20, -58, 10);
    r.globe();
    expect(plans).toHaveBeenCalledTimes(1);
    // The camera rests; the held-back plan runs by itself, from where it stopped.
    await r.run(300);
    expect(plans).toHaveBeenCalledTimes(2);
    expect(plans.mock.calls.at(-1)?.[0].eye[1]).toBeCloseTo(-58);
    plans.mockRestore();
    r.host.destroy();
  });

  it("the renderer asking for a frame (a sort finished, detail streamed) draws one", async () => {
    const r = await rig();
    await settle(r);
    const before = r.renders.length;
    r.hooks().frameWanted();
    await r.run(50);
    expect(r.renders.length).toBe(before + 1);
    r.host.destroy();
  });

  it("a failed tile is tried again five seconds later, at rest", async () => {
    const r = await rig();
    const failed = [...r.pending.keys()][0] ?? "";
    r.pending.get(failed)?.reject(new Error("503"));
    r.pending.delete(failed);
    await r.run(1000);
    const asked = r.loads.filter((uri) => uri === failed).length;
    await r.run(RETRY_FAILED_MS);
    expect(r.loads.filter((uri) => uri === failed).length).toBe(asked + 1);
    expect(r.host.status().error).toBe("503");
    r.host.destroy();
  });

  it("fetches a flight's destination ahead, the views on the way notwithstanding", async () => {
    // Far off, looking away: the root alone.
    const r = await rig({
      position: new Cartesian3(0, -5000, 10),
      direction: new Cartesian3(0, -1, 0),
    });
    await settle(r);
    expect(r.loads).toEqual(["root.glb"]);
    const before = r.loads.length;
    const cancel = prefetchScanDestination({
      position: new Cartesian3(20, -40, 10),
      direction: new Cartesian3(0, 1, 0),
      up: new Cartesian3(0, 0, 1),
    });
    await r.run(50);
    // The level under the root first (two of the three fetch slots), with the camera still
    // where it was.
    expect(r.loads.slice(before).sort()).toEqual(["r-20.glb", "r20.glb"]);
    cancel();
    r.host.destroy();
  });

  it("the budget moving re-plans and draws again", async () => {
    // One tile of 2.5M gaussians: drawn near the desktop's 3M budget.
    const heavy = {
      ...TILESET,
      root: { ...TILESET.root, extras: { gaussians: 2_500_000 }, children: [] },
    };
    const r = await rig({ tileset: heavy });
    await settle(r);
    const before = r.budgets.length;
    let renders = 0;
    // Slow motion frames (50 ms) while drawing near the budget: the adaptive budget cuts.
    for (let i = 0; i < 120 && r.budgets.length === before; i++) {
      r.camera.positionWC = new Cartesian3(-20, -60 + i * 0.01, 10);
      r.globe();
      renders = r.renders.length;
      await vi.advanceTimersByTimeAsync(50);
    }
    expect(r.budgets.length).toBe(before + 1);
    expect(r.budgets.at(-1)).toBeLessThan(3_000_000);
    // The frame that moved it asked for another: drawn without the globe.
    expect(r.renders.length).toBeGreaterThan(renders);
    r.host.destroy();
  });
});
