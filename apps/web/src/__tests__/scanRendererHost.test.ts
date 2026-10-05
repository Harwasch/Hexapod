/**
 * ScanRendererHost with a fake renderer and fake timers: what wakes the overlay, and that
 * nothing else does. Every frame the overlay draws is a `render` call here.
 */

import { Cartesian3, Event, Matrix4, PerspectiveFrustum, type Cesium3DTileset } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { RigidMotion } from "@twin/world";

import {
  FAR_BUDGET_SHARE,
  ScanRendererHost,
  prefetchScanDestination,
  type BackendModule,
} from "@/cesium/scanView/ScanRendererHost";
import type { ScanMotion } from "@/cesium/scanView/scanMotion";
import type { BackendHooks, ScanBackend, ScanPose } from "@/cesium/scanView/types";
import { pickSourceOf } from "@/cesium/sceneSelect/pickSources";
import type * as Telemetry from "@/cesium/telemetry";
import type { PickTile } from "@/lib/splatPick";
import { useSceneObjects } from "@/state/sceneObjects";
import { RETRY_FAILED_MS, TileStreamer } from "@/view/stream";

/**
 * Telemetry's rigid part as the motion link reads it (`telemetryOf(assetId).rigid`), per asset
 * id: a test moves an object as the driver does -- a new `motionVersion` -- and asks the globe
 * for a frame, as the driver's `requestRender` does.
 */
const telemetry = vi.hoisted(
  () =>
    new Map<string, { rigid: { instanceMotions: Map<number, unknown>; motionVersion: number } }>(),
);
vi.mock("@/cesium/telemetry", async (importOriginal) => {
  const actual = await importOriginal<typeof Telemetry>();
  return {
    ...actual,
    telemetryOf: (assetId: string) =>
      (telemetry.get(assetId) as ReturnType<typeof actual.telemetryOf>) ??
      actual.telemetryOf(assetId),
  };
});

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
  /** Set to make every later `render` throw, as a renderer broken mid-session would. */
  fail: { render: Error | null };
  /** Renderers made so far (one per session started). */
  created: () => number;
  destroyed: () => number;
  /** The globe renders one frame (its `postRender`). */
  globe: () => void;
  /** `ms` of display frames (16 ms each), with the globe rendering in each when asked. */
  run: (ms: number, globe?: boolean) => Promise<void>;
  /** Asks for the same scan, seen from afar or up close (SiteManager's far view). */
  setFar: (far: boolean) => void;
}

/** The tileset the next rig's fetch answers with. */
let served: unknown = TILESET;

async function rig(
  options: {
    tileset?: unknown;
    position?: Cartesian3;
    direction?: Cartesian3;
    /** The scan's asset id: its objects and motion are linked by it. */
    assetId?: string;
    /** More of the root's extras (objects, motion). */
    extras?: Record<string, unknown>;
    /** What the renderer can do beyond the basics (setMotion, place, frameDueBy). */
    backend?: Partial<ScanBackend<string>>;
    /** The scan is seen from afar from the start. */
    far?: boolean;
  } = {},
): Promise<Rig> {
  if (options.tileset) served = options.tileset;
  const renders: ScanPose[] = [];
  const loads: string[] = [];
  const pending: Rig["pending"] = new Map();
  const budgets: number[] = [];
  let hooks: BackendHooks | null = null;
  const fail: Rig["fail"] = { render: null };
  let created = 0;
  let destroyed = 0;
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
    render: (pose) => {
      if (fail.render) throw fail.render;
      renders.push(pose);
    },
    isDrawn: () => true,
    setBudget: (drawn) => void budgets.push(drawn),
    destroy: () => {
      destroyed += 1;
    },
    ...options.backend,
  };
  const module: BackendModule = {
    createBackend: (_canvas, _budget, given) => {
      created += 1;
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
  const host = new ScanRendererHost(viewer as never, { backends: () => Promise.resolve(module) });
  const tileset = {
    resource: { url: "https://scan.test/tileset.json" },
    root: {
      computedTransform: Matrix4.IDENTITY.clone(),
      extras: { nativeLod: false, ...options.extras },
    },
    isDestroyed: () => false,
  } as unknown as Cesium3DTileset;
  host.setRenderer("playcanvas");
  const setFar = (far: boolean): void =>
    host.setTarget({
      key: "scan",
      tileset,
      ...(options.assetId ? { assetId: options.assetId } : {}),
      far,
    });
  setFar(options.far === true);
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
    fail,
    created: () => created,
    destroyed: () => destroyed,
    camera,
    viewer,
    globe,
    run,
    setFar,
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

  it("a renderer that throws while drawing is retired, and the globe's render never sees it", async () => {
    const r = await rig();
    await settle(r);
    const failures: string[] = [];
    r.host.onFailure = (message) => failures.push(message);
    r.fail.render = new Error("Cannot read properties of null (reading 'hasCenters')");
    // The camera moved: the overlay draws inside the globe's `postRender`, which CesiumJS
    // raises outside its render-error try (a throw there stops its render loop for good).
    r.camera.positionWC = new Cartesian3(-20, -58, 10);
    expect(() => r.globe()).not.toThrow();
    await r.run(100);
    const status = r.host.status();
    expect(status.active).toBe(false);
    expect(status.error).toContain("hasCenters");
    expect(failures).toHaveLength(1);
    expect(r.destroyed()).toBe(1);
    expect(document.querySelectorAll("canvas[data-scan-renderer]")).toHaveLength(0);
    // Not restarted on the spot to throw again, however often the globe renders...
    for (let i = 0; i < 5; i++) {
      r.camera.positionWC = new Cartesian3(-20, -57 + i, 10);
      expect(() => r.globe()).not.toThrow();
    }
    await r.run(1000, true);
    expect(r.created()).toBe(1);
    // ...until the renderer is chosen again.
    r.fail.render = null;
    r.host.setRenderer("cesium");
    r.host.setRenderer("playcanvas");
    await r.run(400);
    expect(r.created()).toBe(2);
    expect(r.host.status()).toMatchObject({ active: true, error: null });
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

describe("a scan seen from afar", () => {
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

  it("is the same session as up close: a quarter of the budget, nothing culled, no objects", async () => {
    const tile = { checksum: "t", count: 0 } as unknown as PickTile;
    const r = await rig({ assetId: "camp", backend: { pickTiles: () => [tile] } });
    await settle(r);
    const near = r.host.status();
    expect(near).toMatchObject({ active: true, far: false });
    expect(r.renders.at(-1)?.farView).toBe(false);
    expect(pickSourceOf("camp")?.tiles()).toEqual([tile]);
    const loads = r.loads.length;

    // Zoomed out: the site disengages, its scan is still drawn, small.
    r.setFar(true);
    await r.run(400);
    const far = r.host.status();
    expect(far.far).toBe(true);
    expect(far.budget).toBe(Math.round(near.budget * FAR_BUDGET_SHARE));
    expect(r.budgets.at(-1)).toBe(far.budget);
    // Drawn again at once, with nothing culled for size (quality.ts).
    expect(r.renders.at(-1)?.farView).toBe(true);
    // Not a scan to select objects in from there.
    expect(pickSourceOf("camp")?.tiles()).toEqual([]);

    // And back in.
    r.setFar(false);
    await r.run(400);
    expect(r.host.status()).toMatchObject({ active: true, far: false, budget: near.budget });
    expect(r.budgets.at(-1)).toBe(near.budget);
    expect(r.renders.at(-1)?.farView).toBe(false);
    expect(pickSourceOf("camp")?.tiles()).toEqual([tile]);

    // One renderer the whole way: never stopped, never made again, nothing fetched again.
    expect(r.created()).toBe(1);
    expect(r.destroyed()).toBe(0);
    expect(r.loads.length).toBe(loads);
    expect(document.querySelectorAll("canvas[data-scan-renderer]")).toHaveLength(1);
    r.host.destroy();
  });

  it("keeps the full budget's tiles while it draws a quarter of them", async () => {
    // Two regions of 1M gaussians under a coarse root, near enough to want both: up close the
    // 3M budget draws them, from afar a quarter of it cannot -- the root alone is drawn, and
    // the regions stay loaded, so zooming back in fetches nothing.
    const heavy = {
      ...TILESET,
      root: {
        ...TILESET.root,
        extras: { gaussians: 100_000 },
        children: TILESET.root.children.map((child) => ({
          ...child,
          extras: { gaussians: 1_000_000 },
          children: [],
        })),
      },
    };
    const r = await rig({ tileset: heavy });
    await settle(r);
    expect(r.host.status().tiles).toBe(2);
    const loads = r.loads.length;
    r.setFar(true);
    // A move, so the cut is planned again under the new budget.
    r.camera.positionWC = new Cartesian3(-20, -60.5, 10);
    await r.run(1000, true);
    expect(r.host.status()).toMatchObject({ far: true, tiles: 1 });
    expect(r.host.status().cached).toBeGreaterThanOrEqual(2_000_000);
    r.setFar(false);
    await r.run(1000, true);
    expect(r.host.status()).toMatchObject({ far: false, tiles: 2 });
    expect(r.loads.length).toBe(loads);
    expect(r.created()).toBe(1);
    r.host.destroy();
  });

  it("a session that starts from afar draws from afar from its first frame", async () => {
    const r = await rig({ far: true });
    expect(r.budgets[0]).toBe(r.host.status().budget);
    expect(r.renders.length).toBeGreaterThan(0);
    expect(r.renders.every((pose) => pose.farView === true)).toBe(true);
    expect(r.host.status().far).toBe(true);
    r.host.destroy();
  });
});

describe("the scan's objects moving under the overlay", () => {
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
    telemetry.clear();
    useSceneObjects.setState({ objects: {}, poses: {} });
    vi.useRealTimers();
    vi.unstubAllGlobals();
    document.body.innerHTML = "";
  });

  it("draws as a driver moves an object, and nothing at all once the motion stops", async () => {
    const motions: (ScanMotion | null)[] = [];
    const r = await rig({
      assetId: "camp",
      backend: { setMotion: (motion) => void motions.push(motion) },
    });
    await settle(r);
    // The session handed the renderer its first motion (nothing moving) as it started.
    const handedAtStart = motions.length;
    expect(handedAtStart).toBe(1);
    // At rest, with the globe rendering for something else: no frame, no motion handed.
    let before = r.renders.length;
    await r.run(2000, true);
    expect(r.renders.length).toBe(before);
    expect(motions.length).toBe(handedAtStart);

    // Telemetry moves an object: each step a new motion version and a globe frame (the
    // driver's requestRender); the overlay draws each one, with the motion handed first.
    const rigid = { instanceMotions: new Map<number, RigidMotion>(), motionVersion: 0 };
    telemetry.set("camp", { rigid });
    for (let step = 1; step <= 10; step += 1) {
      rigid.motionVersion = step;
      r.globe();
      await r.run(16);
    }
    expect(r.renders.length - before).toBe(10);
    expect(motions.length - handedAtStart).toBe(10);
    expect(r.host.status().motion).toMatchObject({ updates: handedAtStart + 10 });

    // The motion stops (the object stays where it was moved): nothing more is drawn, the
    // globe rendering or not.
    before = r.renders.length;
    await r.run(3000, true);
    await r.run(2000);
    expect(r.renders.length).toBe(before);
    expect(motions.length - handedAtStart).toBe(10);
    r.host.destroy();
    // Stopping hands the renderer no motion at all.
    expect(motions.at(-1)).toBeNull();
  });

  it("draws a split object as it loads and as its pose is set, and nothing at rest", async () => {
    const placed: (readonly number[] | null)[] = [];
    const object = { uri: "objects/7/tileset.json", instance: 7, origin: [0, 0, 0] };
    const r = await rig({
      assetId: "camp",
      extras: { objects: [object] },
      backend: { place: (_mesh, matrix) => void placed.push(matrix) },
    });
    await settle(r);
    // Loaded (its own tileset, its root tile) and placed at its declared pose.
    expect(r.host.status().objects).toBe(1);
    expect(placed).toHaveLength(1);
    const before = r.renders.length;
    await r.run(2000, true);
    expect(r.renders.length).toBe(before);

    // Moved by hand (or a driver): no camera moves, the overlay still draws it.
    useSceneObjects
      .getState()
      .setPose("camp", 7, { translation: [1, 0, 0], rotation: [0, 0, 0, 1] });
    await r.run(100);
    expect(r.renders.length).toBe(before + 1);
    expect(placed).toHaveLength(2);
    expect(placed.at(-1)?.[12]).toBeCloseTo(1);
    // Another asset's pose is not this scan's.
    useSceneObjects
      .getState()
      .setPose("elsewhere", 7, { translation: [5, 0, 0], rotation: [0, 0, 0, 1] });
    await r.run(1000, true);
    expect(r.renders.length).toBe(before + 1);
    r.host.destroy();
  });

  it("draws by the time the renderer says held-back work is due, though nothing else moves", async () => {
    // A renderer that held a moving object's re-sort back (playcanvasBackend's
    // SORT_REFRESH_MS): due at `due`; the frame drawn then does it, and nothing is due after.
    let due: number | null = null;
    const r = await rig({
      backend: {
        frameDueBy: () => {
          if (due !== null && performance.now() >= due) due = null;
          return due;
        },
      },
    });
    await settle(r);
    const before = r.renders.length;
    due = performance.now() + 100;
    // The frame that held it back (the last step of a motion).
    r.hooks().frameWanted();
    await r.run(32);
    expect(r.renders.length).toBe(before + 1);
    await r.run(200);
    expect(r.renders.length).toBe(before + 2);
    expect(due).toBeNull();
    await r.run(3000);
    expect(r.renders.length).toBe(before + 2);
    r.host.destroy();
  });
});
