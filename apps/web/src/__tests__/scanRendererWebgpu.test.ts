/**
 * The PlayCanvas WebGPU trial in ScanRendererHost (docs/WEBGPU_TRIAL.md), with fake renderers
 * and fake timers: WebGPU that does not start, or starts nothing, is drawn over by PlayCanvas
 * on WebGL2 on a fresh canvas, and says why; a WebGPU device lost for good is replaced by WebGL2,
 * which draws again; later scans this visit go straight to WebGL2, and choosing the renderer
 * again tries WebGPU again; and the frame meter counts the camera's motion frames only.
 */

import { Cartesian3, Event, Matrix4, PerspectiveFrustum, type Cesium3DTileset } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ScanRendererHost, type BackendModule } from "@/cesium/scanView/ScanRendererHost";
import type {
  BackendHooks,
  GraphicsApi,
  ScanBackend,
  ScanPose,
  SplatRendererKind,
} from "@/cesium/scanView/types";

/** One root tile and nothing under it: the scan is in as soon as the root is. */
const TILESET = {
  asset: { version: "1.1" },
  geometricError: 20,
  root: {
    refine: "REPLACE",
    geometricError: 0,
    boundingVolume: { box: [0, 0, 0, 40, 0, 0, 0, 40, 0, 0, 0, 10] },
    content: { uri: "root.glb" },
    extras: { gaussians: 1000 },
  },
};

/** A renderer that draws nothing but says it drew, and records what was asked of it. */
interface Fake {
  backend: ScanBackend<string>;
  renders: ScanPose[];
  destroyed: boolean;
  canvas: HTMLCanvasElement | null;
  hooks: BackendHooks | null;
}

function fake(name: SplatRendererKind, api: GraphicsApi, apiNote: string | null = null): Fake {
  const made: Fake = {
    renders: [],
    destroyed: false,
    canvas: null,
    hooks: null,
    backend: {
      name,
      api,
      apiNote,
      loadFactor: 1,
      load: (_url, tile) => Promise.resolve(tile.uri),
      add: () => undefined,
      remove: () => undefined,
      dispose: () => undefined,
      render: (pose) => void made.renders.push(pose),
      isDrawn: () => true,
      setBudget: () => undefined,
      destroy: () => {
        made.destroyed = true;
      },
    },
  };
  return made;
}

interface Rig {
  host: ScanRendererHost;
  /** Every renderer made, by module asked for, in order. */
  made: { module: string; fake: Fake }[];
  camera: { positionWC: Cartesian3; directionWC: Cartesian3 };
  globe: () => void;
  run: (ms: number, globe?: boolean) => Promise<void>;
  overlays: () => HTMLCanvasElement[];
  tileset: (key: string) => { key: string; tileset: Cesium3DTileset };
}

/**
 * A host whose `playcanvas-webgpu` module does `webgpu` (fails to start, or starts a WebGPU or
 * WebGL2 renderer) and whose `playcanvas` module always starts a WebGL2 renderer.
 */
function rig(webgpu: "fails" | "webgpu" | "webgl2" | "unfetched"): Rig {
  const made: Rig["made"] = [];
  const module = (name: string, create: () => Fake | Error): BackendModule => ({
    createBackend: (canvas, _budget, hooks) => {
      const result = create();
      if (result instanceof Error) return Promise.reject(result);
      result.canvas = canvas;
      result.hooks = hooks;
      made.push({ module: name, fake: result });
      return Promise.resolve(result.backend as ScanBackend<unknown>);
    },
  });
  const modules: Record<string, BackendModule> = {
    "playcanvas-webgpu": module("playcanvas-webgpu", () =>
      webgpu === "fails"
        ? new Error("no adapter")
        : webgpu === "webgpu" || webgpu === "unfetched"
          ? fake("playcanvas-webgpu", "webgpu")
          : fake("playcanvas-webgpu", "webgl2", "WebGPU unavailable: this browser has no WebGPU"),
    ),
    playcanvas: module("playcanvas", () => fake("playcanvas", "webgl2")),
  };
  const canvas = document.createElement("canvas");
  Object.defineProperty(canvas, "clientWidth", { value: 960 });
  Object.defineProperty(canvas, "clientHeight", { value: 600 });
  document.body.appendChild(canvas);
  const postRender = new Event();
  const camera = {
    positionWC: new Cartesian3(0, -60, 10),
    directionWC: new Cartesian3(0, 1, 0),
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
  // "unfetched": the WebGPU module's chunk does not arrive the first time it is asked for.
  let unfetched = webgpu === "unfetched" ? 1 : 0;
  const host = new ScanRendererHost(viewer as never, {
    backends: (kind) => {
      if (kind === "playcanvas-webgpu" && unfetched > 0) {
        unfetched -= 1;
        return Promise.reject(new TypeError("Failed to fetch dynamically imported module"));
      }
      const chosen = modules[kind];
      return chosen ? Promise.resolve(chosen) : Promise.reject(new Error(kind));
    },
  });
  const globe = (): void => {
    postRender.raiseEvent();
  };
  return {
    host,
    made,
    camera,
    globe,
    run: async (ms, withGlobe = false) => {
      for (let t = 0; t < ms; t += 16) {
        if (withGlobe) globe();
        await vi.advanceTimersByTimeAsync(16);
      }
    },
    overlays: () => [...document.querySelectorAll<HTMLCanvasElement>("canvas[data-scan-renderer]")],
    tileset: (key) => ({
      key,
      tileset: {
        resource: { url: `https://scan.test/${key}/tileset.json` },
        root: { computedTransform: Matrix4.IDENTITY.clone(), extras: { nativeLod: false } },
        isDestroyed: () => false,
      } as unknown as Cesium3DTileset,
    }),
  };
}

describe("the PlayCanvas WebGPU trial", () => {
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
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response(JSON.stringify(TILESET), { status: 200 }))),
    );
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    document.body.innerHTML = "";
  });

  it("draws with WebGPU where it starts, and says so", async () => {
    const r = rig("webgpu");
    r.host.setRenderer("playcanvas-webgpu");
    r.host.setTarget(r.tileset("a"));
    await r.run(400);
    const status = r.host.status();
    expect(status).toMatchObject({ kind: "playcanvas-webgpu", api: "webgpu", notice: null });
    expect(r.made.map((m) => m.module)).toEqual(["playcanvas-webgpu"]);
    expect(r.made[0]?.fake.renders.length).toBeGreaterThan(0);
    expect(r.overlays().map((c) => c.dataset.api)).toEqual(["webgpu"]);
    r.host.destroy();
  });

  it("draws with PlayCanvas on WebGL2, on a fresh canvas, when WebGPU does not start", async () => {
    const r = rig("fails");
    r.host.setRenderer("playcanvas-webgpu");
    r.host.setTarget(r.tileset("a"));
    await r.run(400);
    const status = r.host.status();
    expect(status.kind).toBe("playcanvas-webgpu");
    expect(status.api).toBe("webgl2");
    expect(status.notice).toBe("WebGPU did not start: no adapter");
    expect(status.error).toBeNull();
    expect(r.made.map((m) => m.module)).toEqual(["playcanvas"]);
    expect(r.made[0]?.fake.renders.length).toBeGreaterThan(0);
    // The canvas WebGPU was tried on is gone; the one drawn on is new, and says who drew it.
    expect(r.overlays()).toHaveLength(1);
    expect(r.overlays()[0]).toBe(r.made[0]?.fake.canvas);
    expect(r.overlays()[0]?.dataset).toMatchObject({ scanRenderer: "playcanvas", api: "webgl2" });
    // A real failure holds for the visit: the next scan goes straight to WebGL2.
    r.host.setTarget(r.tileset("b"));
    await r.run(400);
    expect(r.made.map((m) => m.module)).toEqual(["playcanvas", "playcanvas"]);
    r.host.destroy();
  });

  it("draws with WebGL2 when the WebGPU code did not arrive, and tries WebGPU again next scan", async () => {
    const r = rig("unfetched");
    r.host.setRenderer("playcanvas-webgpu");
    r.host.setTarget(r.tileset("a"));
    await r.run(400);
    const status = r.host.status();
    expect(status).toMatchObject({ kind: "playcanvas-webgpu", active: true, api: "webgl2" });
    // Said as what it is -- the code did not load -- not as WebGPU failing on this device.
    expect(status.notice).toContain("code did not load");
    expect(status.notice).not.toContain("WebGPU did not start");
    expect(r.made.map((m) => m.module)).toEqual(["playcanvas"]);
    expect(r.overlays().map((c) => c.dataset.scanRenderer)).toEqual(["playcanvas"]);
    // A dropped chunk is not held against WebGPU for the visit.
    r.host.setTarget(r.tileset("b"));
    await r.run(400);
    expect(r.made.map((m) => m.module)).toEqual(["playcanvas", "playcanvas-webgpu"]);
    expect(r.host.status()).toMatchObject({ api: "webgpu", notice: null });
    expect(r.overlays().map((c) => c.dataset.scanRenderer)).toEqual(["playcanvas-webgpu"]);
    r.host.destroy();
  });

  it("keeps PlayCanvas's own WebGL2 fallback, and its reason", async () => {
    const r = rig("webgl2");
    r.host.setRenderer("playcanvas-webgpu");
    r.host.setTarget(r.tileset("a"));
    await r.run(400);
    expect(r.host.status()).toMatchObject({
      api: "webgl2",
      notice: "WebGPU unavailable: this browser has no WebGPU",
    });
    expect(r.made.map((m) => m.module)).toEqual(["playcanvas-webgpu"]);
    r.host.destroy();
  });

  it("replaces a lost WebGPU device with WebGL2, which draws the scan again", async () => {
    const r = rig("webgpu");
    r.host.setRenderer("playcanvas-webgpu");
    r.host.setTarget(r.tileset("a"));
    await r.run(400);
    const first = r.made[0]?.fake;
    expect(first?.renders.length).toBeGreaterThan(0);
    first?.hooks?.deviceLost?.("WebGPU device lost: GPU process restarted");
    // Twice, as a renderer might say it: one replacement.
    first?.hooks?.deviceLost?.("WebGPU device lost: GPU process restarted");
    await r.run(400);
    expect(first?.destroyed).toBe(true);
    expect(r.made.map((m) => m.module)).toEqual(["playcanvas-webgpu", "playcanvas"]);
    const second = r.made[1]?.fake;
    expect(second?.renders.length).toBeGreaterThan(0);
    expect(r.host.status()).toMatchObject({
      kind: "playcanvas-webgpu",
      active: true,
      api: "webgl2",
      notice: "WebGPU device lost: GPU process restarted",
    });
    expect(r.overlays()).toHaveLength(1);

    // Another scan this visit goes straight to WebGL2...
    r.host.setTarget(r.tileset("b"));
    await r.run(400);
    expect(r.made.map((m) => m.module)).toEqual(["playcanvas-webgpu", "playcanvas", "playcanvas"]);
    // ...until the trial is chosen again.
    r.host.setRenderer("playcanvas");
    await r.run(400);
    r.host.setRenderer("playcanvas-webgpu");
    await r.run(400);
    expect(r.made.at(-1)?.module).toBe("playcanvas-webgpu");
    expect(r.host.status()).toMatchObject({ api: "webgpu", notice: null });
    r.host.destroy();
  });

  it("a device lost while the session starts is replaced as it arrives", async () => {
    const r = rig("webgpu");
    r.host.setRenderer("playcanvas-webgpu");
    r.host.setTarget(r.tileset("a"));
    // The renderer exists; the session (fetching the tileset, the root) does not yet.
    await vi.advanceTimersByTimeAsync(0);
    const first = r.made[0]?.fake;
    expect(first).toBeDefined();
    first?.hooks?.deviceLost?.("WebGPU device lost: unknown");
    await r.run(800);
    expect(first?.destroyed).toBe(true);
    expect(r.made.map((m) => m.module)).toEqual(["playcanvas-webgpu", "playcanvas"]);
    expect(r.host.status()).toMatchObject({ active: true, api: "webgl2" });
    expect(r.overlays()).toHaveLength(1);
    r.host.destroy();
  });

  it("meters the camera's motion frames, and keeps the last gesture's numbers at rest", async () => {
    const r = rig("webgpu");
    r.host.setRenderer("playcanvas-webgpu");
    r.host.setTarget(r.tileset("a"));
    await r.run(1000);
    // Drawn while the scan came in, with the camera still: no reading.
    expect(r.host.status().meter).toBeNull();
    for (let i = 0; i < 30; i++) {
      r.camera.positionWC = new Cartesian3(0, -60 + (i + 1) * 0.05, 10);
      r.globe();
      await vi.advanceTimersByTimeAsync(16);
    }
    const moving = r.host.status().meter;
    expect(moving?.live).toBe(true);
    expect(moving?.frames).toBe(29); // 30 motion frames
    expect(moving?.fps).toBeCloseTo(62.5, 0);
    await r.run(2000);
    const rested = r.host.status().meter;
    expect(rested?.live).toBe(false);
    expect(rested?.frames).toBe(29);
    r.host.destroy();
  });
});
