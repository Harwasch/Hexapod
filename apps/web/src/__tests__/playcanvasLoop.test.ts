/**
 * PlayCanvas's own update loop in the scan renderer (playcanvasBackend.ts) pauses at rest, and
 * a tile disposed at rest does not keep it running.
 *
 * A disposed tile's GPU resource is destroyed only after a few rendered frames
 * (DESTROY_AFTER_FRAMES), and the loop does not pause while one waits. An eviction at the end
 * of a plan -- an off-screen tile dropped from the cache, a load that landed after it was
 * abandoned -- used to get the one frame it was evicted in and no more: the resource waited for
 * ever and the loop ticked every display frame for as long as the page stayed open. PlayCanvas
 * is faked here (no WebGL in jsdom): what matters is which frames the renderer asks the host
 * for, when it destroys a resource, and whether its loop asks for another tick.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { BackendHooks, ScanPose } from "@/cesium/scanView/types";
import type { TileWork } from "@/cesium/scanView/tileWork";
import type { TileNode } from "@/view/tiles";

/** What the fake PlayCanvas records: resources destroyed, ticks asked for, the app itself. */
const pc = vi.hoisted(() => ({
  destroyed: 0,
  ticksAsked: 0,
  app: null as null | {
    handlers: Map<string, ((...args: unknown[]) => void)[]>;
    gsplatHandlers: Map<string, ((...args: unknown[]) => void)[]>;
    requestAnimationFrame: () => void;
  },
}));

vi.mock("playcanvas", () => {
  class Handlers {
    readonly handlers = new Map<string, ((...args: unknown[]) => void)[]>();
    on(name: string, handler: (...args: unknown[]) => void): void {
      this.handlers.set(name, [...(this.handlers.get(name) ?? []), handler]);
    }
    fire(name: string, ...args: unknown[]): void {
      for (const handler of this.handlers.get(name) ?? []) handler(...args);
    }
  }
  class Entity {
    parent: { removeChild(e: Entity): void } | null = null;
    camera?: Record<string, unknown>;
    gsplat?: Record<string, unknown>;
    constructor(readonly name = "") {}
    addComponent(type: string, data: Record<string, unknown>): void {
      if (type === "camera") this.camera = { ...data };
      if (type === "gsplat") this.gsplat = { ...data, setParameter: () => undefined };
    }
    setPosition = (): void => undefined;
    setLocalPosition = (): void => undefined;
    lookAt = (): void => undefined;
    destroy(): void {
      this.parent?.removeChild(this);
    }
  }
  class Application extends Handlers {
    readonly graphicsDevice = { isWebGPU: false, maxPixelRatio: 1, gl: null };
    readonly scene = Object.assign(new Handlers(), { gsplat: {} });
    readonly systems = { gsplat: new Handlers() };
    readonly root = {
      addChild: (e: Entity) => {
        e.parent = this.root;
      },
      removeChild: (e: Entity) => {
        e.parent = null;
      },
    };
    autoRender = true;
    constructor() {
      super();
      pc.app = {
        handlers: this.handlers,
        gsplatHandlers: this.systems.gsplat.handlers,
        requestAnimationFrame: () => this.requestAnimationFrame(),
      };
    }
    requestAnimationFrame(): void {
      pc.ticksAsked += 1;
    }
    setCanvasFillMode = (): void => undefined;
    setCanvasResolution = (): void => undefined;
    resizeCanvas = (): void => undefined;
    start = (): void => undefined;
    render(): void {
      // Every rendered frame with splats in it: everything sorted, nothing loading.
      this.systems.gsplat.fire("frame:ready", null, null, true, 0);
    }
    destroy = (): void => undefined;
  }
  class GSplatResource {
    destroy(): void {
      pc.destroyed += 1;
    }
  }
  class GSplatData {
    activated = false;
  }
  class Vec3 {
    set(): this {
      return this;
    }
  }
  class Color {
    readonly rgba = [0, 0, 0, 0];
  }
  return {
    Application,
    Entity,
    GSplatResource,
    GSplatData,
    Vec3,
    Color,
    FILLMODE_NONE: "NONE",
    RESOLUTION_AUTO: "AUTO",
    ASPECT_AUTO: 0,
  };
});

/** A decode worker that answers every tile with one splat at once. */
class FakeWorker {
  onmessage: ((event: { data: unknown }) => void) | null = null;
  postMessage(message: { id: number }): void {
    queueMicrotask(() =>
      this.onmessage?.({
        data: {
          id: message.id,
          count: 1,
          properties: { x: new Float32Array(1) },
          order: new Uint32Array([0]),
          origin: [0, 0, 0],
        },
      }),
    );
  }
  terminate = (): void => undefined;
}

const POSE: ScanPose = {
  eye: [0, -10, 2],
  direction: [0, 1, 0],
  up: [0, 0, 1],
  fovy: 1,
  near: 0.1,
  far: 1000,
  width: 960,
  height: 600,
  pixelRatio: 1,
};

const tile = (uri: string) => ({ uri }) as unknown as TileNode;

describe("PlayCanvas's loop in the scan renderer", () => {
  beforeEach(() => {
    vi.stubGlobal("Worker", FakeWorker);
    pc.destroyed = 0;
    pc.ticksAsked = 0;
  });
  afterEach(() => vi.unstubAllGlobals());

  it("lets a tile disposed at rest go within a few frames, and then pauses", async () => {
    const { createBackend, QUIET_TICKS_BEFORE_PAUSE } =
      await import("@/cesium/scanView/playcanvasBackend");
    /** The host's side: a frame asked for is drawn on the next display frame. */
    let wanted = 0;
    const hooks: BackendHooks = {
      frameWanted: () => {
        wanted += 1;
      },
      work: { run: (task: () => unknown) => Promise.resolve(task()) } as unknown as TileWork,
      maxShDegree: 0,
    };
    const backend = await createBackend(document.createElement("canvas"), 1_000_000, hooks);
    const app = pc.app;
    if (!app) throw new Error("no app");
    const tick = (): void => {
      for (const handler of app.handlers.get("frameupdate") ?? []) handler();
      // PlayCanvas asks for its next tick from inside each tick (`app.requestAnimationFrame`,
      // which the renderer gates).
      app.requestAnimationFrame();
    };
    /** One display frame: the host draws if a frame was wanted, then PlayCanvas ticks. */
    const displayFrame = (): void => {
      if (wanted > 0) {
        wanted = 0;
        backend.render(POSE);
      }
      tick();
    };

    const a = await backend.load("https://scan.test/tileset.json", tile("a.glb"));
    const b = await backend.load("https://scan.test/tileset.json", tile("b.glb"));
    backend.add(a);
    backend.add(b);
    backend.render(POSE);
    for (let i = 0; i < 60; i++) displayFrame();
    // At rest: nothing asked of the host, and the loop has paused.
    const asked = pc.ticksAsked;
    for (let i = 0; i < 30; i++) displayFrame();
    expect(pc.ticksAsked).toBe(asked);
    expect(wanted).toBe(0);

    // An off-screen tile is dropped from the cache at the end of a plan: the frame it was
    // dropped in is drawn, and nothing else is moving.
    backend.remove(b);
    backend.dispose(b);
    backend.render(POSE);
    expect(pc.destroyed).toBe(0);
    for (let i = 0; i < QUIET_TICKS_BEFORE_PAUSE + 40; i++) displayFrame();
    // Its resource is gone, and the loop has paused again: no ticks, no frames asked for.
    expect(pc.destroyed).toBe(1);
    const settled = pc.ticksAsked;
    for (let i = 0; i < 60; i++) displayFrame();
    expect(pc.ticksAsked).toBe(settled);
    expect(wanted).toBe(0);

    // A load that lands after it was abandoned is disposed with no frame at all.
    const late = await backend.load("https://scan.test/tileset.json", tile("c.glb"));
    backend.dispose(late);
    for (let i = 0; i < QUIET_TICKS_BEFORE_PAUSE + 40; i++) displayFrame();
    expect(pc.destroyed).toBe(2);
    const after = pc.ticksAsked;
    for (let i = 0; i < 60; i++) displayFrame();
    expect(pc.ticksAsked).toBe(after);
    backend.destroy();
  });
});
