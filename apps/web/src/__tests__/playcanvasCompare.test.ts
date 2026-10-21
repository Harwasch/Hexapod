/**
 * Two dates compared under PlayCanvas (playcanvasBackend.ts, scanView/scanCompare.ts): the side
 * stream is in the work buffer from the start; with a comparison published for the scan it
 * draws (the right-hand date), the left date is streamed into the same app, every tile's copy --
 * the scan's, the left date's, a layer's -- writes the side its date is drawn on, the frame's
 * hook hands that side to the fragment shader and cuts at the divider, and moving the divider
 * changes one uniform of the frame and copies nothing; scene selection picks each date only on
 * its side. PlayCanvas is faked (no WebGL in jsdom): what matters is what each tile and the frame
 * are given. That the pixels are cut there is e2e/compareScene.spec.ts's.
 */
import { SplitDirection } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { LayerLook } from "@/cesium/scanView/layerLook";
import { RULE_KIT } from "@/cesium/scanView/ruleKit";
import {
  COMPARE_HOOK_PS,
  publishScanCompare,
  scanCompareReading,
  SIDE_STREAM,
  SIDE_VARYING,
  SIDE_WRITE_STEP,
  type ScanCompareView,
} from "@/cesium/scanView/scanCompare";
import { lendRuleKit } from "@/cesium/scanView/scanRule";
import { publishScanSun } from "@/state/sunReport";
import { pickSourceOf } from "@/cesium/sceneSelect/pickSources";
import type { BackendHooks, ScanPose } from "@/cesium/scanView/types";
import type { TileWork } from "@/cesium/scanView/tileWork";
import type { TileNode } from "@/view/tiles";

interface FakeComponent {
  modifier: { glsl: string } | null | undefined;
  params: Map<string, unknown>;
  sets: string[];
}

const pc = vi.hoisted(() => {
  const chunks = new Map<string, string>([
    ["gsplatModifyVS", "default modify"],
    ["gsplatModifyPS", "default fragment"],
  ]);
  return {
    entities: [] as { name: string; gsplat?: FakeComponent }[],
    // The user material PlayCanvas copies onto its draw material (`app.scene.gsplat.material`).
    material: {
      chunks,
      params: new Map<string, number[]>(),
      updates: 0,
      getShaderChunks: () => chunks,
      setParameter(name: string, value: unknown) {
        this.params.set(name, Array.from(value as Float32Array));
      },
      update() {
        this.updates += 1;
      },
    },
    // The varyings handed from the frame's vertex hook to its fragment hook.
    varyings: {
      names: [] as string[],
      add(streams: { name: string }[]) {
        this.names.push(...streams.map((v) => v.name));
      },
      remove(names: string[]) {
        this.names = this.names.filter((n) => !names.includes(n));
      },
    },
    // The work buffer's format (`app.scene.gsplat.format`) and the streams added to it.
    format: {
      streams: [] as { name: string; format: number }[],
      addExtraStreams(streams: { name: string; format: number }[]) {
        for (const s of streams)
          if (!this.streams.some((t) => t.name === s.name)) this.streams.push(s);
      },
    },
  };
});

vi.mock("playcanvas", () => {
  class Handlers {
    readonly handlers = new Map<string, ((...args: unknown[]) => void)[]>();
    on(name: string, handler: (...args: unknown[]) => void): void {
      this.handlers.set(name, [...(this.handlers.get(name) ?? []), handler]);
    }
  }
  class Entity {
    parent: { removeChild(e: Entity): void } | null = null;
    camera?: Record<string, unknown>;
    gsplat?: Record<string, unknown> & FakeComponent;
    constructor(readonly name = "") {
      pc.entities.push(this);
    }
    addComponent(type: string, data: Record<string, unknown>): void {
      if (type === "camera") this.camera = { ...data };
      if (type === "gsplat") {
        const component: FakeComponent & Record<string, unknown> = {
          ...data,
          modifier: undefined,
          params: new Map(),
          sets: [],
        };
        component.setWorkBufferModifier = (value: { glsl: string } | null) => {
          component.modifier = value;
        };
        component.setParameter = (name: string, value: unknown) => {
          component.params.set(name, value);
          component.sets.push(name);
        };
        this.gsplat = component;
      }
    }
    setPosition = (): void => undefined;
    setLocalPosition = (): void => undefined;
    setLocalRotation = (): void => undefined;
    lookAt = (): void => undefined;
    destroy(): void {
      this.parent?.removeChild(this);
    }
  }
  class Application extends Handlers {
    readonly graphicsDevice = { isWebGPU: false, maxPixelRatio: 1, gl: null, width: 1000 };
    readonly scene = Object.assign(new Handlers(), {
      gsplat: { material: pc.material, format: pc.format, varyings: pc.varyings },
    });
    readonly systems = { gsplat: new Handlers() };
    readonly assets = {
      add: (): void => undefined,
      remove: (): void => undefined,
      load: (asset: Asset) => asset.loaded(),
    };
    readonly root = {
      addChild: (e: Entity) => {
        e.parent = this.root;
      },
      removeChild: (e: Entity) => {
        e.parent = null;
      },
    };
    autoRender = true;
    requestAnimationFrame = (): void => undefined;
    setCanvasFillMode = (): void => undefined;
    setCanvasResolution = (): void => undefined;
    resizeCanvas = (): void => undefined;
    start = (): void => undefined;
    render = (): void => undefined;
    destroy = (): void => undefined;
  }
  /** A streamed scan's asset (`streamNative`): loads at once. */
  class Asset {
    readonly resource = { aabb: { center: { x: 0, y: 0, z: 0 } } };
    #load: (() => void) | null = null;
    once(name: string, handler: () => void): void {
      if (name === "load") this.#load = handler;
    }
    loaded(): void {
      this.#load?.();
    }
    unload = (): void => undefined;
  }
  class Texture {
    readonly width: number;
    readonly height: number;
    readonly format: number;
    data: Uint8Array | Float32Array | Uint32Array;
    destroyed = false;
    constructor(_device: unknown, options: { width: number; height: number; format: number }) {
      this.width = options.width;
      this.height = options.height;
      this.format = options.format;
      const texels = options.width * options.height;
      this.data =
        options.format === 2
          ? new Uint8Array(texels * 2)
          : options.format === 3
            ? new Float32Array(texels * 4)
            : options.format === 4
              ? new Uint32Array(texels * 4)
              : new Uint8Array(texels * 4);
    }
    lock(): Uint8Array | Float32Array | Uint32Array {
      return this.data;
    }
    unlock = (): void => undefined;
    destroy(): void {
      this.destroyed = true;
    }
  }
  class GSplatResource {
    readonly format = { addExtraStreams: (): void => undefined };
    getTexture = (): null => null;
    destroy = (): void => undefined;
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
    Asset,
    Entity,
    GSplatResource,
    GSplatData,
    Texture,
    Vec3,
    Color,
    FILLMODE_NONE: "NONE",
    RESOLUTION_AUTO: "AUTO",
    ASPECT_AUTO: 0,
    PIXELFORMAT_RGBA8: 1,
    PIXELFORMAT_RG8: 2,
    PIXELFORMAT_RGBA32F: 3,
    PIXELFORMAT_RGBA32U: 4,
    PIXELFORMAT_R32U: 5,
    PIXELFORMAT_R8: 6,
    PIXELFORMAT_R8U: 7,
    TYPE_UINT32: 8,
    FILTER_NEAREST: 0,
    ADDRESS_CLAMP_TO_EDGE: 0,
    Quat: { IDENTITY: {} },
    SHADERLANGUAGE_GLSL: "glsl",
  };
});

/** Tiles named `x_y.glb` decode to two splats around the centre (x, y, 0) of the scan frame. */
class FakeWorker {
  onmessage: ((event: { data: unknown }) => void) | null = null;
  postMessage(message: { id: number; url: string }): void {
    const [x = 0, y = 0] = (/\/(-?\d+)_(-?\d+)\.glb$/.exec(message.url) ?? []).slice(1).map(Number);
    queueMicrotask(() =>
      this.onmessage?.({
        data: {
          id: message.id,
          count: 2,
          properties: {
            x: Float32Array.of(-1, 1),
            y: Float32Array.of(0, 0),
            z: Float32Array.of(0, 0),
            scale_0: Float32Array.of(0.1, 0.1),
            scale_1: Float32Array.of(0.1, 0.1),
            scale_2: Float32Array.of(0.1, 0.1),
            opacity: Float32Array.of(1, 1),
          },
          order: Uint32Array.of(0, 1),
          origin: [x, y, 0],
          checksum: `c${String(x)}_${String(y)}`,
        },
      }),
    );
  }
  terminate = (): void => undefined;
}

const pose: ScanPose = {
  eye: [0, -5, 2],
  direction: [0, 1, 0],
  up: [0, 0, 1],
  fovy: Math.PI / 3,
  near: 0.1,
  far: 1000,
  width: 960,
  height: 600,
  pixelRatio: 1,
};

const tile = (uri: string) => ({ uri }) as unknown as TileNode;
const SCAN = "https://scan.test/june/tileset.json";
const MARCH = "https://scan.test/march/tileset.json";
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const LOOK: LayerLook = { highlight: false, cones: null, fromScan: IDENTITY };

async function backendFor(assetId: string) {
  const { createBackend } = await import("@/cesium/scanView/playcanvasBackend");
  let wanted = 0;
  const hooks: BackendHooks = {
    frameWanted: () => {
      wanted += 1;
    },
    work: { run: (task: () => unknown) => Promise.resolve(task()) } as unknown as TileWork,
    maxShDegree: 0,
    assetId,
  };
  const backend = await createBackend(document.createElement("canvas"), 1_000_000, hooks);
  return { backend, wanted: () => wanted };
}

const component = (entity: unknown): FakeComponent => (entity as { gsplat: FakeComponent }).gsplat;

/** March, left of a divider at `split` of the width, beside June (the scan). */
function view(split = 0.4): ScanCompareView {
  return {
    left: { url: MARCH, assetId: "march", extras: {}, placement: null, root: IDENTITY, owns: [] },
    split,
    toWorld: () => undefined,
  };
}

/** Serves March's tileset: one tile, `9_1.glb`. */
function serveMarch(): string[] {
  const fetched: string[] = [];
  vi.stubGlobal("fetch", (url: string) => {
    fetched.push(url);
    return Promise.resolve({
      ok: true,
      json: () =>
        Promise.resolve({
          root: { geometricError: 0, content: { uri: "9_1.glb" }, extras: { gaussians: 2 } },
        }),
    });
  });
  return fetched;
}

/** Draws frames until March's tile is in the app. */
async function marchTile(backend: { render(pose: ScanPose): void }) {
  let march: (typeof pc.entities)[number] | undefined;
  for (let i = 0; i < 100 && !march?.gsplat; i += 1) {
    backend.render(pose);
    await new Promise((r) => setTimeout(r, 10));
    march = pc.entities.find((e) => e.name === "9_1.glb");
  }
  return march;
}

describe("two dates compared under PlayCanvas", () => {
  beforeEach(() => {
    vi.stubGlobal("Worker", FakeWorker);
    pc.entities.length = 0;
    pc.material.chunks.set("gsplatModifyVS", "default modify");
    pc.material.chunks.set("gsplatModifyPS", "default fragment");
    pc.material.params.clear();
    pc.varyings.names = [];
    pc.format.streams.length = 0;
    // The comparison's session lends the renderer the globe's streaming classes as it starts.
    lendRuleKit(RULE_KIT);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    for (const id of ["june-a", "june-b", "june-c", "june-d"]) {
      publishScanCompare(id, null);
      publishScanSun(id, null);
    }
  });

  it("changes nothing while one date is drawn, and keeps every tile writing its side after one", async () => {
    serveMarch();
    const { backend } = await backendFor("june-a");
    const own = await backend.load(SCAN, tile("1_1.glb"));
    backend.add(own);
    backend.render(pose);
    expect(pc.format.streams).toEqual([]);
    expect(component(own).modifier ?? null).toBeNull();
    expect(pc.material.chunks.get("gsplatModifyVS")).toBe("default modify");
    expect(pc.material.chunks.get("gsplatModifyPS")).toBe("default fragment");
    expect(pc.varyings.names).toEqual([]);
    expect(scanCompareReading("june-a")).toMatchObject({ on: false });
    // The first comparison adds the side's stream; it is kept, and a copy that left it unwritten
    // would draw nothing, so every tile writes its side from then on: none, with no comparison.
    publishScanCompare("june-a", view());
    expect(pc.format.streams).toEqual([{ name: SIDE_STREAM, format: 7 }]);
    publishScanCompare("june-a", null);
    backend.render(pose);
    expect(pc.format.streams).toHaveLength(1);
    expect(component(own).modifier?.glsl).toContain(SIDE_WRITE_STEP);
    expect(component(own).params.get("uHexapodSide")).toBe(0);
    const later = await backend.load(SCAN, tile("3_3.glb"));
    expect(component(later).modifier?.glsl).toContain(SIDE_WRITE_STEP);
    expect(component(later).params.get("uHexapodSide")).toBe(0);
    backend.destroy();
  });

  it("streams the left date beside the scan, each tile's copy writing its date's side", async () => {
    const fetched = serveMarch();
    const { backend } = await backendFor("june-b");
    const own = await backend.load(SCAN, tile("1_1.glb"));
    backend.add(own);
    const fill = await backend.loadLayer?.(SCAN, tile("2_2.glb"));
    if (!fill) throw new Error("PlayCanvas draws layers");
    backend.setLayerLook?.(fill, LOOK);
    publishScanCompare("june-b", view());
    await vi.waitFor(() => expect(fetched).toContain(MARCH), { timeout: 20_000 });
    const march = await marchTile(backend);
    // June's own tiles and its layer right of the divider, March's left of it.
    expect(component(own).modifier?.glsl).toContain(SIDE_WRITE_STEP);
    expect(component(own).params.get("uHexapodSide")).toBe(2);
    expect(component(fill).modifier?.glsl).toContain(SIDE_WRITE_STEP);
    expect(component(fill).params.get("uHexapodSide")).toBe(2);
    expect(component(march).modifier?.glsl).toContain(SIDE_WRITE_STEP);
    expect(component(march).params.get("uHexapodSide")).toBe(1);
    // Bound to none of the scan's objects.
    expect(backend.instanceTiles?.().tiles).toBe(1);
    // One budget for both: the scan streams within half of it, the left date the other half.
    expect(backend.loadFactor).toBe(0.5);
    expect(scanCompareReading("june-b")).toMatchObject({
      on: true,
      dividerPx: 400,
      sides: { left: 1, right: 2 },
    });
    // Ended: every tile drawn whole again (no side), the left date let go.
    publishScanCompare("june-b", null);
    backend.render(pose);
    expect(component(own).params.get("uHexapodSide")).toBe(0);
    expect(component(fill).params.get("uHexapodSide")).toBe(0);
    expect(scanCompareReading("june-b")).toMatchObject({ on: false });
    expect(scanCompareReading("june-b")?.left.dates).toBe(0);
    expect(backend.loadFactor).toBe(1);
    backend.destroy();
  });

  it("cuts at the divider in the frame's hook, and moving it copies nothing", async () => {
    serveMarch();
    const { backend } = await backendFor("june-c");
    const own = await backend.load(SCAN, tile("1_1.glb"));
    backend.add(own);
    publishScanCompare("june-c", view(0.4));
    await marchTile(backend);
    backend.render(pose);
    expect(pc.varyings.names).toEqual([SIDE_VARYING]);
    expect(pc.material.chunks.get("gsplatModifyPS")).toBe(COMPARE_HOOK_PS);
    const vs = pc.material.chunks.get("gsplatModifyVS") ?? "";
    expect(vs).toContain("setHexapodHalf(loadHexapodSide().r);");
    expect(pc.material.params.get("uHexapodDivider")).toEqual([400, 1, 0, 0]);
    // Swiped: one uniform of the frame; no tile is given anything (nothing copied again).
    const before = pc.entities.map((e) => e.gsplat?.sets.length ?? 0);
    for (const split of [0.25, 0.5, 0.75]) {
      publishScanCompare("june-c", view(split));
      backend.render(pose);
      expect(pc.material.params.get("uHexapodDivider")).toEqual([split * 1000, 1, 0, 0]);
    }
    expect(pc.entities.map((e) => e.gsplat?.sets.length ?? 0)).toEqual(before);
    // Ended: PlayCanvas's own chunks back, the varying gone.
    publishScanCompare("june-c", null);
    backend.render(pose);
    expect(pc.material.chunks.get("gsplatModifyVS")).toBe("default modify");
    expect(pc.material.chunks.get("gsplatModifyPS")).toBe("default fragment");
    expect(pc.varyings.names).toEqual([]);
    backend.destroy();
  });

  it("keeps the sunlight's relight in the vertex hook beside the side, and back after", async () => {
    const { backend } = await backendFor("june-d");
    const own = await backend.load(SCAN, tile("1_1.glb"));
    backend.add(own);
    publishScanSun("june-d", {
      dir: [0, 0, 1],
      sunUp: 1,
      strength: 1,
      ambient: 0.35,
      flat: 0.25,
    } as never);
    backend.render(pose);
    publishScanCompare("june-d", view());
    backend.render(pose);
    const vs = pc.material.chunks.get("gsplatModifyVS") ?? "";
    expect(vs).toContain("loadHexapodSun().r");
    expect(vs).toContain("hexapodInnerColor(center, color);");
    expect(vs.match(/void modifySplatColor\(/g)).toHaveLength(1);
    expect(pc.material.params.get("uHexapodSun")).toEqual([0, 0, 1, 1]);
    publishScanCompare("june-d", null);
    backend.render(pose);
    expect(pc.material.chunks.get("gsplatModifyVS")).toContain("uHexapodSunlight");
    expect(pc.material.chunks.get("gsplatModifyVS")).not.toContain("hexapodInnerColor");
    expect(pc.material.chunks.get("gsplatModifyPS")).toBe("default fragment");
    backend.destroy();
  });

  it("has scene selection pick each date only on its side of the divider", async () => {
    serveMarch();
    const { backend } = await backendFor("june-a");
    const own = await backend.load(SCAN, tile("1_1.glb"));
    backend.add(own);
    publishScanCompare("june-a", view());
    await marchTile(backend);
    for (let i = 0; i < 20 && (pickSourceOf("march")?.tiles().length ?? 0) === 0; i += 1) {
      await new Promise((r) => setTimeout(r, 20));
      backend.render(pose);
    }
    const right = pickSourceOf("june-a");
    const left = pickSourceOf("march");
    expect(right?.split?.()).toBe(SplitDirection.RIGHT);
    expect(right?.tiles()).toHaveLength(1);
    expect(left?.split?.()).toBe(SplitDirection.LEFT);
    expect(left?.tiles()).toHaveLength(1);
    expect(left?.tiles()[0]?.checksum).toBe("c9_1");
    // Seen from afar, neither is selected.
    backend.render({ ...pose, farView: true });
    expect(left?.tiles()).toHaveLength(0);
    publishScanCompare("june-a", null);
    expect(pickSourceOf("march")).toBeUndefined();
    backend.destroy();
  });
});
