/**
 * Sunlight under PlayCanvas (playcanvasBackend.ts, scanView/sunHook.ts): nothing while the sun
 * is off; the first sun adds the word stream to the work buffer and puts on every tile a
 * modifier that writes it (an inferred layer's with the skip bit), and the frame's hook goes
 * into the user material with the sun as uniforms -- set per frame without a tile copied again
 * -- and comes out when the sun goes. Under Confidence or Data age the copy pass keeps in the
 * word the look of a black splat, so the hook relights in CesiumJS's order (sun, then look). PlayCanvas is faked (no WebGL in jsdom): what matters is
 * what each tile and the material are given, and when. e2e/sunlightScene.spec.ts draws it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { encodeConfidenceGrid } from "@/dev/cellGridFixture";
import { cellTextureRows, confidenceTexels, decodeCellGrid } from "@/lib/cellGrid";
import type { CellLookView } from "@/lib/cellLook";
import { publishScanCellView } from "@/cesium/scanView/scanCellView";
import type { LayerLook } from "@/cesium/scanView/layerLook";
import {
  publishScanSun,
  scanSunDrawn,
  scanSunReading,
  type ScanSun,
} from "@/cesium/scanView/scanSunlight";
import { SUN_HOOK_VS, SUN_STREAM } from "@/cesium/scanView/sunHook";
import type { BackendHooks, ScanPose } from "@/cesium/scanView/types";
import type { TileWork } from "@/cesium/scanView/tileWork";
import type { TileNode } from "@/view/tiles";

interface FakeComponent {
  modifier: { glsl: string } | null | undefined;
  params: Map<string, unknown>;
  sets: string[];
}

const pc = vi.hoisted(() => {
  const chunks = new Map<string, string>([["gsplatModifyVS", "default modify"]]);
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
    readonly graphicsDevice = { isWebGPU: false, maxPixelRatio: 1, gl: null };
    readonly scene = Object.assign(new Handlers(), {
      gsplat: { material: pc.material, format: pc.format },
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
const URL_ = "https://scan.test/tileset.json";
const MORNING: ScanSun = { dir: [0.87, 0.07, 0.48], sunUp: 1, strength: 1 };
const EVENING: ScanSun = { dir: [-0.87, 0.06, 0.5], sunUp: 1, strength: 1 };

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
const LOOK: LayerLook = {
  highlight: false,
  cones: null,
  fromScan: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
};

describe("Sunlight under PlayCanvas", () => {
  beforeEach(() => {
    vi.stubGlobal("Worker", FakeWorker);
    pc.entities.length = 0;
    pc.format.streams.length = 0;
    pc.material.chunks.set("gsplatModifyVS", "default modify");
    pc.material.params.clear();
    pc.material.updates = 0;
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    for (const id of ["scan-a", "scan-b", "scan-c"]) publishScanSun(id, null);
  });

  it("adds nothing and changes nothing while the sun is off", async () => {
    const { backend } = await backendFor("scan-a");
    const near = await backend.load(URL_, tile("0_0.glb"));
    backend.add(near);
    backend.render(pose);
    expect(component(near).modifier ?? null).toBeNull();
    expect(pc.format.streams).toEqual([]);
    expect(pc.material.chunks.get("gsplatModifyVS")).toBe("default modify");
    expect(pc.material.updates).toBe(0);
    // It draws the scan, and it can relight: the relight's report counts it as drawn.
    expect(scanSunDrawn("scan-a")).toBe(true);
    expect(scanSunReading("scan-a")).toMatchObject({
      renderer: "playcanvas",
      installed: false,
      stream: false,
      sun: null,
    });
    backend.destroy();
    expect(scanSunDrawn("scan-a")).toBe(false);
    expect(scanSunReading("scan-a")).toBeNull();
  });

  it("lights every scan tile by the frame's hook, an inferred layer's not at all", async () => {
    const { backend, wanted } = await backendFor("scan-b");
    const near = await backend.load(URL_, tile("0_0.glb"));
    const layer = await backend.loadLayer?.(URL_, tile("3_3.glb"));
    if (!layer) throw new Error("no layers");
    backend.add(near);
    backend.add(layer);
    backend.setLayerLook?.(layer, LOOK);
    backend.render(pose);

    const before = wanted();
    publishScanSun("scan-b", MORNING);
    expect(wanted()).toBeGreaterThan(before);
    // The word each splat carries: one R32U stream of the work buffer, added once.
    expect(pc.format.streams).toEqual([{ name: SUN_STREAM, format: 5 }]);
    // Every scan tile's copy writes it -- one with no objects and no view too -- and a layer's
    // writes the skip bit.
    const tileGlsl = component(near).modifier?.glsl ?? "";
    expect(tileGlsl).toContain("writeHexapodSun(uvec4(0u, 0u, 0u, 0u));");
    expect(tileGlsl).not.toContain("hexapodInstanceColor(0u");
    expect(component(near).sets.at(-1)).toBe("uInstanceParams");
    const layerGlsl = component(layer).modifier?.glsl ?? "";
    expect(layerGlsl).toContain("writeHexapodSun(uvec4(HEXAPOD_SUN_SKIP, 0u, 0u, 0u));");
    expect(layerGlsl).toContain("color = hexapodLayerColor(center, color");

    // The frame's hook, with the sun as its uniforms.
    backend.render(pose);
    expect(pc.material.chunks.get("gsplatModifyVS")).toBe(SUN_HOOK_VS);
    expect(pc.material.params.get("uHexapodSun")).toEqual(
      [0.87, 0.07, 0.48, 1].map((v) => Math.fround(v)),
    );
    expect(pc.material.params.get("uHexapodSunlight")).toEqual([1, 0, 0, 0]);
    expect(scanSunReading("scan-b")).toMatchObject({ installed: true, stream: true, frames: 1 });

    // The sun moves: uniforms only, no tile copied again.
    component(near).sets.length = 0;
    component(layer).sets.length = 0;
    const updates = pc.material.updates;
    publishScanSun("scan-b", EVENING);
    backend.render(pose);
    expect(pc.material.params.get("uHexapodSun")?.[0]).toBeCloseTo(-0.87, 6);
    expect(pc.material.updates).toBe(updates + 1);
    expect(component(near).sets).toEqual([]);
    expect(component(layer).sets).toEqual([]);
    // A frame with the same sun sets nothing.
    backend.render(pose);
    expect(pc.material.updates).toBe(updates + 1);

    // A tile loaded later writes the word too.
    const late = await backend.load(URL_, tile("40_0.glb"));
    expect(component(late).modifier?.glsl).toContain("writeHexapodSun(");

    // Off: the hook comes out, PlayCanvas's own chunk back; the tiles keep writing the word
    // (a stream is not taken back), which nothing reads.
    publishScanSun("scan-b", null);
    backend.render(pose);
    expect(pc.material.chunks.get("gsplatModifyVS")).toBe("default modify");
    expect(scanSunReading("scan-b")).toMatchObject({ installed: false, stream: true, sun: null });
    // On again: the same stream, the hook back in.
    publishScanSun("scan-b", MORNING);
    backend.render(pose);
    expect(pc.format.streams).toHaveLength(1);
    expect(pc.material.chunks.get("gsplatModifyVS")).toBe(SUN_HOOK_VS);
    backend.destroy();
  });

  it("draws the photograph for a sun of strength 0, the hook in", async () => {
    const { backend } = await backendFor("scan-c");
    publishScanSun("scan-c", { dir: [0, 0, 1], sunUp: 1, strength: 0 });
    const near = await backend.load(URL_, tile("0_0.glb"));
    backend.add(near);
    backend.render(pose);
    // Already on when the renderer started: the stream from the start, the tiles writing it.
    expect(pc.format.streams).toHaveLength(1);
    expect(component(near).modifier?.glsl).toContain("writeHexapodSun(");
    expect(pc.material.params.get("uHexapodSunlight")).toEqual([0, 0, 0, 0]);
    backend.destroy();
  });

  it("has a natively streamed scan write the word too, when the sun was on first", async () => {
    const { backend } = await backendFor("scan-c");
    publishScanSun("scan-c", MORNING);
    const stream = await backend.streamNative?.("https://scan.test/lod-meta.json");
    const entity = pc.entities.find((e) => e.name === "scan");
    expect(entity?.gsplat?.modifier?.glsl).toContain("writeHexapodSun(uvec4(0u, 0u, 0u, 0u));");
    stream?.stop();
    backend.destroy();
  });

  it("keeps the look's black in the word under a cell view, so the sun comes before the look", async () => {
    const { meta, planes } = await encodeConfidenceGrid({
      level: 2,
      i0: 0,
      j0: 0,
      ni: 8,
      nj: 8,
      cell: () => ({ score: 240 }),
    });
    const grid = await decodeCellGrid(meta, planes);
    const view: CellLookView = {
      mode: "confidence",
      rows: cellTextureRows(grid.meta, confidenceTexels(grid)),
      meta: grid.meta,
    };
    const { backend } = await backendFor("scan-c");
    const near = await backend.load(URL_, tile("0_0.glb"));
    backend.add(near);
    publishScanSun("scan-c", MORNING);
    publishScanCellView("scan-c", view);
    const glsl = component(near).modifier?.glsl ?? "";
    // The word: the look drawn for a black splat of the same alpha, then the look itself.
    const word = glsl.indexOf(
      "writeHexapodSun(uvec4(hexapodSunWord(hexapodCellTile(center, vec4(0.0, 0.0, 0.0, color.a), uCellGrid, uCellRule, uCellLevel, uCellTile).rgb), 0u, 0u, 0u));",
    );
    const look = glsl.indexOf(
      "color = hexapodCellTile(center, color, uCellGrid, uCellRule, uCellLevel, uCellTile);",
    );
    expect(word).toBeGreaterThan(0);
    expect(look).toBeGreaterThan(word);
    // No other word written: one per splat.
    expect(glsl.split("writeHexapodSun(").length).toBe(2);
    // Photo again: the plain word (no look, nothing to undo).
    publishScanCellView("scan-c", null);
    expect(component(near).modifier?.glsl).toContain("writeHexapodSun(uvec4(0u, 0u, 0u, 0u));");
    backend.destroy();
  });
});
