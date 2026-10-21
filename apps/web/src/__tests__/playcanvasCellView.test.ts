/**
 * Confidence and Data age under PlayCanvas (playcanvasBackend.ts, scanView/cellModifier.ts):
 * every tile of the scan gets the shared cell look in its work-buffer modifier, reading the
 * view's texture and the grid's own rule, at one level per tile picked from the eye -- and a
 * tile is copied again (a parameter set) only when its level steps. An inferred layer's tiles
 * go grey. Photo takes the look off again. PlayCanvas is faked (no WebGL in jsdom): what
 * matters is which modifier and which parameters each tile is given, and when.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CELL_TILE_GLSL } from "@/cesium/scanView/cellModifier";
import type { LayerLook } from "@/cesium/scanView/layerLook";
import type { BackendHooks, ScanPose } from "@/cesium/scanView/types";
import type { TileWork } from "@/cesium/scanView/tileWork";
import { encodeConfidenceGrid, REASON_BITS } from "@/dev/cellGridFixture";
import { cellTextureRows, confidenceTexels, decodeCellGrid } from "@/lib/cellGrid";
import { cellLevelIndex, pxPerRadianOf, type CellLookView } from "@/lib/cellLook";
import type { TileNode } from "@/view/tiles";

interface FakeComponent {
  modifier: { glsl: string } | null | undefined;
  params: Map<string, unknown>;
  sets: string[];
}

const pc = vi.hoisted(() => ({ entities: [] as { name: string; gsplat?: FakeComponent }[] }));

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
    readonly scene = Object.assign(new Handlers(), { gsplat: {} });
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
  /** A streamed package (lod-meta.json): its octree's bounds, the yard's, in the scan frame. */
  class Asset {
    readonly resource = { aabb: { center: { x: 16, y: 12, z: 2 } } };
    readonly handlers = new Map<string, () => void>();
    once(event: string, handler: () => void): void {
      this.handlers.set(event, handler);
    }
    loaded(): void {
      this.handlers.get("load")?.();
    }
    unload = (): void => undefined;
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

const pose = (eye: [number, number, number]): ScanPose => ({
  eye,
  direction: [0, 1, 0],
  up: [0, 0, 1],
  fovy: Math.PI / 3,
  near: 0.1,
  far: 1000,
  width: 960,
  height: 600,
  pixelRatio: 0.6,
  // The level is measured as CesiumJS measures it: in the globe's pixels, not the overlay's.
  globePixelRatio: 1,
});

const tile = (uri: string) => ({ uri }) as unknown as TileNode;
const URL_ = "https://scan.test/tileset.json";

async function view(): Promise<CellLookView> {
  const { meta, planes } = await encodeConfidenceGrid({
    level: 2,
    i0: 0,
    j0: 0,
    ni: 32,
    nj: 24,
    cell: (i) => (i < 16 ? { score: 60, reasons: REASON_BITS.views } : { score: 240 }),
    coarser: 8,
  });
  const grid = await decodeCellGrid({ ...meta, weakBelow: 100 }, planes);
  return {
    mode: "confidence",
    rows: cellTextureRows(grid.meta, confidenceTexels(grid)),
    meta: grid.meta,
  };
}

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

describe("Confidence and Data age under PlayCanvas", () => {
  beforeEach(() => {
    vi.stubGlobal("Worker", FakeWorker);
    pc.entities.length = 0;
  });
  afterEach(() => vi.unstubAllGlobals());

  it("draws every scan tile with the shared look, a level a tile, re-copied only on a step", async () => {
    const { publishScanCellView, scanCellViewReading } =
      await import("@/cesium/scanView/scanCellView");
    const { backend, wanted } = await backendFor("scan-a");
    const near = await backend.load(URL_, tile("0_0.glb"));
    const far = await backend.load(URL_, tile("0_400.glb"));
    backend.add(near);
    backend.add(far);
    backend.render(pose([0, -5, 2]));
    // Photo: no modifier on a tile with no objects and no motion.
    expect(component(near).modifier ?? null).toBeNull();

    const v = await view();
    const before = wanted();
    publishScanCellView("scan-a", v);
    expect(wanted()).toBeGreaterThan(before);
    for (const entity of [near, far]) {
      const c = component(entity);
      expect(c.modifier?.glsl).toContain(CELL_TILE_GLSL);
      expect(c.modifier?.glsl).toContain("hexapodCellTile(center, color, uCellGrid");
      const grid = c.params.get("uCellGrid") as { width: number; height: number; data: Uint8Array };
      expect(grid.width).toBe(4096);
      expect(Array.from(grid.data.subarray(0, 4))).toEqual(Array.from(v.rows.data.subarray(0, 4)));
      // The grid's own threshold, not weak-v1's default.
      expect(Array.from(c.params.get("uCellRule") as Float32Array)).toEqual([
        100,
        REASON_BITS.unseen,
        Math.fround(0.6),
        0,
      ]);
    }
    // Each tile reads the level CesiumJS's rule gives its centre.
    const ppr = pxPerRadianOf(600 * 1, Math.PI / 3);
    const level = (entity: unknown) =>
      (component(entity).params.get("uCellTile") as Float32Array)[3];
    expect(level(near)).toBe(cellLevelIndex(v.meta, Math.hypot(5, 2), ppr));
    expect(level(far)).toBe(cellLevelIndex(v.meta, Math.hypot(405, 2), ppr));
    expect(level(far)).toBeGreaterThan(level(near) ?? 0);
    expect((component(near).params.get("uCellTile") as Float32Array)[0]).toBe(1);

    // A small move steps neither: nothing is copied again.
    for (const entity of [near, far]) component(entity).sets.length = 0;
    backend.render(pose([0, -5.1, 2]));
    expect(component(near).sets).toEqual([]);
    expect(component(far).sets).toEqual([]);
    // Further off, the near tile steps coarser and the far one does not: only it is copied again.
    backend.render(pose([0, -100, 2]));
    expect(component(near).sets).toEqual(["uCellLevel", "uCellTile"]);
    expect(component(far).sets).toEqual([]);
    expect(level(near)).toBe(cellLevelIndex(v.meta, Math.hypot(100, 2), ppr));
    expect(level(near)).toBe(1);

    const reading = scanCellViewReading("scan-a");
    expect(reading).toMatchObject({
      renderer: "playcanvas",
      mode: "confidence",
      tiles: 2,
      steps: 1,
    });
    expect(reading?.sampled).toBe(4);

    // Photo again: the look comes off, and the tiles are drawn as decoded.
    publishScanCellView("scan-a", null);
    expect(component(near).modifier ?? null).toBeNull();
    expect(component(far).modifier ?? null).toBeNull();
    backend.destroy();
    expect(scanCellViewReading("scan-a")).toBeNull();
  });

  it("draws a natively streamed scan with the look too, one level for all of it", async () => {
    const { publishScanCellView, scanCellViewReading } =
      await import("@/cesium/scanView/scanCellView");
    const { backend } = await backendFor("scan-native");
    const v = await view();
    publishScanCellView("scan-native", v);
    backend.render(pose([16, -40, 30]));
    const stream = await backend.streamNative?.("https://scan.test/sog/lod-meta.json");
    if (!stream) throw new Error("no native streaming");
    const scan = pc.entities.find((e) => e.name === "scan");
    const c = component(scan);
    // The streamed scan is one entity, drawn with the shared look like any tile.
    expect(c.modifier?.glsl).toContain(CELL_TILE_GLSL);
    expect(c.params.get("uCellRule")).toBeDefined();
    // Its level: CesiumJS's rule at the middle of the package's bounds.
    const ppr = pxPerRadianOf(600, Math.PI / 3);
    const level = () => (c.params.get("uCellTile") as Float32Array)[3];
    expect(level()).toBe(cellLevelIndex(v.meta, Math.hypot(0, 52, 28), ppr));
    // From much further, the whole scan steps coarser at once.
    backend.render(pose([16, -1500, 30]));
    expect(level()).toBe(cellLevelIndex(v.meta, Math.hypot(0, 1512, 28), ppr));
    expect(level()).toBeGreaterThan(cellLevelIndex(v.meta, Math.hypot(0, 52, 28), ppr));
    expect(scanCellViewReading("scan-native")).toMatchObject({ tiles: 1, steps: 1 });
    // Photo: the look comes off; stopped, the scan is no longer one of the view's tiles.
    publishScanCellView("scan-native", null);
    expect(c.modifier ?? null).toBeNull();
    stream.stop();
    publishScanCellView("scan-native", v);
    expect(scanCellViewReading("scan-native")).toMatchObject({ tiles: 0 });
    backend.destroy();
  });

  it("draws an inferred layer's splats grey while a view is drawn: no measured colour", async () => {
    const { publishScanCellView } = await import("@/cesium/scanView/scanCellView");
    const { backend } = await backendFor("scan-b");
    const layer = await backend.loadLayer?.(URL_, tile("3_3.glb"));
    if (!layer) throw new Error("no layers");
    const look: LayerLook = {
      highlight: false,
      cones: null,
      fromScan: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    };
    backend.setLayerLook?.(layer, look);
    const c = component(layer);
    expect(c.modifier?.glsl).toContain("hexapodCellGrey(color)");
    expect(Array.from((c.params.get("uCellFill") as Float32Array | undefined) ?? [0])[0]).toBe(0);
    publishScanCellView("scan-b", await view());
    expect(Array.from(c.params.get("uCellFill") as Float32Array)[0]).toBe(1);
    publishScanCellView("scan-b", null);
    expect(Array.from(c.params.get("uCellFill") as Float32Array)[0]).toBe(0);
    backend.destroy();
  });
});
