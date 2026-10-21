/**
 * A per-cell view rule under PlayCanvas (playcanvasBackend.ts, scanView/scanRule.ts): with a
 * rule published for the scan it draws, every tile -- the scan's, an inferred layer's, a split
 * object's -- gets the rule's weight in its work-buffer modifier, its source read from the
 * tileset it came from; a tile of no listed source draws nothing; a split object is weighed
 * whole by its centre; the inferred sources the rule tints get Highlight's look; Data age is
 * drawn from the rule's own owner-age view; the rule off puts every tile back. PlayCanvas is
 * faked (no WebGL in jsdom): what matters is what each tile is given. That each cell's pixels
 * come from one date is e2e/compositorScene.spec.ts's, under PlayCanvas too.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { LayerLook } from "@/cesium/scanView/layerLook";
import { RULE_KIT } from "@/cesium/scanView/ruleKit";
import {
  lendRuleKit,
  PLAYCANVAS_FOOTPRINT,
  publishScanRule,
  RULE_WEIGHT_GLSL,
  ruleOwnerRows,
  ruleSigma,
  scanRuleReading,
  type ScanRuleView,
} from "@/cesium/scanView/scanRule";
import type { BackendHooks, ScanPose } from "@/cesium/scanView/types";
import type { TileWork } from "@/cesium/scanView/tileWork";
import { encodeConfidenceGrid } from "@/dev/cellGridFixture";
import { cellTextureRows, confidenceTexels, decodeCellGrid } from "@/lib/cellGrid";
import { NO_OWNER, ownerPlane } from "@/lib/compositor";
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
  /** Enough of a matrix for `place`: kept, and points moved by its translation. */
  class Mat4 {
    data: number[] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    set(values: number[]): this {
      this.data = [...values];
      return this;
    }
    copy(other: Mat4): this {
      this.data = [...other.data];
      return this;
    }
    transformPoint(point: Vec3, out: Vec3): Vec3 {
      return out;
    }
  }
  class Quat {
    static readonly IDENTITY = new Quat();
    setFromMat4(): this {
      return this;
    }
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
    FILTER_NEAREST: 0,
    ADDRESS_CLAMP_TO_EDGE: 0,
    Quat,
    Mat4,
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
const SCAN = "https://scan.test/march/tileset.json";
const LAYER = "https://scan.test/march/fill/tileset.json";
const OBJECT = "https://scan.test/march/objects/7/tileset.json";
const OTHER = "https://scan.test/elsewhere/tileset.json";
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

const LOOK: LayerLook = {
  highlight: false,
  cones: null,
  fromScan: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
};

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
const param = (entity: unknown, name: string): number[] =>
  Array.from((component(entity).params.get(name) as Float32Array | undefined) ?? []);

/** March (source 0) west of x = 4 m, June (1) east; a fill (2) where neither saw. */
function view(overrides: Partial<ScanRuleView> = {}): ScanRuleView {
  const plane = ownerPlane(
    "latest",
    [
      {
        source: 0,
        epochId: "march",
        observedAt: "2026-03-01T00:00:00Z",
        inferred: false,
        boxes: [{ minX: 0, minY: 0, maxX: 4, maxY: 4 }],
      },
      {
        source: 1,
        epochId: "june",
        observedAt: "2026-06-01T00:00:00Z",
        inferred: false,
        boxes: [{ minX: 4, minY: 0, maxX: 8, maxY: 4 }],
      },
    ],
    null,
  );
  return {
    plane: [plane.i0, plane.j0, plane.ni, plane.nj],
    cellM: plane.cellM,
    owners: ruleOwnerRows(plane),
    sources: new Map([
      [SCAN, { source: 0 }],
      [LAYER, { source: 2 }],
      [OBJECT, { source: 0, whole: true }],
    ]),
    native: 0,
    tinted: 0,
    reach: 2,
    enabled: true,
    dates: [],
    age: null,
    ...overrides,
  };
}

describe("a per-cell view rule under PlayCanvas", () => {
  beforeEach(() => {
    vi.stubGlobal("Worker", FakeWorker);
    pc.entities.length = 0;
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    for (const id of ["rule-a", "rule-b", "rule-c", "rule-d"]) publishScanRule(id, null);
  });

  it("keeps each splat only where its tile's source owns the cell, in the copy pass", async () => {
    const { backend, wanted } = await backendFor("rule-a");
    const scan = await backend.load(SCAN, tile("1_1.glb"));
    const stray = await backend.load(OTHER, tile("2_2.glb"));
    const layer = await backend.loadLayer?.(LAYER, tile("3_3.glb"));
    const object = await backend.load(OBJECT, tile("5_1.glb"));
    if (!layer) throw new Error("no layers");
    for (const e of [scan, stray, layer, object]) backend.add(e);
    backend.setLayerLook?.(layer, LOOK);
    backend.render(pose);
    // No rule: nothing of it in any modifier.
    expect(component(scan).modifier ?? null).toBeNull();
    expect(component(layer).modifier?.glsl).not.toContain("hexapodRuleWeight");

    const before = wanted();
    publishScanRule("rule-a", view());
    expect(wanted()).toBeGreaterThan(before);
    for (const e of [scan, stray, layer, object]) {
      const glsl = component(e).modifier?.glsl ?? "";
      expect(glsl).toContain("color.a *= hexapodRuleWeight(center);");
      // The spread is read after the motion has moved the splat.
      expect(glsl).toContain("hexapodRuleRotation = rotation;");
      // The seam's reach in PlayCanvas's own footprints: it draws a splat twice as wide as
      // CesiumJS for the same spread, so CesiumJS's 2 sigmas are 4 here.
      expect(param(e, "uRuleParams")).toEqual([1, 1, 2 * PLAYCANVAS_FOOTPRINT, 0]);
      expect(param(e, "uRulePlane")).toEqual([0, 0, 8, 4]);
    }
    // Each tile's source, by the tileset it came from: the scan's own, a fill's, none.
    expect(param(scan, "uRuleSource")).toEqual([0, 0, 0, 0]);
    expect(param(layer, "uRuleSource")).toEqual([2, 0, 0, 0]);
    expect(param(stray, "uRuleSource")).toEqual([NO_OWNER, 0, 0, 0]);
    // A split object is weighed whole, by the cell under its anchor: none until it is placed.
    expect(param(object, "uRuleSource").slice(0, 2)).toEqual([0, 1]);
    // The plane's bytes, padded with "no owner".
    const owners = component(scan).params.get("uRuleOwners") as { data: Uint8Array };
    expect(Array.from(owners.data.subarray(0, 8))).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
    expect(owners.data[32]).toBe(NO_OWNER);
    expect(scanRuleReading("rule-a")).toMatchObject({
      renderer: "playcanvas",
      on: true,
      sources: { 0: 2, 2: 1, [NO_OWNER]: 1 },
    });

    // A tile loaded later is ruled too.
    const late = await backend.load(SCAN, tile("6_1.glb"));
    expect(component(late).modifier?.glsl).toContain("hexapodRuleWeight");

    // Off: every tile as it was, the plain scan tile with no modifier at all.
    publishScanRule("rule-a", null);
    expect(component(scan).modifier ?? null).toBeNull();
    expect(component(layer).modifier?.glsl).not.toContain("hexapodRuleWeight");
    expect(scanRuleReading("rule-a")).toMatchObject({ on: false });
    backend.destroy();
    expect(scanRuleReading("rule-a")).toBeNull();
  });

  it("draws every splat as it was while the rule is off for comparison", async () => {
    const { backend } = await backendFor("rule-b");
    const scan = await backend.load(SCAN, tile("1_1.glb"));
    publishScanRule("rule-b", view({ enabled: false, reach: 0 }));
    expect(param(scan, "uRuleParams")).toEqual([0, 1, 0, 0]);
    backend.destroy();
  });

  it("tints the inferred sources the rule names, by the layer's own Highlight", async () => {
    const { backend } = await backendFor("rule-c");
    const layer = await backend.loadLayer?.(LAYER, tile("3_3.glb"));
    if (!layer) throw new Error("no layers");
    backend.setLayerLook?.(layer, LOOK);
    expect(param(layer, "uLayerPattern")[3]).toBe(0);
    publishScanRule("rule-c", view({ tinted: 1 << 2 }));
    expect(param(layer, "uLayerPattern")[3]).toBe(1);
    publishScanRule("rule-c", view({ tinted: 0 }));
    expect(param(layer, "uLayerPattern")[3]).toBe(0);
    backend.destroy();
  });

  it("draws Data age from the rule's own view, and a split object in its own colours", async () => {
    const { meta, planes } = await encodeConfidenceGrid({
      level: 2,
      i0: 0,
      j0: 0,
      ni: 8,
      nj: 4,
      cell: () => ({ score: 240 }),
    });
    const grid = await decodeCellGrid(meta, planes);
    const age = {
      mode: "data-age" as const,
      rows: cellTextureRows(grid.meta, confidenceTexels(grid)),
      meta: grid.meta,
    };
    const { backend } = await backendFor("rule-d");
    const scan = await backend.load(SCAN, tile("1_1.glb"));
    const object = await backend.load(OBJECT, tile("5_1.glb"));
    publishScanRule("rule-d", view({ age }));
    const glsl = component(scan).modifier?.glsl ?? "";
    expect(glsl).toContain("hexapodCellTile(center, color");
    // The look is not put on a split object: it keeps its own colours, as under CesiumJS.
    expect(glsl).toContain("if (uRuleSource.y < 0.5) {");
    expect(param(scan, "uCellTile")[0]).toBeGreaterThan(0);
    expect(component(object).modifier?.glsl).toBe(glsl);
    // Without the rule's view the scan's own (none here) is drawn: the photo.
    publishScanRule("rule-d", view());
    expect(component(scan).modifier?.glsl).not.toContain("hexapodCellTile(center, color");
    backend.destroy();
  });

  it("weighs a split object by its declared origin at its pose, as CesiumJS does", async () => {
    const { backend } = await backendFor("rule-b");
    // The object's tile: its splats' middle 1 m east of the object's origin, in its own frame.
    const object = await backend.load(OBJECT, tile("1_0.glb"));
    publishScanRule("rule-b", view());
    // Not placed yet: no anchor, so nothing of it is drawn (outside every plane).
    expect(param(object, "uRuleSource")[0]).toBe(0);
    expect(param(object, "uRuleSource")[2]).toBeGreaterThan(1e8);
    // Placed with its origin at x = 3.9, in March's cell beside the seam at x = 4; its splats'
    // middle (x = 4.9) is in June's. The anchor is the origin: March's copy is the one drawn,
    // as under CesiumJS (compositorSession.test.ts, the same case).
    backend.place?.(object, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3.9, 1, 0, 1]);
    expect(param(object, "uRuleSource").map((v) => Math.round(v * 10) / 10)).toEqual([
      0, 1, 3.9, 1,
    ]);
    backend.destroy();
  });

  it("gives a natively streamed scan the rule's native source", async () => {
    const { backend } = await backendFor("rule-a");
    publishScanRule("rule-a", view({ native: 1 }));
    const stream = await backend.streamNative?.("https://scan.test/lod-meta.json");
    const entity = pc.entities.find((e) => e.name === "scan");
    expect(param(entity, "uRuleSource")).toEqual([1, 0, 0, 0]);
    stream?.stop();
    backend.destroy();
  });
});

describe("the other dates a rule draws under PlayCanvas", () => {
  beforeEach(() => {
    vi.stubGlobal("Worker", FakeWorker);
    pc.entities.length = 0;
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    publishScanRule("rule-e", null);
  });

  it("are streamed into its one app, ruled by their own source, none of the scan's objects", async () => {
    const JUNE = "https://scan.test/june/tileset.json";
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
    // The rule's session lends the renderer the globe's streaming classes as it starts.
    lendRuleKit(RULE_KIT);
    const { backend } = await backendFor("rule-e");
    const own = await backend.load(SCAN, tile("1_1.glb"));
    backend.add(own);
    publishScanRule(
      "rule-e",
      view({
        sources: new Map([
          [SCAN, { source: 0 }],
          [JUNE, { source: 1 }],
        ]),
        dates: [{ url: JUNE, assetId: "june", extras: {}, placement: null, root: IDENTITY }],
      }),
    );
    // The dates' code is fetched with the first rule that names one, then June's tileset.
    await vi.waitFor(() => expect(fetched).toContain(JUNE), { timeout: 20_000 });
    let june: (typeof pc.entities)[number] | undefined;
    for (let i = 0; i < 50 && !june?.gsplat; i += 1) {
      backend.render(pose);
      await new Promise((r) => setTimeout(r, 10));
      june = pc.entities.find((e) => e.name === "9_1.glb");
    }
    expect(june?.gsplat?.modifier?.glsl).toContain("color.a *= hexapodRuleWeight(center);");
    expect(param(june, "uRuleSource")).toEqual([1, 0, 0, 0]);
    // Bound to none of the scan's objects, and not picked as the scan's.
    expect(backend.instanceTiles?.().tiles).toBe(1);
    // Shown at the next plan, which waits out the replan interval.
    for (let i = 0; i < 20 && scanRuleReading("rule-e")?.dates.tiles !== 1; i += 1) {
      await new Promise((r) => setTimeout(r, 20));
      backend.render(pose);
    }
    expect(scanRuleReading("rule-e")?.dates).toMatchObject({ dates: 1, tiles: 1 });
    // The rule off: the date is let go.
    publishScanRule("rule-e", null);
    expect(scanRuleReading("rule-e")?.dates).toMatchObject({ dates: 0, tiles: 0 });
    backend.destroy();
  });
});

describe("the rule's weight on a colour-only copy", () => {
  it("reads the splat's spread back from the work buffer, where the copy hands none over", () => {
    // PlayCanvas re-copies an SH splat's colour alone, and for a SOG skips
    // modifySplatRotationScale there (GSPLAT_WORKBUFFER_GEOMETRY): the full copy's geometry,
    // in the work buffer, is what the weight reads then.
    const at = RULE_WEIGHT_GLSL.indexOf("#ifdef GSPLAT_WORKBUFFER_GEOMETRY");
    expect(at).toBeGreaterThan(RULE_WEIGHT_GLSL.indexOf("float hexapodRuleWeight(vec3 center) {"));
    expect(RULE_WEIGHT_GLSL).toContain("hexapodRuleRotation = workBufferWorldRotation();");
    expect(RULE_WEIGHT_GLSL).toContain("hexapodRuleScale = workBufferWorldScale();");
    expect(RULE_WEIGHT_GLSL.indexOf("vec4 q = hexapodRuleRotation;")).toBeGreaterThan(at);
  });
});

describe("a splat's footprint as each renderer draws it", () => {
  it("is twice as wide under PlayCanvas as under CesiumJS, for the same spread", () => {
    expect(PLAYCANVAS_FOOTPRINT).toBe(2);
  });
});

describe("a splat's spread across the plane", () => {
  it("is its scales along x and y unrotated, and swapped a quarter turn about up", () => {
    expect(ruleSigma([0, 0, 0, 1], [0.3, 0.1, 0.05])).toEqual([0.3, 0.1]);
    const quarter = Math.SQRT1_2;
    const [sx, sy] = ruleSigma([0, 0, quarter, quarter], [0.3, 0.1, 0.05]);
    expect(sx).toBeCloseTo(0.1, 9);
    expect(sy).toBeCloseTo(0.3, 9);
  });
});
