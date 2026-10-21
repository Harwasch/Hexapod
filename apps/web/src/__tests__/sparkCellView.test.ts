/**
 * Confidence and Data age under Spark (sparkBackend.ts, scanView/cellModifier.ts): every tile
 * of the scan gets the shared cell look in its world modifier, reading the view's texture and
 * the grid's own rule at one level per tile -- regenerated only when its level steps -- and an
 * inferred layer's tiles go grey. Photo takes the look off again. Spark is faked (no WebGL in
 * jsdom): its dynos record what they were built from, so the test reads which look each tile's
 * modifier applies and with which inputs.
 */
import type * as THREE from "three";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CELL_TILE_GLSL } from "@/cesium/scanView/cellModifier";
import type { LayerLook } from "@/cesium/scanView/layerLook";
import type { BackendHooks, ScanPose } from "@/cesium/scanView/types";
import { TileWork } from "@/cesium/scanView/tileWork";
import { encodeConfidenceGrid, REASON_BITS } from "@/dev/cellGridFixture";
import { cellTextureRows, confidenceTexels, decodeCellGrid } from "@/lib/cellGrid";
import { cellLevelIndex, pxPerRadianOf, type CellLookView } from "@/lib/cellLook";
import type { TileNode } from "@/view/tiles";

/** A dyno as the fake records it: its GLSL and statements, and what it was applied to. */
interface Applied {
  globals: string[];
  statements: string[];
  inputs: Record<string, unknown>;
  from: unknown;
}

vi.mock("three", async (importOriginal) => {
  const actual = await importOriginal<typeof THREE>();
  const nothing = (): void => undefined;
  class WebGLRenderer {
    readonly setClearColor = nothing;
    readonly setPixelRatio = nothing;
    readonly setSize = nothing;
    readonly setAnimationLoop = nothing;
    readonly dispose = nothing;
    readonly forceContextLoss = nothing;
    readonly render = nothing;
  }
  return { ...actual, WebGLRenderer };
});

vi.mock("@sparkjsdev/spark", async () => {
  const three = await vi.importActual<typeof THREE>("three");
  class SparkRenderer extends three.Object3D {
    sorting = false;
    sortDirty = false;
    sortTimeoutId = -1;
    readonly display = { mapping: [] };
    readonly current = this.display;
    override dispose = (): void => undefined;
  }
  const uniform = (value: unknown) => ({ value });
  class Dyno {
    constructor(
      readonly options: {
        globals: () => string[];
        statements: (args: {
          inputs: Record<string, string>;
          outputs: Record<string, string>;
        }) => string[];
      },
    ) {}
    apply(inputs: Record<string, unknown>): { gsplat: Applied } {
      const names = Object.fromEntries(Object.keys(inputs).map((k) => [k, k]));
      return {
        gsplat: {
          globals: this.options.globals(),
          statements: this.options.statements({ inputs: names, outputs: { gsplat: "out" } }),
          inputs,
          from: inputs.gsplat,
        },
      };
    }
  }
  const dynoBlock = (
    _inTypes: unknown,
    _outTypes: unknown,
    construct: (args: { gsplat: unknown }) => { gsplat: unknown },
  ) => ({ construct });
  return {
    SparkRenderer,
    dyno: {
      dynoSampler2D: uniform,
      dynoUsampler2D: uniform,
      dynoVec4: uniform,
      dynoMat4: uniform,
      dynoBlock,
      Dyno,
      Gsplat: "Gsplat",
    },
  };
});

const meshes = vi.hoisted(() => ({ centre: [0, 0, 0] as [number, number, number] }));

/** A tile is a mesh around `meshes.centre` (the scan's frame), with no SPZ bytes to digest. */
vi.mock("@/view/sparkStream", async () => {
  const three = await vi.importActual<typeof THREE>("three");
  return {
    loadSplatTile: vi.fn(() => {
      const [x, y, z] = meshes.centre;
      const mesh = Object.assign(new three.Object3D(), {
        maxSh: 0,
        worldModifier: undefined,
        generated: 0,
        versions: 0,
        updateGenerator() {
          this.generated += 1;
        },
        updateVersion() {
          this.versions += 1;
        },
        getBoundingBox: () =>
          new three.Box3(new three.Vector3(x - 1, y - 1, z), new three.Vector3(x + 1, y + 1, z)),
        dispose: (): void => undefined,
      });
      return Promise.resolve({ mesh, bytes: new Uint8Array(0) });
    }),
  };
});

type FakeMesh = THREE.Object3D & {
  worldModifier?: { construct(args: { gsplat: unknown }): { gsplat: Applied } };
  generated: number;
  versions: number;
};

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
  globePixelRatio: 1,
});

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
  const { createBackend } = await import("@/cesium/scanView/sparkBackend");
  const hooks: BackendHooks = {
    frameWanted: () => undefined,
    work: new TileWork(),
    maxShDegree: 3,
    assetId,
  };
  return createBackend(document.createElement("canvas"), 3_000_000, hooks);
}

const URL_ = "https://scan.test/tileset.json";
const tile = (uri: string) => ({ uri }) as unknown as TileNode;

async function load(
  backend: Awaited<ReturnType<typeof backendFor>>,
  centre: [number, number, number],
): Promise<FakeMesh> {
  meshes.centre = centre;
  return (await backend.load(URL_, tile("t.spz"))) as unknown as FakeMesh;
}

/** The cell step of a mesh's world modifier, applied: the last dyno its block applies. */
function cellStep(mesh: FakeMesh): Applied | undefined {
  const out = mesh.worldModifier?.construct({ gsplat: "in" }).gsplat;
  return out?.globals.includes(CELL_TILE_GLSL) ? out : undefined;
}

const vec = (value: unknown): number[] => (value as { value: THREE.Vector4 }).value.toArray();

describe("Confidence and Data age under Spark", () => {
  afterEach(() => vi.clearAllMocks());

  it("draws every scan tile with the shared look, a level a tile, regenerated only on a step", async () => {
    const { publishScanCellView, scanCellViewReading } =
      await import("@/cesium/scanView/scanCellView");
    const backend = await backendFor("spark-a");
    const near = await load(backend, [0, 0, 0]);
    const far = await load(backend, [0, 400, 0]);
    backend.add(near as never);
    backend.add(far as never);
    backend.render(pose([0, -5, 2]));
    expect(near.worldModifier).toBeUndefined();

    const v = await view();
    publishScanCellView("spark-a", v);
    const ppr = pxPerRadianOf(600, Math.PI / 3);
    for (const [mesh, distance] of [
      [near, Math.hypot(5, 2)],
      [far, Math.hypot(405, 2)],
    ] as const) {
      const step = cellStep(mesh);
      expect(step?.statements.join("\n")).toContain("hexapodCellTile(");
      const grid = (step?.inputs.grid as { value: THREE.DataTexture }).value;
      expect(grid.image.width).toBe(4096);
      expect(Array.from((grid.image.data as Uint8Array).subarray(0, 4))).toEqual(
        Array.from(v.rows.data.subarray(0, 4)),
      );
      expect(vec(step?.inputs.rule)).toEqual([100, REASON_BITS.unseen, Math.fround(0.6), 0]);
      expect(vec(step?.inputs.tile)[3]).toBe(cellLevelIndex(v.meta, distance, ppr));
    }
    expect(vec(cellStep(far)?.inputs.tile)[3]).toBeGreaterThan(
      vec(cellStep(near)?.inputs.tile)[3] ?? 0,
    );

    // A small move steps neither tile: nothing is generated again.
    const versions = [near.versions, far.versions];
    backend.render(pose([0, -5.1, 2]));
    expect([near.versions, far.versions]).toEqual(versions);
    // Further off, the near tile steps coarser: only it is generated again.
    backend.render(pose([0, -100, 2]));
    expect(near.versions).toBe((versions[0] ?? 0) + 1);
    expect(far.versions).toBe(versions[1]);
    expect(vec(cellStep(near)?.inputs.tile)[3]).toBe(1);
    expect(scanCellViewReading("spark-a")).toMatchObject({
      renderer: "spark",
      mode: "confidence",
      tiles: 2,
      steps: 1,
    });

    // Photo again: the look comes off.
    publishScanCellView("spark-a", null);
    expect(near.worldModifier).toBeUndefined();
    expect(far.worldModifier).toBeUndefined();
    backend.destroy();
    expect(scanCellViewReading("spark-a")).toBeNull();
  });

  it("draws an inferred layer's splats grey while a view is drawn: no measured colour", async () => {
    const { publishScanCellView } = await import("@/cesium/scanView/scanCellView");
    const backend = await backendFor("spark-b");
    meshes.centre = [3, 3, 0];
    const layer = (await backend.loadLayer?.(URL_, tile("l.spz"))) as unknown as FakeMesh;
    const look: LayerLook = {
      highlight: false,
      cones: null,
      fromScan: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    };
    backend.setLayerLook?.(layer as never, look);
    const step = layer.worldModifier?.construct({ gsplat: "in" }).gsplat;
    expect(step?.statements.join("\n")).toContain("hexapodCellGrey(");
    expect(vec(step?.inputs.fill)[0]).toBe(0);
    const versions = layer.versions;
    publishScanCellView("spark-b", await view());
    expect(vec(step?.inputs.fill)[0]).toBe(1);
    expect(layer.versions).toBeGreaterThan(versions);
    publishScanCellView("spark-b", null);
    expect(vec(step?.inputs.fill)[0]).toBe(0);
    backend.destroy();
  });
});
