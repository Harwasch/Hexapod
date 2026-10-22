/**
 * As captured (ADR 0019) draws no procedural fixture: the synthetic tree the Living Survey's
 * motion model is tested against is generated, never captured, so its model is not drawn while
 * As captured is on (SiteManager `hideFixtures`), under any renderer (a rigged scan is always
 * CesiumJS's), and is drawn again once it is off. A captured site is untouched.
 */
import { Cartesian3, Event, Math as CesiumMath } from "cesium";
import { describe, expect, it, vi } from "vitest";

import type { Site, SiteAsset, SiteSummary } from "@twin/contracts";

import type { CameraController } from "@/cesium/CameraController";
import type { ClippingManager } from "@/cesium/ClippingManager";
import type { PerformanceManager } from "@/cesium/PerformanceManager";
import type * as Tiles from "@/cesium/providers/tiles";
import { SiteManager } from "@/cesium/SiteManager";
import type { SceneEvents } from "@/cesium/types";
import { Emitter } from "@/lib/emitter";

const LON = -82.6966;
const LAT = 28.0389;
/** Metres per degree of latitude there. */
const M_PER_DEG = 110_800;
/** The synthetic tree: some 8 m in radius, standing on the ground. */
const TREE = { center: Cartesian3.fromDegrees(LON, LAT, 4), radius: 7.84 };

function fakeTileset() {
  return {
    show: false,
    preloadWhenHidden: true,
    maximumScreenSpaceError: 16,
    tilesLoaded: true,
    totalMemoryUsageInBytes: 0,
    boundingSphere: TREE,
    loadProgress: new Event(),
    initialTilesLoaded: new Event(),
    allTilesLoaded: new Event(),
    tileFailed: new Event(),
    tileLoad: new Event(),
    tileUnload: new Event(),
    isDestroyed: () => false,
    destroy: vi.fn(),
  };
}
let tileset = fakeTileset();

vi.mock("@/cesium/providers/tiles", async (importOriginal) => ({
  ...(await importOriginal<typeof Tiles>()),
  createSiteTileset: vi.fn(() => Promise.resolve(tileset)),
}));
vi.mock("@/cesium/scanView/ScanRendererHost", () => ({
  prefetchScanDestination: () => () => undefined,
}));
// What a splat tileset gets attached reads the real thing's internals; none of it is at issue.
vi.mock("@/cesium/splatViewCones", () => ({ attachViewCones: () => () => undefined }));
vi.mock("@/cesium/inferredLayers", () => ({ attachInferredLayers: () => () => undefined }));
vi.mock("@/cesium/splatInstances", () => ({ attachInstances: () => () => undefined }));
vi.mock("@/cesium/splitObjects", () => ({ attachSplitObjects: () => () => undefined }));
vi.mock("@/cesium/splatSkin", () => ({ attachSkin: () => () => undefined }));
vi.mock("@/cesium/telemetry", () => ({ attachTelemetry: () => () => undefined }));

const SITE_ID = "tree";

function harness(slug: string) {
  const d = 0.00005;
  const ring = [
    [LON - d, LAT - d],
    [LON + d, LAT - d],
    [LON + d, LAT + d],
    [LON - d, LAT + d],
    [LON - d, LAT - d],
  ];
  const site = {
    id: SITE_ID,
    slug,
    name: "tree",
    centroid: { longitude: LON, latitude: LAT, height: 0 },
    boundary: { type: "Polygon", coordinates: [ring] },
    assets: [
      {
        id: "tree-splat",
        siteId: SITE_ID,
        name: "tree splat",
        representation: "gaussian-splat",
        source: { type: "3d-tiles-url", url: "https://example.invalid/tree/tileset.json" },
        renderConfig: {
          maximumScreenSpaceError: 16,
          clipsWorld: true,
          heightOffsetM: 0,
          rigUrl: "../source/rig.json",
        },
        defaultVisible: true,
      } as unknown as SiteAsset,
    ],
    cameraBookmarks: [],
  } as unknown as Site;
  const summary = {
    id: SITE_ID,
    slug,
    name: "tree",
    centroid: site.centroid,
    areaM2: 100,
  } as unknown as SiteSummary;
  /** Where the camera is: `south` metres south of the tree, `up` metres above the ground. */
  const where = { south: 50_000, up: 30_000 };
  const position = (): Cartesian3 =>
    Cartesian3.fromDegrees(LON, LAT - where.south / M_PER_DEG, where.up);
  const moveEnd: (() => void)[] = [];
  const viewer = {
    camera: {
      get positionWC() {
        return position();
      },
      heading: 0,
      // A 960 x 600 canvas under Cesium's 60 degree field of view.
      frustum: { fovy: 2 * Math.atan(Math.tan(CesiumMath.toRadians(30)) / 1.6) },
      changed: { addEventListener: () => () => undefined },
      moveEnd: {
        addEventListener: (listener: () => void) => {
          moveEnd.push(listener);
          return () => undefined;
        },
      },
    },
    canvas: { clientHeight: 600 },
    scene: {
      globe: { getHeight: () => undefined },
      primitives: { add: vi.fn(), remove: vi.fn() },
      requestRender: vi.fn(),
      sampleHeightSupported: false,
    },
  };
  const camera = {
    pose: () => ({
      longitude: LON,
      latitude: LAT - where.south / M_PER_DEG,
      altitude: where.up,
      height: where.up,
    }),
    isMoving: false,
    refreshPose: vi.fn(),
    setObjectScale: vi.fn(),
    durationFor: () => 4,
    flyTo: vi.fn(),
  };
  const performance = {
    addScreenSpaceErrorSink: vi.fn(),
    addMemorySource: vi.fn(),
    addMotionFrameListener: () => () => undefined,
    reportContext: vi.fn(),
    reportLoading: vi.fn(),
    resetBenchmark: vi.fn(),
    splatMinimumScreenSpaceError: 8,
  };
  const clipping = { setFootprint: vi.fn() };
  const manager = new SiteManager(
    viewer as never,
    new Emitter<SceneEvents>(),
    camera as unknown as CameraController,
    clipping as unknown as ClippingManager,
    performance as unknown as PerformanceManager,
  );
  manager.setCatalog([summary], () => Promise.resolve(site));
  /** Puts the camera there and lets the camera come to rest (`moveEnd`). */
  const moveTo = (south: number, up: number): void => {
    where.south = south;
    where.up = up;
    for (const listener of moveEnd) listener();
  };
  /** The world clip under the site: its footprint, or null. */
  const clip = (): unknown => clipping.setFootprint.mock.calls.at(-1)?.[1] ?? null;
  return { manager, moveTo, clip };
}

describe("a procedural fixture under As captured", () => {
  for (const renderer of ["cesium", "playcanvas"] as const)
    it(`is not drawn while it is on, and is drawn again after (${renderer})`, async () => {
      tileset = fakeTileset();
      const { manager, moveTo, clip } = harness("synthetic-tree");
      manager.setSplatRenderer(renderer);
      moveTo(20, 30);
      await vi.waitFor(() => expect(tileset.show).toBe(true));
      expect(clip()).not.toBeNull();
      manager.hideFixtures(true);
      expect(tileset.show).toBe(false);
      // The world under it is whole again, and no other renderer is handed it.
      expect(clip()).toBeNull();
      expect(manager.scanTarget()).toBeNull();
      // Moving about while hidden draws nothing of it.
      moveTo(300, 260);
      moveTo(20, 30);
      expect(tileset.show).toBe(false);
      manager.hideFixtures(false);
      expect(tileset.show).toBe(true);
      expect(clip()).not.toBeNull();
    });

  it("is not drawn when it loads while As captured is on", async () => {
    tileset = fakeTileset();
    const { manager, moveTo } = harness("synthetic-tree");
    manager.setSplatRenderer("cesium");
    manager.hideFixtures(true);
    moveTo(20, 30);
    await vi.waitFor(() => expect(manager.loadedAssets()).toHaveLength(1));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(tileset.show).toBe(false);
    manager.hideFixtures(false);
    expect(tileset.show).toBe(true);
  });

  it("leaves a captured site drawn", async () => {
    tileset = fakeTileset();
    const { manager, moveTo } = harness("oak-bench-yard");
    manager.setSplatRenderer("cesium");
    moveTo(20, 30);
    await vi.waitFor(() => expect(tileset.show).toBe(true));
    manager.hideFixtures(true);
    expect(tileset.show).toBe(true);
  });
});
