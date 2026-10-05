/**
 * The primary site's splat scan is drawn from afar (SiteManager's far view, farView.ts): not
 * engaged, but big enough on screen, so it is drawn small instead of vanishing as the camera
 * zooms out -- by CesiumJS (the tileset shown) or by a dedicated renderer (`scanTarget`, with
 * `far` outside its key) -- while engagement still decides everything else.
 */
import { Cartesian3, Event, Math as CesiumMath } from "cesium";
import { describe, expect, it, vi } from "vitest";

import type { Site, SiteAsset, SiteSummary } from "@twin/contracts";

import type { CameraController } from "@/cesium/CameraController";
import type { ClippingManager } from "@/cesium/ClippingManager";
import type { PerformanceManager } from "@/cesium/PerformanceManager";
import type * as Tiles from "@/cesium/providers/tiles";
import { cesiumPickSource } from "@/cesium/sceneSelect/cesiumPickSource";
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
    // One splat in one tile, baked where it is: what CesiumJS's pick source reads (incremental).
    gaussianSplatPrimitive: {
      _positions: new Float32Array([0, 0, 4]),
      _numSplats: 1,
      _tileSlots: new Map([
        [
          { content: { _lastSplatTransform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] } },
          { start: 0, count: 1 },
        ],
      ]),
    },
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

function harness() {
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
    slug: "tree",
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
        renderConfig: { maximumScreenSpaceError: 16, clipsWorld: true, heightOffsetM: 0 },
        defaultVisible: true,
      } as unknown as SiteAsset,
    ],
    cameraBookmarks: [],
  } as unknown as Site;
  const summary = {
    id: SITE_ID,
    slug: "tree",
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

describe("a splat scan seen from afar", () => {
  it("a dedicated renderer draws it small, as the same scan, near and far", async () => {
    tileset = fakeTileset();
    const { manager, moveTo, clip } = harness();
    // Loaded from 50 km off: past the far view's limit, so not drawn.
    await manager.activate(SITE_ID);
    expect(manager.scanTarget()).toBeNull();
    // Some 400 m off and 260 m up: well past where the site engages (about 75 m up for a
    // scan this size), and some 33 px across on screen.
    moveTo(300, 260);
    const far = manager.scanTarget();
    expect(far).toMatchObject({ key: `${SITE_ID}:tree-splat`, far: true });
    expect(far?.tileset).toBe(tileset);
    // PlayCanvas draws it, so CesiumJS's copy stays hidden; the world is clipped under it.
    expect(tileset.show).toBe(false);
    expect(clip()).not.toBeNull();
    expect(manager.loadedAssets()[0]?.shown).toBe(false);

    // 5 km off it is under 3 px: not drawn, and the world is whole again.
    moveTo(4000, 3000);
    expect(manager.scanTarget()).toBeNull();
    expect(clip()).toBeNull();

    // Up close it engages: the same scan, the same key, drawn near.
    moveTo(20, 30);
    expect(manager.scanTarget()).toMatchObject({ key: `${SITE_ID}:tree-splat`, far: false });
    // And out again to 400 m: still the same key, now far.
    moveTo(300, 260);
    expect(manager.scanTarget()).toMatchObject({ key: `${SITE_ID}:tree-splat`, far: true });
  });

  it("CesiumJS draws it by its own level of detail, and selects nothing in it", async () => {
    tileset = fakeTileset();
    const { manager, moveTo } = harness();
    manager.setSplatRenderer("cesium");
    // Loaded by proximity, from 400 m: drawn as soon as its tileset is in.
    moveTo(300, 260);
    await vi.waitFor(() => expect(tileset.show).toBe(true));
    expect(manager.scanTarget()).toBeNull();
    expect(manager.loadedAssets()[0]?.shown).toBe(true);
    // Objects are selected once the site is engaged, not in a scan a few dozen pixels across.
    const source = cesiumPickSource(tileset as never);
    expect(source.tiles()).toEqual([]);

    moveTo(4000, 3000);
    expect(tileset.show).toBe(false);
    moveTo(300, 260);
    expect(tileset.show).toBe(true);
    // Engaged: the same tileset, now a scan to work in.
    moveTo(20, 30);
    expect(tileset.show).toBe(true);
    expect(source.tiles()).toHaveLength(1);
  });

  it("is drawn, not a scan the camera is in, though its bounds reach past the camera", async () => {
    // A small footprint whose tiles' bounds span outliers 600 m out (as a splat's root box
    // does): from 400 m the camera is inside them, but the site is not engaged.
    tileset = { ...fakeTileset(), boundingSphere: { center: TREE.center, radius: 600 } };
    const { manager, moveTo } = harness();
    manager.setSplatRenderer("cesium");
    moveTo(300, 260);
    await vi.waitFor(() => expect(tileset.show).toBe(true));
    // The world holds still and loses its floor in a scan up close, not one seen from afar.
    expect(manager.insideSplatScan()).toBe(false);
    moveTo(20, 30);
    expect(manager.insideSplatScan()).toBe(true);
  });
});
