/**
 * A scan drawn at its runtime scale through `SiteManager`: placed, clamped, previewed, saved.
 *
 * `renderConfig.scale` is the factor the catalog says to draw a pipeline-placed splat at, about
 * its root transform's origin (docs/DATA_MODEL.md "Runtime scale"). Everything the catalog puts
 * on the globe already fits it -- in particular the ground samples the clamp rests the model
 * on -- so the viewer scales the tiles and samples the ground exactly where the catalog says.
 * A preview (the Set real size tool) moves the samples by its ratio to the saved scale only,
 * and a saved record replaces the preview with the catalog's own numbers.
 *
 * The tileset is faked down to what placement reads: a root east/north/up at the origin, a
 * bounding sphere that follows the model matrix, the root box's lowest corner. The terrain is
 * the drawn ground (`sampleHeightMostDetailed`), flat, so every lift is checkable by hand.
 */
import {
  BoundingSphere,
  Cartesian3,
  Cartographic,
  Event,
  Math as CesiumMath,
  Matrix3,
  Matrix4,
  Transforms,
} from "cesium";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Site, SiteAsset, SiteSummary } from "@twin/contracts";

import type { CameraController } from "@/cesium/CameraController";
import type { ClippingManager } from "@/cesium/ClippingManager";
import { uniformScale } from "@/cesium/placement";
import type { PerformanceManager } from "@/cesium/PerformanceManager";
import type * as Tiles from "@/cesium/providers/tiles";
import { SiteManager } from "@/cesium/SiteManager";
import type { SceneEvents } from "@/cesium/types";
import { Emitter } from "@/lib/emitter";

const LON = -82.6966;
const LAT = 28.0389;
const ORIGIN_HEIGHT = 20;
const ORIGIN = Cartesian3.fromDegrees(LON, LAT, ORIGIN_HEIGHT);
const ROOT = Transforms.eastNorthUpToFixedFrame(ORIGIN);
/** The drawn ground, everywhere: 5 m above the ellipsoid. */
const GROUND = 5;

/** The local sphere and box of the fake scan: 4 m about a point 3 m up, 1 m east. */
const LOCAL_CENTRE = new Cartesian3(1, 0, 3);
const LOCAL_RADIUS = 4;
const LOWEST_LOCAL_Z = -1;

function fakeTileset() {
  const world = (model: Matrix4) => Matrix4.multiply(model, ROOT, new Matrix4());
  const tileset = {
    show: false,
    preloadWhenHidden: true,
    maximumScreenSpaceError: 16,
    tilesLoaded: true,
    totalMemoryUsageInBytes: 0,
    modelMatrix: Matrix4.clone(Matrix4.IDENTITY),
    get boundingSphere() {
      return BoundingSphere.transform(
        new BoundingSphere(LOCAL_CENTRE, LOCAL_RADIUS),
        world(this.modelMatrix),
      );
    },
    root: {
      transform: ROOT,
      // The root box as Cesium computes it at load (identity model matrix): its lowest corner
      // is LOWEST_LOCAL_Z under the origin.
      boundingVolume: {
        boundingVolume: {
          center: Matrix4.multiplyByPoint(ROOT, new Cartesian3(1, 0, 3), new Cartesian3()),
          halfAxes: Matrix3.multiply(
            Matrix4.getMatrix3(ROOT, new Matrix3()),
            Matrix3.fromScale(new Cartesian3(2, 2, 3 - LOWEST_LOCAL_Z)),
            new Matrix3(),
          ),
        },
      },
    },
    loadProgress: new Event(),
    initialTilesLoaded: new Event(),
    allTilesLoaded: new Event(),
    tileFailed: new Event(),
    tileLoad: new Event(),
    tileUnload: new Event(),
    isDestroyed: () => false,
    destroy: vi.fn(),
  };
  return tileset;
}
type FakeTileset = ReturnType<typeof fakeTileset>;

const made = vi.hoisted(() => ({ tilesets: [] as unknown[] }));
vi.mock("@/cesium/providers/tiles", async (importOriginal) => ({
  ...(await importOriginal<typeof Tiles>()),
  createSiteTileset: vi.fn(() => {
    const tileset = fakeTileset();
    made.tilesets.push(tileset);
    return Promise.resolve(tileset);
  }),
}));
vi.mock("@/cesium/scanView/ScanRendererHost", () => ({
  prefetchScanDestination: () => () => undefined,
}));

const SITE_ID = "site-scaled";
const ASSET_ID = "asset-splat";

/** Cells of the capture's own ground at the saved scale: 2 m under the origin, about it. */
function samplesAt(scale: number) {
  return [
    [0.0002, 0],
    [-0.0002, 0],
    [0, 0.0002],
    [0, -0.0002],
  ].map(([dLon = 0, dLat = 0]) => ({
    lon: LON + dLon * scale,
    lat: LAT + dLat * scale,
    height: ORIGIN_HEIGHT - 2 * scale,
  }));
}

function splat(scale: number, renderConfig: Record<string, unknown> = {}): SiteAsset {
  return {
    id: ASSET_ID,
    siteId: SITE_ID,
    provider: "3d-tiles-url",
    name: "scan",
    representation: "gaussian-splat",
    source: { type: "3d-tiles-url", url: "https://example.invalid/scan/tileset.json" },
    footprint: null,
    renderConfig: {
      maximumScreenSpaceError: 16,
      clipsWorld: false,
      heightOffsetM: 0,
      clampToGround: true,
      groundSamples: samplesAt(scale),
      ...(scale === 1 ? {} : { scale }),
      ...renderConfig,
    },
    defaultVisible: true,
    createdAt: "2026-01-01T00:00:00Z",
  } as unknown as SiteAsset;
}

function site(asset: SiteAsset, half = 0.0003): Site {
  return {
    id: SITE_ID,
    slug: "scaled",
    name: "Scaled",
    centroid: { longitude: LON, latitude: LAT, height: null },
    boundary: {
      type: "Polygon",
      coordinates: [
        [
          [LON - half, LAT - half],
          [LON + half, LAT - half],
          [LON + half, LAT + half],
          [LON - half, LAT + half],
          [LON - half, LAT - half],
        ],
      ],
    },
    metadata: {
      captureId: "capture-1",
      registration: { georef: { lat: LAT, lon: LON, height: ORIGIN_HEIGHT } },
    },
    assets: [asset],
    cameraBookmarks: [],
  } as unknown as Site;
}

function harness(record: Site) {
  made.tilesets.length = 0;
  const events = new Emitter<SceneEvents>();
  const listener = { addEventListener: () => () => undefined };
  /** Every set of points the drawn ground was sampled at, degrees. */
  const sampled: { lon: number; lat: number }[][] = [];
  const viewer = {
    camera: {
      positionWC: Cartesian3.fromDegrees(-110, 35, 18_000_000),
      heading: 0,
      changed: listener,
      moveEnd: listener,
    },
    // No availability: the terrain sample fails, and the drawn ground is the ground.
    terrainProvider: {},
    scene: {
      globe: { getHeight: () => undefined },
      primitives: { add: vi.fn(), remove: vi.fn() },
      requestRender: vi.fn(),
      sampleHeightSupported: true,
      sampleHeightMostDetailed: vi.fn((points: Cartographic[]) => {
        sampled.push(
          points.map((p) => ({
            lon: CesiumMath.toDegrees(p.longitude),
            lat: CesiumMath.toDegrees(p.latitude),
          })),
        );
        return Promise.resolve(
          points.map((p) => new Cartographic(p.longitude, p.latitude, GROUND)),
        );
      }),
    },
  };
  const camera = {
    pose: () => ({ longitude: -110, latitude: 35, altitude: 1e7, height: 1e7 }),
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
    events,
    camera as unknown as CameraController,
    clipping as unknown as ClippingManager,
    performance as unknown as PerformanceManager,
  );
  const summary = {
    id: SITE_ID,
    slug: "scaled",
    name: "Scaled",
    centroid: { longitude: LON, latitude: LAT, height: null },
    areaM2: 400,
  } as unknown as SiteSummary;
  manager.setCatalog([summary], () => Promise.resolve(record));
  const tileset = () => made.tilesets[0] as FakeTileset;
  return { manager, sampled, tileset };
}

/** Where the model matrix puts the root's origin, and how far it scales. */
function placement(tileset: FakeTileset) {
  const origin = Matrix4.multiplyByPoint(tileset.modelMatrix, ORIGIN, new Cartesian3());
  const carto = Cartographic.fromCartesian(origin);
  const horizontal = Cartesian3.distance(
    Cartesian3.fromRadians(carto.longitude, carto.latitude, 0),
    Cartesian3.fromDegrees(LON, LAT, 0),
  );
  return { scale: uniformScale(tileset.modelMatrix), originHeight: carto.height, horizontal };
}

afterEach(() => {
  vi.useRealTimers();
});

describe("a scan at its runtime scale", () => {
  it("is drawn at the catalog's scale about its origin and rests on the ground there", async () => {
    const record = site(splat(0.5));
    const { manager, sampled, tileset } = harness(record);
    await manager.activate(SITE_ID);
    await manager.boundingSphere(record);
    const placed = placement(tileset());
    expect(placed.scale).toBeCloseTo(0.5, 12);
    // The origin only moves up: the cells are 1 m under it at half size, 5 - 19 = -14 m lift.
    expect(placed.horizontal).toBeLessThan(1e-3);
    expect(placed.originHeight).toBeCloseTo(GROUND + 1, 3);
    // The ground was sampled exactly where the catalog says its cells are: untouched.
    const cells = sampled.at(-1)?.slice(1) ?? [];
    const catalog = samplesAt(0.5);
    expect(cells).toHaveLength(catalog.length);
    for (const [i, cell] of cells.entries()) {
      expect(cell.lon).toBeCloseTo(catalog[i]?.lon ?? 0, 9);
      expect(cell.lat).toBeCloseTo(catalog[i]?.lat ?? 0, 9);
    }
    expect(manager.shownScale(SITE_ID)).toBe(0.5);
    // The sphere is the scaled model's.
    expect(tileset().boundingSphere.radius).toBeCloseTo(LOCAL_RADIUS * 0.5, 6);
  });

  it("puts the scaled lowest point on the ground when the capture's ground was never measured", async () => {
    const record = site(splat(0.5, { groundSamples: [] }));
    const { manager, tileset } = harness(record);
    await manager.activate(SITE_ID);
    await manager.boundingSphere(record);
    // The box's lowest corner, a metre under the origin registered, is half a metre under it.
    const placed = placement(tileset());
    expect(placed.scale).toBeCloseTo(0.5, 12);
    expect(placed.originHeight - 0.5 * -LOWEST_LOCAL_Z).toBeCloseTo(GROUND, 2);
  });

  it("is the old translation exactly when nobody resized it", async () => {
    const record = site(splat(1));
    const { manager, tileset } = harness(record);
    await manager.activate(SITE_ID);
    await manager.boundingSphere(record);
    const model = tileset().modelMatrix;
    expect(Matrix3.equals(Matrix4.getMatrix3(model, new Matrix3()), Matrix3.IDENTITY)).toBe(true);
    // Cells 2 m under the origin on 5 m ground: the origin ends 7 m up.
    expect(placement(tileset()).originHeight).toBeCloseTo(GROUND + 2, 3);
  });

  it("previews another scale at once, clamps it after a pause, and puts the saved one back", async () => {
    vi.useFakeTimers();
    const record = site(splat(0.5));
    const { manager, sampled, tileset } = harness(record);
    await manager.activate(SITE_ID);
    await manager.boundingSphere(record);
    const before = sampled.length;

    expect(manager.previewScale(SITE_ID, 0.25)).toBe(true);
    // At once, on the ground sampled before: the cells half a metre under the origin now.
    expect(placement(tileset()).scale).toBeCloseTo(0.25, 12);
    expect(placement(tileset()).originHeight).toBeCloseTo(GROUND + 0.5, 3);
    expect(manager.shownScale(SITE_ID)).toBe(0.25);
    expect(sampled.length).toBe(before);

    // Then sampled afresh, once the scale has held: where the preview puts the cells -- half
    // as far from the origin as the catalog's, the preview's ratio to the saved scale.
    await vi.advanceTimersByTimeAsync(300);
    const cells = sampled.at(-1)?.slice(1) ?? [];
    expect(sampled.length).toBe(before + 1);
    expect(cells[0]?.lon).toBeCloseTo(LON + 0.0002 * 0.25, 9);
    expect(cells[2]?.lat).toBeCloseTo(LAT + 0.0002 * 0.25, 9);

    expect(manager.previewScale(SITE_ID, null)).toBe(true);
    expect(placement(tileset()).scale).toBeCloseTo(0.5, 12);
    expect(placement(tileset()).originHeight).toBeCloseTo(GROUND + 1, 3);
    expect(manager.shownScale(SITE_ID)).toBe(0.5);
  });

  it("takes a saved scale from the record: the catalog's ground, the new boundary", async () => {
    const record = site(splat(0.5));
    const { manager, sampled, tileset } = harness(record);
    await manager.activate(SITE_ID);
    await manager.boundingSphere(record);
    manager.previewScale(SITE_ID, 0.3);

    // Saved at a quarter: the API moved the cells and the boundary with it.
    const saved = site(splat(0.25), 0.00015);
    manager.updateRecord(saved);
    expect(manager.shownScale(SITE_ID)).toBe(0.25);
    expect(placement(tileset()).scale).toBeCloseTo(0.25, 12);
    await vi.waitFor(() => {
      const cells = sampled.at(-1)?.slice(1) ?? [];
      expect(cells[0]?.lon).toBeCloseTo(samplesAt(0.25)[0]?.lon ?? 0, 9);
    });
    await vi.waitFor(() => expect(placement(tileset()).originHeight).toBeCloseTo(GROUND + 0.5, 3));
    expect(manager.activeSite?.boundary).toEqual(saved.boundary);

    // A record refetched for another reason leaves it where it is.
    const count = sampled.length;
    manager.updateRecord({ ...saved, cameraBookmarks: [] });
    expect(sampled.length).toBe(count);
  });

  it("takes the boundary from the refetch that follows the asset the save put in", async () => {
    const record = site(splat(0.5));
    const { manager, tileset } = harness(record);
    await manager.activate(SITE_ID);
    await manager.boundingSphere(record);
    // The save puts the asset in the site's record at once; the boundary is still the old one.
    manager.updateRecord(site(splat(0.25)));
    expect(placement(tileset()).scale).toBeCloseTo(0.25, 12);
    expect(manager.activeSite?.boundary).toEqual(record.boundary);
    // The refetch: the same asset, the boundary the API moved with it.
    const refetched = site(splat(0.25), 0.00015);
    manager.updateRecord(refetched);
    expect(manager.activeSite?.boundary).toEqual(refetched.boundary);
    expect(placement(tileset()).scale).toBeCloseTo(0.25, 12);
  });

  it("previews nothing for a site whose scan the API cannot resize", async () => {
    const record: Site = { ...site(splat(1)), metadata: {} };
    const { manager } = harness(record);
    await manager.activate(SITE_ID);
    expect(manager.previewScale(SITE_ID, 0.5)).toBe(false);
    expect(manager.shownScale(SITE_ID)).toBeNull();
    expect(manager.scalableTileset(SITE_ID)).toBeNull();
  });
});
