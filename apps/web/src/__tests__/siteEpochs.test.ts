/**
 * A site with two dated captures (epochs) in `SiteManager`: which one is shown, how the others
 * are kept, and what each renderer is told.
 *
 * E1 (Mar 1, the drone capture that made the site's frame) and E2 (Jun 1, a phone video) each
 * have a gaussian splat, A1 and A2, both default-visible and A1 listed first: the rule from
 * before epochs would show A1. The tilesets are faked down to what the manager reads: a root
 * east/north/up at the site, a bounding sphere that follows the model matrix, the root box.
 */
import {
  BoundingSphere,
  Cartesian3,
  Event,
  Math as CesiumMath,
  Matrix3,
  Matrix4,
  Transforms,
} from "cesium";
import { describe, expect, it, vi } from "vitest";

import type { Site, SiteAsset, SiteSummary } from "@twin/contracts";

import type { CameraController } from "@/cesium/CameraController";
import type { ClippingManager } from "@/cesium/ClippingManager";
import type { PerformanceManager } from "@/cesium/PerformanceManager";
import { createSiteTileset } from "@/cesium/providers/tiles";
import type * as Tiles from "@/cesium/providers/tiles";
import { SiteManager } from "@/cesium/SiteManager";
import type { SceneEvents } from "@/cesium/types";
import { Emitter } from "@/lib/emitter";

import { A1, A2, E1, E2, SITE_ID, twoEpochSite } from "./epochFixtures";

const LON = -123.88;
const LAT = 46.1335;
const ORIGIN = Cartesian3.fromDegrees(LON, LAT, 10);
const ROOT = Transforms.eastNorthUpToFixedFrame(ORIGIN);

function fakeTileset(root: Matrix4) {
  const world = (model: Matrix4) => Matrix4.multiply(model, root, new Matrix4());
  return {
    show: false,
    preloadWhenHidden: true,
    maximumScreenSpaceError: 16,
    tilesLoaded: true,
    totalMemoryUsageInBytes: 0,
    modelMatrix: Matrix4.clone(Matrix4.IDENTITY),
    get boundingSphere() {
      return BoundingSphere.transform(
        new BoundingSphere(new Cartesian3(0, 0, 2), 6),
        world(this.modelMatrix),
      );
    },
    root: {
      transform: root,
      updateTransform: vi.fn(),
      boundingVolume: {
        boundingVolume: {
          center: Matrix4.multiplyByPoint(root, new Cartesian3(0, 0, 2), new Cartesian3()),
          halfAxes: Matrix3.multiply(
            Matrix4.getMatrix3(root, new Matrix3()),
            Matrix3.fromScale(new Cartesian3(4, 4, 2)),
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
}
type FakeTileset = ReturnType<typeof fakeTileset>;

const made = vi.hoisted(() => ({
  /** The tileset made for each asset, by id. */
  tilesets: new Map<string, unknown>(),
  /** Each asset's root transform, when not the site's (an epoch out of the frame). */
  roots: new Map<string, unknown>(),
  /** An asset whose tileset is held back until `release` is called. */
  held: new Map<string, (release: () => void) => void>(),
}));
vi.mock("@/cesium/providers/tiles", async (importOriginal) => ({
  ...(await importOriginal<typeof Tiles>()),
  createSiteTileset: vi.fn((asset: SiteAsset) => {
    const tileset = fakeTileset((made.roots.get(asset.id) as Matrix4 | undefined) ?? ROOT);
    made.tilesets.set(asset.id, tileset);
    const hold = made.held.get(asset.id);
    if (!hold) return Promise.resolve(tileset);
    return new Promise((resolve) => hold(() => resolve(tileset)));
  }),
}));
vi.mock("@/cesium/scanView/ScanRendererHost", () => ({
  prefetchScanDestination: () => () => undefined,
}));

function harness(record: Site = twoEpochSite()) {
  made.tilesets.clear();
  made.roots.clear();
  made.held.clear();
  vi.mocked(createSiteTileset).mockClear();
  const d = 0.0005;
  const site = {
    ...record,
    centroid: { longitude: LON, latitude: LAT, height: null },
    boundary: {
      type: "Polygon",
      coordinates: [
        [
          [LON - d, LAT - d],
          [LON + d, LAT - d],
          [LON + d, LAT + d],
          [LON - d, LAT + d],
          [LON - d, LAT - d],
        ],
      ],
    },
  } as Site;
  const summary = {
    id: SITE_ID,
    slug: site.slug,
    name: site.name,
    centroid: site.centroid,
    areaM2: 10_000,
  } as unknown as SiteSummary;
  const events = new Emitter<SceneEvents>();
  const epochs: (string | null)[] = [];
  events.on("epoch", ({ epochId }) => epochs.push(epochId));
  const tilesetEvents = vi.fn();
  events.on("tilesets", tilesetEvents);
  const listener = { addEventListener: () => () => undefined };
  const viewer = {
    camera: {
      // At the site, 50 m up: it engages, and its model is drawn.
      positionWC: Cartesian3.fromDegrees(LON, LAT, 60),
      heading: 0,
      changed: listener,
      moveEnd: listener,
      frustum: { fovy: CesiumMath.PI_OVER_THREE },
    },
    canvas: { clientHeight: 800 },
    terrainProvider: {},
    scene: {
      globe: { getHeight: () => undefined },
      primitives: { add: vi.fn(), remove: vi.fn() },
      requestRender: vi.fn(),
      sampleHeightSupported: false,
    },
  };
  const camera = {
    pose: () => ({ longitude: LON, latitude: LAT, altitude: 50, height: 60 }),
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
  const manager = new SiteManager(
    viewer as never,
    events,
    camera as unknown as CameraController,
    { setFootprint: vi.fn() } as unknown as ClippingManager,
    performance as unknown as PerformanceManager,
  );
  manager.setCatalog([summary], () => Promise.resolve(site));
  const tileset = (id: string) => made.tilesets.get(id) as FakeTileset | undefined;
  return { manager, epochs, tilesetEvents, tileset, site };
}

/** Lets the manager's promise chains (tileset created, placed, shown) run. */
const settle = () => new Promise((done) => setTimeout(done, 0));

/**
 * Loads the site and lets it settle: the camera is at the site, so the catalog's proximity
 * check has begun loading it too, and the call that loses that race returns before the
 * model is in.
 */
async function open(manager: SiteManager): Promise<void> {
  await manager.activate(SITE_ID);
  await settle();
}

describe("a site with two epochs", () => {
  it("shows the latest epoch by default", async () => {
    const { manager } = harness();
    await open(manager);
    expect(manager.shownAsset()?.id).toBe(A2.id);
    expect(manager.shownAsset(SITE_ID)?.id).toBe(A2.id);
  });

  it("setEpoch shows the chosen epoch, hides the other, and reuses its tileset", async () => {
    const { manager, tileset } = harness();
    manager.setSplatRenderer("cesium");
    await open(manager);
    await manager.setEpoch(E1.id);
    expect(manager.shownAsset()?.id).toBe(A1.id);
    expect(tileset(A1.id)?.show).toBe(true);
    expect(tileset(A2.id)?.show).toBe(false);
    await manager.setEpoch(E2.id);
    expect(tileset(A2.id)?.show).toBe(true);
    expect(tileset(A1.id)?.show).toBe(false);
    // One tileset per asset, kept: switching back loads nothing again.
    const calls = vi.mocked(createSiteTileset).mock.calls.map(([asset]) => asset.id);
    expect(calls.filter((id) => id === A1.id)).toHaveLength(1);
    expect(calls.filter((id) => id === A2.id)).toHaveLength(1);
  });

  it("refuses an epoch it does not offer", async () => {
    const { manager, epochs } = harness();
    await open(manager);
    await manager.setEpoch("no-such-epoch");
    expect(manager.shownAsset()?.id).toBe(A2.id);
    expect(epochs).toEqual([E2.id]);
  });

  it("says which epoch it shows, and starts over at the latest when the site is loaded again", async () => {
    const { manager, epochs } = harness();
    await open(manager);
    expect(epochs).toEqual([E2.id]);
    await manager.setEpoch(E1.id);
    expect(epochs).toEqual([E2.id, E1.id]);
    // The date chosen stays across Splat / Mesh (this site has no mesh: nothing changes).
    await manager.setRepresentation("mesh");
    await manager.setRepresentation("gaussian-splat");
    expect(manager.shownAsset()?.id).toBe(A1.id);
    manager.deactivate();
    await open(manager);
    expect(epochs.at(-1)).toBe(E2.id);
    expect(manager.shownAsset()?.id).toBe(A2.id);
  });

  it("the dedicated renderer is pointed at the shown epoch, and drops the old one at once", async () => {
    const { manager, tilesetEvents } = harness();
    await open(manager);
    // PlayCanvas, the default, draws the scan: CesiumJS keeps it hidden.
    expect(manager.scanTarget()?.assetId).toBe(A2.id);
    expect(manager.scanTarget()?.key).toBe(`${SITE_ID}:${A2.id}`);
    let release = (): void => undefined;
    made.held.set(A1.id, (r) => (release = r));
    tilesetEvents.mockClear();
    const switching = manager.setEpoch(E1.id);
    await settle();
    // Told at once, before A1 is in: the overlay has nothing to draw rather than the old date.
    expect(tilesetEvents).toHaveBeenCalled();
    expect(manager.scanTarget()).toBeNull();
    release();
    await switching;
    expect(manager.scanTarget()?.assetId).toBe(A1.id);
    expect(manager.scanTarget()?.key).toBe(`${SITE_ID}:${A1.id}`);
  });

  it("a hidden epoch does not preload under CesiumJS", async () => {
    const { manager, tileset } = harness();
    manager.setSplatRenderer("cesium");
    await open(manager);
    await manager.setEpoch(E1.id);
    expect(tileset(A1.id)?.preloadWhenHidden).toBe(true);
    expect(tileset(A2.id)?.preloadWhenHidden).toBe(false);
    // Nor after the renderer is changed and changed back.
    manager.setSplatRenderer("spark");
    manager.setSplatRenderer("cesium");
    expect(tileset(A2.id)?.preloadWhenHidden).toBe(false);
    expect(tileset(A1.id)?.preloadWhenHidden).toBe(true);
  });

  it("a record without its epoch list still shows its newest model", async () => {
    const { manager, epochs } = harness(twoEpochSite({ epochs: undefined }));
    await open(manager);
    expect(manager.shownAsset()?.id).toBe(A2.id);
    expect(epochs).toEqual([E2.id]);
  });

  it("the load pill's Retry on an epoch's asset keeps that epoch", async () => {
    const { manager, epochs } = harness();
    await open(manager);
    await manager.setTemporalAsset(A1.id);
    expect(epochs.at(-1)).toBe(E1.id);
    expect(manager.shownAsset()?.id).toBe(A1.id);
    // It is the epoch that was chosen, not a pin: the date stays across Splat / Mesh.
    await manager.setRepresentation("mesh");
    await manager.setRepresentation("gaussian-splat");
    expect(manager.shownAsset()?.id).toBe(A1.id);
  });

  it("a site without epochs is shown as before, and says it shows none", async () => {
    const legacy = (id: string, extra: Partial<SiteAsset>) =>
      ({ ...A1, id, epochId: null, ...extra });
    const { manager, epochs } = harness(
      twoEpochSite({
        epochs: [],
        assets: [
          legacy("older", { defaultVisible: false, observedAt: "2025-01-01T00:00:00Z" }),
          legacy("default", { defaultVisible: true }),
        ],
      }),
    );
    await open(manager);
    expect(manager.shownAsset()?.id).toBe("default");
    expect(epochs).toEqual([null]);
  });
});
