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
  Cartographic,
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
import { sameRootFrame } from "@/cesium/tilesetScale";
import { Emitter } from "@/lib/emitter";
import { recentLogs } from "@/lib/log";

import { A1, A2, E1, E2, SITE_ID, epoch, splat, twoEpochSite } from "./epochFixtures";

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
  /** Assets whose tileset is gone for good (a 404). */
  missing: new Set<string>(),
}));
vi.mock("@/cesium/providers/tiles", async (importOriginal) => ({
  ...(await importOriginal<typeof Tiles>()),
  createSiteTileset: vi.fn((asset: SiteAsset) => {
    if (made.missing.has(asset.id))
      return Promise.reject(new Error("Request failed with status 404"));
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

function harness(record: Site = twoEpochSite(), options: { sampling?: boolean } = {}) {
  made.tilesets.clear();
  made.roots.clear();
  made.held.clear();
  made.missing.clear();
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
  const toasts: string[] = [];
  events.on("toast", ({ title }) => toasts.push(title));
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
      // The drawn ground, when sampled: 4 m above the ellipsoid everywhere.
      sampleHeightSupported: options.sampling ?? false,
      sampleHeightMostDetailed: vi.fn((points: Cartographic[]) =>
        Promise.resolve(points.map((p) => new Cartographic(p.longitude, p.latitude, 4))),
      ),
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
  return { manager, epochs, tilesetEvents, tileset, site, viewer, toasts };
}

/** The two-epoch site with each splat's render config changed as given. */
function withConfigs(a1: Record<string, unknown>, a2: Record<string, unknown> = {}): Site {
  return twoEpochSite({
    assets: [
      { ...A1, renderConfig: { ...A1.renderConfig, ...a1 } },
      { ...A2, renderConfig: { ...A2.renderConfig, ...a2 } },
    ],
  });
}

/** How high the model matrix puts the site's origin above the ellipsoid (m). */
function originHeight(tileset: FakeTileset | undefined): number {
  const moved = Matrix4.multiplyByPoint(
    tileset?.modelMatrix ?? Matrix4.IDENTITY,
    ORIGIN,
    new Cartesian3(),
  );
  return Cartographic.fromCartesian(moved).height;
}

const warned = (message: string) =>
  recentLogs().some((event) => event.level === "warn" && event.message === message);

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
    // A third date, the latest: E1 is loaded as the frame's anchor, E2 not until it is chosen.
    const E3 = epoch("e3", {
      ordinal: 3,
      observedAt: "2026-09-01T15:00:00Z",
      currentAssetId: "a3",
    });
    const A3 = splat("a3", { epochId: "e3", createdAt: "2026-09-02T00:00:00Z" });
    const { manager, tilesetEvents } = harness(
      twoEpochSite({ epochs: [E1, E2, E3], assets: [A1, A2, A3] }),
    );
    await open(manager);
    // PlayCanvas, the default, draws the scan: CesiumJS keeps it hidden.
    expect(manager.scanTarget()?.assetId).toBe(A3.id);
    expect(manager.scanTarget()?.key).toBe(`${SITE_ID}:${A3.id}`);
    let release = (): void => undefined;
    made.held.set(A2.id, (r) => (release = r));
    tilesetEvents.mockClear();
    const switching = manager.setEpoch(E2.id);
    await settle();
    // Told at once, before A2 is in: the overlay has nothing to draw rather than the old date.
    expect(tilesetEvents).toHaveBeenCalled();
    expect(manager.scanTarget()).toBeNull();
    release();
    await switching;
    expect(manager.scanTarget()?.assetId).toBe(A2.id);
    expect(manager.scanTarget()?.key).toBe(`${SITE_ID}:${A2.id}`);
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
    const legacy = (id: string, extra: Partial<SiteAsset>) => ({
      ...A1,
      id,
      epochId: null,
      ...extra,
    });
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

describe("every epoch of a site is drawn with one placement", () => {
  it("every epoch is drawn with the frame-defining epoch's model matrix", async () => {
    // The anchor is raised 3 m; the phone epoch says nothing. No terrain is needed.
    const { manager, tileset } = harness(
      withConfigs({ clampToGround: false, heightOffsetM: 3 }, { heightOffsetM: 0 }),
    );
    await open(manager);
    expect(manager.shownAsset()?.id).toBe(A2.id);
    // The anchor was loaded for this, hidden, and never streams.
    expect(tileset(A1.id)?.show).toBe(false);
    expect(tileset(A1.id)?.preloadWhenHidden).toBe(false);
    expect(Matrix4.equals(tileset(A1.id)?.modelMatrix, Matrix4.IDENTITY)).toBe(false);
    expect(Matrix4.equals(tileset(A2.id)?.modelMatrix, tileset(A1.id)?.modelMatrix)).toBe(true);
    expect(originHeight(tileset(A2.id))).toBeCloseTo(13, 3);
    // What reads the frame of the scan drawn saw the new matrix at once.
    expect(tileset(A2.id)?.root.updateTransform).toHaveBeenCalled();
    // And the Inspector can say so (EpochFacts).
    expect(manager.sharesPlacement(SITE_ID, A2.id)).toBe(true);
    expect(manager.sharesPlacement(SITE_ID, A1.id)).toBe(false);
    expect(manager.sharesPlacement(SITE_ID, "not-loaded")).toBeNull();
  });

  it("does not say a date shares the placement until it does, and says when that is decided", async () => {
    const { manager, tilesetEvents } = harness(
      withConfigs({ clampToGround: false, heightOffsetM: 3 }),
    );
    let release = (): void => undefined;
    made.held.set(A1.id, (r) => (release = r));
    await open(manager);
    // A2 is in, its anchor is not: whether it will share the placement is not known yet.
    expect(manager.sharesPlacement(SITE_ID, A2.id)).toBeNull();
    tilesetEvents.mockClear();
    release();
    await settle();
    expect(manager.sharesPlacement(SITE_ID, A2.id)).toBe(true);
    // The Inspector re-reads on this (EpochFacts).
    expect(tilesetEvents).toHaveBeenCalled();
  });

  it("a later change to the anchor's placement moves every epoch with it", async () => {
    const { manager, tileset, site } = harness(
      withConfigs({ clampToGround: false, heightOffsetM: 3 }),
    );
    await open(manager);
    const raised = {
      ...A1,
      renderConfig: { ...A1.renderConfig, clampToGround: false, heightOffsetM: 5 },
    };
    manager.updateRecord({ ...site, assets: [raised, A2] });
    expect(originHeight(tileset(A1.id))).toBeCloseTo(15, 3);
    expect(Matrix4.equals(tileset(A2.id)?.modelMatrix, tileset(A1.id)?.modelMatrix)).toBe(true);
    // A change to the follower's own record does not rest it on its own.
    const nudged = { ...A2, renderConfig: { ...A2.renderConfig, heightOffsetM: 9 } };
    manager.updateRecord({ ...site, assets: [raised, nudged] });
    expect(originHeight(tileset(A2.id))).toBeCloseTo(15, 3);
  });

  it("an epoch whose tiles are not in the site frame is placed on its own", async () => {
    const { manager, tileset } = harness(withConfigs({ clampToGround: false, heightOffsetM: 3 }));
    // A2's root is 10 m east of the frame's.
    made.roots.set(
      A2.id,
      Matrix4.multiply(ROOT, Matrix4.fromTranslation(new Cartesian3(10, 0, 0)), new Matrix4()),
    );
    let release = (): void => undefined;
    made.held.set(A1.id, (r) => (release = r));
    await open(manager);
    // Before the guard has run, it is not said to share anything.
    expect(manager.sharesPlacement(SITE_ID, A2.id)).toBeNull();
    release();
    await settle();
    expect(Matrix4.equals(tileset(A2.id)?.modelMatrix, Matrix4.IDENTITY)).toBe(true);
    expect(warned("an epoch's tiles are not in its site's frame; placed on its own")).toBe(true);
    expect(manager.sharesPlacement(SITE_ID, A2.id)).toBe(false);
  });

  it("an epoch whose anchor did not load is placed on its own", async () => {
    const { manager, tileset, toasts } = harness(
      withConfigs({}, { clampToGround: false, heightOffsetM: 2 }),
    );
    made.missing.add(A1.id);
    await open(manager);
    // Nobody chose that date: its failure is the log's, not a toast's.
    expect(toasts).toEqual([]);
    expect(manager.shownAsset()?.id).toBe(A2.id);
    expect(originHeight(tileset(A2.id))).toBeCloseTo(12, 3);
    expect(warned("the site frame's own epoch did not load; this date is placed on its own")).toBe(
      true,
    );
  });

  it("the anchor shown alone is placed exactly as before", async () => {
    const { manager, tileset } = harness(withConfigs({ clampToGround: false, heightOffsetM: 3 }));
    await open(manager);
    await manager.setEpoch(E1.id);
    expect(manager.shownAsset()?.id).toBe(A1.id);
    // Its own placement: its own 3 m about its own frame.
    expect(originHeight(tileset(A1.id))).toBeCloseTo(13, 3);
    expect(Matrix4.equals(tileset(A2.id)?.modelMatrix, tileset(A1.id)?.modelMatrix)).toBe(true);
  });

  it("the anchor rests on the drawn ground, never on another date of its site", async () => {
    const { manager, tileset, viewer } = harness(withConfigs({ clampToGround: true }), {
      sampling: true,
    });
    await open(manager);
    await vi.waitFor(() => expect(viewer.scene.sampleHeightMostDetailed).toHaveBeenCalled());
    await settle();
    const [, excluded] = viewer.scene.sampleHeightMostDetailed.mock.calls.at(-1) as unknown as [
      unknown,
      unknown[],
    ];
    expect(excluded).toContain(tileset(A1.id));
    expect(excluded).toContain(tileset(A2.id));
    // Rested (the box's lowest corner, 10 m up, on the 4 m ground), and the phone epoch with it.
    await vi.waitFor(() => expect(originHeight(tileset(A1.id))).toBeCloseTo(4, 3));
    expect(Matrix4.equals(tileset(A2.id)?.modelMatrix, tileset(A1.id)?.modelMatrix)).toBe(true);
  });

  it("sameRootFrame: a millimetre and 1e-9 in rotation are one frame; 1.1 mm or a 1e-8 turn is another", () => {
    const base = Matrix4.toArray(ROOT);
    const moved = (i: number, by: number) => base.map((v, j) => (j === i ? v + by : v));
    expect(sameRootFrame(base, base)).toBe(true);
    expect(sameRootFrame(undefined, undefined)).toBe(true);
    // Under a millimetre: 0.001 itself is not one in doubles 2,500 km from the centre of the earth.
    expect(sameRootFrame(moved(12, 0.00099), base)).toBe(true);
    expect(sameRootFrame(moved(0, 1e-9), base)).toBe(true);
    expect(sameRootFrame(moved(13, 0.0011), base)).toBe(false);
    expect(sameRootFrame(moved(5, 1e-8), base)).toBe(false);
    expect(sameRootFrame(base, undefined)).toBe(false);
  });
});

describe("a loaded site takes a fresh record of its dates", () => {
  it("a date published while the site is loaded is shown, and can be chosen", async () => {
    const { manager, site, epochs, viewer } = harness(twoEpochSite({ epochs: [E1], assets: [A1] }));
    await open(manager);
    expect(manager.shownAsset()?.id).toBe(A1.id);
    // The site's record is fetched again (watchSiteRecords): E2 has been published since.
    manager.updateRecord({ ...site, epochs: [E1, E2], assets: [A1, A2] });
    await settle();
    // The latest is what is shown when no date was chosen; and either can now be chosen.
    expect(manager.shownAsset()?.id).toBe(A2.id);
    expect(epochs.at(-1)).toBe(E2.id);
    await manager.setEpoch(E1.id);
    expect(manager.shownAsset()?.id).toBe(A1.id);
    await manager.setEpoch(E2.id);
    expect(manager.shownAsset()?.id).toBe(A2.id);
    expect(viewer.scene.primitives.remove).not.toHaveBeenCalled();
  });

  it("a date chosen stays chosen when another is published", async () => {
    const { manager, site } = harness();
    await open(manager);
    await manager.setEpoch(E1.id);
    const E3 = epoch("e3", {
      ordinal: 3,
      observedAt: "2026-09-01T15:00:00Z",
      currentAssetId: "a3",
    });
    const A3 = splat("a3", { epochId: "e3", createdAt: "2026-09-02T00:00:00Z" });
    manager.updateRecord({ ...site, epochs: [E1, E2, E3], assets: [A1, A2, A3] });
    await settle();
    expect(manager.shownAsset()?.id).toBe(A1.id);
    await manager.setEpoch(E3.id);
    expect(manager.shownAsset()?.id).toBe(A3.id);
  });

  it("a re-run's new model replaces the old one on screen, with the shared placement", async () => {
    const { manager, site, tileset, viewer } = harness(
      withConfigs({ clampToGround: false, heightOffsetM: 3 }),
    );
    await open(manager);
    const old = tileset(A2.id);
    // E2 was run again: its new asset is current, the old one superseded (and so not listed).
    const rerun = splat("a2b", { epochId: "e2", createdAt: "2026-07-01T00:00:00Z" });
    manager.updateRecord({
      ...site,
      epochs: [E1, { ...E2, currentAssetId: "a2b" }],
      assets: [site.assets[0] ?? A1, rerun],
    });
    await settle();
    await settle();
    expect(manager.shownAsset()?.id).toBe("a2b");
    expect(viewer.scene.primitives.remove).toHaveBeenCalledWith(old);
    expect(
      manager
        .loadedAssets()
        .map((a) => a.assetId)
        .sort(),
    ).toEqual([A1.id, "a2b"].sort());
    expect(manager.sharesPlacement(SITE_ID, "a2b")).toBe(true);
    expect(originHeight(tileset("a2b"))).toBeCloseTo(13, 3);
  });

  it("when the frame's own date is run again, the other dates follow its new model", async () => {
    const { manager, site, tileset } = harness(
      withConfigs({ clampToGround: false, heightOffsetM: 3 }),
    );
    await open(manager);
    const anchor = splat("a1b", {
      epochId: "e1",
      createdAt: "2026-07-01T00:00:00Z",
      renderConfig: { ...A1.renderConfig, clampToGround: false, heightOffsetM: 5 },
    });
    manager.updateRecord({
      ...site,
      epochs: [{ ...E1, currentAssetId: "a1b" }, E2],
      assets: [anchor, site.assets[1] ?? A2],
    });
    await settle();
    await settle();
    expect(manager.shownAsset()?.id).toBe(A2.id);
    expect(manager.sharesPlacement(SITE_ID, A2.id)).toBe(true);
    expect(originHeight(tileset("a1b"))).toBeCloseTo(15, 3);
    expect(Matrix4.equals(tileset(A2.id)?.modelMatrix, tileset("a1b")?.modelMatrix)).toBe(true);
  });

  it("a record that changes nothing of the dates moves nothing", async () => {
    const { manager, site, tilesetEvents } = harness();
    await open(manager);
    const before = vi.mocked(createSiteTileset).mock.calls.length;
    tilesetEvents.mockClear();
    manager.updateRecord({ ...site, name: "Renamed" });
    await settle();
    expect(vi.mocked(createSiteTileset).mock.calls.length).toBe(before);
    expect(tilesetEvents).not.toHaveBeenCalled();
  });
});
