/**
 * Two dates of a site drawn side by side in `SiteManager` (`setCompareDates`, ADR 0017): each
 * epoch's current model, split at the scene's divider by CesiumJS's own `splitDirection`, both
 * with the site's one placement; and the single date shown again when it ends, however it ends.
 *
 * E1 (Mar 1, the drone capture that made the site's frame) and E2 (Jun 1, a phone video) each
 * have a gaussian splat, A1 and A2 (epochFixtures.ts). The tilesets are faked down to what the
 * manager reads, as in siteEpochs.test.ts.
 */
import {
  BoundingSphere,
  Cartesian3,
  Event,
  Math as CesiumMath,
  Matrix3,
  Matrix4,
  SplitDirection,
  Transforms,
} from "cesium";
import { describe, expect, it, vi } from "vitest";

import type { Site, SiteAsset, SiteSummary } from "@twin/contracts";

import type { CameraController } from "@/cesium/CameraController";
import type { ClippingManager } from "@/cesium/ClippingManager";
import type { PerformanceManager } from "@/cesium/PerformanceManager";
import type * as Tiles from "@/cesium/providers/tiles";
import type * as Footprints from "@/cesium/scanFootprint";
import { SiteManager } from "@/cesium/SiteManager";
import type { SceneEvents } from "@/cesium/types";
import { Emitter } from "@/lib/emitter";
import type { CompareDates } from "@/lib/epochs";

import { A1, A2, E1, E2, SITE_ID, epoch, splat, twoEpochSite } from "./epochFixtures";

const LON = -123.88;
const LAT = 46.1335;
const ROOT = Transforms.eastNorthUpToFixedFrame(Cartesian3.fromDegrees(LON, LAT, 10));

function fakeTileset(root: Matrix4) {
  const world = (model: Matrix4) => Matrix4.multiply(model, root, new Matrix4());
  return {
    show: false,
    preloadWhenHidden: true,
    splitDirection: SplitDirection.NONE as SplitDirection,
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
    // The root tile, as the clip's canopy reads it (`loadCanopy`; its parsing is mocked).
    resource: {
      fetchJson: () => Promise.resolve({ root: { content: { uri: "splat.glb" } } }),
      getDerivedResource: () => ({ fetchArrayBuffer: () => Promise.resolve(new ArrayBuffer(8)) }),
    },
  };
}
type FakeTileset = ReturnType<typeof fakeTileset>;

const made = vi.hoisted(() => ({
  tilesets: new Map<string, unknown>(),
  /** An asset whose tileset is held back until `release` is called. */
  held: new Map<string, (release: () => void) => void>(),
}));
vi.mock("@/cesium/providers/tiles", async (importOriginal) => ({
  ...(await importOriginal<typeof Tiles>()),
  createSiteTileset: vi.fn((asset: SiteAsset) => {
    const tileset = fakeTileset(ROOT);
    made.tilesets.set(asset.id, tileset);
    const hold = made.held.get(asset.id);
    if (!hold) return Promise.resolve(tileset);
    return new Promise((resolve) => hold(() => resolve(tileset)));
  }),
}));
// A scan's tall things, outlined for the clip under it (scanFootprint.ts): one, always.
vi.mock("@/lib/spzPositions", () => ({ spzPickData: () => Promise.resolve({}) }));
vi.mock("@/view/glb", () => ({ spzFromGlb: () => ({}) }));
vi.mock("@/cesium/scanFootprint", async (importOriginal) => ({
  ...(await importOriginal<typeof Footprints>()),
  canopyOutlines: () => [[[0, 0]]],
  groundOf: () => 0,
  outlinesOnGlobe: () => [],
  withOutlines: (footprint: unknown) => ({ withCanopy: footprint }),
}));
vi.mock("@/cesium/scanView/ScanRendererHost", () => ({
  prefetchScanDestination: () => () => undefined,
}));

function harness(record: Site = twoEpochSite(), others: Site[] = []) {
  made.tilesets.clear();
  made.held.clear();
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
  /** Every `compare-dates` the scene said, in order. */
  const said: (CompareDates | null)[] = [];
  events.on("compare-dates", ({ dates }) => said.push(dates));
  const epochs: (string | null)[] = [];
  events.on("epoch", ({ epochId }) => epochs.push(epochId));
  const listener = { addEventListener: () => () => undefined };
  const viewer = {
    camera: {
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
  /** Where the camera is: at the site, 50 m up, unless a test moves it. */
  const at = { altitude: 50 };
  const camera = {
    pose: () => ({
      longitude: LON,
      latitude: LAT,
      altitude: at.altitude,
      height: at.altitude + 10,
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
  const loads: (string | null)[] = [];
  events.on("site-load", ({ load }) => loads.push(load?.phase ?? null));
  const manager = new SiteManager(
    viewer as never,
    events,
    camera as unknown as CameraController,
    clipping as unknown as ClippingManager,
    performance as unknown as PerformanceManager,
  );
  const records = [site, ...others.map((other) => ({ ...site, ...other }))];
  manager.setCatalog(
    records.map((r) => ({ ...summary, id: r.id, slug: r.slug, name: r.name })),
    (id) => Promise.resolve(records.find((r) => r.id === id) ?? null),
  );
  const tileset = (id: string) => made.tilesets.get(id) as FakeTileset | undefined;
  /** Each loaded model: drawn or not, and which side of the divider it keeps. */
  const drawn = () =>
    Object.fromEntries(
      [...made.tilesets.keys()].sort().map((id) => {
        const t = tileset(id);
        return [id, { show: t?.show, split: t?.splitDirection }];
      }),
    );
  /** Moves the camera to `altitude` m and lets the scene take it in (`checkProximity`). */
  const fly = (altitude: number) => {
    at.altitude = altitude;
    viewer.camera.positionWC = Cartesian3.fromDegrees(LON, LAT, altitude + 10);
    (manager as unknown as { checkProximity(force: boolean): void }).checkProximity(true);
  };
  return { manager, said, epochs, tileset, drawn, site, clipping, loads, fly };
}

const settle = () => new Promise((done) => setTimeout(done, 0));

/** Loads the site under CesiumJS, which draws the split, and lets it settle. */
async function open(manager: SiteManager, renderer: "cesium" | "playcanvas" = "cesium") {
  manager.setSplatRenderer(renderer);
  await manager.activate(SITE_ID);
  await settle();
}

const BOTH = { left: E1.id, right: E2.id };
const ALONE = { show: false, split: SplitDirection.NONE };

describe("comparing two dates of a site", () => {
  it("draws both dates' current models, left and right of the divider, with one placement", async () => {
    const { manager, said, tileset, drawn } = harness();
    await open(manager);
    await manager.setCompareDates(BOTH);
    await settle();
    expect(drawn()).toEqual({
      [A1.id]: { show: true, split: SplitDirection.LEFT },
      [A2.id]: { show: true, split: SplitDirection.RIGHT },
    });
    expect(said).toEqual([BOTH]);
    expect(manager.compareDates()).toEqual(BOTH);
    expect(manager.compareDates(SITE_ID)).toEqual(BOTH);
    // The site frame's own date and the later one share its model matrix (followAnchor).
    expect(manager.sharesPlacement(SITE_ID, A2.id)).toBe(true);
    expect(Matrix4.equals(tileset(A1.id)?.modelMatrix, tileset(A2.id)?.modelMatrix)).toBe(true);
    // Either way round.
    await manager.setCompareDates({ left: E2.id, right: E1.id });
    expect(drawn()).toEqual({
      [A1.id]: { show: true, split: SplitDirection.RIGHT },
      [A2.id]: { show: true, split: SplitDirection.LEFT },
    });
  });

  it("turned off, shows the one date chosen before, whole, and the other hidden", async () => {
    const { manager, said, epochs, drawn } = harness();
    await open(manager);
    await manager.setEpoch(E1.id);
    await manager.setCompareDates(BOTH);
    // The single date is kept while two are drawn: it is what comes back.
    expect(manager.shownAsset()?.id).toBe(A1.id);
    await manager.setCompareDates(null);
    expect(drawn()).toEqual({
      [A1.id]: { show: true, split: SplitDirection.NONE },
      [A2.id]: ALONE,
    });
    expect(said).toEqual([BOTH, null]);
    expect(manager.compareDates()).toBeNull();
    expect(epochs.at(-1)).toBe(E1.id);
    // Off again changes nothing and says nothing.
    await manager.setCompareDates(null);
    expect(said).toEqual([BOTH, null]);
  });

  it("refuses one date twice, a date it does not offer, and another renderer's scan", async () => {
    const { manager, said, drawn } = harness();
    await open(manager);
    const before = drawn();
    await manager.setCompareDates({ left: E2.id, right: E2.id });
    await manager.setCompareDates({ left: E1.id, right: "no-such-epoch" });
    expect(drawn()).toEqual(before);
    expect(said).toEqual([]);

    const other = harness();
    await open(other.manager, "playcanvas");
    await other.manager.setCompareDates(BOTH);
    expect(other.said).toEqual([]);
    expect(other.manager.compareDates()).toBeNull();
    expect(other.manager.scanTarget()?.assetId).toBe(A2.id);
  });

  it("ends when another renderer is chosen: that renderer draws the one date, whole", async () => {
    const { manager, said, drawn } = harness();
    await open(manager);
    await manager.setCompareDates(BOTH);
    manager.setSplatRenderer("playcanvas");
    await settle();
    expect(said).toEqual([BOTH, null]);
    expect(drawn()).toEqual({ [A1.id]: ALONE, [A2.id]: ALONE });
    expect(manager.scanTarget()?.assetId).toBe(A2.id);
    // And back under CesiumJS it is the one date, not the two.
    manager.setSplatRenderer("cesium");
    await settle();
    expect(drawn()).toEqual({
      [A1.id]: ALONE,
      [A2.id]: { show: true, split: SplitDirection.NONE },
    });
  });

  it("ended by another renderer, the load pill speaks for the one date again, not a failed one", async () => {
    const { manager, loads } = harness();
    await open(manager);
    await manager.setCompareDates(BOTH);
    // The date the comparison added did not load: the pill says so.
    (manager as unknown as { failLoad(id: string, m: string, r: boolean): void }).failLoad(
      SITE_ID,
      "splat a1 did not load",
      true,
    );
    expect(loads.at(-1)).toBe("error");
    manager.setSplatRenderer("playcanvas");
    await settle();
    // The date shown alone loaded fine: its load is the record now.
    expect(loads.at(-1)).toBe("ready");
  });

  it("ends when one date is chosen", async () => {
    const { manager, said, drawn } = harness();
    await open(manager);
    await manager.setCompareDates(BOTH);
    await manager.setEpoch(E1.id);
    expect(said).toEqual([BOTH, null]);
    expect(drawn()).toEqual({
      [A1.id]: { show: true, split: SplitDirection.NONE },
      [A2.id]: ALONE,
    });
  });

  it("a third date shown before is hidden while two others are compared, and back after", async () => {
    const E3 = epoch("e3", {
      ordinal: 3,
      observedAt: "2026-09-01T15:00:00Z",
      currentAssetId: "a3",
    });
    const A3 = splat("a3", { epochId: "e3", createdAt: "2026-09-02T00:00:00Z" });
    const { manager, drawn } = harness(
      twoEpochSite({ epochs: [E1, E2, E3], assets: [A1, A2, A3] }),
    );
    await open(manager);
    expect(manager.shownAsset()?.id).toBe(A3.id);
    await manager.setCompareDates(BOTH);
    await settle();
    expect(drawn()).toEqual({
      [A1.id]: { show: true, split: SplitDirection.LEFT },
      [A2.id]: { show: true, split: SplitDirection.RIGHT },
      [A3.id]: ALONE,
    });
    await manager.setCompareDates(null);
    expect(drawn()).toEqual({
      [A1.id]: ALONE,
      [A2.id]: ALONE,
      [A3.id]: { show: true, split: SplitDirection.NONE },
    });
  });

  it("a date run again mid-comparison is compared as its new model", async () => {
    const { manager, site, said, drawn } = harness();
    await open(manager);
    await manager.setCompareDates(BOTH);
    // E2 was run again: its new asset is current; the old one is still listed, superseded.
    const rerun = splat("a2b", { epochId: "e2", createdAt: "2026-07-01T00:00:00Z" });
    manager.updateRecord({
      ...site,
      epochs: [E1, { ...E2, currentAssetId: "a2b" }],
      assets: [A1, { ...A2, validTo: "2026-07-01T00:00:00Z" }, rerun],
    });
    await settle();
    await settle();
    expect(said).toEqual([BOTH]);
    expect(drawn()).toEqual({
      [A1.id]: { show: true, split: SplitDirection.LEFT },
      [A2.id]: ALONE,
      a2b: { show: true, split: SplitDirection.RIGHT },
    });
  });

  it("ends when a compared date is withdrawn, and shows the date left", async () => {
    const { manager, site, said, drawn } = harness();
    await open(manager);
    await manager.setCompareDates(BOTH);
    manager.updateRecord({ ...site, epochs: [E1, { ...E2, status: "rejected" }] });
    await settle();
    expect(said).toEqual([BOTH, null]);
    expect(drawn()).toEqual({
      [A1.id]: { show: true, split: SplitDirection.NONE },
      [A2.id]: ALONE,
    });
  });

  it("ends when the site is unloaded, and a fresh load shows one date", async () => {
    const { manager, said, drawn } = harness();
    await open(manager);
    await manager.setCompareDates(BOTH);
    manager.deactivate();
    expect(said).toEqual([BOTH, null]);
    await open(manager);
    expect(manager.compareDates()).toBeNull();
    expect(drawn()[A2.id]).toEqual({ show: true, split: SplitDirection.NONE });
  });

  it("ends when another site becomes the primary one: no split is left on a site out of reach", async () => {
    const rock = splat("rock", { siteId: "site-rock" });
    const other = twoEpochSite({
      id: "site-rock",
      slug: "rock",
      name: "Rock",
      epochs: [],
      assets: [rock],
    });
    const { manager, said, drawn } = harness(twoEpochSite(), [other]);
    await open(manager);
    await manager.setCompareDates(BOTH);
    await manager.activate("site-rock");
    await settle();
    expect(said).toEqual([BOTH, null]);
    expect(manager.compareDates(SITE_ID)).toBeNull();
    const after = drawn();
    expect(after[A1.id]?.split).toBe(SplitDirection.NONE);
    expect(after[A2.id]?.split).toBe(SplitDirection.NONE);
    expect(after.rock?.split).toBe(SplitDirection.NONE);
  });

  it("ends when Splat / Mesh switches to a representation the dates do not both have", async () => {
    const mesh = { ...splat("m2", { epochId: "e2" }), representation: "mesh" } as SiteAsset;
    const { manager, said } = harness(twoEpochSite({ assets: [A1, A2, mesh] }));
    await open(manager);
    await manager.setCompareDates(BOTH);
    await manager.setRepresentation("mesh");
    expect(said).toEqual([BOTH, null]);
  });

  it("says which date a tileset draws, so a pick on either side is about that date", async () => {
    const { manager, tileset } = harness();
    await open(manager);
    const [march, june] = [tileset(A1.id), tileset(A2.id)];
    expect(manager.drawnAssetOf(june ?? {})?.id).toBe(A2.id);
    // The frame's own date, loaded hidden for the placement, is not drawn.
    expect(manager.drawnAssetOf(march ?? {})).toBeNull();
    await manager.setCompareDates(BOTH);
    expect(manager.drawnAssetOf(march ?? {})?.id).toBe(A1.id);
    expect(manager.drawnAssetOf(june ?? {})?.id).toBe(A2.id);
    expect(manager.drawnAssetOf({})).toBeNull();
  });

  it("once ended, says the other date is not drawn, so what was picked on it is let go", async () => {
    const { manager, tileset } = harness();
    await open(manager);
    const march = tileset(A1.id) ?? {};
    await manager.setCompareDates(BOTH);
    expect(manager.drawnAssetOf(march, SITE_ID)?.id).toBe(A1.id);
    await manager.setCompareDates(null);
    expect(manager.drawnAssetOf(march, SITE_ID)).toBeNull();
    expect(manager.drawnAssetOf(tileset(A2.id) ?? {}, SITE_ID)?.id).toBe(A2.id);
  });

  it("ends when an asset is asked for by id (the load pill's Retry)", async () => {
    const { manager, said, drawn } = harness();
    await open(manager);
    await manager.setCompareDates(BOTH);
    await manager.setTemporalAsset(A1.id);
    expect(said).toEqual([BOTH, null]);
    expect(drawn()).toEqual({
      [A1.id]: { show: true, split: SplitDirection.NONE },
      [A2.id]: ALONE,
    });
  });

  it("both dates are hidden when the camera leaves the site, and drawn again on its return", async () => {
    const { manager, drawn, fly } = harness();
    await open(manager);
    await manager.setCompareDates(BOTH);
    fly(100_000);
    expect(drawn()).toEqual({
      [A1.id]: { show: false, split: SplitDirection.LEFT },
      [A2.id]: { show: false, split: SplitDirection.RIGHT },
    });
    fly(50);
    expect(drawn()).toEqual({
      [A1.id]: { show: true, split: SplitDirection.LEFT },
      [A2.id]: { show: true, split: SplitDirection.RIGHT },
    });
  });

  it("the left date is hidden when the camera leaves while the right one is still loading", async () => {
    const E3 = epoch("e3", {
      ordinal: 3,
      observedAt: "2026-09-01T15:00:00Z",
      currentAssetId: "a3",
    });
    const A3 = splat("a3", { epochId: "e3", createdAt: "2026-09-02T00:00:00Z" });
    const { manager, tileset, fly } = harness(
      twoEpochSite({ epochs: [E1, E2, E3], assets: [A1, A2, A3] }),
    );
    await open(manager);
    let release = (): void => undefined;
    made.held.set(A2.id, (r) => (release = r));
    const comparing = manager.setCompareDates(BOTH);
    await settle();
    expect(tileset(A1.id)?.show).toBe(true);
    fly(100_000);
    expect(tileset(A1.id)?.show).toBe(false);
    release();
    await comparing;
    expect(tileset(A2.id)?.show).toBe(false);
    fly(50);
    expect(tileset(A1.id)?.show).toBe(true);
    expect(tileset(A2.id)?.show).toBe(true);
  });

  it("stays across Splat / Mesh when both dates have the representation", async () => {
    const mesh = (id: string, epochId: string) =>
      ({ ...splat(id, { epochId }), representation: "mesh" }) as SiteAsset;
    const [m1, m2] = [mesh("m1", E1.id), mesh("m2", E2.id)];
    const { manager, said, drawn } = harness(twoEpochSite({ assets: [A1, A2, m1, m2] }));
    await open(manager);
    await manager.setCompareDates(BOTH);
    await manager.setRepresentation("mesh");
    await settle();
    expect(said).toEqual([BOTH]);
    expect(drawn()).toMatchObject({
      [A1.id]: ALONE,
      [A2.id]: ALONE,
      m1: { show: true, split: SplitDirection.LEFT },
      m2: { show: true, split: SplitDirection.RIGHT },
    });
    await manager.setRepresentation("gaussian-splat");
    expect(drawn()).toMatchObject({
      [A1.id]: { show: true, split: SplitDirection.LEFT },
      [A2.id]: { show: true, split: SplitDirection.RIGHT },
      m1: ALONE,
      m2: ALONE,
    });
  });

  it("a date still loading when the comparison is turned off is not drawn when it arrives", async () => {
    const E3 = epoch("e3", {
      ordinal: 3,
      observedAt: "2026-09-01T15:00:00Z",
      currentAssetId: "a3",
    });
    const A3 = splat("a3", { epochId: "e3", createdAt: "2026-09-02T00:00:00Z" });
    const { manager, drawn } = harness(
      twoEpochSite({ epochs: [E1, E2, E3], assets: [A1, A2, A3] }),
    );
    await open(manager);
    let release = (): void => undefined;
    made.held.set(A2.id, (r) => (release = r));
    const on = manager.setCompareDates(BOTH);
    await settle();
    const off = manager.setCompareDates(null);
    release();
    await Promise.all([on, off]);
    await settle();
    expect(drawn()).toEqual({
      [A1.id]: ALONE,
      [A2.id]: ALONE,
      [A3.id]: { show: true, split: SplitDirection.NONE },
    });
  });

  it("on, off and on again while a date loads: drawn on its side once it arrives", async () => {
    const E3 = epoch("e3", {
      ordinal: 3,
      observedAt: "2026-09-01T15:00:00Z",
      currentAssetId: "a3",
    });
    const A3 = splat("a3", { epochId: "e3", createdAt: "2026-09-02T00:00:00Z" });
    const { manager, drawn } = harness(
      twoEpochSite({ epochs: [E1, E2, E3], assets: [A1, A2, A3] }),
    );
    await open(manager);
    let release = (): void => undefined;
    made.held.set(A2.id, (r) => (release = r));
    const steps = [manager.setCompareDates(BOTH)];
    await settle();
    steps.push(manager.setCompareDates(null));
    await settle();
    steps.push(manager.setCompareDates(BOTH));
    release();
    await Promise.all(steps);
    await settle();
    expect(drawn()).toEqual({
      [A1.id]: { show: true, split: SplitDirection.LEFT },
      [A2.id]: { show: true, split: SplitDirection.RIGHT },
      [A3.id]: ALONE,
    });
  });

  it("the clip under two dates takes the right-hand date's tall things, as it is that date's", async () => {
    const clips = (asset: SiteAsset, footprint: unknown = null) =>
      ({
        ...asset,
        footprint,
        renderConfig: { ...asset.renderConfig, clipsWorld: true },
      }) as SiteAsset;
    const june = {
      type: "Polygon",
      coordinates: [
        [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 0],
        ],
      ],
    };
    // A third date is shown first, so June's tall things are read only once it is compared.
    const E3 = epoch("e3", {
      ordinal: 3,
      observedAt: "2026-09-01T15:00:00Z",
      currentAssetId: "a3",
    });
    const A3 = splat("a3", { epochId: "e3", createdAt: "2026-09-02T00:00:00Z" });
    const { manager, clipping } = harness(
      twoEpochSite({ epochs: [E1, E2, E3], assets: [clips(A1), clips(A2, june), clips(A3)] }),
    );
    await open(manager);
    // March is the one date chosen; then the two are compared, June on the right.
    await manager.setEpoch(E1.id);
    await settle();
    await manager.setCompareDates(BOTH);
    for (let i = 0; i < 5; i++) await settle();
    expect(clipping.setFootprint.mock.calls.at(-1)?.[1]).toEqual({ withCanopy: june });
  });

  it("a date hidden by the comparison is let go, drawn longest ago first, once splats outgrow the budget", async () => {
    // The scene's one-second tick (with the calibration) is when the budget is checked.
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
    try {
      const E3 = epoch("e3", {
        ordinal: 3,
        observedAt: "2026-09-01T15:00:00Z",
        currentAssetId: "a3",
      });
      const A3 = splat("a3", { epochId: "e3", createdAt: "2026-09-02T00:00:00Z" });
      const { manager, tileset } = harness(
        twoEpochSite({ epochs: [E1, E2, E3], assets: [A1, A2, A3] }),
      );
      await open(manager);
      // September is shown; comparing March and June hides it.
      await manager.setCompareDates(BOTH);
      await settle();
      const ids = () => manager.loadedAssets().map((a) => a.assetId);
      // Within the budget, the hidden date is kept for switching back.
      tileset(A2.id)?.tileLoad.raiseEvent({ extras: { gaussians: 1_000 } });
      vi.advanceTimersByTime(1_000);
      expect(ids()).toContain(A3.id);
      // Past it (more splats than any device keeps), the hidden date goes.
      tileset(A2.id)?.tileLoad.raiseEvent({ extras: { gaussians: 100_000_000 } });
      vi.advanceTimersByTime(1_000);
      expect(ids()).not.toContain(A3.id);
      // Ended, September is loaded again and drawn; June, hidden now, goes next. March is the
      // frame's anchor, whose placement the others copy: it stays.
      await manager.setCompareDates(null);
      await settle();
      expect(tileset(A3.id)?.show).toBe(true);
      vi.advanceTimersByTime(1_000);
      expect(ids().sort()).toEqual([A1.id, A3.id]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("the load pill follows a date the comparison has to load, until both are in", async () => {
    const E3 = epoch("e3", {
      ordinal: 3,
      observedAt: "2026-09-01T15:00:00Z",
      currentAssetId: "a3",
    });
    const A3 = splat("a3", { epochId: "e3", createdAt: "2026-09-02T00:00:00Z" });
    const { manager, loads, tileset } = harness(
      twoEpochSite({ epochs: [E1, E2, E3], assets: [A1, A2, A3] }),
    );
    await open(manager);
    expect(loads.at(-1)).toBe("ready");
    let release = (): void => undefined;
    made.held.set(A2.id, (r) => (release = r));
    const comparing = manager.setCompareDates(BOTH);
    await settle();
    // March is in already; June is being made: the site's model is loading again.
    expect(loads.at(-1)).toBe("model");
    const june = tileset(A2.id);
    if (june) june.tilesLoaded = false;
    release();
    await comparing;
    expect(loads.at(-1)).toBe("streaming");
    if (june) june.tilesLoaded = true;
    june?.allTilesLoaded.raiseEvent();
    expect(loads.at(-1)).toBe("ready");
  });
});
