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
  };
}
type FakeTileset = ReturnType<typeof fakeTileset>;

const made = vi.hoisted(() => ({ tilesets: new Map<string, unknown>() }));
vi.mock("@/cesium/providers/tiles", async (importOriginal) => ({
  ...(await importOriginal<typeof Tiles>()),
  createSiteTileset: vi.fn((asset: SiteAsset) => {
    const tileset = fakeTileset(ROOT);
    made.tilesets.set(asset.id, tileset);
    return Promise.resolve(tileset);
  }),
}));
vi.mock("@/cesium/scanView/ScanRendererHost", () => ({
  prefetchScanDestination: () => () => undefined,
}));

function harness(record: Site = twoEpochSite(), others: Site[] = []) {
  made.tilesets.clear();
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
  return { manager, said, epochs, tileset, drawn, site };
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
});
