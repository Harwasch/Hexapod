/**
 * A per-cell view rule in `SiteManager` (`setViewRule`, ADR 0017 "Per-cell view rules"): the
 * site's dates up to the one chosen drawn as one -- the frame anchor's model and every date
 * that shares its placement -- off by default, ended by Compare › Dates and ending it, refused
 * and ended off CesiumJS, and "As of" moving with the capture-date control. The splat budget
 * counts a date the anchor's primitive draws once.
 *
 * E1 (Mar 1, the frame's own) and E2 (Jun 1) each have a splat, A1 and A2 (epochFixtures.ts).
 * The tilesets are faked down to what the manager reads, as in siteCompareDates.test.ts; the
 * compositor itself (compositorSession.ts) waits for a splat primitive these fakes never get.
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
import { useViewRule } from "@/state/viewRule";
import type { ViewRule } from "@/lib/compositor";
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
  /** A root transform other than the site frame's, by asset id: a date out of the frame. */
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
  made.roots.clear();
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
  /** Every view rule the scene said (state/viewRule.ts), in order. */
  const rules: { rule: ViewRule | null; reason?: string }[] = [];
  useViewRule.setState({ active: null, ended: null });
  // Each harness's own list; one left subscribed after its test only writes to its own.
  useViewRule.subscribe((s, before) => {
    if (s.active === before.active && s.ended === before.ended) return;
    rules.push(s.ended ? { rule: null, reason: s.ended.reason } : { rule: s.active?.rule ?? null });
  });
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
      // The compositor's frame listener (compositorSession.ts).
      preUpdate: new Event(),
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
  return { manager, said, rules, epochs, tileset, drawn, site, clipping, loads, fly };
}

const settle = () => new Promise((done) => setTimeout(done, 0));

/** Loads the site under CesiumJS, which draws the split, and lets it settle. */
async function open(manager: SiteManager, renderer: "cesium" | "playcanvas" | "spark" = "cesium") {
  manager.setSplatRenderer(renderer);
  await manager.activate(SITE_ID);
  await settle();
}

const BOTH = { left: E1.id, right: E2.id };
const ALONE = { show: false, split: SplitDirection.NONE };
const DRAWN = { show: true, split: SplitDirection.NONE };

describe("a per-cell view rule", () => {
  it("draws the frame's own date and every later one that shares its placement, as of the latest", async () => {
    const { manager, rules, drawn } = harness();
    await open(manager);
    expect(useViewRule.getState().active).toBeNull();
    await manager.setViewRule("latest");
    await settle();
    expect(rules).toEqual([{ rule: "latest" }]);
    expect(useViewRule.getState().active).toEqual({ siteId: SITE_ID, rule: "latest" });
    expect(drawn()).toEqual({ [A1.id]: DRAWN, [A2.id]: DRAWN });
    // The date chosen is still the latest: the capture-date control reads "As of" it.
    expect(manager.shownAsset()?.id).toBe(A2.id);
    // Another rule is a uniform change, said, not a new load.
    await manager.setViewRule("ground-truth");
    expect(rules.at(-1)).toEqual({ rule: "ground-truth" });
    expect(drawn()).toEqual({ [A1.id]: DRAWN, [A2.id]: DRAWN });
  });

  it("as of an earlier date, draws the dates up to it", async () => {
    const { manager, rules, drawn } = harness();
    await open(manager);
    await manager.setViewRule("latest");
    await manager.setEpoch(E1.id);
    await settle();
    // Choosing a date moves "As of"; the rule stays.
    expect(rules).toEqual([{ rule: "latest" }]);
    expect(drawn()).toEqual({ [A1.id]: DRAWN, [A2.id]: ALONE });
    await manager.setEpoch(E2.id);
    await settle();
    expect(drawn()).toEqual({ [A1.id]: DRAWN, [A2.id]: DRAWN });
  });

  it("follows the latest request while the first is still loading its code", async () => {
    const { manager, rules, drawn } = harness();
    await open(manager);
    // Off asked before the first rule's code is in: the rule never starts.
    const first = manager.setViewRule("latest");
    await manager.setViewRule(null);
    await first;
    await settle();
    expect(rules).toEqual([]);
    expect(drawn()).toEqual({ [A1.id]: ALONE, [A2.id]: DRAWN });
    // Two rules asked one after the other: the second is what is drawn.
    const one = manager.setViewRule("latest");
    const two = manager.setViewRule("data-age");
    await Promise.all([one, two]);
    expect(useViewRule.getState().active?.rule).toBe("data-age");
  });

  it("turned off, draws the one date chosen again", async () => {
    const { manager, rules, drawn } = harness();
    await open(manager);
    await manager.setViewRule("latest");
    await manager.setViewRule(null);
    await settle();
    expect(rules).toEqual([{ rule: "latest" }, { rule: null }]);
    expect(drawn()).toEqual({ [A1.id]: ALONE, [A2.id]: DRAWN });
    await manager.setViewRule(null);
    expect(rules).toHaveLength(2);
  });

  it("and Compare › Dates are one or the other: each ends the one before, and says so", async () => {
    const { manager, rules, said, drawn } = harness();
    await open(manager);
    await manager.setCompareDates(BOTH);
    await manager.setViewRule("latest");
    await settle();
    expect(said).toEqual([BOTH, null]);
    expect(manager.compareDates()).toBeNull();
    expect(drawn()).toEqual({ [A1.id]: DRAWN, [A2.id]: DRAWN });
    await manager.setCompareDates(BOTH);
    await settle();
    expect(rules).toEqual([{ rule: "latest" }, { rule: null, reason: "compare" }]);
    expect(useViewRule.getState().active).toBeNull();
    expect(drawn()).toEqual({
      [A1.id]: { show: true, split: SplitDirection.LEFT },
      [A2.id]: { show: true, split: SplitDirection.RIGHT },
    });
  });

  it("is drawn under PlayCanvas, the scan it draws the date chosen, and kept across a switch", async () => {
    const other = harness();
    await open(other.manager, "playcanvas");
    await other.manager.setViewRule("latest");
    await settle();
    expect(other.rules).toEqual([{ rule: "latest" }]);
    // PlayCanvas draws every date (scanView/ruleDates.ts): CesiumJS's copies stay hidden, and
    // the scan PlayCanvas draws is the date chosen, which the rule is As of.
    expect(other.drawn()).toEqual({ [A1.id]: ALONE, [A2.id]: ALONE });
    expect(other.manager.scanTarget()?.assetId).toBe(A2.id);
    await other.manager.setViewRule(null);

    const { manager, rules, drawn } = harness();
    await open(manager);
    await manager.setViewRule("latest");
    manager.setSplatRenderer("playcanvas");
    await settle();
    expect(rules).toEqual([{ rule: "latest" }]);
    expect(drawn()).toEqual({ [A1.id]: ALONE, [A2.id]: ALONE });
    manager.setSplatRenderer("cesium");
    await settle();
    expect(rules).toEqual([{ rule: "latest" }]);
    expect(drawn()).toEqual({ [A1.id]: DRAWN, [A2.id]: DRAWN });
  });

  it("is refused under a renderer that draws none, and ends when one is chosen", async () => {
    const other = harness();
    await open(other.manager, "spark");
    await other.manager.setViewRule("latest");
    expect(other.rules).toEqual([]);
    expect(useViewRule.getState().active).toBeNull();

    const { manager, rules, drawn } = harness();
    await open(manager);
    await manager.setViewRule("latest");
    manager.setSplatRenderer("spark");
    await settle();
    expect(rules).toEqual([{ rule: "latest" }, { rule: null, reason: "changed" }]);
    expect(drawn()).toEqual({ [A1.id]: ALONE, [A2.id]: ALONE });
    expect(manager.scanTarget()?.assetId).toBe(A2.id);
  });

  it("is refused for a site with one date", async () => {
    const one = harness(twoEpochSite({ epochs: [E1], assets: [A1] }));
    await open(one.manager);
    await one.manager.setViewRule("latest");
    expect(one.rules).toEqual([]);
  });

  it("leaves out a date placed in another frame", async () => {
    const E3 = epoch("e3", {
      ordinal: 3,
      observedAt: "2026-09-01T15:00:00Z",
      currentAssetId: "a3",
    });
    const A3 = splat("a3", { epochId: "e3", createdAt: "2026-09-02T00:00:00Z" });
    const { manager, drawn } = harness(
      twoEpochSite({ epochs: [E1, E2, E3], assets: [A1, A2, A3] }),
    );
    made.roots.set(A3.id, Transforms.eastNorthUpToFixedFrame(Cartesian3.fromDegrees(LON, LAT, 40)));
    await open(manager);
    await manager.setViewRule("latest");
    await settle();
    await settle();
    expect(manager.sharesPlacement(SITE_ID, A3.id)).toBe(false);
    expect(drawn()).toEqual({ [A1.id]: DRAWN, [A2.id]: DRAWN, [A3.id]: ALONE });
  });

  it("ends, saying why, when one model is asked for by itself (a version, Retry)", async () => {
    const { manager, rules } = harness();
    await open(manager);
    await manager.setViewRule("latest");
    await manager.setTemporalAsset(A1.id);
    expect(rules).toEqual([{ rule: "latest" }, { rule: null, reason: "one-model" }]);
  });

  it("ends when the site is no longer the primary one or unloads", async () => {
    const { manager, rules } = harness();
    await open(manager);
    await manager.setViewRule("latest");
    manager.deactivate(SITE_ID);
    expect(rules).toEqual([{ rule: "latest" }, { rule: null }]);
  });

  it("lets go of no date while on: each is loaded to learn whether it shares the placement", async () => {
    // Evicted, a date after "As of" would be loaded again by the next draw, and again.
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
    try {
      const { manager, tileset } = harness();
      await open(manager);
      await manager.setViewRule("latest");
      await manager.setEpoch(E1.id);
      await settle();
      tileset(A2.id)?.tileLoad.raiseEvent({ extras: { gaussians: 100_000_000 } });
      vi.advanceTimersByTime(1_000);
      expect(manager.loadedAssets().map((a) => a.assetId)).toContain(A2.id);
    } finally {
      vi.useRealTimers();
    }
  });

  it("counts a date another primitive draws once in the splat budget", async () => {
    const { manager, tileset } = harness();
    await open(manager);
    await manager.setViewRule("latest");
    await settle();
    const host = { _liveSplats: 300 };
    Object.assign(tileset(A1.id) ?? {}, { gaussianSplatPrimitive: host });
    // The later date's own primitive draws nothing; its splats are in the host's 300.
    Object.assign(tileset(A2.id) ?? {}, {
      gaussianSplatPrimitive: { _liveSplats: 120, drawnBy: host },
    });
    const splatsDrawn = (manager as unknown as { splatsDrawn(): number }).splatsDrawn();
    expect(splatsDrawn).toBe(300);
  });
});
