/**
 * A fly-to leaves on the click and loads its site during the flight (SiteManager.flyTo).
 *
 * The camera used to wait for the site's record and its model before it moved at all, and for
 * ever when either stalled. These pin the new contract: the first leg leaves at once for the
 * best pose known, the record re-points it at the authored bookmark without restarting from
 * rest, a stall becomes a retryable error in the site's load record, and Retry finishes the
 * job. Without a bookmark, the terrain under the site and then the model's own placement
 * correct the aim, on the way, on landing, or up to 20 s after it -- never once somebody has
 * the camera. CesiumJS's camera and terrain are faked; the poses and the easing are checked.
 */
import {
  BoundingSphere,
  Cartesian3,
  Cartographic,
  Math as CesiumMath,
  Event,
  Matrix4,
  Transforms,
} from "cesium";
import type * as Cesium from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Site, SiteAsset, SiteSummary } from "@twin/contracts";
import { boundingRadiusM } from "@twin/geo";

import type { CameraController, FlyOptions } from "@/cesium/CameraController";
import type { ClippingManager } from "@/cesium/ClippingManager";
import { easingSlope } from "@/cesium/flightRetarget";
import type { PerformanceManager } from "@/cesium/PerformanceManager";
import type { MeasuredGround } from "@/cesium/placement";
import type * as Tiles from "@/cesium/providers/tiles";
import { SiteManager } from "@/cesium/SiteManager";
import type { SceneEvents } from "@/cesium/types";
import { Emitter } from "@/lib/emitter";
import type { SiteLoad } from "@/state/sites";

/** The scan renderer's destination prefetch, recorded instead of run. */
const prefetch = vi.hoisted(() => ({ destinations: [] as unknown[], cancelled: 0 }));
vi.mock("@/cesium/scanView/ScanRendererHost", () => ({
  prefetchScanDestination: (destination: unknown) => {
    prefetch.destinations.push(destination);
    return () => {
      prefetch.cancelled += 1;
    };
  },
}));

/**
 * The terrain at full detail (`sampleTerrainMostDetailed`), as each test sets it: none at all
 * unless one does, as with no terrain layer. `pending` holds the answer back until resolved;
 * `point` holds back only a single point's, the fly-to's own aim (the clamp asks for more).
 */
const terrain = vi.hoisted(() => ({
  height: undefined as number | undefined,
  pending: null as Promise<number> | null,
  point: null as Promise<number> | null,
}));
vi.mock("cesium", async (importOriginal) => ({
  ...(await importOriginal<typeof Cesium>()),
  sampleTerrainMostDetailed: async (_provider: unknown, positions: Cesium.Cartographic[]) => {
    const held = (positions.length === 1 ? terrain.point : null) ?? terrain.pending;
    const height = held ? await held : terrain.height;
    if (height === undefined) throw new Error("no terrain here");
    for (const position of positions) position.height = height;
    return positions;
  },
}));

/** The models a site's assets load as: a placed scan, below (`placedScan`). */
const tilesets = vi.hoisted(() => ({ next: null as (() => unknown) | null }));
vi.mock("@/cesium/providers/tiles", async (importOriginal) => ({
  ...(await importOriginal<typeof Tiles>()),
  createSiteTileset: () =>
    tilesets.next ? Promise.resolve(tilesets.next()) : Promise.reject(new Error("no model")),
}));

const SITE_ID = "11111111-1111-4111-8111-111111111111";
const LON = -122.13;
const LAT = 47.64;

const summary = {
  id: SITE_ID,
  slug: "yard",
  name: "Yard",
  centroid: { longitude: LON, latitude: LAT, height: 100 },
  areaM2: 40_000,
} as unknown as SiteSummary;

const BOOKMARK = {
  id: "b",
  name: "Default",
  isDefault: true,
  longitude: LON + 0.001,
  latitude: LAT - 0.002,
  height: 450,
  heading: 100,
  pitch: -25,
};

/** The site's record: its footprint a square `d` degrees either side of the centre. */
function site(bookmarks = [BOOKMARK], d = 0.001): Site {
  return {
    id: SITE_ID,
    slug: "yard",
    name: "Yard",
    centroid: { longitude: LON, latitude: LAT, height: 100 },
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
    // No assets: nothing for the test to stream, so the load ends at its record.
    assets: [],
    cameraBookmarks: bookmarks,
  } as unknown as Site;
}

interface Flight {
  longitude: number;
  latitude: number;
  height: number;
  options: FlyOptions;
}

function harness() {
  const events = new Emitter<SceneEvents>();
  const flights: Flight[] = [];
  /** Every sphere a pose was asked for, in order: what the flight framed. */
  const spheres: BoundingSphere[] = [];
  const loads: SiteLoad[] = [];
  const toasts: string[] = [];
  /** What the load pill follows (`site-flight`), in order. */
  const flightSites: (string | null)[] = [];
  events.on("site-load", ({ load }) => {
    if (load) loads.push(load);
  });
  events.on("site-flight", (id) => flightSites.push(id));
  events.on("toast", (toast) => toasts.push(toast.title));
  const listener = { addEventListener: () => () => undefined };
  const viewer = {
    camera: {
      positionWC: Cartesian3.fromDegrees(-110, 35, 18_000_000),
      heading: 0,
      changed: listener,
      moveEnd: listener,
    },
    scene: {
      globe: { getHeight: () => undefined },
      primitives: { add: vi.fn(), remove: vi.fn() },
      requestRender: vi.fn(),
      sampleHeightSupported: false,
    },
  };
  /** Where the camera is, as CameraController.pose says; a test moves it. */
  const at = { longitude: -110, latitude: 35, altitude: 18_000_000 };
  const camera = {
    pose: () => ({ ...at, height: at.altitude, metersPerPixel: 10_000 }),
    /** Every flight started, as CameraController counts them; a test adds another's. */
    flights: 0,
    isMoving: false,
    refreshPose: vi.fn(),
    setObjectScale: vi.fn(),
    durationFor: () => 4,
    sphereArrival: (sphere: BoundingSphere) => {
      spheres.push(sphere);
      const center = Cartographic.fromCartesian(sphere.center);
      return {
        longitude: CesiumMath.toDegrees(center.longitude),
        latitude: CesiumMath.toDegrees(center.latitude),
        height: center.height + sphere.radius * 2.3,
        heading: 100,
        pitch: -45,
      };
    },
    flyTo: vi.fn((longitude: number, latitude: number, height: number, options: FlyOptions) => {
      camera.flights += 1;
      flights.push({ longitude, latitude, height, options });
    }),
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
  return { manager, flights, spheres, loads, toasts, events, flightSites, at, viewer, camera };
}

/** Cesium completing a leg: the camera is where the leg was going, and it says so. */
function land(viewer: ReturnType<typeof harness>["viewer"], flight: Flight | undefined): void {
  if (!flight) throw new Error("no such flight");
  viewer.camera.positionWC = Cartesian3.fromDegrees(
    flight.longitude,
    flight.latitude,
    flight.height,
  );
  flight.options.onComplete?.();
}

/** A promise the test settles by hand, as a slow or stalled API would. */
function deferred<T>() {
  let resolve: (value: T) => void = () => undefined;
  let reject: (error: unknown) => void = () => undefined;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

describe("SiteManager.flyTo", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    prefetch.destinations = [];
    prefetch.cancelled = 0;
  });
  afterEach(() => vi.useRealTimers());

  it("leaves on the click for the summary's pose, before the record has answered", async () => {
    const { manager, flights } = harness();
    const record = deferred<Site | null>();
    manager.setCatalog([summary], () => record.promise);
    void manager.flyTo(SITE_ID);
    await vi.advanceTimersByTimeAsync(1);
    expect(flights).toHaveLength(1);
    expect(flights[0]?.longitude).toBeCloseTo(LON, 3);
    expect(flights[0]?.latitude).toBeCloseTo(LAT, 3);
  });

  it("re-points the flight at the bookmark when the record arrives, without stopping first", async () => {
    const { manager, flights, loads } = harness();
    const record = deferred<Site | null>();
    manager.setCatalog([summary], () => record.promise);
    const done = manager.flyTo(SITE_ID);
    await vi.advanceTimersByTimeAsync(1_000);
    record.resolve(site());
    await done;

    expect(flights).toHaveLength(2);
    const second = flights[1];
    expect(second?.longitude).toBe(BOOKMARK.longitude);
    expect(second?.height).toBe(BOOKMARK.height);
    expect(second?.options.pitch).toBe(BOOKMARK.pitch);
    // A quarter of the way into a quadratic in-out leg the camera is moving; the new leg
    // leaves moving too, and lands when the old one would have.
    const easing = second?.options.easing;
    expect(easing && easingSlope(easing, 0)).toBeGreaterThan(0.1);
    expect(second?.options.durationS).toBeCloseTo(3, 1);
    expect(loads.at(-1)?.phase).toBe("ready");
    expect(loads.every((load) => load.flight)).toBe(true);
  });

  it("keeps a better pose that arrives after the leg's time is up but before it completed", async () => {
    const { manager, flights } = harness();
    const record = deferred<Site | null>();
    manager.setCatalog([summary], () => record.promise);
    const done = manager.flyTo(SITE_ID);
    // Past the first leg's 4 s, with Cesium not having called it complete (a hidden tab).
    await vi.advanceTimersByTimeAsync(5_000);
    expect(flights).toHaveLength(1);
    record.resolve(site());
    await done;
    // A fresh leg for the bookmark, from rest, rather than landing at the summary's centre.
    expect(flights).toHaveLength(2);
    expect(flights[1]?.longitude).toBe(BOOKMARK.longitude);
    expect(flights[1]?.height).toBe(BOOKMARK.height);
    const easing = flights[1]?.options.easing;
    expect(easing && easingSlope(easing, 0)).toBeCloseTo(0, 2);
    // The first leg's late completion, replaced, changes nothing.
    flights[0]?.options.onComplete?.();
    expect(flights).toHaveLength(2);
  });

  it("flies to the bookmarks the app last fetched, not the ones it kept from before", async () => {
    const { manager, flights } = harness();
    let fetched = 0;
    manager.setCatalog([summary], () => {
      fetched += 1;
      return Promise.resolve(site());
    });
    await manager.flyTo(SITE_ID);
    expect(flights.at(-1)?.longitude).toBe(BOOKMARK.longitude);
    // The default view is deleted and another saved as default (the site's query refetched;
    // the catalog list is unchanged, so no new catalog reaches the scene).
    const moved = { ...BOOKMARK, id: "c", longitude: LON - 0.003, height: 300 };
    manager.updateRecord(site([moved]));
    await manager.flyTo(SITE_ID);
    expect(flights.at(-1)?.longitude).toBe(moved.longitude);
    expect(flights.at(-1)?.height).toBe(300);
    // All of it from what the app fetched: one record request, the first flight's.
    expect(fetched).toBe(1);
    // A site not loaded yet leaves for the fresh record at once too.
    const other = { ...summary, id: "other" };
    manager.setCatalog([summary, other], () => Promise.reject(new Error("offline")));
    manager.updateRecord({ ...site([moved]), id: "other" });
    await manager.flyTo("other");
    expect(flights.at(-1)?.longitude).toBe(moved.longitude);
  });

  it("flies straight to the bookmark when the record is already known", async () => {
    const { manager, flights } = harness();
    manager.setCatalog([summary], () => Promise.resolve(site()));
    await manager.flyTo(SITE_ID);
    expect(flights).toHaveLength(1);
    expect(flights[0]?.longitude).toBe(BOOKMARK.longitude);
    // An ordinary flight: it leaves from rest.
    const easing = flights[0]?.options.easing;
    expect(easing && easingSlope(easing, 0)).toBeCloseTo(0, 2);
  });

  it("turns a stalled record into a retryable error, and Retry flies on to the bookmark", async () => {
    const { manager, flights, loads } = harness();
    let calls = 0;
    manager.setCatalog([summary], (_id, signal) => {
      calls += 1;
      if (calls === 1) {
        return new Promise<Site | null>((_, reject) =>
          signal?.addEventListener("abort", () => reject(new Error("aborted"))),
        );
      }
      return Promise.resolve(site());
    });
    const first = manager.flyTo(SITE_ID);
    await vi.advanceTimersByTimeAsync(12_000);
    await first;
    const failed = loads.at(-1);
    expect(failed?.phase).toBe("error");
    expect(failed?.retryable).toBe(true);
    expect(failed?.error).toContain("did not answer within 12 s");
    // The camera was not held back by the stall: it left for the summary's pose at once.
    expect(flights).toHaveLength(1);

    await manager.retry(SITE_ID);
    expect(calls).toBe(2);
    expect(flights.at(-1)?.longitude).toBe(BOOKMARK.longitude);
    expect(loads.at(-1)?.phase).toBe("ready");
  });

  it("names the flight's site for the load pill, through a failed record, until the camera leaves", async () => {
    const { manager, flights, loads, flightSites, at } = harness();
    let calls = 0;
    manager.setCatalog([summary], () => {
      calls += 1;
      return calls === 1
        ? Promise.reject(new Error("The server answered 503."))
        : Promise.resolve(site());
    });
    await manager.flyTo(SITE_ID);
    // From the click: the site never became active (its record failed), so this is the only
    // way the pill knows which record to read.
    expect(flightSites).toEqual([SITE_ID]);
    expect(loads.at(-1)).toMatchObject({ phase: "error", retryable: true, flight: true });
    expect(loads.at(-1)?.error).toContain("503");
    // The camera lands at the summary's pose: the pill keeps the site, with its Retry.
    Object.assign(at, { longitude: LON, latitude: LAT, altitude: 600 });
    flights[0]?.options.onComplete?.();
    expect(flightSites).toEqual([SITE_ID]);
    // Retry flies there again, now with the record.
    await manager.retry(SITE_ID);
    expect(flights.at(-1)?.longitude).toBe(BOOKMARK.longitude);
    expect(loads.at(-1)?.phase).toBe("ready");
    flights.at(-1)?.options.onComplete?.();
    expect(flightSites).toEqual([SITE_ID]);
    // The camera leaves (Whole Earth): the pill lets the site go.
    Object.assign(at, { longitude: -110, latitude: 35, altitude: 18_000_000 });
    flights.at(-1)?.options.onCancel?.();
    expect(flightSites).toEqual([SITE_ID, null]);
  });

  it("reports a site the catalog no longer has, without flying anywhere", async () => {
    const { manager, flights, toasts, loads } = harness();
    manager.setCatalog([summary], () => Promise.resolve(null));
    await manager.flyTo(SITE_ID);
    await vi.advanceTimersByTimeAsync(1);
    expect(flights).toHaveLength(0);
    expect(toasts).toContain("Site not found");
    expect(loads.at(-1)).toMatchObject({ phase: "error", retryable: false });
  });

  it("leaves a camera alone once somebody else has taken it", async () => {
    const { manager, flights } = harness();
    const record = deferred<Site | null>();
    manager.setCatalog([summary], () => record.promise);
    const done = manager.flyTo(SITE_ID);
    await vi.advanceTimersByTimeAsync(500);
    // Another fly-to (a search result, say) cancels this one.
    flights[0]?.options.onCancel?.();
    record.resolve(site());
    await done;
    expect(flights).toHaveLength(1);
  });

  it("prefetches each leg's destination for the scan renderer, and drops the ones abandoned", async () => {
    const { manager, flights } = harness();
    const record = deferred<Site | null>();
    manager.setCatalog([summary], () => record.promise);
    const done = manager.flyTo(SITE_ID);
    await vi.advanceTimersByTimeAsync(800);
    record.resolve(site());
    await done;
    // The summary's leg, then the bookmark's, which replaced it.
    expect(prefetch.destinations).toHaveLength(2);
    expect(prefetch.cancelled).toBe(1);
    const destination = prefetch.destinations[1] as {
      position: Cartesian3;
      heading: number;
      pitch: number;
    };
    expect(
      Cartesian3.distance(
        destination.position,
        Cartesian3.fromDegrees(BOOKMARK.longitude, BOOKMARK.latitude, BOOKMARK.height),
      ),
    ).toBeLessThan(1e-6);
    expect(destination.heading).toBeCloseTo(CesiumMath.toRadians(BOOKMARK.heading), 9);
    expect(destination.pitch).toBeCloseTo(CesiumMath.toRadians(BOOKMARK.pitch), 9);
    // Somebody else takes the camera: nothing is fetched for a destination nobody reaches.
    flights[1]?.options.onCancel?.();
    expect(prefetch.cancelled).toBe(2);
  });
});

/** A site the pipeline registered: no catalog height (a phone has none), no bookmark. */
const PLACED_ID = "22222222-2222-4222-8222-222222222222";
const placedSummary = {
  id: PLACED_ID,
  slug: "camp",
  name: "Camp",
  centroid: { longitude: LON, latitude: LAT, height: null },
  areaM2: 22_000,
} as unknown as SiteSummary;

/** Degrees for metres east and north of the site's placed point. */
const east = (m: number) => m / (111_320 * Math.cos((LAT * Math.PI) / 180));
const north = (m: number) => m / 111_320;

/** The capture's densest ground cells: a disc 20 m in radius around its origin, at z 0.1 m. */
const CELLS: MeasuredGround[] = Array.from({ length: 64 }, (_, i) => {
  const r = 20 * Math.sqrt((i + 0.5) / 64);
  const a = i * 2.399963;
  return { lon: LON + east(r * Math.cos(a)), lat: LAT + north(r * Math.sin(a)), height: 0.1 };
});

const SCAN = {
  id: "44444444-4444-4444-8444-444444444444",
  siteId: PLACED_ID,
  name: "Camp splat",
  representation: "gaussian-splat",
  source: { type: "3d-tiles-url", url: "https://example.invalid/runs/1/splat/tileset.json" },
  renderConfig: {
    clampToGround: true,
    groundSamples: CELLS,
    clipsWorld: true,
    clipFootprint: "catalog",
    heightOffsetM: 0,
  },
  defaultVisible: true,
} as unknown as SiteAsset;

/** Its record: the footprint is the manifest's box, 75 m either way with the floaters in it. */
function placedSite(assets: SiteAsset[] = []): Site {
  const [x, y] = [east(75), north(75)];
  return {
    ...site([], 0),
    id: PLACED_ID,
    slug: "camp",
    name: "Camp",
    centroid: { longitude: LON, latitude: LAT, height: null },
    boundary: {
      type: "Polygon",
      coordinates: [
        [
          [LON - x, LAT - y],
          [LON + x, LAT - y],
          [LON + x, LAT + y],
          [LON - x, LAT + y],
          [LON - x, LAT - y],
        ],
      ],
    },
    assets,
  } as unknown as Site;
}

/**
 * A pipeline scan's tileset, as far as SiteManager reads one. Its root transform is the
 * capture's east-north-up frame at the placed point, on the ellipsoid (a phone has no usable
 * height), and floaters have stretched its bounds to 128 m around a point 6 m west and 21 m
 * up: the Camp scan's. Like Cesium's, the bounds follow the model matrix the clamp sets.
 */
function placedScan() {
  const frame = Transforms.eastNorthUpToFixedFrame(Cartesian3.fromDegrees(LON, LAT, 0));
  const floaters = new Cartesian3(-6, 1.1, 20.9);
  let modelMatrix = Matrix4.clone(Matrix4.IDENTITY);
  const computed = () => Matrix4.multiply(modelMatrix, frame, new Matrix4());
  return {
    show: false,
    preloadWhenHidden: true,
    maximumScreenSpaceError: 16,
    tilesLoaded: true,
    totalMemoryUsageInBytes: 0,
    get modelMatrix() {
      return modelMatrix;
    },
    set modelMatrix(matrix: Matrix4) {
      modelMatrix = matrix;
    },
    root: {
      get computedTransform() {
        return computed();
      },
    },
    get boundingSphere() {
      return new BoundingSphere(
        Matrix4.multiplyByPoint(computed(), floaters, new Cartesian3()),
        128,
      );
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

describe("SiteManager.flyTo without a bookmark", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    terrain.height = undefined;
    terrain.pending = null;
    terrain.point = null;
    tilesets.next = null;
  });
  afterEach(() => vi.useRealTimers());

  it("re-points the flight on the way for the record's footprint", async () => {
    const { manager, flights, viewer } = harness();
    // Three kilometres out: the footprint is more than a few per cent of the way left.
    viewer.camera.positionWC = Cartesian3.fromDegrees(LON, LAT - 0.03, 1_500);
    const record = deferred<Site | null>();
    manager.setCatalog([summary], () => record.promise);
    const done = manager.flyTo(SITE_ID);
    await vi.advanceTimersByTimeAsync(1_000);
    record.resolve(site([], 0.004));
    await done;
    expect(flights).toHaveLength(2);
    const radius = boundingRadiusM(site([], 0.004).boundary);
    expect(flights[1]?.longitude).toBeCloseTo(LON, 6);
    expect(flights[1]?.height).toBeCloseTo(100 + radius * 2.3, 3);
    // Leaving at the speed the camera already had.
    const easing = flights[1]?.options.easing;
    expect(easing && easingSlope(easing, 0)).toBeGreaterThan(0.1);
    // It lands where it was going: nothing was kept for later.
    land(viewer, flights[1]);
    expect(flights).toHaveLength(2);
  });

  it("flies on landing to a correction too small to re-point for on the way", async () => {
    const { manager, flights, viewer } = harness();
    const record = deferred<Site | null>();
    manager.setCatalog([summary], () => record.promise);
    const done = manager.flyTo(SITE_ID);
    await vi.advanceTimersByTimeAsync(1_000);
    record.resolve(site([]));
    await done;
    // Tens of metres, against a leg from 18,000 km up: not worth a new leg on the way.
    expect(flights).toHaveLength(1);
    land(viewer, flights[0]);
    expect(flights).toHaveLength(2);
    const radius = Math.max(boundingRadiusM(site([]).boundary), 20);
    expect(flights[1]?.height).toBeCloseTo(100 + radius * 2.3, 3);
    expect(Math.abs((flights[1]?.height ?? 0) - (flights[0]?.height ?? 0))).toBeGreaterThan(10);
    // From rest: the camera had stopped.
    const easing = flights[1]?.options.easing;
    expect(easing && easingSlope(easing, 0)).toBeCloseTo(0, 2);
    land(viewer, flights[1]);
    expect(flights).toHaveLength(2);
  });

  it("aims at the terrain under a site the catalog has no height for, up to 20 s after landing", async () => {
    const { manager, flights, viewer } = harness();
    const late = deferred<number>();
    terrain.pending = late.promise;
    manager.setCatalog([placedSummary], () => Promise.resolve(placedSite()));
    await manager.flyTo(PLACED_ID);
    // The globe had no terrain loaded there: the first leg aims at the ellipsoid.
    expect(flights).toHaveLength(1);
    land(viewer, flights[0]);
    await vi.advanceTimersByTimeAsync(15_000);
    late.resolve(1_500);
    await vi.advanceTimersByTimeAsync(1);
    expect(flights).toHaveLength(2);
    expect((flights[1]?.height ?? 0) - (flights[0]?.height ?? 0)).toBeCloseTo(1_500, 3);
    expect(flights[1]?.longitude).toBeCloseTo(flights[0]?.longitude ?? 0, 9);
  });

  it("lets the camera be once 20 s have passed since it landed", async () => {
    const { manager, flights, viewer } = harness();
    const late = deferred<number>();
    terrain.pending = late.promise;
    manager.setCatalog([placedSummary], () => Promise.resolve(placedSite()));
    await manager.flyTo(PLACED_ID);
    land(viewer, flights[0]);
    await vi.advanceTimersByTimeAsync(21_000);
    late.resolve(1_500);
    await vi.advanceTimersByTimeAsync(1);
    expect(flights).toHaveLength(1);
  });

  it("never moves a camera somebody has moved since it landed, or another flight has", async () => {
    const touched = async (take: (h: ReturnType<typeof harness>) => void) => {
      const h = harness();
      const late = deferred<number>();
      terrain.pending = late.promise;
      h.manager.setCatalog([placedSummary], () => Promise.resolve(placedSite()));
      await h.manager.flyTo(PLACED_ID);
      land(h.viewer, h.flights[0]);
      await vi.advanceTimersByTimeAsync(2_000);
      take(h);
      late.resolve(1_500);
      await vi.advanceTimersByTimeAsync(1);
      return h.flights.length;
    };
    // A drag of a few metres.
    const dragged = await touched(({ viewer, flights }) => {
      const at = flights[0];
      viewer.camera.positionWC = Cartesian3.fromDegrees(
        (at?.longitude ?? 0) + east(3),
        at?.latitude ?? 0,
        at?.height ?? 0,
      );
    });
    expect(dragged).toBe(1);
    // A turn on the spot.
    const turned = await touched(({ viewer }) => {
      viewer.camera.heading += 0.2;
    });
    expect(turned).toBe(1);
    // Another flight (an object flown to), before it has moved the camera at all.
    const taken = await touched(({ camera }) => {
      camera.flights += 1;
    });
    expect(taken).toBe(1);
  });

  it("frames a placed scan at its placement origin, at the size of its ground, not its floaters", async () => {
    const { manager, flights, spheres, viewer } = harness();
    terrain.height = 1_500;
    tilesets.next = placedScan;
    manager.setCatalog([placedSummary], () => Promise.resolve(placedSite([SCAN])));
    await manager.flyTo(PLACED_ID);
    await vi.advanceTimersByTimeAsync(1);
    // The terrain and the model both came while the camera was still thousands of kilometres
    // out, too close to the first leg's aim to re-point for: kept, and flown on landing.
    expect(flights).toHaveLength(1);
    land(viewer, flights[0]);
    expect(flights).toHaveLength(2);
    const framed = spheres.at(-1);
    if (!framed) throw new Error("nothing framed");
    const centre = Cartographic.fromCartesian(framed.center);
    // The origin, lifted by the clamp onto the terrain: its cells' z of 0.1 m rest at 1,500 m.
    expect(CesiumMath.toDegrees(centre.longitude)).toBeCloseTo(LON, 7);
    expect(CesiumMath.toDegrees(centre.latitude)).toBeCloseTo(LAT, 7);
    expect(centre.height).toBeCloseTo(1_499.9, 2);
    // Ninety per cent of a disc of cells 20 m in radius, not the floaters' 128 m.
    expect(framed.radius).toBeGreaterThan(15);
    expect(framed.radius).toBeLessThan(20);
    expect(flights[1]?.height).toBeCloseTo(centre.height + framed.radius * 2.3, 3);
    // Anything else asking for the model's bounds still gets all of them.
    expect((await manager.boundingSphere(placedSite([SCAN])))?.radius).toBe(128);
  });

  it("keeps the model's framing when the terrain under the site answers after it", async () => {
    const { manager, flights, viewer } = harness();
    terrain.height = 1_500;
    const late = deferred<number>();
    terrain.point = late.promise;
    tilesets.next = placedScan;
    manager.setCatalog([placedSummary], () => Promise.resolve(placedSite([SCAN])));
    await manager.flyTo(PLACED_ID);
    land(viewer, flights[0]);
    land(viewer, flights[1]);
    expect(flights).toHaveLength(2);
    // The model rests where the clamp measured; the footprint at the terrain is a worse aim.
    late.resolve(1_400);
    await vi.advanceTimersByTimeAsync(1);
    expect(flights).toHaveLength(2);
  });
});

describe("site records reach the scene from the query cache", () => {
  it("hands over each site record the API answers, and nothing else", async () => {
    const { QueryClient } = await import("@tanstack/react-query");
    const { queryKeys, watchSiteRecords } = await import("@/api/queries");
    const client = new QueryClient();
    const seen: string[] = [];
    const stop = watchSiteRecords(client, (record) =>
      seen.push(`${record.id}:${String(record.cameraBookmarks.length)}`),
    );
    // A bookmark saved: the site's query is invalidated and refetched.
    await client.query({ queryKey: queryKeys.site(SITE_ID), queryFn: () => site() });
    await client.query({
      queryKey: queryKeys.site(SITE_ID),
      queryFn: () => site([BOOKMARK, { ...BOOKMARK, id: "c", isDefault: false }]),
      staleTime: 0,
    });
    // Not a site record: the catalog list, a failed fetch.
    await client.query({ queryKey: queryKeys.sites, queryFn: () => [summary] });
    await client
      .query({
        queryKey: queryKeys.site("gone"),
        queryFn: () => Promise.reject(new Error("503")),
        retry: false,
      })
      .catch(() => undefined);
    expect(seen).toEqual([`${SITE_ID}:1`, `${SITE_ID}:2`]);
    stop();
    client.setQueryData(queryKeys.site(SITE_ID), site([]));
    expect(seen).toHaveLength(2);
  });
});

describe("the site load store", () => {
  it("keeps one record per site, drops it when the site unloads, and routes Retry to the scene", async () => {
    const { useSites } = await import("@/state/sites");
    const load: SiteLoad = {
      phase: "error",
      progress: 0.05,
      error: "The catalog did not answer within 12 s",
      retryable: true,
      attempt: 1,
      flight: true,
      startedAt: 0,
    };
    useSites.getState().setSiteLoad(SITE_ID, load);
    expect(useSites.getState().siteLoads[SITE_ID]).toEqual(load);
    // A no-op until the scene installs one.
    expect(() => useSites.getState().retrySiteLoad(SITE_ID)).not.toThrow();
    const retry = vi.fn();
    useSites.getState().setSiteLoadRetry(retry);
    useSites.getState().retrySiteLoad(SITE_ID);
    expect(retry).toHaveBeenCalledWith(SITE_ID);
    useSites.getState().setSiteLoad(SITE_ID, null);
    expect(useSites.getState().siteLoads).not.toHaveProperty(SITE_ID);
  });
});
