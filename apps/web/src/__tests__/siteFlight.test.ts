/**
 * A fly-to leaves on the click and loads its site during the flight (SiteManager.flyTo).
 *
 * The camera used to wait for the site's record and its model before it moved at all, and for
 * ever when either stalled. These pin the new contract: the first leg leaves at once for the
 * best pose known, the record re-points it at the authored bookmark without restarting from
 * rest, a stall becomes a retryable error in the site's load record, and Retry finishes the
 * job. CesiumJS's camera is faked; the poses and the easing are what is checked.
 */
import { type BoundingSphere, Cartesian3, Cartographic, Math as CesiumMath } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Site, SiteSummary } from "@twin/contracts";

import type { CameraController, FlyOptions } from "@/cesium/CameraController";
import type { ClippingManager } from "@/cesium/ClippingManager";
import { easingSlope } from "@/cesium/flightRetarget";
import type { PerformanceManager } from "@/cesium/PerformanceManager";
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

function site(bookmarks = [BOOKMARK]): Site {
  const d = 0.001;
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
    isMoving: false,
    refreshPose: vi.fn(),
    setObjectScale: vi.fn(),
    durationFor: () => 4,
    sphereArrival: (sphere: BoundingSphere) => {
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
  return { manager, flights, loads, toasts, events, flightSites, at };
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
