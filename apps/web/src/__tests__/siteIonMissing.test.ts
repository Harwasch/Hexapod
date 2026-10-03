/**
 * A site whose shown model is an ion asset the map key cannot see (the seeded San Francisco
 * site, ion asset 1415196: apps/api/app/seed/data.py).
 *
 * Ion's 404 for such an asset is skipped quietly -- logged once, remembered on this device so
 * it is not asked for again (cesium/ion.ts), no toast -- since it also loads by proximity near
 * the Bay Area scans. But the quiet skip returned before the site's load record was told, so
 * when that asset was the one shown the load pill said "Loading 3D model" for ever. Here it
 * ends in an error with Retry, ion is asked once (a missing asset is not retried on a timer),
 * and Retry asks ion again rather than refusing from memory.
 */
import { Cartesian3 } from "cesium";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Site, SiteAsset, SiteSummary } from "@twin/contracts";

import type { CameraController } from "@/cesium/CameraController";
import type { ClippingManager } from "@/cesium/ClippingManager";
import { IonAssetMissingError, ionAssetKnownMissing } from "@/cesium/ion";
import type { PerformanceManager } from "@/cesium/PerformanceManager";
import { createSiteTileset } from "@/cesium/providers/tiles";
import type * as Tiles from "@/cesium/providers/tiles";
import { SiteManager } from "@/cesium/SiteManager";
import type { SceneEvents } from "@/cesium/types";
import { Emitter } from "@/lib/emitter";
import type { SiteLoad } from "@/state/sites";

const ION_ASSET = 1415196;

vi.mock("@/cesium/providers/tiles", async (importOriginal) => {
  const ion = await import("@/cesium/ion");
  return {
    ...(await importOriginal<typeof Tiles>()),
    // As createSiteTileset does for an ion asset: refused from memory when ion said 404 for
    // it lately, else asked of ion -- which says 404.
    createSiteTileset: vi.fn((asset: SiteAsset) => {
      if (asset.source.type !== "cesium-ion") return Promise.reject(new Error("not ion"));
      const id = asset.source.assetId;
      if (ion.ionAssetKnownMissing(id)) return Promise.reject(new ion.IonAssetMissingError(id));
      return Promise.reject(new Error(`Request failed with status 404 (ion asset ${String(id)})`));
    }),
  };
});
vi.mock("@/cesium/scanView/ScanRendererHost", () => ({
  prefetchScanDestination: () => () => undefined,
}));

const SITE_ID = "sf";
const LON = -122.42;
const LAT = 37.77;

function harness() {
  const d = 0.01;
  const mesh = {
    id: `${SITE_ID}-mesh`,
    siteId: SITE_ID,
    name: "San Francisco mesh",
    representation: "mesh",
    source: { type: "cesium-ion", assetId: ION_ASSET },
    renderConfig: { maximumScreenSpaceError: 16, clipsWorld: false, heightOffsetM: 0 },
    defaultVisible: true,
  } as unknown as SiteAsset;
  const site = {
    id: SITE_ID,
    slug: "sf",
    name: "San Francisco",
    centroid: { longitude: LON, latitude: LAT, height: 20 },
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
    assets: [mesh],
    cameraBookmarks: [],
  } as unknown as Site;
  const summary = {
    id: SITE_ID,
    slug: "sf",
    name: "San Francisco",
    centroid: site.centroid,
    areaM2: 4_000_000,
  } as unknown as SiteSummary;
  const events = new Emitter<SceneEvents>();
  const loads: SiteLoad[] = [];
  const toasts: string[] = [];
  const assetStates: string[] = [];
  events.on("site-load", ({ load }) => {
    if (load) loads.push(load);
  });
  events.on("toast", (toast) => toasts.push(toast.title));
  events.on("asset", ({ patch }) => {
    if (patch.loadState) assetStates.push(patch.loadState);
  });
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
  const manager = new SiteManager(
    viewer as never,
    events,
    camera as unknown as CameraController,
    { setFootprint: vi.fn() } as unknown as ClippingManager,
    performance as unknown as PerformanceManager,
  );
  manager.setCatalog([summary], () => Promise.resolve(site));
  return { manager, loads, toasts, assetStates };
}

describe("a shown ion asset the map key cannot see", () => {
  afterEach(() => {
    localStorage.clear();
    vi.mocked(createSiteTileset).mockClear();
  });

  it("ends the site's load in an error with Retry, quietly, and asks ion once", async () => {
    const { manager, loads, toasts, assetStates } = harness();
    await manager.activate(SITE_ID);
    await vi.waitFor(() => expect(loads.at(-1)?.phase).toBe("error"));
    const last = loads.at(-1);
    // Not "Loading 3D model" for ever: an error the pill shows with its Retry. Not one a
    // retry on a timer can fix, so the record says so (Retry then shows the asset again).
    expect(last).toMatchObject({ phase: "error", retryable: false });
    expect(last?.error).toBe(
      "San Francisco mesh did not load: Cesium ion has no asset with this ID that the map key can see.",
    );
    expect(assetStates.at(-1)).toBe("error");
    // Quiet: no toast, one request, remembered on this device.
    expect(toasts).toEqual([]);
    expect(vi.mocked(createSiteTileset)).toHaveBeenCalledTimes(1);
    expect(loads.every((load) => load.attempt <= 1)).toBe(true);
    expect(ionAssetKnownMissing(ION_ASSET)).toBe(true);
  });

  it("refused from memory on a later visit, still ends in an error at once", async () => {
    const first = harness();
    await first.manager.activate(SITE_ID);
    await vi.waitFor(() => expect(first.loads.at(-1)?.phase).toBe("error"));
    vi.mocked(createSiteTileset).mockClear();

    const { manager, loads, toasts } = harness();
    await manager.activate(SITE_ID);
    await vi.waitFor(() => expect(loads.at(-1)?.phase).toBe("error"));
    // The remembered miss is permanent to withRetry too: no attempt on a timer.
    await expect(vi.mocked(createSiteTileset).mock.results[0]?.value).rejects.toBeInstanceOf(
      IonAssetMissingError,
    );
    expect(vi.mocked(createSiteTileset)).toHaveBeenCalledTimes(1);
    expect(loads.at(-1)?.retryable).toBe(false);
    expect(toasts).toEqual([]);
  });

  it("asks ion again on Retry rather than refusing from memory", async () => {
    const { manager, loads } = harness();
    await manager.activate(SITE_ID);
    await vi.waitFor(() => expect(loads.at(-1)?.phase).toBe("error"));
    vi.mocked(createSiteTileset).mockClear();
    // The pill's Retry for an asset (siteLoad.ts, retrySiteLoad).
    const retried = manager.setTemporalAsset(`${SITE_ID}-mesh`);
    expect(ionAssetKnownMissing(ION_ASSET)).toBe(false);
    await retried;
    await vi.waitFor(() => expect(loads.at(-1)?.phase).toBe("error"));
    // Ion itself was asked (a 404), not the memory.
    await expect(vi.mocked(createSiteTileset).mock.results[0]?.value).rejects.toThrow("404");
    expect(ionAssetKnownMissing(ION_ASSET)).toBe(true);
  });
});
