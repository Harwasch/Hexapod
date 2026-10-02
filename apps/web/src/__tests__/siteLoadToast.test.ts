/**
 * A site's model that fails to load is said once: by the load pill beside Splat / Mesh /
 * Points, which reads the site's load record. SiteManager raises no toast for the active
 * site's shown model; it still does for a failure no pill speaks for (a site loading in the
 * background, not the one the HUD describes).
 */
import { Cartesian3 } from "cesium";
import { describe, expect, it, vi } from "vitest";

import type { Site, SiteAsset, SiteSummary } from "@twin/contracts";

import type { CameraController } from "@/cesium/CameraController";
import type { ClippingManager } from "@/cesium/ClippingManager";
import type { PerformanceManager } from "@/cesium/PerformanceManager";
import type * as Tiles from "@/cesium/providers/tiles";
import { SiteManager } from "@/cesium/SiteManager";
import type { SceneEvents } from "@/cesium/types";
import { Emitter } from "@/lib/emitter";
import type { SiteLoad } from "@/state/sites";

vi.mock("@/cesium/providers/tiles", async (importOriginal) => ({
  ...(await importOriginal<typeof Tiles>()),
  // A 404 is permanent, so the load fails at once instead of retrying on a timer.
  createSiteTileset: vi.fn(() => Promise.reject(new Error("Request failed with status 404"))),
}));
vi.mock("@/cesium/scanView/ScanRendererHost", () => ({
  prefetchScanDestination: () => () => undefined,
}));

function siteWith(id: string, slug: string): { summary: SiteSummary; site: Site } {
  const lon = -122.13;
  const lat = 47.64;
  const d = 0.001;
  const asset = {
    id: `${id}-mesh`,
    siteId: id,
    name: `${slug} mesh (LOD)`,
    representation: "mesh",
    source: { type: "3d-tiles-url", url: "https://example.invalid/mesh/tileset.json" },
    renderConfig: { maximumScreenSpaceError: 16, clipsWorld: false, heightOffsetM: 0 },
    defaultVisible: true,
  } as unknown as SiteAsset;
  const site = {
    id,
    slug,
    name: slug,
    centroid: { longitude: lon, latitude: lat, height: 100 },
    boundary: {
      type: "Polygon",
      coordinates: [
        [
          [lon - d, lat - d],
          [lon + d, lat - d],
          [lon + d, lat + d],
          [lon - d, lat + d],
          [lon - d, lat - d],
        ],
      ],
    },
    assets: [asset],
    cameraBookmarks: [],
  } as unknown as Site;
  const summary = {
    id,
    slug,
    name: slug,
    centroid: site.centroid,
    areaM2: 40_000,
  } as unknown as SiteSummary;
  return { summary, site };
}

function harness() {
  const events = new Emitter<SceneEvents>();
  const toasts: string[] = [];
  const loads: { siteId: string; load: SiteLoad }[] = [];
  events.on("toast", (toast) => toasts.push(toast.title));
  events.on("site-load", ({ siteId, load }) => {
    if (load) loads.push({ siteId, load });
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
  return { manager, toasts, loads };
}

describe("a model that fails to load", () => {
  it("is said by the active site's load record only, not by a toast as well", async () => {
    const { manager, toasts, loads } = harness();
    const a = siteWith("site-a", "yard");
    manager.setCatalog([a.summary], () => Promise.resolve(a.site));
    await manager.activate("site-a");
    await vi.waitFor(() => expect(loads.at(-1)?.load.phase).toBe("error"));
    expect(loads.at(-1)).toMatchObject({ siteId: "site-a", load: { retryable: false } });
    expect(toasts).toEqual([]);
  });

  it("still raises a toast for a site loading in the background, which no pill speaks for", async () => {
    const { manager, toasts } = harness();
    const a = siteWith("site-a", "yard");
    const b = siteWith("site-b", "orchard");
    a.site.assets = [];
    manager.setCatalog([a.summary, b.summary], (id) =>
      Promise.resolve(id === "site-a" ? a.site : b.site),
    );
    await manager.activate("site-a");
    await manager.activate("site-b", { primary: false });
    await vi.waitFor(() => expect(toasts).toEqual(["orchard mesh (LOD) failed to load"]));
  });
});
