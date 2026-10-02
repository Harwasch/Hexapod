/**
 * A site's load record follows the model shown (SiteManager.setRepresentation).
 *
 * The record is what the load pill reads first. After the mesh failed, switching back to a
 * representation that was fine used to keep saying "Couldn't load the 3D model · Retry": the
 * switch never touched the record, and `watchFirstTiles` leaves an errored record alone. Here
 * the point cloud loads, the mesh fails for good (a 404), and switching back says ready.
 */
import { Cartesian3, Event } from "cesium";
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

/** A tileset as far as SiteManager reads one, with its first view already in. */
function fakeTileset() {
  return {
    show: false,
    preloadWhenHidden: true,
    maximumScreenSpaceError: 16,
    tilesLoaded: true,
    totalMemoryUsageInBytes: 0,
    boundingSphere: { center: Cartesian3.fromDegrees(-122.13, 47.64, 100), radius: 50 },
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

vi.mock("@/cesium/providers/tiles", async (importOriginal) => ({
  ...(await importOriginal<typeof Tiles>()),
  // The mesh is gone for good (a 404 is permanent: no retries on a timer); the cloud loads.
  createSiteTileset: vi.fn((asset: SiteAsset) =>
    asset.representation === "mesh"
      ? Promise.reject(new Error("Request failed with status 404"))
      : Promise.resolve(fakeTileset()),
  ),
}));
vi.mock("@/cesium/scanView/ScanRendererHost", () => ({
  prefetchScanDestination: () => () => undefined,
}));

const SITE_ID = "site-a";
const LON = -122.13;
const LAT = 47.64;

function asset(representation: "mesh" | "point-cloud", defaultVisible: boolean): SiteAsset {
  return {
    id: `${SITE_ID}-${representation}`,
    siteId: SITE_ID,
    name: `yard ${representation}`,
    representation,
    source: { type: "3d-tiles-url", url: `https://example.invalid/${representation}/tileset.json` },
    renderConfig: { maximumScreenSpaceError: 16, clipsWorld: false, heightOffsetM: 0 },
    defaultVisible,
  } as unknown as SiteAsset;
}

function harness() {
  const d = 0.001;
  const site = {
    id: SITE_ID,
    slug: "yard",
    name: "yard",
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
    assets: [asset("point-cloud", true), asset("mesh", false)],
    cameraBookmarks: [],
  } as unknown as Site;
  const summary = {
    id: SITE_ID,
    slug: "yard",
    name: "yard",
    centroid: site.centroid,
    areaM2: 40_000,
  } as unknown as SiteSummary;
  const events = new Emitter<SceneEvents>();
  const loads: SiteLoad[] = [];
  events.on("site-load", ({ load }) => {
    if (load) loads.push(load);
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
  return { manager, loads };
}

describe("a site's load record follows the model shown", () => {
  it("says ready again once a representation that loaded is shown after one that failed", async () => {
    const { manager, loads } = harness();
    await manager.activate(SITE_ID);
    expect(loads.at(-1)?.phase).toBe("ready");

    await manager.setRepresentation("mesh");
    await vi.waitFor(() => expect(loads.at(-1)?.phase).toBe("error"));
    expect(loads.at(-1)?.error).toContain("yard mesh did not load");

    await manager.setRepresentation("point-cloud");
    expect(loads.at(-1)).toMatchObject({ phase: "ready", error: null });
  });

  it("starts over on the model it now follows, from the asset's own Retry too", async () => {
    const { manager, loads } = harness();
    await manager.activate(SITE_ID);
    await manager.setRepresentation("mesh");
    await vi.waitFor(() => expect(loads.at(-1)?.phase).toBe("error"));
    const before = loads.length;
    // The pill's asset Retry shows the same asset again (siteLoad.ts, retrySiteLoad).
    await manager.setTemporalAsset(`${SITE_ID}-mesh`);
    const after = loads.slice(before);
    // Loading again, then the same permanent failure: not a stale error that never moved.
    expect(after[0]).toMatchObject({ phase: "model", error: null });
    await vi.waitFor(() => expect(loads.at(-1)?.phase).toBe("error"));
  });
});
