import {
  BoundingSphere,
  Cartesian3,
  Cartographic,
  Math as CesiumMath,
  Matrix3,
  Matrix4,
  type Cesium3DTileset,
  type Rectangle,
  type Scene,
  type CesiumWidget,
  type Cesium3DTile,
  sampleTerrainMostDetailed,
} from "cesium";

import type { Footprint, Representation, Site, SiteAsset, SiteSummary } from "@twin/contracts";
import { boundingRadiusM, centerOf, circleFootprint, haversineDistance } from "@twin/geo";

import {
  detailScreenSpaceScale,
  deviceSplatBudget,
  deviceSplatCeiling,
  isHandheld,
} from "@/lib/detail";
import { AdaptiveSplatBudget } from "@/lib/splatBudget";

import { DEFAULT_SPLAT_RENDERER } from "@/state/settings";
import type { SiteLoad } from "@/state/sites";

import { prefetchScanDestination } from "./scanView/ScanRendererHost";
import type { SplatRendererKind } from "./scanView/types";
import type { Emitter } from "@/lib/emitter";
import { createLogger, describeError } from "@/lib/log";
import { withRetry } from "@/lib/retry";
import { throttleProgress } from "@/lib/throttle";
import { withTimeout } from "@/lib/timeout";
import { timed } from "@/lib/timing";

import type { CameraController } from "./CameraController";
import type { ClippingManager } from "./ClippingManager";
import {
  quadraticInOut,
  retarget,
  samePose,
  type ArrivalPose,
  type Easing,
} from "./flightRetarget";
import { isIonAuthError, isIonNotFound } from "./ion";
import { devicePixelError, type PerformanceManager } from "./PerformanceManager";
import { groundAt, measuredClamp, type MeasuredGround } from "./placement";
import { createSiteTileset, tileCacheBudget } from "./providers/tiles";
import { SPLAT_BYTES_ESTIMATE, SplatCount, splatMemory } from "./splatCount";
import { splatTilesetOf } from "./splatInternals";
import { attachInferredLayers } from "./inferredLayers";
import { attachSplitObjects } from "./splitObjects";
import { attachInstances } from "./splatInstances";
import { attachSkin } from "./splatSkin";
import { attachTelemetry } from "./telemetry";
import { attachViewCones } from "./splatViewCones";
import type { SceneEvents } from "./types";

const log = createLogger("sites");

const REPRESENTATION_ORDER: Representation[] = [
  "gaussian-splat",
  "mesh",
  "point-cloud",
  "terrain",
  "imagery",
];
const ACTIVATE_DISTANCE_M = 40_000;
const DEACTIVATE_DISTANCE_M = 400_000;
const NEAR_ALTITUDE_M = 6_000;
/** Sites smaller than this are ranked as if they were this big, so tiny objects do not win by default. */
const MIN_SITE_RADIUS_M = 30;
/** Object scale engages within this distance of a hand-sized model's surface. */
const OBJECT_SCALE_REACH_M = 25;
/**
 * A site's model takes over from the world below this many footprint radii of altitude and
 * within this many radii horizontally; it hands back further out (hysteresis).
 */
const ENGAGE_ALTITUDE_RADII = 2.5;
const DISENGAGE_ALTITUDE_RADII = 3.5;
const ENGAGE_DISTANCE_RADII = 3;
const DISENGAGE_DISTANCE_RADII = 4.5;
/**
 * The site's catalog record must answer within this (ms). The API scales to zero between
 * visits (fly.toml), and a cold start takes a few seconds; a request that has not answered by
 * now is stalled, not slow, and is cut off so the HUD can say so and offer Retry.
 */
const DETAIL_TIMEOUT_MS = 12_000;
/**
 * One attempt at creating a site's tileset -- the tile proxy probe, `tileset.json`, ion's
 * endpoint -- must answer within this (ms). A stall used to hold the load forever: withRetry
 * only retries what fails. Now it fails, and the next attempt goes.
 */
const TILESET_ATTEMPT_TIMEOUT_MS = 15_000;
/**
 * A site whose record failed is not fetched again by proximity for this long (ms): the camera
 * moving near it re-checks every 400 ms, and an API that is down would be asked each time.
 * A fly-to or Retry asks at once.
 */
const PROXIMITY_RETRY_MS = 30_000;
/** A flight re-pointed on the way never lands sooner than this (s). */
const RETARGET_MIN_S = 1.2;
/**
 * After landing, a better destination (the authored bookmark, the model's real bounds) still
 * moves the camera for this long (ms), and only if nobody has touched the camera since.
 */
const SETTLE_WINDOW_MS = 8_000;
/** Where the load bar stands as each phase begins (state/sites.ts `SiteLoad`). */
const LOAD_PROGRESS = { details: 0.05, model: 0.15, streaming: 0.3 } as const;

interface AssetHandle {
  asset: SiteAsset;
  tileset: Cesium3DTileset | null;
  loading: Promise<Cesium3DTileset | null> | null;
  /** Resolves once a clamp-to-ground placement has been applied (or was not requested). */
  placed: Promise<void>;
  unsubscribe: (() => void)[];
  /** The tile-coverage clip refresh is registered once per tileset. */
  coverageWatched?: boolean;
  /** A splat tileset's loaded gaussians (splatCount.ts): Cesium's byte count misses them. */
  splats?: SplatCount;
}

/**
 * A read-only view of one loaded asset, for callers that need to reach the tileset without
 * being handed the mutable `AssetHandle` it lives in. Plain data plus the tileset itself.
 */
export interface LoadedSiteAsset {
  readonly siteId: string;
  readonly siteSlug: string;
  readonly assetId: string;
  readonly assetName: string;
  readonly representation: Representation;
  /** Where the tileset was fetched from, or null for an ion-hosted asset. */
  readonly sourceUrl: string | null;
  /**
   * The catalog's Living Survey rig claim for this asset, relative to `sourceUrl`, or null.
   * Read, never probed — see `livingRigs.ts`.
   */
  readonly rigPath: string | null;
  /** Whether the site is engaged, so the tileset is actually drawn. */
  readonly shown: boolean;
  readonly tileset: Cesium3DTileset;
}

interface ActiveSite {
  site: Site;
  representation: Representation;
  temporalAssetId: string | null;
  handles: Map<string, AssetHandle>;
  /** Radius of the authored footprint (at least MIN_SITE_RADIUS_M), the yardstick for engagement. */
  radius: number;
  /**
   * Engaged: the camera is close enough that the model's detail matters, so the model is
   * drawn and takes over from the world. Further out the model stays loaded but hidden and
   * the world is left seamless: from 20 km up a 5 km patch of another capture, with a hard
   * edge, is a blemish rather than information.
   */
  engaged: boolean;
}

/** The flight `flyTo` is steering: where it is headed, and enough to re-point it smoothly. */
interface SiteFlight {
  /** Which `flyTo` call this belongs to; a later call supersedes it. */
  serial: number;
  siteId: string;
  pose: ArrivalPose;
  /** `performance.now()` when this leg left. */
  startedAt: number;
  durationS: number;
  easing: Easing;
  /** Chord from where this leg left to `pose` (m). */
  lengthM: number;
  state: "flying" | "landed" | "cancelled";
  /** Where and when it landed, to tell whether anybody has moved the camera since. */
  landed: { at: number; position: Cartesian3; heading: number } | null;
}

/**
 * Settles with the promise's value if it already has one (a cached record, the built-in demo
 * site), or with undefined at the next macrotask: what a fly-to may use without waiting.
 */
function settledNow<T>(promise: Promise<T>): Promise<T | undefined> {
  return Promise.race([
    promise.catch(() => undefined),
    new Promise<undefined>((resolve) => setTimeout(() => resolve(undefined), 0)),
  ]);
}

/** Splat tiles kept in memory, as a multiple of what the Detail budget draws: the view,
 *  its coarser ancestors, and what was looked at a moment ago, so looking back finds it. */
const SPLAT_CACHE_FACTOR = 2.5;
/** Ceilings on a splat tileset's cache (bytes): a desktop, and a phone or tablet. */
const SPLAT_CACHE_CEILING = { desktop: 1536 * 1024 * 1024, handheld: 512 * 1024 * 1024 };

/**
 * A splat tileset's cache, sized once its tiles say what a splat costs in memory. Cesium's
 * default (the tile cache budget, 384 MB on a desktop) held 1M splats with spherical
 * harmonics, 4.8M without -- against a 3M budget, so a look around evicted the view just
 * left and looking back fetched, decoded and uploaded it again.
 */
function sizeSplatCache(tileset: Cesium3DTileset, tile: Cesium3DTile, budget: number): void {
  const content = tile.content as
    { geometryByteLength?: number; pointsLength?: number } | undefined;
  const bytes = content?.geometryByteLength ?? 0;
  const points = content?.pointsLength ?? 0;
  if (!(bytes > 0 && points > 0)) return;
  const perSplat = bytes / points;
  const ceiling = isHandheld() ? SPLAT_CACHE_CEILING.handheld : SPLAT_CACHE_CEILING.desktop;
  const wanted = Math.min(ceiling, budget * SPLAT_CACHE_FACTOR * perSplat);
  const floor = tileCacheBudget().cacheBytes;
  const cacheBytes = Math.round(Math.max(floor, wanted));
  // Only ever grows: a tile of a coarser level (fewer bytes a splat) must not shrink it back.
  if (cacheBytes <= tileset.cacheBytes) return;
  tileset.cacheBytes = cacheBytes;
  tileset.maximumCacheOverflowBytes = Math.round(cacheBytes / 2);
}

/**
 * Loads a site's reality models into the world when they are useful (fly-to or
 * proximity), switches representations without moving the camera, and keeps
 * the coarse world clipped underneath the active model.
 */
export class SiteManager {
  private readonly scene: Scene;
  private summaries: SiteSummary[] = [];
  /**
   * Fetches a site's catalog record: null when the catalog has no such site, a rejection for
   * any other failure. `signal` is aborted at DETAIL_TIMEOUT_MS.
   */
  private detailResolver: (id: string, signal?: AbortSignal) => Promise<Site | null> = () =>
    Promise.resolve(null);
  /** Records already fetched, so a second flight to a site leaves for its bookmark at once. */
  private readonly details = new Map<string, Site>();
  /** Record fetches in flight, shared by a flight and the activation it starts. */
  private readonly pendingDetails = new Map<string, Promise<Site | null>>();
  /** Each site's load, as the HUD shows it (`site-load` events, state/sites.ts). */
  private readonly loads = new Map<string, SiteLoad>();
  /** When each site's record last failed (`Date.now()`), for PROXIMITY_RETRY_MS. */
  private readonly detailFailedAt = new Map<string, number>();
  private flight: SiteFlight | null = null;
  private flightSerial = 0;
  /** Stops the scan renderer's prefetch of the current leg's destination. */
  private cancelPrefetch: (() => void) | null = null;
  /** Every site currently loaded in the scene, keyed by site id. Sites can overlap (a hand-sized
   *  object registered on top of a campus), so several stay loaded at once. */
  private readonly loaded = new Map<string, ActiveSite>();
  /** Who draws splat scans: CesiumJS, or a dedicated renderer over the globe
   *  (scanView/ScanRendererHost.ts) while CesiumJS keeps the tileset, hidden, for its frame
   *  and its solids. */
  private splatRenderer: SplatRendererKind = DEFAULT_SPLAT_RENDERER;
  /** The site the representation switcher, clipping and the HUD refer to. */
  private primaryId: string | null = null;
  private nearId: string | null = null;
  private objectScale = false;
  private screenSpaceError = 16;
  /** The gaussians this device draws at once (lib/detail.ts), read once. */
  private readonly splatDetail = deviceSplatBudget();
  /** The same choice as a factor on splat screen-space error. */
  private readonly splatDetailScale = detailScreenSpaceScale(this.splatDetail);
  /** What a view may draw, below that ceiling, from motion frame times (lib/splatBudget.ts). */
  private readonly splatCeiling = deviceSplatCeiling();
  private readonly splatBudget = new AdaptiveSplatBudget(this.splatCeiling, this.splatDetail);
  private pixelRatio = 1;
  /** Ground metres per pixel at the view centre when the errors were last applied. */
  private metersPerPixel = Number.POSITIVE_INFINITY;
  private appliedMetersPerPixel = Number.POSITIVE_INFINITY;
  private flightTarget: string | null = null;
  /**
   * The site the latest fly-to is taking the camera to, for the HUD's load pill
   * (`site-flight`), until the camera has left it (`checkProximity`). Unlike `flightTarget` it
   * outlives the flight: a site whose record failed never becomes active, and once the camera
   * has landed at the summary's pose the pill must still say why nothing loaded and offer Retry.
   */
  private flightSiteId: string | null = null;
  private readonly unsubscribe: (() => void)[] = [];
  private lastProximityCheck = 0;

  constructor(
    private readonly viewer: CesiumWidget,
    private readonly events: Emitter<SceneEvents>,
    private readonly camera: CameraController,
    private readonly clipping: ClippingManager,
    private readonly performance: PerformanceManager,
  ) {
    this.scene = viewer.scene;
    this.performance.addScreenSpaceErrorSink("sites", (sse, pixelRatio) =>
      this.applyScreenSpaceError(sse, pixelRatio),
    );
    this.performance.addMemorySource("sites", () => this.memoryUsage());
    // Splats are budgeted by count, against the Detail choice (splatCount.ts): the group's
    // pressure is the higher of the two ratios.
    // Drawn, not loaded: what the budget limits is what a frame draws. Tiles only cached
    // (turned away from, or replaced by their children) are the tileset cache's to trim, so
    // a view is never coarsened for splats it is not drawing.
    this.performance.addMemorySource("sites", () =>
      splatMemory(this.splatsDrawn(), this.splatBudget.budget),
    );
    this.unsubscribe.push(
      this.performance.addMotionFrameListener((intervalMs) => {
        if (!this.splatBudget.frame(intervalMs, this.splatsDrawn())) return;
        log.info("splat budget", { budget: this.splatBudget.budget, ceiling: this.splatCeiling });
      }),
    );
    const calibrationTimer = setInterval(() => this.refreshCalibration(), CALIBRATION_TICK_MS);
    this.unsubscribe.push(
      viewer.camera.changed.addEventListener(() => this.checkProximity()),
      viewer.camera.moveEnd.addEventListener(() => this.checkProximity(true)),
      () => clearInterval(calibrationTimer),
    );
  }

  /** Catalog summaries used for proximity activation; details are fetched lazily. */
  setCatalog(
    summaries: SiteSummary[],
    resolver: (id: string, signal?: AbortSignal) => Promise<Site | null>,
  ): void {
    this.summaries = summaries;
    this.detailResolver = resolver;
    // A new catalog may carry edited sites (a new bookmark); fetch their records afresh.
    this.details.clear();
    this.checkProximity(true);
  }

  /**
   * A fresh copy of a site's record, fetched elsewhere in the app (the query cache,
   * `watchSiteRecords`): a bookmark saved or deleted, the site edited. The records kept for
   * flights (`details`) and the loaded site's own copy were otherwise cleared only with a new
   * catalog, and a bookmark changes nothing the catalog list carries -- so the next fly-to used
   * to leave for a deleted bookmark, or for the footprint of a site that now has one. Every
   * fly-to used to fetch the record afresh; the copy kept now has to be kept fresh instead.
   *
   * A loaded site takes the new bookmarks only: its assets are in the scene as they were
   * loaded, and swapping them is a reload, not an update.
   */
  updateRecord(site: Site): void {
    this.details.set(site.id, site);
    const entry = this.loaded.get(site.id);
    if (entry) entry.site = { ...entry.site, cameraBookmarks: site.cameraBookmarks };
  }

  private get active(): ActiveSite | null {
    return this.primaryId ? (this.loaded.get(this.primaryId) ?? null) : null;
  }

  get activeSite(): Site | null {
    return this.active?.site ?? null;
  }

  get activeRepresentation(): Representation | null {
    return this.active?.representation ?? null;
  }

  availableRepresentations(): Representation[] {
    if (!this.active) return [];
    const present = new Set(this.active.site.assets.map((a) => a.representation));
    return REPRESENTATION_ORDER.filter((r) => present.has(r));
  }

  /**
   * The loaded tileset for one site's representation, or null while it is absent or loading.
   *
   * Read-only on purpose. Tilesets live in private `AssetHandle`s and nothing outside this
   * class may keep one or change it; the Living Survey needs to *read* one to deform its splat
   * texture, and this is the whole of that need.
   */
  tilesetFor(siteId: string, representation: Representation): Cesium3DTileset | null {
    const entry = this.loaded.get(siteId);
    if (!entry) return null;
    const asset = this.pickAsset(entry, representation);
    return asset ? (entry.handles.get(asset.id)?.tileset ?? null) : null;
  }

  /**
   * Every asset with a tileset in the scene right now, as flat read-only views.
   *
   * The Living Survey reconciles against this list rather than tracking `asset` events one by
   * one: a list that is always the truth cannot drift out of step with the scene, and a site
   * dropped by `checkProximity` disappears from it in the same turn its tileset is destroyed.
   */
  loadedAssets(): LoadedSiteAsset[] {
    const views: LoadedSiteAsset[] = [];
    for (const { entry, handle } of this.handles()) {
      const tileset = handle.tileset;
      if (!tileset) continue;
      views.push({
        siteId: entry.site.id,
        siteSlug: entry.site.slug,
        assetId: handle.asset.id,
        assetName: handle.asset.name,
        representation: handle.asset.representation,
        sourceUrl: handle.asset.source.type === "3d-tiles-url" ? handle.asset.source.url : null,
        rigPath: handle.asset.renderConfig.rigUrl ?? null,
        shown: tileset.show,
        tileset,
      });
    }
    return views;
  }

  /** Temporal versions of the active representation, newest first. */
  temporalVersions(): SiteAsset[] {
    if (!this.active) return [];
    const rep = this.active.representation;
    return this.active.site.assets
      .filter((a) => a.representation === rep && a.observedAt)
      .sort((a, b) => (b.observedAt ?? "").localeCompare(a.observedAt ?? ""));
  }

  /**
   * Loads a site's assets into the scene without moving the camera. `flight` marks the load as
   * one a fly-to is waiting on (the HUD shows those). Resolves null when the record could not be
   * had; the site's load record (`site-load`) says whether it is gone or failed, and why.
   */
  async activate(
    siteId: string,
    options: { primary?: boolean; flight?: boolean } = {},
  ): Promise<Site | null> {
    const makePrimary = options.primary ?? true;
    const existing = this.loaded.get(siteId);
    if (existing) {
      if (makePrimary) this.setPrimary(siteId);
      if (options.flight && this.loads.has(siteId)) this.reportLoad(siteId, { flight: true });
      return existing.site;
    }
    const failedAt = this.detailFailedAt.get(siteId);
    if (!options.flight && failedAt !== undefined && Date.now() - failedAt < PROXIMITY_RETRY_MS)
      return null;
    if (!this.pendingDetails.has(siteId) || options.flight) {
      const flight = options.flight === true || (this.loads.get(siteId)?.flight ?? false);
      this.startLoad(siteId, "details", flight);
    }
    let site: Site | null;
    try {
      site = await this.detailFor(siteId);
    } catch (error) {
      log.warn("site details failed", { site: siteId, error: describeError(error) });
      this.detailFailedAt.set(siteId, Date.now());
      this.failLoad(siteId, `The site's details did not load: ${describeError(error)}`, true);
      return null;
    }
    if (!site) {
      this.failLoad(siteId, "The catalog no longer has this site.", false);
      return null;
    }
    this.detailFailedAt.delete(siteId);
    if (this.loaded.has(siteId)) return site;
    this.reportLoad(siteId, { phase: "model", progress: LOAD_PROGRESS.model });
    const defaultAsset = site.assets.find((a) => a.defaultVisible) ?? site.assets[0];
    const representation = defaultAsset?.representation ?? "gaussian-splat";
    const entry: ActiveSite = {
      site,
      representation,
      temporalAssetId: null,
      handles: new Map(),
      radius: Math.max(boundingRadiusM(site.boundary), MIN_SITE_RADIUS_M),
      engaged: false,
    };
    entry.engaged = this.shouldEngage(entry, false);
    this.loaded.set(siteId, entry);
    if (makePrimary || !this.primaryId) this.setPrimary(siteId);
    await this.showRepresentation(entry, representation);
    return site;
  }

  private setFlightSite(siteId: string | null): void {
    if (this.flightSiteId === siteId) return;
    this.flightSiteId = siteId;
    this.events.emit("site-flight", siteId);
  }

  private setPrimary(siteId: string): void {
    if (this.primaryId === siteId) return;
    const entry = this.loaded.get(siteId);
    if (!entry) return;
    this.primaryId = siteId;
    this.performance.resetBenchmark();
    this.events.emit("site-active", siteId);
    this.events.emit("representation", { siteId, representation: entry.representation });
    this.events.emit("tilesets", this.activeTilesetLabels());
  }

  /** Unloads one site, or every site when no id is given. */
  deactivate(siteId?: string): void {
    const ids = siteId ? [siteId] : Array.from(this.loaded.keys());
    for (const id of ids) {
      const entry = this.loaded.get(id);
      if (!entry) continue;
      for (const handle of entry.handles.values()) this.disposeHandle(handle);
      this.clipping.setFootprint(id, null);
      this.loaded.delete(id);
      if (this.loads.delete(id)) this.events.emit("site-load", { siteId: id, load: null });
      if (this.flightSiteId === id) this.setFlightSite(null);
    }
    if (this.primaryId && !this.loaded.has(this.primaryId)) {
      this.primaryId = null;
      const next = this.loaded.keys().next();
      if (!next.done) this.setPrimary(next.value);
      else this.events.emit("site-active", null);
    }
    if (this.loaded.size === 0) {
      this.camera.setObjectScale(false);
      this.objectScale = false;
    }
    this.events.emit("tilesets", this.activeTilesetLabels());
    this.scene.requestRender();
  }

  /**
   * Flies to a site and loads it, at the same time. The camera leaves on the click, for the
   * best pose known at that moment, and the site's record and model load during the flight.
   *
   * It used to wait first: for the site's record (a GET that meets the API's cold start), then
   * for the model's tileset (a proxy probe and `tileset.json`, retried up to four times), and
   * without a bookmark for the model to be clamped to the terrain -- seconds of a dead click
   * before anything moved, and forever if one of those stalled. Now:
   *
   * 1. The flight leaves at once, for the authored bookmark if the record is already here (a
   *    second visit, the built-in demo), else for the catalog summary's centre and size.
   * 2. The record and the model load meanwhile (`activate`), with deadlines on both, and the
   *    flight itself is what Cesium preloads destination tiles for (`preloadFlightDestinations`).
   * 3. As better poses arrive -- the bookmark with the record, the model's real bounds once it
   *    rests on the ground -- the flight is re-pointed without a jolt: the new leg leaves at
   *    the speed the camera already has (flightRetarget.ts). After landing, a better pose still
   *    moves a camera nobody has touched, for a few seconds; never one somebody has.
   *
   * Progress and failure go to the site's load record (`site-load`, state/sites.ts), where the
   * HUD shows them with Retry; a failed record leaves the camera at the summary's pose.
   */
  async flyTo(siteId: string): Promise<void> {
    const summary = this.summaries.find((s) => s.id === siteId);
    // Protected from proximity unloading from the very start: the catalog can finish loading
    // while the site details are still being fetched, and the camera is usually far away.
    this.flightTarget = siteId;
    // The pill speaks for this site from the click: its record is what it waits on first.
    this.setFlightSite(siteId);
    const serial = ++this.flightSerial;
    const detail = this.detailFor(siteId);
    // Swallowed here: `activate` turns a failure into the site's load record.
    detail.catch(() => undefined);
    const activation = this.activate(siteId, { flight: true });
    const early =
      this.loaded.get(siteId)?.site ?? this.details.get(siteId) ?? (await settledNow(detail));
    if (serial !== this.flightSerial) return;
    if (early === null) {
      this.siteNotFound(siteId);
      return;
    }
    const first = early
      ? this.arrivalFor(early)
      : summary
        ? this.summaryArrival(summary)
        : undefined;
    if (first) this.startFlight(serial, siteId, first);

    let site: Site | null;
    try {
      site = early ?? (await detail);
    } catch {
      // The record failed; the load record says so and offers Retry. Wherever the camera is
      // going (the summary's pose), it is allowed to arrive.
      if (!first && this.flightTarget === siteId) this.flightTarget = null;
      return;
    }
    if (serial !== this.flightSerial) return;
    if (!site) {
      this.siteNotFound(siteId);
      return;
    }
    if (!early) {
      const pose = this.arrivalFor(site);
      if (first) this.steer(serial, pose);
      else this.startFlight(serial, siteId, pose);
    }
    // With an authored bookmark that is the destination. Without one, the model's own bounds
    // are, once it is loaded and resting on the ground (a clamped object has no usable
    // catalog height, so its record's sphere can be metres off).
    if (this.bookmarkOf(site)) return;
    await activation;
    if (serial !== this.flightSerial) return;
    const sphere = await this.boundingSphere(site);
    if (sphere && serial === this.flightSerial)
      this.steer(serial, this.camera.sphereArrival(sphere));
  }

  private siteNotFound(siteId: string): void {
    if (this.flightTarget === siteId) this.flightTarget = null;
    // Said by the toast below; there is nothing for the pill to retry.
    if (this.flightSiteId === siteId) this.setFlightSite(null);
    this.events.emit("toast", {
      tone: "error",
      title: "Site not found",
      body: "The catalog no longer has this site.",
    });
  }

  /** The default camera bookmark of a site, if it has any. */
  private bookmarkOf(site: Site): Site["cameraBookmarks"][number] | undefined {
    return site.cameraBookmarks.find((b) => b.isDefault) ?? site.cameraBookmarks[0];
  }

  /** Where a site's record says to arrive: its bookmark, else above its footprint. */
  private arrivalFor(site: Site): ArrivalPose {
    const bookmark = this.bookmarkOf(site);
    if (bookmark) {
      return {
        longitude: bookmark.longitude,
        latitude: bookmark.latitude,
        height: bookmark.height,
        heading: bookmark.heading,
        pitch: bookmark.pitch,
      };
    }
    const center = centerOf(site.boundary);
    const radius = Math.max(boundingRadiusM(site.boundary), 20);
    const ground = this.groundGuess(center.longitude, center.latitude, site.centroid.height);
    return this.camera.sphereArrival(
      new BoundingSphere(Cartesian3.fromDegrees(center.longitude, center.latitude, ground), radius),
    );
  }

  /** Where the catalog summary says to arrive: above its centre, at its size. */
  private summaryArrival(summary: SiteSummary): ArrivalPose {
    const { longitude, latitude, height } = summary.centroid;
    const radius = Math.max(Math.sqrt(summary.areaM2 / Math.PI), MIN_SITE_RADIUS_M);
    const ground = this.groundGuess(longitude, latitude, height);
    return this.camera.sphereArrival(
      new BoundingSphere(Cartesian3.fromDegrees(longitude, latitude, ground), radius),
    );
  }

  /**
   * The ground height to aim at: the catalog's when it has one, else the globe's terrain as far
   * as it has loaded (a CPU lookup), else the ellipsoid. Only the first leg aims this way; the
   * record or the model corrects it on the way.
   */
  private groundGuess(longitude: number, latitude: number, height?: number | null): number {
    if (height !== undefined && height !== null && Number.isFinite(height)) return height;
    return this.scene.globe.getHeight(Cartographic.fromDegrees(longitude, latitude)) ?? 0;
  }

  /** The first leg of a fly-to: Cesium's usual quadratic in-out, at the usual pace. */
  private startFlight(serial: number, siteId: string, pose: ArrivalPose): void {
    const destination = Cartesian3.fromDegrees(pose.longitude, pose.latitude, pose.height);
    const lengthM = Cartesian3.distance(this.viewer.camera.positionWC, destination);
    this.fly(serial, siteId, pose, this.camera.durationFor(destination), quadraticInOut, lengthM);
  }

  private fly(
    serial: number,
    siteId: string,
    pose: ArrivalPose,
    durationS: number,
    easing: Easing,
    lengthM: number,
  ): void {
    const flight: SiteFlight = {
      serial,
      siteId,
      pose,
      startedAt: performance.now(),
      durationS,
      easing,
      lengthM,
      state: "flying",
      landed: null,
    };
    // Before `camera.flyTo`: it cancels the leg this replaces, synchronously, and that leg's
    // onCancel must find itself already superseded.
    this.flight = flight;
    this.flightTarget = siteId;
    // A scan drawn by a dedicated renderer (Spark, PlayCanvas) streams through an overlay that
    // Cesium's own destination preloading never reaches; fetch what it will show from where
    // this leg ends, during the flight. A re-pointed leg replaces the prefetch; the prefetch
    // ends itself on arrival or after 15 s (scanView/ScanRendererHost.ts).
    this.cancelPrefetch?.();
    this.cancelPrefetch = prefetchScanDestination({
      position: Cartesian3.fromDegrees(pose.longitude, pose.latitude, pose.height),
      heading: CesiumMath.toRadians(pose.heading),
      pitch: CesiumMath.toRadians(pose.pitch),
    });
    const settle = () => {
      if (this.flightTarget === siteId) this.flightTarget = null;
      this.checkProximity(true);
    };
    this.camera.flyTo(pose.longitude, pose.latitude, pose.height, {
      heading: pose.heading,
      pitch: pose.pitch,
      durationS,
      easing,
      onComplete: () => {
        if (this.flight !== flight) return;
        this.cancelPrefetch = null;
        const camera = this.viewer.camera;
        flight.state = "landed";
        flight.landed = {
          at: performance.now(),
          position: Cartesian3.clone(camera.positionWC),
          heading: camera.heading,
        };
        settle();
      },
      onCancel: () => {
        // Replaced by our own re-pointing: the new leg carries on.
        if (this.flight !== flight) return;
        // Somebody else took the camera (another fly-to, a search result): let it go, and stop
        // fetching for a destination nobody is going to.
        flight.state = "cancelled";
        this.cancelPrefetch?.();
        this.cancelPrefetch = null;
        settle();
      },
    });
  }

  /**
   * Points the current fly-to at a better pose: mid-flight at the camera's current speed, or
   * after landing if nobody has moved the camera since. Nothing happens when the pose is
   * effectively where the flight is already going, when another fly-to has superseded this
   * one, or when somebody has taken the camera.
   */
  private steer(serial: number, pose: ArrivalPose): void {
    const flight = this.flight;
    if (flight?.serial !== serial || flight.state === "cancelled") return;
    const camera = this.viewer.camera;
    const destination = Cartesian3.fromDegrees(pose.longitude, pose.latitude, pose.height);
    const previous = Cartesian3.fromDegrees(
      flight.pose.longitude,
      flight.pose.latitude,
      flight.pose.height,
    );
    const remainingM = Cartesian3.distance(camera.positionWC, destination);
    // A few per cent of the distance still to fly, and 5 cm beside a hand-sized object.
    const tolerance = Math.max(0.05, 0.03 * Cartesian3.distance(camera.positionWC, previous));
    if (samePose(flight.pose, pose, Cartesian3.distance(previous, destination), tolerance)) return;
    if (flight.state === "flying") {
      // The leg's time can be up before Cesium has called it complete (a hidden tab draws no
      // frames; slow frames finish late), and then there is no speed to carry on with: the
      // camera is at the old pose, or still on its way and about to jump there, at rest. A
      // fresh leg from rest, then. Dropping the better pose here landed the camera at the
      // summary's centroid, and the settle window never saw it.
      const next = retarget(
        {
          elapsedS: (performance.now() - flight.startedAt) / 1000,
          durationS: flight.durationS,
          easing: flight.easing,
          lengthM: flight.lengthM,
        },
        remainingM,
        RETARGET_MIN_S,
      ) ?? { durationS: this.camera.durationFor(destination), easing: quadraticInOut };
      log.info("flight re-pointed", { site: flight.siteId, durationS: next.durationS });
      this.fly(serial, flight.siteId, pose, next.durationS, next.easing, remainingM);
      return;
    }
    const landed = flight.landed;
    if (!landed || performance.now() - landed.at > SETTLE_WINDOW_MS) return;
    const untouched =
      Cartesian3.distance(landed.position, camera.positionWC) < 0.01 &&
      Math.abs(landed.heading - camera.heading) < 1e-4;
    if (!untouched) return;
    log.info("settling on a better pose", { site: flight.siteId });
    this.fly(
      serial,
      flight.siteId,
      pose,
      this.camera.durationFor(destination),
      quadraticInOut,
      remainingM,
    );
  }

  /**
   * Tries a failed site load again (the HUD's Retry, through state/sites.ts): the model when
   * the site is loaded, otherwise its record -- flying there again if a fly-to had asked for
   * it, so the camera ends at the bookmark it never got.
   */
  async retry(siteId: string): Promise<void> {
    const load = this.loads.get(siteId);
    if (load?.phase !== "error") return;
    this.detailFailedAt.delete(siteId);
    const entry = this.loaded.get(siteId);
    if (entry) {
      this.startLoad(siteId, "model", load.flight);
      await this.showRepresentation(entry, entry.representation);
      return;
    }
    if (load.flight) await this.flyTo(siteId);
    else await this.activate(siteId, { primary: false });
  }

  /** The site's record, from the cache, a fetch already in flight, or a new one with a deadline. */
  private detailFor(siteId: string): Promise<Site | null> {
    const known = this.details.get(siteId);
    if (known) return Promise.resolve(known);
    const pending = this.pendingDetails.get(siteId);
    if (pending) return pending;
    const controller = new AbortController();
    const request = withTimeout(this.detailResolver(siteId, controller.signal), DETAIL_TIMEOUT_MS, {
      what: "The catalog",
      controller,
    })
      .then((site) => {
        if (site) this.details.set(siteId, site);
        return site;
      })
      .finally(() => this.pendingDetails.delete(siteId));
    this.pendingDetails.set(siteId, request);
    return request;
  }

  /** Starts (or restarts, for a retry) a site's load record at `phase`. */
  private startLoad(siteId: string, phase: "details" | "model", flight: boolean): void {
    this.loads.delete(siteId);
    this.reportLoad(siteId, {
      phase,
      progress: LOAD_PROGRESS[phase],
      error: null,
      retryable: false,
      attempt: 1,
      flight,
      startedAt: Date.now(),
    });
  }

  /**
   * Updates a site's load record and tells the HUD, which gets the whole record. Progress
   * never goes backwards, and a change of under a per cent is not worth an event.
   */
  private reportLoad(siteId: string, patch: Partial<SiteLoad>): void {
    const current = this.loads.get(siteId);
    const next: SiteLoad = {
      phase: "details",
      progress: 0,
      error: null,
      retryable: false,
      attempt: 1,
      flight: false,
      startedAt: Date.now(),
      ...current,
      ...patch,
    };
    if (current) next.progress = Math.max(current.progress, next.progress);
    if (next.phase === "ready") next.progress = 1;
    if (
      current?.phase === next.phase &&
      current.error === next.error &&
      current.attempt === next.attempt &&
      current.flight === next.flight &&
      next.progress - current.progress < 0.01
    )
      return;
    this.loads.set(siteId, next);
    this.events.emit("site-load", { siteId, load: next });
  }

  private failLoad(siteId: string, message: string, retryable: boolean): void {
    this.reportLoad(siteId, { phase: "error", error: message, retryable });
  }

  /**
   * Follows the shown model's first view into the site's load record until it is in:
   * tiles still to come against the most there ever were, then `ready` on Cesium's
   * `initialTilesLoaded`. A scan another renderer draws is never streamed here, so for that
   * one the tileset itself is the end of the load.
   */
  private watchFirstTiles(
    siteId: string,
    handle: AssetHandle,
    tileset: Cesium3DTileset,
    asset: SiteAsset,
  ): void {
    const load = this.loads.get(siteId);
    if (!load || load.phase === "ready" || load.phase === "error") return;
    if (!this.cesiumDraws(asset) || tileset.tilesLoaded) {
      this.reportLoad(siteId, { phase: "ready" });
      return;
    }
    this.reportLoad(siteId, { phase: "streaming", progress: LOAD_PROGRESS.streaming });
    let peak = 0;
    const span = 1 - LOAD_PROGRESS.streaming;
    // Only while this asset is the one shown: Splat / Mesh / Points can change mid-stream, and
    // the record then belongs to the representation shown now (`restartLoad`).
    const shown = (): boolean => {
      const entry = this.loaded.get(siteId);
      return entry !== undefined && this.pickAsset(entry, entry.representation)?.id === asset.id;
    };
    const offProgress = tileset.loadProgress.addEventListener(
      (pending: number, processing: number) => {
        if (!shown()) return;
        const left = pending + processing;
        peak = Math.max(peak, left);
        if (peak > 0) {
          this.reportLoad(siteId, { progress: LOAD_PROGRESS.streaming + span * (1 - left / peak) });
        }
      },
    );
    // `initialTilesLoaded` is raised once in a tileset's life; a model shown again later (a
    // representation switched back to) finishes on `allTilesLoaded`, raised on every drain.
    const done = (): void => {
      offProgress();
      offReady();
      offAll();
      if (shown() && this.loads.get(siteId)?.phase === "streaming")
        this.reportLoad(siteId, { phase: "ready" });
    };
    const offReady = tileset.initialTilesLoaded.addEventListener(done);
    const offAll = tileset.allTilesLoaded.addEventListener(done);
    handle.unsubscribe.push(offProgress, offReady, offAll);
  }

  /** Bounding sphere of the active representation (loaded tileset) or the footprint. */
  async boundingSphere(site: Site): Promise<BoundingSphere | null> {
    const entry = this.loaded.get(site.id);
    const handle = entry ? this.handleFor(entry) : null;
    const tileset = handle?.tileset ?? (handle?.loading ? await handle.loading : null);
    if (tileset && handle) {
      // A clamped model moves once the terrain height is known; fly to where it will be.
      await handle.placed;
      return tileset.boundingSphere;
    }
    const center = centerOf(site.boundary);
    const radius = Math.max(boundingRadiusM(site.boundary), 20);
    return new BoundingSphere(
      Cartesian3.fromDegrees(center.longitude, center.latitude, site.centroid.height ?? 0),
      radius,
    );
  }

  async setRepresentation(representation: Representation): Promise<void> {
    if (!this.active || this.active.representation === representation) return;
    this.active.representation = representation;
    this.active.temporalAssetId = null;
    this.events.emit("representation", { siteId: this.active.site.id, representation });
    this.restartLoad(this.active);
    await this.showRepresentation(this.active, representation);
  }

  async setTemporalAsset(assetId: string): Promise<void> {
    if (!this.active) return;
    const asset = this.active.site.assets.find((a) => a.id === assetId);
    if (!asset) return;
    this.active.temporalAssetId = assetId;
    if (asset.representation !== this.active.representation) {
      this.active.representation = asset.representation;
      this.events.emit("representation", {
        siteId: this.active.site.id,
        representation: asset.representation,
      });
    }
    this.restartLoad(this.active);
    await this.showRepresentation(this.active, asset.representation);
  }

  /**
   * Another model is about to be shown at a loaded site (Splat / Mesh / Points, a version, the
   * asset's Retry): the site's load record is about the model shown, so one that ended in an
   * error, or is still following the previous model, starts over at `model`. A model already
   * loaded is `ready` at once (`watchFirstTiles`). It used to keep the error: after the mesh
   * failed, switching back to a splat that was fine still said "Couldn't load the 3D model",
   * since `watchFirstTiles` leaves an errored record alone. A record that reached `ready`
   * stays: from there the asset's own state speaks (features/sites/siteLoad.ts).
   */
  private restartLoad(active: ActiveSite): void {
    const siteId = active.site.id;
    const load = this.loads.get(siteId);
    if (!load || load.phase === "ready") return;
    this.startLoad(siteId, "model", load.flight);
  }

  /** Every asset handle of every loaded site. */
  private *handles(): IterableIterator<{ entry: ActiveSite; handle: AssetHandle }> {
    for (const entry of this.loaded.values())
      for (const handle of entry.handles.values()) yield { entry, handle };
  }

  /** Called by the debug panel. */
  setDebug(options: { boundingVolumes?: boolean; wireframe?: boolean }): void {
    for (const { handle } of this.handles()) {
      if (!handle.tileset) continue;
      if (options.boundingVolumes !== undefined)
        handle.tileset.debugShowBoundingVolume = options.boundingVolumes;
      if (options.wireframe !== undefined) handle.tileset.debugWireframe = options.wireframe;
    }
    this.scene.requestRender();
  }

  activeTilesetLabels(): string[] {
    const labels: string[] = [];
    for (const { entry, handle } of this.handles()) {
      if (handle.tileset?.show) labels.push(`${entry.site.name} · ${handle.asset.name}`);
    }
    return labels;
  }

  private handleFor(entry: ActiveSite): AssetHandle | null {
    const asset = this.pickAsset(entry, entry.representation);
    return asset ? (entry.handles.get(asset.id) ?? null) : null;
  }

  private pickAsset(entry: ActiveSite, representation: Representation): SiteAsset | null {
    const candidates = entry.site.assets.filter((a) => a.representation === representation);
    if (entry.temporalAssetId) {
      const chosen = candidates.find((a) => a.id === entry.temporalAssetId);
      if (chosen) return chosen;
    }
    return (
      candidates.find((a) => a.defaultVisible) ??
      candidates.sort((a, b) => (b.observedAt ?? "").localeCompare(a.observedAt ?? ""))[0] ??
      null
    );
  }

  private async showRepresentation(
    active: ActiveSite,
    representation: Representation,
  ): Promise<void> {
    const asset = this.pickAsset(active, representation);
    for (const handle of active.handles.values()) {
      if (handle.tileset && handle.asset.id !== asset?.id) handle.tileset.show = false;
    }
    if (!asset) {
      this.clipping.setFootprint(active.site.id, null);
      this.events.emit("tilesets", this.activeTilesetLabels());
      // Nothing to load for this representation: the site's load is as done as it gets.
      if (this.loads.get(active.site.id)?.phase === "model")
        this.reportLoad(active.site.id, { phase: "ready" });
      return;
    }
    const tileset = await this.ensureTileset(active, asset);
    if (
      !tileset ||
      this.loaded.get(active.site.id) !== active ||
      this.pickAsset(active, active.representation)?.id !== asset.id
    )
      return;
    tileset.show = active.engaged && this.cesiumDraws(asset);
    if (active.engaged) this.applyClip(active, asset, tileset);
    else this.clipping.setFootprint(active.site.id, null);
    const handle = active.handles.get(asset.id);
    if (handle) this.watchFirstTiles(active.site.id, handle, tileset, asset);
    this.events.emit("tilesets", this.activeTilesetLabels());
    this.scene.requestRender();
  }

  /**
   * Whether the camera is close enough to a site for its model to take over from the world,
   * with hysteresis so the hand-over does not flicker at the threshold. A flight target is
   * always engaged, so the model is there on arrival.
   */
  private shouldEngage(entry: ActiveSite, current: boolean): boolean {
    if (this.flightTarget === entry.site.id) return true;
    const pose = this.camera.pose();
    const center = centerOf(entry.site.boundary);
    const distance = haversineDistance(
      { longitude: pose.longitude, latitude: pose.latitude },
      center,
    );
    const altitudeRadii = current ? DISENGAGE_ALTITUDE_RADII : ENGAGE_ALTITUDE_RADII;
    const distanceRadii = current ? DISENGAGE_DISTANCE_RADII : ENGAGE_DISTANCE_RADII;
    // Height above the site's own ground when the catalog knows it. The pose's altitude is
    // above whatever surface the globe can report, which right after a long flight, or with
    // the globe hidden under the photorealistic world, can be sea level: a model 190 m below
    // the camera then reads as 350 m away and hands back to the world on arrival.
    const ground = entry.site.centroid.height;
    const altitude =
      ground !== undefined && ground !== null && Number.isFinite(ground)
        ? Math.max(0, pose.height - ground)
        : pose.altitude;
    return altitude < entry.radius * altitudeRadii && distance < entry.radius * distanceRadii;
  }

  private setEngaged(entry: ActiveSite, engaged: boolean): void {
    if (entry.engaged === engaged) return;
    entry.engaged = engaged;
    const handle = this.handleFor(entry);
    const tileset = handle?.tileset;
    if (!handle || !tileset) return;
    tileset.show = engaged && this.cesiumDraws(handle.asset);
    if (engaged) this.applyClip(entry, handle.asset, tileset);
    else this.clipping.setFootprint(entry.site.id, null);
    log.info(engaged ? "site engaged" : "site disengaged", { site: entry.site.slug });
    this.events.emit("tilesets", this.activeTilesetLabels());
    this.scene.requestRender();
  }

  private async ensureTileset(
    active: ActiveSite,
    asset: SiteAsset,
  ): Promise<Cesium3DTileset | null> {
    let handle = active.handles.get(asset.id);
    if (handle?.tileset) return handle.tileset;
    if (handle?.loading) return handle.loading;
    handle = { asset, tileset: null, loading: null, placed: Promise.resolve(), unsubscribe: [] };
    active.handles.set(asset.id, handle);
    this.events.emit("asset", { id: asset.id, patch: { loadState: "loading", error: null } });
    handle.loading = timed(
      "site.asset.load",
      () =>
        withRetry(
          () =>
            withTimeout(
              createSiteTileset(asset, { maximumScreenSpaceError: this.screenSpaceError }),
              TILESET_ATTEMPT_TIMEOUT_MS,
              // An attempt that answers after its deadline made a tileset nobody will add.
              { what: asset.name, onLate: (late) => late.destroy() },
            ),
          {
            permanent: (error) => isIonAuthError(error) || isIonNotFound(error),
            onRetry: (error, attempt) => {
              log.info("asset retrying", { asset: asset.id, attempt, error: describeError(error) });
              if (this.showsAsset(active, asset))
                this.reportLoad(active.site.id, { attempt: attempt + 1 });
            },
          },
        ),
      {
        asset: asset.id,
        representation: asset.representation,
      },
    )
      .then((tileset) => {
        if (this.loaded.get(active.site.id) !== active) {
          tileset.destroy();
          return null;
        }
        // Drawn by another renderer: loaded for its frame and solids, never streamed here.
        if (!this.cesiumDraws(asset)) tileset.preloadWhenHidden = false;
        this.scene.primitives.add(tileset);
        this.attachTileset(handle, tileset, asset);
        return tileset;
      })
      .catch((error: unknown) => {
        const message = isIonAuthError(error)
          ? "Cesium ion refused this asset: the map key has no access to it."
          : isIonNotFound(error)
            ? "Cesium ion has no asset with this ID that the map key can see."
            : describeError(error);
        log.warn("asset failed", { asset: asset.id, error: message });
        this.events.emit("asset", { id: asset.id, patch: { loadState: "error", error: message } });
        const shown = this.showsAsset(active, asset);
        if (shown) {
          const permanent = isIonAuthError(error) || isIonNotFound(error);
          this.failLoad(active.site.id, `${asset.name} did not load: ${message}`, !permanent);
        }
        // The active site's model is said once, where the operator is looking: the load pill
        // beside Splat / Mesh / Points reads the record above ("Couldn't load the 3D model ·
        // Retry", `SiteLoadStatus`). A toast as well would say it twice, and with the asset's
        // file name rather than the site's. Any other failure has no pill to speak for it.
        if (!shown || this.primaryId !== active.site.id) {
          this.events.emit("toast", {
            tone: "error",
            title: `${asset.name} failed to load`,
            body: message,
            id: `asset-${asset.id}`,
          });
        }
        if (isIonAuthError(error)) this.events.emit("token", "invalid");
        return null;
      })
      .finally(() => {
        if (handle) handle.loading = null;
      });
    return handle.loading;
  }

  /** Whether `asset` is the one a loaded site shows, so its load is the site's load. */
  private showsAsset(active: ActiveSite, asset: SiteAsset): boolean {
    return (
      this.loaded.get(active.site.id) === active &&
      this.pickAsset(active, active.representation)?.id === asset.id
    );
  }

  private attachTileset(handle: AssetHandle, tileset: Cesium3DTileset, asset: SiteAsset): void {
    handle.tileset = tileset;
    this.applyScreenSpaceError(this.screenSpaceError, this.pixelRatio);
    if (asset.renderConfig.clampToGround) handle.placed = this.clampToGround(tileset, asset);
    const offset = asset.renderConfig.heightOffsetM ?? 0;
    if (offset !== 0 && !asset.renderConfig.clampToGround) {
      const center = Cartographic.fromCartesian(tileset.boundingSphere.center);
      const surface = Cartesian3.fromRadians(center.longitude, center.latitude, 0);
      const lifted = Cartesian3.fromRadians(center.longitude, center.latitude, offset);
      tileset.modelMatrix = Matrix4.fromTranslation(
        Cartesian3.subtract(lifted, surface, new Cartesian3()),
      );
    }
    const radius = tileset.boundingSphere.radius;
    this.events.emit("asset", {
      id: asset.id,
      patch: {
        loadState: "ready",
        error: null,
        boundingRadiusM: radius,
        screenSpaceError: tileset.maximumScreenSpaceError,
      },
    });
    if (asset.representation === "gaussian-splat") {
      const splats = new SplatCount();
      handle.splats = splats;
      handle.unsubscribe.push(
        // Loaded splats are counted for the budget; which tiles go when memory runs short is
        // the tileset cache's choice (least recently used), now that splat tiles report
        // their bytes (engine patch, GaussianSplat3DTileContent.geometryByteLength).
        tileset.tileLoad.addEventListener((tile: Cesium3DTile) => {
          splats.load(tile);
          sizeSplatCache(tileset, tile, this.splatCeiling);
        }),
        tileset.tileUnload.addEventListener((tile: Cesium3DTile) => splats.unload(tile)),
        // What the capture never saw, faded from the views it never had (lib/viewCones.ts).
        attachViewCones(tileset),
        // What an image model filled in where it never looked, beside it (lib/inferred.ts).
        attachInferredLayers(tileset, this.scene, asset.id),
        attachInstances(tileset, this.scene, asset.id),
        // Movable objects split into tilesets of their own, placed by their poses (C4).
        attachSplitObjects(tileset, this.scene, asset.id),
        // Objects that move by their skins, once a driver sets handles (lib/skin.ts).
        attachSkin(tileset, this.scene, asset.id),
        // Objects a live pose stream moves (lib/telemetry.ts), when the scan binds any.
        attachTelemetry(tileset, this.scene, asset.id),
      );
    }
    // Loading progress reaches the store at most four times a second (throttleProgress): Cesium
    // reports it up to once a rendered frame, and every asset patch re-renders whatever reads
    // the assets (the representation switcher reads them all). The end of loading goes
    // through at once, and the quality policy hears every report, unthrottled.
    const reportProgress = throttleProgress((pending, processing) => {
      const counted = (handle.splats?.total ?? 0) * SPLAT_BYTES_ESTIMATE;
      this.events.emit("asset", {
        id: asset.id,
        patch: {
          progress: { pending, processing },
          memoryMb: Math.round(Math.max(tileset.totalMemoryUsageInBytes, counted) / 1048576),
          screenSpaceError: tileset.maximumScreenSpaceError,
        },
      });
    });
    handle.unsubscribe.push(
      tileset.loadProgress.addEventListener((pending: number, processing: number) => {
        this.performance.reportLoading("sites", pending, processing);
        reportProgress(pending, processing);
      }),
      () => reportProgress.cancel(),
      tileset.initialTilesLoaded.addEventListener(() => this.camera.refreshPose()),
      tileset.allTilesLoaded.addEventListener(() => this.camera.refreshPose()),
      tileset.tileFailed.addEventListener((detail: { message?: string; url?: string }) => {
        log.warn("tile failed", { asset: asset.id, message: detail.message, url: detail.url });
      }),
    );
  }

  /** Memory held by the visible site tilesets against their configured cache budget. Splat
   *  tilesets are budgeted by what they draw instead (splatMemory); their cache holds more. */
  private memoryUsage(): { bytes: number; budget: number } {
    const { cacheBytes, maximumCacheOverflowBytes } = tileCacheBudget();
    let bytes = 0;
    for (const { handle } of this.handles()) {
      if (handle.tileset?.show && !handle.splats) bytes += handle.tileset.totalMemoryUsageInBytes;
    }
    return { bytes, budget: cacheBytes + maximumCacheOverflowBytes };
  }

  /** Gaussians drawn across the visible splat tilesets: what their snapshots hold. */
  private splatsDrawn(): number {
    let total = 0;
    for (const { handle } of this.handles()) {
      if (!handle.tileset?.show || !handle.splats) continue;
      const primitive = splatTilesetOf(handle.tileset).gaussianSplatPrimitive;
      // Live slots, not the slot range: hidden tiles stay resident but are not drawn.
      total += primitive?._liveSplats ?? primitive?._numSplats ?? 0;
    }
    return total;
  }

  /**
   * Clips the coarse world under the model. With `clipFootprint: "tileset"` the clip follows
   * the tiles' real coverage; that is only known once the first branching sub-tileset has
   * loaded, so until then the root box is used and the clip is re-derived on tile loads.
   */
  /**
   * Rests a model on the ground. Downloaded objects and phone scans carry no usable ellipsoid
   * height, so the catalog stores where, and the viewer works out how high once the terrain is
   * known.
   *
   * Two ways of working it out, and which one is used depends only on whether the capture's own
   * ground was ever measured:
   *
   * * **measured** — the asset carries `groundSamples`, the capture's own ground cell by cell
   *   as ellipsoid heights (`splat_ground` measured them; the worker put them on the asset).
   *   The ground is sampled at exactly those points and the median per-cell difference is the
   *   lift. Slope cancels, because both sides of every subtraction are at the same place. This
   *   is the subtraction a person used to do by hand into `heightOffsetM`.
   * * **bounding box** — no samples, so the lowest corner of the root box goes on the ground
   *   under the centre, exactly as before B4, `heightOffsetM` and all. Every asset that
   *   predates the measured path takes this branch and does not move by a millimetre.
   *
   * `heightOffsetM` is still added in both branches. On the measured branch it is no longer
   * *needed* — nothing about the clamp requires correcting any more — but it is an authored
   * value, and an asset somebody deliberately nudged should stay nudged.
   */
  private async clampToGround(tileset: Cesium3DTileset, asset: SiteAsset): Promise<void> {
    const sphere = tileset.boundingSphere;
    const center = Cartographic.fromCartesian(sphere.center);
    const measured = asset.renderConfig.groundSamples ?? [];
    // The centre first, then one point per measured cell, so index 0 is always the centre and
    // the bounding-box fallback is available even when every cell's sample fails.
    const wanted: MeasuredGround[] = [
      {
        lon: CesiumMath.toDegrees(center.longitude),
        lat: CesiumMath.toDegrees(center.latitude),
        height: center.height,
      },
      ...measured,
    ];
    const cartographics = () => wanted.map((p) => Cartographic.fromDegrees(p.lon, p.lat));
    const terrain = await sampleTerrainMostDetailed(
      this.viewer.terrainProvider,
      cartographics(),
    ).catch(() => undefined);
    // The ground that is actually drawn may be a mesh (the photorealistic world, another
    // site's model) sitting metres from the terrain; rest on what is visible when there is
    // something plausible there.
    const drawn = this.scene.sampleHeightSupported
      ? await this.scene.sampleHeightMostDetailed(cartographics(), [tileset]).catch(() => undefined)
      : undefined;
    if (tileset.isDestroyed()) return;
    const ground = wanted.map((_, i) =>
      groundAt(terrain?.[i]?.height, drawn?.[i]?.height, DRAWN_GROUND_TOLERANCE_M),
    );
    const offset = asset.renderConfig.heightOffsetM ?? 0;
    const clamp = measured.length > 0 ? measuredClamp(measured, ground.slice(1)) : null;
    let lift: number;
    if (clamp) {
      lift = clamp.liftM + offset;
      log.info("clamped model to its own measured ground", {
        asset: asset.id,
        cells: clamp.cells,
        of: measured.length,
        lift,
        spread: Math.round(clamp.spreadM * 100) / 100,
      });
    } else {
      const under = ground[0];
      if (under === undefined) return;
      // Lowest point of the root bounding box when there is one (a sphere would float a flat
      // object by the difference between its radius and its half height).
      const bottom = lowestHeight(tileset) ?? center.height - sphere.radius;
      lift = under + offset - bottom;
      log.info("clamped model to ground", {
        asset: asset.id,
        ground: Math.round(under),
        lift,
        measuredCells: measured.length,
      });
    }
    const from = Cartesian3.fromRadians(center.longitude, center.latitude, center.height);
    const to = Cartesian3.fromRadians(center.longitude, center.latitude, center.height + lift);
    tileset.modelMatrix = Matrix4.fromTranslation(Cartesian3.subtract(to, from, new Cartesian3()));
    this.scene.requestRender();
  }

  private applyClip(active: ActiveSite, asset: SiteAsset, tileset: Cesium3DTileset): void {
    const blended =
      asset.representation === "gaussian-splat" || asset.representation === "point-cloud";
    if (asset.renderConfig.clipsWorld && blended) {
      // Splats and point clouds are drawn over the opaque globe with a depth test, so terrain
      // does not fight them; their ground layer covers it. Cutting the terrain instead leaves
      // a see-through hole wherever the capture is sparse or between points (tile boxes
      // include outliers, so no tile-derived footprint is tight). Only the global 3D tileset
      // is cut, so buildings from OSM or Google do not poke through the model. The authored
      // footprint comes first: a splat's root box spans every outlier splat (the demo's is
      // 1.7 × 2.8 km around a campus) and would blank the photorealistic world for blocks.
      const footprint = asset.footprint ?? active.site.boundary ?? footprintFromTileset(tileset);
      this.clipping.setFootprint(active.site.id, footprint, { globe: false, world: true });
      return;
    }
    let footprint: Footprint | null = null;
    let provisional = false;
    const authored = asset.footprint ?? active.site.boundary;
    if (asset.renderConfig.clipsWorld) {
      if (asset.renderConfig.clipFootprint === "tileset") {
        const coverage = tighter(coverageFromTileset(tileset), authored);
        provisional = coverage === null;
        footprint = coverage ?? authored;
      } else {
        footprint = authored;
      }
    }
    this.clipping.setFootprint(active.site.id, footprint);
    if (asset.renderConfig.clipFootprint !== "tileset") return;
    const handle = active.handles.get(asset.id);
    if (!handle || handle.coverageWatched) return;
    handle.coverageWatched = true;
    let applied = provisional ? "" : coverageKey(footprint);
    let timer: ReturnType<typeof setTimeout> | null = null;
    // Sub-tilesets stream in over time; re-derive the coverage after each burst of tile loads
    // and swap the clip only when it actually changed (rebuilding it is not free). Never
    // mid-gesture: rasterising the new polygons is a visible hitch, so it waits for rest.
    const refresh = (): void => {
      timer = null;
      if (this.loaded.get(active.site.id) !== active || !tileset.show) return;
      if (this.camera.isMoving) {
        timer = setTimeout(refresh, COVERAGE_REFRESH_MS);
        return;
      }
      const coverage = tighter(coverageFromTileset(tileset), authored);
      if (!coverage) return;
      const key = coverageKey(coverage);
      if (key === applied) return;
      applied = key;
      this.clipping.setFootprint(active.site.id, coverage);
      this.scene.requestRender();
    };
    const off = tileset.tileLoad.addEventListener(() => {
      if (timer !== null) return;
      timer = setTimeout(refresh, COVERAGE_REFRESH_MS);
    });
    handle.unsubscribe.push(() => {
      off();
      if (timer !== null) clearTimeout(timer);
    });
  }

  private applyScreenSpaceError(sse: number, pixelRatio: number): void {
    this.screenSpaceError = sse;
    this.pixelRatio = pixelRatio;
    this.metersPerPixel = this.camera.pose().metersPerPixel;
    this.appliedMetersPerPixel = this.metersPerPixel;
    for (const { handle } of this.handles()) {
      if (!handle.tileset) continue;
      const configured = handle.asset.renderConfig.maximumScreenSpaceError;
      // A per-asset value acts as a floor for quality (never coarser than configured), while
      // splats have a hard floor on refinement because of their per-frame CPU sort.
      let next = configured ? Math.min(configured, sse) : sse;
      // The device's Detail choice then scales it (lib/detail.ts): a pipeline scan is a
      // level-of-detail tileset holding every gaussian, and this is what decides how many of
      // them this device draws. A single-tile scan (the committed tree, anything packaged
      // before the hierarchy) has nothing to refine, so it is drawn whole either way.
      if (handle.asset.representation === "gaussian-splat")
        next =
          Math.max(next, this.performance.splatMinimumScreenSpaceError) * this.splatDetailScale;
      // Cesium measures the error in CSS pixels; hand it device pixels so a HiDPI screen
      // gets the detail it can show, and a resolution cut also lightens the tile load. The
      // asset's calibration comes last: a tiler's geometric errors say nothing about texture
      // sharpness, and a survey mesh may need a finer error than the world to look as sharp.
      next =
        devicePixelError(next, pixelRatio) *
        calibrationFor(handle.asset.renderConfig.screenSpaceErrorScale ?? 1, this.metersPerPixel);
      next = Math.round(next * 16) / 16;
      if (handle.tileset.maximumScreenSpaceError === next) continue;
      handle.tileset.maximumScreenSpaceError = next;
      this.events.emit("asset", {
        id: handle.asset.id,
        patch: { screenSpaceError: handle.tileset.maximumScreenSpaceError },
      });
      // Tile selection only runs during a frame; without this the new detail never loads.
      this.scene.requestRender();
    }
  }

  /**
   * Loads every site the camera is near, unloads the ones it has left, chooses which loaded
   * site is primary (the one the switcher, clipping and HUD describe), and switches the
   * camera to object scale beside a hand-sized model. Sites overlap: a 14 cm rock can be
   * registered on top of a campus, and both must stay loaded while you look at either.
   */
  private checkProximity(force = false): void {
    const now = performance.now();
    if (!force && now - this.lastProximityCheck < 400) return;
    this.lastProximityCheck = now;
    const pose = this.camera.pose();
    const here = { longitude: pose.longitude, latitude: pose.latitude };
    const ranked = this.summaries
      .map((summary) => {
        const distance = haversineDistance(here, summary.centroid);
        const radius = Math.max(Math.sqrt(summary.areaM2 / Math.PI), MIN_SITE_RADIUS_M);
        // Distance in units of the site's own size, so a campus 1 km away still outranks a
        // rock 1 km away, while the rock wins once you are standing beside it.
        return { summary, distance, radius, score: distance / radius };
      })
      .sort((a, b) => a.score - b.score);

    for (const { summary, distance } of ranked) {
      const entry = this.loaded.get(summary.id);
      if (
        entry &&
        distance > DEACTIVATE_DISTANCE_M &&
        pose.altitude > DEACTIVATE_DISTANCE_M / 4 &&
        this.flightTarget !== summary.id
      ) {
        log.info("unloading distant site", { site: entry.site.slug });
        this.deactivate(summary.id);
      } else if (!entry && distance < ACTIVATE_DISTANCE_M && pose.altitude < ACTIVATE_DISTANCE_M) {
        void this.activate(summary.id, { primary: false });
      }
    }

    // The pill follows a fly-to's destination until the camera has left it, once the flight is
    // over (landed or taken over): further out than a site hands its model back to the world,
    // or higher than sites load at all. Landing at a site whose record failed keeps it, with
    // its error and Retry; a small site registered inside it (a rock on a campus) does not
    // take the pill away, as "the nearest site" would.
    const flown = this.flightSiteId;
    if (flown !== null && this.flightTarget !== flown) {
      const there = ranked.find((r) => r.summary.id === flown);
      const stays =
        there !== undefined &&
        there.distance < there.radius * DISENGAGE_DISTANCE_RADII &&
        pose.altitude < ACTIVATE_DISTANCE_M;
      if (!stays) this.setFlightSite(null);
    }

    for (const entry of this.loaded.values())
      this.setEngaged(entry, this.shouldEngage(entry, entry.engaged));

    const best = ranked.find((r) => this.loaded.has(r.summary.id));
    if (best && this.flightTarget === null && best.summary.id !== this.primaryId)
      this.setPrimary(best.summary.id);

    const nearest = ranked[0];
    const near =
      nearest && nearest.distance < nearest.radius * 6 && pose.altitude < NEAR_ALTITUDE_M
        ? nearest.summary.id
        : null;
    if (near !== this.nearId) {
      this.nearId = near;
      this.events.emit("site-near", near);
    }
    this.performance.reportContext(pose.altitude, near !== null);
    this.updateObjectScale();
    this.refreshCalibration();
  }

  /**
   * The calibration taper depends on the view scale; re-apply it once the camera rests
   * (never mid-gesture, a changed error pops tiles) and the scale moved a good step. Also
   * ticked on a timer: in request-render mode a programmatic move may not be followed by a
   * frame, so moveEnd alone is not a reliable trigger.
   */
  private refreshCalibration(): void {
    if (this.camera.isMoving || this.loaded.size === 0) return;
    const metersPerPixel = this.camera.pose().metersPerPixel;
    if (!Number.isFinite(metersPerPixel)) return;
    const moved = Math.abs(Math.log(metersPerPixel / this.appliedMetersPerPixel));
    if (moved > CALIBRATION_STEP || !Number.isFinite(moved))
      this.applyScreenSpaceError(this.screenSpaceError, this.pixelRatio);
  }

  /** Whether the camera is inside a shown splat scan's bounds: the scan is the whole view. */
  insideSplatScan(): boolean {
    const cameraPosition = this.viewer.camera.positionWC;
    for (const { entry, handle } of this.handles()) {
      const tileset = handle.tileset;
      // Drawn by CesiumJS (shown) or by a dedicated renderer (engaged, hidden here).
      if (!tileset || !handle.splats || !(tileset.show || this.scanDrawnElsewhere(entry, handle)))
        continue;
      const sphere = tileset.boundingSphere;
      if (Cartesian3.distance(cameraPosition, sphere.center) < sphere.radius) return true;
    }
    return false;
  }

  /**
   * Whether CesiumJS draws this asset itself: anything but a splat under another renderer, and
   * a Living Survey scan whatever the renderer -- its motion is CesiumJS's splat shader
   * (LivingSurveyManager), which no other renderer has.
   */
  private cesiumDraws(asset: SiteAsset): boolean {
    return (
      this.splatRenderer === "cesium" ||
      asset.representation !== "gaussian-splat" ||
      Boolean(asset.renderConfig.rigUrl)
    );
  }

  private scanDrawnElsewhere(entry: ActiveSite, handle: AssetHandle): boolean {
    return (
      !this.cesiumDraws(handle.asset) &&
      entry.engaged &&
      entry.representation === "gaussian-splat" &&
      this.pickAsset(entry, entry.representation)?.id === handle.asset.id
    );
  }

  /**
   * Chooses who draws splat scans. Another renderer than CesiumJS hides the splat tilesets
   * (and stops them streaming: nothing is preloaded while hidden), but keeps them loaded for
   * their frame, placement and packaged solids.
   */
  setSplatRenderer(kind: SplatRendererKind): void {
    if (kind === this.splatRenderer) return;
    this.splatRenderer = kind;
    for (const { entry, handle } of this.handles()) {
      const tileset = handle.tileset;
      if (!tileset || handle.asset.representation !== "gaussian-splat") continue;
      tileset.preloadWhenHidden = this.cesiumDraws(handle.asset);
      const current = this.pickAsset(entry, entry.representation)?.id === handle.asset.id;
      tileset.show = current && entry.engaged && this.cesiumDraws(handle.asset);
    }
    this.scene.requestRender();
  }

  /** The engaged splat scan a dedicated renderer should draw, if any. */
  scanTarget(): { key: string; tileset: Cesium3DTileset; assetId: string } | null {
    if (this.splatRenderer === "cesium") return null;
    const active = this.active;
    if (!active?.engaged || active.representation !== "gaussian-splat") return null;
    const asset = this.pickAsset(active, active.representation);
    const handle = asset ? active.handles.get(asset.id) : undefined;
    if (!asset || this.cesiumDraws(asset) || !handle?.tileset || handle.tileset.isDestroyed())
      return null;
    return { key: `${active.site.id}:${asset.id}`, tileset: handle.tileset, assetId: asset.id };
  }

  /** Object scale while the camera is within reach of a hand-sized loaded model. */
  private updateObjectScale(): void {
    const cameraPosition = this.viewer.camera.positionWC;
    let near = false;
    for (const { handle } of this.handles()) {
      const tileset = handle.tileset;
      if (!tileset?.show || tileset.boundingSphere.radius >= OBJECT_SCALE_RADIUS_M) continue;
      const reach = tileset.boundingSphere.radius + OBJECT_SCALE_REACH_M;
      if (Cartesian3.distance(cameraPosition, tileset.boundingSphere.center) < reach) {
        near = true;
        break;
      }
    }
    if (near !== this.objectScale) {
      this.objectScale = near;
      this.camera.setObjectScale(near);
    }
  }

  private disposeHandle(handle: AssetHandle): void {
    for (const off of handle.unsubscribe) off();
    if (handle.tileset) {
      this.scene.primitives.remove(handle.tileset);
      handle.tileset = null;
    }
    this.events.emit("asset", {
      id: handle.asset.id,
      patch: { loadState: "idle", progress: { pending: 0, processing: 0 } },
    });
  }

  destroy(): void {
    for (const off of this.unsubscribe) off();
    this.cancelPrefetch?.();
    this.cancelPrefetch = null;
    this.deactivate();
  }
}

/**
 * Derives a lon/lat footprint from a loaded tileset's root bounding volume:
 * a region's rectangle, an oriented box's horizontal corners, or a sphere's circle.
 */
export function footprintFromTileset(tileset: Cesium3DTileset): Footprint | null {
  return footprintFromTile(tileset.root);
}

/**
 * The area a tileset really covers: the union of its first-level tiles' footprints. A root
 * bounding box is usually much larger than the capture (this demo's is an L inside a rectangle),
 * and clipping the whole box leaves a see-through hole around the model. Sub-tileset entries
 * carry bounding volumes as soon as the root document is parsed, so this needs no tile loads.
 * Returns null when the root has no children, so callers fall back to the root box.
 */
export function coverageFromTileset(tileset: Cesium3DTileset): Footprint | null {
  // Walk down single-child chains (a root that only wraps one external tileset) to the first
  // level that actually branches; an unloaded chain end means "not known yet".
  let tile: Cesium3DTile = tileset.root;
  for (let depth = 0; depth < MAX_COVERAGE_DEPTH; depth++) {
    const only = tile.children.length === 1 ? tile.children[0] : undefined;
    if (!only) break;
    tile = only;
  }
  if (tile.children.length < 2) return null;
  // Then take the finest loaded level below it, a few levels deep, so the clip tightens as
  // sub-tilesets stream in. Tiles whose children have not loaded contribute their own box.
  const polygons: [number, number][][][] = [];
  // External tileset references add a single-child wrapper per level; only branching counts.
  const visit = (node: Cesium3DTile, level: number): void => {
    if (polygons.length >= MAX_COVERAGE_TILES) return;
    const only = node.children.length === 1 ? node.children[0] : undefined;
    if (only) {
      visit(only, level);
      return;
    }
    if (level < COVERAGE_LEVELS && node.children.length > 1) {
      for (const child of node.children) visit(child, level + 1);
      return;
    }
    // Spheres are never tight (a coarse tile's sphere reaches far past its content and,
    // cut out of the world, shows as a black circle of sky); only boxes and regions count.
    if (!hasTightVolume(node)) return;
    const footprint = footprintFromTile(node);
    if (footprint?.type === "Polygon") polygons.push(footprint.coordinates as [number, number][][]);
  };
  visit(tile, 0);
  if (polygons.length === 0) return null;
  return { type: "MultiPolygon", coordinates: polygons };
}

/** Whether a tile's bounding volume is a region or an oriented box (a sphere is never tight). */
function hasTightVolume(tile: Cesium3DTile): boolean {
  const volume = (tile as unknown as { boundingVolume?: TileVolume }).boundingVolume;
  return Boolean(volume?.rectangle ?? volume?.boundingVolume?.halfAxes);
}

/**
 * The tile-derived coverage is only used while it is at least as tight as the authored
 * footprint: before the fine tiles have loaded, coarse tiles' volumes reach well past the
 * data, and cutting the world along them leaves holes around the model.
 */
export function tighter(coverage: Footprint | null, authored: Footprint | null): Footprint | null {
  if (!coverage) return null;
  if (!authored) return coverage;
  return bboxArea(coverage) <= bboxArea(authored) * COVERAGE_SLACK ? coverage : null;
}

/** Bounding-box area of a footprint in square degrees (only ever compared at one latitude). */
function bboxArea(footprint: Footprint): number {
  let west = Number.POSITIVE_INFINITY;
  let south = Number.POSITIVE_INFINITY;
  let east = Number.NEGATIVE_INFINITY;
  let north = Number.NEGATIVE_INFINITY;
  const polygons = footprint.type === "Polygon" ? [footprint.coordinates] : footprint.coordinates;
  for (const polygon of polygons)
    for (const point of polygon[0] ?? []) {
      const [lon, lat] = point;
      if (lon === undefined || lat === undefined) continue;
      west = Math.min(west, lon);
      east = Math.max(east, lon);
      south = Math.min(south, lat);
      north = Math.max(north, lat);
    }
  return Number.isFinite(west) ? Math.max(0, east - west) * Math.max(0, north - south) : 0;
}

/** Ellipsoid height of the lowest corner of a tileset's root oriented bounding box. */
function lowestHeight(tileset: Cesium3DTileset): number | undefined {
  const inner = (tileset.root as unknown as { boundingVolume?: TileVolume }).boundingVolume
    ?.boundingVolume;
  const halfAxes = inner?.halfAxes;
  if (!inner?.center || !halfAxes) return undefined;
  let lowest = Number.POSITIVE_INFINITY;
  const axes = [0, 1, 2].map((i) => Matrix3.getColumn(halfAxes, i, new Cartesian3()));
  for (const sx of [-1, 1])
    for (const sy of [-1, 1])
      for (const sz of [-1, 1]) {
        const corner = Cartesian3.clone(inner.center, new Cartesian3());
        for (const [axis, sign] of [
          [axes[0], sx],
          [axes[1], sy],
          [axes[2], sz],
        ] as const) {
          if (axis)
            Cartesian3.add(
              corner,
              Cartesian3.multiplyByScalar(axis, sign, new Cartesian3()),
              corner,
            );
        }
        lowest = Math.min(lowest, Cartographic.fromCartesian(corner).height);
      }
  return Number.isFinite(lowest) ? lowest : undefined;
}

/** Lon/lat footprint of one tile: a region's rectangle, an oriented box's horizontal corners, or a sphere's circle. */
export function footprintFromTile(tile: Cesium3DTile): Footprint | null {
  // `Cesium3DTile.boundingVolume` (a TileBoundingVolume) is not in the public typings but is
  // stable at runtime; fall back to the public bounding sphere when it is absent.
  const volume = (tile as unknown as { boundingVolume?: TileVolume }).boundingVolume ?? {
    boundingVolume: {
      center: tile.boundingSphere.center,
      radius: tile.boundingSphere.radius,
    },
  };
  if (volume.rectangle) {
    const r = volume.rectangle;
    return rectanglePolygon(
      CesiumMath.toDegrees(r.west),
      CesiumMath.toDegrees(r.south),
      CesiumMath.toDegrees(r.east),
      CesiumMath.toDegrees(r.north),
    );
  }
  const inner = volume.boundingVolume;
  if (inner?.center && inner.halfAxes) {
    const col0 = Matrix3.getColumn(inner.halfAxes, 0, new Cartesian3());
    const col1 = Matrix3.getColumn(inner.halfAxes, 1, new Cartesian3());
    const col2 = Matrix3.getColumn(inner.halfAxes, 2, new Cartesian3());
    // Pick the two axes most parallel to the local horizontal plane.
    const up = Cartesian3.normalize(inner.center, new Cartesian3());
    const axes = [col0, col1, col2].sort(
      (a, b) =>
        Math.abs(Cartesian3.dot(Cartesian3.normalize(a, new Cartesian3()), up)) -
        Math.abs(Cartesian3.dot(Cartesian3.normalize(b, new Cartesian3()), up)),
    );
    const [ax, ay] = axes as [Cartesian3, Cartesian3, Cartesian3];
    const ring: [number, number][] = [];
    for (const [sx, sy] of [
      [1, 1],
      [-1, 1],
      [-1, -1],
      [1, -1],
    ] as const) {
      const point = Cartesian3.add(
        inner.center,
        Cartesian3.add(
          Cartesian3.multiplyByScalar(ax, sx, new Cartesian3()),
          Cartesian3.multiplyByScalar(ay, sy, new Cartesian3()),
          new Cartesian3(),
        ),
        new Cartesian3(),
      );
      const carto = Cartographic.fromCartesian(point);
      ring.push([CesiumMath.toDegrees(carto.longitude), CesiumMath.toDegrees(carto.latitude)]);
    }
    const first = ring[0];
    if (!first) return null;
    ring.push(first);
    return { type: "Polygon", coordinates: [ring] };
  }
  if (inner?.center && inner.radius) {
    const carto = Cartographic.fromCartesian(inner.center);
    return circleFootprint(
      {
        longitude: CesiumMath.toDegrees(carto.longitude),
        latitude: CesiumMath.toDegrees(carto.latitude),
      },
      inner.radius,
      32,
    );
  }
  return null;
}

/**
 * An asset's screen-space error calibration is a distance measure. A tiler assigns geometric
 * error from geometry, so at kilometres a survey mesh's coarse levels look softer than their
 * error says and need the full factor (the SF mesh: 0.5 px draws its detail 7 km up where
 * 2 px draws a grey blob); up close the finest levels already carry textures many times
 * finer than a screen pixel, and asking for 0.5 px there fetches several times the data for
 * no visible gain (measured 317 m up: 2 px settles sharp at 217 tiles, 0.5 px never settled).
 * The factor therefore fades from the asset's value at 8 m per pixel to 1 at 0.5 m per pixel.
 */
export function calibrationFor(scale: number, metersPerPixel: number): number {
  if (scale === 1 || !Number.isFinite(metersPerPixel)) return scale === 1 ? 1 : scale;
  const t =
    (Math.log(metersPerPixel) - Math.log(CALIBRATION_NEAR_MPP)) /
    (Math.log(CALIBRATION_FAR_MPP) - Math.log(CALIBRATION_NEAR_MPP));
  const clamped = Math.min(1, Math.max(0, t));
  return 1 + (scale - 1) * clamped;
}
const CALIBRATION_NEAR_MPP = 0.5;
const CALIBRATION_FAR_MPP = 8;
/** Re-apply the taper when the view scale moved by this much in log space (about 35 %). */
const CALIBRATION_STEP = 0.3;
const CALIBRATION_TICK_MS = 1000;

/** Clipping polygons are rasterised into one texture; keep the count bounded. */
const MAX_COVERAGE_TILES = 96;
/** Models smaller than this radius get the millimetre zoom floor and near plane. */
const OBJECT_SCALE_RADIUS_M = 30;
const MAX_COVERAGE_DEPTH = 4;
/** How many branching levels below the first split the coverage follows (4^3 boxes at most). */
const COVERAGE_LEVELS = 3;
const COVERAGE_REFRESH_MS = 1000;
/** Tile coverage may exceed the authored footprint's bounding box by this factor and still be used. */
const COVERAGE_SLACK = 1.15;
/** A drawn surface further than this from the terrain is something else (a roof, a tree), not ground. */
const DRAWN_GROUND_TOLERANCE_M = 60;

/** Cheap identity for a coverage footprint: polygon count plus the first coordinate of each. */
function coverageKey(footprint: Footprint | null): string {
  if (footprint?.type !== "MultiPolygon") return "";
  return footprint.coordinates
    .map((polygon) => polygon[0]?.[0]?.map((n) => n.toFixed(5)).join(",") ?? "-")
    .join("|");
}

interface TileVolume {
  rectangle?: Rectangle;
  boundingVolume?: { center?: Cartesian3; halfAxes?: Matrix3; radius?: number };
}

function rectanglePolygon(west: number, south: number, east: number, north: number): Footprint {
  return {
    type: "Polygon",
    coordinates: [
      [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
      ],
    ],
  };
}

export function degrees(radians: number): number {
  return CesiumMath.toDegrees(radians);
}
