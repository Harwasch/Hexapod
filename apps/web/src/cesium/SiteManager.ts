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
  type Viewer,
} from "cesium";

import type { Footprint, Representation, Site, SiteAsset, SiteSummary } from "@twin/contracts";
import { boundingRadiusM, centerOf, circleFootprint, haversineDistance } from "@twin/geo";

import type { Emitter } from "@/lib/emitter";
import { createLogger, describeError } from "@/lib/log";
import { timed } from "@/lib/timing";

import type { CameraController } from "./CameraController";
import type { ClippingManager } from "./ClippingManager";
import { isIonAuthError, isIonNotFound } from "./ion";
import type { PerformanceManager } from "./PerformanceManager";
import {
  SPLAT_MIN_SCREEN_SPACE_ERROR,
  createSiteTileset,
  tileCacheBudget,
} from "./providers/tiles";
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

interface AssetHandle {
  asset: SiteAsset;
  tileset: Cesium3DTileset | null;
  loading: Promise<Cesium3DTileset | null> | null;
  unsubscribe: (() => void)[];
}

interface ActiveSite {
  site: Site;
  representation: Representation;
  temporalAssetId: string | null;
  handles: Map<string, AssetHandle>;
}

/**
 * Loads a site's reality models into the world when they are useful (fly-to or
 * proximity), switches representations without moving the camera, and keeps
 * the coarse world clipped underneath the active model.
 */
export class SiteManager {
  private readonly scene: Scene;
  private summaries: SiteSummary[] = [];
  private detailResolver: (id: string) => Promise<Site | null> = () => Promise.resolve(null);
  private active: ActiveSite | null = null;
  private nearId: string | null = null;
  private screenSpaceError = 16;
  private flightTarget: string | null = null;
  private readonly unsubscribe: (() => void)[] = [];
  private lastProximityCheck = 0;

  constructor(
    private readonly viewer: Viewer,
    private readonly events: Emitter<SceneEvents>,
    private readonly camera: CameraController,
    private readonly clipping: ClippingManager,
    private readonly performance: PerformanceManager,
  ) {
    this.scene = viewer.scene;
    this.performance.bindScreenSpaceErrorSink((sse) => this.applyScreenSpaceError(sse));
    this.performance.bindMemorySource(() => this.memoryUsage());
    this.unsubscribe.push(
      viewer.camera.changed.addEventListener(() => this.checkProximity()),
      viewer.camera.moveEnd.addEventListener(() => this.checkProximity(true)),
    );
  }

  /** Catalog summaries used for proximity activation; details are fetched lazily. */
  setCatalog(summaries: SiteSummary[], resolver: (id: string) => Promise<Site | null>): void {
    this.summaries = summaries;
    this.detailResolver = resolver;
    this.checkProximity(true);
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

  /** Temporal versions of the active representation, newest first. */
  temporalVersions(): SiteAsset[] {
    if (!this.active) return [];
    const rep = this.active.representation;
    return this.active.site.assets
      .filter((a) => a.representation === rep && a.observedAt)
      .sort((a, b) => (b.observedAt ?? "").localeCompare(a.observedAt ?? ""));
  }

  /** Loads a site's assets into the scene without moving the camera. */
  async activate(siteId: string): Promise<Site | null> {
    if (this.active?.site.id === siteId) return this.active.site;
    const site = await this.detailResolver(siteId);
    if (!site) return null;
    if (this.active) this.deactivate();
    const defaultAsset = site.assets.find((a) => a.defaultVisible) ?? site.assets[0];
    const representation = defaultAsset?.representation ?? "gaussian-splat";
    this.active = { site, representation, temporalAssetId: null, handles: new Map() };
    this.events.emit("site-active", site.id);
    this.events.emit("representation", { siteId: site.id, representation });
    await this.showRepresentation(representation);
    return site;
  }

  deactivate(): void {
    if (!this.active) return;
    const { site, handles } = this.active;
    for (const handle of handles.values()) this.disposeHandle(handle);
    this.clipping.setFootprint(site.id, null);
    this.active = null;
    this.events.emit("site-active", null);
    this.events.emit("tilesets", []);
    this.scene.requestRender();
  }

  /** Flies to a site and loads it. The camera pose comes from the default bookmark or the model bounds. */
  async flyTo(siteId: string): Promise<void> {
    const summary = this.summaries.find((s) => s.id === siteId);
    const site = await this.activate(siteId);
    if (!site) {
      this.events.emit("toast", {
        tone: "error",
        title: "Site not found",
        body: "The catalog no longer has this site.",
      });
      return;
    }
    const bookmark = site.cameraBookmarks.find((b) => b.isDefault) ?? site.cameraBookmarks[0];
    this.flightTarget = site.id;
    const onComplete = () => {
      this.flightTarget = null;
      this.checkProximity(true);
    };
    if (bookmark) {
      this.camera.flyToBookmark(bookmark, { onComplete });
      return;
    }
    const sphere = await this.boundingSphere(site);
    if (sphere) {
      this.camera.flyToBoundingSphere(sphere, { onComplete });
      return;
    }
    const centroid = summary?.centroid ?? site.centroid;
    this.camera.flyTo(centroid.longitude, centroid.latitude, (centroid.height ?? 0) + 400, {
      pitch: -35,
      onComplete,
    });
  }

  /** Bounding sphere of the active representation (loaded tileset) or the footprint. */
  async boundingSphere(site: Site): Promise<BoundingSphere | null> {
    const handle = this.active?.site.id === site.id ? this.activeHandle() : null;
    const tileset = handle?.tileset ?? (handle?.loading ? await handle.loading : null);
    if (tileset) return tileset.boundingSphere;
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
    await this.showRepresentation(representation);
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
    await this.showRepresentation(asset.representation);
  }

  /** Called by the debug panel. */
  setDebug(options: { boundingVolumes?: boolean; wireframe?: boolean }): void {
    for (const handle of this.active?.handles.values() ?? []) {
      if (!handle.tileset) continue;
      if (options.boundingVolumes !== undefined)
        handle.tileset.debugShowBoundingVolume = options.boundingVolumes;
      if (options.wireframe !== undefined) handle.tileset.debugWireframe = options.wireframe;
    }
    this.scene.requestRender();
  }

  activeTilesetLabels(): string[] {
    const labels: string[] = [];
    for (const handle of this.active?.handles.values() ?? []) {
      if (handle.tileset?.show)
        labels.push(`${this.active?.site.name ?? "site"} · ${handle.asset.name}`);
    }
    return labels;
  }

  private activeHandle(): AssetHandle | null {
    if (!this.active) return null;
    const asset = this.pickAsset(this.active.representation);
    return asset ? (this.active.handles.get(asset.id) ?? null) : null;
  }

  private pickAsset(representation: Representation): SiteAsset | null {
    if (!this.active) return null;
    const candidates = this.active.site.assets.filter((a) => a.representation === representation);
    if (this.active.temporalAssetId) {
      const chosen = candidates.find((a) => a.id === this.active?.temporalAssetId);
      if (chosen) return chosen;
    }
    return (
      candidates.find((a) => a.defaultVisible) ??
      candidates.sort((a, b) => (b.observedAt ?? "").localeCompare(a.observedAt ?? ""))[0] ??
      null
    );
  }

  private async showRepresentation(representation: Representation): Promise<void> {
    const active = this.active;
    if (!active) return;
    const asset = this.pickAsset(representation);
    for (const handle of active.handles.values()) {
      if (handle.tileset && handle.asset.id !== asset?.id) handle.tileset.show = false;
    }
    if (!asset) {
      this.clipping.setFootprint(active.site.id, null);
      this.events.emit("tilesets", this.activeTilesetLabels());
      return;
    }
    const tileset = await this.ensureTileset(active, asset);
    if (
      !tileset ||
      this.active !== active ||
      this.pickAsset(active.representation)?.id !== asset.id
    )
      return;
    tileset.show = true;
    let footprint: Footprint | null = null;
    if (asset.renderConfig.clipsWorld) {
      footprint =
        asset.renderConfig.clipFootprint === "tileset"
          ? (footprintFromTileset(tileset) ?? asset.footprint ?? active.site.boundary)
          : (asset.footprint ?? active.site.boundary);
    }
    this.clipping.setFootprint(active.site.id, footprint);
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
    handle = { asset, tileset: null, loading: null, unsubscribe: [] };
    active.handles.set(asset.id, handle);
    this.events.emit("asset", { id: asset.id, patch: { loadState: "loading", error: null } });
    handle.loading = timed(
      "site.asset.load",
      () => createSiteTileset(asset, { maximumScreenSpaceError: this.screenSpaceError }),
      {
        asset: asset.id,
        representation: asset.representation,
      },
    )
      .then((tileset) => {
        if (this.active !== active) {
          tileset.destroy();
          return null;
        }
        this.scene.primitives.add(tileset);
        this.attachTileset(handle, tileset, asset);
        return tileset;
      })
      .catch((error: unknown) => {
        const message = isIonAuthError(error)
          ? "Cesium ion rejected the asset request. Provide VITE_CESIUM_ION_ACCESS_TOKEN with access to this asset."
          : isIonNotFound(error)
            ? "Asset not found on Cesium ion (check the asset ID and that your token can read it)."
            : describeError(error);
        log.warn("asset failed", { asset: asset.id, error: message });
        this.events.emit("asset", { id: asset.id, patch: { loadState: "error", error: message } });
        this.events.emit("toast", {
          tone: "error",
          title: `${asset.name} failed to load`,
          body: message,
          id: `asset-${asset.id}`,
        });
        if (isIonAuthError(error)) this.events.emit("token", "invalid");
        return null;
      })
      .finally(() => {
        if (handle) handle.loading = null;
      });
    return handle.loading;
  }

  private attachTileset(handle: AssetHandle, tileset: Cesium3DTileset, asset: SiteAsset): void {
    handle.tileset = tileset;
    const offset = asset.renderConfig.heightOffsetM ?? 0;
    if (offset !== 0) {
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
    handle.unsubscribe.push(
      tileset.loadProgress.addEventListener((pending: number, processing: number) => {
        this.performance.reportLoading(pending, processing);
        this.events.emit("asset", {
          id: asset.id,
          patch: {
            progress: { pending, processing },
            memoryMb: Math.round(tileset.totalMemoryUsageInBytes / 1048576),
            screenSpaceError: tileset.maximumScreenSpaceError,
          },
        });
      }),
      tileset.tileFailed.addEventListener((detail: { message?: string; url?: string }) => {
        log.warn("tile failed", { asset: asset.id, message: detail.message, url: detail.url });
      }),
    );
  }

  /** Memory held by the visible site tilesets against their configured cache budget. */
  private memoryUsage(): { bytes: number; budget: number } {
    const { cacheBytes, maximumCacheOverflowBytes } = tileCacheBudget();
    let bytes = 0;
    for (const handle of this.active?.handles.values() ?? []) {
      if (handle.tileset?.show) bytes += handle.tileset.totalMemoryUsageInBytes;
    }
    return { bytes, budget: cacheBytes + maximumCacheOverflowBytes };
  }

  private applyScreenSpaceError(sse: number): void {
    this.screenSpaceError = sse;
    for (const handle of this.active?.handles.values() ?? []) {
      if (!handle.tileset) continue;
      const configured = handle.asset.renderConfig.maximumScreenSpaceError;
      // A per-asset value acts as a floor for quality (never coarser than configured), while
      // splats have a hard floor on refinement because of their per-frame CPU sort.
      let next = configured ? Math.min(configured, sse) : sse;
      if (handle.asset.representation === "gaussian-splat")
        next = Math.max(next, SPLAT_MIN_SCREEN_SPACE_ERROR);
      handle.tileset.maximumScreenSpaceError = next;
      this.events.emit("asset", {
        id: handle.asset.id,
        patch: { screenSpaceError: handle.tileset.maximumScreenSpaceError },
      });
    }
  }

  /** Loads nearby sites automatically and unloads far ones; also tracks the "near site" for the HUD. */
  private checkProximity(force = false): void {
    const now = performance.now();
    if (!force && now - this.lastProximityCheck < 400) return;
    this.lastProximityCheck = now;
    const pose = this.camera.pose();
    const here = { longitude: pose.longitude, latitude: pose.latitude };
    let nearest: { summary: SiteSummary; distance: number } | null = null;
    for (const summary of this.summaries) {
      const distance = haversineDistance(here, summary.centroid);
      if (!nearest || distance < nearest.distance) nearest = { summary, distance };
    }
    const active = this.active;
    if (active) {
      const activeSummary = this.summaries.find((s) => s.id === active.site.id);
      const distance = activeSummary ? haversineDistance(here, activeSummary.centroid) : 0;
      if (
        distance > DEACTIVATE_DISTANCE_M &&
        pose.altitude > DEACTIVATE_DISTANCE_M / 4 &&
        this.flightTarget !== active.site.id
      ) {
        log.info("unloading distant site", { site: active.site.slug });
        this.deactivate();
      }
    }
    if (
      nearest &&
      nearest.distance < ACTIVATE_DISTANCE_M &&
      pose.altitude < ACTIVATE_DISTANCE_M &&
      this.active?.site.id !== nearest.summary.id
    ) {
      void this.activate(nearest.summary.id);
    }
    const radius = nearest ? Math.max(Math.sqrt(nearest.summary.areaM2 / Math.PI) * 6, 400) : 0;
    const near =
      nearest && nearest.distance < radius && pose.altitude < NEAR_ALTITUDE_M
        ? nearest.summary.id
        : null;
    if (near !== this.nearId) {
      this.nearId = near;
      this.events.emit("site-near", near);
    }
    this.performance.reportContext(pose.altitude, near !== null);
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
    this.deactivate();
  }
}

/**
 * Derives a lon/lat footprint from a loaded tileset's root bounding volume:
 * a region's rectangle, an oriented box's horizontal corners, or a sphere's circle.
 */
export function footprintFromTileset(tileset: Cesium3DTileset): Footprint | null {
  // `Cesium3DTile.boundingVolume` (a TileBoundingVolume) is not in the public typings but is
  // stable at runtime; fall back to the public bounding sphere when it is absent.
  const volume = (tileset.root as unknown as { boundingVolume?: TileVolume }).boundingVolume ?? {
    boundingVolume: {
      center: tileset.root.boundingSphere.center,
      radius: tileset.root.boundingSphere.radius,
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
