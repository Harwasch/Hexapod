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
  type Cesium3DTile,
  sampleTerrainMostDetailed,
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
import { createSiteTileset, tileCacheBudget } from "./providers/tiles";
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
  /** Resolves once a clamp-to-ground placement has been applied (or was not requested). */
  placed: Promise<void>;
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
    this.performance.resetBenchmark();
    this.events.emit("site-active", site.id);
    this.events.emit("representation", { siteId: site.id, representation });
    await this.showRepresentation(representation);
    return site;
  }

  deactivate(): void {
    if (!this.active) return;
    const { site, handles } = this.active;
    this.camera.setObjectScale(false);
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
    this.applyClip(active, asset, tileset);
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
    // Hand-sized models need a millimetre zoom floor and a close near plane.
    this.camera.setObjectScale(tileset.boundingSphere.radius < OBJECT_SCALE_RADIUS_M);
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
      tileset.initialTilesLoaded.addEventListener(() => this.camera.refreshPose()),
      tileset.allTilesLoaded.addEventListener(() => this.camera.refreshPose()),
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

  /**
   * Clips the coarse world under the model. With `clipFootprint: "tileset"` the clip follows
   * the tiles' real coverage; that is only known once the first branching sub-tileset has
   * loaded, so until then the root box is used and the clip is re-derived on tile loads.
   */
  /**
   * Rests a model's lowest point on the terrain under its centre. Downloaded objects and
   * phone scans carry no usable ellipsoid height, so the catalog stores where, and the
   * viewer works out how high once the terrain is known.
   */
  private async clampToGround(tileset: Cesium3DTileset, asset: SiteAsset): Promise<void> {
    const sphere = tileset.boundingSphere;
    const center = Cartographic.fromCartesian(sphere.center);
    const [sample] = await sampleTerrainMostDetailed(this.viewer.terrainProvider, [
      Cartographic.fromRadians(center.longitude, center.latitude),
    ]).catch(() => [undefined]);
    const ground = sample?.height;
    if (ground === undefined || !Number.isFinite(ground) || tileset.isDestroyed()) return;
    // Lowest point of the root bounding box when there is one (a sphere would float a flat
    // object by the difference between its radius and its half height).
    const bottom = lowestHeight(tileset) ?? center.height - sphere.radius;
    const lift = ground + (asset.renderConfig.heightOffsetM ?? 0) - bottom;
    const from = Cartesian3.fromRadians(center.longitude, center.latitude, center.height);
    const to = Cartesian3.fromRadians(center.longitude, center.latitude, center.height + lift);
    tileset.modelMatrix = Matrix4.fromTranslation(Cartesian3.subtract(to, from, new Cartesian3()));
    log.info("clamped model to ground", { asset: asset.id, ground: Math.round(ground), lift });
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
      // is cut, so buildings from OSM or Google do not poke through the model.
      const footprint = footprintFromTileset(tileset) ?? asset.footprint ?? active.site.boundary;
      this.clipping.setFootprint(active.site.id, footprint, { globe: false, world: true });
      return;
    }
    let footprint: Footprint | null = null;
    let provisional = false;
    if (asset.renderConfig.clipsWorld) {
      if (asset.renderConfig.clipFootprint === "tileset") {
        const coverage = coverageFromTileset(tileset);
        provisional = coverage === null;
        footprint =
          coverage ?? footprintFromTileset(tileset) ?? asset.footprint ?? active.site.boundary;
      } else {
        footprint = asset.footprint ?? active.site.boundary;
      }
    }
    this.clipping.setFootprint(active.site.id, footprint);
    if (asset.renderConfig.clipFootprint !== "tileset") return;
    const handle = active.handles.get(asset.id);
    if (!handle) return;
    let applied = provisional ? "" : coverageKey(footprint);
    let timer: ReturnType<typeof setTimeout> | null = null;
    // Sub-tilesets stream in over time; re-derive the coverage after each burst of tile loads
    // and swap the clip only when it actually changed (rebuilding it is not free).
    const off = tileset.tileLoad.addEventListener(() => {
      if (timer !== null) return;
      timer = setTimeout(() => {
        timer = null;
        if (this.active !== active || !tileset.show) return;
        const coverage = coverageFromTileset(tileset);
        if (!coverage) return;
        const key = coverageKey(coverage);
        if (key === applied) return;
        applied = key;
        this.clipping.setFootprint(active.site.id, coverage);
        this.scene.requestRender();
      }, COVERAGE_REFRESH_MS);
    });
    handle.unsubscribe.push(() => {
      off();
      if (timer !== null) clearTimeout(timer);
    });
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
        next = Math.max(next, this.performance.splatMinimumScreenSpaceError);
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
    const footprint = footprintFromTile(node);
    if (footprint?.type === "Polygon") polygons.push(footprint.coordinates as [number, number][][]);
  };
  visit(tile, 0);
  if (polygons.length === 0) return null;
  return { type: "MultiPolygon", coordinates: polygons };
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

/** Clipping polygons are rasterised into one texture; keep the count bounded. */
const MAX_COVERAGE_TILES = 96;
/** Models smaller than this radius get the millimetre zoom floor and near plane. */
const OBJECT_SCALE_RADIUS_M = 30;
const MAX_COVERAGE_DEPTH = 4;
/** How many branching levels below the first split the coverage follows (4^3 boxes at most). */
const COVERAGE_LEVELS = 3;
const COVERAGE_REFRESH_MS = 1000;

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
