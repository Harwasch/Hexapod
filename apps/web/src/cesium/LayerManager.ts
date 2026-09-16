import {
  Cesium3DTileset,
  EllipsoidTerrainProvider,
  ImageryLayer,
  MVTDataProvider,
  SplitDirection,
  UrlTemplateImageryProvider,
  type DataSource,
  type ImageryProvider,
  type Scene,
  type Terrain,
  type Viewer,
} from "cesium";

import type { Layer } from "@twin/contracts";

import type { Emitter } from "@/lib/emitter";
import { createLogger, describeError } from "@/lib/log";
import { timed } from "@/lib/timing";

import type { ClippingManager } from "./ClippingManager";
import type { PerformanceManager } from "./PerformanceManager";
import { isIonAuthError } from "./ion";
import {
  createImageryProvider,
  createNaturalEarthProvider,
  isImagerySource,
} from "./providers/imagery";
import { resolveStacLayer } from "./providers/stac";
import { createTerrain, isTerrainSource } from "./providers/terrain";
import { createLayerTileset, is3DSource, tileCacheBudget } from "./providers/tiles";
import { createDataSource, isVectorSource } from "./providers/vector";
import type { SceneEvents } from "./types";

const log = createLogger("layers");
/** The world mesh renders at half the sites' screen-space error (8 CSS px on balanced). */
const WORLD_SSE_FRACTION = 0.5;
/** Idle refinement never asks the world mesh for finer than this many device pixels of error. */
const WORLD_MIN_DEVICE_PX = 2;

type Handle =
  | { kind: "imagery"; layer: ImageryLayer }
  | { kind: "terrain"; terrain: Terrain }
  | { kind: "tileset"; tileset: Cesium3DTileset }
  | { kind: "mvt"; provider: MVTDataProvider }
  | { kind: "datasource"; dataSource: DataSource };

interface Entry {
  layer: Layer;
  handle: Handle | null;
  visible: boolean;
  opacity: number;
  loading: Promise<void> | null;
  generation: number;
  abort: AbortController | null;
}

/**
 * Turns catalog layers into Cesium primitives, imagery layers, terrain and
 * data sources. React never touches Cesium objects; it calls this manager.
 */
export class LayerManager {
  private readonly scene: Scene;
  private readonly entries = new Map<string, Entry>();
  private fallbackBasemap: ImageryLayer | null = null;
  private worldTilesetId: string | null = null;
  private worldSse = 16;
  private worldPixelRatio = 1;
  private performance: PerformanceManager | null = null;

  /** Lets the world tileset report its loading and memory to the adaptive quality policy. */
  bindPerformance(performance: PerformanceManager): void {
    this.performance = performance;
    performance.addMemorySource(() => {
      const tileset = this.worldTileset;
      if (!tileset) return { bytes: 0, budget: 1 };
      const { cacheBytes, maximumCacheOverflowBytes } = tileCacheBudget();
      return {
        bytes: tileset.totalMemoryUsageInBytes,
        budget: cacheBytes + maximumCacheOverflowBytes,
      };
    });
  }
  private splitLeft: string | null = null;
  private splitRight: string | null = null;
  private generation = 0;

  constructor(
    private readonly viewer: Viewer,
    private readonly events: Emitter<SceneEvents>,
    private readonly clipping: ClippingManager,
  ) {
    this.scene = viewer.scene;
  }

  /** Registers catalog layers (idempotent; keeps runtime state for known ids). */
  register(layers: Layer[]): void {
    const seen = new Set<string>();
    for (const layer of layers) {
      seen.add(layer.id);
      const existing = this.entries.get(layer.id);
      if (existing) {
        existing.layer = layer;
        continue;
      }
      this.entries.set(layer.id, {
        layer,
        handle: null,
        visible: false,
        opacity: layer.render.opacity ?? 1,
        loading: null,
        generation: 0,
        abort: null,
      });
      this.events.emit("layer", {
        id: layer.id,
        patch: {
          visible: false,
          opacity: layer.render.opacity ?? 1,
          loadState: "idle",
          error: null,
        },
      });
    }
    for (const id of Array.from(this.entries.keys())) {
      if (!seen.has(id)) this.unregister(id);
    }
  }

  unregister(id: string): void {
    const entry = this.entries.get(id);
    if (!entry) return;
    this.dispose(entry);
    this.entries.delete(id);
  }

  get(id: string): Layer | undefined {
    return this.entries.get(id)?.layer;
  }

  layerIdForDataSource(dataSource: DataSource): string | null {
    for (const [id, entry] of this.entries) {
      if (entry.handle?.kind === "datasource" && entry.handle.dataSource === dataSource) return id;
    }
    return null;
  }

  layerIdForPrimitive(primitive: unknown): string | null {
    for (const [id, entry] of this.entries) {
      const handle = entry.handle;
      if (!handle) continue;
      if (handle.kind === "tileset" && handle.tileset === primitive) return id;
      if (handle.kind === "mvt" && handle.provider.tileset === primitive) return id;
    }
    return null;
  }

  /** Shows or hides a layer, loading it on first use. Exclusive groups switch off siblings. */
  async setVisible(id: string, visible: boolean): Promise<void> {
    const entry = this.entries.get(id);
    if (!entry) return;
    if (visible) {
      const group = entry.layer.render.exclusiveGroup;
      if (group) {
        for (const [otherId, other] of this.entries) {
          if (otherId !== id && other.visible && other.layer.render.exclusiveGroup === group) {
            await this.setVisible(otherId, false);
          }
        }
      }
    }
    entry.visible = visible;
    this.events.emit("layer", { id, patch: { visible } });
    if (visible) {
      await this.ensureLoaded(entry);
      this.applyVisibility(entry);
    } else {
      this.applyVisibility(entry);
      if (entry.layer.source.type === "cesium-ion-terrain") this.applyTerrainOff();
    }
    this.emitTilesets();
  }

  setOpacity(id: string, opacity: number): void {
    const entry = this.entries.get(id);
    if (!entry) return;
    entry.opacity = opacity;
    this.events.emit("layer", { id, patch: { opacity } });
    const handle = entry.handle;
    if (handle?.kind === "imagery") handle.layer.alpha = opacity;
    if (handle?.kind === "tileset") handle.tileset.style = undefined;
    this.scene.requestRender();
  }

  /** Compare mode: constrain two layers to the two halves of the screen. */
  setSplit(leftId: string | null, rightId: string | null): void {
    this.splitLeft = leftId;
    this.splitRight = rightId;
    for (const [id, entry] of this.entries) {
      const direction =
        id === leftId
          ? SplitDirection.LEFT
          : id === rightId
            ? SplitDirection.RIGHT
            : SplitDirection.NONE;
      const handle = entry.handle;
      if (handle?.kind === "imagery") handle.layer.splitDirection = direction;
      if (handle?.kind === "tileset") handle.tileset.splitDirection = direction;
    }
    this.scene.requestRender();
  }

  setSplitPosition(fraction: number): void {
    this.scene.splitPosition = Math.min(1, Math.max(0, fraction));
    this.scene.requestRender();
  }

  get split(): { left: string | null; right: string | null } {
    return { left: this.splitLeft, right: this.splitRight };
  }

  /**
   * Detail of the global mesh follows the adaptive screen-space error, at half the site
   * value (Google's tiles are coarse at Cesium's default of 16) and in device pixels. The
   * ladder's tile steps and memory pressure reach it through the same sink.
   */
  applyWorldScreenSpaceError(siteSse: number, pixelRatio: number): void {
    this.worldSse = siteSse;
    this.worldPixelRatio = pixelRatio;
    const tileset = this.worldTileset;
    if (!tileset) return;
    const next = Math.max(
      WORLD_MIN_DEVICE_PX,
      Math.round(((siteSse * WORLD_SSE_FRACTION) / pixelRatio) * 4) / 4,
    );
    if (tileset.maximumScreenSpaceError === next) return;
    tileset.maximumScreenSpaceError = next;
    this.scene.requestRender();
  }

  /** Google Photorealistic tileset when loaded (drives the world mode + clipping). */
  get worldTileset(): Cesium3DTileset | null {
    const id = this.worldTilesetId;
    const handle = id ? this.entries.get(id)?.handle : null;
    return handle?.kind === "tileset" ? handle.tileset : null;
  }

  activeTilesetLabels(): string[] {
    const labels: string[] = [];
    for (const entry of this.entries.values()) {
      if (entry.visible && (entry.handle?.kind === "tileset" || entry.handle?.kind === "mvt"))
        labels.push(entry.layer.name);
    }
    return labels;
  }

  /** Ensures at least one basemap is on screen, even with no ion access. */
  async ensureFallbackBasemap(): Promise<void> {
    const anyImagery = Array.from(this.entries.values()).some(
      (e) => e.visible && e.handle?.kind === "imagery",
    );
    if (anyImagery || this.fallbackBasemap) return;
    try {
      const provider = await createNaturalEarthProvider();
      this.fallbackBasemap = this.viewer.imageryLayers.addImageryProvider(provider, 0);
      log.info("using Natural Earth II fallback basemap");
    } catch (error) {
      log.error("fallback basemap failed", { error: describeError(error) });
    }
  }

  private removeFallbackBasemap(): void {
    if (this.fallbackBasemap) {
      this.viewer.imageryLayers.remove(this.fallbackBasemap, true);
      this.fallbackBasemap = null;
    }
  }

  private async ensureLoaded(entry: Entry): Promise<void> {
    if (entry.handle) return;
    if (entry.loading) return entry.loading;
    const id = entry.layer.id;
    const generation = ++this.generation;
    entry.generation = generation;
    entry.abort = new AbortController();
    this.events.emit("layer", { id, patch: { loadState: "loading", error: null } });
    entry.loading = timed("layer.load", () => this.load(entry, entry.abort?.signal), { id })
      .then((handle) => {
        if (entry.generation !== generation || !this.entries.has(id)) {
          this.disposeHandle(handle);
          return;
        }
        entry.handle = handle;
        this.events.emit("layer", { id, patch: { loadState: "ready", error: null } });
        if (handle.kind === "imagery") this.removeFallbackBasemap();
      })
      .catch((error: unknown) => {
        const message = isIonAuthError(error)
          ? "Cesium ion rejected the request. Check VITE_CESIUM_ION_ACCESS_TOKEN and the asset's access."
          : describeError(error);
        log.warn("layer failed", { id, error: message });
        entry.visible = false;
        this.events.emit("layer", {
          id,
          patch: { loadState: "error", error: message, visible: false },
        });
        this.events.emit("toast", {
          tone: "warning",
          title: `${entry.layer.name} could not load`,
          body: message,
        });
        if (isIonAuthError(error)) this.events.emit("token", "invalid");
        if (isImagerySource(entry.layer.source)) void this.ensureFallbackBasemap();
      })
      .finally(() => {
        entry.loading = null;
        entry.abort = null;
      });
    return entry.loading;
  }

  private async load(entry: Entry, signal?: AbortSignal): Promise<Handle> {
    const layer = entry.layer;
    const source = layer.source;
    if (isTerrainSource(source)) {
      const terrain = createTerrain(layer);
      await new Promise<void>((resolve, reject) => {
        terrain.readyEvent.addEventListener(() => resolve());
        terrain.errorEvent.addEventListener((error) =>
          reject(error instanceof Error ? error : new Error(String(error))),
        );
        this.scene.setTerrain(terrain);
      });
      return { kind: "terrain", terrain };
    }
    if (isImagerySource(source)) {
      const provider = await createImageryProvider(layer);
      return { kind: "imagery", layer: this.addImagery(provider, entry) };
    }
    if (isVectorSource(source)) {
      const dataSource = await createDataSource(layer);
      dataSource.show = false;
      await this.viewer.dataSources.add(dataSource);
      return { kind: "datasource", dataSource };
    }
    if (is3DSource(source)) {
      const created = await createLayerTileset(layer);
      if (created instanceof MVTDataProvider) {
        this.scene.primitives.add(created);
        return { kind: "mvt", provider: created };
      }
      this.scene.primitives.add(created);
      if (source.type === "google-photorealistic") {
        this.worldTilesetId = layer.id;
        this.clipping.setWorldTileset(created);
        this.applyWorldScreenSpaceError(this.worldSse, this.worldPixelRatio);
        // The idle refinement waits for the world's tiles too, and its memory counts.
        created.loadProgress.addEventListener((pending: number, processing: number) =>
          this.performance?.reportLoading("world", pending, processing),
        );
      }
      return { kind: "tileset", tileset: created };
    }
    if (source.type === "stac") {
      const resolved = await resolveStacLayer(layer, signal);
      switch (resolved.kind) {
        case "3d-tiles": {
          const tileset = await Cesium3DTileset.fromUrl(resolved.url, { show: false });
          this.scene.primitives.add(tileset);
          return { kind: "tileset", tileset };
        }
        case "xyz":
          return {
            kind: "imagery",
            layer: this.addImagery(
              new UrlTemplateImageryProvider({ url: resolved.urlTemplate }),
              entry,
            ),
          };
        case "imagery":
          return { kind: "imagery", layer: this.addImagery(resolved.provider, entry) };
        case "unsupported":
          throw new Error(resolved.reason);
      }
    }
    throw new Error(`unsupported layer source ${source.type}`);
  }

  private addImagery(provider: ImageryProvider, entry: Entry): ImageryLayer {
    const isBasemap = entry.layer.render.exclusiveGroup === "basemap";
    const layer = new ImageryLayer(provider, { alpha: entry.opacity, show: false });
    if (isBasemap) this.viewer.imageryLayers.add(layer, this.fallbackBasemap ? 1 : 0);
    else this.viewer.imageryLayers.add(layer);
    return layer;
  }

  private applyVisibility(entry: Entry): void {
    const handle = entry.handle;
    if (!handle) return;
    switch (handle.kind) {
      case "imagery":
        handle.layer.show = entry.visible;
        handle.layer.alpha = entry.opacity;
        break;
      case "terrain":
        if (entry.visible) this.scene.setTerrain(handle.terrain);
        break;
      case "tileset":
        handle.tileset.show = entry.visible;
        if (entry.layer.source.type === "google-photorealistic") this.applyWorldMode(entry.visible);
        break;
      case "mvt":
        handle.provider.show = entry.visible;
        break;
      case "datasource":
        handle.dataSource.show = entry.visible;
        break;
    }
    this.scene.requestRender();
  }

  private applyTerrainOff(): void {
    const anyTerrain = Array.from(this.entries.values()).some(
      (e) => e.visible && e.handle?.kind === "terrain",
    );
    if (!anyTerrain) this.scene.terrainProvider = new EllipsoidTerrainProvider();
  }

  /** Photorealistic world: the global mesh replaces the globe surface (it includes terrain). */
  private applyWorldMode(photorealistic: boolean): void {
    this.clipping.setWorldMode(photorealistic);
    this.events.emit("world", photorealistic ? "Google Photorealistic 3D Tiles" : "Open world");
  }

  private emitTilesets(): void {
    this.events.emit("tilesets", this.activeTilesetLabels());
  }

  private dispose(entry: Entry): void {
    entry.abort?.abort();
    entry.generation = -1;
    if (entry.handle) this.disposeHandle(entry.handle);
    entry.handle = null;
  }

  private disposeHandle(handle: Handle): void {
    switch (handle.kind) {
      case "imagery":
        this.viewer.imageryLayers.remove(handle.layer, true);
        break;
      case "terrain":
        this.scene.terrainProvider = new EllipsoidTerrainProvider();
        break;
      case "tileset":
        if (this.clipping && this.worldTileset === handle.tileset) {
          this.clipping.setWorldTileset(null);
          this.worldTilesetId = null;
          this.applyWorldMode(false);
        }
        this.scene.primitives.remove(handle.tileset);
        break;
      case "mvt":
        this.scene.primitives.remove(handle.provider);
        break;
      case "datasource":
        void this.viewer.dataSources.remove(handle.dataSource, true);
        break;
    }
  }

  destroy(): void {
    for (const entry of this.entries.values()) this.dispose(entry);
    this.entries.clear();
  }
}
