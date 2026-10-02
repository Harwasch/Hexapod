import {
  Cesium3DTileset,
  MVTDataProvider,
  createGooglePhotorealistic3DTileset,
  type Cesium3DTileset as TilesetType,
} from "cesium";

import type { Layer, SiteAsset } from "@twin/contracts";

import { deviceSplatBudget, isHandheld } from "@/lib/detail";
import { tileUrl } from "@/lib/tileProxy";

import { incrementalSplats, keepOffscreenSplats } from "../splatInternals";

/** The incremental splat texture starts at this many times the device's splat budget: room
 *  for a batch's arrivals beside what they replace, and for freed ranges not yet reused. */
const INCREMENTAL_HEADROOM = 1.6;

export interface TilesetQuality {
  maximumScreenSpaceError: number;
}

const MB = 1024 * 1024;

interface DeviceHints {
  /** `navigator.deviceMemory` (GB): Chromium only, and capped at 8 there. */
  deviceMemory?: number | undefined;
  handheld: boolean;
  /** `navigator.connection.saveData`: the user asked for less data. */
  saveData: boolean;
}

function deviceHints(): DeviceHints {
  const nav =
    typeof navigator === "undefined"
      ? undefined
      : (navigator as Navigator & {
          deviceMemory?: number;
          connection?: { saveData?: boolean };
        });
  return {
    deviceMemory: nav?.deviceMemory,
    handheld: isHandheld(),
    saveData: nav?.connection?.saveData === true,
  };
}

/**
 * The device's memory in GB as the budgets read it. Safari and Firefox report none: a phone or
 * tablet that does not say is taken as the smallest (an iPhone's tab is killed long before its
 * RAM is used), a desktop that does not say as 8 GB, as the splat budget takes it
 * (lib/detail.ts) -- every Mac sold for years has at least that, and the 4 GB it used to be
 * taken as held a Mac to a phone's cache.
 */
export function deviceMemoryGb(hints: DeviceHints = deviceHints()): number {
  return hints.deviceMemory ?? (hints.handheld ? 2 : 8);
}

/** A phone or tablet, or a user who asked to save data: held to the smaller budgets. */
export function constrainedDevice(hints: DeviceHints = deviceHints()): boolean {
  return hints.handheld || hints.saveData;
}

/**
 * Tile cache budget for one tileset, sized to the device. `navigator.deviceMemory` is coarse,
 * so this errs on the small side: a tileset that overruns its budget gets its screen-space
 * error raised by the PerformanceManager rather than filling the GPU. Several tilesets share
 * one device total besides (`totalTileCacheBudget`).
 */
export function tileCacheBudget(hints: DeviceHints = deviceHints()): {
  cacheBytes: number;
  maximumCacheOverflowBytes: number;
} {
  if (constrainedDevice(hints)) {
    return { cacheBytes: 192 * MB, maximumCacheOverflowBytes: 96 * MB };
  }
  const memory = deviceMemoryGb(hints);
  if (memory >= 16) return { cacheBytes: 512 * MB, maximumCacheOverflowBytes: 256 * MB };
  if (memory >= 8) return { cacheBytes: 384 * MB, maximumCacheOverflowBytes: 192 * MB };
  return { cacheBytes: 256 * MB, maximumCacheOverflowBytes: 128 * MB };
}

/**
 * The tile cache every mesh, point-cloud and world tileset shares, on top of each one's own
 * budget. CesiumJS gives each tileset its own cache: with the Google world, a site's mesh and
 * its point cloud preloading hidden, a phone held three budgets -- over a gigabyte of tiles, in
 * a tab iOS kills at about one and a half. Splat tilesets are budgeted by what they draw
 * (SiteManager's splat cache) and stay out of it.
 */
export function totalTileCacheBudget(hints: DeviceHints = deviceHints()): {
  cacheBytes: number;
  maximumCacheOverflowBytes: number;
} {
  const total = (cache: number) => ({
    cacheBytes: cache * MB,
    maximumCacheOverflowBytes: (cache * MB) / 2,
  });
  if (constrainedDevice(hints)) return total(256);
  const memory = deviceMemoryGb(hints);
  if (memory >= 16) return total(1024);
  if (memory >= 8) return total(768);
  return total(384);
}

/** No shared tileset is left less cache than this: one view's tiles. */
const MIN_SHARE_BYTES = 64 * MB;

/**
 * Each tileset's share of the device total, `count` sharing it: an equal part, never more
 * than one tileset's own budget, never less than one view's tiles (`MIN_SHARE_BYTES`).
 */
export function tileCacheShare(
  count: number,
  hints: DeviceHints = deviceHints(),
): { cacheBytes: number; maximumCacheOverflowBytes: number } {
  const own = tileCacheBudget(hints).cacheBytes;
  const total = totalTileCacheBudget(hints).cacheBytes;
  const cacheBytes = Math.round(
    Math.min(own, Math.max(MIN_SHARE_BYTES, total / Math.max(1, count))),
  );
  return { cacheBytes, maximumCacheOverflowBytes: Math.round(cacheBytes / 2) };
}

/** The tilesets sharing the device total, live. */
const sharing = new Set<TilesetType>();

function reshare(): void {
  for (const tileset of [...sharing]) if (tileset.isDestroyed()) sharing.delete(tileset);
  const share = tileCacheShare(sharing.size);
  for (const tileset of sharing) {
    tileset.cacheBytes = share.cacheBytes;
    tileset.maximumCacheOverflowBytes = share.maximumCacheOverflowBytes;
  }
}

/**
 * Puts `tileset` in the device total (`totalTileCacheBudget`): every tileset in it gets an
 * equal share again, now and when one is destroyed. CesiumJS has no event for that, so the
 * tileset's `destroy` is wrapped on the instance.
 */
export function shareTileCache(tileset: TilesetType): void {
  if (sharing.has(tileset)) return;
  sharing.add(tileset);
  const destroy = tileset.destroy.bind(tileset);
  tileset.destroy = () => {
    sharing.delete(tileset);
    reshare();
    return destroy();
  };
  reshare();
}

/** What the tilesets in the device total hold, against it: a memory source for the
 *  PerformanceManager, so the total coarsens the view once tiles a view needs exceed it. */
export function sharedTileCacheUsage(): { bytes: number; budget: number } {
  let bytes = 0;
  for (const tileset of sharing) {
    if (!tileset.isDestroyed()) bytes += tileset.totalMemoryUsageInBytes;
  }
  const { cacheBytes, maximumCacheOverflowBytes } = totalTileCacheBudget();
  return { bytes, budget: cacheBytes + maximumCacheOverflowBytes };
}

const COMMON: Cesium3DTileset.ConstructorOptions = {
  dynamicScreenSpaceError: true,
  foveatedScreenSpaceError: true,
  preloadFlightDestinations: true,
  skipLevelOfDetail: false,
  ...tileCacheBudget(),
};

/** Loads a site asset (splat, mesh or point cloud) as a 3D Tileset. */
export async function createSiteTileset(
  asset: SiteAsset,
  quality: TilesetQuality,
): Promise<TilesetType> {
  const options: Cesium3DTileset.ConstructorOptions = {
    ...COMMON,
    maximumScreenSpaceError:
      asset.renderConfig.maximumScreenSpaceError ?? quality.maximumScreenSpaceError,
    show: false,
    preloadWhenHidden: true,
    // Load the level the view needs instead of every level on the way (measured on the SF
    // mesh 160 m up: 265k triangles in the time it took 145k without). The near, deep part of
    // a pitched view is what waits longest otherwise. Splats and point clouds keep the
    // ordinary traversal.
    skipLevelOfDetail: asset.representation === "mesh",
    // Never enableCollision: Cesium then ray-casts every loaded tile's triangles on the CPU
    // every frame to find the height under the camera (measured 130 ms per frame on the
    // Google world, 400 ms on a drone mesh). The camera floor is kept by CameraController
    // with one depth sample when the camera comes to rest instead.
    enableCollision: false,
  };
  if (asset.representation === "point-cloud") {
    const shading = asset.renderConfig.pointCloudShading;
    options.pointCloudShading = {
      attenuation: shading?.attenuation ?? true,
      eyeDomeLighting: shading?.eyeDomeLighting ?? true,
      maximumAttenuation: shading?.maximumAttenuation ?? undefined,
    };
  }
  const tileset = await (asset.source.type === "cesium-ion"
    ? Cesium3DTileset.fromIonAssetId(asset.source.assetId, options)
    : Cesium3DTileset.fromUrl(await tileUrl(asset.source.url), options));
  // A splat is drawn from one snapshot of its selected tiles, held while the camera moves
  // (splatMotionGate.ts); out-of-view tiles stay in it, coarse, so turning shows no hole.
  // A tile uploads alone into its own slot of one texture, so refining costs the tiles
  // refined, not every splat drawn (incrementalSplats).
  if (asset.representation === "gaussian-splat") {
    keepOffscreenSplats(tileset);
    // Walking a scan moves the camera a tile's width in seconds: Cesium's default skips
    // requests for tiles smaller than 60 frames of travel, which is every fine tile nearby.
    tileset.cullRequestsWhileMoving = false;
    incrementalSplats(tileset, deviceSplatBudget() * INCREMENTAL_HEADROOM);
  } else {
    shareTileCache(tileset);
  }
  return tileset;
}

/** Loads a catalog 3D layer (ion tileset, tileset URL, MVT or Google Photorealistic). */
export async function createLayerTileset(layer: Layer): Promise<TilesetType | MVTDataProvider> {
  const source = layer.source;
  const options: Cesium3DTileset.ConstructorOptions = {
    ...COMMON,
    maximumScreenSpaceError: layer.render.maximumScreenSpaceError ?? 16,
    show: false,
  };
  // Every 3D layer shares the device total with the sites' meshes (`shareTileCache`).
  const shared = (tileset: TilesetType): TilesetType => {
    shareTileCache(tileset);
    return tileset;
  };
  switch (source.type) {
    case "cesium-ion-3d-tiles":
      return shared(await Cesium3DTileset.fromIonAssetId(source.assetId, options));
    case "3d-tiles-url":
      return shared(await Cesium3DTileset.fromUrl(await tileUrl(source.url), options));
    case "google-photorealistic":
      // The world tileset's detail is driven by the PerformanceManager (see LayerManager);
      // the floor under the camera comes from a depth sample at rest, never from
      // enableCollision (see createSiteTileset).
      return shared(
        await createGooglePhotorealistic3DTileset(
          { onlyUsingWithGoogleGeocoder: true },
          { ...options, show: false, enableCollision: false, skipLevelOfDetail: true },
        ),
      );
    case "mvt":
      return MVTDataProvider.fromUrl(source.urlTemplate, {
        minZoom: source.minZoom ?? 0,
        maxZoom: source.maxZoom ?? 14,
      });
    default:
      throw new Error(`unsupported 3D source ${source.type}`);
  }
}

export function is3DSource(source: Layer["source"]): boolean {
  return ["cesium-ion-3d-tiles", "3d-tiles-url", "google-photorealistic", "mvt"].includes(
    source.type,
  );
}
