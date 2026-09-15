import {
  Cesium3DTileset,
  MVTDataProvider,
  createGooglePhotorealistic3DTileset,
  type Cesium3DTileset as TilesetType,
} from "cesium";

import type { Layer, SiteAsset } from "@twin/contracts";

export interface TilesetQuality {
  maximumScreenSpaceError: number;
}

const MB = 1024 * 1024;

/**
 * Tile cache budget sized to the device. `navigator.deviceMemory` is coarse (and absent on
 * Firefox/Safari) so this errs on the small side: a splat that overruns the budget gets its
 * screen-space error raised by the PerformanceManager rather than filling the GPU.
 */
export function tileCacheBudget(): { cacheBytes: number; maximumCacheOverflowBytes: number } {
  const deviceMemoryGb = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 4;
  if (deviceMemoryGb >= 16) return { cacheBytes: 512 * MB, maximumCacheOverflowBytes: 256 * MB };
  if (deviceMemoryGb >= 8) return { cacheBytes: 384 * MB, maximumCacheOverflowBytes: 192 * MB };
  return { cacheBytes: 256 * MB, maximumCacheOverflowBytes: 128 * MB };
}

/**
 * Gaussian splats are sorted on the CPU every time the camera moves, so their cost scales with
 * the number of splats on screen far more steeply than a mesh does. Never refine them below this.
 */
export const SPLAT_MIN_SCREEN_SPACE_ERROR = 8;

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
  };
  if (asset.representation === "point-cloud") {
    const shading = asset.renderConfig.pointCloudShading;
    options.pointCloudShading = {
      attenuation: shading?.attenuation ?? true,
      eyeDomeLighting: shading?.eyeDomeLighting ?? true,
      maximumAttenuation: shading?.maximumAttenuation ?? undefined,
    };
  }
  return asset.source.type === "cesium-ion"
    ? Cesium3DTileset.fromIonAssetId(asset.source.assetId, options)
    : Cesium3DTileset.fromUrl(asset.source.url, options);
}

/** Loads a catalog 3D layer (ion tileset, tileset URL, MVT or Google Photorealistic). */
export async function createLayerTileset(layer: Layer): Promise<TilesetType | MVTDataProvider> {
  const source = layer.source;
  const options: Cesium3DTileset.ConstructorOptions = {
    ...COMMON,
    maximumScreenSpaceError: layer.render.maximumScreenSpaceError ?? 16,
    show: false,
  };
  switch (source.type) {
    case "cesium-ion-3d-tiles":
      return Cesium3DTileset.fromIonAssetId(source.assetId, options);
    case "3d-tiles-url":
      return Cesium3DTileset.fromUrl(source.url, options);
    case "google-photorealistic":
      return createGooglePhotorealistic3DTileset(
        { onlyUsingWithGoogleGeocoder: true },
        { ...options, show: false },
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
