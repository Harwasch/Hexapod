/**
 * Typed, validated access to Vite environment variables.
 * Everything here is public (inlined into the bundle). Secrets never live here.
 */

function optionalString(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed === "" ? undefined : trimmed;
}

function optionalAssetId(value: string | undefined): number | undefined {
  const trimmed = optionalString(value);
  if (!trimmed) return undefined;
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

function flag(value: string | undefined, fallback = false): boolean {
  const trimmed = optionalString(value)?.toLowerCase();
  if (!trimmed) return fallback;
  return trimmed === "true" || trimmed === "1" || trimmed === "yes";
}

export interface AppEnv {
  ionAccessToken: string | undefined;
  defaultSplatAssetId: number | undefined;
  defaultMeshAssetId: number | undefined;
  defaultPointCloudAssetId: number | undefined;
  apiBaseUrl: string;
  /**
   * Where `catalog.json` is published, for the offline catalog. Public by construction —
   * it is the bucket's public read URL, the same origin the tiles themselves come from.
   */
  offlineCatalogUrl: string | undefined;
  photorealisticEnabled: boolean;
  devToolsEnabled: boolean;
  isDev: boolean;
}

export function readEnv(source: ImportMetaEnv = import.meta.env): AppEnv {
  return {
    ionAccessToken: optionalString(source.VITE_CESIUM_ION_ACCESS_TOKEN),
    defaultSplatAssetId: optionalAssetId(source.VITE_DEFAULT_SPLAT_ASSET_ID),
    defaultMeshAssetId: optionalAssetId(source.VITE_DEFAULT_MESH_ASSET_ID),
    defaultPointCloudAssetId: optionalAssetId(source.VITE_DEFAULT_POINTCLOUD_ASSET_ID),
    apiBaseUrl: optionalString(source.VITE_API_BASE_URL) ?? "",
    offlineCatalogUrl: optionalString(source.VITE_OFFLINE_CATALOG_URL),
    // On unless switched off: the ion token in use must have access to Google's tiles.
    photorealisticEnabled: flag(source.VITE_ENABLE_PHOTOREALISTIC, true),
    devToolsEnabled: source.DEV || flag(source.VITE_ENABLE_DEV_TOOLS),
    isDev: source.DEV,
  };
}

export const env: AppEnv = readEnv();
