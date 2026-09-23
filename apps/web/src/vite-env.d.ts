/// <reference types="vite/client" />

import type { CesiumSceneManager } from "./cesium/CesiumSceneManager";

declare global {
  const CESIUM_BASE_URL: string;

  interface Window {
    /** Development-only handle to the scene manager for debugging and browser automation. */
    __twin?: CesiumSceneManager | undefined;
  }

  interface ImportMetaEnv {
    readonly VITE_CESIUM_ION_ACCESS_TOKEN?: string;
    readonly VITE_DEFAULT_SPLAT_ASSET_ID?: string;
    readonly VITE_DEFAULT_MESH_ASSET_ID?: string;
    readonly VITE_DEFAULT_POINTCLOUD_ASSET_ID?: string;
    readonly VITE_API_BASE_URL?: string;
    readonly VITE_OFFLINE_CATALOG_URL?: string;
    readonly VITE_ENABLE_PHOTOREALISTIC?: string;
    readonly VITE_ENABLE_DEV_TOOLS?: string;
  }
}

export {};
