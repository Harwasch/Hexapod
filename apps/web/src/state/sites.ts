import { create } from "zustand";

import type { Representation } from "@twin/contracts";

import type { LoadState } from "./layers";

export interface AssetRuntime {
  loadState: LoadState;
  error: string | null;
  progress: { pending: number; processing: number };
  screenSpaceError: number | null;
  memoryMb: number;
  boundingRadiusM: number | null;
}

interface SitesState {
  /** Site whose assets are loaded into the scene (activated by fly-to or proximity). */
  activeSiteId: string | null;
  /** Site the camera is currently inside/near (drives representation control visibility). */
  nearSiteId: string | null;
  representation: Record<string, Representation>;
  /** Selected temporal version (asset id) per site, when several exist. */
  temporalAsset: Record<string, string>;
  assets: Record<string, AssetRuntime>;
  setActiveSite: (id: string | null) => void;
  setNearSite: (id: string | null) => void;
  setRepresentation: (siteId: string, representation: Representation) => void;
  setTemporalAsset: (siteId: string, assetId: string) => void;
  updateAsset: (assetId: string, patch: Partial<AssetRuntime>) => void;
  clearAssets: (assetIds: string[]) => void;
}

export const defaultAssetRuntime: AssetRuntime = {
  loadState: "idle",
  error: null,
  progress: { pending: 0, processing: 0 },
  screenSpaceError: null,
  memoryMb: 0,
  boundingRadiusM: null,
};

export const useSites = create<SitesState>()((set) => ({
  activeSiteId: null,
  nearSiteId: null,
  representation: {},
  temporalAsset: {},
  assets: {},
  setActiveSite: (activeSiteId) => set({ activeSiteId }),
  setNearSite: (nearSiteId) => set({ nearSiteId }),
  setRepresentation: (siteId, representation) =>
    set((s) => ({ representation: { ...s.representation, [siteId]: representation } })),
  setTemporalAsset: (siteId, assetId) =>
    set((s) => ({ temporalAsset: { ...s.temporalAsset, [siteId]: assetId } })),
  updateAsset: (assetId, patch) =>
    set((s) => ({
      assets: {
        ...s.assets,
        [assetId]: { ...(s.assets[assetId] ?? defaultAssetRuntime), ...patch },
      },
    })),
  clearAssets: (assetIds) =>
    set((s) => ({
      assets: Object.fromEntries(
        Object.entries(s.assets).filter(([key]) => !assetIds.includes(key)),
      ),
    })),
}));
