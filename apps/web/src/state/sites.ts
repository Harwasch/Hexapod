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

/**
 * Where one site's load stands -- its catalog record, its model, the model's first tiles --
 * for the HUD's progress pill and the Retry beside an error. A fly-to leaves on the click and
 * loads during the flight (SiteManager.flyTo), so this is how the screen shows that the
 * destination is still arriving, and why, if it never does.
 *
 * - `details`: the site's record is being fetched from the catalog.
 * - `model`: the default representation's tileset is being created (tileset.json; ion's
 *   endpoint). `attempt` counts tries; a stall is cut off and retried (lib/timeout.ts).
 * - `streaming`: the model is in the scene and its first view's tiles are arriving.
 * - `ready`: those tiles are in. The record stays until the site is unloaded.
 * - `error`: `error` says what failed, in words for the screen; `retryable` is false when
 *   trying again cannot help (the site is gone; ion refused the key or has no such asset).
 */
export type SiteLoadPhase = "details" | "model" | "streaming" | "ready" | "error";

export interface SiteLoad {
  phase: SiteLoadPhase;
  /** 0..1, a rough share of the load done; it never goes backwards within one attempt. */
  progress: number;
  error: string | null;
  retryable: boolean;
  /** Which try at creating the model this is (1 for the first). */
  attempt: number;
  /** A fly-to asked for this site, so the camera is on its way there; false for a site
   *  loaded because the camera came near it. */
  flight: boolean;
  /** When this load (or its last retry) began, `Date.now()`. */
  startedAt: number;
}

interface SitesState {
  /** Site whose assets are loaded into the scene (activated by fly-to or proximity). */
  activeSiteId: string | null;
  /** Site the camera is currently inside/near (drives representation control visibility). */
  nearSiteId: string | null;
  /** Loaded site the camera still frames, from near or a few km out (keeps its controls up). */
  inViewSiteId: string | null;
  representation: Record<string, Representation>;
  /** Selected temporal version (asset id) per site, when several exist. */
  temporalAsset: Record<string, string>;
  assets: Record<string, AssetRuntime>;
  /** Per-site load state, by site id; absent once a site is unloaded. */
  siteLoads: Record<string, SiteLoad>;
  /**
   * The site the latest fly-to is taking the camera to, until the camera has left it
   * (SiteManager, `site-flight`); null otherwise. The load pill follows it before
   * `activeSiteId`: a site becomes active only once its record has arrived, so a record that
   * is slow, or never comes (a cold or failing API, no network), would otherwise be loading
   * and failing where nothing on screen speaks for it -- and its Retry out of reach.
   */
  flightSiteId: string | null;
  /**
   * Tries a failed site load again: the record if that is what failed (flying there again if
   * a fly-to asked for it), otherwise the model. Installed by the scene (SceneBridge); a no-op
   * until it is.
   */
  retrySiteLoad: (siteId: string) => void;
  setActiveSite: (id: string | null) => void;
  setNearSite: (id: string | null) => void;
  setInViewSite: (id: string | null) => void;
  setRepresentation: (siteId: string, representation: Representation) => void;
  setTemporalAsset: (siteId: string, assetId: string) => void;
  updateAsset: (assetId: string, patch: Partial<AssetRuntime>) => void;
  clearAssets: (assetIds: string[]) => void;
  /** The scene's view of one site's load, or null when the site is unloaded. */
  setSiteLoad: (siteId: string, load: SiteLoad | null) => void;
  setSiteLoadRetry: (retry: (siteId: string) => void) => void;
  setFlightSite: (siteId: string | null) => void;
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
  inViewSiteId: null,
  representation: {},
  temporalAsset: {},
  assets: {},
  siteLoads: {},
  flightSiteId: null,
  retrySiteLoad: () => undefined,
  setActiveSite: (activeSiteId) => set({ activeSiteId }),
  setNearSite: (nearSiteId) => set({ nearSiteId }),
  setInViewSite: (inViewSiteId) => set({ inViewSiteId }),
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
  setSiteLoad: (siteId, load) =>
    set((s) => {
      if (load) return { siteLoads: { ...s.siteLoads, [siteId]: load } };
      if (!(siteId in s.siteLoads)) return {};
      const { [siteId]: _gone, ...rest } = s.siteLoads;
      return { siteLoads: rest };
    }),
  setSiteLoadRetry: (retrySiteLoad) => set({ retrySiteLoad }),
  setFlightSite: (flightSiteId) => set({ flightSiteId }),
}));
