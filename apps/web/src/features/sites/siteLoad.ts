import type { Representation, SiteAsset } from "@twin/contracts";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import type { AssetRuntime, SiteLoad as SiteLoadRecord } from "@/state/sites";

/**
 * Load feedback for the site's 3D model, from two sources in the store:
 *
 * - the scene's per-site record (`siteLoads`, written by `SiteManager` as a fly-to fetches
 *   the site's details and creates its model during the flight), which says what a first
 *   visit is waiting on and offers the scene's own retry (`retrySiteLoad(siteId)`); and
 * - the per-asset runtime (`AssetRuntime.loadState`, `.error`, `.progress`), which still
 *   covers what the site record does not: a model streaming after the site counts as
 *   ready, and switching Splat / Mesh / Points at a site already loaded.
 *
 * The site record wins while it is not ready; after that the asset's own state speaks.
 */
export type SiteLoad =
  { phase: "loading"; percent: number | null } | { phase: "error"; message: string } | null;

/** What the site-level record says, or null once it is ready (or absent). */
export function fromSiteRecord(record: SiteLoadRecord | undefined): SiteLoad {
  if (!record || record.phase === "ready") return null;
  if (record.phase === "error")
    return { phase: "error", message: record.error ?? "The site could not be loaded." };
  const percent = Math.min(99, Math.max(0, Math.round(record.progress * 100)));
  return { phase: "loading", percent: percent > 0 ? percent : null };
}

/** The asset the scene shows for a representation, chosen the way `SiteManager.pickAsset` does. */
export function shownAsset(
  assets: readonly SiteAsset[],
  representation: Representation | undefined,
  temporalAssetId: string | undefined,
): SiteAsset | null {
  if (!representation) return null;
  const candidates = assets.filter((a) => a.representation === representation);
  return (
    candidates.find((a) => a.id === temporalAssetId) ??
    candidates.find((a) => a.defaultVisible) ??
    [...candidates].sort((a, b) => (b.observedAt ?? "").localeCompare(a.observedAt ?? ""))[0] ??
    null
  );
}

/** Tiles still on their way: requested, or arrived and being processed. */
export function outstanding(runtime: Pick<AssetRuntime, "progress">): number {
  return runtime.progress.pending + runtime.progress.processing;
}

/**
 * Where the model's first load stands.
 *
 * Cesium reports tiles outstanding, never a total, so progress is measured against `peak`, the
 * most seen outstanding since the load began: a percentage that only rises once the tile
 * requests stop growing. `settled` is true once the first load has drained; after that,
 * tiles streaming in as the camera moves are not news.
 */
export function siteLoad(
  runtime: AssetRuntime | undefined,
  peak: number,
  settled: boolean,
): SiteLoad {
  if (!runtime) return null;
  if (runtime.loadState === "error")
    return { phase: "error", message: runtime.error ?? "The model could not be loaded." };
  if (runtime.loadState === "loading") return { phase: "loading", percent: null };
  if (runtime.loadState !== "ready" || settled) return null;
  const left = outstanding(runtime);
  if (left === 0) return null;
  const done = peak > 0 ? 1 - left / peak : 0;
  return { phase: "loading", percent: Math.min(99, Math.max(0, Math.round(done * 100))) };
}

/**
 * Asks the scene for the failed model again, leaving the camera and the rest of the site as
 * they are. `SiteManager` creates a tileset for any asset it is asked to show that has none,
 * and a failed load leaves none, so showing the same asset again is the retry.
 */
export async function retrySiteLoad(
  scene: CesiumSceneManager | null,
  assetId: string,
): Promise<void> {
  await scene?.sites.setTemporalAsset(assetId);
}
