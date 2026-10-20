import type { Epoch, Site } from "@twin/contracts";

import { currentAsset, epochOf, pickShownAsset, siteEpochs } from "@/lib/epochs";

type SiteAsset = Site["assets"][number];

/** The coverage overlay of the scan view.html shows: where it is, and how much of the keep
 *  tier held-out frames verified (absent when accuracy was not measured). */
export interface ShownCoverage {
  url: string;
  verifiedPct?: number;
}

/** The splat view.html shows, the dated capture (epoch) it is, and the coverage overlay that
 *  belongs to it (or none). */
export interface ShownScan {
  asset: SiteAsset | undefined;
  epoch: Epoch | null;
  coverage: ShownCoverage | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Which splat view.html shows, and whether the site's coverage overlay is about it.
 *
 * The scan is the one the globe shows (lib/epochs.ts `pickShownAsset`): on a site with dated
 * captures, the latest epoch's current revision; on a legacy site, its default-visible splat,
 * else its first.
 *
 * The overlay's URL is on the site's metadata, a mirror the worker writes for the site's first
 * epoch only (ADR 0013, decision 8). So on a site with epochs the overlay goes with the scan
 * only when the scan is the current asset of epoch 1 (ordinal 1): laid over another date's
 * tiles, or over a superseded revision of epoch 1, it would describe a scan that is not there.
 * A legacy site (no epochs) shows its scan with the coverage its run measured.
 */
export function shownScan(site: Site): ShownScan {
  const epochs = siteEpochs(site);
  const asset = pickShownAsset(site.assets, epochs, "gaussian-splat") ?? undefined;
  const epoch = epochOf(asset, epochs);
  const first = epochs.find((candidate) => candidate.ordinal === 1);
  const mirrored =
    epochs.length === 0 ||
    (first !== undefined &&
      asset !== undefined &&
      currentAsset(first, site.assets, "gaussian-splat")?.id === asset.id);
  const metadata: unknown = site.metadata;
  const url = isRecord(metadata) ? metadata.coverageUrl : undefined;
  if (!asset || !mirrored || typeof url !== "string" || !url)
    return { asset, epoch, coverage: null };
  const verified = isRecord(metadata) ? metadata.keepVerifiedPct : undefined;
  return {
    asset,
    epoch,
    coverage: typeof verified === "number" ? { url, verifiedPct: verified } : { url },
  };
}
