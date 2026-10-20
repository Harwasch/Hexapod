import type { Site } from "@twin/contracts";

type SiteAsset = Site["assets"][number];

/** The coverage overlay of the scan view.html shows: where it is, and how much of the keep
 *  tier held-out frames verified (absent when accuracy was not measured). */
export interface ShownCoverage {
  url: string;
  verifiedPct?: number;
}

/** The splat view.html shows, and the coverage overlay that belongs to it (or none). */
export interface ShownScan {
  asset: SiteAsset | undefined;
  coverage: ShownCoverage | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** A site's epochs, oldest first, or none: read defensively, since an older API, a cached
 *  record or a test's partial site may not carry them. */
function epochsOf(site: Site): Site["epochs"] {
  const listed: unknown = (site as { epochs?: unknown }).epochs;
  if (!Array.isArray(listed)) return [];
  return [...(listed as Site["epochs"])].sort((a, b) => a.ordinal - b.ordinal);
}

/**
 * Which splat view.html shows, and whether the site's coverage overlay is about it.
 *
 * The overlay's URL is on the site's metadata, a mirror the worker writes for the site's first
 * epoch only (ADR 0013, decision 8). So on a site with epochs the scan shown is the current
 * revision of its oldest epoch that has tiles -- epoch 1's, the one the mirror describes and the
 * site opens on -- and the overlay goes with it only when that epoch is epoch 1. A re-run of
 * epoch 1 puts its new revision after epoch 2's asset in `site.assets`, which is why the scan is
 * not simply the first splat listed: that would lay epoch 1's coverage over epoch 2's tiles. A
 * legacy site (no epochs) shows its first splat with the coverage its run measured.
 */
export function shownScan(site: Site): ShownScan {
  const splats = site.assets.filter((asset) => asset.representation === "gaussian-splat");
  const epochs = epochsOf(site);
  let asset: SiteAsset | undefined;
  let mirrored = epochs.length === 0;
  if (epochs.length === 0) {
    asset = splats[0];
  } else {
    for (const epoch of epochs) {
      const current = splats.find((candidate) => candidate.id === epoch.currentAssetId);
      if (current) {
        asset = current;
        mirrored = epoch.ordinal === 1;
        break;
      }
    }
    asset ??= splats[0];
  }
  const metadata: unknown = site.metadata;
  const url = isRecord(metadata) ? metadata.coverageUrl : undefined;
  if (!asset || !mirrored || typeof url !== "string" || !url) return { asset, coverage: null };
  const verified = isRecord(metadata) ? metadata.keepVerifiedPct : undefined;
  return {
    asset,
    coverage: typeof verified === "number" ? { url, verifiedPct: verified } : { url },
  };
}
