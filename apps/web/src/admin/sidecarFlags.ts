/**
 * Sidecars a republish could not carry, by asset and by site.
 *
 * When a run publishes new tiles, the worker carries each sidecar beside the old ones —
 * objects, skins, a plant rig, collision, the streamed LOD — only where it still holds for
 * the new splats, and flags the asset for every kind it drops (`sidecarFlags`, with words
 * for a person: "Objects need re-segmenting"; docs/SCENE_OBJECTS.md, section 8). Nothing
 * else says so: the scan simply stops showing its objects. These are the rules for showing
 * it; `SidecarNotice.tsx` draws them.
 */
import type { SiteAsset } from "@twin/contracts";

export type SidecarFlag = SiteAsset["sidecarFlags"][number];

export interface FlaggedAsset {
  id: string;
  name: string;
  siteId: string | null;
  flags: SidecarFlag[];
}

/** Every asset with at least one flag, flags in a stable order (by kind). */
export function flaggedAssets(assets: readonly SiteAsset[] | undefined): FlaggedAsset[] {
  return (assets ?? [])
    .filter((asset) => asset.sidecarFlags.length > 0)
    .map((asset) => ({
      id: asset.id,
      name: asset.name,
      siteId: asset.siteId,
      flags: [...asset.sidecarFlags].sort((a, b) => a.kind.localeCompare(b.kind)),
    }));
}

/** The flagged assets of each site, by site id. */
export function flaggedBySite(
  assets: readonly SiteAsset[] | undefined,
): Record<string, FlaggedAsset[]> {
  const bySite: Record<string, FlaggedAsset[]> = {};
  for (const asset of flaggedAssets(assets)) {
    if (asset.siteId === null) continue;
    (bySite[asset.siteId] ??= []).push(asset);
  }
  return bySite;
}

/** What a flag's tooltip says: why it was dropped, and by which run, when. */
export function flagDetail(flag: SidecarFlag): string {
  const when = new Date(flag.flaggedAt);
  const at = Number.isNaN(when.getTime()) ? flag.flaggedAt : when.toISOString().slice(0, 16);
  const run = flag.jobId ? ` by run ${flag.jobId.slice(0, 8)}` : "";
  return `${flag.reason} (dropped${run}, ${at.replace("T", " ")} UTC)`;
}
