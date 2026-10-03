/**
 * "Objects need re-segmenting": what a republish dropped, shown where a person looks.
 *
 * Small on purpose — a badge per dropped kind beside the asset it belongs to, its reason in
 * the tooltip — and gone once the kind is attached again (the API clears the flag).
 */
import { AlertTriangle } from "lucide-react";

import { GlassBadge } from "@twin/ui";

import { flagDetail, type FlaggedAsset, type SidecarFlag } from "./sidecarFlags";

export function SidecarNotice({ flags }: { flags: readonly SidecarFlag[] }) {
  if (flags.length === 0) return null;
  return (
    <span className="admin-flags">
      {flags.map((flag, index) => (
        <GlassBadge
          key={`${flag.kind}:${index}`}
          tone="warning"
          title={flagDetail(flag)}
          data-testid="sidecar-flag"
        >
          <AlertTriangle size={11} aria-hidden="true" /> {flag.action}
        </GlassBadge>
      ))}
    </span>
  );
}

/** Every flagged asset, with the site it is on: nothing when nothing is flagged. */
export function FlaggedAssets({
  assets,
  siteNames,
}: {
  assets: readonly FlaggedAsset[];
  siteNames: Record<string, string>;
}) {
  if (assets.length === 0) return null;
  return (
    <section className="admin-card" aria-labelledby="flags-heading" data-testid="flagged-assets">
      <div className="admin-card__head">
        <h2 id="flags-heading">Scans missing what was published beside them</h2>
        <span className="admin-note">{assets.length} flagged</span>
      </div>
      <p className="admin-note">
        A run published new tiles, and these could not be carried onto them: the scan shows without
        each until it is made again for the new tiles and attached. A notice goes once it is.
      </p>
      <ul className="admin-flag-list">
        {assets.map((asset) => (
          <li key={asset.id}>
            <span className="admin-strong">{asset.name}</span>{" "}
            <span className="admin-dim">
              {asset.siteId ? (siteNames[asset.siteId] ?? "unnamed site") : "no site"}
            </span>{" "}
            <SidecarNotice flags={asset.flags} />
          </li>
        ))}
      </ul>
    </section>
  );
}
