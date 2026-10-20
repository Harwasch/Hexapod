import type { Site, SiteAsset } from "@twin/contracts";

import { useScene } from "@/cesium/SceneContext";
import {
  alignmentLabel,
  epochOf,
  formatEpochDate,
  formatEpochDateTime,
  frameAnchor,
  observedAtFromUpload,
  pickShownAsset,
  platformLabel,
  siteEpochs,
} from "@/lib/epochs";
import { useSites } from "@/state/sites";

/** How the shown date is placed, in words: with the frame's own date, or on its own. */
function placementLine(
  anchorAsset: SiteAsset | null,
  anchorDate: string | null,
  isAnchor: boolean,
  shares: boolean | null,
): string {
  const clamped = anchorAsset?.renderConfig.clampToGround === true;
  const measured = clamped && (anchorAsset?.renderConfig.groundSamples?.length ?? 0) > 0;
  if (isAnchor) {
    const rests = measured
      ? "Rests on its own measured ground"
      : clamped
        ? "Rests on the ground under it"
        : "Placed as registered";
    return `${rests}; every date of this site shares this placement`;
  }
  // The scene placed it on its own: its tiles are not in the site frame, or the frame's own
  // date did not load (SiteManager.followAnchor, which logs which).
  if (shares === false || !anchorDate)
    return "Placed on its own, not with the placement the site's other dates share";
  return clamped
    ? `Shares the site frame's placement, rested on the ${anchorDate} capture's ground`
    : `Shares the placement of the ${anchorDate} capture, which made the site frame`;
}

/**
 * Date shown, in the Inspector: which of a site's dated captures (epochs, lib/epochs.ts) is on
 * screen, whether it was observed or filled in, what captured it, how it is aligned with the
 * rest of the site and how it is placed. Nothing for a site whose shown model belongs to no
 * epoch: every site before epochs reads as it did.
 */
export function EpochFacts({ site }: { site: Site }) {
  const scene = useScene();
  const representation = useSites((s) => s.representation[site.id]);
  const shownEpoch = useSites((s) => s.epoch[site.id] ?? null);
  const epochs = siteEpochs(site);
  const asset = representation
    ? pickShownAsset(site.assets, epochs, representation, { epochId: shownEpoch })
    : null;
  const epoch = epochOf(asset, epochs);
  if (!asset || !epoch || !representation) return null;
  const anchor = frameAnchor(epochs, site.assets, representation);
  const isAnchor = anchor?.epoch.id === epoch.id;
  const anchorDate = anchor ? formatEpochDate(anchor.epoch.observedAt) : null;
  const shares = scene?.sites.sharesPlacement(site.id, asset.id) ?? null;
  return (
    <section aria-label={`Date shown: ${site.name}`} data-testid="inspector-epoch">
      <p className="glass-eyebrow">Date shown · {site.name}</p>
      <dl className="dl" style={{ marginTop: "0.3rem", fontSize: "var(--text-xs)" }}>
        <dt>Date</dt>
        <dd data-testid="inspector-epoch-date">
          {formatEpochDateTime(epoch.observedAt)}
          {observedAtFromUpload(epoch) && " · upload date; capture time not recorded"}
        </dd>
        <dt>Source</dt>
        <dd data-testid="inspector-epoch-source">
          {asset.sourceKind === "inferred"
            ? "Inferred: filled in by a model, not observed"
            : "Observed (captured)"}
        </dd>
        <dt>Platform</dt>
        <dd data-testid="inspector-epoch-platform">{platformLabel(epoch.platform)}</dd>
        <dt>Alignment</dt>
        <dd data-testid="inspector-epoch-alignment">
          {alignmentLabel(epoch, anchor?.epoch ?? null, epochs).long}
        </dd>
        <dt>Placement</dt>
        <dd data-testid="inspector-epoch-placement">
          {placementLine(anchor?.asset ?? null, anchorDate, isAnchor, shares)}
        </dd>
      </dl>
    </section>
  );
}
