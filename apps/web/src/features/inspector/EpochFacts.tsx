import { useEffect, useState } from "react";

import type { Epoch, Representation, Site, SiteAsset } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import { confidenceFact } from "@/lib/confidenceFacts";
import {
  alignmentLabel,
  epochOf,
  epochOptions,
  formatEpochDate,
  formatEpochDateTime,
  frameAnchor,
  observedAtFromUpload,
  pickShownAsset,
  platformLabel,
  siteEpochs,
} from "@/lib/epochs";
import { useSites } from "@/state/sites";
import { useToasts } from "@/state/toasts";

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
  // Not decided yet (the frame's own date still loading), or not in the scene: claim nothing.
  if (shares === null)
    return `Not placed yet: waiting for the ${anchorDate} capture, whose placement every date shares`;
  return clamped
    ? `Shares the site frame's placement, rested on the ${anchorDate} capture's ground`
    : `Shares the placement of the ${anchorDate} capture, which made the site frame`;
}

/**
 * Date shown, in the Inspector: which of a site's dated captures (epochs, lib/epochs.ts) is on
 * screen, whether it was observed or filled in, what captured it, how it is aligned with the
 * rest of the site and how it is placed. Nothing for a site whose shown model belongs to no
 * epoch: every site before epochs reads as it did. While two dates are compared (Compare ›
 * Dates), each of the two, named by its side of the divider.
 */
export function EpochFacts({ site }: { site: Site }) {
  const scene = useScene();
  // Whether the shown date shares the placement is the scene's to say, and it may say so
  // after this renders (its anchor still loading): read again whenever its models change.
  const [, refresh] = useState(0);
  useEffect(() => scene?.events.on("tilesets", () => refresh((n) => n + 1)), [scene]);
  const representation = useSites((s) => s.representation[site.id]);
  const shownEpoch = useSites((s) => s.epoch[site.id] ?? null);
  const pair = useSites((s) => (s.compareDates?.siteId === site.id ? s.compareDates : null));
  if (!representation) return null;
  const epochs = siteEpochs(site);
  if (pair) {
    const options = epochOptions(epochs, site.assets, representation);
    return (["left", "right"] as const).map((side) => {
      const option = options.find((o) => o.epoch.id === pair[side]);
      return option ? (
        <DateFacts
          key={side}
          site={site}
          asset={option.asset}
          epoch={option.epoch}
          representation={representation}
          heading={`${side === "left" ? "Left" : "Right"} of the divider`}
          testId={`inspector-epoch-${side}`}
        />
      ) : null;
    });
  }
  const asset = pickShownAsset(site.assets, epochs, representation, { epochId: shownEpoch });
  const epoch = epochOf(asset, epochs);
  if (!asset || !epoch) return null;
  return (
    <DateFacts
      site={site}
      asset={asset}
      epoch={epoch}
      representation={representation}
      heading="Date shown"
      testId="inspector-epoch"
    />
  );
}

function DateFacts({
  site,
  asset,
  epoch,
  representation,
  heading,
  testId,
}: {
  site: Site;
  asset: SiteAsset;
  epoch: Epoch;
  representation: Representation;
  heading: string;
  testId: string;
}) {
  const scene = useScene();
  const [planning, setPlanning] = useState(false);
  const epochs = siteEpochs(site);
  const anchor = frameAnchor(epochs, site.assets, representation);
  const isAnchor = anchor?.epoch.id === epoch.id;
  const anchorDate = anchor ? formatEpochDate(anchor.epoch.observedAt) : null;
  const shares = scene?.sites.sharesPlacement(site.id, asset.id) ?? null;
  // The root's extras, once the model is in the scene (under any renderer: CesiumJS loads it).
  const loaded = scene?.sites.loadedAssets().find((a) => a.assetId === asset.id);
  const extras = loaded
    ? ((loaded.tileset.root as { extras?: unknown } | undefined)?.extras ?? null)
    : undefined;
  return (
    <section aria-label={`${heading}: ${site.name}`} data-testid={testId}>
      <p className="glass-eyebrow">
        {heading} · {site.name}
      </p>
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
        {representation === "gaussian-splat" && (
          <>
            <dt>Confidence</dt>
            <dd data-testid="inspector-epoch-confidence">{confidenceFact(extras)}</dd>
          </>
        )}
      </dl>
      {/* Mission zones and the API only, so under every renderer; the flow loads on demand. */}
      {representation === "gaussian-splat" && (
        <GlassButton
          size="sm"
          disabled={planning}
          onClick={() => {
            setPlanning(true);
            void import("@/features/rescan/rescanWeak")
              .then((m) => m.rescanWeak(scene, site, epoch.id))
              // A chunk gone after a deploy: say so rather than nothing happening.
              .catch(() =>
                useToasts.getState().push({ tone: "error", title: "Could not plan the re-scan" }),
              )
              .finally(() => setPlanning(false));
          }}
        >
          Plan a re-scan of weak areas
        </GlassButton>
      )}
    </section>
  );
}
