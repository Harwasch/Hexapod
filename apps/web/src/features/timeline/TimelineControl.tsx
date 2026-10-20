import { Clock } from "lucide-react";

import { GlassPanel, GlassSegmentedControl } from "@twin/ui";

import { useSite } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import {
  alignmentLabel,
  epochDateLabel,
  epochOptions,
  frameAnchor,
  platformLabel,
  shownOption,
  siteEpochs,
} from "@/lib/epochs";
import { useSites } from "@/state/sites";

/**
 * Capture date: which of the active site's dated captures (epochs, lib/epochs.ts) is shown,
 * each named by its date, what captured it and how it is placed against the rest of the site.
 * Shown only when the representation on screen has two or more; nothing is faked, so one
 * capture means no control.
 *
 * Choosing a date asks the scene (`SiteManager.setEpoch`); the control then follows what the
 * scene says it shows (the store's `epoch`, from the `epoch` event), never its own click.
 */
export function TimelineControl() {
  const scene = useScene();
  const activeSiteId = useSites((s) => s.activeSiteId);
  const representation = useSites((s) =>
    activeSiteId ? s.representation[activeSiteId] : undefined,
  );
  const shown = useSites((s) => (activeSiteId ? (s.epoch[activeSiteId] ?? null) : null));
  const site = useSite(activeSiteId).data;
  if (!site || !representation) return null;
  const epochs = siteEpochs(site);
  const options = epochOptions(epochs, site.assets, representation);
  if (options.length < 2) return null;
  const value = shownOption(options, shown)?.epoch.id ?? "";
  const anchor = frameAnchor(epochs, site.assets, representation)?.epoch ?? null;
  return (
    <GlassPanel
      strong
      pill
      className="timeline"
      role="group"
      aria-label="Capture date"
      data-testid="timeline"
    >
      <Clock size={14} className="glass-muted" aria-hidden="true" />
      <GlassSegmentedControl
        aria-label="Capture date"
        data-testid="timeline-epochs"
        value={value}
        onValueChange={(epochId) => void scene?.sites.setEpoch(epochId)}
        options={options.map(({ epoch }) => {
          const date = epochDateLabel(epoch);
          const platform = platformLabel(epoch.platform);
          const alignment = alignmentLabel(epoch, anchor, epochs);
          return {
            value: epoch.id,
            label: (
              <span className="timeline__option">
                <span className="timeline__date">{date}</span>
                <span className="timeline__meta">
                  {platform} · {alignment.short}
                </span>
              </span>
            ),
            ariaLabel: `${date}, ${platform}, ${alignment.long}`,
          };
        })}
      />
    </GlassPanel>
  );
}
