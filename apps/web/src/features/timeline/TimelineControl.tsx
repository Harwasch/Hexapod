import { Clock } from "lucide-react";

import { GlassPanel, GlassSegmentedControl } from "@twin/ui";

import { useSite } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { formatDate } from "@/lib/format";
import { useSites } from "@/state/sites";

/**
 * Subtle temporal control shown only when the active representation has more than
 * one dated capture. Nothing is faked: one version means no timeline.
 */
export function TimelineControl() {
  const scene = useScene();
  const activeSiteId = useSites((s) => s.activeSiteId);
  const representation = useSites((s) =>
    activeSiteId ? s.representation[activeSiteId] : undefined,
  );
  const temporal = useSites((s) => (activeSiteId ? s.temporalAsset[activeSiteId] : undefined));
  const site = useSite(activeSiteId).data;
  if (!site || !representation) return null;
  const versions = site.assets
    .filter((a) => a.representation === representation && a.observedAt)
    .sort((a, b) => (a.observedAt ?? "").localeCompare(b.observedAt ?? ""));
  if (versions.length < 2) return null;
  const current =
    temporal ?? versions.find((a) => a.defaultVisible)?.id ?? versions.at(-1)?.id ?? "";
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
        value={current}
        onValueChange={(assetId) => {
          useSites.getState().setTemporalAsset(site.id, assetId);
          void scene?.sites.setTemporalAsset(assetId);
        }}
        options={versions.map((a) => ({ value: a.id, label: formatDate(a.observedAt) }))}
      />
    </GlassPanel>
  );
}
