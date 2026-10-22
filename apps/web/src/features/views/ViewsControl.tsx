import { Palette } from "lucide-react";

import { GlassButton, GlassPanel, GlassSegmentedControl } from "@twin/ui";

import { useSite } from "@/api/queries";
import { epochOptions, pickShownAsset, siteEpochs } from "@/lib/epochs";
import {
  DRAW_WITH_CESIUM,
  drawWithCesium,
  NEEDS_CESIUM_NOTICE,
  VIEW_MODES,
  viewModeDrawable,
} from "@/lib/viewModes";
import { useSplatRenderer } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useViewMode } from "@/state/viewMode";
import { useViewRule } from "@/state/viewRule";

import { CellLegend } from "./CellLegend";
import { RuleControl } from "./RuleControl";
import { VIEW_RULE_TEXT } from "./ruleText";

/**
 * How the active splat scan is coloured: Photo, Confidence or Data age (state/viewMode.ts),
 * with the key to what is drawn. The analytic views are CesiumJS's alone (lib/viewModes.ts):
 * under another renderer they are offered disabled, with the reason and the one-click switch,
 * and Photo is what the control says is shown, because it is.
 */
export function ViewsControl({ siteId }: { siteId: string }) {
  const mode = useViewMode((s) => s.mode);
  const setMode = useViewMode((s) => s.set);
  const renderer = useSplatRenderer();
  const shownEpoch = useSites((s) => s.epoch[siteId] ?? null);
  const site = useSite(siteId).data;
  const asset = site
    ? pickShownAsset(site.assets, siteEpochs(site), "gaussian-splat", { epochId: shownEpoch })
    : null;
  const report = useViewMode((s) => (asset ? s.cells[asset.id] : undefined));
  const ruleOn = useViewRule((s) => s.active?.siteId === siteId);
  const onCesium = renderer === "cesium";
  const shown = onCesium && !ruleOn ? mode : "photo";
  const epochs = site ? siteEpochs(site) : [];
  const dates = site ? epochOptions(epochs, site.assets, "gaussian-splat").length : 0;
  return (
    <GlassPanel
      strong
      pill
      className="views"
      role="group"
      aria-label="Splat view"
      data-testid="views"
    >
      <Palette size={14} className="glass-muted" aria-hidden="true" />
      <GlassSegmentedControl
        aria-label="Splat view"
        data-testid="views-modes"
        value={shown}
        onValueChange={setMode}
        options={VIEW_MODES.map(({ value, label }) => ({
          value,
          label,
          disabled: !viewModeDrawable(value, renderer) || (ruleOn && value !== "photo"),
        }))}
      />
      {ruleOn && (
        <span className="views__notice" role="note" data-testid="views-rule-holds">
          {VIEW_RULE_TEXT.holdsViews}
        </span>
      )}
      {!onCesium && (
        <div className="views__notice" role="note" data-testid="views-needs-cesium">
          <span>{NEEDS_CESIUM_NOTICE}</span>
          <GlassButton size="sm" variant="ghost" onClick={drawWithCesium}>
            {DRAW_WITH_CESIUM}
          </GlassButton>
        </div>
      )}
      {onCesium && shown !== "photo" && <CellLegend mode={shown} report={report} />}
      {dates >= 2 && <RuleControl siteId={siteId} epochs={epochs} onCesium={onCesium} />}
    </GlassPanel>
  );
}
