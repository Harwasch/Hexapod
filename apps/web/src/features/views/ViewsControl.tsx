import { Palette, ScanEye } from "lucide-react";

import { GlassButton, GlassPanel, GlassSegmentedControl, GlassSwitch } from "@twin/ui";

import { useSite } from "@/api/queries";
import { AS_CAPTURED_TEXT } from "@/lib/asCapturedText";
import { epochOptions, pickShownAsset, siteEpochs } from "@/lib/epochs";
import {
  DRAW_WITH_CESIUM,
  drawWithCesium,
  NEEDS_CESIUM_NOTICE,
  VIEW_MODES,
  viewModeDrawable,
} from "@/lib/viewModes";
import { useAsCaptured } from "@/state/asCaptured";
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
 * and Photo is what the control says is shown, because it is. Beside them, the As captured
 * switch (state/asCaptured.ts): the scan as measured, nothing simulated or inferred drawn.
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
  const asCaptured = useAsCaptured((s) => s.on);
  const setAsCaptured = useAsCaptured((s) => s.set);
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
      <span className="views__rule" title={AS_CAPTURED_TEXT.hint}>
        <GlassSwitch
          id="views-as-captured"
          checked={asCaptured}
          onCheckedChange={setAsCaptured}
          aria-label={AS_CAPTURED_TEXT.label}
        />
        <label className="views__rule-label" htmlFor="views-as-captured">
          {AS_CAPTURED_TEXT.label}
        </label>
      </span>
    </GlassPanel>
  );
}

/**
 * Said in the HUD strip for as long as As captured is on (state/asCaptured.ts), whatever the
 * site, with the way back: a viewer who arrived on a `?asCaptured=1` link may never have seen
 * the switch (Splat view, ViewsControl). Loaded on demand (ViewsSlot.tsx).
 */
export function AsCapturedChip() {
  const set = useAsCaptured((s) => s.set);
  return (
    <GlassPanel
      strong
      pill
      className="views as-captured"
      role="status"
      aria-label={AS_CAPTURED_TEXT.label}
      data-testid="as-captured-chip"
    >
      <ScanEye size={14} className="glass-muted" aria-hidden="true" />
      <span>{AS_CAPTURED_TEXT.chip}</span>
      <GlassButton size="sm" variant="ghost" onClick={() => set(false)}>
        {AS_CAPTURED_TEXT.off}
      </GlassButton>
    </GlassPanel>
  );
}
