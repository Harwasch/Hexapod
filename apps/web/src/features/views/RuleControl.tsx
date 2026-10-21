import { GlassSelect } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import type { ViewRule } from "@/lib/compositor";
import { AGE_RAMP, rampCss } from "@/lib/cellRamps";
import { AGE_BUCKETS } from "@/lib/dataAge";
import { AS_CAPTURED_TEXT } from "@/lib/asCapturedText";
import { formatEpochDate, observedAtFromUpload } from "@/lib/epochs";
import type { Epoch } from "@twin/contracts";
import { useAsCaptured } from "@/state/asCaptured";
import { useLiving } from "@/state/living";
import { useSites } from "@/state/sites";
import { useViewRule, type ViewRuleLegend } from "@/state/viewRule";

import { VIEW_RULE_ENDED, VIEW_RULE_TEXT as RULE_TEXT, VIEW_RULES } from "./ruleText";

/**
 * A date as the key names it: its capture date, or -- dated only by its upload -- said to be
 * the upload's, its capture time unknown (the rule ranks it below every dated date).
 */
function dateOf(epochs: readonly Epoch[], epochId: string): string {
  const epoch = epochs.find((e) => e.id === epochId);
  if (!epoch) return epochId;
  const date = formatEpochDate(epoch.observedAt);
  return observedAtFromUpload(epoch) ? `Uploaded ${date}, capture time unknown` : date;
}

/**
 * The key to a rule: each date taking part, the cells it owns and how its coverage was read
 * (a date without a confidence grid covers its tiles' bounds, which is said), and the cells an
 * inferred layer fills. Under Data age, each date's age swatch.
 */
function RuleLegend({
  legend,
  epochs,
  rule,
}: {
  legend: ViewRuleLegend;
  epochs: readonly Epoch[];
  rule: ViewRule;
}) {
  return (
    <div className="cell-legend" role="note" aria-label="Per-cell key" data-testid="rule-legend">
      <ul className="cell-legend__rows">
        {legend.dates.map((d) => {
          const bucket = d.ageBucket;
          return (
            <li key={d.epochId} data-testid="rule-legend-date">
              {rule === "data-age" && bucket !== null && (
                <span
                  className="cell-legend__swatch"
                  style={{ background: rampCss(AGE_RAMP[bucket] ?? [0.5, 0.5, 0.5]) }}
                  aria-hidden="true"
                />
              )}
              {dateOf(epochs, d.epochId)}:{" "}
              {d.coverage === "loading"
                ? "reading what it covers…"
                : `${String(d.cells)} cells${d.coverage === "tile-bounds" ? ` (${RULE_TEXT.tileBounds})` : ""}`}
              {rule === "data-age" && bucket !== null && (
                <> · {(AGE_BUCKETS[bucket]?.label ?? "").toLowerCase()}</>
              )}
            </li>
          );
        })}
        {legend.inferredCells > 0 && rule !== "ground-truth" && (
          <li data-testid="rule-legend-inferred">
            Inferred: {String(legend.inferredCells)} cells no date saw
          </li>
        )}
        {legend.leftOut.map((d) => {
          const date = dateOf(epochs, d.epochId);
          return (
            <li key={d.epochId} className="cell-legend__skipped" data-testid="rule-legend-left-out">
              {d.reason === "frame"
                ? `${date}${d.asOf ? ", the As of date," : ""} is not drawn: ${RULE_TEXT.ownFrame}`
                : `${date}${d.asOf ? ", the As of date," : ""} is not drawn yet: still loading`}
            </li>
          );
        })}
        {legend.left > 0 && (
          <li className="cell-legend__skipped">
            {String(legend.left)} more layers are not drawn (sixteen at most)
          </li>
        )}
      </ul>
      <p className="cell-legend__policy">{String(legend.cellM)} m cells</p>
    </div>
  );
}

/**
 * Per cell: one of the per-cell view rules (lib/compositor.ts, ADR 0017), or One date. Offered
 * while the site has two dates or more under a renderer that draws the rules -- CesiumJS or
 * PlayCanvas (`drawable`; the caller's notice covers Spark and the WebGPU trial); disabled, and
 * says why, while a Living Survey rig moves the scan or with more than four dates up to As of
 * (lib/ruleCap.ts). The scene owns
 * the rule: the select shows what it says it draws (state/viewRule.ts), never its own click,
 * and why the last one ended when nobody turned it off. While As captured is on the rule
 * chosen is drawn as Ground truth (compositorSession.ts), and the control says so.
 */
export function RuleControl({
  siteId,
  epochs,
  drawable,
  tooMany = false,
}: {
  siteId: string;
  epochs: readonly Epoch[];
  /** The renderer drawing splats draws the rules (lib/viewModes.ts `perCellDrawable`). */
  drawable: boolean;
  /** More than four dates up to As of (lib/ruleCap.ts): none can be drawn as one. */
  tooMany?: boolean;
}) {
  const scene = useScene();
  const active = useViewRule((s) => (s.active?.siteId === siteId ? s.active.rule : null));
  const ended = useViewRule((s) => (s.ended?.siteId === siteId ? s.ended.reason : null));
  const legend = useViewRule((s) => s.legend);
  const compare = useSites((s) => s.compareDates?.siteId === siteId);
  const rigged = useLiving((s) => s.status.sites.some((site) => site.siteId === siteId));
  const held = useAsCaptured((s) => s.on);
  const disabled = !drawable || rigged || (tooMany && !active);
  return (
    <div className="views__rule" data-testid="views-rule">
      <label className="views__rule-label" htmlFor="views-rule-select">
        {RULE_TEXT.label}
      </label>
      <GlassSelect
        id="views-rule-select"
        value={active ?? ""}
        disabled={disabled}
        onChange={(e) => void scene?.sites.setViewRule((e.target.value || null) as ViewRule | null)}
      >
        <option value="">{RULE_TEXT.off}</option>
        {VIEW_RULES.map((r) => (
          <option key={r.value} value={r.value} title={r.hint}>
            {r.label}
          </option>
        ))}
      </GlassSelect>
      {tooMany && !active && drawable && (
        <span className="views__notice" role="note" data-testid="views-rule-dates">
          {RULE_TEXT.tooMany}
        </span>
      )}
      {rigged && drawable && (
        <span className="views__notice" role="note" data-testid="views-rule-living">
          {RULE_TEXT.living}
        </span>
      )}
      {!active && compare && !disabled && ended !== "compare" && (
        <span className="views__notice" role="note" data-testid="views-rule-compare">
          {RULE_TEXT.compare}
        </span>
      )}
      {!active && ended && (
        <span className="views__notice" role="status" data-testid="views-rule-ended">
          {VIEW_RULE_ENDED[ended]}
        </span>
      )}
      {active && held && active !== "ground-truth" && (
        <span className="views__notice" role="note" data-testid="views-rule-as-captured">
          {AS_CAPTURED_TEXT.rule}
        </span>
      )}
      {active && legend && (
        <RuleLegend legend={legend} epochs={epochs} rule={held ? "ground-truth" : active} />
      )}
    </div>
  );
}
