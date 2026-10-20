import { AGE_RAMP, CONFIDENCE_RAMP, rampCss } from "@/lib/cellRamps";
import { AGE_BUCKETS } from "@/lib/dataAge";
import type { CellViewReport, ViewMode } from "@/state/viewMode";

function Swatch({ color, striped = false }: { color: string; striped?: boolean }) {
  return (
    <span
      className={
        striped ? "cell-legend__swatch cell-legend__swatch--striped" : "cell-legend__swatch"
      }
      style={{ background: color }}
      aria-hidden="true"
    />
  );
}

const GREY = "rgb(128 128 128)";

/** The confidence key: what `weak-v1` calls each colour (design M4-6 C3). */
function ConfidenceKey() {
  return (
    <ul className="cell-legend__rows">
      <li>
        <Swatch color={rampCss(CONFIDENCE_RAMP[1] ?? [1, 0, 0])} />
        Weak: under half its splats pass the capture checks, or unseen
      </li>
      <li>
        <Swatch color={rampCss(CONFIDENCE_RAMP[3] ?? [1, 1, 0])} />
        Fair
      </li>
      <li>
        <Swatch color={rampCss(CONFIDENCE_RAMP[5] ?? [0, 1, 0])} />
        Strong
      </li>
      <li>
        <Swatch color={GREY} />
        No evidence
      </li>
    </ul>
  );
}

/** The data-age key for the shown date: its age, and whether later dates were compared. */
function AgeKey({ report }: { report: Extract<CellViewReport, { kind: "ready" }> }) {
  const bucket = report.ageBucket ?? null;
  const newer = report.newer;
  return (
    <ul className="cell-legend__rows">
      <li data-testid="cell-legend-age">
        {bucket === null ? (
          <>
            <Swatch color={GREY} />
            This model has no capture date
          </>
        ) : (
          <>
            <Swatch color={rampCss(AGE_RAMP[bucket] ?? [0.5, 0.5, 0.5])} />
            This date: {(AGE_BUCKETS[bucket]?.label ?? "").toLowerCase()} old
          </>
        )}
      </li>
      <li data-testid="cell-legend-newer">
        <Swatch color={GREY} striped />
        {!newer || newer.dates === 0
          ? "No later date of this site"
          : newer.cells > 0
            ? "Newer data here: switch date"
            : "No later date has evidence where this one does"}
      </li>
      {newer && newer.skipped.length > 0 && (
        <li className="cell-legend__skipped" data-testid="cell-legend-skipped">
          Not compared: {newer.skipped.join("; ")}
        </li>
      )}
    </ul>
  );
}

/** The key to an analytic view, or what stands in for one (loading, not measured, failed). */
export function CellLegend({
  mode,
  report,
}: {
  mode: Exclude<ViewMode, "photo">;
  report: CellViewReport | undefined;
}) {
  let body;
  if (!report || report.kind === "loading") {
    body = <p className="cell-legend__state">Loading…</p>;
  } else if (report.kind === "not-measured") {
    body = (
      <p className="cell-legend__state">
        Not measured: this model came without capture evidence, so nothing is tinted
      </p>
    );
  } else if (report.kind === "failed") {
    body = <p className="cell-legend__state">Could not load: {report.message}</p>;
  } else if (mode === "confidence") {
    body = <ConfidenceKey />;
  } else if (report.mode === "data-age") {
    body = <AgeKey report={report} />;
  } else {
    body = <p className="cell-legend__state">Loading…</p>;
  }
  return (
    <div
      className="cell-legend"
      role="note"
      aria-label={mode === "confidence" ? "Confidence key" : "Data age key"}
      data-testid="cell-legend"
    >
      {body}
      {mode === "confidence" && report?.kind === "ready" && (
        <p className="cell-legend__policy">weak-v1</p>
      )}
    </div>
  );
}
