import { useId, type ReactNode } from "react";

import { GlassButton } from "@twin/ui";

import { analysisBlocker } from "../../../researchAvailability";
import type { useLandAnalysis } from "../../useLandAnalysis";
import { isStale, OVERVIEW_SPECS, type OverviewKey, type Raster } from "./overviewData";

type Analysis = ReturnType<typeof useLandAnalysis>;

/**
 * An analysis Details offers on demand (bare-earth terrain, seasonal signatures): what it reads,
 * the one button that starts it, and, beside a disabled button, why it is disabled. Its
 * progress shows on the card's running line, like every Overview run; once it has a result for
 * this outline the result's own details take its place.
 */
export function OverviewAnalysis({
  spec: key,
  analysis,
  raster,
  revision,
  ready,
  canEdit,
  busy,
  unavailable,
  onStart,
  children,
  testId,
  titled = true,
}: {
  spec: OverviewKey;
  analysis: Analysis;
  raster: Raster | null;
  revision: number;
  ready: boolean;
  canEdit: boolean;
  /** Another start is on its way. */
  busy: boolean;
  /** Why this land cannot have it at all (outside the United States). */
  unavailable?: string | null;
  onStart: () => void;
  /** What it reads and how, in a sentence or two. */
  children: ReactNode;
  testId: string;
  /** Off where the section around it already names the analysis. */
  titled?: boolean;
}) {
  const reasonId = useId();
  const spec = OVERVIEW_SPECS[key];
  const current = raster ? !isStale(raster, revision) : false;
  if (current && !analysis.running) return null;
  const failed = analysis.failed && analysis.run && !current ? analysis.run : null;
  const reason = analysisBlocker({ ready, canEdit, run: analysis.run ?? undefined }) ?? unavailable;
  return (
    <div className="overview-analysis" data-testid={testId}>
      {titled && <h5>{spec.name}</h5>}
      {children}
      {raster && !current && (
        <p className="land-muted">Made for an earlier outline of this land.</p>
      )}
      {failed && (
        <p className="land-alert" role="alert">
          {failed.status === "cancelled"
            ? `${spec.name} was stopped.`
            : `${spec.failed} ${failed.error ?? "The source did not answer."}`}
        </p>
      )}
      <div className="land-row">
        <GlassButton
          size="sm"
          disabled={Boolean(reason) || busy}
          loading={busy}
          aria-describedby={reason ? reasonId : undefined}
          onClick={onStart}
        >
          {spec.start}
        </GlassButton>
      </div>
      {reason && (
        <p id={reasonId} className="land-muted" role="status">
          {reason}
        </p>
      )}
    </div>
  );
}
