import type { ResearchRun } from "@twin/contracts";

import { env } from "@/app/env";

import { workerIdleNote, workerLine } from "./researchAvailability";
import { useResearchStatus } from "./researchApi";

/**
 * Why a queued research run has not started: "Waking the research worker" while it sleeps
 * between runs, or "none has checked in for 14 min" when one went quiet mid-run. Research
 * runs on a separate process (`python -m app.research`) that exits when idle and is started
 * again by the enqueue; without this a queued run looked the same whether the worker was
 * busy, asleep or absent.
 * Renders nothing unless `run` is queued. While it is, the status is refreshed every 10 s,
 * so the notice clears by itself when a worker starts.
 *
 * `className` lets a caller that already draws a status box (ecology, solar) show the line
 * as a footnote inside it instead of a second box.
 */
export function ResearchWorkerNotice({
  run,
  className = "land-notice",
}: {
  run: Pick<ResearchRun, "status"> | undefined;
  className?: string;
}) {
  const status = useResearchStatus(run?.status === "queued");
  const line = workerLine(status.data, run);
  if (!line) return null;
  const offline = status.data?.worker.state !== "online";
  return (
    <p className={className} role="status" data-testid="research-worker-notice">
      {line}
      {env.isDev && offline && (
        <span className="land-footnote">
          {" "}
          Developers: start one with <code>pnpm dev:research</code>.
        </span>
      )}
    </p>
  );
}

/**
 * Said before anyone asks, when no research worker is online: that a sleeping one starts on
 * demand, or that one went quiet and a new analysis would wait in the queue
 * ("Analyses run on a separate research worker, which has not checked in for 2 h."). The
 * caller leaves it out while its own run is queued, where ResearchWorkerNotice says more.
 */
export function ResearchWorkerIdle({ className = "land-muted" }: { className?: string }) {
  const status = useResearchStatus();
  const note = workerIdleNote(status.data);
  if (!note) return null;
  return (
    <p className={className} data-testid="research-worker-idle">
      {note}
      {env.isDev && (
        <span className="land-footnote">
          {" "}
          Developers: start one with <code>pnpm dev:research</code>.
        </span>
      )}
    </p>
  );
}
