import type { ResearchRun } from "@twin/contracts";

import { env } from "@/app/env";

import { workerLine } from "./researchAvailability";
import { useResearchStatus } from "./researchApi";

/**
 * Why a queued research run has not started: "Waiting for the research worker: none has
 * checked in for 14 min." Research runs on a separate process (`python -m app.research`),
 * and without this a queued run looked the same whether the worker was busy, idle or absent.
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
