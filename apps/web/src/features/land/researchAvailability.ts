import type { ResearchRun, ResearchStatus } from "@twin/contracts";

/**
 * Why research is waiting or a research button is disabled, in words, from what
 * GET /research/status says about the worker (apps/api/app/services/research.py:
 * research_status). Pure, so the copy is tested without a server: ResearchWorkerNotice, the
 * lenses and the agent card only place it.
 *
 * Every age is the server's (the database clock), so a skewed browser clock never shows
 * the wrong minutes. A status from an older server, or one still loading, has no `worker`:
 * then nothing is said about the worker rather than something guessed.
 */

/** "12 s", "4 min", "3 h", "2 days": whole units, rounded down, never "0 min". */
export function formatAgo(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min`;
  if (s < 86_400) return `${Math.floor(s / 3600)} h`;
  const days = Math.floor(s / 86_400);
  return `${days} ${days === 1 ? "day" : "days"}`;
}

type Worker = ResearchStatus["worker"];

function workerOf(status: ResearchStatus | undefined): Worker | null {
  return status?.worker ?? null;
}

function ago(worker: Worker): string {
  return formatAgo(worker.secondsSinceSeen ?? 0);
}

/** What a queued run is waiting for. Null unless `run` is queued and the worker is known. */
export function workerLine(
  status: ResearchStatus | undefined,
  run: Pick<ResearchRun, "status"> | undefined,
): string | null {
  const worker = workerOf(status);
  if (run?.status !== "queued" || !worker) return null;
  switch (worker.state) {
    case "online":
      return worker.busy
        ? `Queued behind another research request. The research worker checked in ${ago(worker)} ago.`
        : `Starting. The research worker checked in ${ago(worker)} ago.`;
    case "stale":
      return `Waiting for the research worker: none has checked in for ${ago(worker)}. Your request is saved and starts when a worker runs.`;
    case "stopped":
      return `Waiting for the research worker: the last one stopped ${ago(worker)} ago. Your request is saved and starts when a worker runs.`;
    case "never":
      return "Waiting for a research worker: none has checked in on this server yet. Your request is saved and starts when one runs.";
  }
}

/**
 * Said before anyone asks, when no worker is online: a new analysis would wait in the queue.
 * The caller leaves it out while the land's own run is queued, where `workerLine` says more.
 */
export function workerIdleNote(status: ResearchStatus | undefined): string | null {
  const worker = workerOf(status);
  if (!worker || worker.state === "online") return null;
  const which =
    worker.state === "never"
      ? "has never checked in here"
      : worker.state === "stopped"
        ? `stopped ${ago(worker)} ago`
        : `has not checked in for ${ago(worker)}`;
  return `Analyses run on a separate research worker, which ${which}. New requests will wait in the queue.`;
}

export interface AnalysisInputs {
  ready: boolean;
  canEdit: boolean;
  run: Pick<ResearchRun, "status" | "question" | "focusLabel"> | undefined;
}

/**
 * Why an analysis button is disabled. A queued run's reason leaves the worker out:
 * ResearchWorkerNotice, next to the run, says what it is waiting for and for how long, so
 * saying it here too would repeat the worker's age beside every disabled button.
 */
export function analysisBlocker({ ready, canEdit, run }: AnalysisInputs): string | null {
  if (!ready) return "Connecting to your workspace…";
  if (!canEdit) return "Your workspace role can view research but not start it.";
  if (run?.status === "queued")
    return "One request runs at a time. Cancel the queued request to start something else.";
  if (run?.status === "running") {
    const label = run.focusLabel ?? run.question;
    const short = label.length > 60 ? `${label.slice(0, 60).trimEnd()}…` : label;
    return `One request runs at a time. Wait for “${short}” to finish, or stop it.`;
  }
  return null;
}

/** Shown only once the question box has been used: an empty box is not an error at first. */
export const TYPE_A_QUESTION = "Type a question to ask the agent.";
export const AGENT_NOT_SET_UP =
  "The agent isn't set up on this server. The lens analyses still work.";

export interface AskInputs extends AnalysisInputs {
  /** From /research/status: undefined while it loads, null when it could not be loaded. */
  modelConfigured: boolean | null | undefined;
  pending: boolean;
  recoveryError: boolean;
  question: string;
}

/** Why the agent's send button is disabled: the analysis reasons first, then the question's. */
export function askBlocker(inputs: AskInputs): string | null {
  const blocked = analysisBlocker(inputs);
  if (blocked) return blocked;
  if (inputs.modelConfigured === undefined) return "Checking whether the agent is available…";
  if (inputs.modelConfigured === null)
    return "Could not check whether the agent is available. Reload the page to try again.";
  if (!inputs.modelConfigured) return AGENT_NOT_SET_UP;
  if (inputs.pending) return "Confirm or discard the captured request above first.";
  if (inputs.recoveryError) return "Resolve the unreadable saved question above first.";
  if (!inputs.question.trim()) return TYPE_A_QUESTION;
  return null;
}
