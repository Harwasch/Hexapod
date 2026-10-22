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

/**
 * How long a queued run may wait for a sleeping worker before the words stop saying it is
 * waking up. The research worker exits after five idle minutes, and queueing a run asks Fly
 * to start it again (apps/api/app/services/worker_wake.py); its first act is a claim. A
 * product choice, not a measurement: generous next to a machine start, short enough that a
 * wake that never happened (no Fly token, a local server with no worker) is not hidden.
 */
export const WAKE_GRACE_S = 120;

/**
 * Asleep, not missing: the worker stopped itself when it ran out of work ("stopped"), or
 * has not run on this server since its presence was last cleared ("never"). Presence is
 * written only while a run is active, so both are the normal resting state.
 */
function asleep(worker: Worker): boolean {
  return worker.state === "stopped" || worker.state === "never";
}

/** What a queued run is waiting for. Null unless `run` is queued and the worker is known. */
export function workerLine(
  status: ResearchStatus | undefined,
  run: Pick<ResearchRun, "status"> | undefined,
): string | null {
  const worker = workerOf(status);
  if (run?.status !== "queued" || !worker) return null;
  if (asleep(worker)) {
    const waited = status?.queue?.oldestQueuedSeconds ?? null;
    if (waited !== null && waited >= WAKE_GRACE_S)
      return `The research worker has not started yet: this request has waited ${formatAgo(waited)}. It is saved and starts when the worker runs.`;
    return "Waking the research worker: it sleeps when there is nothing to do and starts for your request. Your request is saved.";
  }
  switch (worker.state) {
    case "online":
      return worker.busy
        ? `Queued behind another research request. The research worker checked in ${ago(worker)} ago.`
        : `Starting. The research worker checked in ${ago(worker)} ago.`;
    case "stale":
      return `Waiting for the research worker: none has checked in for ${ago(worker)}. Your request is saved and starts when a worker runs.`;
  }
  return null;
}

/**
 * Said before anyone asks, when no worker is online. A sleeping worker is the normal state, so
 * the note says it starts on demand; only a worker that went quiet mid-run ("stale") is said
 * to be missing, and then a new analysis would wait in the queue. The caller leaves it out
 * while the land's own run is queued, where `workerLine` says more.
 */
export function workerIdleNote(status: ResearchStatus | undefined): string | null {
  const worker = workerOf(status);
  if (!worker || worker.state === "online") return null;
  if (asleep(worker))
    return "Analyses run on a separate research worker. It sleeps when there is nothing to do and starts when you ask, so the first request can take a moment longer.";
  return `Analyses run on a separate research worker, which has not checked in for ${ago(worker)}. New requests will wait in the queue.`;
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
