/**
 * Product metrics for the planning flow, recorded per session so pilots can be read from the
 * dev panel: time to first draft, time to approval, redrafts, and whether the operator edited
 * the title or objective before approving.
 */

import { recordSpan } from "./timing";

export interface PlanningSession {
  openedAt: number;
  firstDraftAt: number | null;
  redrafts: number;
  edited: boolean;
}

export interface PlanningOutcome {
  toFirstDraftMs: number | null;
  toApproveMs: number;
  redrafts: number;
  edited: boolean;
}

let current: PlanningSession | null = null;
const outcomes: PlanningOutcome[] = [];

export function planningOpened(): void {
  current = { openedAt: performance.now(), firstDraftAt: null, redrafts: 0, edited: false };
}

export function planningDrafted(isRedraft: boolean): void {
  if (!current) planningOpened();
  if (!current) return;
  if (isRedraft) current.redrafts += 1;
  else current.firstDraftAt ??= performance.now();
}

export function planningEdited(): void {
  if (current) current.edited = true;
}

export function planningApproved(): PlanningOutcome | null {
  if (!current) return null;
  const now = performance.now();
  const outcome: PlanningOutcome = {
    toFirstDraftMs: current.firstDraftAt === null ? null : current.firstDraftAt - current.openedAt,
    toApproveMs: now - current.openedAt,
    redrafts: current.redrafts,
    edited: current.edited,
  };
  outcomes.push(outcome);
  recordSpan("planning", outcome.toApproveMs, { ...outcome });
  current = null;
  return outcome;
}

export function planningOutcomes(): readonly PlanningOutcome[] {
  return outcomes;
}

/** Medians over the session, the numbers the PRD tracks. */
export function planningSummary(): {
  plans: number;
  medianToApproveMs: number | null;
  medianRedrafts: number | null;
  approvedUneditedPct: number | null;
} {
  if (outcomes.length === 0)
    return { plans: 0, medianToApproveMs: null, medianRedrafts: null, approvedUneditedPct: null };
  const median = (values: number[]) => {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const upper = sorted[mid] ?? 0;
    const lower = sorted[mid - 1] ?? upper;
    return sorted.length % 2 ? upper : (lower + upper) / 2;
  };
  return {
    plans: outcomes.length,
    medianToApproveMs: median(outcomes.map((o) => o.toApproveMs)),
    medianRedrafts: median(outcomes.map((o) => o.redrafts)),
    approvedUneditedPct: Math.round(
      (outcomes.filter((o) => !o.edited).length / outcomes.length) * 100,
    ),
  };
}

/** Test hook. */
export function resetPlanningMetrics(): void {
  current = null;
  outcomes.length = 0;
}
