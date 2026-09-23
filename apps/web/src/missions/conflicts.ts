/**
 * Conflicts between a candidate plan and the plans already on the books: a machine booked
 * twice on overlapping days, or a zone another active plan is still treating. Pure, so the
 * review can warn and the planner request can carry the same facts.
 */

import type { PlanStep } from "@twin/contracts";

import type { Plan } from "./types";

export interface BusyWindow {
  machineId: string;
  /** ISO dates; start inclusive, end exclusive. */
  startDate: string;
  endDate: string;
}

export interface PlanConflict {
  kind: "machine" | "zone";
  id: string;
  planId: string;
  planTitle: string;
  /** For machine conflicts: the overlapping days as ISO dates. */
  from?: string;
  to?: string;
}

export interface CandidatePlan {
  id?: string;
  startDate: string;
  zoneIds: string[];
  machineIds: string[];
  steps: PlanStep[];
}

const DAY_MS = 86_400_000;

function addDays(iso: string, days: number): string {
  const date = new Date(`${iso}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return iso;
  return new Date(date.getTime() + days * DAY_MS).toISOString().slice(0, 10);
}

/** Plans that still book machines: anything not finished or cancelled. */
export function isActivePlan(plan: Pick<Plan, "state">): boolean {
  return !["Done", "Cancelled"].includes(plan.state);
}

/** Days each machine is booked by a plan, from its steps (or the whole span when it has none). */
export function busyWindows(plan: CandidatePlan): BusyWindow[] {
  const out: BusyWindow[] = [];
  const steps = plan.steps.filter((s) => s.machineIds.length > 0);
  if (steps.length === 0) {
    const days = Math.max(1, ...plan.steps.map((s) => s.startDay + Math.max(s.days, 1)));
    for (const machineId of plan.machineIds)
      out.push({ machineId, startDate: plan.startDate, endDate: addDays(plan.startDate, days) });
    return out;
  }
  for (const step of steps) {
    const startDate = addDays(plan.startDate, Math.max(step.startDay, 0));
    const endDate = addDays(startDate, Math.max(step.days, 1));
    for (const machineId of step.machineIds) out.push({ machineId, startDate, endDate });
  }
  return out;
}

function overlap(a: BusyWindow, b: BusyWindow): { from: string; to: string } | null {
  const from = a.startDate > b.startDate ? a.startDate : b.startDate;
  const to = a.endDate < b.endDate ? a.endDate : b.endDate;
  return from < to ? { from, to } : null;
}

/** Windows of every active plan except the candidate itself, for the planner request. */
export function bookedWindows(
  plans: Plan[],
  excludeId?: string,
): { plan: Plan; busy: BusyWindow[] }[] {
  return plans
    .filter((p) => p.id !== excludeId && isActivePlan(p) && p.startDate)
    .map((p) => ({
      plan: p,
      busy: busyWindows({
        startDate: p.startDate ?? "",
        zoneIds: p.zoneIds,
        machineIds: p.machineIds ?? [],
        steps: p.steps ?? [],
      }),
    }));
}

export function planConflicts(candidate: CandidatePlan, plans: Plan[]): PlanConflict[] {
  const conflicts: PlanConflict[] = [];
  const mine = busyWindows(candidate);
  for (const { plan, busy } of bookedWindows(plans, candidate.id)) {
    for (const window of busy) {
      for (const own of mine) {
        if (own.machineId !== window.machineId) continue;
        const hit = overlap(own, window);
        if (!hit) continue;
        const existing = conflicts.find(
          (c) => c.kind === "machine" && c.id === own.machineId && c.planId === plan.id,
        );
        if (existing) {
          if (hit.from < (existing.from ?? hit.from)) existing.from = hit.from;
          if (hit.to > (existing.to ?? hit.to)) existing.to = hit.to;
        } else {
          conflicts.push({
            kind: "machine",
            id: own.machineId,
            planId: plan.id,
            planTitle: plan.title,
            from: hit.from,
            to: hit.to,
          });
        }
      }
    }
    for (const zoneId of candidate.zoneIds) {
      if (plan.zoneIds.includes(zoneId) && plan.status !== "ok")
        conflicts.push({ kind: "zone", id: zoneId, planId: plan.id, planTitle: plan.title });
    }
  }
  return conflicts;
}

/** One sentence per conflict, in the operator's words. */
export function describeConflict(conflict: PlanConflict): string {
  if (conflict.kind === "machine")
    return `${conflict.id} is booked by “${conflict.planTitle}” ${conflict.from ?? ""} → ${conflict.to ?? ""}.`;
  return `${conflict.id} is still in scope of “${conflict.planTitle}”.`;
}
