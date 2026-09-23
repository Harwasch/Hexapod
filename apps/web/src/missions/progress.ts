/**
 * Execution progress of a plan against its schedule, from the work log: acres logged in
 * the plan's zones by its machines since it started, versus where the schedule says it
 * should be today. Pure, so the console and tests read the same number.
 */

import type { Plan, WorkLogRow } from "./types";

export interface PlanProgress {
  /** Acres logged against the plan since its start. */
  acresDone: number;
  /** Acres in scope (the plan's estimate). */
  acresTotal: number;
  /** 0–100, from acres. */
  actualPct: number;
  /** 0–100, where the schedule says the plan should be today. */
  expectedPct: number;
  /** Positive when behind, negative when ahead, in calendar days. */
  driftDays: number;
  /** "On schedule", "Behind by 3 d", "Ahead by 1 d", "Not started", "Done". */
  label: string;
  tone: "ok" | "warn" | "idle";
}

const DAY_MS = 86_400_000;

/** Work-log dates are "MM-DD"; the year is the plan's start year (or the next, across new year). */
function rowDate(row: WorkLogRow, startDate: string): string | null {
  const match = /^(\d{2})-(\d{2})$/.exec(row.date);
  if (!match) return /^\d{4}-\d{2}-\d{2}$/.test(row.date) ? row.date : null;
  const year = Number(startDate.slice(0, 4));
  const candidate = `${year}-${match[1]}-${match[2]}`;
  return candidate < startDate.slice(0, 10) && startDate.slice(5, 7) === "12" && match[1] === "01"
    ? `${year + 1}-${match[1]}-${match[2]}`
    : candidate;
}

function zoneOfRow(row: WorkLogRow): string | null {
  return /\b(Z-\d{2})\b/i.exec(row.zone)?.[1]?.toUpperCase() ?? null;
}

export function planProgress(plan: Plan, rows: readonly WorkLogRow[], today: Date): PlanProgress {
  const startDate = plan.startDate ?? "";
  const acresTotal = Number(plan.stats[0]?.value) || 0;
  const dispatched =
    plan.state === "Dispatched" || plan.state === "Paused" || plan.state === "Done";
  const todayIso = today.toISOString().slice(0, 10);
  let acresDone = 0;
  if (startDate) {
    for (const row of rows) {
      const date = rowDate(row, startDate);
      const zone = zoneOfRow(row);
      if (!date || !zone || date < startDate || date > todayIso) continue;
      if (!plan.zoneIds.includes(zone)) continue;
      if (plan.machineIds?.length && !plan.machineIds.includes(row.machine)) continue;
      acresDone += row.acres ?? 0;
    }
  }
  const actualPct = acresTotal > 0 ? Math.min(100, Math.round((acresDone / acresTotal) * 100)) : 0;
  const days = Math.max(
    1,
    plan.steps?.reduce((m, s) => Math.max(m, s.startDay + Math.max(s.days, 1)), 1) ?? 1,
  );
  const elapsed = startDate
    ? (new Date(`${todayIso}T00:00:00Z`).getTime() - new Date(`${startDate}T00:00:00Z`).getTime()) /
      DAY_MS
    : 0;
  const expectedPct =
    plan.ongoing || !dispatched
      ? 0
      : Math.max(0, Math.min(100, Math.round((elapsed / days) * 100)));
  const driftDays = plan.ongoing ? 0 : Math.round(((expectedPct - actualPct) / 100) * days);
  if (plan.state === "Done")
    return {
      acresDone,
      acresTotal,
      actualPct: 100,
      expectedPct: 100,
      driftDays: 0,
      label: "Done",
      tone: "ok",
    };
  if (!dispatched)
    return {
      acresDone,
      acresTotal,
      actualPct,
      expectedPct,
      driftDays: 0,
      label: "Not started",
      tone: "idle",
    };
  const label =
    driftDays > 0
      ? `Behind by ${driftDays} d`
      : driftDays < 0
        ? `Ahead by ${-driftDays} d`
        : "On schedule";
  return {
    acresDone,
    acresTotal,
    actualPct,
    expectedPct,
    driftDays,
    label,
    tone: driftDays > 0 ? "warn" : "ok",
  };
}

/** The refinement the operator sends to replan from where the fleet actually is. */
export function replanRefinement(progress: PlanProgress, today: Date): string {
  const done = `${Math.round(progress.acresDone)} of ${Math.round(progress.acresTotal)} acres are done as of ${today.toISOString().slice(0, 10)}`;
  if (progress.driftDays > 0)
    return `We are ${progress.driftDays} days behind: ${done}. Resequence the remaining work from today.`;
  return `${done}. Resequence the remaining work from today.`;
}
