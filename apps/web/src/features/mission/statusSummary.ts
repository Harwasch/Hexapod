/**
 * The words on the status line, as pure functions of the mission state (tested in
 * `__tests__/statusLine.test.tsx`).
 */

import type { Project } from "@/missions/types";
import type { AgentLogEntry } from "@/state/mission";

export interface FleetCounts {
  machines: number;
  /** Machines out on the ground: working, or stopped there needing attention. */
  working: number;
  attention: number;
}

/** A KPI from the provider by its label, as a whole number, when it is one. */
function stat(project: Pick<Project, "fleetStats">, label: RegExp): number | null {
  const value = project.fleetStats.find((s) => label.test(s.label))?.value;
  if (value === undefined) return null;
  const n = Number.parseInt(value, 10);
  return Number.isFinite(n) ? n : null;
}

/**
 * How many machines are working and how many need attention.
 *
 * The Fleet window's KPIs come first ("Working now", "Need attention" in `fleetStats`) so the
 * status line and the window can never disagree; without them the machines are counted by
 * status — a machine that needs attention is still out working, which is how the KPIs count.
 */
export function fleetCounts(project: Pick<Project, "machines" | "fleetStats">): FleetCounts {
  const counted = {
    working: project.machines.filter((m) => m.status !== "idle").length,
    attention: project.machines.filter((m) => m.status === "attention").length,
  };
  return {
    machines: project.machines.length,
    working: stat(project, /working/i) ?? counted.working,
    attention: stat(project, /need.*attention/i) ?? counted.attention,
  };
}

export function sumFleet(counts: FleetCounts[]): FleetCounts {
  return counts.reduce(
    (sum, c) => ({
      machines: sum.machines + c.machines,
      working: sum.working + c.working,
      attention: sum.attention + c.attention,
    }),
    { machines: 0, working: 0, attention: 0 },
  );
}

function plural(n: number, one: string, many = `${one}s`): string {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`;
}

/**
 * "4 working · 2 need attention" for the site in view. The status line puts its "Fleet:"
 * label in front, which a phone drops from view (not from screen readers) to fit the counts.
 */
export function fleetLine(counts: FleetCounts): string {
  if (counts.machines === 0) return "no machines registered";
  const parts = [`${counts.working.toLocaleString()} working`];
  if (counts.attention > 0) parts.push(`${counts.attention.toLocaleString()} need attention`);
  if (counts.working === 0 && counts.attention === 0)
    parts[0] = `${plural(counts.machines, "machine")} idle`;
  return parts.join(" · ");
}

/** At globe scale with no site: "3 sites · 6 machines · 2 alerts". */
export function globeLine(sites: number, counts: FleetCounts): string {
  const parts = [plural(sites, "site"), plural(counts.machines, "machine")];
  parts.push(counts.attention > 0 ? plural(counts.attention, "alert") : "no alerts");
  return parts.join(" · ");
}

/** A reply stays on the status line this long, then the line returns to the running task. */
export const REPLY_HOLD_MS = 15_000;

export interface AgentLine {
  /** What the agent is doing, or just said. */
  text: string;
  /** Other tasks running or queued besides the one shown. */
  more: number;
  /** Something is running right now. */
  busy: boolean;
  idle: boolean;
}

/**
 * The agent's one line: a plan being drafted, else the last exchange while it is fresh (the
 * answer to what the operator just asked belongs where they are looking), else the task
 * running now, else idle. `fresh` says the last log entry is younger than `REPLY_HOLD_MS`
 * (`isFresh`); the caller keeps the clock, so this stays pure.
 */
export function agentLine(
  project: Pick<Project, "agent"> | null,
  log: readonly AgentLogEntry[],
  drafting: boolean,
  fresh: boolean,
): AgentLine {
  const actions = project?.agent.actions ?? [];
  const running = actions.filter((a) => a.status === "run");
  const queued = actions.filter((a) => a.status === "wait");
  const open = running.length + queued.length;
  if (drafting) return { text: "Drafting a plan…", more: open, busy: true, idle: false };
  const last = log.at(-1);
  if (last && fresh) {
    // Asked and not yet answered: say so, rather than carry on about something else.
    if (last.role === "you")
      return { text: `On it: “${last.text}”`, more: open, busy: true, idle: false };
    return { text: last.text, more: open, busy: running.length > 0, idle: false };
  }
  const current = running[0];
  if (current) return { text: current.text, more: open - 1, busy: true, idle: false };
  return { text: "idle", more: queued.length, busy: false, idle: true };
}

/** Whether a log entry still holds the agent's line at `now`. */
export function isFresh(entry: Pick<AgentLogEntry, "at"> | undefined, now: number): boolean {
  return entry !== undefined && now - entry.at < REPLY_HOLD_MS;
}
