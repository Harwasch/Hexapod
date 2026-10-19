import { footprintAreaM2 } from "@twin/geo";

import { areaLabel } from "../../landFlow";
import type { Action, Scenario, Step } from "./planModel";

const WORDS = [
  "no",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "eleven",
  "twelve",
];

/** "three" for 3; digits past twelve. */
export function countWord(count: number): string {
  return WORDS[count] ?? count.toLocaleString();
}

function capital(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function joinWords(parts: string[]): string {
  if (parts.length <= 1) return parts.join("");
  return `${parts.slice(0, -1).join(", ")} and ${parts.at(-1) ?? ""}`;
}

/**
 * Where a plan stands, in the order a person acts on it: made before the land changed, open
 * questions to settle, ready for review, approved and waiting to be scheduled, or scheduled.
 */
export type PlanState = "stale" | "blocked" | "ready" | "approved" | "scheduled";

export function planState(action: Action): PlanState {
  if (action.status === "scheduled") return "scheduled";
  if (action.staleReasons.length) return "stale";
  if (action.status === "approved") return "approved";
  if (action.constraints?.some((constraint) => !constraint.resolved)) return "blocked";
  return "ready";
}

export const STATE_LABEL: Record<PlanState, string> = {
  stale: "Needs a fresh look",
  blocked: "Open questions",
  ready: "Ready to review",
  approved: "Approved",
  scheduled: "Scheduled",
};

const STATE_PHRASE: Record<PlanState, string> = {
  stale: "made before the land changed",
  blocked: "with questions to settle",
  ready: "ready to review",
  approved: "approved and ready to schedule",
  scheduled: "scheduled as a mission",
};

const STATE_SHORT: Record<PlanState, [string, string]> = {
  stale: ["needs a fresh look", "need a fresh look"],
  blocked: ["draft", "drafts"],
  ready: ["draft", "drafts"],
  approved: ["approved", "approved"],
  scheduled: ["scheduled", "scheduled"],
};

/** What a plan is for, when its scenario says: "restoration" or "solar plan". */
export function planKind(action: Action, scenarios: Scenario[]): string {
  const scenario = action.scenario && scenarios.find((item) => item.id === action.scenario?.id);
  if (scenario?.inputs.kind === "restoration") return "restoration";
  if (scenario?.inputs.kind === "solar") return "solar plan";
  return "plan";
}

/** "a three-step restoration" */
export function planNoun(action: Action, scenarios: Scenario[]): string {
  const steps = action.steps.length;
  return `a ${countWord(steps)}-step ${planKind(action, scenarios)}`;
}

/**
 * The card's sentence about every plan on this land, the shown one first: "Two plans: a
 * three-step restoration ready to review, and one draft."
 */
export function plansHeadline(actions: Action[], scenarios: Scenario[], shown: Action | null) {
  if (!actions.length) return "No plans yet.";
  const first = shown ?? actions[0];
  if (!first) return "No plans yet.";
  if (actions.length === 1)
    return `One plan: ${planNoun(first, scenarios)}, ${STATE_PHRASE[planState(first)]}.`;
  if (actions.length === 2) {
    const other = actions.find((item) => item.id !== first.id) ?? first;
    const state = planState(other);
    const short =
      state === "stale" ? "one that needs a fresh look" : `one ${STATE_SHORT[state][0]}`;
    return `Two plans: ${planNoun(first, scenarios)} ${STATE_PHRASE[planState(first)]}, and ${short}.`;
  }
  const groups: [PlanState[], string, string][] = [
    [["scheduled"], "scheduled", "scheduled"],
    [["approved"], "approved", "approved"],
    [["ready", "blocked"], "draft", "drafts"],
    [["stale"], "needs a fresh look", "need a fresh look"],
  ];
  const parts = groups.flatMap(([states, one, many]) => {
    const count = actions.filter((item) => states.includes(planState(item))).length;
    return count ? [`${countWord(count)} ${count === 1 ? one : many}`] : [];
  });
  return `${capital(countWord(actions.length))} plans: ${joinWords(parts)}.`;
}

/** The server's stale reasons, in plain words and without identifiers. */
export function staleReasonText(reasons: string[]): string[] {
  const out = new Set<string>();
  for (const reason of reasons) {
    if (/boundary/i.test(reason)) out.add("The land's outline changed.");
    else if (/scenario/i.test(reason)) out.add("The scenario it came from changed.");
    else if (/feature|inventory/i.test(reason)) out.add("A mapped asset it uses changed.");
    else out.add(reason);
  }
  return [...out];
}

/** "4.2 ac" for a pin; square metres under a tenth of an acre. */
export function shortArea(squareMetres: number): string {
  const acres = squareMetres / 4046.8564224;
  if (acres < 0.1) return `${Math.round(squareMetres).toLocaleString()} m²`;
  return `${acres.toLocaleString(undefined, { maximumFractionDigits: acres < 10 ? 1 : 0 })} ac`;
}

export function stepArea(step: Step): string {
  return step.footprint ? shortArea(footprintAreaM2(step.footprint)) : "Whole area";
}

/** An amount in its currency, without cents: "$4,600". */
export function money(value: number, currency: string): string {
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency,
      maximumFractionDigits: 0,
    }).format(value);
  } catch {
    return `${Math.round(value).toLocaleString()} ${currency}`;
  }
}

function addDays(date: string, days: number): Date {
  const value = new Date(`${date}T00:00:00`);
  value.setDate(value.getDate() + days);
  return value;
}

function shortDate(date: Date): string {
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** "Oct 14" for an ISO date. */
export function dayLabel(date: string | null | undefined): string {
  if (!date) return "";
  const value = new Date(`${date}T00:00:00`);
  return Number.isNaN(value.getTime()) ? date : shortDate(value);
}

/**
 * When a step happens. Before scheduling, its place in the plan ("Day 1–3"); once scheduled,
 * where it stands on the calendar ("Starts Oct 14", "Now, until Oct 16", "Ended Oct 2").
 */
export function stepTiming(
  step: Step,
  action: Pick<Action, "status" | "startDate">,
  today = new Date(),
) {
  const first = step.startDay + 1;
  const last = step.startDay + step.days;
  if (action.status !== "scheduled" || !Number.isFinite(first) || !Number.isFinite(last))
    return last > first ? `Day ${first}–${last}` : `Day ${first}`;
  const start = addDays(action.startDate, step.startDay);
  const end = addDays(action.startDate, step.startDay + step.days - 1);
  const now = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  if (end < now) return `Ended ${shortDate(end)}`;
  if (start <= now) return `Now, until ${shortDate(end)}`;
  return `Starts ${shortDate(start)}`;
}

/** The known cost and what is still missing: "$4,600 · 1 step has no estimate". */
export function costSummary(action: Pick<Action, "currency" | "totalKnownCost" | "uncostedSteps">) {
  const known = action.totalKnownCost > 0 ? money(action.totalKnownCost, action.currency) : null;
  const missing =
    action.uncostedSteps === 0
      ? null
      : action.uncostedSteps === 1
        ? "1 step has no estimate"
        : `${action.uncostedSteps} steps have no estimate`;
  if (known && missing) return `${known} · ${missing}`;
  return known ?? (missing ? `Not estimated yet · ${missing}` : money(0, action.currency));
}

export interface ScenarioNumber {
  label: string;
  value: string;
  /** For drawing a bar against the other scenarios (costs share a currency to compare). */
  amount?: number;
}

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Three numbers that let two possibilities be compared at a glance: area, cost, outcome. */
export function scenarioNumbers(scenario: Pick<Scenario, "inputs" | "result">): ScenarioNumber[] {
  const { inputs, result } = scenario;
  const currency = String(result.summary.currency ?? inputs.currency);
  if (inputs.kind === "restoration") {
    const changing = result.rows.reduce((sum, row) => {
      const change = num(row.changeHa);
      return change && change > 0 ? sum + change : sum;
    }, 0);
    const best = [...result.rows]
      .map((row) => ({
        name: String(row.class ?? ""),
        from: num(row.baselinePercent) ?? 0,
        to: num(row.targetPercent) ?? 0,
      }))
      .sort((a, b) => b.to - b.from - (a.to - a.from))[0];
    const total = num(result.summary.totalCost);
    const goals = result.ecology?.species.length ?? 0;
    return [
      {
        label: "Area changing",
        value: changing > 0 ? areaLabel(changing * 10_000) : "No change in cover",
      },
      {
        label: "Cost",
        value: total === null ? "Not calculated" : money(total, currency),
        ...(total === null ? {} : { amount: total }),
      },
      {
        label: "Outcome",
        value:
          best && best.to > best.from
            ? `${best.name} ${Math.round(best.from)}% → ${Math.round(best.to)}%`
            : goals
              ? `${countWord(goals)} species ${goals === 1 ? "goal" : "goals"}`
              : "No target set",
      },
    ];
  }
  const equity = num(result.summary.initialEquity) ?? num(inputs.installedCost);
  const kwh = num(result.summary.firstYearGenerationKwh);
  const payback = result.summary.paybackYear;
  return [
    {
      label: "Panel area",
      value: inputs.usableRoofAreaM2 > 0 ? areaLabel(inputs.usableRoofAreaM2) : "Not set",
    },
    {
      label: "Upfront cost",
      value: equity === null ? "Not calculated" : money(equity, currency),
      ...(equity === null ? {} : { amount: equity }),
    },
    {
      label: "Outcome",
      value: [
        kwh === null ? null : `${Math.round(kwh).toLocaleString()} kWh a year`,
        typeof payback === "number"
          ? `pays back in year ${payback}`
          : `no payback in ${inputs.years} years`,
      ]
        .filter(Boolean)
        .join(", "),
    },
  ];
}

export function scenarioKindLabel(scenario: Scenario): string {
  return scenario.inputs.kind === "restoration" ? "Restoration" : "Solar";
}

/** The comparison view's sentence. */
export function scenariosHeadline(scenarios: Scenario[]): string {
  if (!scenarios.length) return "No possibilities to compare yet.";
  if (scenarios.length === 1) return `One possibility: ${scenarios[0]?.name ?? ""}.`;
  return `${capital(countWord(scenarios.length))} possibilities, side by side.`;
}
