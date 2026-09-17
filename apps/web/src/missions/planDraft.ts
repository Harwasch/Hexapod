/**
 * Bridges between the planner API (a structured draft) and the console's `Plan` records.
 * Pure functions so the flow is unit-tested without a scene.
 */

import type { PlanCadence, PlanDraft, PlanDraftRequest } from "@twin/contracts";

import type { Plan, Project } from "./types";

export interface DraftSeed {
  goal: string;
  zoneIds?: string[];
  machineIds?: string[];
  refinement?: string;
  previous?: PlanDraft | null;
}

const CADENCE_LABEL: Record<PlanCadence, string> = {
  once: "One-off",
  daily: "Daily",
  weekly: "Weekly",
  monthly: "Monthly",
  seasonal: "Seasonal",
};

/** The context the planner needs: the goal plus the project's zones, machines and plans. */
export function buildDraftRequest(project: Project, seed: DraftSeed): PlanDraftRequest {
  const request: PlanDraftRequest = {
    goal: seed.goal.trim(),
    projectName: project.name,
    machines: project.machines.map((m) => ({
      id: m.id,
      name: m.name,
      model: m.model,
      status: m.status,
      task: m.task,
      batteryPct: m.batteryPct,
    })),
    zones: project.zones.map((z) => ({
      id: z.id,
      name: z.name,
      acres: z.acres,
      task: z.task,
      progressPct: z.progressPct,
      treated: z.treated,
      note: z.note,
    })),
    existingPlans: project.plans.map((p) => ({ id: p.id, title: p.title, zoneIds: p.zoneIds })),
    preferredZoneIds: seed.zoneIds ?? [],
    preferredMachineIds: seed.machineIds ?? [],
    today: new Date().toISOString().slice(0, 10),
  };
  if (seed.refinement) request.refinement = seed.refinement;
  if (seed.previous) {
    const { source: _source, model: _model, note: _note, ...body } = seed.previous;
    request.previousDraft = body;
  }
  return request;
}

function formatDay(iso: string): string {
  const date = new Date(`${iso}T00:00:00`);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

/** "Z-14 West bench" once, whether or not the name already carries the id. */
export function idLabel(id: string, name: string): string {
  return name.startsWith(id) ? name : `${id} ${name}`;
}

function slug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 40);
}

/** Turns an approved draft into the console's plan record (scheduled, nothing done yet). */
export function planFromDraft(
  draft: PlanDraft,
  project: Project,
  options: { id?: string; goal: string },
): Plan {
  const zones = project.zones.filter((z) => draft.zoneIds.includes(z.id));
  const machines = project.machines.filter((m) => draft.machineIds.includes(m.id));
  const acres = draft.estimates.acres || zones.reduce((sum, z) => sum + z.acres, 0);
  const ongoing = draft.cadence !== "once" || !draft.endDate;
  const machineList = machines.map((m) => idLabel(m.id, m.name)).join(", ") || "unassigned";
  return {
    id: options.id ?? `plan-${slug(draft.title) || "new"}-${Date.now().toString(36)}`,
    title: draft.title,
    state: "Scheduled",
    status: "idle",
    meta: `${CADENCE_LABEL[draft.cadence]} · ${draft.zoneIds.join(", ") || "no zones"} · ${Math.round(acres)} acres`,
    progressPct: 0,
    progressLabel: "Not started",
    objective: draft.objective,
    stats: [
      { value: String(Math.round(acres)), label: "Acres in scope" },
      { value: String(machines.length), label: "Machines" },
      {
        value: `${Math.round(draft.estimates.machineHours)} h`,
        label: "Machine-hours",
        accent: true,
      },
    ],
    facts: [
      { k: "Cadence", v: CADENCE_LABEL[draft.cadence] },
      { k: "Starts", v: formatDay(draft.startDate) },
      { k: "Machines", v: machineList },
      { k: "Area", v: zones.map((z) => idLabel(z.id, z.name)).join(", ") || "none" },
      ...(draft.risks.length ? [{ k: "Risks", v: draft.risks.join(" ") }] : []),
    ],
    agentNote:
      draft.source === "claude"
        ? `Drafted by Claude${draft.model ? ` (${draft.model})` : ""} and approved by you.`
        : "Rule-based draft approved by you; no model was configured when it was drafted.",
    action: "Pause plan",
    span: ongoing
      ? `${CADENCE_LABEL[draft.cadence]} from ${formatDay(draft.startDate)} · no end date`
      : `${formatDay(draft.startDate)} – ${formatDay(draft.endDate ?? draft.startDate)}`,
    start: formatDay(draft.startDate),
    end: ongoing ? "Ongoing" : formatDay(draft.endDate ?? draft.startDate),
    todayPct: 0,
    ongoing,
    zoneIds: [...draft.zoneIds],
    machineIds: [...draft.machineIds],
    steps: draft.steps ?? [],
    goal: options.goal,
    source: { kind: draft.source, model: draft.model ?? null },
  };
}

/** One line for the agent stream once a draft is ready. */
export function describeDraft(draft: PlanDraft): string {
  const parts = [
    `Drafted “${draft.title}”`,
    `${draft.steps?.length ?? 0} steps`,
    `${draft.machineIds.length} machine${draft.machineIds.length === 1 ? "" : "s"}`,
    `${Math.round(draft.estimates.acres)} acres`,
  ];
  const tail = draft.questions.length
    ? ` I have ${draft.questions.length} question${draft.questions.length === 1 ? "" : "s"} before you approve.`
    : " Review it in the Plans window.";
  return `${parts.join(" · ")}.${tail}`;
}
