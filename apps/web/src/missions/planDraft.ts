/**
 * Bridges between the planner API (a structured draft) and the console's `Plan` records.
 * Pure functions so the flow is unit-tested without a scene.
 */

import type {
  PlanCadence,
  PlanDraft,
  PlanDraftRequest,
  PlanRecord,
  PlanRecordCreate,
  PlanRecordStatus,
  PlanStep,
} from "@twin/contracts";

import { isAreaZone, zoneFromFootprint } from "./areas";
import { bookedWindows } from "./conflicts";
import { isLandZone } from "./landGround";
import { learnedRates } from "./rates";
import type { Plan, PlanOverlay, Project, Zone } from "./types";

export interface DraftSeed {
  goal: string;
  zoneIds?: string[];
  machineIds?: string[];
  refinement?: string;
  previous?: PlanDraft | null;
  /** The plan being revised, left out of the booked windows sent to the planner. */
  replacePlanId?: string | null;
  answers?: Record<string, string | number | boolean>;
}

const CADENCE_LABEL: Record<PlanCadence, string> = {
  once: "One-off",
  daily: "Daily",
  weekly: "Weekly",
  monthly: "Monthly",
  seasonal: "Seasonal",
};

/**
 * Whether the planner gets a zone's outline to size work from. Areas found in the console, a
 * Land's zones (a fence as its line) and every zone of a real project do; a simulated
 * provider's zones do not, because their stated acres are the demo's story, not their
 * footprints' (lift this once the demo's acres equal its outlines).
 */
function sendsGeometry(project: Project, zone: Zone): boolean {
  return isAreaZone(zone) || isLandZone(zone) || !project.simulated;
}

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
      ...(sendsGeometry(project, z) ? { geometry: z.land?.line ?? z.footprint } : {}),
    })),
    existingPlans: project.plans
      .filter((p) => p.id !== seed.replacePlanId)
      .map((p) => {
        const booked = bookedWindows([p]).find((b) => b.plan.id === p.id);
        return {
          id: p.id,
          title: p.title,
          zoneIds: p.zoneIds,
          status: p.state === "Done" ? "done" : p.state === "Cancelled" ? "cancelled" : "scheduled",
          busy: booked?.busy ?? [],
        };
      }),
    rates: learnedRates(project.workLog),
    preferredZoneIds: seed.zoneIds ?? [],
    preferredMachineIds: seed.machineIds ?? [],
    today: new Date().toISOString().slice(0, 10),
  };
  if (seed.refinement) request.refinement = seed.refinement;
  if (seed.answers && Object.keys(seed.answers).length) request.answers = seed.answers;
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

/**
 * "Z-14 West bench" once, whether or not the name already carries the id. Land zones and
 * land-action areas ("L-…", "LA-…") have opaque ids and go by their name alone.
 */
export function idLabel(id: string, name: string): string {
  if (id.startsWith("L-") || id.startsWith("LA-")) return name;
  return name.startsWith(id) ? name : `${id} ${name}`;
}

/** Machine-hours as a stat: minutes when a run is short, so a fence check never reads "0 h". */
export function formatMachineHours(hours: number): string {
  if (hours > 0 && hours < 1) return `${Math.max(1, Math.round(hours * 60))} min`;
  return `${Math.round(hours)} h`;
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
  const named = draft.zoneIds.map((id) => {
    const zone = zones.find((z) => z.id === id);
    return zone ? idLabel(zone.id, zone.name) : id;
  });
  // Work along a line (a fence) has a path, not an area: say how far.
  const pathM = draft.estimates.pathM ?? null;
  const alongLine = draft.estimates.acres < 0.5 && pathM !== null && pathM > 0;
  const size = alongLine ? `${(pathM / 1000).toFixed(1)} km` : `${Math.round(acres)} acres`;
  return {
    id: options.id ?? `plan-${slug(draft.title) || "new"}-${Date.now().toString(36)}`,
    title: draft.title,
    state: "Scheduled",
    status: "idle",
    meta: `${CADENCE_LABEL[draft.cadence]} · ${named.join(", ") || "no zones"} · ${size}`,
    progressPct: 0,
    progressLabel: "Not started",
    objective: draft.objective,
    stats: [
      alongLine
        ? { value: `${(pathM / 1000).toFixed(1)} km`, label: "Path" }
        : { value: String(Math.round(acres)), label: "Acres in scope" },
      { value: String(machines.length), label: "Machines" },
      {
        value: formatMachineHours(draft.estimates.machineHours),
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
    assumptions: draft.assumptions ?? [],
    goal: options.goal,
    source: { kind: draft.source, model: draft.model ?? null },
    startDate: draft.startDate,
  };
}

/**
 * The API body for approving a draft (create) or a revision (same shape plus a note). The
 * ground found in the console travels with it: drawn and outlined areas, and a Land's zones
 * (a line as its corridor), each with where its outline came from.
 */
export function recordBodyFromDraft(
  draft: PlanDraft,
  goal: string,
  projectId: string,
  siteId: string | null,
  zones: readonly Zone[] = [],
): PlanRecordCreate {
  return {
    projectId,
    siteId,
    areas: zones
      .filter((z) => (isAreaZone(z) || isLandZone(z)) && draft.zoneIds.includes(z.id))
      .map((z) => ({
        id: z.id,
        name: z.name,
        footprint: z.footprint,
        ...(z.attribution ? { attribution: z.attribution.slice(0, 300) } : {}),
      })),
    title: draft.title.trim() || "New plan",
    objective: draft.objective,
    goal,
    cadence: draft.cadence,
    startDate: draft.startDate,
    endDate: draft.endDate ?? null,
    zoneIds: draft.zoneIds,
    machineIds: draft.machineIds,
    steps: draft.steps ?? [],
    estimates: draft.estimates,
    assumptions: draft.assumptions ?? [],
    risks: draft.risks,
    questions: draft.questions,
    source: draft.source,
    model: draft.model ?? null,
  };
}

const STATUS_PRESENTATION: Record<
  PlanRecordStatus,
  { status: Plan["status"]; state: string; action: string }
> = {
  scheduled: { status: "idle", state: "Scheduled", action: "Dispatch to fleet" },
  dispatched: { status: "run", state: "Dispatched", action: "Pause plan" },
  paused: { status: "idle", state: "Paused", action: "Resume" },
  done: { status: "ok", state: "Done", action: "Archive" },
  cancelled: { status: "idle", state: "Cancelled", action: "Reopen" },
};

/** How a lifecycle status reads in the console. */
export function presentStatus(status: PlanRecordStatus): {
  status: Plan["status"];
  state: string;
  action: string;
} {
  return STATUS_PRESENTATION[status];
}

/**
 * The ground a stored plan carries, as zones, with where each outline came from. A workspace's
 * plan's ground is private, so it never reaches browser storage.
 */
export function zonesFromRecord(record: PlanRecord): Zone[] {
  return record.areas.map((a) => {
    const zone = zoneFromFootprint(a.id, a.name, a.footprint);
    return {
      ...zone,
      ...(a.attribution ? { attribution: a.attribution, note: a.attribution } : {}),
      ...(record.private ? { private: true } : {}),
    };
  });
}

/** A persisted plan as the console's record, with the lifecycle state the API holds. */
export function planFromRecord(record: PlanRecord, project: Project): Plan {
  const draft: PlanDraft = {
    title: record.title,
    objective: record.objective,
    zoneIds: record.zoneIds,
    machineIds: record.machineIds,
    cadence: record.cadence,
    startDate: record.startDate,
    endDate: record.endDate,
    estimates: record.estimates,
    steps: record.steps,
    assumptions: record.assumptions,
    risks: record.risks,
    questions: record.questions,
    clarifications: [],
    source: record.source,
    model: record.model,
    note: "",
  };
  const plan = planFromDraft(draft, project, { id: record.id, goal: record.goal });
  const presentation = presentStatus(record.status);
  return {
    ...plan,
    ...presentation,
    ...(record.landActionId
      ? {
          agentNote:
            "Scheduled from a land action and approved in Land; machine-hours were not estimated.",
        }
      : {}),
    scope: record.private ? "workspace" : "public",
    landId: record.landId,
    landActionId: record.landActionId,
    revision: record.revision,
    revisions: record.revisions.map((r) => ({
      revision: r.revision,
      note: r.note,
      createdAt: r.createdAt,
      title: r.title,
    })),
    persisted: true,
  };
}

/** One line for the agent stream once a draft is ready. */
export function describeDraft(draft: PlanDraft): string {
  const pathM = draft.estimates.pathM ?? 0;
  const parts = [
    `Drafted “${draft.title}”`,
    `${draft.steps?.length ?? 0} steps`,
    `${draft.machineIds.length} machine${draft.machineIds.length === 1 ? "" : "s"}`,
    draft.estimates.acres < 0.5 && pathM > 0
      ? `${(pathM / 1000).toFixed(1)} km path`
      : `${Math.round(draft.estimates.acres)} acres`,
  ];
  return `${parts.join(" · ")}. Approve it, drag the outline, or tell me what to change.`;
}

/** Zones in step order (a zone appears once, at its first step) with the machines assigned. */
export function overlayFor(scope: {
  zoneIds: string[];
  machineIds?: string[];
  steps?: PlanStep[];
}): PlanOverlay {
  const zones: PlanOverlay["zones"] = [];
  for (const step of scope.steps ?? []) {
    if (!step.zoneId || zones.some((z) => z.zoneId === step.zoneId)) continue;
    zones.push({ zoneId: step.zoneId, machineIds: step.machineIds });
  }
  for (const zoneId of scope.zoneIds) {
    if (!zones.some((z) => z.zoneId === zoneId))
      zones.push({ zoneId, machineIds: scope.machineIds ?? [] });
  }
  return { zones };
}

export interface ScheduleLane {
  machineId: string;
  bars: { title: string; startDay: number; days: number; zoneId: string | null }[];
}

/** One lane per machine with its steps as bars; unassigned steps land in a "Fleet" lane. */
export function scheduleLanes(steps: PlanStep[]): { lanes: ScheduleLane[]; totalDays: number } {
  const lanes = new Map<string, ScheduleLane>();
  let totalDays = 1;
  for (const step of steps) {
    const days = Math.max(Number.isFinite(step.days) ? step.days : 1, 1);
    const startDay = Number.isFinite(step.startDay) ? Math.max(step.startDay, 0) : 0;
    totalDays = Math.max(totalDays, startDay + days);
    const owners = step.machineIds.length ? step.machineIds : ["Fleet"];
    for (const machineId of owners) {
      const lane = lanes.get(machineId) ?? { machineId, bars: [] };
      lane.bars.push({
        title: step.title,
        startDay: step.startDay,
        days,
        zoneId: step.zoneId ?? null,
      });
      lanes.set(machineId, lane);
    }
  }
  return { lanes: [...lanes.values()], totalDays };
}

function listDiff(before: string[], after: string[], noun: string): string[] {
  const added = after.filter((x) => !before.includes(x));
  const removed = before.filter((x) => !after.includes(x));
  const out: string[] = [];
  if (added.length) out.push(`Added ${noun} ${added.join(", ")}.`);
  if (removed.length) out.push(`Dropped ${noun} ${removed.join(", ")}.`);
  return out;
}

/** Plain sentences describing what a redraft changed; empty when nothing material moved. */
export function diffDrafts(before: PlanDraft, after: PlanDraft): string[] {
  const out = [
    ...listDiff(before.zoneIds, after.zoneIds, "zone"),
    ...listDiff(before.machineIds, after.machineIds, "machine"),
  ];
  if (before.cadence !== after.cadence) out.push(`Cadence ${before.cadence} → ${after.cadence}.`);
  if (before.endDate !== after.endDate)
    out.push(`Ends ${before.endDate ?? "ongoing"} → ${after.endDate ?? "ongoing"}.`);
  const b = before.estimates;
  const a = after.estimates;
  if (Math.round(b.machineHours) !== Math.round(a.machineHours))
    out.push(`Machine-hours ${Math.round(b.machineHours)} → ${Math.round(a.machineHours)}.`);
  if (b.calendarDays !== a.calendarDays)
    out.push(`Duration ${b.calendarDays} d → ${a.calendarDays} d.`);
  const stepsBefore = (before.steps ?? []).length;
  const stepsAfter = (after.steps ?? []).length;
  if (stepsBefore !== stepsAfter) out.push(`Steps ${stepsBefore} → ${stepsAfter}.`);
  return out;
}

/** Room one schedule date tick needs: "Oct 16" in the 12 px mono tick face, plus a gap. */
const TICK_LABEL_W = 52;
const TICK_STEPS = [1, 2, 3, 7, 14];

/** Days between labelled ticks: the smallest step whose labels cannot overlap. */
export function tickStep(dayWidth: number): number {
  return TICK_STEPS.find((step) => step * dayWidth >= TICK_LABEL_W) ?? 28;
}
