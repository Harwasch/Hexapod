import type { components } from "@twin/contracts";
import { defaultImportMapping, parseInventoryImport } from "./inventoryImport";
export type ActionDraft = components["schemas"]["LandActionCreate"];
export type ActionRecord = components["schemas"]["LandActionRead"];
export type ActionEdit = Pick<ActionRecord, "id" | "landId" | "revision" | "status" | "missionId">;
export interface ActionDraftSnapshot {
  version: 1;
  landId: string;
  draft: ActionDraft;
  editing: ActionEdit | null;
}
export const actionDraftKey = (scope: string, landId: string) =>
  `living-world-land-draft:${encodeURIComponent(scope)}:action:${landId}`;
export const actionEdit = (record: ActionRecord): ActionEdit => ({
  id: record.id,
  landId: record.landId,
  revision: record.revision,
  status: record.status,
  missionId: record.missionId,
});
export function serializeActionDraft(value: unknown) {
  return JSON.stringify(value, (_key, item: unknown) =>
    typeof item === "number" && !Number.isFinite(item) ? { __actionNumber: "NaN" } : item,
  );
}
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("The saved action has invalid fields.");
  return value as Record<string, unknown>;
}
function list(value: unknown, max: number): unknown[] {
  if (!Array.isArray(value) || value.length > max)
    throw new Error("The saved action list is invalid.");
  return value as unknown[];
}
function strings(value: unknown, max: number) {
  if (list(value, max).some((item) => typeof item !== "string"))
    throw new Error("The saved action text is invalid.");
}
function fields(value: Record<string, unknown>, text: string[], numeric: string[]) {
  if (
    text.some((key) => typeof value[key] !== "string") ||
    numeric.some(
      (key) =>
        typeof value[key] !== "number" ||
        (!Number.isFinite(value[key]) && !Number.isNaN(value[key])),
    )
  )
    throw new Error("The saved action field types are invalid.");
}
function reference(value: unknown) {
  const row = object(value);
  if (typeof row.id !== "string" || !Number.isInteger(row.revision) || Number(row.revision) < 1)
    throw new Error("The saved action reference is invalid.");
}
function footprint(value: unknown) {
  const parsed = parseInventoryImport(JSON.stringify(value), "geojson", defaultImportMapping)[0];
  if (
    !parsed?.geometry ||
    parsed.error ||
    !["Polygon", "MultiPolygon"].includes(parsed.geometry.type)
  )
    throw new Error("The saved work area geometry is invalid.");
}
export function parseActionDraft(text: string, landId: string): ActionDraftSnapshot {
  if (text.length > 4_000_000) throw new Error("This action draft exceeds the recovery limit.");
  const value = object(
    JSON.parse(text, (_key, item: unknown) =>
      item &&
      typeof item === "object" &&
      Object.keys(item).length === 1 &&
      (item as Record<string, unknown>).__actionNumber === "NaN"
        ? NaN
        : item,
    ),
  );
  if (value.version !== 1 || value.landId !== landId)
    throw new Error("This action draft belongs to different land or has an unsupported format.");
  const draft = object(value.draft);
  fields(
    draft,
    ["requestKey", "title", "objective", "startDate", "currency"],
    ["boundaryRevision"],
  );
  if (
    !/^[a-f0-9-]{36}$/i.test(String(draft.requestKey)) ||
    !Number.isInteger(draft.boundaryRevision) ||
    Number(draft.boundaryRevision) < 1
  )
    throw new Error("The saved action request or boundary revision is invalid.");
  if (draft.scenario != null) reference(draft.scenario);
  list(draft.features ?? [], 200).forEach(reference);
  strings(draft.evidenceIds ?? [], 100);
  strings(draft.assumptions ?? [], 100);
  list(draft.exclusions ?? [], 50).forEach(footprint);
  for (const item of list(draft.steps, 250)) {
    const step = object(item);
    fields(step, ["id", "title", "successMeasure"], ["startDay", "days"]);
    fields({ detail: "", costBasis: "", ...step }, ["detail", "costBasis"], []);
    strings(step.resources ?? [], 50);
    strings(step.machineIds ?? [], 100);
    strings(step.dependsOn ?? [], 100);
    if (step.estimatedCost != null) fields(step, [], ["estimatedCost"]);
    if (step.footprint != null) footprint(step.footprint);
  }
  for (const item of list(draft.constraints ?? [], 100)) {
    const constraint = { resolved: false, resolution: "", ...object(item) };
    fields(constraint, ["text", "resolution"], []);
    if (typeof constraint.resolved !== "boolean")
      throw new Error("The saved constraint state is invalid.");
    strings((constraint as Record<string, unknown>).evidenceIds ?? [], 100);
  }
  if (value.editing != null) {
    const editing = object(value.editing);
    reference(editing);
    if (
      editing.landId !== landId ||
      !["draft", "approved", "scheduled"].includes(String(editing.status)) ||
      (editing.missionId != null && typeof editing.missionId !== "string")
    )
      throw new Error("The saved action revision is invalid.");
  }
  return value as unknown as ActionDraftSnapshot;
}
export function actionPayload(draft: ActionDraft): ActionDraft {
  return {
    requestKey: draft.requestKey,
    title: draft.title,
    objective: draft.objective,
    boundaryRevision: draft.boundaryRevision,
    scenario: draft.scenario ?? null,
    features: draft.features ?? [],
    evidenceIds: draft.evidenceIds ?? [],
    startDate: draft.startDate,
    currency: draft.currency,
    exclusions: draft.exclusions ?? [],
    assumptions: (draft.assumptions ?? []).map((line) => line.trim()).filter(Boolean),
    constraints: (draft.constraints ?? []).map((item) => ({
      text: item.text,
      resolved: item.resolved ?? false,
      resolution: item.resolution ?? "",
      evidenceIds: item.evidenceIds ?? [],
    })),
    steps: draft.steps.map((step) => ({
      id: step.id,
      title: step.title,
      detail: step.detail ?? "",
      startDay: step.startDay,
      days: step.days,
      resources: (step.resources ?? []).map((line) => line.trim()).filter(Boolean),
      machineIds: step.machineIds ?? [],
      estimatedCost: step.estimatedCost ?? null,
      costBasis: step.costBasis ?? "",
      dependsOn: step.dependsOn ?? [],
      footprint: step.footprint ?? null,
      successMeasure: step.successMeasure,
    })),
  };
}
function stable(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, item]) => [key, stable(item)]),
    );
  return value;
}
export function sameAction(a: ActionDraft, b: ActionDraft): boolean {
  return (
    serializeActionDraft(stable({ ...actionPayload(a), requestKey: null })) ===
    serializeActionDraft(stable({ ...actionPayload(b), requestKey: null }))
  );
}
export function downloadActionDraft(value: unknown) {
  const url = URL.createObjectURL(
    new Blob([typeof value === "string" ? value : serializeActionDraft(value)], {
      type: "application/json",
    }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = "action-draft.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * A reason the draft cannot be saved yet, and the DOM id of the field that fixes it. The plan
 * editor (Plans lens) sets `noValidate`: the browser's bubbles cannot point into a collapsed
 * <details>, so this list is what stops Save, and the server's 422 stays the backstop.
 */
export interface DraftProblem {
  id: string;
  message: string;
}

type StepField =
  "title" | "detail" | "start" | "days" | "success" | "area" | "resources" | "cost" | "cost-basis";

/** The plan editor's field ids, shared with actionDraftProblems so the two cannot drift. */
export const actionFieldId = {
  title: "action-title",
  objective: "action-objective",
  start: "action-start",
  currency: "action-currency",
  assumptions: "action-assumptions",
  exclusions: "action-exclusions",
  step: (index: number, field: StepField) => `action-step-${index}-${field}`,
  constraint: (index: number, field: "text" | "resolution") =>
    `action-constraint-${index}-${field}`,
};

const blank = (text: string | null | undefined) => !text?.trim();
// The server strips whitespace before it checks a length, so trimmed text is what counts.
const over = (text: string | null | undefined, max: number) => (text?.trim().length ?? 0) > max;
const DAY_MS = 86_400_000;

/**
 * Everything the server would refuse in this draft, worded as the fix in the plan editor's own
 * words (Plan name, Goal, Done when, Question…) and in the order the editor shows its fields:
 * name, goal and start date, each step top to bottom, areas to avoid, then the folded money
 * and questions sections. Mirrors the validators in apps/api/app/schemas/land_actions.py
 * (LandActionCreate, LandActionStep, ActionConstraint): keep the two in step. Geometry
 * validity and vertex counts are left to the server; the form only offers footprints the
 * inventory already validated.
 */
export function actionDraftProblems(draft: ActionDraft): DraftProblem[] {
  const problems: DraftProblem[] = [];
  const add = (id: string, message: string) => problems.push({ id, message });
  const ids = actionFieldId;
  if (blank(draft.title)) add(ids.title, "Add a plan name.");
  else if (over(draft.title, 200)) add(ids.title, "Shorten the plan name to 200 characters.");
  if (blank(draft.objective)) add(ids.objective, "Describe the goal.");
  else if (over(draft.objective, 2000)) add(ids.objective, "Shorten the goal to 2,000 characters.");
  const start = /^\d{4}-\d{2}-\d{2}$/.test(draft.startDate) ? Date.parse(draft.startDate) : NaN;
  const lastDay = Math.max(0, ...draft.steps.map((step) => step.startDay + step.days));
  if (Number.isNaN(start)) add(ids.start, "Choose a start date.");
  // Python's date ends at 9999-12-31; the server refuses a schedule that runs past it.
  else if (Number.isFinite(lastDay) && new Date(start + lastDay * DAY_MS).getUTCFullYear() > 9999)
    add(ids.start, "Choose an earlier start date. This schedule would run past the year 9999.");

  const byId = new Map(draft.steps.map((step) => [step.id, step]));
  let footprints = 0;
  draft.steps.forEach((step, index) => {
    const n = index + 1;
    const at = (field: StepField) => ids.step(index, field);
    if (blank(step.title)) add(at("title"), `Step ${n}: add a step name.`);
    else if (over(step.title, 200))
      add(at("title"), `Step ${n}: shorten the step name to 200 characters.`);
    if (step.footprint && ++footprints === 50)
      add(
        at("area"),
        `Step ${n}: at most 49 steps can have an area of their own. Use the whole area for the rest.`,
      );
    // The form shows day startDay + 1, so the server's 0..36500 reads as 1..36501.
    if (!Number.isFinite(step.startDay) || step.startDay < 0)
      add(at("start"), `Step ${n}: enter a start day (1 or later).`);
    else if (!Number.isInteger(step.startDay))
      add(at("start"), `Step ${n}: enter the start day as a whole number.`);
    else if (step.startDay > 36500) add(at("start"), `Step ${n}: start on day 36,501 or earlier.`);
    if (!Number.isFinite(step.days) || step.days < 1)
      add(at("days"), `Step ${n}: enter a duration of at least 1 day.`);
    else if (!Number.isInteger(step.days))
      add(at("days"), `Step ${n}: enter the duration in whole days.`);
    else if (step.days > 3650)
      add(at("days"), `Step ${n}: keep the duration to 3,650 days or fewer.`);
    const cost = step.estimatedCost;
    if (cost != null) {
      if (!Number.isFinite(cost) || cost < 0)
        add(at("cost"), `Step ${n}: enter a cost of 0 or more, or leave it blank.`);
      else if (cost > 1e12)
        add(at("cost"), `Step ${n}: enter a cost of at most 1,000,000,000,000 ${draft.currency}.`);
      if (Number.isFinite(cost) && blank(step.costBasis))
        add(
          at("cost-basis"),
          `Step ${n}: say where its ${cost.toLocaleString()} ${draft.currency} cost comes from.`,
        );
    }
    if (over(step.costBasis, 2000))
      add(at("cost-basis"), `Step ${n}: shorten where the cost comes from to 2,000 characters.`);
    if (blank(step.successMeasure))
      add(at("success"), `Step ${n}: say how you will know it is done.`);
    else if (over(step.successMeasure, 2000))
      add(at("success"), `Step ${n}: shorten “Done when” to 2,000 characters.`);
    if (over(step.detail, 1000))
      add(at("detail"), `Step ${n}: shorten what the work involves to 1,000 characters.`);
    // actionPayload drops blank resource lines, so only the lines it sends count.
    const resources = (step.resources ?? []).map((line) => line.trim()).filter(Boolean);
    if (resources.length > 50) add(at("resources"), `Step ${n}: list up to 50 resources.`);
    else if (resources.some((line) => line.length > 200))
      add(at("resources"), `Step ${n}: keep each resource to 200 characters.`);
    for (const dependency of step.dependsOn ?? []) {
      const before = byId.get(dependency);
      if (!before || before.id === step.id) continue;
      const finish = before.startDay + before.days;
      if (Number.isFinite(finish) && Number.isFinite(step.startDay) && finish > step.startDay)
        add(
          at("start"),
          `Step ${n} starts on day ${step.startDay + 1}, but “${before.title.trim() || "its prerequisite"}” finishes on day ${finish}. Start it on day ${finish + 1} or later.`,
        );
    }
  });
  if ((draft.exclusions ?? []).length > 50)
    add(ids.exclusions, "Avoid at most 50 areas: remove some, or clear them all.");
  if (!/^[A-Z]{3}$/.test(draft.currency))
    add(ids.currency, "Use a three-letter currency code, like USD.");

  (draft.constraints ?? []).forEach((constraint, index) => {
    const n = index + 1;
    if (blank(constraint.text))
      add(ids.constraint(index, "text"), `Question ${n}: write it out or remove it.`);
    else if (over(constraint.text, 2000))
      add(ids.constraint(index, "text"), `Question ${n}: shorten it to 2,000 characters.`);
    if (constraint.resolved && blank(constraint.resolution))
      add(ids.constraint(index, "resolution"), `Question ${n} is marked settled: say how.`);
    else if (over(constraint.resolution, 2000))
      add(
        ids.constraint(index, "resolution"),
        `Question ${n}: shorten how it was settled to 2,000 characters.`,
      );
  });
  const assumptions = (draft.assumptions ?? []).map((line) => line.trim()).filter(Boolean);
  if (assumptions.length > 100) add(ids.assumptions, "List up to 100 assumptions, one per line.");
  // A scenario's assumptions arrive as one entry of up to 5,000 characters.
  if (assumptions.some((line) => line.length > 2000))
    add(
      ids.assumptions,
      "Keep each assumption to 2,000 characters. Split a long one over several lines.",
    );
  return problems;
}

export interface ApprovalBlocker {
  kind: "stale" | "constraints" | "note";
  message: string;
  /** Fixed by revising the plan: the review offers its Revise button beside it. */
  revise?: boolean;
}

/**
 * Why Approve (a draft) or Schedule (an approved plan) is disabled, in the order to fix them.
 * Mirrors reviewable() in apps/api/app/services/land_actions.py, which refuses a stale revision
 * or an unresolved constraint, and LandActionReview, which requires the note.
 */
export function approvalBlockers(
  action: Pick<ActionRecord, "staleReasons" | "constraints">,
  note: string,
  mode: "approve" | "schedule",
): ApprovalBlocker[] {
  const blockers: ApprovalBlocker[] = [];
  if (action.staleReasons.length)
    blockers.push({
      kind: "stale",
      message:
        "Something this plan relies on changed since this revision. Revise the plan to check it against the land as it is now.",
      revise: true,
    });
  const open = (action.constraints ?? []).filter((constraint) => !constraint.resolved).length;
  if (open)
    blockers.push({
      kind: "constraints",
      message: `Settle ${open} open question${open > 1 ? "s" : ""}: revise the plan, tick Settled and say how.`,
      revise: true,
    });
  if (!note.trim())
    blockers.push({
      kind: "note",
      message:
        mode === "approve"
          ? "Add a review note. It is stored with the approval."
          : "Add a scheduling note. It is stored with the scheduled mission.",
    });
  return blockers;
}
