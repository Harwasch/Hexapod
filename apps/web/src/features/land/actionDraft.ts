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
