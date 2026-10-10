import type { components } from "@twin/contracts";
import { defaultImportMapping, inventoryCategories, parseInventoryImport } from "./inventoryImport";
import type { InventoryShape, Vertex } from "./inventoryGeometry";
export type AssetDraft = components["schemas"]["LandFeatureCreate"];
export type AssetRecord = components["schemas"]["LandFeatureRead"];
export interface InventoryDraftSnapshot {
  version: 1;
  landId: string;
  boundaryRevision: number;
  draft: AssetDraft;
  editing: AssetRecord | null;
  revisionNote: string;
  working: InventoryShape | null;
}
export function parseInventoryDraft(text: string, landId: string): InventoryDraftSnapshot {
  if (text.length > 4_000_000) throw new Error("This asset draft exceeds the recovery limit.");
  const value = JSON.parse(text) as InventoryDraftSnapshot | null;
  if (
    value?.version !== 1 ||
    value.landId !== landId ||
    !Number.isInteger(value.boundaryRevision) ||
    typeof value.revisionNote !== "string"
  )
    throw new Error("This asset draft cannot be recovered for the current land.");
  const validate = (draft: AssetDraft) => {
    if (
      !draft ||
      typeof draft.name !== "string" ||
      !inventoryCategories.includes(draft.category) ||
      !draft.source ||
      typeof draft.source.label !== "string" ||
      typeof draft.source.method !== "string" ||
      (draft.status !== undefined &&
        !["candidate", "confirmed", "retired"].includes(draft.status)) ||
      (draft.description !== undefined && typeof draft.description !== "string")
    )
      throw new Error("The saved asset details are invalid.");
    const geometry = parseInventoryImport(
      JSON.stringify(draft.geometry),
      "geojson",
      defaultImportMapping,
    )[0];
    if (!geometry?.geometry || geometry.error)
      throw new Error("The saved asset geometry is invalid.");
    if (
      draft.attributes &&
      (Array.isArray(draft.attributes) ||
        typeof draft.attributes !== "object" ||
        Object.entries(draft.attributes).some(
          ([key, v]) =>
            key.length > 100 ||
            !(
              v === null ||
              typeof v === "string" ||
              typeof v === "boolean" ||
              (typeof v === "number" && Number.isFinite(v))
            ),
        ))
    )
      throw new Error("The saved asset attributes are invalid.");
    if (
      draft.evidenceIds &&
      (!Array.isArray(draft.evidenceIds) || draft.evidenceIds.some((id) => typeof id !== "string"))
    )
      throw new Error("The saved evidence references are invalid.");
    return { ...draft, geometry: geometry.geometry };
  };
  value.draft = validate(value.draft);
  if (
    typeof value.draft.requestKey !== "string" ||
    !/^[a-f0-9-]{36}$/i.test(value.draft.requestKey)
  )
    throw new Error("The saved request key is invalid.");
  if (value.editing) {
    if (
      value.editing.landId !== landId ||
      typeof value.editing.id !== "string" ||
      !Number.isInteger(value.editing.revision) ||
      value.editing.revision < 1
    )
      throw new Error("The saved asset revision is invalid.");
    validate(value.editing);
  }
  if (value.working) {
    const working = value.working;
    if (
      !["Point", "LineString", "Polygon", "MultiPolygon"].includes(working.type) ||
      !Array.isArray(working.parts) ||
      !working.parts.length ||
      working.parts.length > 20000
    )
      throw new Error("The unfinished geometry is invalid.");
    let vertices = 0,
      rings = 0;
    working.parts = working.parts.map((part) => {
      if (!Array.isArray(part) || !part.length || (rings += part.length) > 20000)
        throw new Error("The unfinished geometry rings are invalid.");
      return part.map((ring) => {
        if (!Array.isArray(ring)) throw new Error("The unfinished geometry ring is invalid.");
        return ring.map((point): Vertex => {
          if (
            !Array.isArray(point) ||
            point.length !== 2 ||
            ++vertices > 20000 ||
            point.some((n: unknown) => n !== null && (typeof n !== "number" || !Number.isFinite(n)))
          )
            throw new Error("The unfinished geometry vertices are invalid.");
          return [point[0] ?? NaN, point[1] ?? NaN];
        });
      });
    });
  }
  return value;
}
export const assetFields = [
  "name",
  "category",
  "status",
  "description",
  "geometry",
  "source",
  "attributes",
  "evidenceIds",
  "externalRef",
] as const;
export type AssetField = (typeof assetFields)[number];
function normalized(value: AssetDraft): AssetDraft {
  return {
    ...value,
    status: value.status ?? "candidate",
    description: value.description ?? "",
    attributes: value.attributes ?? {},
    evidenceIds: value.evidenceIds ?? [],
    externalRef: value.externalRef ?? null,
    source: {
      ...value.source,
      meaning: value.source.meaning ?? "study-area",
      url: value.source.url ?? null,
      recordId: value.source.recordId ?? null,
      observedAt: value.source.observedAt ?? null,
      attribution: value.source.attribution ?? null,
    },
  };
}
function stable(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (value && typeof value === "object")
    return `{${Object.entries(value)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, v]) => `${JSON.stringify(key)}:${stable(v)}`)
      .join(",")}}`;
  return JSON.stringify(value) ?? "undefined";
}
export function mergeAssetDraft(
  baseline: AssetDraft,
  draft: AssetDraft,
  latest: AssetDraft,
  working = false,
): { merged: AssetDraft; conflicts: AssetField[] } {
  const before = normalized(baseline),
    mine = normalized(draft),
    current = normalized(latest);
  const conflicts: AssetField[] = [];
  const merged = { ...mine };
  for (const key of assetFields) {
    const changed = stable(mine[key]) !== stable(before[key]) || (key === "geometry" && working);
    if (!changed) Object.assign(merged, { [key]: current[key] });
    else if (
      stable(current[key]) !== stable(before[key]) &&
      (stable(current[key]) !== stable(mine[key]) || (key === "geometry" && working))
    )
      conflicts.push(key);
  }
  return { merged, conflicts };
}
export function downloadAssetDraft(value: unknown) {
  const url = URL.createObjectURL(
    new Blob([typeof value === "string" ? value : JSON.stringify(value, null, 2)], {
      type: "application/json",
    }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "asset-draft.json";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function sameAssetContent(a: AssetDraft, b: AssetDraft): boolean {
  const left = normalized(a),
    right = normalized(b);
  return assetFields.every((field) => stable(left[field]) === stable(right[field]));
}
