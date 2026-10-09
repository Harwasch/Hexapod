import type { components } from "@twin/contracts";
export type BoundaryReference = components["schemas"]["BoundarySourceReference"];

export function parseBoundaryReferences(raw: unknown): BoundaryReference[] {
  if (!Array.isArray(raw) || raw.length > 100)
    throw new Error("Use at most 100 drawing-guide sources.");
  return raw.map((value: unknown) => {
    if (!value || typeof value !== "object") throw new Error("Invalid drawing-guide source.");
    const row = value as Record<string, unknown>;
    if (
      typeof row.method !== "string" ||
      !["drawn", "imported", "parcel", "mapped-feature", "imagery", "corridor"].includes(
        row.method,
      ) ||
      typeof row.label !== "string" ||
      !row.label ||
      row.label.length > 300
    )
      throw new Error("Invalid drawing-guide source.");
    for (const [key, limit] of [
      ["url", 10000],
      ["recordId", 300],
      ["attribution", 2000],
      ["observedAt", 100],
    ] as const)
      if (row[key] != null && (typeof row[key] !== "string" || row[key].length > limit))
        throw new Error("Invalid drawing-guide source metadata.");
    if (
      (row.url != null && (typeof row.url !== "string" || !/^https?:\/\//i.test(row.url))) ||
      (row.observedAt != null &&
        (typeof row.observedAt !== "string" || !Number.isFinite(Date.parse(row.observedAt)))) ||
      (row.meaning != null &&
        (typeof row.meaning !== "string" ||
          !["study-area", "recorded-parcel", "physical-feature"].includes(row.meaning)))
    )
      throw new Error("Invalid drawing-guide source locator or meaning.");
    return {
      method: row.method,
      label: row.label,
      url: row.url ?? null,
      recordId: row.recordId ?? null,
      attribution: row.attribution ?? null,
      observedAt: row.observedAt ?? null,
      meaning: row.meaning ?? "study-area",
    } as BoundaryReference;
  });
}
export function drawingReferences(
  sources: components["schemas"]["BoundarySource"][],
): BoundaryReference[] {
  const flat = sources.flatMap((source) => [source, ...(source.references ?? [])]);
  return parseBoundaryReferences([
    ...new Map(
      flat.map((source) => {
        const { method, label, url, recordId, attribution, observedAt, meaning } = source;
        const reference = { method, label, url, recordId, attribution, observedAt, meaning };
        return [JSON.stringify(reference), reference];
      }),
    ).values(),
  ]);
}
