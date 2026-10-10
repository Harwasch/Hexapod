import type { components } from "@twin/contracts";
import { importBoundary } from "./geometry";

type Draft = components["schemas"]["SurveyCreate"];
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Invalid survey record.");
  return value as Record<string, unknown>;
}
const text = (v: unknown, max: number) => typeof v === "string" && v.length <= max;
const numeric = (v: unknown) => v == null || (typeof v === "number" && Number.isFinite(v));
const strata = ["canopy", "shrub", "herb", "ground", "aquatic"];
/** Accept unfinished values but reject malformed structures before rendering private drafts. */
export function parseSurveyDraft(raw: string): Draft {
  if (raw.length > 2_000_000) throw new Error("Survey files are limited to 2 MB.");
  const v = object(JSON.parse(raw));
  if (
    !text(v.name, 200) ||
    !text(v.observer, 200) ||
    !text(v.observedOn, 10) ||
    !text(v.methodNotes, 5000) ||
    !Number.isInteger(v.boundaryRevision) ||
    !text(v.requestKey, 36) ||
    !["visual-cover", "point-intercept"].includes(String(v.method)) ||
    !["census", "random", "systematic", "purposive"].includes(String(v.design)) ||
    !Array.isArray(v.assessedStrata) ||
    v.assessedStrata.length > 5 ||
    !v.assessedStrata.every((s) => strata.includes(String(s))) ||
    !Array.isArray(v.plots) ||
    v.plots.length > 100
  )
    throw new Error("The survey draft has invalid fields.");
  for (const value of v.plots) {
    const p = object(value);
    if (
      !text(p.label, 100) ||
      !numeric(p.samplePoints) ||
      (p.completeInventory != null && typeof p.completeInventory !== "boolean") ||
      !Array.isArray(p.observations) ||
      p.observations.length > 200
    )
      throw new Error("The survey draft has an invalid plot.");
    importBoundary(JSON.stringify(p.boundary));
    for (const value of p.observations) {
      const o = object(value);
      if (
        !text(o.taxon, 200) ||
        !strata.includes(String(o.stratum)) ||
        !["verified", "tentative", "unidentified"].includes(String(o.identification)) ||
        !numeric(o.percentCover) ||
        !numeric(o.hits) ||
        (o.notes != null && !text(o.notes, 1000))
      )
        throw new Error("The survey draft has an invalid observation.");
    }
  }
  // A saved export also contains server metadata; send only writable survey fields.
  return {
    requestKey: v.requestKey,
    name: v.name,
    boundaryRevision: v.boundaryRevision,
    observedOn: v.observedOn,
    observer: v.observer,
    method: v.method,
    design: v.design,
    assessedStrata: v.assessedStrata,
    methodNotes: v.methodNotes,
    plots: v.plots,
    supersedesId: v.supersedesId ?? null,
  } as unknown as Draft;
}
export function parseSurveyCorners(raw: string | null): number[][] {
  if (!raw || raw.length > 100000) return [];
  try {
    const value: unknown = JSON.parse(raw);
    if (
      Array.isArray(value) &&
      value.length <= 1000 &&
      value.every(
        (p) =>
          Array.isArray(p) &&
          p.length === 2 &&
          p.every(Number.isFinite) &&
          typeof p[0] === "number" &&
          typeof p[1] === "number" &&
          Math.abs(p[0]) <= 180 &&
          Math.abs(p[1]) <= 90,
      )
    )
      return value as number[][];
  } catch {
    /* Keep the valid survey even if a sketch was damaged. */
  }
  return [];
}
