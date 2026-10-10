import type { components } from "@twin/contracts";
import {
  restorationDefaults,
  solarDefaults,
  type Scenario,
  type ScenarioInputs,
} from "./scenarioDefaults";

export type ScenarioPayload = components["schemas"]["ScenarioCreate"];
export type ScenarioEdit = Pick<Scenario, "id" | "revision" | "landId">;
export interface ScenarioDraft {
  version: 1;
  landId: string;
  requestKey: string;
  payload: ScenarioPayload;
  editing: ScenarioEdit | null;
  monitoringText: string;
}
export const scenarioDraftKey = (scope: string, landId: string) =>
  `living-world-land-draft:${encodeURIComponent(scope)}:scenario:${landId}`;
export function serializeScenarioDraft(value: unknown): string {
  return JSON.stringify(value, (_key, item: unknown) =>
    typeof item === "number" && !Number.isFinite(item) ? { __scenarioNumber: "NaN" } : item,
  );
}
type Shape =
  "text" | "number" | "nullable-number" | "nullable-text" | [Shape] | { [key: string]: Shape };
const textList: Shape = ["text"];
const responseShape: Shape = {
  startYear: "number",
  asymptoteLow: "number",
  asymptoteHigh: "number",
  annualRateLow: "number",
  annualRateHigh: "number",
  basis: "text",
  evidenceIds: textList,
};
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("The saved scenario has invalid fields.");
  return value as Record<string, unknown>;
}
function check(value: unknown, shape: Shape): void {
  if (Array.isArray(shape)) {
    if (!Array.isArray(value) || value.length > 1000)
      throw new Error("The saved scenario list is invalid.");
    value.forEach((item) => check(item, shape[0]));
  } else if (typeof shape === "object") {
    const row = object(value);
    for (const [key, field] of Object.entries(shape)) check(row[key], field);
  } else if (!(
    (shape.startsWith("nullable") && value === null) ||
    ((shape === "text" || shape === "nullable-text") && typeof value === "string") ||
    ((shape === "number" || shape === "nullable-number") &&
      typeof value === "number" &&
      (Number.isFinite(value) || Number.isNaN(value)))
  ))
    throw new Error("The saved scenario field types are invalid.");
}
function inputShape(defaults: ScenarioInputs): Record<string, Shape> {
  return Object.fromEntries(
    Object.entries(defaults)
      .filter(([, value]) => !Array.isArray(value))
      .map(([key, value]) => [
        key,
        value === null ? "nullable-number" : typeof value === "number" ? "number" : "text",
      ]),
  );
}
function validateInputs(raw: unknown): ScenarioInputs {
  const value = object(raw);
  if (value.kind === "solar") {
    check(value, inputShape(solarDefaults()));
  } else if (value.kind === "restoration") {
    check(value, {
      ...inputShape(restorationDefaults()),
      cover: [
        { name: "text", baselinePercent: "number", targetPercent: "number", evidenceBasis: "text" },
      ],
      treatments: [
        { name: "text", objective: "text", areaHa: "number", costPerHa: "number", year: "number" },
      ],
      monitoringYears: ["number"],
    });
    if (value.ecology != null) {
      const plan = object(value.ecology);
      check(plan, {
        referenceBasis: "text",
        siteConstraints: "text",
        referenceEvidenceIds: textList,
        speciesTargets: [
          {
            taxon: "text",
            stratum: "text",
            baselineSurveyId: "nullable-text",
            baselinePercent: "nullable-number",
            baselineBasis: "text",
            targetLow: "number",
            targetHigh: "number",
            targetYear: "number",
            targetBasis: "text",
            rationale: "text",
            evidenceIds: textList,
            treatmentNames: textList,
            monitoringMethod: "text",
            monitoringSeason: "text",
            responseIfOffTrack: "text",
          },
        ],
      });
      for (const target of plan.speciesTargets as Record<string, unknown>[])
        if (target.response != null) check(target.response, responseShape);
    }
  } else throw new Error("The saved scenario type is unsupported.");
  return value as ScenarioInputs;
}
export function parseScenarioDraft(text: string, landId: string): ScenarioDraft {
  if (text.length > 2_000_000) throw new Error("This scenario draft exceeds the recovery limit.");
  const value = object(
    JSON.parse(text, (_key, item: unknown) =>
      item &&
      typeof item === "object" &&
      Object.keys(item).length === 1 &&
      (item as Record<string, unknown>).__scenarioNumber === "NaN"
        ? NaN
        : item,
    ),
  );
  if (
    value.version !== 1 ||
    value.landId !== landId ||
    typeof value.monitoringText !== "string" ||
    typeof value.requestKey !== "string" ||
    !/^[a-f0-9-]{36}$/i.test(value.requestKey)
  )
    throw new Error("This scenario draft cannot be recovered for the current land.");
  const payload = object(value.payload);
  check(payload, {
    name: "text",
    boundaryRevision: "number",
    evidenceIds: textList,
    fieldSurveyIds: textList,
    solarAssessmentId: "nullable-text",
  });
  if (!Number.isInteger(payload.boundaryRevision) || Number(payload.boundaryRevision) < 1)
    throw new Error("The saved boundary revision is invalid.");
  validateInputs(payload.inputs);
  if (value.editing != null) {
    const editing = object(value.editing);
    if (
      editing.landId !== landId ||
      typeof editing.id !== "string" ||
      !Number.isInteger(editing.revision) ||
      Number(editing.revision) < 1
    )
      throw new Error("The saved scenario revision is invalid.");
  }
  return value as unknown as ScenarioDraft;
}
export function scenarioPayload(value: ScenarioPayload): ScenarioPayload {
  return {
    name: value.name,
    boundaryRevision: value.boundaryRevision,
    inputs: value.inputs,
    evidenceIds: value.evidenceIds ?? [],
    fieldSurveyIds: value.fieldSurveyIds ?? [],
    solarAssessmentId: value.solarAssessmentId ?? null,
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
export function sameScenario(a: ScenarioPayload, b: ScenarioPayload): boolean {
  const normalized = (value: ScenarioPayload) => {
    const inputs =
      value.inputs.kind === "solar"
        ? { ...solarDefaults(), ...value.inputs }
        : { ...restorationDefaults(), ...value.inputs, ecology: value.inputs.ecology ?? null };
    return stable({ ...scenarioPayload(value), inputs });
  };
  return serializeScenarioDraft(normalized(a)) === serializeScenarioDraft(normalized(b));
}
export function downloadScenarioDraft(value: unknown) {
  const url = URL.createObjectURL(
    new Blob([typeof value === "string" ? value : serializeScenarioDraft(value)], {
      type: "application/json",
    }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "scenario-draft.json";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
