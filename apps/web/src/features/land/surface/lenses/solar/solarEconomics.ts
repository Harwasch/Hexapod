import type { components } from "@twin/contracts";

import type { SolarAssessment } from "./solarDraft";

export type SolarInputs = components["schemas"]["SolarInputs"];
export type Scenario = components["schemas"]["ScenarioRead"];
export type ScenarioResult = components["schemas"]["ScenarioResult"];
export type ScenarioCreate = components["schemas"]["ScenarioCreate"];
export type ScenarioEdit = Pick<Scenario, "id" | "revision" | "landId">;

/** Planning values for a solar cash-flow scenario; every one is an editable assumption. */
export function solarDefaults(): SolarInputs {
  return {
    kind: "solar",
    usableRoofAreaM2: 0,
    moduleEfficiency: 0.2,
    annualPlaneIrradiationKwhM2: 0,
    irradiationBasis: "",
    tiltDegrees: 0,
    azimuthDegrees: 180,
    shadeLoss: 0.1,
    systemLoss: 0.14,
    degradationPerYear: 0.005,
    selfConsumptionFraction: 0.8,
    purchaseRatePerKwh: 0.2,
    exportRatePerKwh: 0.05,
    tariffEscalation: 0.02,
    installedCost: 0,
    upfrontIncentive: 0,
    annualMaintenanceCost: 0,
    maintenanceEscalation: 0.02,
    discountRate: 0.06,
    financedFraction: 0,
    loanInterestRate: 0.06,
    loanYears: 10,
    years: 25,
    replacementYear: null,
    replacementCost: 0,
    currency: "USD",
    assumptions:
      "Planning assumptions. Verify usable roof area, resource, costs and tariffs before making a decision.",
  };
}

/** The array and loss values a linked assessment pins; the server rejects any change to them. */
export const physicalFields = new Set<keyof SolarInputs>([
  "usableRoofAreaM2",
  "moduleEfficiency",
  "annualPlaneIrradiationKwhM2",
  "tiltDegrees",
  "azimuthDegrees",
  "shadeLoss",
  "systemLoss",
]);

/** Costs-and-payback inputs from a complete hourly assessment: its physics, default money. */
export function solarFinance(assessment: SolarAssessment): SolarInputs {
  if (!assessment.metadata.completeYear || assessment.metadata.annualGenerationKwh == null)
    throw new Error("A complete weather year is required for annual economics.");
  const request = assessment.request;
  return {
    ...solarDefaults(),
    usableRoofAreaM2: request.moduleAreaM2,
    moduleEfficiency: request.moduleEfficiency,
    annualPlaneIrradiationKwhM2: assessment.metadata.unshadedPlaneIrradiationKwhM2,
    tiltDegrees: request.tiltDegrees,
    azimuthDegrees: request.azimuthDegrees,
    shadeLoss: request.additionalShadeLoss,
    systemLoss: request.systemLoss,
    irradiationBasis: `NASA POWER ${request.year}; saved hourly AC generation includes orientation, temperature, horizon and inverter effects.`,
    assumptions: request.assumptions,
  };
}

type NumericInput = {
  [K in keyof SolarInputs]-?: NonNullable<SolarInputs[K]> extends number ? K : never;
}[keyof SolarInputs];

export interface MoneyField {
  key: NumericInput;
  label: string;
  percent?: boolean;
  min?: number;
  max?: number;
  /** Whole numbers only (years). */
  whole?: boolean;
  optional?: boolean;
}

/** The few values that decide payback, shown first. */
export const MAIN_FIELDS: MoneyField[] = [
  { key: "installedCost", label: "Installed cost", min: 0 },
  { key: "purchaseRatePerKwh", label: "Electricity price you avoid (per kWh)", min: 0 },
  { key: "exportRatePerKwh", label: "Payment for exported power (per kWh)", min: 0 },
  {
    key: "selfConsumptionFraction",
    label: "Power used on site (%)",
    percent: true,
    min: 0,
    max: 1,
  },
];

/** Everything else, grouped the way the old scenario form grouped it. */
export const MORE_FIELDS: { title: string; fields: MoneyField[] }[] = [
  {
    title: "Array and sunlight",
    fields: [
      { key: "usableRoofAreaM2", label: "Panel area (m²)", min: 0 },
      {
        key: "annualPlaneIrradiationKwhM2",
        label: "Yearly sunlight on the panels (kWh/m²)",
        min: 0,
        max: 5000,
      },
      { key: "tiltDegrees", label: "Tilt (degrees)", min: 0, max: 90 },
      {
        key: "azimuthDegrees",
        label: "Facing, clockwise from north (degrees)",
        min: 0,
        max: 359.999,
      },
      { key: "moduleEfficiency", label: "Panel efficiency (%)", percent: true, min: 0, max: 0.5 },
      { key: "shadeLoss", label: "Shade loss (%)", percent: true, min: 0, max: 1 },
      { key: "systemLoss", label: "Other losses (%)", percent: true, min: 0, max: 1 },
      {
        key: "degradationPerYear",
        label: "Yearly panel wear (%)",
        percent: true,
        min: 0,
        max: 0.1,
      },
    ],
  },
  {
    title: "Prices and running costs",
    fields: [
      {
        key: "tariffEscalation",
        label: "Yearly price change (%)",
        percent: true,
        min: -0.1,
        max: 0.2,
      },
      { key: "upfrontIncentive", label: "Up-front incentive", min: 0 },
      { key: "annualMaintenanceCost", label: "Yearly upkeep", min: 0 },
      {
        key: "maintenanceEscalation",
        label: "Yearly upkeep change (%)",
        percent: true,
        min: -0.1,
        max: 0.2,
      },
      { key: "replacementCost", label: "Replacement cost", min: 0 },
      {
        key: "replacementYear",
        label: "Replacement year (blank if none)",
        min: 1,
        max: 40,
        whole: true,
        optional: true,
      },
    ],
  },
  {
    title: "Financing and time",
    fields: [
      { key: "years", label: "Years to model", min: 1, max: 40, whole: true },
      { key: "discountRate", label: "Discount rate (%)", percent: true, min: 0, max: 0.5 },
      { key: "financedFraction", label: "Share financed (%)", percent: true, min: 0, max: 1 },
      { key: "loanInterestRate", label: "Loan interest (%)", percent: true, min: 0, max: 0.5 },
      { key: "loanYears", label: "Loan term (years)", min: 1, max: 40, whole: true },
    ],
  },
];

const ALL_FIELDS = [...MAIN_FIELDS, ...MORE_FIELDS.flatMap((group) => group.fields)];

/** The first value that stops a calculation, in words, or null. Mirrors the server's rules. */
export function economicsProblem(payload: ScenarioCreate): string | null {
  const inputs = payload.inputs;
  if (inputs.kind !== "solar") return "Only solar costs are worked out here.";
  if (!payload.name.trim()) return "Give these costs a name.";
  for (const field of ALL_FIELDS) {
    const value = inputs[field.key];
    if (value == null && field.optional) continue;
    if (typeof value !== "number" || !Number.isFinite(value))
      return `Fill in ${field.label.replace(/ \(.*\)$/, "").toLowerCase()}.`;
    if (
      (field.min !== undefined && value < field.min) ||
      (field.max !== undefined && value > field.max) ||
      (field.whole && value % 1 !== 0)
    )
      return `Check ${field.label.replace(/ \(.*\)$/, "").toLowerCase()}.`;
  }
  if (!(inputs.usableRoofAreaM2 > 0)) return "Enter the panel area.";
  if (!(inputs.moduleEfficiency > 0)) return "Enter the panel efficiency.";
  if (inputs.upfrontIncentive > inputs.installedCost)
    return "The incentive cannot be more than the installed cost.";
  if (inputs.financedFraction > 0 && inputs.loanYears > inputs.years)
    return "Model at least the whole loan term.";
  if (inputs.replacementYear != null && inputs.replacementYear > inputs.years)
    return "The replacement year must fall within the years modeled.";
  if (!/^[A-Z]{3}$/.test(inputs.currency)) return "Use a three-letter currency code, like USD.";
  if (!inputs.irradiationBasis.trim()) return "Say where the sunlight figure came from.";
  if (!inputs.assumptions.trim()) return "Add a note about the assumptions.";
  return null;
}

/* ---- Drafts kept in this browser (`…:solar-scenario:<landId>`) ---------------------- */

export interface EconomicsDraft {
  version: 1;
  landId: string;
  requestKey: string;
  payload: ScenarioCreate;
  editing: ScenarioEdit | null;
}

export const economicsDraftKey = (scope: string, landId: string) =>
  `living-world-land-draft:${encodeURIComponent(scope)}:solar-scenario:${landId}`;

/** JSON that keeps blank (NaN) numbers blank. */
export function serializeEconomics(value: unknown): string {
  return JSON.stringify(value, (_key, item: unknown) =>
    typeof item === "number" && !Number.isFinite(item) ? { __scenarioNumber: "NaN" } : item,
  );
}

function record(value: unknown, message: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(message);
  return value as Record<string, unknown>;
}

export function parseEconomicsDraft(text: string, landId: string): EconomicsDraft {
  if (text.length > 2_000_000) throw new Error("This draft is larger than a browser can keep.");
  const value = record(
    JSON.parse(text, (_key, item: unknown) =>
      item &&
      typeof item === "object" &&
      Object.keys(item).length === 1 &&
      (item as Record<string, unknown>).__scenarioNumber === "NaN"
        ? Number.NaN
        : item,
    ),
    "The saved costs are damaged.",
  );
  if (
    value.version !== 1 ||
    value.landId !== landId ||
    typeof value.requestKey !== "string" ||
    !/^[a-f0-9-]{36}$/i.test(value.requestKey)
  )
    throw new Error("The saved costs belong to other land.");
  const payload = record(value.payload, "The saved costs are damaged.");
  if (
    typeof payload.name !== "string" ||
    !Number.isInteger(payload.boundaryRevision) ||
    Number(payload.boundaryRevision) < 1 ||
    !(payload.solarAssessmentId === null || typeof payload.solarAssessmentId === "string")
  )
    throw new Error("The saved costs are damaged.");
  const inputs = record(payload.inputs, "The saved costs are damaged.");
  if (inputs.kind !== "solar") throw new Error("The saved costs are not for solar.");
  for (const [key, expected] of Object.entries(solarDefaults())) {
    const actual = inputs[key];
    if (expected === null) {
      if (actual !== null && actual !== undefined && typeof actual !== "number")
        throw new Error("The saved costs are damaged.");
    } else if (typeof actual !== typeof expected) throw new Error("The saved costs are damaged.");
  }
  if (value.editing != null) {
    const editing = record(value.editing, "The saved costs are damaged.");
    if (
      editing.landId !== landId ||
      typeof editing.id !== "string" ||
      !Number.isInteger(editing.revision) ||
      Number(editing.revision) < 1
    )
      throw new Error("The saved costs are damaged.");
  }
  return value as unknown as EconomicsDraft;
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

/** The writable part of a scenario, with defaults filled in, so two can be compared. */
function normalized(value: ScenarioCreate) {
  return stable({
    name: value.name,
    boundaryRevision: value.boundaryRevision,
    inputs: { ...solarDefaults(), ...value.inputs },
    evidenceIds: value.evidenceIds ?? [],
    fieldSurveyIds: value.fieldSurveyIds ?? [],
    solarAssessmentId: value.solarAssessmentId ?? null,
  });
}

export function sameScenario(a: ScenarioCreate, b: ScenarioCreate): boolean {
  return serializeEconomics(normalized(a)) === serializeEconomics(normalized(b));
}

export function downloadJson(value: unknown, filename: string): void {
  const url = URL.createObjectURL(
    new Blob([typeof value === "string" ? value : serializeEconomics(value)], {
      type: "application/json",
    }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** A saved scenario's headline numbers, read defensively from its summary. */
export function scenarioNumbers(result: ScenarioResult) {
  const summary = result.summary;
  const number = (key: string) => {
    const value = summary[key];
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  };
  return {
    payback: summary.paybackYear ?? null,
    netPresentValue: number("netPresentValue"),
    firstYearKwh: number("firstYearGenerationKwh"),
    lifetime: number("lifetimeNetCashFlow"),
    equity: number("initialEquity"),
    loanPayment: number("annualLoanPayment"),
    capacityKw: number("capacityKwDc"),
    currency: typeof summary.currency === "string" ? summary.currency : "USD",
  };
}
