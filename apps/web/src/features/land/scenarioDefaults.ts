import type { components } from "@twin/contracts";
export type SolarInputs = components["schemas"]["SolarInputs"];
export type RestorationInputs = components["schemas"]["RestorationInputs"];
export type ScenarioInputs = SolarInputs | RestorationInputs;
export type Scenario = components["schemas"]["ScenarioRead"];
export type ScenarioResult = components["schemas"]["ScenarioResult"];

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
export function restorationDefaults(): RestorationInputs {
  return {
    kind: "restoration",
    referenceEcosystem: "",
    surveyDate: new Date().toISOString().slice(0, 10),
    surveyMethod: "User estimate for scenario planning",
    confidence: "user-estimate",
    cover: [
      {
        name: "Unclassified",
        baselinePercent: 100,
        targetPercent: 0,
        evidenceBasis: "Not yet surveyed",
      },
      {
        name: "Reference vegetation",
        baselinePercent: 0,
        targetPercent: 100,
        evidenceBasis: "Proposed target; verify locally",
      },
    ],
    treatments: [],
    monitoringYears: [1, 3, 5],
    monitoringCostPerVisit: 0,
    contingencyFraction: 0.15,
    discountRate: 0.04,
    currency: "USD",
    assumptions: "Scenario targets require field assessment and local ecological advice.",
  };
}
export const solarGroups: {
  title: string;
  fields: { key: keyof SolarInputs; label: string; percent?: boolean }[];
}[] = [
  {
    title: "Roof and solar resource",
    fields: [
      { key: "usableRoofAreaM2", label: "Usable module area after exclusions (m²)" },
      {
        key: "annualPlaneIrradiationKwhM2",
        label: "Annual irradiation at the module plane (kWh/m²/year)",
      },
      { key: "tiltDegrees", label: "Module tilt (degrees)" },
      { key: "azimuthDegrees", label: "Azimuth (degrees clockwise from north)" },
      { key: "moduleEfficiency", label: "Module efficiency (%)", percent: true },
      { key: "shadeLoss", label: "Shade loss (%)", percent: true },
      { key: "systemLoss", label: "Other system losses (%)", percent: true },
      { key: "degradationPerYear", label: "Annual degradation (%)", percent: true },
    ],
  },
  {
    title: "Energy value and costs",
    fields: [
      { key: "selfConsumptionFraction", label: "Generation used on site (%)", percent: true },
      { key: "purchaseRatePerKwh", label: "Avoided electricity price (currency/kWh)" },
      { key: "exportRatePerKwh", label: "Export payment (currency/kWh)" },
      { key: "tariffEscalation", label: "Annual tariff change (%)", percent: true },
      { key: "installedCost", label: "Installed cost" },
      { key: "upfrontIncentive", label: "Upfront incentive" },
      { key: "annualMaintenanceCost", label: "Annual maintenance cost" },
      { key: "maintenanceEscalation", label: "Annual maintenance cost change (%)", percent: true },
      { key: "replacementCost", label: "Replacement cost" },
      { key: "replacementYear", label: "Replacement year (blank if none)" },
    ],
  },
  {
    title: "Financing and horizon",
    fields: [
      { key: "years", label: "Years to model" },
      { key: "discountRate", label: "Annual discount rate (%)", percent: true },
      { key: "financedFraction", label: "Net cost financed (%)", percent: true },
      { key: "loanInterestRate", label: "Annual loan interest (%)", percent: true },
      { key: "loanYears", label: "Loan term (years)" },
    ],
  },
];

const labels: Record<string, string> = {
  capacityKwDc: "DC capacity (kW)",
  firstYearGenerationKwh: "First-year generation (kWh)",
  netPresentValue: "Net present value",
  initialEquity: "Initial equity",
  annualLoanPayment: "Annual loan payment",
  paybackYear: "Equity recovery year",
  equityRecoveryYear: "Equity recovery year",
  lifetimeNetCashFlow: "Lifetime net cash flow",
  areaHa: "Area (ha)",
  costPerHa: "Cost per hectare",
  presentValueCost: "Present value of costs",
  totalCost: "Total cost",
  baselinePercent: "Current cover (%)",
  targetPercent: "Target cover (%)",
  baselineHa: "Current area (ha)",
  targetHa: "Target area (ha)",
  changeHa: "Area change (ha)",
  unclassifiedBaselinePercent: "Unclassified current cover (%)",
  generationKwh: "Generation (kWh)",
  energyValue: "Energy value",
  netCashFlow: "Net cash flow",
  cumulativeCashFlow: "Cumulative cash flow",
  discountedCashFlow: "Discounted cash flow",
  maintenanceCost: "Maintenance cost",
  debtService: "Loan payment",
  replacementCost: "Replacement cost",
};
export const fieldLabel = (key: string) =>
  labels[key] ??
  key.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/^./, (letter) => letter.toUpperCase());
export const valueLabel = (value: number | string | null | undefined) =>
  typeof value === "number"
    ? value.toLocaleString(undefined, { maximumFractionDigits: 2 })
    : value === undefined
      ? "—"
      : (value ?? "Not reached");
