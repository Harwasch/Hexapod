import type { components, LandArea } from "@twin/contracts";

export type Action = components["schemas"]["LandActionRead"];
export type ActionDraft = components["schemas"]["LandActionCreate"];
export type Step = components["schemas"]["LandActionStep"];
export type Scenario = components["schemas"]["ScenarioRead"];
export type Area = components["schemas"]["Polygon"] | components["schemas"]["MultiPolygon"];

/** One colour per step, on the map and beside the step in the card. */
export const STEP_COLORS = [
  "#c7a6f0",
  "#7fd8c0",
  "#f0b254",
  "#8fbcf2",
  "#ef9ac4",
  "#b9e07a",
  "#f2d16b",
  "#9db8f0",
] as const;
/** Areas the work must stay out of. */
export const AVOID_COLOR = "#e8806f";
/** The plan's work area outline: the lens colour. */
export const WORK_AREA_COLOR = "#ece6f7";

export function stepColor(index: number): string {
  return STEP_COLORS[index % STEP_COLORS.length] ?? STEP_COLORS[0];
}

export function newStep(startDay: number): Step {
  return {
    id: crypto.randomUUID().slice(0, 8),
    title: "",
    detail: "",
    startDay,
    days: 1,
    successMeasure: "",
  };
}

/** A saved plan as an editable draft (the server's read-only fields left out). */
export function editablePlan(action: Action): ActionDraft {
  return {
    requestKey: crypto.randomUUID(),
    title: action.title,
    objective: action.objective,
    boundaryRevision: action.boundaryRevision,
    scenario: action.scenario,
    features: action.features,
    evidenceIds: action.evidenceIds,
    startDate: action.startDate,
    currency: action.currency,
    steps: action.steps,
    constraints: action.constraints,
    exclusions: action.exclusions,
    assumptions: action.assumptions,
  };
}

export function freshPlan(land: LandArea): ActionDraft {
  return {
    requestKey: crypto.randomUUID(),
    title: "",
    objective: "",
    boundaryRevision: land.revision,
    startDate: new Date().toISOString().slice(0, 10),
    currency: "USD",
    steps: [newStep(0)],
    constraints: [],
    evidenceIds: [],
    features: [],
    exclusions: [],
    assumptions: [],
  };
}

/** A plan seeded from a scenario: its treatments and monitoring visits become steps. */
export function planFromScenario(land: LandArea, scenario: Scenario): ActionDraft {
  const draft = freshPlan(land);
  draft.title = scenario.name;
  draft.scenario = { id: scenario.id, revision: scenario.revision };
  draft.boundaryRevision = scenario.boundaryRevision;
  draft.evidenceIds = scenario.evidenceIds;
  draft.currency = scenario.inputs.currency;
  draft.assumptions = [scenario.inputs.assumptions];
  const inputs = scenario.inputs;
  if (inputs.kind === "restoration") {
    draft.objective = `Restore toward ${inputs.referenceEcosystem}`;
    draft.steps = inputs.treatments.map((treatment, index) => ({
      ...newStep(index),
      title: treatment.name,
      detail: treatment.objective,
      startDay: treatment.year * 365,
      days: 1,
      estimatedCost: treatment.areaHa * treatment.costPerHa,
      costBasis:
        "Area × cost per hectare from the selected scenario; review duration and scheduling.",
      successMeasure: treatment.objective,
    }));
    draft.steps.push(
      ...inputs.monitoringYears.map((year, index) => ({
        ...newStep(index),
        title: `Monitor restoration in year ${year}`,
        startDay: year * 365,
        days: 1,
        estimatedCost: inputs.monitoringCostPerVisit,
        costBasis: "Monitoring cost per visit from the selected restoration scenario.",
        successMeasure:
          "Repeat the recorded survey method and compare measured cover with scenario targets.",
      })),
    );
    draft.assumptions.push(
      "Scenario years are converted to 365-day intervals. One-day step durations are placeholders for review before approval.",
    );
    if (!draft.steps.length) draft.steps = [newStep(0)];
    draft.constraints = [
      {
        text: "Confirm treatment footprints, timing, permissions and monitoring with site evidence.",
        resolved: false,
      },
    ];
  } else {
    draft.objective = "Evaluate and deliver the selected solar option";
    draft.steps = [
      {
        ...newStep(0),
        title: "Site and roof assessment",
        successMeasure: "Verify roof capacity, usable area, shading and electrical connection.",
      },
      {
        ...newStep(1),
        title: "Installation proposal",
        startDay: 1,
        successMeasure: "Review a site-specific design and contractor quote.",
      },
    ];
    draft.constraints = [
      {
        text: "Verify roof structure, design, permits and connection requirements before installation.",
        resolved: false,
      },
    ];
  }
  return draft;
}

/** Saves a JSON copy through the browser's download prompt. */
export function downloadJson(value: unknown, name: string): void {
  const url = URL.createObjectURL(
    new Blob([typeof value === "string" ? value : JSON.stringify(value, null, 2)], {
      type: "application/json",
    }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
