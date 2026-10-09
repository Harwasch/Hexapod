import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { LandArea } from "@twin/contracts";
import { api } from "@/api/client";
import { LandScenarios } from "@/features/land/LandScenarios";
import {
  restorationDefaults,
  solarDefaults,
  type Scenario,
  type ScenarioResult,
} from "@/features/land/scenarioDefaults";

const land: LandArea = {
  id: "land",
  name: "Ranch",
  description: "",
  revision: 1,
  areaM2: 10000,
  perimeterM: 400,
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 0],
      ],
    ],
  },
  source: { method: "drawn", label: "Drawn" },
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};
const result: ScenarioResult = {
  algorithm: "solar-cash-flow/1",
  summary: { capacityKwDc: 20, currency: "USD", netPresentValue: 1000 },
  rows: [],
  limitations: ["Assumed roof resource"],
};
const scenario: Scenario = {
  id: "scenario",
  landId: land.id,
  name: "Solar option",
  boundaryRevision: 1,
  revision: 1,
  inputs: solarDefaults(),
  result,
  stale: false,
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};
function mount() {
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={cache}>
      <LandScenarios land={land} />
    </QueryClientProvider>,
  );
}
afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
});

it("invalidates a preview when assumptions change and preserves the create request key", async () => {
  vi.spyOn(api, "GET").mockResolvedValue({ data: [], response: new Response() });
  const post = vi.spyOn(api, "POST").mockResolvedValue({ data: result, response: new Response() });
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Solar and economics" }));
  fireEvent.change(screen.getByLabelText("Usable module area after exclusions (m²)"), {
    target: { value: "100" },
  });
  fireEvent.change(screen.getByLabelText("Annual irradiation at the module plane (kWh/m²/year)"), {
    target: { value: "1000" },
  });
  fireEvent.change(screen.getByLabelText("Solar resource source and method"), {
    target: { value: "User supplied reference" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Calculate scenario" }));
  await screen.findByRole("region", { name: "Scenario result" });
  expect(screen.getByRole("button", { name: "Save scenario" })).toBeEnabled();
  fireEvent.change(screen.getByLabelText("Installed cost"), { target: { value: "12000" } });
  expect(screen.getByRole("button", { name: "Save scenario" })).toBeDisabled();
  expect(screen.queryByRole("region", { name: "Scenario result" })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Calculate scenario" }));
  await screen.findByRole("region", { name: "Scenario result" });
  const calls = post.mock.calls as unknown as [
    string,
    { body: { requestKey: string; inputs: { installedCost: number } } },
  ][];
  expect(calls[0]?.[1].body.requestKey).toBe(calls[1]?.[1].body.requestKey);
  expect(calls[1]?.[1].body.inputs.installedCost).toBe(12000);
});

it("compares saved values without recomputing or hiding a stale boundary", async () => {
  const restored: Scenario = {
    ...scenario,
    id: "restoration",
    name: "Meadow option",
    stale: true,
    inputs: restorationDefaults(),
    result: {
      algorithm: "restoration-cover-and-cost/1",
      summary: { totalCost: 5000, currency: "USD" },
      rows: [],
      limitations: [],
    },
  };
  vi.spyOn(api, "GET").mockImplementation((path) =>
    Promise.resolve({
      data: String(path).endsWith("/scenarios") ? [scenario, restored] : [],
      response: new Response(),
    }),
  );
  mount();
  await screen.findByRole("checkbox", { name: /Meadow option.*earlier boundary/ });
  fireEvent.click(screen.getByRole("checkbox", { name: /Solar option/ }));
  fireEvent.click(screen.getByRole("checkbox", { name: /Meadow option/ }));
  await waitFor(() =>
    expect(screen.getByRole("table", { name: /Scenario comparison/ })).toBeVisible(),
  );
  expect(screen.getAllByText("5,000").length).toBeGreaterThan(0);
});

it("pins field survey references without converting overlapping species into cover classes", async () => {
  vi.spyOn(api, "GET").mockImplementation((path) =>
    Promise.resolve({
      data: String(path).endsWith("/surveys")
        ? [{ id: "survey", name: "Measured plots", boundaryRevision: 1, observedOn: "2025-07-01" }]
        : [],
      response: new Response(),
    }),
  );
  const post = vi.spyOn(api, "POST").mockResolvedValue({ data: result, response: new Response() });
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Restoration and cover" }));
  fireEvent.click(await screen.findByRole("checkbox", { name: /Measured plots/ }));
  fireEvent.change(screen.getByLabelText("Reference ecosystem"), {
    target: { value: "Locally evaluated meadow reference" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Calculate scenario" }));
  await screen.findByRole("region", { name: "Scenario result" });
  const calls = post.mock.calls as unknown as [
    string,
    { body: { fieldSurveyIds: string[]; inputs: ReturnType<typeof restorationDefaults> } },
  ][];
  expect(calls[0]?.[0]).toBe("/api/v1/land/{land_id}/scenarios/preview");
  expect(calls[0]?.[1].body.fieldSurveyIds).toEqual(["survey"]);
  expect(calls[0]?.[1].body.inputs.cover).toEqual(restorationDefaults().cover);
});

it("recovers unfinished species targets, blank numbers and monitoring text without restoring a preview", async () => {
  vi.spyOn(api, "GET").mockResolvedValue({ data: [], response: new Response() });
  const first = mount();
  fireEvent.click(screen.getByRole("button", { name: "Restoration and cover" }));
  fireEvent.change(screen.getByLabelText("Scenario name"), {
    target: { value: "Recovered meadow" },
  });
  fireEvent.change(screen.getByLabelText("Monitoring years (comma separated)"), {
    target: { value: "1, 3," },
  });
  fireEvent.change(screen.getByLabelText("Cost per monitoring visit"), { target: { value: "" } });
  fireEvent.click(screen.getByRole("button", { name: "Add species targets and monitoring" }));
  first.unmount();
  vi.mocked(api.GET).mockImplementation((path) =>
    Promise.resolve(
      String(path).includes("/requests/")
        ? { error: { title: "Not saved" }, response: new Response(null, { status: 404 }) }
        : { data: [], response: new Response() },
    ),
  );
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Recover scenario draft" }));
  await screen.findByDisplayValue("Recovered meadow");
  expect(screen.getByLabelText("Monitoring years (comma separated)")).toHaveValue("1, 3,");
  expect(screen.getByLabelText("Cost per monitoring visit")).toHaveValue(null);
  expect(screen.getByRole("region", { name: "Species targets and monitoring" })).toBeVisible();
  expect(screen.getByRole("button", { name: "Save scenario" })).toBeDisabled();
  expect(screen.queryByRole("region", { name: "Scenario result" })).toBeNull();
});

it("reconciles a lost save against its original request even when the scenario was subsequently revised", async () => {
  const { scenarioDraftKey, serializeScenarioDraft } =
    await import("@/features/land/scenarioDraft");
  const requestKey = crypto.randomUUID();
  localStorage.setItem(
    scenarioDraftKey("pilot", land.id),
    serializeScenarioDraft({
      version: 1,
      landId: land.id,
      requestKey,
      editing: null,
      monitoringText: "1, 3, 5",
      payload: {
        name: scenario.name,
        boundaryRevision: 1,
        inputs: solarDefaults(),
        evidenceIds: [],
        fieldSurveyIds: [],
        solarAssessmentId: null,
      },
    }),
  );
  const current = { ...scenario, revision: 2, name: "Later user revision" };
  vi.spyOn(api, "GET").mockImplementation((path) =>
    Promise.resolve({
      data: String(path).includes("/requests/")
        ? { saved: scenario, current }
        : String(path).endsWith("/scenarios")
          ? [current]
          : [],
      response: new Response(),
    }),
  );
  const post = vi.spyOn(api, "POST");
  const put = vi.spyOn(api, "PUT");
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Recover scenario draft" }));
  await screen.findByText(
    "This draft was already saved. Opened the current scenario without creating another revision.",
  );
  expect(localStorage.getItem(scenarioDraftKey("pilot", land.id))).toBeNull();
  expect(post).not.toHaveBeenCalled();
  expect(put).not.toHaveBeenCalled();
  expect(screen.queryByLabelText("Scenario name")).toBeNull();
});

it("keeps a concurrent revision unchanged while recovering local assumptions as a separate option", async () => {
  const { scenarioDraftKey, serializeScenarioDraft } =
    await import("@/features/land/scenarioDraft");
  localStorage.setItem(
    scenarioDraftKey("pilot", land.id),
    serializeScenarioDraft({
      version: 1,
      landId: land.id,
      requestKey: crypto.randomUUID(),
      editing: { id: scenario.id, revision: 1, landId: land.id },
      monitoringText: "1, 3, 5",
      payload: {
        name: "Local assumptions",
        boundaryRevision: 1,
        inputs: solarDefaults(),
        evidenceIds: [],
        fieldSurveyIds: [],
        solarAssessmentId: null,
      },
    }),
  );
  vi.spyOn(api, "GET").mockImplementation((path) =>
    Promise.resolve(
      String(path).includes("/requests/")
        ? { error: {}, response: new Response(null, { status: 404 }) }
        : {
            data: String(path).endsWith("/{scenario_id}") ? { ...scenario, revision: 2 } : [],
            response: new Response(),
          },
    ),
  );
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Recover scenario draft" }));
  await screen.findByRole("region", { name: "Review newer scenario revision" });
  expect(screen.getByRole("button", { name: "Save scenario revision" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Keep my assumptions as a new scenario" }));
  expect(screen.getByLabelText("Scenario name")).toHaveValue("Local assumptions");
  expect(screen.queryByRole("region", { name: "Review newer scenario revision" })).toBeNull();
  expect(screen.getByRole("button", { name: "Save scenario" })).toBeDisabled();
});

it("retains an invalid recovery file for download instead of replacing it with a new draft", async () => {
  const { scenarioDraftKey } = await import("@/features/land/scenarioDraft");
  const key = scenarioDraftKey("pilot", land.id),
    invalid = JSON.stringify({ version: 1, landId: "different-land" });
  localStorage.setItem(key, invalid);
  vi.spyOn(api, "GET").mockResolvedValue({ data: [], response: new Response() });
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Recover scenario draft" }));
  await screen.findByRole("alert");
  expect(localStorage.getItem(key)).toBe(invalid);
  expect(screen.getByRole("button", { name: "Download scenario draft" })).toBeEnabled();
  expect(screen.queryByLabelText("Scenario name")).toBeNull();
});

it("keeps the current form editable and offers a download when browser storage is full", async () => {
  vi.spyOn(api, "GET").mockResolvedValue({ data: [], response: new Response() });
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new DOMException("Full", "QuotaExceededError");
  });
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Solar and economics" }));
  fireEvent.change(screen.getByLabelText("Scenario name"), { target: { value: "Keep my work" } });
  await screen.findByText(
    "This browser could not preserve the scenario draft. Download a copy before leaving.",
  );
  expect(screen.getByLabelText("Scenario name")).toHaveValue("Keep my work");
  expect(screen.getByRole("button", { name: "Download scenario draft" })).toBeEnabled();
});
