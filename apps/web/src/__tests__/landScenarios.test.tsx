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
