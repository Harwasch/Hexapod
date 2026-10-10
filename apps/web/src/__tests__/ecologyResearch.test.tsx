import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components, LandArea } from "@twin/contracts";
import { api } from "@/api/client";
import { EcologyResearch } from "@/features/land/EcologyResearch";
import { ecologyNames } from "@/features/land/ecologyRequest";
import { useLandContext } from "@/state/landContext";

const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Public software fixture",
  description: "",
  revision: 1,
  areaM2: 10000,
  perimeterM: 500,
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
  source: { method: "drawn", label: "Fixture" },
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-77.055, 38.887],
        [-77.05, 38.887],
        [-77.05, 38.889],
        [-77.055, 38.887],
      ],
    ],
  },
};
const inv: components["schemas"]["InvestigationRead"] = {
  id: "22222222-2222-4222-8222-222222222222",
  landId: land.id,
  title: "Ecological context and name review",
  question: "Ecology",
  boundaryRevision: 1,
  createdAt: "2026-10-09",
  stale: false,
};
const detail: components["schemas"]["InvestigationDetail"] = {
  investigation: inv,
  page: { offset: 0, limit: 200, totals: {} },
  runs: [],
  evidence: [],
  findings: [],
  artifacts: [],
  messages: [],
};
function mount() {
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <EcologyResearch land={land} surveyNames={["Quercus alba", "Quercus alba", "Carex sp."]} />
    </QueryClientProvider>,
  );
}
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  localStorage.clear();
  useLandContext.getState().clear();
});

it("reviews supplied names without inventing an identification or silently truncating a survey", () => {
  expect(ecologyNames(" Quercus   alba \n\n Carex sp. ", "")).toEqual([
    { scientificName: "Quercus alba", kingdom: null },
    { scientificName: "Carex sp.", kingdom: null },
  ]);
  expect(ecologyNames("Quercus alba\nquercus ALBA", "Plantae")).toBeNull();
  expect(
    ecologyNames(Array.from({ length: 21 }, (_, i) => `Taxon ${i}`).join("\n"), ""),
  ).toBeNull();
  vi.spyOn(api, "GET").mockResolvedValue({ data: [], response: new Response() });
  mount();
  fireEvent.click(screen.getByText("Choose sources and review species names"));
  fireEvent.click(screen.getByRole("button", { name: "Use names from the displayed survey" }));
  expect(screen.getByLabelText("Scientific names to review")).toHaveValue(
    "Quercus alba\nCarex sp.",
  );
  expect(screen.getByLabelText("Kingdom hint")).toHaveValue("");
});

it("recovers source choices and reuses the request key after a lost queue response", async () => {
  vi.spyOn(api, "GET").mockImplementation((path: string) =>
    Promise.resolve({
      data: path.includes("/investigations/{") ? detail : [],
      response: new Response(),
    }),
  );
  const post = vi.spyOn(api, "POST").mockImplementation((path: string) => {
    if (path.endsWith("/investigations"))
      return Promise.resolve({ data: inv, response: new Response() });
    return Promise.reject(new Error("Connection lost after queueing"));
  });
  const view = mount();
  fireEvent.click(screen.getByText("Choose sources and review species names"));
  fireEvent.change(screen.getByLabelText("Scientific names to review"), {
    target: { value: "Quercus alba" },
  });
  fireEvent.change(screen.getByLabelText("Kingdom hint"), { target: { value: "Plantae" } });
  fireEvent.click(screen.getByLabelText(/EPA regional context/));
  fireEvent.click(screen.getByRole("button", { name: "Research ecological context" }));
  await screen.findByRole("alert");
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Research ecological context" })).toBeEnabled(),
  );
  expect(post).toHaveBeenCalledTimes(2);
  const first = post.mock.calls[1]?.[1];
  view.unmount();
  mount();
  fireEvent.click(screen.getByText("Choose sources and review species names"));
  expect(screen.getByLabelText(/EPA regional context/)).not.toBeChecked();
  expect(screen.getByLabelText("Kingdom hint")).toHaveValue("Plantae");
  expect(screen.getByLabelText("Scientific names to review")).toHaveValue("Quercus alba");
  fireEvent.click(screen.getByRole("button", { name: "Research ecological context" }));
  await waitFor(() => expect(post).toHaveBeenCalledTimes(3));
  expect(post.mock.calls[2]?.[1]).toEqual(first);
});

it("carries the selected investigation into an agent follow-up and labels stale evidence", async () => {
  const completed = {
    ...detail,
    investigation: { ...inv, stale: true },
    runs: [
      {
        id: "33333333-3333-4333-8333-333333333333",
        investigationId: inv.id,
        question: "Ecology",
        kind: "ecology" as const,
        status: "succeeded" as const,
        budget: {},
        attempt: 1,
        error: null,
        createdAt: "2026-10-09",
        startedAt: null,
        finishedAt: null,
      },
    ],
  };
  vi.spyOn(api, "GET").mockImplementation((path: string) =>
    Promise.resolve({
      data: path.endsWith("/events") ? [] : path.includes("/investigations/{") ? completed : [inv],
      response: new Response(),
    }),
  );
  mount();
  fireEvent.click(
    await screen.findByRole("button", { name: "Explore these results with the agent" }),
  );
  expect(screen.getByText(/These results use boundary revision 1/)).toBeVisible();
  expect(useLandContext.getState().selectedInvestigationId).toBe(inv.id);
  expect(useLandContext.getState().section).toBe("discover");
  expect(useLandContext.getState().researchQuestion).toContain("saved ecological evidence");
});
