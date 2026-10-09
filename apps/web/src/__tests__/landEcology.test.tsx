import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components, LandArea } from "@twin/contracts";
import { api } from "@/api/client";
import { LandEcology } from "@/features/land/LandEcology";
import { SurveySummaryView } from "@/features/land/SurveySummaryView";
import { parseSurveyCorners, parseSurveyDraft } from "@/features/land/surveyDraft";
import { useLandContext } from "@/state/landContext";

const boundary = {
  type: "Polygon" as const,
  coordinates: [
    [
      [-77.055, 38.887],
      [-77.05, 38.887],
      [-77.05, 38.889],
      [-77.055, 38.887],
    ],
  ],
};
const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Software fixture",
  description: "",
  revision: 1,
  areaM2: 10000,
  perimeterM: 500,
  boundary,
  source: { method: "drawn", label: "Synthetic" },
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};
const draft: components["schemas"]["SurveyCreate"] = {
  requestKey: "22222222-2222-4222-8222-222222222222",
  name: "Unfinished",
  observedOn: "2025-07-01",
  observer: "",
  boundaryRevision: 1,
  method: "visual-cover",
  design: "purposive",
  assessedStrata: ["herb"],
  methodNotes: "",
  plots: [
    {
      label: "Plot 1",
      boundary,
      observations: [
        { taxon: "", stratum: "herb", identification: "tentative", percentCover: null },
      ],
    },
  ],
};
const summary: components["schemas"]["SurveySummary"] = {
  algorithm: "field-cover-area-weighted-v1",
  landAreaM2: 10000,
  sampledAreaM2: 1000,
  sampledFraction: 0.1,
  plotAreasM2: [1000],
  species: [
    {
      taxon: "Species A",
      stratum: "herb",
      identifications: ["tentative"],
      meanPercent: 80,
      minimumPercent: 80,
      maximumPercent: 80,
      measuredPlots: 1,
      assessedAreaM2: 1000,
      assessedSampleFraction: 1,
    },
    {
      taxon: "Species B",
      stratum: "canopy",
      identifications: ["verified"],
      meanPercent: 90,
      minimumPercent: 90,
      maximumPercent: 90,
      measuredPlots: 1,
      assessedAreaM2: 1000,
      assessedSampleFraction: 1,
    },
  ],
  limitations: ["Synthetic observations."],
};
function mount() {
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <LandEcology land={land} />
    </QueryClientProvider>,
  );
}
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  localStorage.clear();
  useLandContext.getState().clear();
});
it("preserves unfinished observations without substituting measured zeros", () => {
  expect(
    parseSurveyDraft(JSON.stringify(draft)).plots[0]?.observations[0]?.percentCover,
  ).toBeNull();
  expect(() =>
    parseSurveyDraft(JSON.stringify({ ...draft, plots: [{ label: "Broken", observations: [] }] })),
  ).toThrow();
  expect(() =>
    parseSurveyDraft(
      JSON.stringify({ ...draft, plots: [{ ...draft.plots[0], observations: [{ taxon: [] }] }] }),
    ),
  ).toThrow();
  expect(parseSurveyCorners("[[181,40]]")).toEqual([]);
  expect(parseSurveyCorners("[[-77.05,38.88]]")).toEqual([[-77.05, 38.88]]);
});
it("shows overlapping species percentages without normalizing the totals", () => {
  render(<SurveySummaryView summary={summary} />);
  expect(screen.getByText("80.00%")).toBeVisible();
  expect(screen.getByText("90.00%")).toBeVisible();
  expect(screen.getByText(/10.00% of the land boundary/)).toBeVisible();
  expect(screen.getByText(/totals may exceed 100%/)).toBeVisible();
});
it("requires explicit measurements and clears them when the method changes", () => {
  vi.spyOn(api, "GET").mockResolvedValue({ data: [], response: new Response() });
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Record a field survey" }));
  fireEvent.click(screen.getByRole("button", { name: "Use entire land as the surveyed plot" }));
  fireEvent.click(screen.getByRole("button", { name: "Add species observation" }));
  expect(screen.getByLabelText("Cover (%)")).toHaveValue(null);
  fireEvent.change(screen.getByLabelText("Cover (%)"), { target: { value: "45" } });
  fireEvent.change(screen.getByLabelText("Method", { exact: true }), {
    target: { value: "point-intercept" },
  });
  expect(screen.getByLabelText("Points with taxon")).toHaveValue(null);
  expect(screen.getByLabelText("Sampled points")).toHaveValue(null);
  expect(screen.getByLabelText(/Complete inventory/)).not.toBeChecked();
});
it("invalidates a reviewed survey when observations change and recovers the unfinished draft", async () => {
  vi.spyOn(api, "GET").mockResolvedValue({ data: [], response: new Response() });
  const post = vi.spyOn(api, "POST").mockResolvedValue({ data: summary, response: new Response() });
  const view = mount();
  fireEvent.click(screen.getByRole("button", { name: "Record a field survey" }));
  fireEvent.change(screen.getByLabelText("Observer", { exact: true }), {
    target: { value: "Test observer" },
  });
  fireEvent.change(screen.getByLabelText("Survey method and limitations"), {
    target: { value: "Synthetic fixture" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Use entire land as the surveyed plot" }));
  fireEvent.click(screen.getByRole("button", { name: "Review survey" }));
  await screen.findByRole("button", { name: "Save immutable survey" });
  fireEvent.change(screen.getByLabelText("Survey name"), {
    target: { value: "Revised field survey" },
  });
  expect(screen.queryByRole("button", { name: "Save immutable survey" })).toBeNull();
  expect(post).toHaveBeenCalledTimes(1);
  view.unmount();
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Resume survey" }));
  await waitFor(() =>
    expect(screen.getByLabelText("Survey name")).toHaveValue("Revised field survey"),
  );
  expect(screen.getByLabelText("Observer", { exact: true })).toHaveValue("Test observer");
});
