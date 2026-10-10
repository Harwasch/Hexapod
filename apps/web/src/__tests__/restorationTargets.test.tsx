import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components } from "@twin/contracts";
import { api } from "@/api/client";
import { RestorationTargets } from "@/features/land/RestorationTargets";
import { RestorationTargetResults } from "@/features/land/RestorationTargetResults";

type Plan = components["schemas"]["RestorationEcology"];
const target: components["schemas"]["SpeciesTarget"] = {
  taxon: "Species A",
  stratum: "herb",
  baselinePercent: null,
  baselineBasis: "Unknown, needs field measurements",
  targetLow: 60,
  targetHigh: 80,
  targetYear: 5,
  targetBasis: "hypothetical",
  rationale: "Hypothetical target",
  monitoringMethod: "Repeat plots",
  monitoringSeason: "Summer",
  responseIfOffTrack: "Reassess site constraints",
};
function Editor() {
  const [plan, setPlan] = useState<Plan | null>(null);
  return (
    <>
      <RestorationTargets
        landId="land"
        plan={plan}
        onChange={setPlan}
        surveys={[]}
        treatmentNames={[]}
      />
      <output aria-label="Stored plan">{JSON.stringify(plan)}</output>
    </>
  );
}
function wrap(children: React.ReactNode) {
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      {children}
    </QueryClientProvider>,
  );
}
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it("starts with unknown baselines and blank response parameters rather than invented measurements", () => {
  vi.spyOn(api, "GET").mockResolvedValue({ data: [], response: new Response() });
  wrap(<Editor />);
  fireEvent.click(screen.getByRole("button", { name: "Add species targets and monitoring" }));
  const baseline = screen.getByLabelText("Assumed baseline cover (%) — blank means unknown");
  expect(baseline).toHaveValue(null);
  fireEvent.change(baseline, { target: { value: "0" } });
  expect(screen.getByLabelText("Stored plan").textContent).toContain('"baselinePercent":0');
  fireEvent.change(baseline, { target: { value: "" } });
  expect(screen.getByLabelText("Stored plan").textContent).toContain('"baselinePercent":null');
  fireEvent.click(screen.getByText("Optional conditional response curve"));
  fireEvent.click(screen.getByRole("button", { name: "Enter response assumptions" }));
  expect(screen.getByLabelText("Response rate lower bound (per year)")).toHaveValue(null);
  expect(screen.getByLabelText("Asymptotic cover lower bound (%)")).toHaveValue(null);
  fireEvent.click(screen.getByRole("button", { name: "Remove response curve" }));
  expect(screen.queryByLabelText("Response rate lower bound (per year)")).toBeNull();
});

it("shows independent species goals and unknown baselines without normalizing overlapping cover", () => {
  wrap(
    <RestorationTargetResults
      result={{
        algorithm: "restoration-species-targets-and-conditional-response/1",
        referenceBasis: "Local reference pending",
        referenceEvidenceIds: [],
        siteConstraints: "Soils need checking",
        limitations: [],
        species: [
          {
            target,
            baselinePercent: null,
            baselineScope: "unknown",
            targetRelation: "not-modeled",
            limitations: [],
          },
          {
            target: {
              ...target,
              taxon: "Species B",
              stratum: "canopy",
              targetLow: 90,
              targetHigh: 100,
            },
            baselinePercent: 90,
            baselineScope: "surveyed-plots",
            observedOn: "2025-07-01",
            assessedLandFraction: 0.2,
            targetRelation: "not-modeled",
            limitations: [],
            identificationStatus: ["tentative"],
          },
        ],
      }}
    />,
  );
  expect(screen.getByText(/60–80%/)).toBeVisible();
  expect(screen.getByText(/90–100%/)).toBeVisible();
  expect(screen.getByText(/Unknown · not measured/)).toBeVisible();
  expect(screen.getByText(/assessed plots cover 20%/)).toBeVisible();
  expect(screen.queryByRole("img")).toBeNull();
});

it("renders a conditional interval and monitoring plan without presenting it as a probability", () => {
  wrap(
    <RestorationTargetResults
      result={{
        referenceBasis: "Fixture",
        referenceEvidenceIds: [],
        siteConstraints: "Fixture",
        limitations: [],
        species: [
          {
            target: {
              ...target,
              response: {
                startYear: 0,
                asymptoteLow: 80,
                asymptoteHigh: 100,
                annualRateLow: 0.2,
                annualRateHigh: 0.5,
                basis: "User-supplied hypothetical response",
              },
            },
            baselinePercent: 20,
            baselineScope: "entered-assumption",
            targetRelation: "overlaps-target",
            projection: [
              { year: 0, low: 20, high: 20 },
              { year: 5, low: 57.93, high: 93.43 },
            ],
            limitations: [],
          },
        ],
      }}
    />,
  );
  expect(screen.getByRole("img")).toHaveAccessibleName(/conditional cover envelope/);
  expect(screen.getByText(/not a measured trend or probability interval/)).toBeVisible();
  fireEvent.click(screen.getByText("Response method and entered assumptions"));
  const table = screen.getByRole("table", { name: "Conditional cover envelope (%)" });
  expect(within(table).getByText("57.93")).toBeVisible();
  fireEvent.click(screen.getByText("Monitoring and response plan"));
  expect(screen.getByText("Repeat plots")).toBeVisible();
});
