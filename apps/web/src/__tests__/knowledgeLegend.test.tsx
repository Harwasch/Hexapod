/**
 * The fog legend's "Fetch latest satellite look": what it says when the run finishes comes from
 * the raster the run stored, not from the run's status. A 30-day window under cloud or snow
 * still finishes "succeeded" with no clear cells, and the fog does not count it.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components } from "@twin/contracts";

import { api } from "@/api/client";
import { KnowledgeLegend } from "@/features/knowledge/KnowledgeLegend";
import { useKnowledge } from "@/state/knowledge";
import { useLand } from "@/state/land";

import { land } from "./landGroundFixture";

vi.mock("@/cesium/SceneContext", () => ({ useScene: () => null }));

type Run = components["schemas"]["RunRead"];
type Observation = components["schemas"]["VegetationObservation"];

const INVESTIGATION = "22222222-2222-4222-8222-222222222222";
const RUN = "33333333-3333-4333-8333-333333333333";
const RASTER = "44444444-4444-4444-8444-444444444444";
const CLOUDY =
  "Scenes were found but the examined quality masks contain no clear land samples within this boundary.";

afterEach(() => {
  cleanup();
  useLand.getState().clear();
  useKnowledge.getState().setVisible(false);
  vi.restoreAllMocks();
  localStorage.clear();
});

function observation(validCells: number, explanation: string): Observation {
  return {
    period: { startDate: "2026-09-17", endDate: "2026-10-17" },
    band: 1,
    sceneId: validCells ? "S2B_MSIL2A_20261010" : null,
    acquiredAt: validCells ? "2026-10-10T18:40:00Z" : null,
    candidateCount: 3,
    catalogTruncated: false,
    qualityCandidatesExamined: 3,
    validCells,
    coverageFraction: null,
    meanNdvi: null,
    commonMeanNdvi: null,
    qualityCounts: {},
    explanation,
  };
}

/** Mounts the legend with a run that has already finished as `status`, storing `stored`. */
async function fetchLook(status: Run["status"], stored: Observation[] | null) {
  useLand.getState().select(land);
  vi.spyOn(globalThis, "fetch").mockReturnValue(new Promise<Response>(() => undefined));
  const run = {
    id: RUN,
    investigationId: INVESTIGATION,
    kind: "raster",
    status,
    error: null,
    question: "Store the latest clear Sentinel-2 look at this land as dated evidence.",
    attempt: 1,
    createdAt: "2026-10-17T00:00:00Z",
    startedAt: "2026-10-17T00:00:01Z",
    finishedAt: "2026-10-17T00:01:00Z",
  };
  vi.spyOn(api, "GET").mockImplementation(((path: string) =>
    Promise.resolve({
      data:
        path === "/api/v1/research/status"
          ? undefined
          : path === "/api/v1/land/rasters/{raster_id}"
            ? { id: RASTER, metadata: { vegetation: { observations: stored ?? [] } } }
            : {
                investigation: { id: INVESTIGATION },
                runs: [run],
                messages: [],
                findings: [],
                evidence: [],
                artifacts: stored
                  ? [
                      {
                        id: "a1",
                        runId: RUN,
                        title: "Vegetation through time",
                        method: "",
                        evidenceIds: [],
                        output: { kind: "raster", rasterId: RASTER },
                      },
                    ]
                  : [],
                page: { offset: 0, limit: 100, totals: {} },
              },
      response: new Response(),
    })) as never);
  vi.spyOn(api, "POST").mockImplementation(((path: string) =>
    Promise.resolve({
      data: path.endsWith("/runs") ? run : { id: INVESTIGATION },
      response: new Response(),
    })) as never);
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <KnowledgeLegend />
    </QueryClientProvider>,
  );
  fireEvent.click(screen.getByTestId("knowledge-details"));
  fireEvent.click(await screen.findByTestId("knowledge-satellite"));
}

it("says nothing was added when the window had no clear scene, though the run succeeded", async () => {
  await fetchLook("succeeded", [observation(0, CLOUDY)]);
  expect(
    await screen.findByText(
      `No clear satellite scene in the last 30 days, so nothing was added to the fog. ${CLOUDY}`,
    ),
  ).toBeInTheDocument();
  expect(screen.queryByText(/The fog now counts it/)).toBeNull();
});

it("says the fog counts a look only when it stored clear cells, with the scene's date", async () => {
  await fetchLook("succeeded", [observation(812, "One selected acquisition.")]);
  expect(
    await screen.findByText("Sentinel-2 scene of 2026-10-10 stored. The fog now counts it."),
  ).toBeInTheDocument();
});

it("says a partial run that stored no raster added nothing", async () => {
  await fetchLook("partial", null);
  expect(
    await screen.findByText(
      "The satellite look stopped before a scene was stored. Nothing was added to the fog.",
    ),
  ).toBeInTheDocument();
  expect(screen.queryByText(/The fog now counts it/)).toBeNull();
});
