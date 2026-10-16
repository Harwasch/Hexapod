/**
 * A queued research run says why it waits, and every disabled research button says why it is
 * disabled (LandResearch + ResearchWorkerNotice + researchAvailability.ts), from what
 * /research/status reports about the research worker.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { LandArea, ResearchRun, ResearchStatus } from "@twin/contracts";

import { api } from "@/api/client";
import { EcologyResearch } from "@/features/land/EcologyResearch";
import { LandResearch } from "@/features/land/LandResearch";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";

vi.mock("@/cesium/SceneContext", () => ({ useScene: () => null }));

const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Oak Bench",
  description: "",
  revision: 1,
  areaM2: 100,
  perimeterM: 40,
  createdAt: "2026-10-16",
  updatedAt: "2026-10-16",
  source: { method: "drawn", label: "Fixture" },
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-120.73, 35.52],
        [-120.71, 35.52],
        [-120.71, 35.54],
        [-120.73, 35.52],
      ],
    ],
  },
};
const inv = {
  id: "22222222-2222-4222-8222-222222222222",
  landId: land.id,
  title: "Land overview",
  question: "What is useful and interesting about this land?",
  boundaryRevision: 1,
  createdAt: "2026-10-16",
  stale: false,
};
const queuedRun: ResearchRun = {
  id: "33333333-3333-4333-8333-333333333333",
  investigationId: inv.id,
  question: inv.question,
  kind: "overview",
  status: "queued",
  budget: { maxSteps: 16, maxSeconds: 180, maxOutputTokens: 12000, maxWebSearches: 6 },
  attempt: 0,
  error: null,
  createdAt: "2026-10-16T18:00:00Z",
  startedAt: null,
  finishedAt: null,
};

function researchStatus(
  worker: Partial<ResearchStatus["worker"]>,
  modelConfigured = true,
): ResearchStatus {
  return {
    modelConfigured,
    model: modelConfigured ? "fixture" : null,
    workerRequired: true,
    worker: {
      state: "online",
      lastSeenAt: "2026-10-16T17:46:00Z",
      secondsSinceSeen: 3,
      busy: false,
      checkInEverySeconds: 15,
      ...worker,
    },
    queue: {
      queued: 1,
      oldestQueuedAt: "2026-10-16T17:46:00Z",
      oldestQueuedSeconds: 840,
      running: 0,
    },
  };
}

function mount(status: ResearchStatus, runs: ResearchRun[] = [queuedRun]) {
  useLand.getState().select(land);
  // The live progress stream stays connecting: this file is about the status, not the stream.
  vi.spyOn(globalThis, "fetch").mockReturnValue(new Promise<Response>(() => undefined));
  const get = vi.spyOn(api, "GET").mockImplementation(((path: string) =>
    Promise.resolve({
      data: path.endsWith("/status")
        ? status
        : path.endsWith("/investigations")
          ? [inv]
          : path.endsWith("/events")
            ? []
            : {
                investigation: inv,
                runs,
                messages: [],
                findings: [],
                evidence: [],
                artifacts: [],
                page: { offset: 0, limit: 100, totals: {} },
              },
      response: new Response(),
    })) as never);
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <LandResearch land={land} />
    </QueryClientProvider>,
  );
  return get;
}

afterEach(() => {
  cleanup();
  useLandContext.getState().clear();
  useLand.getState().clear();
  vi.restoreAllMocks();
  localStorage.clear();
});

it("explains a queued run that no worker has picked up, and says why analyses wait", async () => {
  mount(researchStatus({ state: "stale", secondsSinceSeen: 840 }));
  expect(
    await screen.findByText(/none has checked in for 14 min/, { selector: "p" }),
  ).toHaveAttribute("data-testid", "research-worker-notice");
  const terrain = screen.getByRole("button", { name: "Analyze terrain" });
  expect(terrain).toBeDisabled();
  expect(terrain).toHaveAttribute("aria-describedby", "land-research-blocked");
  expect(document.getElementById("land-research-blocked")?.textContent).toMatch(
    /^One request runs at a time/,
  );
  for (const name of ["Analyze land cover", "Compare vegetation", "Discover photos and maps"])
    expect(screen.getByRole("button", { name })).toHaveAttribute(
      "aria-describedby",
      "land-research-blocked",
    );
  expect(screen.getByRole("button", { name: /Cancel queued request/ })).toBeInTheDocument();
  // The worker's age is said once, by the notice, not again by every disabled button's reason.
  expect(document.body.textContent?.match(/14 min/g)).toHaveLength(1);
  // The idle footnote is left out while the land's own run explains itself.
  expect(screen.queryByTestId("research-worker-idle")).not.toBeInTheDocument();
});

it("says when the worker is online and the request is about to start", async () => {
  mount(researchStatus({ state: "online", busy: true, secondsSinceSeen: 6 }));
  expect(
    await screen.findByText(
      "Queued behind another research request. The research worker checked in 6 s ago.",
    ),
  ).toBeInTheDocument();
});

it("polls the status while a run waits, so the notice clears when a worker starts", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try {
    const get = mount(researchStatus({ state: "never", secondsSinceSeen: null }));
    await screen.findByText(/none has checked in on this server yet/);
    const calls = () =>
      (get.mock.calls as unknown as [string][]).filter(([path]) => path.endsWith("/status")).length;
    const before = calls();
    await vi.advanceTimersByTimeAsync(10_500);
    await waitFor(() => expect(calls()).toBeGreaterThan(before));
  } finally {
    vi.useRealTimers();
  }
});

it("moves the AI-configuration notice next to Investigate, as its reason", async () => {
  mount(researchStatus({}, false), []);
  const input = await screen.findByLabelText("Follow your curiosity");
  fireEvent.change(input, { target: { value: "What grew here before?" } });
  await waitFor(() =>
    expect(document.getElementById("land-ask-blocked")?.textContent).toBe(
      "AI investigation isn't configured on this server. The analyses above still work.",
    ),
  );
  expect(screen.getByRole("button", { name: "Investigate" })).toHaveAttribute(
    "aria-describedby",
    "land-ask-blocked",
  );
  expect(screen.queryByText(/AI investigation is not configured/)).not.toBeInTheDocument();
});

it("asks for a question only after the box has been used", async () => {
  mount(researchStatus({}), []);
  const input = await screen.findByLabelText("Follow your curiosity");
  await waitFor(() => expect(screen.getByRole("button", { name: "Investigate" })).toBeDisabled());
  expect(document.getElementById("land-ask-blocked")).toBeNull();
  fireEvent.focus(input);
  expect(document.getElementById("land-ask-blocked")?.textContent).toBe(
    "Type a question to investigate.",
  );
});

it("warns before anyone asks when no worker is online", async () => {
  mount(researchStatus({ state: "stale", secondsSinceSeen: 7200 }), []);
  expect(await screen.findByTestId("research-worker-idle")).toHaveTextContent(
    "Analyses run on a separate research worker, which has not checked in for 2 h. New requests will wait in the queue.",
  );
  expect(screen.getByRole("button", { name: "Analyze terrain" })).toBeEnabled();
  expect(document.getElementById("land-research-blocked")).toBeNull();
});

it("explains a queued ecology run inside its own status row", async () => {
  vi.spyOn(api, "GET").mockImplementation(((path: string) =>
    Promise.resolve({
      data: path.endsWith("/status")
        ? researchStatus({ state: "stopped", secondsSinceSeen: 300 })
        : path.endsWith("/investigations")
          ? [{ ...inv, title: "Ecological context and name review" }]
          : path.endsWith("/events")
            ? []
            : {
                investigation: { ...inv, title: "Ecological context and name review" },
                runs: [{ ...queuedRun, kind: "ecology" }],
                messages: [],
                findings: [],
                evidence: [],
                artifacts: [],
                page: { offset: 0, limit: 200, totals: {} },
              },
      response: new Response(),
    })) as never);
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <EcologyResearch land={land} surveyNames={[]} />
    </QueryClientProvider>,
  );
  expect(await screen.findByText(/the last one stopped 5 min ago/, { selector: "p" })).toHaveClass(
    "land-footnote",
  );
});
