/**
 * A queued research run says why it waits, wherever it was started (a lens or the agent card),
 * its stop button cancels the queued request by that name, and every disabled research button
 * says why it is disabled (ResearchWorkerNotice + researchAvailability.ts), from what
 * /research/status reports about the research worker.
 */

import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { components, LandArea, ResearchRun, ResearchStatus } from "@twin/contracts";

import { api } from "@/api/client";
import { env } from "@/app/env";
import { TYPE_A_QUESTION } from "@/features/land/researchAvailability";
import { AgentCard } from "@/features/land/surface/agent/AgentCard";
import { useAgentUi } from "@/features/land/surface/agent/agentStore";
import { EcologyLens } from "@/features/land/surface/lenses/EcologyLens";
import { OverviewLens } from "@/features/land/surface/lenses/OverviewLens";
import { useOverviewMap } from "@/features/land/surface/lenses/overview/useOverview";
import { SolarLens } from "@/features/land/surface/lenses/SolarLens";
import { WaterLens } from "@/features/land/surface/lenses/WaterLens";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { useLandMap } from "@/state/landMap";

vi.mock("@/cesium/SceneContext", () => ({ useScene: () => null }));

type Investigation = components["schemas"]["InvestigationRead"];
type Detail = components["schemas"]["InvestigationDetail"];
type Raster = components["schemas"]["LandRasterRead"];

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

function investigation(id: string, title: string, kinds?: string[]): Investigation {
  return {
    id,
    landId: land.id,
    title,
    question: title,
    boundaryRevision: 1,
    createdAt: "2026-10-16",
    stale: false,
    ...(kinds ? { kinds: kinds as Investigation["kinds"] } : {}),
  };
}

function queuedRun(
  kind: ResearchRun["kind"],
  question = "What is useful and interesting about this land?",
  status: ResearchRun["status"] = "queued",
): ResearchRun {
  return {
    id: "33333333-3333-4333-8333-333333333333",
    investigationId: "inv",
    question,
    kind,
    status,
    budget: { maxSteps: 16, maxSeconds: 180, maxOutputTokens: 12000, maxWebSearches: 6 },
    attempt: 0,
    error: null,
    createdAt: "2026-10-16T18:00:00Z",
    startedAt: null,
    finishedAt: null,
  };
}

function detail(item: Investigation, runs: ResearchRun[], rasterId?: string): Detail {
  return {
    investigation: item,
    runs,
    messages: [],
    findings: [],
    evidence: [],
    artifacts: rasterId
      ? [
          {
            id: `artifact-${rasterId}`,
            runId: runs[0]?.id ?? "run",
            title: item.title,
            method: "Synthetic",
            evidenceIds: [],
            output: { kind: "raster", rasterId },
          },
        ]
      : [],
    page: { offset: 0, limit: 100, totals: {} },
  };
}

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

const terrainBand: components["schemas"]["RasterBand"] = {
  index: 1,
  name: "Surface elevation",
  unit: "m",
  validCells: 8,
  minimum: 62,
  maximum: 78,
  mean: 70,
  standardDeviation: 4,
  percentiles: { "50": 70 },
  histogramEdges: [62, 70, 78],
  histogramCounts: [5, 3],
  palette: "viridis",
};
const terrain: Raster = {
  id: "terrain",
  landId: land.id,
  runId: "run-terrain",
  boundaryRevision: 1,
  request: { dataset: "cop-dem-glo-30" },
  sha256: "a".repeat(64),
  byteSize: 100,
  stale: false,
  createdAt: "2026-10-16T12:00:00Z",
  metadata: {
    algorithm: "fixture-v1",
    method: "Synthetic fixture",
    bounds: [-120.73, 35.52, -120.71, 35.54],
    crs: "EPSG:4326",
    resolutionM: 30,
    width: 4,
    height: 4,
    boundaryCells: 10,
    validCells: 8,
    coverageFraction: 0.8,
    sampledAreaM2: 7200,
    downloadedBytes: 100,
    warnings: [],
    sources: [],
    bands: [terrainBand, { ...terrainBand, index: 2, name: "Surface slope", unit: "degrees" }],
  },
};

interface World {
  status?: ResearchStatus;
  investigations?: Investigation[];
  details?: Record<string, Detail>;
  rasters?: Raster[];
}

function serve(world: World) {
  useLand.getState().select(land);
  // The live progress stream stays connecting: this file is about the status, not the stream.
  vi.spyOn(globalThis, "fetch").mockReturnValue(new Promise<Response>(() => undefined));
  const posts: string[] = [];
  vi.spyOn(api, "POST").mockImplementation(((path: string) => {
    posts.push(path);
    return Promise.resolve({
      data: queuedRun("raster", "", "cancelled"),
      response: new Response(),
    });
  }) as never);
  const get = vi.spyOn(api, "GET").mockImplementation(((
    path: string,
    init?: { params?: { path?: Record<string, string> } },
  ) => {
    let data: unknown = [];
    if (path === "/api/v1/research/status") data = world.status;
    else if (path === "/api/v1/land/{land_id}/investigations") data = world.investigations ?? [];
    else if (path === "/api/v1/research/investigations/{investigation_id}")
      data = world.details?.[init?.params?.path?.investigation_id ?? ""];
    else if (path === "/api/v1/land/{land_id}/rasters") data = world.rasters ?? [];
    return Promise.resolve({ data, response: new Response() });
  }) as never);
  return { get, posts };
}

function mount(node: ReactNode) {
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      {node}
    </QueryClientProvider>,
  );
}

function describedBy(element: HTMLElement): HTMLElement | null {
  return document.getElementById(element.getAttribute("aria-describedby") ?? "");
}

afterEach(() => {
  cleanup();
  useLandContext.getState().clear();
  useLandMap.setState({ pins: {}, focus: null });
  useOverviewMap.setState({ views: {}, opacity: 0.8 });
  useAgentUi.setState({ highlight: null, closed: null, autoShown: null, focusTick: 0 });
  useLand.getState().clear();
  vi.restoreAllMocks();
  localStorage.clear();
});

const terrainInv = investigation("inv-terrain", "Surface elevation and slope");

describe("the Overview lens", () => {
  it("explains a queued run that no worker has picked up, and cancels it by that name", async () => {
    const { posts } = serve({
      status: researchStatus({ state: "stale", secondsSinceSeen: 840 }),
      investigations: [terrainInv],
      details: { "inv-terrain": detail(terrainInv, [queuedRun("raster")]) },
    });
    mount(<OverviewLens land={land} />);
    expect(
      await screen.findByText(/none has checked in for 14 min/, { selector: "p" }),
    ).toHaveAttribute("data-testid", "research-worker-notice");
    expect(screen.getByText("Waiting to read the terrain…")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Read the terrain" })).not.toBeInTheDocument();
    // The worker's age is said once, by the notice, not again by an idle note.
    expect(document.body.textContent?.match(/14 min/g)).toHaveLength(1);
    expect(screen.queryByTestId("research-worker-idle")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel queued request" }));
    await waitFor(() => expect(posts).toContain("/api/v1/research/runs/{run_id}/cancel"));
  });

  it("says when the worker is online and the request is about to start", async () => {
    serve({
      status: researchStatus({ state: "online", busy: true, secondsSinceSeen: 6 }),
      investigations: [terrainInv],
      details: { "inv-terrain": detail(terrainInv, [queuedRun("raster")]) },
    });
    mount(<OverviewLens land={land} />);
    expect(
      await screen.findByText(
        "Queued behind another research request. The research worker checked in 6 s ago.",
      ),
    ).toBeInTheDocument();
  });

  it("polls the status while a run waits, so the notice clears when a worker starts", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const { get } = serve({
        status: researchStatus({ state: "never", secondsSinceSeen: null }),
        investigations: [terrainInv],
        details: { "inv-terrain": detail(terrainInv, [queuedRun("raster")]) },
      });
      mount(<OverviewLens land={land} />);
      await screen.findByText(/none has checked in on this server yet/);
      const calls = () =>
        (get.mock.calls as unknown as [string][]).filter(([path]) => path.endsWith("/status"))
          .length;
      const before = calls();
      await vi.advanceTimersByTimeAsync(10_500);
      await waitFor(() => expect(calls()).toBeGreaterThan(before));
    } finally {
      vi.useRealTimers();
    }
  });

  it("says why months cannot be compared while a comparison runs", async () => {
    const greenInv = investigation("inv-green", "Vegetation through time");
    serve({
      status: researchStatus({}),
      investigations: [terrainInv, greenInv],
      details: {
        "inv-terrain": detail(terrainInv, [queuedRun("raster", "", "succeeded")], "terrain"),
        "inv-green": detail(greenInv, [
          queuedRun(
            "raster",
            "Compare dated vegetation signals on common clear cells inside this land.",
            "running",
          ),
        ]),
      },
      rasters: [terrain],
    });
    mount(<OverviewLens land={land} />);
    expect(await screen.findByText("Comparing greenness…")).toBeVisible();
    expect(screen.getByRole("button", { name: "Stop" })).toBeVisible();
    fireEvent.click(screen.getByText("Details and sources"));
    const compare = screen.getByRole("button", { name: "Compare these months" });
    expect(compare).toBeDisabled();
    expect(describedBy(compare)?.textContent).toMatch(
      /^One request runs at a time\. Wait for “Compare dated vegetation signals/,
    );
  });

  it("warns before anyone asks when no worker is online", async () => {
    serve({ status: researchStatus({ state: "stale", secondsSinceSeen: 7200 }) });
    mount(<OverviewLens land={land} />);
    const idle = await screen.findByTestId("research-worker-idle");
    expect(idle).toHaveTextContent(
      "Analyses run on a separate research worker, which has not checked in for 2 h. New requests will wait in the queue.",
    );
    if (env.isDev) expect(idle).toHaveTextContent("pnpm dev:research");
    expect(screen.getByRole("button", { name: "Read the terrain" })).toBeEnabled();
    // Said once: Details holds no second copy while the card says it.
    expect(screen.getAllByTestId("research-worker-idle")).toHaveLength(1);
  });
});

describe("the agent card", () => {
  const agentInv = investigation("22222222-2222-4222-8222-222222222222", "Where could a pond go?", [
    "investigation",
  ]);

  it("says beside the send button that the agent isn't set up, as its reason", async () => {
    serve({ status: researchStatus({}, false) });
    mount(<AgentCard land={land} />);
    const send = await screen.findByRole("button", { name: "Ask the agent" });
    await waitFor(() =>
      expect(describedBy(send)?.textContent).toBe(
        "The agent isn't set up on this server. The lens analyses still work.",
      ),
    );
    expect(send).toBeDisabled();
    expect(screen.queryByText(/AI investigation/)).not.toBeInTheDocument();
  });

  it("asks for a question only after the box has been used, then says a worker is missing", async () => {
    serve({ status: researchStatus({ state: "stale", secondsSinceSeen: 7200 }) });
    mount(<AgentCard land={land} />);
    const input = await screen.findByLabelText("Ask about this land");
    const send = screen.getByRole("button", { name: "Ask the agent" });
    await waitFor(() => expect(send).toBeDisabled());
    // The status has loaded: nothing else stands in the way but the empty box.
    await waitFor(() => expect(screen.queryByTestId("land-agent-blocked")).toBeNull());
    expect(send).not.toHaveAttribute("aria-describedby");
    expect(screen.queryByTestId("research-worker-idle")).toBeNull();
    fireEvent.focus(input);
    expect(screen.getByTestId("land-agent-blocked")).toHaveTextContent(TYPE_A_QUESTION);
    expect(describedBy(send)).toHaveTextContent(TYPE_A_QUESTION);
    expect(screen.getByTestId("research-worker-idle")).toHaveTextContent(
      "which has not checked in for 2 h",
    );
    fireEvent.change(input, { target: { value: "What grew here?" } });
    await waitFor(() => expect(send).toBeEnabled());
    expect(screen.queryByTestId("land-agent-blocked")).toBeNull();
  });

  it("explains a queued question and cancels it by that name", async () => {
    useLandContext.getState().selectInvestigation(agentInv.id);
    const { posts } = serve({
      status: researchStatus({ state: "stopped", secondsSinceSeen: 300 }),
      investigations: [agentInv],
      details: {
        [agentInv.id]: detail(agentInv, [queuedRun("investigation", "Where could a pond go?")]),
      },
    });
    mount(<AgentCard land={land} />);
    expect(
      await screen.findByText(/the last one stopped 5 min ago/, { selector: "p" }),
    ).toHaveAttribute("data-testid", "research-worker-notice");
    fireEvent.click(screen.getByRole("button", { name: "Cancel queued request" }));
    await waitFor(() => expect(posts).toContain("/api/v1/research/runs/{run_id}/cancel"));
  });
});

describe("the other lenses", () => {
  it("explains a queued ecology lookup inside its own progress", async () => {
    const ecology = investigation("inv-eco", "Ecological context and name review");
    serve({
      status: researchStatus({ state: "stopped", secondsSinceSeen: 300 }),
      investigations: [ecology],
      details: { "inv-eco": detail(ecology, [queuedRun("ecology")]) },
    });
    mount(<EcologyLens land={land} />);
    const notice = await screen.findByText(/the last one stopped 5 min ago/, { selector: "p" });
    expect(notice).toHaveClass("land-muted");
    expect(notice.closest(".land-eco-running")).not.toBeNull();
    expect(screen.getByRole("button", { name: "Cancel queued request" })).toBeVisible();
  });

  it("explains a queued solar model", async () => {
    const solar = investigation("inv-solar", "Hourly solar assessment · 2025");
    serve({
      status: researchStatus({ state: "stale", secondsSinceSeen: 840 }),
      investigations: [solar],
      details: { "inv-solar": detail(solar, [queuedRun("solar", "Calculate hourly generation")]) },
    });
    mount(<SolarLens land={land} />);
    expect(
      await screen.findByText(/none has checked in for 14 min/, { selector: "p" }),
    ).toBeVisible();
    expect(
      screen.getByText("Waiting to start. It takes about a minute once it runs."),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "Cancel queued request" })).toBeVisible();
  });

  it("explains a queued check of water, soil and climate", async () => {
    const overview = investigation("inv-overview", "Land overview");
    serve({
      status: researchStatus({ state: "never", secondsSinceSeen: null }),
      investigations: [overview],
      details: { "inv-overview": detail(overview, [queuedRun("overview")]) },
    });
    mount(<WaterLens land={land} />);
    const notice = await screen.findByText(/none has checked in on this server yet/, {
      selector: "p",
    });
    expect(notice).toHaveAttribute("data-testid", "research-worker-notice");
    const progress = screen.getByText(/Checking public maps/).closest(".bundle-progress");
    expect(
      within(progress as HTMLElement).getByRole("button", { name: "Cancel queued request" }),
    ).toBeVisible();
  });
});
