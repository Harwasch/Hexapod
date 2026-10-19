import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  InvestigationDetail,
  LandArea,
  LandFinding,
  ResearchEvent,
  ResearchRun,
} from "@twin/contracts";

import { api } from "@/api/client";
import { askAgent, useAgentBus } from "@/features/land/surface/agentBus";
import { AgentCard } from "@/features/land/surface/agent/AgentCard";
import { LandAgent, LandAgentInline } from "@/features/land/surface/LandAgent";
import { useUi } from "@/state/ui";
import {
  findingPlace,
  locationAnchor,
  locationArea,
  runProgress,
  splitAnswer,
  stepsFromEvents,
} from "@/features/land/surface/agent/agentEvents";
import { useAgentUi } from "@/features/land/surface/agent/agentStore";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { currentLeaders, useLandMap } from "@/state/landMap";

vi.mock("@/cesium/SceneContext", () => ({ useScene: () => null }));

const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "North field",
  description: "",
  revision: 1,
  areaM2: 100_000,
  perimeterM: 1300,
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
  source: { method: "drawn", label: "Drawn" },
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-122.14, 47.64],
        [-122.13, 47.64],
        [-122.13, 47.65],
        [-122.14, 47.65],
        [-122.14, 47.64],
      ],
    ],
  },
};

const INV = "22222222-2222-4222-8222-222222222222";
const RUN = "33333333-3333-4333-8333-333333333333";
const EVIDENCE = "44444444-4444-4444-8444-444444444444";

function investigation(id = INV, question = "Where could a pond go?", kinds = ["investigation"]) {
  return {
    id,
    landId: land.id,
    title: question,
    question,
    boundaryRevision: 1,
    createdAt: "2026-10-10T10:00:00Z",
    stale: false,
    kinds: kinds as ("investigation" | "raster")[],
  };
}

function run(status: ResearchRun["status"]): ResearchRun {
  return {
    id: RUN,
    investigationId: INV,
    question: "Where could a pond go?",
    kind: "investigation",
    status,
    budget: { maxSteps: 16, maxSeconds: 180, maxOutputTokens: 12000, maxWebSearches: 6 },
    attempt: 1,
    error: null,
    createdAt: "2026-10-10T10:00:00Z",
    startedAt: "2026-10-10T10:00:01Z",
    finishedAt: null,
  };
}

function finding(id: string, title: string, extra: Partial<LandFinding> = {}): LandFinding {
  return {
    id,
    runId: RUN,
    title,
    summary: `${title}, in more words.`,
    category: "physical",
    evidenceIds: [EVIDENCE],
    confidence: "inferred",
    uncertainty: "From one source.",
    disposition: "visible",
    suggestedQuestions: [],
    ...extra,
  };
}

function detail(
  status: ResearchRun["status"],
  extra: Partial<InvestigationDetail> = {},
): InvestigationDetail {
  return {
    investigation: investigation(),
    runs: [run(status)],
    messages: [],
    findings: [],
    evidence: [],
    artifacts: [],
    page: { offset: 0, limit: 100, totals: {} },
    ...extra,
  };
}

let seq = 0;
function event(kind: string, payload: Record<string, unknown> = {}, location?: unknown) {
  seq += 1;
  return {
    sequence: seq,
    kind,
    payload,
    location: location ?? null,
    createdAt: "2026-10-10T10:00:02Z",
  } as ResearchEvent;
}

const POINT = { kind: "point", point: [-122.135, 47.645], meaning: "sample-point" };
const BOX = { kind: "bbox", bbox: [-122.14, 47.64, -122.13, 47.65], meaning: "query-bbox" };

interface Fixture {
  configured?: boolean;
  list?: ReturnType<typeof investigation>[];
  detail?: InvestigationDetail;
  events?: ResearchEvent[];
}

function mount(fixture: Fixture = {}) {
  useLand.getState().select(land);
  const get = vi.spyOn(api, "GET").mockImplementation((path: string, init?: unknown) => {
    const options = init as { params?: { query?: { after?: number } } } | undefined;
    const data = path.endsWith("/status")
      ? { modelConfigured: fixture.configured ?? true, model: "fixture" }
      : path.endsWith("/investigations")
        ? (fixture.list ?? [])
        : path.endsWith("/events")
          ? (fixture.events ?? []).filter(
              (item) => item.sequence > (options?.params?.query?.after ?? 0),
            )
          : path.endsWith("/stream")
            ? null
            : path.includes("/evidence/")
              ? null
              : (fixture.detail ?? detail("succeeded"));
    return Promise.resolve({ data, response: new Response() });
  });
  const view = render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <AgentCard land={land} />
    </QueryClientProvider>,
  );
  return { get, view };
}

afterEach(() => {
  cleanup();
  useLandContext.getState().clear();
  useLandMap.setState({ pins: {}, focus: null });
  useAgentUi.setState({ highlight: null, closed: null, autoShown: null, focusTick: 0 });
  useAgentBus.setState({ request: null });
  useUi.setState({ activePanel: null });
  vi.restoreAllMocks();
  localStorage.clear();
});

describe("steps in plain words", () => {
  it("joins a tool's lookup and answer to the step that chose it and hides bookkeeping", () => {
    const steps = stepsFromEvents(
      [
        event("queued", { kind: "investigation" }),
        event("started", { attempt: 1 }),
        event("model", { step: 1 }),
        event(
          "progress",
          { message: "Checking the ground height", action: "retrieve_source" },
          POINT,
        ),
        event("looking", { provider: "usgs-elevation", message: "Checking USGS elevation" }, POINT),
        event("source", { provider: "usgs-elevation", status: "available" }, POINT),
        event("finding", { id: "f1", title: "Ground height" }, POINT),
        event("tool-result", { action: "retrieve_source", result: "ok" }, POINT),
        event("progress", {}),
        event("model", { step: 2 }),
      ],
      true,
    );
    expect(steps.map((step) => step.text)).toEqual([
      "Checking the ground height",
      "Found: Ground height",
      "Thinking about the next step",
    ]);
    expect(steps[0]).toMatchObject({
      source: "USGS elevation",
      note: "Found data",
      state: "done",
      location: POINT,
    });
    expect(steps[1]).toMatchObject({ state: "finding", findingId: "f1" });
    expect(steps[2]?.state).toBe("active");
  });

  it("keeps the agent's own words when it publishes what it found", () => {
    const steps = stepsFromEvents(
      [
        event("model", { step: 3 }),
        event("progress", {
          message: "Found a basin outside the flood zone",
          action: "publish_finding",
        }),
        event("finding", { id: "f9", title: "Natural basin" }, BOX),
        event("tool-result", { action: "publish_finding", result: "ok" }),
      ],
      false,
    );
    expect(steps).toEqual([
      expect.objectContaining({
        text: "Found a basin outside the flood zone",
        state: "finding",
        findingId: "f9",
        location: BOX,
      }),
    ]);
  });

  it("lists sources looked up without the model, and drops thinking once it is done", () => {
    const steps = stepsFromEvents(
      [
        event("looking", { provider: "fema-flood-zones" }, BOX),
        event("source", { provider: "fema-flood-zones", status: "empty" }, BOX),
        event("artifact", { id: "a1", kind: "map", title: "Flood zones" }, BOX),
        event("model", { step: 3 }),
        event("succeeded", { message: "Done" }),
      ],
      false,
    );
    expect(steps.map((step) => [step.text, step.note, step.state])).toEqual([
      ["Checking FEMA flood maps", "Nothing recorded here", "done"],
      ["Made a map: Flood zones", undefined, "done"],
    ]);
  });

  it("finds places: a point, a box's middle, a search circle", () => {
    expect(locationAnchor(POINT as never)).toEqual([-122.135, 47.645]);
    const middle = locationAnchor(BOX as never);
    expect(middle?.[0]).toBeCloseTo(-122.135, 6);
    expect(middle?.[1]).toBeCloseTo(47.645, 6);
    const circle = locationArea({ ...POINT, radiusM: 500, meaning: "query-radius" } as never);
    expect(circle?.type).toBe("Polygon");
    expect(locationArea(POINT as never)).toBeNull();
    const boxed = locationArea(BOX as never);
    expect(boxed?.type === "Polygon" && boxed.coordinates[0]).toHaveLength(5);
    const place = findingPlace(finding("f1", "Wet corner"), new Map([["f1", POINT as never]]));
    expect(place?.anchor).toEqual([-122.135, 47.645]);
  });

  it("splits the answer from its citations and measures the budget used", () => {
    expect(splitAnswer(`The basin holds water.\n\nEvidence: ${EVIDENCE}`)).toEqual({
      text: "The basin holds water.",
      evidenceIds: [EVIDENCE],
    });
    const started = Date.parse("2026-10-10T10:00:01Z");
    expect(runProgress(run("running"), [], started + 90_000)).toBeCloseTo(0.5);
    expect(runProgress(run("succeeded"), [])).toBe(1);
  });
});

describe("the agent card", () => {
  it("is one line to ask, and starts nothing on its own", async () => {
    const post = vi.spyOn(api, "POST");
    mount();
    expect(await screen.findByLabelText("Ask about this land")).toHaveValue("");
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Ask the agent" })).toBeDisabled(),
    );
    expect(post).not.toHaveBeenCalled();
  });

  it("says plainly when the agent isn't set up on this server", async () => {
    mount({ configured: false });
    expect(
      await screen.findByText(
        "The agent isn't set up on this server. The lens analyses still work.",
      ),
    ).toBeVisible();
    fireEvent.change(screen.getByLabelText("Ask about this land"), {
      target: { value: "What grows here?" },
    });
    expect(screen.getByRole("button", { name: "Ask the agent" })).toBeDisabled();
  });

  it("sends the question as an agent investigation", async () => {
    const post = vi.spyOn(api, "POST").mockImplementation((path: string) =>
      Promise.resolve({
        data: path.endsWith("/investigations") ? investigation() : run("queued"),
        response: new Response(),
      }),
    );
    mount();
    fireEvent.change(await screen.findByLabelText("Ask about this land"), {
      target: { value: "Where could a pond go?" },
    });
    const send = screen.getByRole("button", { name: "Ask the agent" });
    await waitFor(() => expect(send).toBeEnabled());
    fireEvent.click(send);
    await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
    expect(post.mock.calls[0]?.[0]).toBe("/api/v1/land/{land_id}/investigations");
    expect(post.mock.calls[1]?.[0]).toBe("/api/v1/research/investigations/{investigation_id}/runs");
    expect((post.mock.calls[1]?.[1] as unknown as { body: unknown }).body).toMatchObject({
      kind: "investigation",
      question: "Where could a pond go?",
      budget: { maxSeconds: 180, maxWebSearches: 6 },
    });
    await waitFor(() => expect(useLandContext.getState().selectedInvestigationId).toBe(INV));
  });

  it("takes a question from a lens and waits for the person to send it", async () => {
    const post = vi.spyOn(api, "POST");
    mount();
    act(() =>
      askAgent(
        "Could this basin hold a pond?",
        { artifactId: EVIDENCE, featureIndex: 2 },
        "Northwest basin",
      ),
    );
    expect(await screen.findByLabelText("Ask about this land")).toHaveValue(
      "Could this basin hold a pond?",
    );
    expect(screen.getByText("Northwest basin")).toBeVisible();
    expect(useLandContext.getState().researchFocus).toEqual({
      artifactId: EVIDENCE,
      featureIndex: 2,
      label: "Northwest basin",
    });
    expect(useAgentBus.getState().request).toBeNull();
    expect(post).not.toHaveBeenCalled();
  });

  it("shows the step it is on with a line to where it is looking, and can cancel", async () => {
    const post = vi
      .spyOn(api, "POST")
      .mockResolvedValue({ data: run("cancelled"), response: new Response() });
    useLandContext.getState().selectInvestigation(INV);
    mount({
      list: [investigation()],
      detail: detail("running"),
      events: [
        event("queued", { kind: "investigation" }),
        event("model", { step: 1 }),
        event("progress", { message: "Checking flood maps", action: "retrieve_source" }, BOX),
        event("looking", { provider: "fema-flood-zones" }, BOX),
      ],
    });
    const steps = await screen.findByRole("list", { name: "What the agent is doing" });
    await within(steps).findByText("Checking flood maps");
    expect(within(steps).getByText("FEMA flood maps")).toBeVisible();
    expect(screen.queryByText("Waiting to start")).not.toBeInTheDocument();
    await waitFor(() =>
      expect(currentLeaders()).toContainEqual(
        expect.objectContaining({
          tone: "agent",
          anchor: [expect.closeTo(-122.135, 6), expect.closeTo(47.645, 6)],
        }),
      ),
    );
    // The box it is querying is outlined while it looks.
    expect(useLandContext.getState().layers["agent-looking"]?.features).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() =>
      expect(post).toHaveBeenCalledWith(
        "/api/v1/research/runs/{run_id}/cancel",
        expect.objectContaining({ params: { path: { run_id: RUN } } }),
      ),
    );
  });

  it("gives a short answer, pins what it found and offers follow-ups", async () => {
    useLandContext.getState().selectInvestigation(INV);
    const boundary = {
      type: "Polygon" as const,
      coordinates: [
        [
          [-122.139, 47.641],
          [-122.138, 47.641],
          [-122.138, 47.642],
          [-122.139, 47.642],
          [-122.139, 47.641],
        ],
      ],
    };
    mount({
      list: [investigation()],
      detail: detail("succeeded", {
        messages: [
          {
            id: "m1",
            role: "user",
            content: "Where could a pond go?",
            createdAt: "2026-10-10T10:00:00Z",
          },
          {
            id: "m2",
            role: "assistant",
            content: `The northwest basin holds water.\n\nEvidence: ${EVIDENCE}`,
            createdAt: "2026-10-10T10:02:00Z",
          },
        ],
        findings: [
          finding("f1", "Natural basin", {
            boundary,
            suggestedQuestions: ["How big could the pond be?"],
          }),
          finding("f2", "Clay soil holds water"),
          finding("f3", "Old fence line", { disposition: "dismissed" }),
        ],
      }),
      events: [
        event("finding", { id: "f1", title: "Natural basin" }),
        event("finding", { id: "f2", title: "Clay soil holds water" }, POINT),
        event("succeeded", { message: "Done" }),
      ],
    });
    expect(await screen.findByText("The northwest basin holds water.")).toBeVisible();
    const found = screen.getByRole("list", { name: "What the agent found" });
    expect(within(found).getByText("Natural basin")).toBeVisible();
    expect(within(found).queryByText("Old fence line")).not.toBeInTheDocument();
    await waitFor(() => expect(useLandMap.getState().pins.agent).toHaveLength(2));
    expect(useLandMap.getState().pins.agent?.every((pin) => pin.tone === "agent")).toBe(true);
    expect(useLandContext.getState().layers["agent-findings"]?.features).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "How big could the pond be?" }));
    expect(screen.getByLabelText("Ask about this land")).toHaveValue("How big could the pond be?");
    fireEvent.click(screen.getByRole("button", { name: "New question" }));
    expect(useLandContext.getState().selectedInvestigationId).toBeNull();
    await waitFor(() => expect(useLandMap.getState().pins.agent).toBeUndefined());
  });

  it("keeps earlier questions a tap away, without the lenses' analyses", async () => {
    mount({
      list: [
        investigation(),
        investigation("55555555-5555-4555-8555-555555555555", "What grew here in 1950?"),
        investigation("66666666-6666-4666-8666-666666666666", "Land cover in 2021", ["raster"]),
      ],
    });
    const history = await screen.findByRole("list", { name: "Earlier questions" });
    expect(within(history).getAllByRole("button")).toHaveLength(2);
    expect(within(history).queryByText("Land cover in 2021")).not.toBeInTheDocument();
    fireEvent.click(within(history).getByText("What grew here in 1950?"));
    expect(useLandContext.getState().selectedInvestigationId).toBe(
      "55555555-5555-4555-8555-555555555555",
    );
  });

  it("asks about a map feature picked on the map", async () => {
    useLandContext.getState().setLayer({
      id: EVIDENCE,
      researchArtifactId: EVIDENCE,
      title: "Flood zones",
      features: [
        {
          id: "0",
          label: "Zone AE",
          geometry: { type: "Point", coordinates: [-122.135, 47.645] },
        },
      ],
    });
    useLandContext.getState().selectMapFeature(EVIDENCE, "0", "map");
    mount();
    const picked = await screen.findByRole("region", { name: "Selected map feature" });
    fireEvent.click(within(picked).getByRole("button", { name: "Ask about this feature" }));
    expect(useLandContext.getState().researchFocus).toEqual({
      artifactId: EVIDENCE,
      featureIndex: 0,
      label: "Zone AE",
    });
    expect(screen.getByLabelText("Ask about this land")).toHaveValue(
      "What can you tell me about this feature?",
    );
    fireEvent.click(within(picked).getByRole("button", { name: "Clear map selection" }));
    expect(useLandContext.getState().selectedMapFeature).toBeNull();
  });
});

describe("one agent card at every width", () => {
  function host(compact: boolean) {
    vi.spyOn(window, "matchMedia").mockImplementation(
      (query: string) =>
        ({
          matches: compact && query.includes("899px"),
          media: query,
          onchange: null,
          addListener: () => undefined,
          removeListener: () => undefined,
          addEventListener: () => undefined,
          removeEventListener: () => undefined,
          dispatchEvent: () => false,
        }) as MediaQueryList,
    );
    mount();
    cleanup();
    useUi.setState({ activePanel: "land" });
    return render(
      <QueryClientProvider client={new QueryClient()}>
        <div data-testid="dock">
          <LandAgent />
        </div>
        <div data-testid="sheet">
          <LandAgentInline />
        </div>
      </QueryClientProvider>,
    );
  }

  it("lives in the right dock on wide screens", async () => {
    host(false);
    expect(await within(screen.getByTestId("dock")).findByTestId("land-agent")).toBeVisible();
    expect(within(screen.getByTestId("sheet")).queryByTestId("land-agent")).toBeNull();
  });

  it("moves into the left sheet on narrow screens", async () => {
    host(true);
    expect(await within(screen.getByTestId("sheet")).findByTestId("land-agent")).toBeVisible();
    expect(within(screen.getByTestId("dock")).queryByTestId("land-agent")).toBeNull();
  });

  it("stays away while there is no saved land open", () => {
    host(false);
    act(() => useUi.setState({ activePanel: null }));
    expect(screen.queryByTestId("land-agent")).toBeNull();
  });
});
