import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { components, LandArea } from "@twin/contracts";

import { api } from "@/api/client";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { SceneContext, SceneRegistry } from "@/cesium/SceneContext";
import {
  actionDraftKey,
  actionEdit,
  actionPayload,
  serializeActionDraft,
} from "@/features/land/actionDraft";
import { restorationDefaults, solarDefaults } from "@/features/land/scenarioDefaults";
import { scenarioDraftKey, serializeScenarioDraft } from "@/features/land/scenarioDraft";
import { useAgentBus } from "@/features/land/surface/agentBus";
import { useLens } from "@/features/land/surface/lenses";
import { PlansLens } from "@/features/land/surface/lenses/PlansLens";
import { hatchLines, outlineFromCorners } from "@/features/land/surface/lenses/plans/plansMap";
import {
  plansHeadline,
  scenarioNumbers,
  staleReasonText,
  stepTiming,
} from "@/features/land/surface/lenses/plans/plansFormat";
import { useLandContext } from "@/state/landContext";
import { useLandMap } from "@/state/landMap";
import type * as IdentityModule from "@/state/landIdentity";
import { useMission } from "@/state/mission";

let canEdit = true;
vi.mock("@/state/landIdentity", async (original) => ({
  ...(await original<typeof IdentityModule>()),
  useLandCanEdit: () => canEdit,
}));

type Action = components["schemas"]["LandActionRead"];
type Scenario = components["schemas"]["ScenarioRead"];
type Polygon = components["schemas"]["Polygon"];

const square = (west: number, south: number, size: number): Polygon => ({
  type: "Polygon",
  coordinates: [
    [
      [west, south],
      [west + size, south],
      [west + size, south + size],
      [west, south + size],
      [west, south],
    ],
  ],
});
const boundary = square(-122.102, 47.6, 0.004);
const mowArea = square(-122.101, 47.601, 0.001);
const pond = square(-122.099, 47.602, 0.0005);
const land: LandArea = {
  id: "land",
  name: "Ranch",
  description: "",
  revision: 1,
  areaM2: 120_000,
  perimeterM: 1_400,
  boundary,
  source: { method: "drawn", label: "Drawn" },
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};
const meadow: Action = {
  id: "plan-meadow",
  landId: "land",
  revision: 1,
  title: "Meadow restoration",
  objective: "Native meadow",
  boundaryRevision: 1,
  scenario: { id: "scenario-meadow", revision: 1 },
  startDate: "2026-10-10",
  currency: "USD",
  steps: [
    {
      id: "mow",
      title: "Mow",
      startDay: 0,
      days: 3,
      successMeasure: "Cut to 10 cm",
      footprint: mowArea,
      estimatedCost: 1200,
      costBasis: "Contractor quote",
    },
    {
      id: "seed",
      title: "Seed native mix",
      startDay: 4,
      days: 1,
      successMeasure: "Seeded",
      footprint: mowArea,
      estimatedCost: 3400,
      costBasis: "Seed price × area",
    },
    { id: "monitor", title: "Monitor cover", startDay: 365, days: 1, successMeasure: "Survey" },
  ],
  exclusions: [pond],
  status: "draft",
  staleReasons: [],
  totalKnownCost: 4600,
  uncostedSteps: 1,
  effectiveBoundary: boundary,
  approvedBy: null,
  approvedAt: null,
  approvalNote: null,
  missionId: null,
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};
const fence: Action = {
  ...meadow,
  id: "plan-fence",
  title: "Fence repair",
  scenario: null,
  exclusions: [],
  steps: [{ id: "fix", title: "Fix posts", startDay: 0, days: 1, successMeasure: "Upright" }],
  totalKnownCost: 0,
};
const restorationScenario: Scenario = {
  id: "scenario-meadow",
  landId: "land",
  name: "Meadow option",
  boundaryRevision: 1,
  revision: 1,
  inputs: {
    ...restorationDefaults(),
    referenceEcosystem: "Native meadow",
    treatments: [{ name: "Mow", objective: "Open the sward", areaHa: 2, costPerHa: 500, year: 0 }],
  },
  result: {
    algorithm: "restoration-cover-and-cost/1",
    summary: { areaHa: 12, totalCost: 5000, presentValueCost: 4800, currency: "USD" },
    rows: [
      { class: "Unclassified", baselinePercent: 100, targetPercent: 0, changeHa: -12 },
      { class: "Reference vegetation", baselinePercent: 0, targetPercent: 100, changeHa: 12 },
    ],
    limitations: [],
  },
  stale: false,
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};
const solarScenario: Scenario = {
  ...restorationScenario,
  id: "scenario-solar",
  name: "Solar option",
  inputs: { ...solarDefaults(), usableRoofAreaM2: 100, installedCost: 12000 },
  result: {
    algorithm: "solar-cash-flow/1",
    summary: {
      capacityKwDc: 20,
      firstYearGenerationKwh: 28000,
      initialEquity: 12000,
      paybackYear: 9,
      currency: "USD",
    },
    rows: [],
    limitations: [],
  },
};

interface Routes {
  actions?: Action[];
  scenarios?: Scenario[];
  record?: Action;
  surveys?: unknown[];
  requests?: unknown;
  mission?: unknown;
}
const ok = (data: unknown) => Promise.resolve({ data, response: new Response() });
const missing = () =>
  Promise.resolve({ error: { title: "Not found" }, response: new Response(null, { status: 404 }) });
function route(routes: Routes) {
  const handler = (path: string) => {
    if (path.includes("/requests/")) return routes.requests ? ok(routes.requests) : missing();
    if (path.endsWith("/actions")) return ok(routes.actions ?? []);
    if (path.endsWith("/mission")) return ok(routes.mission ?? null);
    if (path.endsWith("/revisions")) return ok(routes.actions?.slice(0, 1) ?? []);
    if (path.endsWith("/actions/{action_id}"))
      return ok(routes.record ?? routes.actions?.[0] ?? null);
    if (path.endsWith("/scenarios")) return ok(routes.scenarios ?? []);
    if (path.endsWith("/surveys")) return ok(routes.surveys ?? []);
    return ok([]);
  };
  const spy = vi.isMockFunction(api.GET) ? vi.mocked(api.GET) : vi.spyOn(api, "GET");
  spy.mockImplementation(((path: string) => handler(path)) as typeof api.GET);
}

function mount(routes: Routes, scene?: SceneRegistry) {
  route(routes);
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  const wrap = (children: ReactNode) =>
    scene ? <SceneContext.Provider value={scene}>{children}</SceneContext.Provider> : children;
  return render(
    <QueryClientProvider client={cache}>{wrap(<PlansLens land={land} />)}</QueryClientProvider>,
  );
}

beforeEach(() => {
  canEdit = true;
});
afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
  useLandContext.getState().clear();
  useLens.getState().setLens("overview");
  useMission.setState({ project: null });
});

describe("plans card", () => {
  it("offers one way to start when there are no plans", async () => {
    mount({});
    expect(await screen.findByRole("heading", { name: "No plans yet." })).toBeVisible();
    expect(screen.getByRole("button", { name: "Make a plan" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Compare possibilities" }));
    expect(
      await screen.findByRole("heading", { name: "No possibilities to compare yet." }),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "New restoration scenario" })).toBeVisible();
  });

  it("sums up every plan, lists one plan's steps and draws them on the map", async () => {
    const view = mount({ actions: [meadow, fence], scenarios: [restorationScenario] });
    expect(
      await screen.findByRole("heading", {
        name: "Two plans: a three-step restoration ready to review, and one draft.",
      }),
    ).toBeVisible();
    const steps = screen.getByRole("list", { name: "Steps" });
    expect(within(steps).getAllByRole("button")).toHaveLength(3);
    expect(within(steps).getByText("$1,200")).toBeVisible();
    expect(within(steps).getByText("Whole area")).toBeVisible();
    expect(screen.getByText("$4,600 · 1 step has no estimate")).toBeVisible();
    expect(screen.getByText(/avoiding one area/)).toBeVisible();

    const layers = useLandContext.getState().layers;
    expect(layers["plans-steps"]?.features.map((feature) => feature.id)).toEqual(["mow", "seed"]);
    expect(layers["plans-work-area"]?.features).toHaveLength(1);
    const avoid = layers["plans-avoid"]?.features ?? [];
    expect(avoid[0]?.geometry).toEqual(pond);
    expect(avoid.slice(1).every((feature) => feature.geometry.type === "LineString")).toBe(true);
    const pins = useLandMap.getState().pins.plans ?? [];
    expect(pins.map((pin) => pin.label)).toEqual([
      expect.stringMatching(/^1 · Mow · [\d.]+ ac$/),
      expect.stringMatching(/^2 · Seed native mix · [\d.]+ ac$/),
      "3 · Monitor cover · whole area",
    ]);

    fireEvent.click(screen.getByRole("button", { name: "Fence repair" }));
    expect(
      await screen.findByRole("heading", {
        name: "Two plans: a one-step plan ready to review, and one draft.",
      }),
    ).toBeVisible();
    await waitFor(() => expect(useLandContext.getState().layers["plans-avoid"]).toBeUndefined());

    view.unmount();
    expect(Object.keys(useLandContext.getState().layers).some((id) => id.startsWith("plans"))).toBe(
      false,
    );
    expect(useLandMap.getState().pins.plans).toBeUndefined();
  });

  it("approves a reviewed draft with a note, and never on its own", async () => {
    const post = vi
      .spyOn(api, "POST")
      .mockResolvedValue({ data: { ...meadow, status: "approved" }, response: new Response() });
    mount({ actions: [meadow], scenarios: [restorationScenario] });
    fireEvent.click(await screen.findByRole("button", { name: "Review and approve" }));
    const approve = screen.getByRole("button", { name: "Approve revision 1" });
    expect(approve).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Review note"), { target: { value: "Walked it" } });
    fireEvent.click(approve);
    await waitFor(() => expect(post).toHaveBeenCalledOnce());
    const [path, options] = post.mock.calls[0] as unknown as [string, { body: unknown }];
    expect(path).toBe("/api/v1/land/{land_id}/actions/{action_id}/approve");
    expect(options.body).toEqual({ expectedRevision: 1, note: "Walked it" });
  });

  it("asks for a revision, in plain words, when the land changed or questions are open", async () => {
    mount({
      actions: [
        {
          ...meadow,
          staleReasons: [
            "The land boundary has changed; review the action against the current boundary.",
            "Inventory feature 0b9c5a52-1111-4a5b-9c1d-2f6e8a7b9c0d has changed since this action was drafted.",
          ],
        },
      ],
    });
    expect(await screen.findByText("The land changed since this plan was made.")).toBeVisible();
    expect(screen.getByText("The land's outline changed.")).toBeVisible();
    expect(screen.getByText("A mapped asset it uses changed.")).toBeVisible();
    expect(screen.queryByText(/0b9c5a52/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Review and approve" })).toBeNull();
    expect(screen.getByRole("button", { name: "Revise plan" })).toBeVisible();
    cleanup();

    mount({ actions: [{ ...meadow, constraints: [{ text: "Confirm access", resolved: false }] }] });
    expect(await screen.findByText(/One question to settle/)).toBeVisible();
    expect(screen.queryByRole("button", { name: "Review and approve" })).toBeNull();
  });

  it("schedules an approved plan as a mission under the open project", async () => {
    useMission.setState({
      project: { id: "project-1", name: "North field" } as ReturnType<
        typeof useMission.getState
      >["project"],
    });
    const post = vi.spyOn(api, "POST").mockResolvedValue({ data: {}, response: new Response() });
    mount({ actions: [{ ...meadow, status: "approved", approvedAt: "2026-10-10" }] });
    fireEvent.click(await screen.findByRole("button", { name: "Schedule as a mission" }));
    fireEvent.change(screen.getByLabelText("Scheduling note"), { target: { value: "Go" } });
    fireEvent.change(screen.getByLabelText("Schedule under"), { target: { value: "project-1" } });
    fireEvent.click(screen.getByRole("button", { name: "Schedule mission" }));
    await waitFor(() => expect(post).toHaveBeenCalledOnce());
    const [path, options] = post.mock.calls[0] as unknown as [string, { body: unknown }];
    expect(path).toBe("/api/v1/land/{land_id}/actions/{action_id}/mission");
    expect(options.body).toEqual({ expectedRevision: 1, note: "Go", projectId: "project-1" });
  });

  it("shows the mission a scheduled plan became", async () => {
    mount({
      actions: [{ ...meadow, status: "scheduled", missionId: "mission-1" }],
      mission: { startDate: "2026-10-10", endDate: "2026-10-12", status: "scheduled" },
    });
    expect(await screen.findByText(/Mission scheduled for/)).toBeVisible();
    expect(screen.queryByRole("button", { name: /Approve/ })).toBeNull();
  });

  it("is read-only for viewers", async () => {
    canEdit = false;
    mount({ actions: [meadow] });
    expect(await screen.findByText("Waiting for review.")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Review and approve" })).toBeNull();
    expect(screen.queryByRole("button", { name: "New plan" })).toBeNull();
    cleanup();
    mount({});
    await screen.findByRole("heading", { name: "No plans yet." });
    expect(screen.queryByRole("button", { name: "Make a plan" })).toBeNull();
  });

  it("hands the agent a question instead of starting it", async () => {
    mount({ actions: [meadow] });
    fireEvent.click(await screen.findByRole("button", { name: /Ask about this plan/ }));
    expect(useAgentBus.getState().request?.question).toContain("Meadow restoration");
  });
});

describe("making and revising plans", () => {
  it("saves a new plan and returns to it", async () => {
    const post = vi
      .spyOn(api, "POST")
      .mockResolvedValue({ data: { ...fence, id: "new-plan" }, response: new Response() });
    mount({});
    fireEvent.click(await screen.findByRole("button", { name: "Make a plan" }));
    fireEvent.change(screen.getByLabelText("Plan name"), { target: { value: "Clear brush" } });
    fireEvent.change(screen.getByLabelText("Goal"), { target: { value: "Open understory" } });
    fireEvent.change(screen.getByLabelText("Step 1 name"), { target: { value: "Cut" } });
    fireEvent.change(screen.getByLabelText("Step 1 success measure"), {
      target: { value: "No brush" },
    });
    fireEvent.change(screen.getByLabelText("Step 1 cost"), { target: { value: "800" } });
    fireEvent.change(screen.getByLabelText("Step 1 cost basis"), { target: { value: "Quote" } });
    fireEvent.click(screen.getByRole("button", { name: "Save plan" }));
    await waitFor(() => expect(post).toHaveBeenCalledOnce());
    const [path, options] = post.mock.calls[0] as unknown as [
      string,
      { body: components["schemas"]["LandActionCreate"] },
    ];
    expect(path).toBe("/api/v1/land/{land_id}/actions");
    expect(options.body).toMatchObject({
      title: "Clear brush",
      objective: "Open understory",
      boundaryRevision: 1,
      steps: [{ title: "Cut", successMeasure: "No brush", estimatedCost: 800, costBasis: "Quote" }],
    });
    await waitFor(() => expect(screen.queryByLabelText("Plan name")).toBeNull());
    expect(localStorage.getItem(actionDraftKey("pilot", land.id))).toBeNull();
  });

  it("revises without sending read-only fields and does not approve on save", async () => {
    const put = vi
      .spyOn(api, "PUT")
      .mockResolvedValue({ data: { ...meadow, revision: 2 }, response: new Response() });
    const post = vi.spyOn(api, "POST");
    mount({ actions: [meadow] });
    fireEvent.click(await screen.findByRole("button", { name: "Revise" }));
    fireEvent.change(screen.getByLabelText("Plan name"), { target: { value: "Revised meadow" } });
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(put).toHaveBeenCalledOnce());
    const body = (put.mock.calls[0] as unknown as [string, { body: Record<string, unknown> }])[1]
      .body;
    expect(body).toMatchObject({
      title: "Revised meadow",
      expectedRevision: 1,
      boundaryRevision: 1,
    });
    for (const key of ["id", "landId", "status", "approvedAt", "effectiveBoundary", "missionId"])
      expect(body).not.toHaveProperty(key);
    expect(post).not.toHaveBeenCalled();
  });

  it("draws a step's area on the map, corner by corner", async () => {
    const corners = [
      [-122.1015, 47.6015],
      [-122.1005, 47.6015],
      [-122.1005, 47.6025],
    ];
    let pending: ((value: null) => void) | null = null;
    const registry = new SceneRegistry();
    registry.set({
      areas: {
        pickGround: vi.fn(() => {
          const next = corners.shift();
          if (next) return Promise.resolve({ longitude: next[0], latitude: next[1], x: 0, y: 0 });
          return new Promise((resolve) => {
            pending = resolve;
          });
        }),
        cancelPick: vi.fn(() => {
          pending?.(null);
          pending = null;
        }),
      },
      camera: { flyToRectangle: vi.fn() },
    } as unknown as CesiumSceneManager);
    mount({}, registry);
    fireEvent.click(await screen.findByRole("button", { name: "Make a plan" }));
    fireEvent.click(screen.getByRole("button", { name: "Draw on map" }));
    expect(await screen.findByText("Click the map to add corners · 3 so far")).toBeVisible();
    expect(useLandContext.getState().layers["plans-drawing"]?.features).toHaveLength(4);
    fireEvent.click(screen.getByRole("button", { name: "Finish area" }));
    await waitFor(() =>
      expect(useLandContext.getState().layers["plans-steps"]?.features).toHaveLength(1),
    );
    expect(useLandContext.getState().layers["plans-drawing"]).toBeUndefined();
    expect(useLandContext.getState().pointPicker).toBeNull();
    expect(screen.getByRole("button", { name: "Redraw" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Use the whole area" })).toBeVisible();
  });

  it("keeps a changed draft when leaving the editor, and recovers it with blank numbers intact", async () => {
    const view = mount({});
    fireEvent.click(await screen.findByRole("button", { name: "Make a plan" }));
    fireEvent.change(screen.getByLabelText("Plan name"), { target: { value: "Unfinished work" } });
    fireEvent.change(screen.getByLabelText("Step 1 duration (days)"), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Back to plans" }));
    expect(await screen.findByRole("region", { name: "Recover plan draft" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Make a plan" })).toBeNull();
    view.unmount();

    const post = vi.spyOn(api, "POST");
    const put = vi.spyOn(api, "PUT");
    mount({});
    fireEvent.click(await screen.findByRole("button", { name: "Recover plan draft" }));
    expect(await screen.findByDisplayValue("Unfinished work")).toBeVisible();
    expect(screen.getByLabelText("Step 1 duration (days)")).toHaveValue(null);
    expect(post).not.toHaveBeenCalled();
    expect(put).not.toHaveBeenCalled();
  });

  it("keeps the form working and offers a download when browser storage is full", async () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("Full", "QuotaExceededError");
    });
    mount({});
    fireEvent.click(await screen.findByRole("button", { name: "Make a plan" }));
    fireEvent.change(screen.getByLabelText("Plan name"), { target: { value: "Keep my work" } });
    expect(
      await screen.findByText(
        "This browser could not keep a copy of this draft. Download one before leaving.",
      ),
    ).toBeVisible();
    expect(screen.getByLabelText("Plan name")).toHaveValue("Keep my work");
    expect(screen.getByRole("button", { name: "Download draft" })).toBeEnabled();
  });

  it("leaves nothing behind when an untouched new plan is closed", async () => {
    mount({});
    fireEvent.click(await screen.findByRole("button", { name: "Make a plan" }));
    fireEvent.click(screen.getByRole("button", { name: "Back to plans" }));
    expect(await screen.findByRole("button", { name: "Make a plan" })).toBeVisible();
    expect(localStorage.getItem(actionDraftKey("pilot", land.id))).toBeNull();
  });

  it("opens a draft that was in fact saved, without approving or scheduling it", async () => {
    const key = actionDraftKey("pilot", land.id);
    localStorage.setItem(
      key,
      serializeActionDraft({
        version: 1,
        landId: land.id,
        editing: null,
        draft: { ...actionPayload(meadow), requestKey: crypto.randomUUID() },
      }),
    );
    const approved: Action = { ...meadow, status: "approved", approvedAt: "2026-10-10" };
    const post = vi.spyOn(api, "POST");
    mount({
      actions: [approved],
      record: approved,
      requests: { saved: meadow, current: approved },
    });
    fireEvent.click(await screen.findByRole("button", { name: "Recover plan draft" }));
    expect(
      await screen.findByText(
        "This draft was already saved. Opened the plan without approving or scheduling it.",
      ),
    ).toBeVisible();
    expect(localStorage.getItem(key)).toBeNull();
    expect(await screen.findByRole("button", { name: "Schedule as a mission" })).toBeVisible();
    expect(post).not.toHaveBeenCalled();
  });

  it("asks for a choice when the plan was approved while an edit was closed", async () => {
    localStorage.setItem(
      actionDraftKey("pilot", land.id),
      serializeActionDraft({
        version: 1,
        landId: land.id,
        editing: actionEdit(meadow),
        draft: { ...actionPayload(meadow), requestKey: crypto.randomUUID(), title: "My changes" },
      }),
    );
    mount({ actions: [meadow], record: { ...meadow, status: "approved" } });
    fireEvent.click(await screen.findByRole("button", { name: "Recover plan draft" }));
    const region = await screen.findByRole("region", { name: "Review changed plan" });
    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
    fireEvent.click(within(region).getByRole("button", { name: "Keep my work as a new plan" }));
    expect(screen.getByLabelText("Plan name")).toHaveValue("My changes");
    expect(screen.getByRole("button", { name: "Save plan" })).toBeEnabled();
  });
});

describe("comparing possibilities", () => {
  it("compares area, cost and outcome, and starts a plan from a scenario", async () => {
    mount({ actions: [], scenarios: [restorationScenario, solarScenario] });
    fireEvent.click(await screen.findByRole("button", { name: "Compare possibilities" }));
    const table = await screen.findByRole("table", { name: "Scenario comparison" });
    expect(within(table).getByText("$5,000")).toBeVisible();
    expect(within(table).getByText("Reference vegetation 0% → 100%")).toBeVisible();
    expect(within(table).getByText("28,000 kWh a year, pays back in year 9")).toBeVisible();
    fireEvent.click(within(table).getByRole("button", { name: "Open Solar option in Solar" }));
    expect(useLens.getState().lens).toBe("solar");
    fireEvent.click(within(table).getByRole("button", { name: "Start a plan from Meadow option" }));
    expect(await screen.findByLabelText("Plan name")).toHaveValue("Meadow option");
    expect(screen.getByLabelText("Step 1 name")).toHaveValue("Mow");
    expect(screen.getByLabelText("Step 2 name")).toHaveValue("Monitor restoration in year 1");
  });

  it("calculates a restoration scenario with its surveys before it can be saved", async () => {
    const result = restorationScenario.result;
    const post = vi
      .spyOn(api, "POST")
      .mockResolvedValue({ data: result, response: new Response() });
    mount({
      surveys: [
        { id: "survey", name: "Measured plots", boundaryRevision: 1, observedOn: "2025-07-01" },
      ],
    });
    fireEvent.click(await screen.findByRole("button", { name: "Compare possibilities" }));
    fireEvent.click(await screen.findByRole("button", { name: "New restoration scenario" }));
    fireEvent.change(screen.getByLabelText("Reference ecosystem"), {
      target: { value: "Native meadow" },
    });
    fireEvent.click(await screen.findByRole("checkbox", { name: /Measured plots/ }));
    expect(screen.getByRole("button", { name: "Save scenario" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Calculate scenario" }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Save scenario" })).toBeEnabled(),
    );
    const [path, options] = post.mock.calls[0] as unknown as [
      string,
      { body: { fieldSurveyIds: string[]; inputs: { cover: unknown } } },
    ];
    expect(path).toBe("/api/v1/land/{land_id}/scenarios/preview");
    expect(options.body.fieldSurveyIds).toEqual(["survey"]);
    expect(options.body.inputs.cover).toEqual(restorationDefaults().cover);
    fireEvent.change(screen.getByLabelText("Cost per monitoring visit"), {
      target: { value: "90" },
    });
    expect(screen.getByRole("button", { name: "Save scenario" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Calculate scenario" }));
    await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
    const calls = post.mock.calls as unknown as [string, { body: { requestKey: string } }][];
    expect(calls[1]?.[1].body.requestKey).toBe(calls[0]?.[1].body.requestKey);
  });

  it("opens a scenario draft that was in fact saved, without saving it again", async () => {
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
          name: restorationScenario.name,
          boundaryRevision: 1,
          inputs: restorationScenario.inputs,
          evidenceIds: [],
          fieldSurveyIds: [],
          solarAssessmentId: null,
        },
      }),
    );
    const current = { ...restorationScenario, revision: 2, name: "Later meadow", stale: true };
    const post = vi.spyOn(api, "POST");
    const put = vi.spyOn(api, "PUT");
    mount({
      scenarios: [current],
      requests: { saved: restorationScenario, current },
    });
    fireEvent.click(await screen.findByRole("button", { name: /Compare possibilities/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Recover scenario draft" }));
    expect(
      await screen.findByText(
        "This draft was already saved. Opened the scenario without saving it again.",
      ),
    ).toBeVisible();
    expect(localStorage.getItem(scenarioDraftKey("pilot", land.id))).toBeNull();
    const table = await screen.findByRole("table", { name: "Scenario comparison" });
    expect(within(table).getByText("Earlier outline")).toBeVisible();
    expect(post).not.toHaveBeenCalled();
    expect(put).not.toHaveBeenCalled();
  });

  it("keeps a newer scenario revision intact while recovering local assumptions", async () => {
    localStorage.setItem(
      scenarioDraftKey("pilot", land.id),
      serializeScenarioDraft({
        version: 1,
        landId: land.id,
        requestKey: crypto.randomUUID(),
        editing: { id: restorationScenario.id, revision: 1, landId: land.id },
        monitoringText: "1, 3, 5",
        payload: {
          name: "Local assumptions",
          boundaryRevision: 1,
          inputs: restorationScenario.inputs,
          evidenceIds: [],
          fieldSurveyIds: [],
          solarAssessmentId: null,
        },
      }),
    );
    vi.spyOn(api, "GET");
    mount({ scenarios: [{ ...restorationScenario, revision: 2 }] });
    vi.mocked(api.GET).mockImplementation(((path: string) =>
      path.includes("/requests/")
        ? missing()
        : ok(
            path.endsWith("/{scenario_id}") ? { ...restorationScenario, revision: 2 } : [],
          )) as typeof api.GET);
    fireEvent.click(await screen.findByRole("button", { name: /Compare possibilities/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Recover scenario draft" }));
    await screen.findByRole("region", { name: "Review newer scenario revision" });
    expect(screen.getByRole("button", { name: "Save scenario revision" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Keep my assumptions as a new scenario" }));
    expect(screen.getByLabelText("Scenario name")).toHaveValue("Local assumptions");
    expect(screen.queryByRole("region", { name: "Review newer scenario revision" })).toBeNull();
    expect(screen.getByRole("button", { name: "Save scenario" })).toBeDisabled();
  });

  it("keeps an unreadable scenario draft for download instead of replacing it", async () => {
    const key = scenarioDraftKey("pilot", land.id);
    const invalid = JSON.stringify({ version: 1, landId: "different-land" });
    localStorage.setItem(key, invalid);
    mount({});
    fireEvent.click(await screen.findByRole("button", { name: /Compare possibilities/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Recover scenario draft" }));
    await screen.findByRole("alert");
    expect(localStorage.getItem(key)).toBe(invalid);
    expect(screen.getByRole("button", { name: "Download scenario draft" })).toBeEnabled();
    expect(screen.queryByLabelText("Scenario name")).toBeNull();
    expect(screen.getByRole("button", { name: "New restoration scenario" })).toBeVisible();
  });

  it("recovers a restoration draft with its species goals, and never edits a solar draft here", async () => {
    const draft = (
      inputs: ReturnType<typeof restorationDefaults> | ReturnType<typeof solarDefaults>,
    ) =>
      serializeScenarioDraft({
        version: 1,
        landId: land.id,
        requestKey: crypto.randomUUID(),
        editing: null,
        monitoringText: "1, 3,",
        payload: {
          name: "Recovered meadow",
          boundaryRevision: 1,
          inputs,
          evidenceIds: [],
          fieldSurveyIds: [],
          solarAssessmentId: null,
        },
      });
    localStorage.setItem(
      scenarioDraftKey("pilot", land.id),
      draft({
        ...restorationDefaults(),
        monitoringCostPerVisit: Number.NaN,
        ecology: {
          referenceBasis: "",
          siteConstraints: "",
          referenceEvidenceIds: [],
          speciesTargets: [],
        },
      }),
    );
    const view = mount({});
    expect(await screen.findByText("An unfinished scenario is waiting")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: /Compare possibilities/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Recover scenario draft" }));
    expect(await screen.findByDisplayValue("Recovered meadow")).toBeVisible();
    expect(screen.getByLabelText("Monitoring years (comma separated)")).toHaveValue("1, 3,");
    expect(screen.getByLabelText("Cost per monitoring visit")).toHaveValue(null);
    expect(
      screen.getByRole("region", { name: "Species targets and monitoring" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save scenario" })).toBeDisabled();
    view.unmount();

    localStorage.setItem(scenarioDraftKey("pilot", land.id), draft(solarDefaults()));
    mount({});
    fireEvent.click(await screen.findByRole("button", { name: "Compare possibilities" }));
    const region = await screen.findByRole("region", { name: "Recover scenario draft" });
    expect(within(region).queryByRole("button", { name: "Recover scenario draft" })).toBeNull();
    fireEvent.click(within(region).getByRole("button", { name: "Discard scenario draft" }));
    expect(localStorage.getItem(scenarioDraftKey("pilot", land.id))).toBeNull();
  });
});

describe("plan wording and map shapes", () => {
  it("writes the headline from the plans' states", () => {
    const approved = { ...meadow, id: "b", status: "approved" as const };
    const scheduled = { ...meadow, id: "c", status: "scheduled" as const };
    expect(plansHeadline([meadow], [restorationScenario], meadow)).toBe(
      "One plan: a three-step restoration, ready to review.",
    );
    expect(plansHeadline([meadow, approved, scheduled, fence], [], meadow)).toBe(
      "Four plans: one scheduled, one approved and two drafts.",
    );
    expect(staleReasonText(["The supporting scenario has changed; review it."])).toEqual([
      "The scenario it came from changed.",
    ]);
  });

  it("dates steps once a plan is scheduled", () => {
    const step = { id: "s", title: "Mow", startDay: 2, days: 3, successMeasure: "Done" };
    expect(stepTiming(step, { status: "draft", startDate: "2026-10-10" })).toBe("Day 3–5");
    const today = new Date(2026, 9, 13);
    expect(stepTiming(step, { status: "scheduled", startDate: "2026-10-10" }, today)).toMatch(
      /^Now, until /,
    );
    expect(stepTiming(step, { status: "scheduled", startDate: "2026-10-20" }, today)).toMatch(
      /^Starts /,
    );
  });

  it("hatches an area to avoid with lines inside it", () => {
    const lines = hatchLines(pond);
    expect(lines.length).toBeGreaterThanOrEqual(3);
    for (const line of lines)
      for (const [lon = 0, lat = 0] of line) {
        expect(lon).toBeGreaterThanOrEqual(-122.099 - 1e-9);
        expect(lon).toBeLessThanOrEqual(-122.0985 + 1e-9);
        expect(lat).toBeGreaterThanOrEqual(47.602 - 1e-9);
        expect(lat).toBeLessThanOrEqual(47.6025 + 1e-9);
      }
  });

  it("refuses an outline that crosses itself", () => {
    expect(
      outlineFromCorners([
        [0, 0],
        [1, 1],
        [1, 0],
        [0, 1],
      ]),
    ).toEqual({ error: "The outline crosses itself. Undo a corner and try again." });
    expect(
      "area" in
        outlineFromCorners([
          [0, 0],
          [1, 0],
          [1, 1],
        ]),
    ).toBe(true);
  });

  it("reads three comparable numbers from each kind of scenario", () => {
    expect(scenarioNumbers(restorationScenario).map((item) => item.label)).toEqual([
      "Area changing",
      "Cost",
      "Outcome",
    ]);
    expect(scenarioNumbers(solarScenario)[1]).toMatchObject({
      label: "Upfront cost",
      amount: 12000,
    });
  });
});

it("never leaves its pick running after the lens closes", async () => {
  let pending: ((value: null) => void) | null = null;
  const cancelPick = vi.fn(() => {
    pending?.(null);
    pending = null;
  });
  const registry = new SceneRegistry();
  registry.set({
    areas: {
      pickGround: vi.fn(
        () =>
          new Promise((resolve) => {
            pending = resolve;
          }),
      ),
      cancelPick,
    },
    camera: { flyToRectangle: vi.fn() },
  } as unknown as CesiumSceneManager);
  const view = mount({}, registry);
  fireEvent.click(await screen.findByRole("button", { name: "Make a plan" }));
  fireEvent.click(screen.getByRole("button", { name: "Draw an area to avoid" }));
  await waitFor(() => expect(useLandContext.getState().pointPicker).toBe("plans-draw"));
  act(() => view.unmount());
  expect(cancelPick).toHaveBeenCalled();
  expect(useLandContext.getState().pointPicker).toBeNull();
});
