/**
 * The Land → Plan loop on the Plans lens: "Plan work on this land", scheduling a plan as a
 * mission (where it is filed, Plan and Fleet refreshed, "Open in Plan"), why Approve or
 * Schedule is disabled, and the checklist that stops Save. Ported from the Land panel's
 * landActions tests onto the map-first lens.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { components, LandArea } from "@twin/contracts";

import { api } from "@/api/client";
import {
  actionDraftKey,
  actionEdit,
  actionPayload,
  serializeActionDraft,
} from "@/features/land/actionDraft";
import { useLens } from "@/features/land/surface/lenses";
import { PlansLens } from "@/features/land/surface/lenses/PlansLens";
import type * as PlanningModule from "@/features/mission/landPlanning";
import { anywhereProject } from "@/missions/anywhere";
import type { Project } from "@/missions/types";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import type * as IdentityModule from "@/state/landIdentity";
import { useMission } from "@/state/mission";
import { useUi } from "@/state/ui";

let canEdit = true;
vi.mock("@/state/landIdentity", async (original) => ({
  ...(await original<typeof IdentityModule>()),
  useLandCanEdit: () => canEdit,
}));
const planOnLand = vi.hoisted(() => vi.fn((_land: unknown) => Promise.resolve()));
const openLandMission = vi.hoisted(() =>
  vi.fn((_land: unknown, _record: unknown, _client: unknown) => Promise.resolve()),
);
vi.mock("@/features/mission/landPlanning", async (original) => ({
  ...(await original<typeof PlanningModule>()),
  planOnLand,
  openLandMission,
}));

type Action = components["schemas"]["LandActionRead"];
const boundary: LandArea["boundary"] = {
  type: "Polygon",
  coordinates: [
    [
      [-122.102, 47.6],
      [-122.098, 47.6],
      [-122.098, 47.604],
      [-122.102, 47.604],
      [-122.102, 47.6],
    ],
  ],
};
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
const survey: Action = {
  id: "action",
  landId: "land",
  revision: 1,
  title: "Field survey",
  objective: "Measure baseline cover",
  boundaryRevision: 1,
  startDate: "2026-10-10",
  currency: "USD",
  steps: [
    { id: "survey", title: "Survey meadow", startDay: 0, days: 2, successMeasure: "Five quadrats" },
  ],
  status: "draft",
  staleReasons: [],
  totalKnownCost: 0,
  uncostedSteps: 1,
  effectiveBoundary: boundary,
  approvedBy: null,
  approvedAt: null,
  approvalNote: null,
  missionId: null,
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};
const approved: Action = {
  ...survey,
  status: "approved",
  revision: 2,
  approvedAt: "2026-10-09",
  approvedBy: "operator",
  approvalNote: "Reviewed",
};
const mission = {
  id: "mission-1",
  projectId: "land:land",
  title: approved.title,
  startDate: "2026-10-10",
  endDate: "2026-10-11",
  status: "scheduled",
};

interface Routes {
  actions?: () => Action[];
  record?: Action;
  mission?: unknown;
  requests?: unknown;
}
const ok = (data: unknown) => Promise.resolve({ data, response: new Response() });
function route(routes: Routes) {
  const spy = vi.isMockFunction(api.GET) ? vi.mocked(api.GET) : vi.spyOn(api, "GET");
  spy.mockImplementation(((path: string) => {
    const actions = routes.actions?.() ?? [];
    if (path.includes("/requests/"))
      return routes.requests
        ? ok(routes.requests)
        : Promise.resolve({ error: {}, response: new Response(null, { status: 404 }) });
    if (path.endsWith("/actions")) return ok(actions);
    if (path.endsWith("/mission")) return ok(routes.mission ?? null);
    if (path.endsWith("/actions/{action_id}")) return ok(routes.record ?? actions[0] ?? null);
    if (path === "/api/v1/land/{land_id}") return ok(land);
    return ok([]);
  }) as typeof api.GET);
}
function mount(routes: Routes) {
  route(routes);
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  render(
    <QueryClientProvider client={cache}>
      <PlansLens land={land} />
    </QueryClientProvider>,
  );
  return cache;
}
const setProject = (project: Partial<Project>) =>
  useMission.setState({ project: { ...anywhereProject(), ...project } });

beforeEach(() => {
  canEdit = true;
});
afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
  planOnLand.mockClear();
  openLandMission.mockClear();
  useLandContext.getState().clear();
  useLens.getState().setLens("overview");
  useMission.setState({ project: null });
  useUi.getState().setPanel(null);
  useLand.setState({ active: null });
});

describe("Plan work on this land", () => {
  it("leads the card, with or without plans, and opens Plan on this land", async () => {
    mount({});
    const work = await screen.findByTestId("land-plan-work");
    expect(work).toHaveAccessibleName("Plan work on this land");
    expect(work).toBeEnabled();
    // The one primary action of an empty card: making a step-by-step plan is the other way.
    expect(await screen.findByRole("button", { name: "Make a plan" })).not.toHaveClass(
      "glass-button--primary",
    );
    fireEvent.click(work);
    expect(planOnLand).toHaveBeenCalledWith(land);
    cleanup();

    mount({ actions: () => [survey] });
    await screen.findByRole("button", { name: "Review and approve" });
    expect(screen.getByTestId("land-plan-work")).toBeEnabled();
  });

  it("says why a viewer cannot plan work here", async () => {
    canEdit = false;
    mount({ actions: () => [survey] });
    const work = await screen.findByTestId("land-plan-work");
    expect(work).toBeDisabled();
    expect(work).toHaveAccessibleDescription(
      "Viewers can see this land's plans. A workspace editor can plan work here.",
    );
    fireEvent.click(work);
    expect(planOnLand).not.toHaveBeenCalled();
  });
});

describe("scheduling a plan as a mission", () => {
  it("files it under the land's own plans, refreshes Plan and Fleet, and opens it in Plan", async () => {
    let scheduled = false;
    const post = vi.spyOn(api, "POST").mockImplementation((() => {
      scheduled = true;
      return ok(mission);
    }) as typeof api.POST);
    const cache = mount({
      actions: () => [
        scheduled ? { ...approved, status: "scheduled", missionId: "mission-1" } : approved,
      ],
      mission,
    });
    const invalidate = vi.spyOn(cache, "invalidateQueries");
    fireEvent.click(await screen.findByRole("button", { name: "Schedule as a mission" }));
    const under = screen.getByLabelText("Schedule under");
    expect(under).toHaveValue("land:land");
    expect(
      within(under)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["Ranch · this land's plans"]);
    fireEvent.change(screen.getByLabelText("Scheduling note"), { target: { value: "Go ahead" } });
    fireEvent.click(screen.getByRole("button", { name: "Schedule mission" }));
    await waitFor(() => expect(post).toHaveBeenCalledOnce());
    const [path, options] = post.mock.calls[0] as unknown as [string, { body: unknown }];
    expect(path).toBe("/api/v1/land/{land_id}/actions/{action_id}/mission");
    expect(options.body).toEqual({ expectedRevision: 2, note: "Go ahead", projectId: "land:land" });
    await waitFor(() => expect(invalidate).toHaveBeenCalledWith({ queryKey: ["workspace-plans"] }));
    const status = await screen.findByTestId("land-mission-status");
    expect(status).toHaveTextContent(
      "Mission scheduled for Oct 10 through Oct 11 under this land's own plans. Status: scheduled.",
    );
    fireEvent.click(screen.getByTestId("land-mission-open-plan"));
    expect(openLandMission).toHaveBeenCalledWith(land, mission, cache);
    expect(screen.getByTestId("land-mission-open-plan")).toHaveAccessibleName("Open in Plan");
  });

  it("files it with the fleet whose project works this land, simulated or not", async () => {
    setProject({
      id: "fleet",
      name: "North field",
      simulated: true,
      machines: [{ id: "m1", position: { longitude: -122.1, latitude: 47.602 } }] as never,
    });
    const post = vi.spyOn(api, "POST").mockImplementation((() => ok(mission)) as typeof api.POST);
    mount({ actions: () => [approved] });
    fireEvent.click(await screen.findByRole("button", { name: "Schedule as a mission" }));
    const under = screen.getByLabelText("Schedule under");
    expect(under).toHaveValue("fleet");
    expect(
      within(under)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual([
      "Ranch · this land's plans",
      "North field · current mission project (simulated fleet)",
    ]);
    fireEvent.change(screen.getByLabelText("Scheduling note"), { target: { value: "Go" } });
    fireEvent.click(screen.getByRole("button", { name: "Schedule mission" }));
    await waitFor(() => expect(post).toHaveBeenCalledOnce());
    expect(
      (post.mock.calls[0]?.[1] as unknown as { body: { projectId: string } }).body,
    ).toMatchObject({ projectId: "fleet" });
  });

  it("files it with a fleet's project that came into view after the plan did, until one is chosen", async () => {
    // The plan is on screen before the map reaches the fleet whose project works this land.
    setProject({ id: "land:land", name: "Ranch" });
    mount({ actions: () => [approved] });
    const schedule = await screen.findByRole("button", { name: "Schedule as a mission" });
    const fleet = (id: string, name: string) => {
      act(() => {
        setProject({
          id,
          name,
          machines: [{ id: "m1", position: { longitude: -122.1, latitude: 47.602 } }] as never,
        });
      });
    };
    fleet("fleet", "North field");
    fireEvent.click(schedule);
    const under = screen.getByLabelText("Schedule under");
    expect(under).toHaveValue("fleet");
    // Chosen, it stays chosen when another project comes into view.
    fireEvent.change(under, { target: { value: "land:land" } });
    fleet("fleet-2", "South field");
    expect(screen.getByLabelText("Schedule under")).toHaveValue("land:land");
  });

  it("never offers another land's project, and names no project by its id", async () => {
    setProject({ id: "land:other", name: "Creek Field" });
    mount({ actions: () => [approved] });
    fireEvent.click(await screen.findByRole("button", { name: "Schedule as a mission" }));
    expect(screen.getByLabelText("Schedule under")).toHaveValue("land:land");
    expect(screen.getAllByRole("option").map((option) => option.textContent)).toEqual([
      "Ranch · this land's plans",
    ]);
    cleanup();

    mount({
      actions: () => [{ ...approved, status: "scheduled", missionId: "mission-2" }],
      mission: { ...mission, id: "mission-2", projectId: "site-elsewhere", endDate: "2026-10-10" },
    });
    // A one-day mission names its day once.
    expect(await screen.findByTestId("land-mission-status")).toHaveTextContent(
      /^Mission scheduled for Oct 10 under another mission project\. Status: scheduled\.$/,
    );
    expect(screen.queryByText(/site-elsewhere/)).toBeNull();
  });
});

describe("why Approve or Schedule is disabled", () => {
  it("says beside a disabled Approve why, quietly until the note is left empty", async () => {
    const post = vi.spyOn(api, "POST").mockImplementation((() => ok(approved)) as typeof api.POST);
    mount({ actions: () => [survey] });
    fireEvent.click(await screen.findByRole("button", { name: "Review and approve" }));
    const approve = screen.getByRole("button", { name: "Approve revision 1" });
    expect(approve).toBeDisabled();
    const reasons = screen.getByRole("list", { name: "Before approving" });
    const reason = within(reasons)
      .getByText("Add a review note. It is stored with the approval.")
      .closest("li");
    expect(approve).toHaveAccessibleDescription(
      "Add a review note. It is stored with the approval.",
    );
    // A hint until the note is visited and left empty, then a warning on the note itself.
    const note = screen.getByLabelText("Review note");
    expect(reason).toHaveClass("is-hint");
    expect(note).not.toHaveAttribute("aria-invalid");
    fireEvent.blur(note);
    expect(reason).not.toHaveClass("is-hint");
    expect(note).toHaveAttribute("aria-invalid", "true");
    expect(note).toHaveAccessibleDescription("Add a review note. It is stored with the approval.");
    fireEvent.change(note, { target: { value: "Walked it" } });
    expect(screen.queryByRole("list", { name: "Before approving" })).toBeNull();
    expect(approve).toBeEnabled();
    expect(approve).not.toHaveAttribute("aria-describedby");
    expect(note).not.toHaveAttribute("aria-invalid");
    fireEvent.click(approve);
    await waitFor(() => expect(post).toHaveBeenCalledOnce());
  });

  it("offers the revision beside a reason only revising fixes", async () => {
    // The server never approves a plan with open questions; should one arrive, Schedule
    // says so and the revision is a click away.
    mount({
      actions: () => [{ ...approved, constraints: [{ text: "Confirm access", resolved: false }] }],
    });
    fireEvent.click(await screen.findByRole("button", { name: "Schedule as a mission" }));
    const schedule = screen.getByRole("button", { name: "Schedule mission" });
    expect(schedule).toBeDisabled();
    // The reasons' words, not the Revise button's.
    expect(schedule).toHaveAccessibleDescription(
      "Settle 1 open question: revise the plan, tick Settled and say how. Add a scheduling note. It is stored with the scheduled mission.",
    );
    const reasons = screen.getByRole("list", { name: "Before scheduling" });
    fireEvent.change(screen.getByLabelText("Scheduling note"), { target: { value: "Go" } });
    expect(schedule).toBeDisabled();
    fireEvent.click(within(reasons).getByRole("button", { name: "Revise plan" }));
    expect(await screen.findByLabelText("Plan name")).toHaveValue("Field survey");
    expect(screen.getByLabelText("Question 1")).toHaveValue("Confirm access");
  });

  it("names open questions on a plan the land changed under, and how to settle them", async () => {
    mount({
      actions: () => [
        {
          ...survey,
          staleReasons: ["The land boundary has changed."],
          constraints: [{ text: "Confirm access", resolved: false }],
        },
      ],
    });
    expect(await screen.findByText(/One question also to settle\./)).toBeVisible();
    cleanup();
    mount({
      actions: () => [{ ...survey, constraints: [{ text: "Confirm access", resolved: false }] }],
    });
    expect(
      await screen.findByText(
        "One question to settle before this plan can be approved. Revise it, tick Settled and say how.",
      ),
    ).toBeVisible();
  });
});

describe("the checklist that stops Save", () => {
  it("focuses the first problem of an empty plan and lists the rest", async () => {
    const post = vi.spyOn(api, "POST");
    mount({});
    fireEvent.click(await screen.findByRole("button", { name: "Make a plan" }));
    expect(screen.getByRole("form", { name: "New plan" })).toHaveAttribute("novalidate");
    fireEvent.click(screen.getByRole("button", { name: "Save plan" }));
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Before saving, fix 4 things:");
    expect(
      within(alert)
        .getAllByRole("button")
        .map((item) => item.textContent),
    ).toEqual([
      "Add a plan name.",
      "Describe the goal.",
      "Step 1: add a step name.",
      "Step 1: say how you will know it is done.",
    ]);
    const name = screen.getByLabelText("Plan name");
    expect(name).toHaveFocus();
    expect(name).toHaveAttribute("aria-invalid", "true");
    expect(name).toHaveAccessibleDescription("Add a plan name.");
    expect(screen.getByLabelText("Step 1 name")).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByRole("button", { name: "Save plan" })).toHaveAccessibleDescription(
      /^Before saving, fix 4 things:/,
    );
    expect(document.querySelectorAll(".plans-field-problem")).toHaveLength(4);
    // Typing a name clears its mark and its reason; the rest stay listed.
    fireEvent.change(name, { target: { value: "Fence line" } });
    expect(name).not.toHaveAttribute("aria-invalid");
    expect(screen.getByRole("alert")).toHaveTextContent("Before saving, fix 3 things:");
    expect(post).not.toHaveBeenCalled();
  });

  it("opens a folded section to reach its field, and saves once nothing is left", async () => {
    const post = vi
      .spyOn(api, "POST")
      .mockImplementation((() => ok({ ...survey, id: "new" })) as typeof api.POST);
    mount({});
    fireEvent.click(await screen.findByRole("button", { name: "Make a plan" }));
    fireEvent.change(screen.getByLabelText("Plan name"), { target: { value: "Fence line" } });
    fireEvent.change(screen.getByLabelText("Goal"), { target: { value: "Keep it clear" } });
    fireEvent.change(screen.getByLabelText("Step 1 name"), { target: { value: "Clear brush" } });
    fireEvent.change(screen.getByLabelText("Step 1 success measure"), {
      target: { value: "No brush over 30 cm" },
    });
    // Two fields folded away: the currency, and a question added and left empty.
    fireEvent.change(screen.getByLabelText("Currency"), { target: { value: "us" } });
    fireEvent.click(screen.getByRole("button", { name: "Add a question" }));
    const money = screen.getByText("Money, sources and references").closest("details");
    const questions = screen.getByText("Questions to settle and assumptions").closest("details");
    if (!money || !questions) throw new Error("The folded sections are missing");
    expect(money.open).toBe(false);
    expect(questions.open).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "Save plan" }));
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Before saving, fix 2 things:");
    const currency = screen.getByLabelText("Currency");
    expect(money.open).toBe(true);
    expect(currency).toHaveFocus();
    expect(currency).toHaveAccessibleDescription("Use a three-letter currency code, like USD.");
    // Each listed problem is a way back to its field.
    fireEvent.click(within(alert).getByRole("button", { name: /Question 1/ }));
    expect(questions.open).toBe(true);
    expect(screen.getByLabelText("Question 1")).toHaveFocus();
    expect(post).not.toHaveBeenCalled();

    // The list follows the plan: fixing the last problem clears it, and Save goes through.
    fireEvent.change(currency, { target: { value: "usd" } });
    fireEvent.change(screen.getByLabelText("Question 1"), { target: { value: "Gate code?" } });
    expect(screen.queryByRole("alert")).toBeNull();
    expect(currency).not.toHaveAttribute("aria-invalid");
    fireEvent.click(screen.getByRole("button", { name: "Save plan" }));
    await waitFor(() => expect(post).toHaveBeenCalledOnce());
  });

  it("asks a costed step where its cost comes from, under the step", async () => {
    mount({});
    fireEvent.click(await screen.findByRole("button", { name: "Make a plan" }));
    fireEvent.change(screen.getByLabelText("Plan name"), { target: { value: "Fence line" } });
    fireEvent.change(screen.getByLabelText("Goal"), { target: { value: "Keep it clear" } });
    fireEvent.change(screen.getByLabelText("Step 1 name"), { target: { value: "Clear brush" } });
    fireEvent.change(screen.getByLabelText("Step 1 success measure"), {
      target: { value: "No brush" },
    });
    fireEvent.change(screen.getByLabelText("Step 1 cost"), { target: { value: "450" } });
    fireEvent.click(screen.getByRole("button", { name: "Save plan" }));
    const basis = screen.getByLabelText("Step 1 cost basis");
    expect(basis).toHaveFocus();
    expect(basis).toHaveAccessibleDescription("Step 1: say where its 450 USD cost comes from.");
  });

  it("says why Save is disabled while the plan changed elsewhere", async () => {
    localStorage.setItem(
      actionDraftKey("pilot", land.id),
      serializeActionDraft({
        version: 1,
        landId: land.id,
        editing: actionEdit(survey),
        draft: { ...actionPayload(survey), requestKey: crypto.randomUUID(), title: "My changes" },
      }),
    );
    mount({ actions: () => [survey], record: { ...survey, status: "approved" } });
    fireEvent.click(await screen.findByRole("button", { name: "Recover plan draft" }));
    await screen.findByRole("region", { name: "Review changed plan" });
    const save = screen.getByRole("button", { name: "Save changes" });
    expect(save).toBeDisabled();
    expect(save).toHaveAccessibleDescription(
      "Choose how to handle the changed plan above before saving.",
    );
  });
});

it("opens Land on the Plans lens from Plan's “Revise the action in Land”", async () => {
  route({});
  const { openLandActions } = await vi.importActual<typeof PlanningModule>(
    "@/features/mission/landPlanning",
  );
  useLens.getState().setLens("assets");
  await openLandActions(land.id);
  expect(useLand.getState().active?.id).toBe(land.id);
  expect(useUi.getState().activePanel).toBe("land");
  expect(useLens.getState().lens).toBe("plans");
});
