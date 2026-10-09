import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components, LandArea } from "@twin/contracts";
import { api } from "@/api/client";
import { LandActions } from "@/features/land/LandActions";

type Action = components["schemas"]["LandActionRead"];
const boundary: LandArea["boundary"] = {
  type: "Polygon",
  coordinates: [
    [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 0],
    ],
  ],
};
const land: LandArea = {
  id: "land",
  name: "Ranch",
  description: "",
  revision: 1,
  areaM2: 1000,
  perimeterM: 100,
  boundary,
  source: { method: "drawn", label: "Drawn" },
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};
const action: Action = {
  id: "action",
  landId: "land",
  revision: 1,
  title: "Field survey",
  objective: "Measure baseline cover",
  boundaryRevision: 1,
  startDate: "2026-10-10",
  currency: "USD",
  steps: [
    {
      id: "survey",
      title: "Survey meadow",
      startDay: 0,
      days: 2,
      successMeasure: "Record five quadrats",
    },
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
function mount(row: Action) {
  vi.spyOn(api, "GET").mockImplementation(((path: string) =>
    Promise.resolve({
      data: path.endsWith("/actions") ? [row] : path.endsWith("/{action_id}") ? row : [],
      response: new Response(),
    })) as typeof api.GET);
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={cache}>
      <LandActions land={land} />
    </QueryClientProvider>,
  );
}
afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
});

it("edits an action without submitting read-only approval fields and does not approve on save", async () => {
  const put = vi
    .spyOn(api, "PUT")
    .mockResolvedValue({ data: { ...action, revision: 2 }, response: new Response() });
  const post = vi.spyOn(api, "POST");
  mount(action);
  fireEvent.click(await screen.findByRole("button", { name: /Field survey.*draft/ }));
  fireEvent.click(screen.getByRole("button", { name: "Revise action" }));
  fireEvent.change(screen.getByLabelText("Action title"), { target: { value: "Revised survey" } });
  fireEvent.click(screen.getByRole("button", { name: "Save action revision" }));
  await waitFor(() => expect(put).toHaveBeenCalledOnce());
  const calls = put.mock.calls as unknown as [string, { body: Record<string, unknown> }][];
  expect(calls[0]?.[1].body).toMatchObject({
    title: "Revised survey",
    expectedRevision: 1,
    boundaryRevision: 1,
  });
  for (const key of [
    "id",
    "landId",
    "status",
    "approvedAt",
    "effectiveBoundary",
    "missionId",
    "staleReasons",
  ])
    expect(calls[0]?.[1].body).not.toHaveProperty(key);
  expect(post).not.toHaveBeenCalled();
});

it("keeps approval disabled when supporting information changed or constraints remain unresolved", async () => {
  const post = vi.spyOn(api, "POST");
  mount({
    ...action,
    staleReasons: ["The boundary changed"],
    constraints: [{ text: "Confirm access", resolved: false }],
  });
  fireEvent.click(await screen.findByRole("button", { name: /Field survey.*draft/ }));
  fireEvent.change(screen.getByLabelText("Review note"), { target: { value: "Reviewed" } });
  expect(screen.getByRole("button", { name: "Approve revision 1" })).toBeDisabled();
  expect(screen.getByText("The boundary changed")).toBeVisible();
  expect(post).not.toHaveBeenCalled();
});

it("recovers unfinished work and blank step timing without approving or scheduling", async () => {
  const first = mount(action);
  fireEvent.click(screen.getByRole("button", { name: "Plan an action" }));
  fireEvent.change(screen.getByLabelText("Action title"), {
    target: { value: "Unfinished field work" },
  });
  fireEvent.change(screen.getByLabelText("Duration (days)"), { target: { value: "" } });
  fireEvent.change(screen.getByLabelText("Assumptions (one per line)"), {
    target: { value: "Keep this line\n\nUnresolved assumption" },
  });
  first.unmount();
  mount(action);
  vi.mocked(api.GET).mockImplementation((path) =>
    Promise.resolve(
      String(path).includes("/requests/")
        ? { error: {}, response: new Response(null, { status: 404 }) }
        : { data: String(path).endsWith("/actions") ? [action] : [], response: new Response() },
    ),
  );
  const post = vi.spyOn(api, "POST"),
    put = vi.spyOn(api, "PUT");
  fireEvent.click(screen.getByRole("button", { name: "Recover action draft" }));
  await screen.findByDisplayValue("Unfinished field work");
  expect(screen.getByLabelText("Duration (days)")).toHaveValue(null);
  expect(screen.getByLabelText("Assumptions (one per line)")).toHaveValue(
    "Keep this line\n\nUnresolved assumption",
  );
  expect(post).not.toHaveBeenCalled();
  expect(put).not.toHaveBeenCalled();
  expect(screen.queryByRole("button", { name: /Approve revision/ })).toBeNull();
});

it("opens a successfully saved action without repeating its approval or mission handoff", async () => {
  const { actionDraftKey, actionPayload, serializeActionDraft } =
    await import("@/features/land/actionDraft");
  const key = actionDraftKey("pilot", land.id);
  localStorage.setItem(
    key,
    serializeActionDraft({
      version: 1,
      landId: land.id,
      editing: null,
      draft: { ...actionPayload(action), requestKey: crypto.randomUUID() },
    }),
  );
  mount(action);
  const approved: Action = {
    ...action,
    status: "approved",
    approvedAt: "2026-10-09",
    approvedBy: "operator",
    approvalNote: "Reviewed",
  };
  vi.mocked(api.GET).mockImplementation((path) =>
    Promise.resolve({
      data: String(path).includes("/requests/")
        ? { saved: action, current: approved }
        : String(path).endsWith("/{action_id}")
          ? approved
          : String(path).endsWith("/actions")
            ? [approved]
            : [],
      response: new Response(),
    }),
  );
  const post = vi.spyOn(api, "POST"),
    put = vi.spyOn(api, "PUT");
  fireEvent.click(screen.getByRole("button", { name: "Recover action draft" }));
  await screen.findByText(
    "This draft was already saved. Opened the current action without approving or scheduling it.",
  );
  await screen.findByRole("article", { name: "Action review" });
  expect(post).not.toHaveBeenCalled();
  expect(put).not.toHaveBeenCalled();
  expect(localStorage.getItem(key)).toBeNull();
  expect(screen.getByLabelText("Scheduling note")).toHaveValue("");
  expect(screen.getByRole("button", { name: "Schedule approved action" })).toBeDisabled();
});

it("requires review if an action was approved while the unfinished edit was closed", async () => {
  const { actionDraftKey, actionEdit, actionPayload, serializeActionDraft } =
    await import("@/features/land/actionDraft");
  localStorage.setItem(
    actionDraftKey("pilot", land.id),
    serializeActionDraft({
      version: 1,
      landId: land.id,
      editing: actionEdit(action),
      draft: {
        ...actionPayload(action),
        requestKey: crypto.randomUUID(),
        title: "My pending changes",
      },
    }),
  );
  mount(action);
  const approved: Action = { ...action, status: "approved" };
  vi.mocked(api.GET).mockImplementation((path) =>
    Promise.resolve(
      String(path).includes("/requests/")
        ? { error: {}, response: new Response(null, { status: 404 }) }
        : { data: String(path).endsWith("/{action_id}") ? approved : [], response: new Response() },
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Recover action draft" }));
  await screen.findByRole("region", { name: "Review changed action" });
  expect(screen.getByRole("button", { name: "Save action revision" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Keep my work as a new action" }));
  expect(screen.getByLabelText("Action title")).toHaveValue("My pending changes");
  expect(screen.getByRole("button", { name: "Save action draft" })).toBeEnabled();
});
