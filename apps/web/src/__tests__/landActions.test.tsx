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
      data: path.endsWith("/actions") ? [row] : [],
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
