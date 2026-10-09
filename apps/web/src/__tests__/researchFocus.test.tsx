import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components, LandArea } from "@twin/contracts";
import { api } from "@/api/client";
import { LandResearch } from "@/features/land/LandResearch";
import { LandMapSelection } from "@/features/land/LandMapSelection";
import {
  researchDraftKey,
  parseResearchDraft,
  type ResearchDraft,
} from "@/features/land/researchDraft";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";

vi.mock("@/cesium/SceneContext", () => ({ useScene: () => null }));
const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Public fixture",
  description: "",
  revision: 1,
  areaM2: 100,
  perimeterM: 40,
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
  source: { method: "drawn", label: "Fixture" },
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-77.05, 38.89],
        [-77.04, 38.89],
        [-77.04, 38.9],
        [-77.05, 38.89],
      ],
    ],
  },
};
const inv = {
  id: "22222222-2222-4222-8222-222222222222",
  landId: land.id,
  title: "Fixture investigation",
  question: "Explore",
  boundaryRevision: 1,
  createdAt: "2026-10-09",
  stale: false,
};
const focus = {
  artifactId: "33333333-3333-4333-8333-333333333333",
  featureIndex: 0,
  label: "Survey point",
};
function mount() {
  useLand.getState().select(land);
  vi.spyOn(api, "GET").mockImplementation((path: string) =>
    Promise.resolve({
      data: path.endsWith("/status")
        ? { modelConfigured: true, model: "fixture" }
        : path.endsWith("/investigations")
          ? [inv]
          : {
              investigation: inv,
              runs: [],
              messages: [],
              findings: [],
              evidence: [],
              artifacts: [],
              page: { offset: 0, limit: 100, totals: {} },
            },
      response: new Response(),
    }),
  );
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <LandResearch land={land} />
      <LandMapSelection land={land} />
    </QueryClientProvider>,
  );
}
afterEach(() => {
  cleanup();
  useLandContext.getState().clear();
  vi.restoreAllMocks();
  localStorage.clear();
});

it("carries map selection into the question while preserving an existing draft", () => {
  useLandContext.getState().setLayer({
    id: focus.artifactId,
    researchArtifactId: focus.artifactId,
    title: "Survey",
    features: [
      {
        id: "0",
        label: focus.label,
        value: 0,
        geometry: { type: "Point", coordinates: [-77.05, 38.89] },
      },
    ],
  });
  useLandContext.getState().selectMapFeature(focus.artifactId, "0", "list");
  useLandContext.getState().setResearchQuestion("Explain the uncertainty");
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Ask about this feature" }));
  expect(useLandContext.getState().researchFocus).toEqual(focus);
  expect(screen.getByLabelText("Follow your curiosity")).toHaveValue("Explain the uncertainty");
  fireEvent.click(screen.getByRole("button", { name: "Clear question focus" }));
  expect(useLandContext.getState().researchFocus).toBeNull();
  expect(screen.getByLabelText("Follow your curiosity")).toHaveValue("Explain the uncertainty");
});

it("retries the same pinned reference and request key after a lost queue response", async () => {
  const post = vi.spyOn(api, "POST").mockRejectedValue(new Error("Lost queue response"));
  useLandContext.getState().setResearchQuestion("What about this?");
  useLandContext.getState().setResearchFocus(focus);
  mount();
  const ask = screen.getByRole("button", { name: "Investigate" });
  await waitFor(() => expect(ask).toBeEnabled());
  fireEvent.click(ask);
  await screen.findByRole("alert");
  fireEvent.click(await screen.findByRole("button", { name: "Retry captured request" }));
  await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
  const first = (post.mock.calls[0]?.[1] as unknown as { body: components["schemas"]["RunCreate"] })
    .body;
  expect(first).toMatchObject({ focus: { artifactId: focus.artifactId, featureIndex: 0 } });
  expect(first?.focus).not.toHaveProperty("label");
  expect(
    (post.mock.calls[1]?.[1] as unknown as { body: components["schemas"]["RunCreate"] }).body,
  ).toEqual(first);
});

it("retains a newer question and focus when an earlier request finishes", async () => {
  let resolve!: (value: { data: unknown; response: Response }) => void;
  vi.spyOn(api, "POST").mockImplementation(
    () =>
      new Promise((done) => {
        resolve = done;
      }),
  );
  useLandContext.getState().setResearchQuestion("First question");
  useLandContext.getState().setResearchFocus(focus);
  mount();
  const ask = screen.getByRole("button", { name: "Investigate" });
  await waitFor(() => expect(ask).toBeEnabled());
  fireEvent.click(ask);
  await waitFor(() => expect(resolve).toBeDefined());
  act(() => {
    useLandContext.getState().setResearchQuestion("Next question");
    useLandContext
      .getState()
      .setResearchFocus({ ...focus, featureIndex: 1, label: "Another point" });
    resolve({ data: {}, response: new Response() });
  });
  await waitFor(() => expect(ask).toBeEnabled());
  expect(useLandContext.getState().researchQuestion).toBe("Next question");
  expect(useLandContext.getState().researchFocus?.featureIndex).toBe(1);
});

it("recovers a draft and retries the original request after a reload without overwriting a newer question", async () => {
  const post = vi
    .spyOn(api, "POST")
    .mockRejectedValueOnce(new Error("Lost response"))
    .mockResolvedValue({ data: {}, response: new Response() });
  useLandContext.getState().setResearchQuestion("First question");
  useLandContext.getState().setResearchFocus(focus);
  const view = mount();
  await waitFor(() => expect(screen.getByRole("button", { name: "Investigate" })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: "Investigate" }));
  await screen.findByRole("button", { name: "Retry captured request" });
  const first = post.mock.calls[0]?.[1];
  fireEvent.change(screen.getByLabelText("Follow your curiosity"), {
    target: { value: "A newer question" },
  });
  fireEvent.change(screen.getByLabelText("Time budget"), { target: { value: "600" } });
  const stored = parseResearchDraft(
    localStorage.getItem(researchDraftKey("pilot", land.id))!,
    land.id,
  );
  expect(stored.question).toBe("A newer question");
  expect(stored.pending?.question).toBe("First question");
  expect(stored.pending?.budget.maxSeconds).toBe(180);
  view.unmount();
  useLandContext.getState().clear();
  mount();
  expect(screen.getByLabelText("Follow your curiosity")).toHaveValue("A newer question");
  expect(screen.getByLabelText("Time budget")).toHaveValue("600");
  expect(useLandContext.getState().researchFocus).toEqual(focus);
  expect(post).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: "Retry captured request" }));
  await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
  expect(post.mock.calls[1]?.[1]).toEqual(first);
  await waitFor(() =>
    expect(
      screen.queryByRole("region", { name: "Captured research request" }),
    ).not.toBeInTheDocument(),
  );
  expect(screen.getByLabelText("Follow your curiosity")).toHaveValue("A newer question");
  expect(
    parseResearchDraft(localStorage.getItem(researchDraftKey("pilot", land.id))!, land.id).pending,
  ).toBeNull();
});

it("rejects a recovered investigation belonging to different land before queueing research", async () => {
  const post = vi.spyOn(api, "POST").mockRejectedValue(new Error("Lost response"));
  useLandContext.getState().setResearchQuestion("Inspect this feature");
  const view = mount();
  await waitFor(() => expect(screen.getByRole("button", { name: "Investigate" })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: "Investigate" }));
  await screen.findByRole("button", { name: "Retry captured request" });
  view.unmount();
  useLandContext.getState().clear();
  mount();
  vi.mocked(api.GET).mockResolvedValue({
    data: { investigation: { ...inv, landId: "other-land" } },
    response: new Response(),
  });
  fireEvent.click(screen.getByRole("button", { name: "Retry captured request" }));
  await screen.findByText(
    "This captured investigation does not match the selected land and boundary revision.",
  );
  expect(post).toHaveBeenCalledTimes(1);
});

it("keeps the live question usable and reports failed browser storage", async () => {
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new Error("Storage full");
  });
  const post = vi.spyOn(api, "POST").mockResolvedValue({ data: {}, response: new Response() });
  useLandContext.getState().setResearchQuestion("Research this land");
  mount();
  await screen.findByText(/This browser could not save or restore/);
  await waitFor(() => expect(screen.getByRole("button", { name: "Investigate" })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: "Investigate" }));
  await waitFor(() => expect(post).toHaveBeenCalledOnce());
});

it("validates captured limits, source references, land identity and draft size", () => {
  const draft: ResearchDraft = {
    version: 1,
    landId: land.id,
    boundaryRevision: 1,
    question: "A question",
    focus,
    budget: { maxSteps: 16, maxSeconds: 180, maxOutputTokens: 12000, maxWebSearches: 6 },
    newTopic: false,
    investigationId: inv.id,
    pending: null,
  };
  expect(parseResearchDraft(JSON.stringify(draft), land.id)).toEqual(draft);
  for (const broken of [
    { ...draft, landId: "other-land" },
    { ...draft, version: 2 },
    { ...draft, focus: { ...focus, featureIndex: -1 } },
    { ...draft, budget: { ...draft.budget, maxSeconds: 999999 } },
    { ...draft, pending: { investigationId: inv.id } },
  ])
    expect(() => parseResearchDraft(JSON.stringify(broken), land.id)).toThrow();
  expect(() => parseResearchDraft(" ".repeat(100001), land.id)).toThrow();
  expect(researchDraftKey("alice/workspace", land.id)).not.toBe(
    researchDraftKey("bob/workspace", land.id),
  );
});

it("preserves an unreadable recovery record until explicitly discarded", async () => {
  const key = researchDraftKey("pilot", land.id);
  const raw = JSON.stringify({ version: 42, question: "Keep this future-format draft" });
  localStorage.setItem(key, raw);
  mount();
  await screen.findByRole("region", { name: "Unreadable saved question" });
  expect(localStorage.getItem(key)).toBe(raw);
  fireEvent.change(screen.getByLabelText("Follow your curiosity"), {
    target: { value: "A new question" },
  });
  expect(localStorage.getItem(key)).toBe(raw);
  fireEvent.click(screen.getByRole("button", { name: "Discard unreadable record" }));
  expect(parseResearchDraft(localStorage.getItem(key)!, land.id).question).toBe("A new question");
});
