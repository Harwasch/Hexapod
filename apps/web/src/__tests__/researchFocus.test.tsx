import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components, LandArea } from "@twin/contracts";
import { api } from "@/api/client";
import { LandResearch } from "@/features/land/LandResearch";
import { LandMapSelection } from "@/features/land/LandMapSelection";
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
  await waitFor(() => expect(ask).toBeEnabled());
  fireEvent.click(ask);
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
