import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components, LandArea } from "@twin/contracts";
import { api } from "@/api/client";
import { LandSavedViews } from "@/features/land/LandSavedViews";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { useViewer } from "@/state/viewer";
const { flyTo } = vi.hoisted(() => ({ flyTo: vi.fn() }));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => ({ camera: { flyTo } }) }));
const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Synthetic land",
  description: "",
  revision: 2,
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 0],
      ],
    ],
  },
  source: { method: "drawn", label: "Fixture" },
  areaM2: 1,
  perimeterM: 1,
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};
const view: components["schemas"]["LandViewRead"] = {
  id: "22222222-2222-4222-8222-222222222222",
  landId: land.id,
  name: "Study",
  revision: 1,
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
  state: {
    camera: { longitude: 2, latitude: 3, height: 400, heading: 50, pitch: -80, roll: 2 },
    boundaryRevision: 1,
    section: "records",
    inventoryVisible: false,
    artifactIds: [],
    rasters: [],
  },
};
const opened: components["schemas"]["LandViewOpen"] = {
  view,
  state: view.state,
  maps: [
    {
      id: "33333333-3333-4333-8333-333333333333",
      title: "Research map",
      features: [{ label: "Point", geometry: { type: "Point", coordinates: [2, 3] }, value: null }],
    },
  ],
  rasters: [
    {
      id: "44444444-4444-4444-8444-444444444444",
      kind: "raster",
      band: 2,
      opacity: 0.4,
      bounds: [1, 2, 3, 4],
      attribution: "Fixture",
      categorical: false,
    },
  ],
  warnings: ["Saved boundary is older."],
};
function show() {
  useLand.getState().select(land);
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <LandSavedViews land={land} />
    </QueryClientProvider>,
  );
  fireEvent.click(screen.getByText("Saved exploration views"));
}
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  flyTo.mockClear();
  useLand.getState().clear();
  useLandContext.getState().clear();
});
it("retries the captured view unchanged after a lost response", async () => {
  vi.spyOn(api, "GET").mockResolvedValue({ data: [], response: new Response() });
  const post = vi
    .spyOn(api, "POST")
    .mockRejectedValueOnce(new Error("Response lost"))
    .mockResolvedValue({ data: view, response: new Response() });
  useLandContext
    .getState()
    .setLayer({ id: "map", researchArtifactId: opened.maps[0]!.id, title: "Map", features: [] });
  show();
  fireEvent.change(screen.getByRole("textbox", { name: "View name" }), {
    target: { value: "Study" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save this view" }));
  await screen.findByRole("alert");
  useViewer.setState({ camera: { ...useViewer.getState().camera, longitude: 100 } });
  useLandContext.getState().removeLayer("map");
  fireEvent.click(screen.getByRole("button", { name: "Retry captured view save" }));
  await screen.findByRole("status");
  const calls = post.mock.calls as unknown as [string, { body: unknown }][];
  expect(calls).toHaveLength(2);
  expect(calls[0]?.[1].body).toEqual(calls[1]?.[1].body);
  expect(calls[0]?.[1].body).toMatchObject({ state: { artifactIds: [opened.maps[0]!.id] } });
});
it("restores map sources, band and camera while preserving unfinished map overlays", async () => {
  vi.spyOn(api, "GET").mockImplementation(
    (path: string) =>
      Promise.resolve({
        data: path.endsWith("/{view_id}") ? opened : [view],
        response: new Response(),
      }) as never,
  );
  const draft = {
    id: "inventory-draft",
    title: "Unfinished asset",
    features: [
      { id: "draft", label: "Pole", geometry: { type: "Point" as const, coordinates: [0, 0] } },
    ],
  };
  useLandContext.getState().setLayer(draft);
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Open Study" }));
  await screen.findByText("Saved boundary is older.");
  expect(flyTo).toHaveBeenCalledWith(2, 3, 400, { heading: 50, pitch: -80, roll: 2 });
  expect(useLandContext.getState().layers["inventory-draft"]).toEqual(draft);
  expect(useLandContext.getState().inventoryVisible).toBe(false);
  expect(useLandContext.getState().rasters[opened.rasters[0]!.id]).toMatchObject({
    band: 2,
    opacity: 0.4,
  });
  expect(useLandContext.getState().section).toBe("records");
});
it("does not move the camera when the user changes land during a slow open", async () => {
  let finish: ((value: unknown) => void) | undefined;
  vi.spyOn(api, "GET").mockImplementation((path: string) =>
    path.endsWith("/{view_id}")
      ? (new Promise((resolve) => {
          finish = resolve;
        }) as never)
      : (Promise.resolve({ data: [view], response: new Response() }) as never),
  );
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Open Study" }));
  await waitFor(() => expect(finish).toBeDefined());
  useLand.getState().select({ ...land, id: "other" });
  finish?.({ data: opened, response: new Response() });
  await waitFor(() => expect(screen.getByRole("button", { name: "Open Study" })).toBeEnabled());
  expect(flyTo).not.toHaveBeenCalled();
});
