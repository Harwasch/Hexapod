import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { LandCreate } from "@twin/contracts";
import { ApiError, api } from "@/api/client";
import { LandDraftRecovery } from "@/features/land/LandDraftRecovery";
import { useLand } from "@/state/land";

const key = "living-world-land-draft:pilot";
vi.mock("@/cesium/SceneContext", () => ({
  useScene: () => ({
    camera: { flyToRectangle: vi.fn() },
    areas: { cancelPick: vi.fn(), edit: vi.fn() },
  }),
}));
const draft: LandCreate = {
  name: "Unfinished ranch",
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
  source: { method: "drawn", label: "Drawn" },
};
beforeEach(() => {
  localStorage.clear();
  useLand.getState().clear();
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it("recovers an unsaved boundary after reload and removes it when discarded", async () => {
  const first = render(<LandDraftRecovery scope="pilot" />);
  act(() => useLand.getState().propose(draft));
  expect(localStorage.getItem(key)).toContain("Unfinished ranch");
  first.unmount();
  useLand.getState().clear();
  render(<LandDraftRecovery scope="pilot" />);
  fireEvent.click(screen.getByRole("button", { name: "Resume boundary draft" }));
  await waitFor(() => expect(useLand.getState().draft?.name).toBe(draft.name));
  act(() => useLand.getState().cancel());
  expect(localStorage.getItem(key)).toBeNull();
});

it("keeps a newer saved revision intact until a separate draft is explicitly requested", async () => {
  localStorage.setItem(
    key,
    JSON.stringify({ version: 1, draft, activeId: "land", revision: 1, savedAt: "2026-10-09" }),
  );
  vi.spyOn(api, "GET").mockResolvedValue({
    data: { ...draft, id: "land", revision: 2 },
    response: new Response(),
  });
  render(<LandDraftRecovery scope="pilot" />);
  fireEvent.click(screen.getByRole("button", { name: "Resume boundary draft" }));
  await screen.findByRole("button", { name: "Restore as a separate area" });
  expect(useLand.getState().draft).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Restore as a separate area" }));
  await waitFor(() => expect(useLand.getState().draft?.name).toBe(draft.name));
  expect(useLand.getState().active).toBeNull();
});

it("does not display a different workspace's stored draft", () => {
  localStorage.setItem(
    "living-world-land-draft:someone%2Fprivate",
    JSON.stringify({ version: 1, draft }),
  );
  render(<LandDraftRecovery scope="pilot" />);
  expect(screen.queryByText(/Resume Unfinished ranch/)).toBeNull();
});

it("recovers an unfinished corridor with its width and units before it becomes a boundary", async () => {
  const first = render(<LandDraftRecovery scope="pilot" />);
  act(() => {
    useLand.getState().begin("corridor");
    useLand.getState().setCorridorWidth(150);
    useLand.getState().setCorridorUnit("m");
    useLand.getState().addPoint([-77.05, 38.888]);
    useLand.getState().addPoint([-77.049, 38.889]);
  });
  const width = useLand.getState().corridorWidth;
  expect(width).toBeCloseTo(45.72);
  expect(localStorage.getItem(key)).toContain('"mode":"corridor"');
  first.unmount();
  useLand.getState().clear();
  render(<LandDraftRecovery scope="pilot" />);
  expect(screen.getByText(/2 drawn points/)).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Resume drawing" }));
  await waitFor(() => expect(useLand.getState().mode).toBe("corridor"));
  expect(useLand.getState().points).toEqual([
    [-77.05, 38.888],
    [-77.049, 38.889],
  ]);
  expect(useLand.getState().corridorWidth).toBe(width);
  expect(useLand.getState().corridorUnit).toBe("m");
  expect(useLand.getState().draft).toBeNull();
  act(() => useLand.getState().cancel());
  expect(localStorage.getItem(key)).toBeNull();
});

it("does not overwrite a recoverable drawing when merely browsing another saved area", () => {
  localStorage.setItem(
    key,
    JSON.stringify({
      version: 2,
      draft: null,
      activeId: null,
      sketch: { mode: "draw", points: [[0, 0]], width: 100, unit: "ft" },
    }),
  );
  render(<LandDraftRecovery scope="pilot" />);
  act(() => useLand.getState().clear());
  expect(screen.getByRole("button", { name: "Resume drawing" })).toBeVisible();
  expect(localStorage.getItem(key)).toContain('"points":[[0,0]]');
  act(() => {
    useLand.getState().begin("draw");
    useLand.getState().addPoint([1, 1]);
  });
  expect(screen.queryByRole("button", { name: "Resume drawing" })).toBeNull();
  expect(localStorage.getItem(key)).toContain('"points":[[1,1]]');
});

it("preserves a sketch as a separate area when the saved boundary revision has changed", async () => {
  localStorage.setItem(
    key,
    JSON.stringify({
      version: 2,
      draft: null,
      activeId: "land",
      revision: 1,
      sketch: {
        mode: "draw",
        points: [
          [0, 0],
          [1, 0],
        ],
        width: 100,
        unit: "ft",
      },
    }),
  );
  vi.spyOn(api, "GET").mockResolvedValue({
    data: { ...draft, id: "land", revision: 2 },
    response: new Response(),
  });
  render(<LandDraftRecovery scope="pilot" />);
  fireEvent.click(screen.getByRole("button", { name: "Resume drawing" }));
  fireEvent.click(await screen.findByRole("button", { name: "Restore as a separate area" }));
  await waitFor(() => expect(useLand.getState().mode).toBe("draw"));
  expect(useLand.getState().active).toBeNull();
  expect(useLand.getState().points).toHaveLength(2);
});

it("rejects corrupt sketch coordinates without breaking the workspace", () => {
  localStorage.setItem(
    key,
    JSON.stringify({
      version: 2,
      draft: null,
      sketch: { mode: "draw", points: [[200, 0]], width: 100, unit: "ft" },
    }),
  );
  render(<LandDraftRecovery scope="pilot" />);
  expect(screen.queryByRole("button", { name: "Resume drawing" })).toBeNull();
  expect(useLand.getState().points).toEqual([]);
});

it("recovers a draft as a separate area when its original land has been deleted", async () => {
  localStorage.setItem(
    key,
    JSON.stringify({ version: 1, draft, activeId: "deleted", revision: 1 }),
  );
  vi.spyOn(api, "GET").mockRejectedValue(new ApiError(404, undefined, "Land not found"));
  render(<LandDraftRecovery scope="pilot" />);
  fireEvent.click(screen.getByRole("button", { name: "Resume boundary draft" }));
  fireEvent.click(await screen.findByRole("button", { name: "Restore as a separate area" }));
  await waitFor(() => expect(useLand.getState().draft?.name).toBe(draft.name));
  expect(useLand.getState().active).toBeNull();
});

it.each(["split", "difference"] as const)(
  "recovers the %s operation with its original boundary",
  async (operation) => {
    const first = render(<LandDraftRecovery scope="pilot" />);
    act(() => {
      useLand.getState().propose(draft);
      if (operation === "difference") useLand.getState().setBoundaryOperation(operation);
      useLand.getState().begin(operation === "split" ? "split" : "draw");
      useLand.getState().addPoint([0.5, -0.1]);
      useLand.getState().addPoint([0.5, 1.1]);
    });
    first.unmount();
    useLand.getState().clear();
    render(<LandDraftRecovery scope="pilot" />);
    fireEvent.click(screen.getByRole("button", { name: "Resume drawing" }));
    await waitFor(() => expect(useLand.getState().points).toHaveLength(2));
    expect(useLand.getState().draft).toEqual(draft);
    expect(useLand.getState().mode).toBe(operation === "split" ? "split" : "draw");
    expect(useLand.getState().boundaryOperation).toBe(operation === "split" ? null : "difference");
  },
);
