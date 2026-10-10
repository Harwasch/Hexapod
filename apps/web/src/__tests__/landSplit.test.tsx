import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { Footprint, LandCreate } from "@twin/contracts";
import { api } from "@/api/client";
import { LandBoundarySplit } from "@/features/land/LandBoundarySplit";
import { nearestBoundaryPoint } from "@/features/land/boundarySnap";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { clearHistory, undo } from "@/state/history";
vi.mock("@/cesium/SceneContext", () => ({
  useScene: () => ({ camera: { flyToRectangle: vi.fn() } }),
}));
const shape = (west: number, east: number): Footprint => ({
  type: "Polygon",
  coordinates: [
    [
      [west, 0],
      [east, 0],
      [east, 1],
      [west, 1],
      [west, 0],
    ],
  ],
});
const draft: LandCreate = {
  name: "Synthetic area",
  boundary: shape(0, 2),
  source: { method: "drawn", label: "Fixture" },
};
const parts = [0, 1].map((i) => ({ boundary: shape(i, i + 1), areaM2: 100, perimeterM: 40 }));
beforeEach(() => {
  useLand.getState().clear();
  useLandContext.getState().clear();
  clearHistory();
  useLand.getState().propose(draft);
  useLand.getState().begin("split");
  useLand.getState().addPoint([1, -1]);
  useLand.getState().addPoint([1, 2]);
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
it("previews parts without changing land, applies only selected pieces and supports undo", async () => {
  const post = vi
    .spyOn(api, "POST")
    .mockResolvedValueOnce({ data: { parts, selection: null }, response: new Response() })
    .mockResolvedValueOnce({ data: { parts, selection: parts[0] }, response: new Response() });
  render(<LandBoundarySplit />);
  fireEvent.click(screen.getByRole("button", { name: "Preview split" }));
  const choice = await screen.findByRole("checkbox", { name: /Part 1/ });
  expect(useLand.getState().draft).toBe(draft);
  expect(screen.getByRole("button", { name: "Keep selected pieces" })).toBeDisabled();
  fireEvent.click(choice);
  expect(useLandContext.getState().layers["boundary-split"]?.selectedIds).toEqual(["0"]);
  fireEvent.click(screen.getByRole("button", { name: "Keep selected pieces" }));
  await waitFor(() => expect(useLand.getState().mode).toBe("browse"));
  expect(post).toHaveBeenLastCalledWith("/api/v1/land/split", {
    body: {
      boundary: draft.boundary,
      coordinates: [
        [1, -1],
        [1, 2],
      ],
      keepParts: [0],
    },
  });
  expect(useLand.getState().draft?.boundary).toEqual(parts[0]!.boundary);
  act(() => {
    undo();
  });
  expect(useLand.getState().draft?.boundary).toEqual(draft.boundary);
});
it("ignores a delayed preview after the cut line changes", async () => {
  let release!: (value: Awaited<ReturnType<typeof api.POST>>) => void;
  vi.spyOn(api, "POST").mockImplementation(
    () =>
      new Promise((resolve) => {
        release = resolve;
      }),
  );
  render(<LandBoundarySplit />);
  fireEvent.click(screen.getByRole("button", { name: "Preview split" }));
  act(() => useLand.getState().addPoint([2, 2]));
  await act(async () => {
    release({ data: { parts, selection: null }, response: new Response() });
    await Promise.resolve();
  });
  expect(screen.queryByRole("checkbox")).toBeNull();
  expect(useLand.getState().draft).toBe(draft);
});
it("snaps to corners, edges and holes while excluding the moving vertex and adjacent edges", () => {
  expect(nearestBoundaryPoint(shape(0, 2), [1, 0.02])).toEqual([1, 0]);
  expect(nearestBoundaryPoint(shape(0, 2), [-0.01, -0.01])).toEqual([0, 0]);
  const away = nearestBoundaryPoint(shape(0, 2), [0.01, 0.01], { polygon: 0, ring: 0, vertex: 0 });
  expect(away?.[1]).toBe(1);
  const withHole: Footprint = {
    type: "Polygon",
    coordinates: [
      ...(shape(0, 2).coordinates as number[][][]),
      [
        [0.4, 0.4],
        [0.6, 0.4],
        [0.6, 0.6],
        [0.4, 0.6],
        [0.4, 0.4],
      ],
    ],
  };
  expect(nearestBoundaryPoint(withHole, [0.5, 0.39])).toEqual([0.5, 0.4]);
});
