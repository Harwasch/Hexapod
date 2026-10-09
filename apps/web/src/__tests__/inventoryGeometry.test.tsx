import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { LandMapGeometry } from "@twin/contracts";
import { api } from "@/api/client";
import { InventoryGeometryEditor } from "@/features/land/InventoryGeometryEditor";
import {
  closeInventoryShape,
  keepGeometryHistory,
  openInventoryShape,
} from "@/features/land/inventoryGeometry";
import { useLandContext } from "@/state/landContext";
import { useUi } from "@/state/ui";
const scene = vi.hoisted(() => ({ areas: { pickGround: vi.fn(), cancelPick: vi.fn() } }));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));
const geometry: LandMapGeometry = {
  type: "MultiPolygon",
  coordinates: [
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4],
        [0, 0],
      ],
      [
        [1, 1],
        [1, 2],
        [2, 2],
        [2, 1],
        [1, 1],
      ],
    ],
    [
      [
        [6, 0],
        [8, 0],
        [8, 2],
        [6, 0],
      ],
    ],
  ],
};
beforeEach(() => {
  useLandContext.setState({ section: "inventory" });
  useUi.setState({ activePanel: "land" });
});
afterEach(() => {
  cleanup();
  useLandContext.getState().clear();
  vi.restoreAllMocks();
  vi.clearAllMocks();
});
it("preserves every part and hole without exposing a duplicate closing vertex", () => {
  const shape = openInventoryShape(geometry);
  expect(shape.parts[0]?.[0]).toHaveLength(4);
  expect(closeInventoryShape(shape)).toEqual(geometry);
  expect(() =>
    closeInventoryShape({
      type: "Polygon",
      parts: [
        [
          [
            [1, 1],
            [2, 2],
            [NaN, 3],
          ],
        ],
      ],
    }),
  ).toThrow("every vertex");
  expect(() =>
    closeInventoryShape({
      type: "Polygon",
      parts: [
        [
          [
            [1, 1],
            [1, 1],
            [2, 2],
          ],
        ],
      ],
    }),
  ).toThrow("three distinct");
  const big = {
    type: "LineString" as const,
    parts: [[Array.from({ length: 20000 }, () => [1, 1] as [number, number])]],
  };
  expect(keepGeometryHistory(Array.from({ length: 30 }, () => big))).toHaveLength(5);
});
it("keeps holes through edits, invalidates preview, and restores removed parts with undo", async () => {
  const post = vi.spyOn(api, "POST").mockResolvedValue({
    data: {
      geometry,
      intersectsLand: true,
      distanceM: 0,
      areaM2: 100,
      lengthM: null,
      perimeterM: 50,
    },
    response: new Response(),
  });
  const apply = vi.fn();
  render(
    <InventoryGeometryEditor
      landId="land"
      boundaryRevision={1}
      geometry={geometry}
      onApply={apply}
      onCancel={vi.fn()}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Remove area part" }));
  expect(screen.getByLabelText("Area part").querySelectorAll("option")).toHaveLength(1);
  fireEvent.click(screen.getByRole("button", { name: "Undo geometry" }));
  expect(screen.getByLabelText("Area part").querySelectorAll("option")).toHaveLength(2);
  fireEvent.change(screen.getByLabelText("Boundary ring"), { target: { value: "1" } });
  fireEvent.change(screen.getByLabelText("Vertex 1 longitude"), { target: { value: "1.1" } });
  fireEvent.click(screen.getByRole("button", { name: "Preview geometry" }));
  await screen.findByText("Geometry is valid.");
  const calls = post.mock.calls as unknown as [string, { body: { geometry: typeof geometry } }][];
  const body = calls[0]![1].body;
  expect(body.geometry.coordinates).toEqual([
    [
      geometry.coordinates[0]![0],
      [
        [1.1, 1],
        [1, 2],
        [2, 2],
        [2, 1],
        [1.1, 1],
      ],
    ],
    geometry.coordinates[1],
  ]);
  fireEvent.change(screen.getByLabelText("Vertex 1 latitude"), { target: { value: "1.2" } });
  expect(screen.queryByText("Geometry is valid.")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Apply geometry to draft" })).toBeDisabled();
  expect(apply).not.toHaveBeenCalled();
});
it("ignores stale preview results and invalidates review after a boundary revision", async () => {
  let resolve!: (value: unknown) => void;
  vi.spyOn(api, "POST").mockImplementation(
    () =>
      new Promise((r) => {
        resolve = r;
      }) as never,
  );
  const props = {
    landId: "land",
    boundaryRevision: 1,
    geometry,
    onApply: vi.fn(),
    onCancel: vi.fn(),
  };
  const view = render(<InventoryGeometryEditor {...props} />);
  fireEvent.click(screen.getByRole("button", { name: "Preview geometry" }));
  fireEvent.change(screen.getByLabelText("Vertex 1 longitude"), { target: { value: "0.1" } });
  await act(
    async () =>
      await Promise.resolve(
        resolve({
          data: {
            geometry,
            intersectsLand: true,
            distanceM: 0,
            areaM2: 100,
            lengthM: null,
            perimeterM: 50,
          },
          response: new Response(),
        }),
      ),
  );
  expect(screen.getByRole("button", { name: "Apply geometry to draft" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Preview geometry" }));
  await act(
    async () =>
      await Promise.resolve(
        resolve({
          data: {
            geometry,
            intersectsLand: true,
            distanceM: 0,
            areaM2: 100,
            lengthM: null,
            perimeterM: 50,
          },
          response: new Response(),
        }),
      ),
  );
  expect(screen.getByRole("button", { name: "Apply geometry to draft" })).toBeEnabled();
  view.rerender(<InventoryGeometryEditor {...props} boundaryRevision={2} />);
  expect(screen.getByRole("button", { name: "Apply geometry to draft" })).toBeDisabled();
});
it("cancels map picking when hidden and does not claim another picker's selection", async () => {
  let resolve!: (value: unknown) => void;
  scene.areas.pickGround.mockImplementation(
    () =>
      new Promise((r) => {
        resolve = r;
      }),
  );
  render(
    <InventoryGeometryEditor
      landId="land"
      boundaryRevision={1}
      geometry={{ type: "Point", coordinates: [1, 2] }}
      onApply={vi.fn()}
      onCancel={vi.fn()}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Pick vertex 1 on map" }));
  expect(useLandContext.getState().pointPicker).toBe("inventory-geometry:land");
  act(() => {
    useLandContext.getState().setSection("ecology");
  });
  await waitFor(() => expect(scene.areas.cancelPick).toHaveBeenCalledOnce());
  act(() => {
    useLandContext.getState().setPointPicker("other-tool");
  });
  await act(async () => {
    resolve({ longitude: 3, latitude: 4 });
    await Promise.resolve();
  });
  expect(screen.getByLabelText("Vertex 1 longitude")).toHaveValue(1);
  expect(useLandContext.getState().pointPicker).toBe("other-tool");
});
