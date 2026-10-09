import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ResearchMapExplorer } from "@/features/land/ResearchMapExplorer";
import { researchMapBounds, mapValue } from "@/features/land/researchMap";
import { useLandContext } from "@/state/landContext";
import type { ResearchArtifact } from "@twin/contracts";
const { flyToRectangle } = vi.hoisted(() => ({ flyToRectangle: vi.fn() }));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => ({ camera: { flyToRectangle } }) }));
const output: Extract<ResearchArtifact["output"], { kind: "map" }> = {
  kind: "map",
  unit: "m",
  legend: "Synthetic field samples",
  features: Array.from({ length: 125 }, (_, i) => ({
    label: `Sample ${i}`,
    geometry: { type: "Point", coordinates: [-77 + i * 0.00001, 38.9] },
    value: i === 1 ? null : i,
  })),
};
afterEach(() => {
  cleanup();
  useLandContext.getState().clear();
  vi.clearAllMocks();
});
it("links filtered list selection to the map without losing the filter or moving the camera", () => {
  render(<ResearchMapExplorer artifactId="map" title="Samples" output={output} />);
  fireEvent.click(screen.getByText("Explore mapped features"));
  fireEvent.change(screen.getByRole("searchbox", { name: "Filter Samples" }), {
    target: { value: "Sample 12" },
  });
  const list = screen.getByRole("list", { name: "Samples features" });
  expect(within(list).getAllByRole("listitem")).toHaveLength(6);
  fireEvent.click(screen.getByRole("button", { name: "Sample 120 Point · 120 m" }));
  expect(useLandContext.getState().layers.map?.selectedIds).toEqual(["120"]);
  expect(screen.getByRole("searchbox", { name: "Filter Samples" })).toHaveValue("Sample 12");
  expect(flyToRectangle).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Focus on Sample 120" }));
  expect(flyToRectangle).toHaveBeenCalledOnce();
  expect(useLandContext.getState().layers.map?.unit).toBe("m");
});
it("opens the page containing a picked map feature and clears stale selection on hide", async () => {
  render(<ResearchMapExplorer artifactId="map" title="Samples" output={output} />);
  fireEvent.click(screen.getByRole("button", { name: "Show map layer" }));
  act(() => useLandContext.getState().selectMapFeature("map", "120", "map"));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Sample 120 Point · 120 m" })).toHaveAttribute(
      "aria-pressed",
      "true",
    ),
  );
  expect(
    within(screen.getByRole("list", { name: "Samples features" })).getAllByRole("listitem"),
  ).toHaveLength(25);
  fireEvent.click(screen.getByRole("button", { name: "Hide map layer" }));
  expect(useLandContext.getState().selectedMapFeature).toBeNull();
  expect(useLandContext.getState().layers.map).toBeUndefined();
});
it("preserves zero and missing values and frames a very large line without argument spreading", () => {
  expect(mapValue(0, "m")).toBe("0 m");
  expect(mapValue(null, "m")).toBe("No value");
  const bounds = researchMapBounds([
    {
      geometry: {
        type: "LineString",
        coordinates: Array.from({ length: 150000 }, (_, i) => [-77 + i / 1_000_000, 38.9]),
      },
    },
  ]);
  expect(bounds?.west).toBeLessThan(-77);
  expect(bounds?.east).toBeGreaterThan(-76.851);
  expect(researchMapBounds([])).toBeNull();
});
it("retains selection when replacing a layer and clears it when the feature disappears", () => {
  const layer = {
    id: "map",
    researchArtifactId: "map",
    title: "Samples",
    features: output.features.map((feature, index) => ({ ...feature, id: String(index) })),
  };
  useLandContext.getState().setLayer(layer);
  useLandContext.getState().selectMapFeature("map", "0");
  useLandContext.getState().setLayer({ ...layer, title: "Updated" });
  expect(useLandContext.getState().layers.map?.selectedIds).toEqual(["0"]);
  useLandContext.getState().setLayer({ ...layer, features: layer.features.slice(1) });
  expect(useLandContext.getState().selectedMapFeature).toBeNull();
  expect(useLandContext.getState().layers.map?.selectedIds).toEqual([]);
});
