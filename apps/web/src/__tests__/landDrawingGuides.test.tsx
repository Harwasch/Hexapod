import {
  act,
  cleanup,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type * as Osm from "@/missions/osm";
import { mappedFeatureDraft } from "@/features/land/mappedFeature";
import { LandParcelPicker } from "@/features/land/surface/LandParcelPicker";
import { useMappedFeaturePick } from "@/features/land/surface/landFlow";
import type { OsmCandidate } from "@/missions/osm";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { api } from "@/api/client";

const osm = vi.hoisted(() => ({ answer: null as OsmCandidate | null }));
vi.mock("@/missions/osm", async (importOriginal) => ({
  ...(await importOriginal<typeof Osm>()),
  fetchOsmContaining: () => Promise.resolve(osm.answer),
}));
vi.mock("@/cesium/SceneContext", () => ({
  useScene: () => ({
    camera: { flyToRectangle: vi.fn() },
    areas: { cancelPick: vi.fn(), edit: vi.fn() },
    // A canvas with no layout: the review's framing (landFraming.ts) measures nothing to fix.
    viewer: { canvas: document.createElement("canvas") },
  }),
}));

afterEach(() => {
  cleanup();
  useLand.getState().clear();
  useLandContext.getState().clear();
  vi.restoreAllMocks();
});

it("keeps only reviewed candidates as drawing guides without writing a boundary", () => {
  const post = vi.spyOn(api, "POST");
  useLand.getState().begin("candidates");
  const geometry = {
    type: "LineString" as const,
    coordinates: [
      [-77.052, 38.888],
      [-77.048, 38.888],
    ],
  };
  useLandContext.getState().setCandidates([
    {
      id: "line-a",
      label: "Public fixture line",
      geometry,
      distanceM: 0,
      properties: {},
      source: { method: "mapped-feature", meaning: "physical-feature", label: "Synthetic fixture" },
    },
    {
      id: "line-b",
      label: "Other fixture",
      geometry,
      distanceM: 1,
      properties: {},
      source: { method: "mapped-feature", meaning: "physical-feature", label: "Synthetic fixture" },
    },
  ]);
  render(<LandParcelPicker />);
  fireEvent.click(screen.getByRole("checkbox", { name: "Public fixture line" }));
  fireEvent.click(screen.getByRole("button", { name: "Trace over them" }));
  expect(useLand.getState().mode).toBe("corridor");
  expect(useLand.getState().points).toEqual([]);
  expect(useLand.getState().draft).toBeNull();
  expect(useLand.getState().traceSources[0]?.label).toBe("Synthetic fixture");
  expect(useLandContext.getState().layers["drawing-guides"]?.features).toEqual([
    { id: "line-a", label: "Public fixture line", geometry },
  ]);
  expect(post).not.toHaveBeenCalled();
  useLandContext.getState().removeLayer("drawing-guides");
  expect(useLand.getState().traceSources[0]?.label).toBe("Synthetic fixture");
});

/** The Land surface's "find the mapped area here" (pick mode), built by mappedFeatureDraft. */
it("proposes the mapped area found at a picked point with the same source as before", async () => {
  const footprint = {
    type: "Polygon" as const,
    coordinates: [
      [
        [-120.05, 34.69],
        [-120.03, 34.69],
        [-120.03, 34.71],
        [-120.05, 34.69],
      ],
    ],
  };
  osm.answer = { id: "osm-w1", name: "Creek pasture", footprint, acres: 412 };
  renderHook(() => useMappedFeaturePick(true));
  act(() => {
    useLand.getState().begin("pick");
    useLand.getState().addPoint([-120.04, 34.698]);
  });
  await waitFor(() => expect(useLand.getState().draft).not.toBeNull());
  // The literal the pick built before the extraction, field for field.
  expect(useLand.getState().draft).toEqual({
    name: "Creek pasture",
    description: "",
    boundary: footprint,
    source: {
      method: "mapped-feature",
      meaning: "physical-feature",
      label: "OpenStreetMap feature",
      attribution: "Outline © OpenStreetMap contributors (ODbL)",
      url: "https://www.openstreetmap.org/copyright",
    },
  });
  expect(mappedFeatureDraft(osm.answer)).toEqual(useLand.getState().draft);
});
