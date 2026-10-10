import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components } from "@twin/contracts";
import { api } from "@/api/client";
import { LandRasterView } from "@/features/land/LandRasterView";
import { useLandContext } from "@/state/landContext";
import { useSelection } from "@/state/selection";

vi.mock("@/cesium/SceneContext", () => ({
  useScene: () => ({ camera: { flyToRectangle: vi.fn() } }),
}));
const elevation: components["schemas"]["RasterBand"] = {
  index: 1,
  name: "Surface elevation",
  unit: "m",
  validCells: 8,
  minimum: 0,
  maximum: 100,
  mean: 20,
  standardDeviation: 10,
  percentiles: { "50": 20 },
  histogramEdges: [0, 50, 100],
  histogramCounts: [6, 2],
  palette: "viridis",
};
const raster: components["schemas"]["LandRasterRead"] = {
  id: "raster",
  landId: "land",
  runId: "run",
  boundaryRevision: 1,
  request: { dataset: "cop-dem-glo-30" },
  sha256: "a".repeat(64),
  byteSize: 100,
  stale: false,
  createdAt: "2026-10-09T12:00:00Z",
  metadata: {
    algorithm: "terrain-v1",
    method: "Synthetic fixture",
    bounds: [-77.055, 38.886, -77.045, 38.891],
    crs: "EPSG:4326",
    resolutionM: 30,
    width: 4,
    height: 4,
    boundaryCells: 10,
    validCells: 8,
    coverageFraction: 0.8,
    sampledAreaM2: 7200,
    downloadedBytes: 100,
    warnings: ["This surface model is not a ground survey."],
    sources: [],
    bands: [
      elevation,
      {
        ...elevation,
        index: 2,
        name: "Surface slope",
        unit: "degrees",
        validCells: 6,
        palette: "magma",
      },
    ],
  },
};
function mount(value = raster) {
  vi.spyOn(api, "GET").mockImplementation(((path: string) =>
    Promise.resolve({
      data: path.endsWith("/sample")
        ? {
            longitude: -77.05,
            latitude: 38.888,
            values: value.request.dataset === "esa-worldcover-2021" ? [10] : [0, null],
            units: ["m", "degrees"],
            resolutionM: 30,
            interpretation: "Nearest analysis-grid cell; not an on-site measurement.",
          }
        : value,
      response: new Response(),
    })) as typeof api.GET);
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  render(
    <QueryClientProvider client={cache}>
      <LandRasterView id="raster" />
    </QueryClientProvider>,
  );
}
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  useLandContext.getState().clear();
  useSelection.getState().setSelection(null);
});
it("switches map bands and opacity while preserving land scope and showing band-specific coverage", async () => {
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Show terrain map" }));
  expect(useLandContext.getState().rasters.raster?.band).toBe(1);
  expect(screen.getByText(/80% coverage/)).toBeVisible();
  fireEvent.change(screen.getByLabelText("Map measurement"), { target: { value: "2" } });
  expect(useLandContext.getState().rasters.raster?.band).toBe(2);
  expect(screen.getByText(/60% coverage/)).toBeVisible();
  fireEvent.change(screen.getByRole("slider"), { target: { value: "0.4" } });
  expect(useLandContext.getState().rasters.raster?.opacity).toBe(0.4);
  fireEvent.click(screen.getByRole("button", { name: "Hide terrain map" }));
  expect(useLandContext.getState().rasters).toEqual({});
});
it("labels stale boundaries and keeps zero elevation distinct from missing slope", async () => {
  useSelection.getState().setSelection({
    kind: "ground",
    title: "Public park test point",
    longitude: -77.05,
    latitude: 38.888,
    height: null,
    terrainHeight: null,
    at: 1,
  });
  mount({ ...raster, stale: true });
  expect(await screen.findByText(/uses an older boundary/)).toBeVisible();
  fireEvent.click(screen.getByText("Sample a point"));
  fireEvent.click(screen.getByRole("button", { name: "Sample selected point" }));
  await waitFor(() => expect(screen.getByText("Surface elevation: 0 m")).toBeVisible());
  expect(screen.getByText("Surface slope: No data")).toBeVisible();
});
it("does not offer a map for unresolved small areas", async () => {
  mount({
    ...raster,
    metadata: {
      ...raster.metadata,
      boundaryCells: 0,
      validCells: 0,
      coverageFraction: null,
      bands: [
        {
          ...elevation,
          validCells: 0,
          mean: null,
          minimum: null,
          maximum: null,
          percentiles: {},
          histogramCounts: [],
          histogramEdges: [],
        },
      ],
    },
  });
  expect(await screen.findByRole("button", { name: "Show terrain map" })).toBeDisabled();
  expect(screen.getByText(/No grid cells inside/)).toBeVisible();
});

it("reopens the measurement and opacity of the map that is already visible", async () => {
  useLandContext.getState().setRaster({
    id: "raster",
    band: 2,
    opacity: 0.4,
    bounds: raster.metadata.bounds,
    attribution: "Fixture",
  });
  mount();
  expect(await screen.findByRole("combobox")).toHaveValue("2");
  expect(screen.getByRole("slider")).toHaveValue("0.4");
  fireEvent.click(screen.getByRole("button", { name: "Hide terrain map" }));
  expect(screen.getByRole("combobox")).toHaveValue("2");
});

it("shows categorical cover proportions without arithmetic summaries and decodes point classes", async () => {
  useSelection
    .getState()
    .setSelection({
      kind: "ground",
      title: "Public park test point",
      longitude: -77.05,
      latitude: 38.888,
      height: null,
      terrainHeight: null,
      at: 1,
    });
  mount({
    ...raster,
    request: { dataset: "esa-worldcover-2021", resolutionM: 10 },
    metadata: {
      ...raster.metadata,
      resolutionM: 10,
      bands: [
        {
          ...elevation,
          name: "Land cover (2021)",
          palette: "categorical",
          unit: "class code",
          minimum: null,
          maximum: null,
          mean: null,
          standardDeviation: null,
          percentiles: {},
          histogramCounts: [],
          histogramEdges: [],
          classes: [
            {
              code: 10,
              label: "Tree cover",
              color: "#006400",
              cells: 6,
              fraction: 0.75,
              sampledAreaM2: 600,
            },
            {
              code: 50,
              label: "Built-up",
              color: "#fa0000",
              cells: 2,
              fraction: 0.25,
              sampledAreaM2: 200,
            },
          ],
        },
      ],
    },
  });
  expect(await screen.findByRole("region", { name: "Land-cover analysis" })).toBeVisible();
  expect(screen.getByText("75%")).toBeVisible();
  expect(screen.getByText(/do not identify species/)).toBeVisible();
  expect(screen.queryByText("Mean")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Show land-cover map" }));
  expect(useLandContext.getState().rasters.raster?.categorical).toBe(true);
  fireEvent.click(screen.getByText("Sample a point"));
  fireEvent.click(screen.getByRole("button", { name: "Sample selected point" }));
  expect(await screen.findByText("Land cover (2021): Tree cover")).toBeVisible();
});
