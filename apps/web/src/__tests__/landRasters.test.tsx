import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components } from "@twin/contracts";
import { api } from "@/api/client";
import { LandRasterView } from "@/features/land/LandRasterView";
import { RasterDetails } from "@/features/land/surface/lenses/overview/RasterDetails";
import { useLandContext } from "@/state/landContext";

type Raster = components["schemas"]["LandRasterRead"];

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
const raster: Raster = {
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
        minimum: 0,
        maximum: 45,
        percentiles: { "50": 45 },
        histogramEdges: [0, 45],
        histogramCounts: [6],
        palette: "magma",
      },
    ],
  },
};
const landCover: Raster = {
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
            code: 80,
            label: "Permanent water bodies",
            color: "#0064c8",
            cells: 2,
            fraction: 0.25,
            sampledAreaM2: 200,
          },
        ],
      },
    ],
  },
};

function mount(value = raster) {
  vi.spyOn(api, "GET").mockImplementation((() =>
    Promise.resolve({ data: value, response: new Response() })) as never);
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
});

it("puts a research result's raster on the map and switches its band", async () => {
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Show on map" }));
  expect(useLandContext.getState().rasters.raster?.band).toBe(1);
  fireEvent.change(screen.getByLabelText("Map measurement"), { target: { value: "2" } });
  expect(useLandContext.getState().rasters.raster?.band).toBe(2);
  fireEvent.click(screen.getByRole("button", { name: "Hide from map" }));
  expect(useLandContext.getState().rasters).toEqual({});
  expect(screen.getByLabelText("Map measurement")).toHaveValue("2");
});

it("reopens the band of the map that is already visible", async () => {
  useLandContext.getState().setRaster({
    id: "raster",
    band: 2,
    opacity: 0.4,
    bounds: raster.metadata.bounds,
    attribution: "Fixture",
  });
  mount();
  expect(await screen.findByLabelText("Map measurement")).toHaveValue("2");
  expect(screen.getByRole("button", { name: "Hide from map" })).toBeVisible();
});

it("does not offer a map for areas smaller than the grid", async () => {
  mount({
    ...raster,
    metadata: {
      ...raster.metadata,
      boundaryCells: 0,
      validCells: 0,
      coverageFraction: null,
      bands: [{ ...elevation, validCells: 0, mean: null, minimum: null, maximum: null }],
    },
  });
  expect(await screen.findByRole("button", { name: "Show on map" })).toBeDisabled();
  expect(screen.getByText(/No grid cells fall inside/)).toBeVisible();
});

it("shows full statistics in percent and plain units, with warnings and the method", () => {
  render(<RasterDetails raster={{ ...raster, stale: true }} units="metric" />);
  expect(screen.getByText(/an earlier one/)).toBeVisible();
  expect(screen.getByText(/80% of the land has data/)).toBeVisible();
  const slope = screen.getByRole("region", { name: "Slope statistics" });
  // 45 degrees is a 100% slope; zero elevation stays a number, not "no data".
  expect(slope).toHaveTextContent("Median100%");
  expect(screen.getByRole("region", { name: "Elevation statistics" })).toHaveTextContent(
    "Lowest0 m",
  );
  expect(screen.getAllByRole("img", { name: /how the land's cells spread/ })).toHaveLength(2);
  expect(screen.getByText("This surface model is not a ground survey.")).toBeVisible();
  expect(screen.getByText(/Synthetic fixture/)).toBeVisible();
});

it("lists only the land-cover classes present, by plain name, with their share", () => {
  render(<RasterDetails raster={landCover} units="metric" />);
  const classes = screen.getByRole("region", { name: "Land-cover classes" });
  expect(classes).toHaveTextContent("Tree cover75%");
  expect(classes).toHaveTextContent("Open water");
  expect(classes).toHaveTextContent("(Permanent water bodies)");
  expect(screen.getByText(/do not identify species/)).toBeVisible();
  expect(screen.queryByText("Average")).not.toBeInTheDocument();
});

it("downloads the GeoTIFF", async () => {
  const paths: string[] = [];
  vi.spyOn(api, "GET").mockImplementation(((path: string) => {
    paths.push(path);
    return Promise.resolve({ data: new Blob(["tif"]), response: new Response() });
  }) as never);
  Object.assign(URL, { createObjectURL: () => "blob:fixture", revokeObjectURL: () => undefined });
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
  render(<RasterDetails raster={raster} units="metric" />);
  fireEvent.click(screen.getByRole("button", { name: "Download GeoTIFF" }));
  await waitFor(() => expect(click).toHaveBeenCalled());
  expect(paths).toEqual(["/api/v1/land/rasters/{raster_id}/download"]);
});
