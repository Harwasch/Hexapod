/**
 * The land-scale datasets behind Show once and Rehearsal: the seasonal Sentinel-2 signature
 * and the USGS 3DEP bare-earth raster. LandResearch starts them (rasterRuns.ts holds the
 * requests) and LandRasterView labels what comes back.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components, LandArea, ResearchStatus } from "@twin/contracts";

import { api } from "@/api/client";
import { LandRasterView } from "@/features/land/LandRasterView";
import { LandResearch } from "@/features/land/LandResearch";
import { coversUs, RASTER_RUNS, rasterRequest, signaturePeriods } from "@/features/land/rasterRuns";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";

vi.mock("@/cesium/SceneContext", () => ({
  useScene: () => ({ camera: { flyToRectangle: vi.fn() } }),
}));

afterEach(() => {
  cleanup();
  useLandContext.getState().clear();
  useLand.getState().clear();
  vi.restoreAllMocks();
  localStorage.clear();
});

it("asks for four 30-day seasons of the latest complete year at 10 m", () => {
  const seasons = signaturePeriods("2026-10-17");
  expect(seasons).toEqual([
    { startDate: "2025-01-01", endDate: "2025-01-30" },
    { startDate: "2025-04-01", endDate: "2025-04-30" },
    { startDate: "2025-07-01", endDate: "2025-07-30" },
    { startDate: "2025-10-01", endDate: "2025-10-30" },
  ]);
  expect(rasterRequest("sentinel-2-signature", seasons)).toEqual({
    dataset: "sentinel-2-signature",
    resolutionM: 10,
    maxDimension: 512,
    periods: seasons,
  });
  expect(rasterRequest("usgs-3dep-13")).toEqual({
    dataset: "usgs-3dep-13",
    resolutionM: 10,
    maxDimension: 512,
  });
  expect(RASTER_RUNS["usgs-3dep-13"].title).toBe("Bare-earth elevation and slope (USGS 3DEP)");
  // Each budget covers the processor's limit (wall_seconds in raster_runner.py) and the save.
  const processing: Record<string, number> = {
    "sentinel-2-signature": 200,
    "sentinel-2-ndvi": 165,
  };
  for (const [dataset, run] of Object.entries(RASTER_RUNS))
    expect(run.maxSeconds).toBeGreaterThanOrEqual((processing[dataset] ?? 90) + 15);
  expect(coversUs(-120.04, 34.698)).toBe(true);
  expect(coversUs(2.35, 48.85)).toBe(false);
});

const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Oak Bench",
  description: "",
  revision: 1,
  areaM2: 100,
  perimeterM: 40,
  createdAt: "2026-10-16",
  updatedAt: "2026-10-16",
  source: { method: "drawn", label: "Fixture" },
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-120.05, 34.69],
        [-120.03, 34.69],
        [-120.03, 34.71],
        [-120.05, 34.69],
      ],
    ],
  },
};
const inv = {
  id: "22222222-2222-4222-8222-222222222222",
  landId: land.id,
  title: "Land overview",
  question: "What is useful and interesting about this land?",
  boundaryRevision: 1,
  createdAt: "2026-10-16",
  stale: false,
};
const status: ResearchStatus = {
  modelConfigured: true,
  model: "fixture",
  workerRequired: true,
  worker: {
    state: "online",
    lastSeenAt: "2026-10-16T17:46:00Z",
    secondsSinceSeen: 3,
    busy: false,
    checkInEverySeconds: 15,
  },
  queue: { queued: 0, oldestQueuedAt: null, oldestQueuedSeconds: null, running: 0 },
};

function mountResearch(area: LandArea = land) {
  useLand.getState().select(area);
  vi.spyOn(globalThis, "fetch").mockReturnValue(new Promise<Response>(() => undefined));
  vi.spyOn(api, "GET").mockImplementation(((path: string) =>
    Promise.resolve({
      data: path.endsWith("/status")
        ? status
        : path.endsWith("/investigations")
          ? [inv]
          : path.endsWith("/events")
            ? []
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
    })) as never);
  const post = vi.spyOn(api, "POST").mockImplementation(((path: string) =>
    Promise.resolve({
      data: path.endsWith("/investigations") ? { ...inv, id: `inv-${post.mock.calls.length}` } : {},
      response: new Response(),
    })) as never);
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <LandResearch land={area} />
    </QueryClientProvider>,
  );
  return post;
}

function runBodies(post: ReturnType<typeof mountResearch>) {
  return (post.mock.calls as unknown as [string, { body: Record<string, unknown> }][])
    .filter(([path]) => path.endsWith("/runs"))
    .map(([, options]) => options.body);
}

it("prepares seasonal signatures and bare-earth terrain as saved raster runs", async () => {
  const post = mountResearch();
  fireEvent.click(await screen.findByRole("button", { name: "Prepare 10 m signatures" }));
  await waitFor(() => expect(runBodies(post)).toHaveLength(1));
  expect(runBodies(post)[0]).toMatchObject({
    kind: "raster",
    question: RASTER_RUNS["sentinel-2-signature"].question,
    analysis: {
      dataset: "sentinel-2-signature",
      resolutionM: 10,
      maxDimension: 512,
      periods: signaturePeriods(),
    },
    // Up to 200 s of processing and 15 s to save; a default 180 s budget would cut it off.
    budget: { maxSeconds: 300 },
  });
  expect(post.mock.calls[0]?.[1]).toMatchObject({
    body: { title: "Seasonal 10 m signatures" },
  });
  const terrain = await screen.findByRole("button", { name: "Analyze bare-earth terrain" });
  await waitFor(() => expect(terrain).toBeEnabled());
  fireEvent.click(terrain);
  await waitFor(() => expect(runBodies(post)).toHaveLength(2));
  expect(runBodies(post)[1]).toMatchObject({
    analysis: { dataset: "usgs-3dep-13", resolutionM: 10 },
    budget: { maxSeconds: 180 },
  });
  expect(runBodies(post)[1]?.analysis).not.toHaveProperty("periods");
});

it("says why bare-earth terrain is unavailable outside the United States", async () => {
  mountResearch({
    ...land,
    boundary: {
      type: "Polygon",
      coordinates: [
        [
          [2.35, 48.85],
          [2.36, 48.85],
          [2.36, 48.86],
          [2.35, 48.85],
        ],
      ],
    },
  });
  const terrain = await screen.findByRole("button", { name: "Analyze bare-earth terrain" });
  expect(terrain).toBeDisabled();
  expect(terrain).toHaveAttribute("aria-describedby", "land-research-outside-us");
  expect(document.getElementById("land-research-outside-us")?.textContent).toMatch(
    /outside the United States/,
  );
  expect(screen.getByRole("button", { name: "Prepare 10 m signatures" })).toBeEnabled();
});

const baseBand: components["schemas"]["RasterBand"] = {
  index: 1,
  name: "",
  unit: "reflectance",
  validCells: 80,
  minimum: 0.1,
  maximum: 0.3,
  mean: 0.2,
  standardDeviation: 0.05,
  percentiles: { "50": 0.2 },
  histogramEdges: [0.1, 0.2, 0.3],
  histogramCounts: [40, 40],
  palette: "viridis",
};
const keys = ["green", "red", "nir", "swir16", "ndvi"] as const;
const labels = ["Green (B03)", "Red (B04)", "NIR (B08)", "SWIR (B11)", "NDVI"];
const dates = ["2025-04-16", "2025-07-16"];
const signatureRaster: components["schemas"]["LandRasterRead"] = {
  id: "raster",
  landId: land.id,
  runId: "run",
  boundaryRevision: 1,
  request: {
    dataset: "sentinel-2-signature",
    resolutionM: 10,
    periods: [
      { startDate: "2025-04-01", endDate: "2025-04-30" },
      { startDate: "2025-07-01", endDate: "2025-07-30" },
    ],
  },
  sha256: "a".repeat(64),
  byteSize: 100,
  stale: false,
  createdAt: "2026-10-17T12:00:00Z",
  metadata: {
    algorithm: "sentinel-2-signature-v1",
    method: "Synthetic fixture",
    bounds: [-120.05, 34.69, -120.03, 34.71],
    crs: "EPSG:4326",
    resolutionM: 10,
    width: 10,
    height: 10,
    boundaryCells: 100,
    validCells: 90,
    coverageFraction: 0.9,
    sampledAreaM2: 9000,
    downloadedBytes: 100,
    warnings: ["Similar spectral signature at 10 m across 2 dates; not a species identification."],
    sources: [],
    bands: dates.flatMap((date, window) =>
      keys.map((key, i) => ({
        ...baseBand,
        index: window * 5 + i + 1,
        name: `${date} ${labels[i]}`,
        unit: key === "ndvi" ? "NDVI" : "reflectance",
      })),
    ),
    vegetation: {
      observations: dates.map((date, window) => ({
        period: { startDate: `${date.slice(0, 7)}-01`, endDate: `${date.slice(0, 7)}-30` },
        band: window * 5 + 5,
        sceneId: `S2B_T10SGD_${date.replaceAll("-", "")}T185325_L2A`,
        acquiredAt: `${date}T18:54:50Z`,
        candidateCount: 4,
        catalogTruncated: false,
        qualityCandidatesExamined: 2,
        validCells: 85,
        coverageFraction: 0.85,
        meanNdvi: 0.4,
        commonMeanNdvi: 0.41,
        qualityCounts: { Vegetation: 85 },
        explanation: "One selected acquisition.",
      })),
      commonCells: 80,
      commonCoverageFraction: 0.8,
      changeBand: null,
      changeCells: 0,
      meanChange: null,
    },
    signature: {
      features: dates.flatMap((date, window) =>
        keys.map((key, i) => ({
          band: window * 5 + i + 1,
          window,
          key,
          name: `${date} ${labels[i]}`,
          commonMean: 0.2,
          commonStandardDeviation: 0.05,
        })),
      ),
      commonCells: 80,
      commonCoverageFraction: 0.8,
      standardization: "z-score per feature",
      similarity: "cosine of standardized vectors",
    },
  },
};

function mountRaster(value: components["schemas"]["LandRasterRead"]) {
  vi.spyOn(api, "GET").mockImplementation((() =>
    Promise.resolve({ data: value, response: new Response() })) as never);
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <LandRasterView id="raster" />
    </QueryClientProvider>,
  );
}

it("maps a signature by its dated NDVI bands and keeps every band in the download", async () => {
  mountRaster(signatureRaster);
  expect(
    await screen.findByText(/Sentinel-2 Collection 1 seasonal signatures · 10 m analysis grid/),
  ).toBeVisible();
  expect(screen.getByText(/80 cells clear in every season carry a full 10-band/)).toBeVisible();
  const options = screen.getAllByRole("option");
  expect(options.map((option) => option.textContent)).toEqual([
    "2025-04-16 NDVI (NDVI)",
    "2025-07-16 NDVI (NDVI)",
  ]);
  fireEvent.click(screen.getByRole("button", { name: "Show vegetation map" }));
  expect(useLandContext.getState().rasters.raster?.band).toBe(5);
  fireEvent.click(screen.getByRole("button", { name: "2025-07 observation" }));
  expect(useLandContext.getState().rasters.raster?.band).toBe(10);
  expect(screen.getByText(/all 10 measurement bands/)).toBeInTheDocument();
});

it("names the bare-earth source of a 3DEP raster", async () => {
  mountRaster({
    ...signatureRaster,
    request: { dataset: "usgs-3dep-13", resolutionM: 10 },
    metadata: {
      ...signatureRaster.metadata,
      algorithm: "usgs-3dep-13-v1",
      vegetation: null,
      signature: null,
      bands: [
        { ...baseBand, index: 1, name: "Bare-earth elevation", unit: "m" },
        { ...baseBand, index: 2, name: "Bare-earth slope", unit: "degrees", palette: "magma" },
      ],
    },
  });
  expect(
    await screen.findByText(/USGS 3DEP 1\/3 arc-second bare-earth model · 10 m analysis grid/),
  ).toBeVisible();
  expect(screen.getByRole("region", { name: "Terrain analysis" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Show terrain map" })).toBeEnabled();
  expect(screen.getByText(/both measurement bands/)).toBeInTheDocument();
});
