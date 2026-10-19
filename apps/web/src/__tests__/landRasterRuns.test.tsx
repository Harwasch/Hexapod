/**
 * The land-scale datasets behind Show once and Rehearsal: the seasonal Sentinel-2 signature
 * and the USGS 3DEP bare-earth raster. The Overview lens offers them on demand in Details
 * (rasterRuns.ts holds the requests and budgets), reads its terrain from bare earth where the
 * land has it, and LandRasterView / RasterDetails label what comes back.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { components, LandArea } from "@twin/contracts";
import type * as Identity from "@/state/landIdentity";

import { api } from "@/api/client";
import { LandRasterView } from "@/features/land/LandRasterView";
import { coversUs, RASTER_RUNS, rasterRequest, signaturePeriods } from "@/features/land/rasterRuns";
import { sourceName } from "@/features/land/surface/agent/agentEvents";
import { isAgentInvestigation } from "@/features/land/surface/agent/agentStore";
import { OverviewLens } from "@/features/land/surface/lenses/OverviewLens";
import {
  OUTSIDE_US,
  rasterKind,
  terrainOf,
} from "@/features/land/surface/lenses/overview/overviewData";
import { useOverviewMap } from "@/features/land/surface/lenses/overview/useOverview";
import { useLandContext } from "@/state/landContext";
import { useLandMap } from "@/state/landMap";
import { useSettings } from "@/state/settings";

type Raster = components["schemas"]["LandRasterRead"];
type RasterBand = components["schemas"]["RasterBand"];
type Investigation = components["schemas"]["InvestigationRead"];
type Detail = components["schemas"]["InvestigationDetail"];
type Run = components["schemas"]["RunRead"];

const identity = vi.hoisted(() => ({ canEdit: true }));
vi.mock("@/state/landIdentity", async (original) => ({
  ...(await original<typeof Identity>()),
  useLandCanEdit: () => identity.canEdit,
}));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => null }));

beforeEach(() => {
  identity.canEdit = true;
  useSettings.setState({ units: "metric" });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  useLandContext.getState().clear();
  useLandMap.setState({ pins: {}, focus: null });
  useOverviewMap.setState({ views: {}, opacity: 0.8 });
  localStorage.clear();
});

describe("the requests", () => {
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

  it("names both datasets for the agent's steps and keeps their runs out of its questions", () => {
    expect(sourceName("usgs-3dep-13")).toBe("USGS 3DEP bare-earth elevation");
    expect(sourceName("sentinel-2-signature")).toBe("Sentinel-2 seasonal signatures");
    for (const title of [RASTER_RUNS["usgs-3dep-13"].title, "Seasonal 10 m signatures"])
      expect(
        isAgentInvestigation({
          id: "i",
          landId: "l",
          title,
          question: title,
          boundaryRevision: 1,
          createdAt: "2026-10-17",
          stale: false,
        }),
      ).toBe(false);
  });
});

// ---- Fixtures ---------------------------------------------------------------

const LAND = "11111111-1111-4111-8111-111111111111";
const ranch: LandArea = {
  id: LAND,
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
const paris: LandArea = {
  ...ranch,
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
};

const baseBand: RasterBand = {
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

function terrainBands(name: "Surface" | "Bare-earth", low: number): RasterBand[] {
  return [
    {
      ...baseBand,
      index: 1,
      name: `${name} elevation`,
      unit: "m",
      minimum: low,
      maximum: low + 40,
      mean: low + 20,
      percentiles: { "5": low, "50": low + 20, "95": low + 40 },
      histogramEdges: [low, low + 20, low + 40],
    },
    {
      ...baseBand,
      index: 2,
      name: `${name} slope`,
      unit: "degrees",
      minimum: 0.2,
      maximum: 8,
      mean: 2,
      percentiles: { "50": 1.5, "95": 6 },
      histogramEdges: [0, 4, 8],
      palette: "magma",
    },
  ];
}

const keys = ["green", "red", "nir", "swir16", "ndvi"] as const;
const labels = ["Green (B03)", "Red (B04)", "NIR (B08)", "SWIR (B11)", "NDVI"];
const dates = ["2025-04-16", "2025-07-16"];

const signatureRaster: Raster = {
  id: "signature",
  landId: LAND,
  runId: "run-signature",
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

function terrainRaster(
  id: string,
  dataset: "cop-dem-glo-30" | "usgs-3dep-13",
  extra: Partial<Raster> = {},
): Raster {
  const bare = dataset === "usgs-3dep-13";
  return {
    ...signatureRaster,
    id,
    runId: `run-${id}`,
    request: { dataset, resolutionM: bare ? 10 : 30 },
    metadata: {
      ...signatureRaster.metadata,
      algorithm: `${dataset}-v1`,
      resolutionM: bare ? 10 : 30,
      warnings: [],
      vegetation: null,
      signature: null,
      // Bare earth sits under the canopy the surface model includes: lower here.
      bands: terrainBands(bare ? "Bare-earth" : "Surface", bare ? 300 : 310),
    },
    ...extra,
  };
}

const surface = terrainRaster("surface", "cop-dem-glo-30");
const bareEarth = terrainRaster("bare", "usgs-3dep-13");

function investigation(id: string, title: string): Investigation {
  return {
    id,
    landId: LAND,
    title,
    question: title,
    boundaryRevision: 1,
    createdAt: "2026-10-17T12:00:00Z",
    stale: false,
  };
}

function run(id: string, status: Run["status"]): Run {
  return {
    id,
    investigationId: "inv",
    kind: "raster",
    question: "",
    status,
    error: null,
    attempt: 1,
    budget: {},
    createdAt: "2026-10-17T12:00:00Z",
    startedAt: null,
    finishedAt: null,
  };
}

function detail(item: Investigation, runs: Run[], rasterId?: string): Detail {
  return {
    investigation: item,
    runs,
    findings: [],
    evidence: [],
    messages: [],
    page: { limit: 100, offset: 0, totals: {} },
    artifacts: rasterId
      ? [
          {
            id: `artifact-${rasterId}`,
            runId: runs[0]?.id ?? "run",
            title: item.title,
            method: "Synthetic",
            evidenceIds: [],
            output: { kind: "raster", rasterId },
          },
        ]
      : [],
  };
}

interface Init {
  params?: { path?: Record<string, string> };
  body?: Record<string, unknown>;
}

interface Fixture {
  investigations?: Investigation[];
  details?: Record<string, Detail>;
  rasters?: Raster[];
}

function mountOverview(land: LandArea, fixture: Fixture) {
  const posts: { path: string; init?: Init }[] = [];
  vi.spyOn(api, "GET").mockImplementation(((path: string, init?: Init) => {
    let data: unknown = [];
    if (path === "/api/v1/land/{land_id}/investigations") data = fixture.investigations ?? [];
    else if (path === "/api/v1/research/investigations/{investigation_id}")
      data = fixture.details?.[init?.params?.path?.investigation_id ?? ""];
    else if (path === "/api/v1/land/{land_id}/rasters") data = fixture.rasters ?? [];
    else if (path === "/api/v1/research/status") data = undefined;
    return Promise.resolve({ data, response: new Response() });
  }) as never);
  vi.spyOn(api, "POST").mockImplementation(((path: string, init?: Init) => {
    posts.push({ path, init });
    const data: unknown =
      path === "/api/v1/land/{land_id}/investigations"
        ? investigation(
            `inv-${posts.length}`,
            typeof init?.body?.title === "string" ? init.body.title : "",
          )
        : run(`run-${posts.length}`, "queued");
    return Promise.resolve({ data, response: new Response() });
  }) as never);
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <OverviewLens land={land} />
    </QueryClientProvider>,
  );
  return posts;
}

function runBodies(posts: { path: string; init?: Init }[]) {
  return posts.filter((call) => call.path.endsWith("/runs")).map((call) => call.init?.body);
}

function describedBy(element: HTMLElement): HTMLElement | null {
  return document.getElementById(element.getAttribute("aria-describedby") ?? "");
}

// ---- The Overview lens --------------------------------------------------------

describe("bare-earth terrain and seasonal signatures in the Overview", () => {
  it("prepares seasonal signatures and bare-earth terrain from Details as saved raster runs", async () => {
    const posts = mountOverview(ranch, {});
    fireEvent.click(await screen.findByText("Details and sources"));
    const signatures = screen.getByTestId("overview-signatures");
    expect(signatures).toHaveTextContent(/not the same species/);
    fireEvent.click(within(signatures).getByRole("button", { name: "Prepare 10 m signatures" }));
    await waitFor(() => expect(runBodies(posts)).toHaveLength(1));
    expect(posts[0]).toMatchObject({ init: { body: { title: "Seasonal 10 m signatures" } } });
    expect(runBodies(posts)[0]).toMatchObject({
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

    const terrain = within(screen.getByTestId("overview-bare-earth")).getByRole("button", {
      name: "Analyze bare-earth terrain",
    });
    await waitFor(() => expect(terrain).toBeEnabled());
    fireEvent.click(terrain);
    await waitFor(() => expect(runBodies(posts)).toHaveLength(2));
    expect(posts[2]).toMatchObject({
      init: { body: { title: "Bare-earth elevation and slope (USGS 3DEP)" } },
    });
    expect(runBodies(posts)[1]).toMatchObject({
      analysis: { dataset: "usgs-3dep-13", resolutionM: 10 },
      budget: { maxSeconds: 180 },
    });
    expect(runBodies(posts)[1]?.analysis).not.toHaveProperty("periods");
  });

  it("says why bare-earth terrain is unavailable outside the United States", async () => {
    mountOverview(paris, {});
    fireEvent.click(await screen.findByText("Details and sources"));
    const terrain = screen.getByRole("button", { name: "Analyze bare-earth terrain" });
    expect(terrain).toBeDisabled();
    expect(describedBy(terrain)).toHaveTextContent(OUTSIDE_US);
    expect(describedBy(terrain)?.textContent).toMatch(/outside the United States/);
    expect(screen.getByRole("button", { name: "Prepare 10 m signatures" })).toBeEnabled();
  });

  it("says why viewers cannot start them", async () => {
    identity.canEdit = false;
    mountOverview(ranch, {});
    fireEvent.click(await screen.findByText("Details and sources"));
    const signatures = screen.getByRole("button", { name: "Prepare 10 m signatures" });
    expect(signatures).toBeDisabled();
    expect(describedBy(signatures)).toHaveTextContent(
      "Your workspace role can view research but not start it.",
    );
  });

  it("reads the terrain from bare earth where the land has it, and names the source", async () => {
    const surfaceRun = investigation("inv-surface", RASTER_RUNS["cop-dem-glo-30"].title);
    const bareRun = investigation("inv-bare", RASTER_RUNS["usgs-3dep-13"].title);
    mountOverview(ranch, {
      investigations: [surfaceRun, bareRun],
      details: {
        "inv-surface": detail(surfaceRun, [run("r1", "succeeded")], "surface"),
        "inv-bare": detail(bareRun, [run("r2", "succeeded")], "bare"),
      },
      rasters: [bareEarth, surface],
    });
    const height = await screen.findByRole("button", { name: /^Height/ });
    expect(height).toHaveTextContent("300–340 m above sea level");
    expect(height).toHaveTextContent("Bare earth · USGS 3DEP, 10 m");
    expect(screen.getByRole("button", { name: /^Slope/ })).toHaveTextContent(
      "Bare earth · USGS 3DEP, 10 m",
    );
    // The bare-earth raster is the one on the map.
    await waitFor(() => expect(useLandContext.getState().rasters.bare?.band).toBe(1));
    expect(useLandContext.getState().rasters.surface).toBeUndefined();
    // Nothing more to offer for bare earth; Details keeps both models' numbers.
    fireEvent.click(screen.getByText("Details and sources"));
    expect(screen.queryByTestId("overview-bare-earth")).not.toBeInTheDocument();
    const terrain = screen.getByRole("region", { name: "Terrain details" });
    expect(terrain).toHaveTextContent("USGS 3DEP 1/3 arc-second bare-earth model · 10 m grid");
    expect(terrain).toHaveTextContent("Also saved: Copernicus GLO-30 surface model");
    expect(terrain).toHaveTextContent(/trees and buildings left out \(bare earth\)/);
  });

  it("keeps the worldwide surface model where there is no bare earth, and says so", async () => {
    const surfaceRun = investigation("inv-surface", RASTER_RUNS["cop-dem-glo-30"].title);
    mountOverview(ranch, {
      investigations: [surfaceRun],
      details: { "inv-surface": detail(surfaceRun, [run("r1", "succeeded")], "surface") },
      rasters: [surface],
    });
    expect(await screen.findByRole("button", { name: /^Height/ })).toHaveTextContent(
      "Surface incl. trees and buildings · Copernicus, 30 m",
    );
    fireEvent.click(screen.getByText("Details and sources"));
    expect(screen.getByTestId("overview-bare-earth")).toBeVisible();
  });

  it("prefers terrain made for this outline over bare earth made for an earlier one", () => {
    const stale = { ...bareEarth, stale: true, boundaryRevision: 0 };
    expect(terrainOf({ terrain: surface, bareEarth: stale })).toBe(surface);
    expect(terrainOf({ terrain: surface, bareEarth })).toBe(bareEarth);
    expect(terrainOf({ terrain: null, bareEarth: stale })).toBe(stale);
  });

  it("puts a season's NDVI from the signatures on the map from Details", async () => {
    const signatureRun = investigation("inv-signature", "Seasonal 10 m signatures");
    mountOverview(ranch, {
      investigations: [signatureRun],
      details: {
        "inv-signature": detail(signatureRun, [run("r1", "succeeded")], "signature"),
      },
      rasters: [signatureRaster],
    });
    fireEvent.click(await screen.findByText("Details and sources"));
    const section = screen.getByRole("region", { name: "Seasonal signature details" });
    expect(section).toHaveTextContent(
      "80 cells clear in every season carry a full 10-band signature",
    );
    // Prepared for this outline: nothing to start.
    expect(screen.queryByTestId("overview-signatures")).not.toBeInTheDocument();
    const select = screen.getByLabelText("Map measurement");
    expect(
      within(select)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["Nothing", "Seasonal greenness, April 2025", "Seasonal greenness, July 2025"]);
    fireEvent.change(select, { target: { value: "signature:10" } });
    await waitFor(() => expect(useLandContext.getState().rasters.signature?.band).toBe(10));
    expect(
      screen.getByRole("img", { name: "Map colours: Greenness, July 2025, from Bare to Green" }),
    ).toBeVisible();
    fireEvent.click(within(section).getByRole("button", { name: "2025-04 observation" }));
    await waitFor(() => expect(useLandContext.getState().rasters.signature?.band).toBe(5));
  });
});

// ---- LandRasterView (a research result naming a raster) ---------------------------

function mountRaster(value: Raster) {
  vi.spyOn(api, "GET").mockImplementation((() =>
    Promise.resolve({ data: value, response: new Response() })) as never);
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <LandRasterView id={value.id} />
    </QueryClientProvider>,
  );
}

describe("a saved raster in a research result", () => {
  it("labels a signature as one, maps its dated NDVI bands and keeps every band in the download", async () => {
    expect(rasterKind(signatureRaster)).toBe("signature");
    mountRaster(signatureRaster);
    const region = await screen.findByRole("region", { name: "Seasonal signature analysis" });
    expect(within(region).getByText(/Sentinel-2 seasonal signatures · 10 m grid/)).toBeVisible();
    expect(region).toHaveTextContent(
      "80 cells clear in every season carry a full 10-band signature",
    );
    const options = within(screen.getByLabelText("Map measurement")).getAllByRole("option");
    expect(options.map((option) => option.textContent)).toEqual([
      "2025-04-16 NDVI (NDVI)",
      "2025-07-16 NDVI (NDVI)",
    ]);
    // The latest clear season first, as a vegetation run's latest month.
    fireEvent.click(screen.getByRole("button", { name: "Show on map" }));
    expect(useLandContext.getState().rasters.signature?.band).toBe(10);
    fireEvent.click(screen.getByRole("button", { name: "2025-04 observation" }));
    expect(useLandContext.getState().rasters.signature?.band).toBe(5);
    expect(region).toHaveTextContent(/the download holds all 10 measurement bands/);
  });

  it("names the bare-earth source of a 3DEP raster", async () => {
    mountRaster(bareEarth);
    const region = await screen.findByRole("region", { name: "Terrain analysis" });
    expect(
      within(region).getByText(/USGS 3DEP 1\/3 arc-second bare-earth model · 10 m grid/),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "Show on map" })).toBeEnabled();
    expect(region).toHaveTextContent(/trees and buildings left out \(bare earth\)/);
    expect(region).not.toHaveTextContent(/including buildings and tree tops/);
  });
});
