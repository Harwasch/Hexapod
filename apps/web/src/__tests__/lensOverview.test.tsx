import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { components, LandArea } from "@twin/contracts";
import type * as Identity from "@/state/landIdentity";

import { api } from "@/api/client";
import { OverviewLens } from "@/features/land/surface/lenses/OverviewLens";
import {
  coverPhrase,
  elevationPhrase,
  greenness,
  overviewHeadline,
  slopePercent,
  slopeSummary,
  slopeValue,
  slopeWord,
  type Raster,
  type RasterBand,
  type RasterClass,
  type Timeline,
} from "@/features/land/surface/lenses/overview/overviewData";
import { useOverviewMap } from "@/features/land/surface/lenses/overview/useOverview";
import { useLandContext } from "@/state/landContext";
import { useLandMap } from "@/state/landMap";
import { useSelection } from "@/state/selection";
import { useSettings } from "@/state/settings";

type Investigation = components["schemas"]["InvestigationRead"];
type Detail = components["schemas"]["InvestigationDetail"];
type Run = components["schemas"]["RunRead"];

const identity = vi.hoisted(() => ({ canEdit: true }));
vi.mock("@/state/landIdentity", async (original) => ({
  ...(await original<typeof Identity>()),
  useLandCanEdit: () => identity.canEdit,
}));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => null }));

const LAND = "11111111-1111-4111-8111-111111111111";
const land: LandArea = {
  id: LAND,
  name: "Public fixture",
  description: "",
  revision: 1,
  areaM2: 40000,
  perimeterM: 800,
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-77.055, 38.886],
        [-77.045, 38.886],
        [-77.045, 38.891],
        [-77.055, 38.891],
        [-77.055, 38.886],
      ],
    ],
  },
  source: { method: "drawn", label: "Synthetic" },
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};

function band(overrides: Partial<RasterBand>): RasterBand {
  return {
    index: 1,
    name: "Surface elevation",
    unit: "m",
    validCells: 8,
    minimum: 62.2,
    maximum: 78.4,
    mean: 70,
    standardDeviation: 4,
    percentiles: { "5": 63, "25": 66, "50": 70, "75": 74, "95": 77 },
    histogramEdges: [62, 70, 78],
    histogramCounts: [5, 3],
    palette: "viridis",
    ...overrides,
  };
}

function raster(id: string, dataset: Raster["request"]["dataset"], bands: RasterBand[]): Raster {
  return {
    id,
    landId: LAND,
    runId: `run-${id}`,
    boundaryRevision: 1,
    request: { dataset },
    sha256: "a".repeat(64),
    byteSize: 100,
    stale: false,
    createdAt: "2026-10-09T12:00:00Z",
    metadata: {
      algorithm: "fixture-v1",
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
      warnings: [],
      sources: [
        {
          id: "fixture-source",
          attribution: "Fixture attribution",
          catalogDatetime: null,
          catalogUrl: "https://example.org/catalog",
          etag: null,
          lastModified: null,
          license: "CC-BY-4.0",
          licenseUrl: "https://example.org/license",
          observationPeriod: "2021",
          url: "https://example.org/data",
        },
      ],
      bands,
    },
  };
}

const terrain = raster("terrain", "cop-dem-glo-30", [
  band({}),
  band({
    index: 2,
    name: "Surface slope",
    unit: "degrees",
    validCells: 6,
    minimum: 0.2,
    maximum: 8,
    mean: 2,
    percentiles: { "50": 1.5, "95": 6 },
    palette: "magma",
  }),
]);

function cls(code: number, label: string, color: string, cells: number): RasterClass {
  return { code, label, color, cells, fraction: cells / 100, sampledAreaM2: cells * 100 };
}

const cover = raster("cover", "esa-worldcover-2021", [
  band({
    name: "Land cover (2021)",
    unit: "class code",
    palette: "categorical",
    minimum: null,
    maximum: null,
    mean: null,
    standardDeviation: null,
    percentiles: {},
    histogramEdges: [],
    histogramCounts: [],
    classes: [
      cls(10, "Tree cover", "#006400", 26),
      cls(30, "Grassland", "#ffff4c", 64),
      cls(50, "Built-up", "#fa0000", 10),
      cls(80, "Permanent water bodies", "#0064c8", 0),
    ],
  }),
]);

function investigation(id: string, title: string, revision = 1): Investigation {
  return {
    id,
    landId: LAND,
    title,
    boundaryRevision: revision,
    createdAt: "2026-10-09T12:00:00Z",
    stale: false,
  };
}

function run(id: string, status: Run["status"], error: string | null = null): Run {
  return {
    id,
    investigationId: "inv",
    kind: "raster",
    question: "",
    status,
    error,
    attempt: 1,
    budget: {},
    createdAt: "2026-10-09T12:00:00Z",
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

interface Fixture {
  investigations?: Investigation[];
  details?: Record<string, Detail>;
  rasters?: Raster[];
}

interface Init {
  params?: { path?: Record<string, string>; query?: Record<string, unknown> };
  body?: Record<string, unknown>;
}

interface Call {
  path: string;
  init?: Init;
}

function mockApi(fixture: Fixture) {
  const gets: Call[] = [];
  const posts: Call[] = [];
  vi.spyOn(api, "GET").mockImplementation(((path: string, init?: Init) => {
    gets.push({ path, init });
    let data: unknown = [];
    if (path === "/api/v1/land/{land_id}/investigations") data = fixture.investigations ?? [];
    else if (path === "/api/v1/research/investigations/{investigation_id}")
      data = fixture.details?.[init?.params?.path?.investigation_id ?? ""];
    else if (path === "/api/v1/land/{land_id}/rasters") data = fixture.rasters ?? [];
    else if (path === "/api/v1/land/rasters/{raster_id}/sample")
      data =
        init?.params?.path?.raster_id === "terrain"
          ? {
              values: [71.4, 1.7],
              units: ["m", "degrees"],
              resolutionM: 30,
              interpretation: "Nearest cell.",
            }
          : {
              values: [30],
              units: ["class code"],
              resolutionM: 10,
              interpretation: "Nearest cell.",
            };
    return Promise.resolve({ data, response: new Response() });
  }) as never);
  vi.spyOn(api, "POST").mockImplementation(((path: string, init?: Init) => {
    posts.push({ path, init });
    const data: unknown =
      path === "/api/v1/land/{land_id}/investigations"
        ? investigation("new", "Started")
        : run("run-new", path.endsWith("/cancel") ? "cancelled" : "queued");
    return Promise.resolve({ data, response: new Response() });
  }) as never);
  return { gets, posts };
}

function mount(value = land) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={client}>
      <OverviewLens land={value} />
    </QueryClientProvider>,
  );
}

function runBodies(posts: Call[]) {
  return posts.filter((call) => call.path.endsWith("/runs")).map((call) => call.init?.body);
}

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
  useSelection.getState().setSelection(null);
});

describe("Overview lens", () => {
  it("starts nothing on its own and reads the terrain when asked", async () => {
    const { posts } = mockApi({});
    mount();
    const start = await screen.findByRole("button", { name: "Read the terrain" });
    expect(screen.getByText(/how high and how steep/)).toBeVisible();
    expect(posts).toHaveLength(0);
    fireEvent.click(start);
    await waitFor(() => expect(runBodies(posts)).toHaveLength(1));
    expect(posts[0]).toMatchObject({
      path: "/api/v1/land/{land_id}/investigations",
      init: { body: { title: "Surface elevation and slope" } },
    });
    expect(runBodies(posts)[0]).toMatchObject({
      kind: "raster",
      analysis: { dataset: "cop-dem-glo-30", resolutionM: 30, maxDimension: 512 },
    });
  });

  it("offers no start to someone who can only view", async () => {
    identity.canEdit = false;
    mockApi({});
    mount();
    expect(await screen.findByText(/Someone who can edit this land/)).toBeVisible();
    expect(screen.queryByRole("button", { name: "Read the terrain" })).not.toBeInTheDocument();
  });

  it("summarises terrain and cover in plain words and draws one layer at a time", async () => {
    const terrainRun = investigation("inv-terrain", "Surface elevation and slope");
    const coverRun = investigation("inv-cover", "Land cover in 2021");
    mockApi({
      investigations: [terrainRun, coverRun],
      details: {
        "inv-terrain": detail(terrainRun, [run("r1", "succeeded")], "terrain"),
        "inv-cover": detail(coverRun, [run("r2", "succeeded")], "cover"),
      },
      rasters: [cover, terrain],
    });
    mount();
    expect(
      await screen.findByRole("heading", {
        name: "Mostly gentle slopes, 62–78 m above sea level, about two-thirds grassland.",
      }),
    ).toBeVisible();
    expect(screen.getByText("Gentle, typically 3% · up to 11%")).toBeVisible();
    // The size is on the land chip already; the facts are the layers.
    expect(screen.queryByText("Size")).not.toBeInTheDocument();
    // Elevation shows first, with its legend under its fact.
    await waitFor(() => expect(useLandContext.getState().rasters.terrain?.band).toBe(1));
    expect(
      screen.getByRole("img", { name: "Map colours: Elevation, from 62 m to 78 m" }),
    ).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: /^Slope/ }));
    await waitFor(() => expect(useLandContext.getState().rasters.terrain?.band).toBe(2));

    fireEvent.click(screen.getByRole("button", { name: /^Land cover/ }));
    await waitFor(() => expect(Object.keys(useLandContext.getState().rasters)).toEqual(["cover"]));
    expect(useLandContext.getState().rasters.cover?.categorical).toBe(true);
    // Shown, the land-cover fact lists every class found, with its colour: it is the legend.
    const coverFact = screen.getByRole("button", { name: /^Land cover/ });
    expect(coverFact).toHaveTextContent("Grassland 64%Tree cover 26%Built-up 10%");
    expect(coverFact).not.toHaveTextContent("Open water");
    expect(screen.queryByRole("img", { name: /Map colours/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Land cover/ })).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    // Pressing the shown layer again clears the map.
    fireEvent.click(screen.getByRole("button", { name: /^Land cover/ }));
    await waitFor(() => expect(useLandContext.getState().rasters).toEqual({}));
    // Greenness has not been compared yet: it is the one next step.
    expect(screen.getByRole("button", { name: "Compare greenness with last year" })).toBeVisible();
  });

  it("offers land cover as the one next step after the terrain", async () => {
    const terrainRun = investigation("inv-terrain", "Surface elevation and slope");
    const { posts } = mockApi({
      investigations: [terrainRun],
      details: { "inv-terrain": detail(terrainRun, [run("r1", "succeeded")], "terrain") },
      rasters: [terrain],
    });
    mount();
    fireEvent.click(await screen.findByRole("button", { name: "Map land cover" }));
    expect(screen.queryByRole("button", { name: /Compare greenness/ })).not.toBeInTheDocument();
    await waitFor(() => expect(runBodies(posts)).toHaveLength(1));
    expect(runBodies(posts)[0]).toMatchObject({
      analysis: { dataset: "esa-worldcover-2021", resolutionM: 10 },
    });
  });

  it("shows a running analysis on one line and can stop it", async () => {
    const terrainRun = investigation("inv-terrain", "Surface elevation and slope");
    const { posts } = mockApi({
      investigations: [terrainRun],
      details: { "inv-terrain": detail(terrainRun, [run("r1", "running")]) },
    });
    mount();
    const status = await screen.findByText("Reading the terrain…");
    expect(status.closest("[role=status]")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Read the terrain" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    await waitFor(() =>
      expect(posts).toContainEqual({
        path: "/api/v1/research/runs/{run_id}/cancel",
        init: { params: { path: { run_id: "r1" } } },
      }),
    );
  });

  it("says what failed and offers to try again", async () => {
    const terrainRun = investigation("inv-terrain", "Surface elevation and slope");
    mockApi({
      investigations: [terrainRun],
      details: {
        "inv-terrain": detail(terrainRun, [run("r1", "failed", "The elevation tiles timed out.")]),
      },
    });
    mount();
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The terrain could not be read. The elevation tiles timed out.",
    );
    expect(screen.getByRole("button", { name: "Try again" })).toBeVisible();
  });

  it("notes results made for an earlier outline and runs them again", async () => {
    const terrainRun = investigation("inv-terrain", "Surface elevation and slope", 1);
    const { posts } = mockApi({
      investigations: [terrainRun],
      details: { "inv-terrain": detail(terrainRun, [run("r1", "succeeded")], "terrain") },
      rasters: [terrain],
    });
    mount({ ...land, revision: 2 });
    expect(await screen.findByText(/Made for an earlier outline/)).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Run again" }));
    await waitFor(() => expect(runBodies(posts)).toHaveLength(1));
    expect(runBodies(posts)[0]).toMatchObject({ analysis: { dataset: "cop-dem-glo-30" } });
  });

  it("reads the spot picked with What's here and pins it", async () => {
    const terrainRun = investigation("inv-terrain", "Surface elevation and slope");
    const coverRun = investigation("inv-cover", "Land cover in 2021");
    mockApi({
      investigations: [terrainRun, coverRun],
      details: {
        "inv-terrain": detail(terrainRun, [run("r1", "succeeded")], "terrain"),
        "inv-cover": detail(coverRun, [run("r2", "succeeded")], "cover"),
      },
      rasters: [cover, terrain],
    });
    const view = mount();
    expect(await screen.findByText(/choose What's here/)).toBeVisible();
    act(() =>
      useSelection.getState().setSelection({
        kind: "ground",
        title: "Location",
        longitude: -77.05,
        latitude: 38.888,
        height: null,
        terrainHeight: null,
        at: 1,
      }),
    );
    expect(await screen.findByText("71 m · 3% slope · Grassland")).toBeVisible();
    await waitFor(() =>
      expect(useLandMap.getState().pins.overview?.[0]).toMatchObject({
        position: [-77.05, 38.888],
        label: "71 m · 3% slope · Grassland",
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Clear this spot" }));
    await waitFor(() => expect(useLandMap.getState().pins.overview).toBeUndefined());
    // Closing the lens takes its raster off the map.
    view.unmount();
    expect(useLandContext.getState().rasters).toEqual({});
  });

  it("ignores spots outside the land", async () => {
    const terrainRun = investigation("inv-terrain", "Surface elevation and slope");
    const { gets } = mockApi({
      investigations: [terrainRun],
      details: { "inv-terrain": detail(terrainRun, [run("r1", "succeeded")], "terrain") },
      rasters: [terrain],
    });
    useSelection.getState().setSelection({
      kind: "ground",
      title: "Location",
      longitude: -70,
      latitude: 40,
      height: null,
      terrainHeight: null,
      at: 1,
    });
    mount();
    expect(await screen.findByText(/choose What's here/)).toBeVisible();
    expect(gets.some((call) => call.path.endsWith("/sample"))).toBe(false);
  });

  it("keeps the full tools in Details: measurement, statistics, classes and month choice", async () => {
    const terrainRun = investigation("inv-terrain", "Surface elevation and slope");
    const coverRun = investigation("inv-cover", "Land cover in 2021");
    const { posts } = mockApi({
      investigations: [terrainRun, coverRun],
      details: {
        "inv-terrain": detail(terrainRun, [run("r1", "succeeded")], "terrain"),
        "inv-cover": detail(coverRun, [run("r2", "succeeded")], "cover"),
      },
      rasters: [cover, terrain],
    });
    mount();
    fireEvent.click(await screen.findByText("Details and sources"));
    fireEvent.change(screen.getByLabelText("Map measurement"), { target: { value: "slope" } });
    await waitFor(() => expect(useLandContext.getState().rasters.terrain?.band).toBe(2));
    fireEvent.change(screen.getByLabelText(/Map opacity/), { target: { value: "0.4" } });
    expect(useLandContext.getState().rasters.terrain?.opacity).toBe(0.4);
    expect(screen.getByRole("region", { name: "Elevation statistics" })).toBeVisible();
    const classes = screen.getByRole("region", { name: "Land-cover classes" });
    expect(classes).toHaveTextContent("Built-up10%");
    expect(classes).not.toHaveTextContent("Open water");
    expect(screen.getAllByRole("button", { name: "Download GeoTIFF" })).toHaveLength(2);
    expect(screen.getAllByRole("link", { name: "CC-BY-4.0" })[0]).toHaveAttribute(
      "href",
      "https://example.org/license",
    );
    fireEvent.change(screen.getByLabelText("Observation month 1"), {
      target: { value: "2024-07" },
    });
    fireEvent.change(screen.getByLabelText("Observation month 2"), {
      target: { value: "2025-07" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Compare these months" }));
    await waitFor(() => expect(runBodies(posts)).toHaveLength(1));
    expect(runBodies(posts)[0]).toMatchObject({
      analysis: {
        dataset: "sentinel-2-ndvi",
        resolutionM: 20,
        periods: [
          { startDate: "2024-07-01", endDate: "2024-07-31" },
          { startDate: "2025-07-01", endDate: "2025-07-31" },
        ],
      },
    });
  });
});

describe("Overview words", () => {
  it("turns slope degrees into percent and plain words", () => {
    expect(slopePercent(45)).toBeCloseTo(100);
    expect(slopeWord(0.5)).toBe("flat");
    expect(slopeWord(3)).toBe("gentle");
    expect(slopeWord(15)).toBe("moderate");
    expect(slopeWord(16)).toBe("steep");
    const summary = slopeSummary(terrain.metadata.bands[1] ?? null);
    expect(summary && slopeValue(summary)).toBe("Gentle, typically 3% · up to 11%");
    expect(slopeSummary({ ...band({}), validCells: 0 })).toBeNull();
  });

  it("names heights in the chosen units, below sea level included", () => {
    expect(elevationPhrase(62.2, 78.4, "metric")).toBe("62–78 m above sea level");
    expect(elevationPhrase(62.2, 78.4, "imperial")).toBe("204–257 ft above sea level");
    expect(elevationPhrase(-80, -60, "metric")).toBe("60–80 m below sea level");
    expect(elevationPhrase(-3, 12, "metric")).toBe("from 3 m below to 12 m above sea level");
    expect(elevationPhrase(70.2, 70.4, "metric")).toBe("about 70 m above sea level");
  });

  it("describes cover by its largest classes", () => {
    const grass = cls(30, "Grassland", "#ffff4c", 64);
    const trees = cls(10, "Tree cover", "#006400", 26);
    expect(coverPhrase([grass, trees])).toBe("about two-thirds grassland");
    expect(coverPhrase([{ ...trees, fraction: 0.95 }])).toBe("almost all tree cover");
    expect(
      coverPhrase([
        { ...grass, fraction: 0.2 },
        { ...trees, fraction: 0.18 },
      ]),
    ).toBe("a mix of grassland and tree cover");
    expect(coverPhrase([cls(50, "Built-up", "#fa0000", 80)])).toBe("mainly built-up land");
  });

  it("compares greenness without inventing a trend", () => {
    const series: Timeline = {
      observations: [2024, 2025].map((year, index) => ({
        period: { startDate: `${year}-07-01`, endDate: `${year}-07-31` },
        band: index + 1,
        sceneId: null,
        acquiredAt: `${year}-07-16T12:00:00Z`,
        candidateCount: 5,
        catalogTruncated: false,
        qualityCandidatesExamined: 2,
        validCells: 80,
        coverageFraction: 0.8,
        meanNdvi: 0.5,
        commonMeanNdvi: 0.5,
        qualityCounts: {},
        explanation: "",
      })),
      commonCells: 80,
      commonCoverageFraction: 0.8,
      changeBand: 3,
      changeCells: 80,
      meanChange: 0.05,
    };
    expect(greenness(series)?.value).toMatch(/^Slightly greener than in /);
    expect(greenness(series)?.note).toMatch(/80% of the land/);
    expect(greenness({ ...series, meanChange: -0.3 })?.value).toMatch(/^Much less green than/);
    expect(greenness({ ...series, meanChange: 0.01 })?.value).toMatch(/^About as green as/);
    expect(greenness({ ...series, meanChange: null, changeCells: 0 })?.value).toBe(
      "Clouds hid the comparison",
    );
  });

  it("builds one headline from what exists", () => {
    expect(overviewHeadline({ terrain, cover: null, vegetation: null }, "metric")).toBe(
      "Mostly gentle slopes, 62–78 m above sea level.",
    );
    expect(overviewHeadline({ terrain: null, cover, vegetation: null }, "metric")).toBe(
      "About two-thirds grassland.",
    );
    expect(overviewHeadline({ terrain: null, cover: null, vegetation: null }, "metric")).toBeNull();
  });
});
