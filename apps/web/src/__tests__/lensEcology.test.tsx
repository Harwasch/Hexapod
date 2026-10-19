import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { components, LandArea } from "@twin/contracts";

import { api } from "@/api/client";
import { parseSurveyCorners, parseSurveyDraft } from "@/features/land/surveyDraft";
import { useAgentBus } from "@/features/land/surface/agentBus";
import { useLens } from "@/features/land/surface/lenses";
import { EcologyLens } from "@/features/land/surface/lenses/EcologyLens";
import {
  DEFAULT_REQUEST,
  ECOLOGY_TITLE,
  ecologyHeadline,
  ecologyNames,
  plainSiteName,
  readEcology,
  titleCase,
} from "@/features/land/surface/lenses/ecology/ecologyData";
import { useLandContext } from "@/state/landContext";
import type * as LandIdentity from "@/state/landIdentity";
import { useLandMap } from "@/state/landMap";
import { useShowOnce } from "@/state/showOnce";
import { useUi } from "@/state/ui";

const scene = vi.hoisted(() => ({
  areas: { pickGround: vi.fn(), cancelPick: vi.fn() },
  camera: { flyToRectangle: vi.fn() },
}));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));
const access = vi.hoisted(() => ({ canEdit: true }));
vi.mock("@/state/landIdentity", async (importOriginal) => ({
  ...(await importOriginal<typeof LandIdentity>()),
  useLandCanEdit: () => access.canEdit,
}));

type Detail = components["schemas"]["InvestigationDetail"];
type Survey = components["schemas"]["SurveyRead"];
type Summary = components["schemas"]["SurveySummary"];

const boundary = {
  type: "Polygon" as const,
  coordinates: [
    [
      [-77.055, 38.887],
      [-77.05, 38.887],
      [-77.05, 38.889],
      [-77.055, 38.887],
    ],
  ],
};
const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Software fixture",
  description: "",
  revision: 2,
  areaM2: 10000,
  perimeterM: 500,
  boundary,
  source: { method: "drawn", label: "Synthetic" },
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};
const inv: components["schemas"]["InvestigationRead"] = {
  id: "22222222-2222-4222-8222-222222222222",
  landId: land.id,
  title: ECOLOGY_TITLE,
  question: ECOLOGY_TITLE,
  boundaryRevision: 2,
  createdAt: "2026-10-09T10:00:00Z",
  stale: false,
};
const RUN = "33333333-3333-4333-8333-333333333333";
const run: components["schemas"]["RunRead"] = {
  id: RUN,
  investigationId: inv.id,
  question: ECOLOGY_TITLE,
  kind: "ecology",
  status: "succeeded",
  budget: {},
  attempt: 1,
  error: null,
  createdAt: "2026-10-09",
  startedAt: null,
  finishedAt: null,
  analysis: DEFAULT_REQUEST,
};
function evidence(id: string, provider: string, excerpt: unknown) {
  return {
    id,
    runId: RUN,
    provider,
    title: `${provider} record`,
    license: provider === "gbif-occurrences" ? "CC0" : "Public domain",
    attribution: `${provider} attribution`,
    excerpt: JSON.stringify(excerpt),
    relevanceNote: "Fixture",
    retrievedAt: "2026-10-09",
    spatialRelevance: "regional" as const,
  };
}
function table(
  id: string,
  title: string,
  evidenceIds: string[],
  columns: string[],
  rows: Record<string, string | number | boolean | null>[],
) {
  return {
    id,
    runId: RUN,
    title,
    method: `${title} method.`,
    evidenceIds,
    output: {
      kind: "table" as const,
      columns: columns.map((key) => ({ key, label: key })),
      rows,
    },
  };
}
function finding(id: string, title: string, evidenceIds: string[]) {
  return {
    id,
    runId: RUN,
    title,
    summary: `${title} summary`,
    category: "ecology" as const,
    confidence: "supported" as const,
    uncertainty: `${title} caveat`,
    evidenceIds,
    disposition: "visible" as const,
  };
}
const detail: Detail = {
  investigation: inv,
  page: { offset: 0, limit: 200, totals: {} },
  runs: [run],
  messages: [],
  evidence: [
    evidence("e-regions", "epa-ecoregions", { records: [] }),
    evidence("e-sites", "usda-ecological-sites", { samplePoint: [-77.052, 38.8875] }),
    evidence("e-gbif-1", "gbif-occurrences", { species: "Quercus alba" }),
    evidence("e-names", "col-taxonomy", {}),
  ],
  findings: [
    finding("f-regions", "EPA regional ecological context (2011)", ["e-regions"]),
    finding("f-sites", "USDA soil-linked ecological site references", ["e-sites"]),
    finding("f-gbif", "GBIF biodiversity observations", ["e-gbif-1"]),
    finding("f-names", "Name review · Quercus alba", ["e-names"]),
  ],
  artifacts: [
    table(
      "a-regions",
      "EPA regional ecological context (2011)",
      ["e-regions"],
      ["US_L4CODE", "US_L4NAME", "US_L3NAME", "NA_L1NAME"],
      [
        {
          US_L4CODE: "64c",
          US_L4NAME: "Piedmont Uplands",
          US_L3NAME: "Northern Piedmont",
          NA_L1NAME: "EASTERN TEMPERATE FORESTS",
        },
      ],
    ),
    table(
      "a-sites",
      "USDA soil-linked ecological site references",
      ["e-sites"],
      ["muname", "compname", "comppct_r", "ecoclassid", "ecoclassname"],
      [
        {
          muname: "Glenelg loam",
          compname: "Glenelg",
          comppct_r: 85,
          ecoclassid: "F148XY001MD",
          ecoclassname: 'Loamy Upland 16-20" PZ',
        },
      ],
    ),
    table(
      "a-gbif",
      "Biodiversity observation sample",
      ["e-gbif-1"],
      ["species", "date", "basis", "publisher"],
      [
        { species: "Quercus alba", date: "2015-05-01", basis: "HUMAN_OBSERVATION", publisher: "A" },
        {
          species: "Cardinalis cardinalis",
          date: "2019-06",
          basis: "HUMAN_OBSERVATION",
          publisher: "B",
        },
        {
          species: "Quercus alba",
          date: "2021-04-02T10:00:00",
          basis: "PRESERVED_SPECIMEN",
          publisher: "C",
        },
      ],
    ),
    table(
      "a-names",
      "Name review · Quercus alba",
      ["e-names"],
      ["supplied", "matched", "accepted", "rank", "matchType", "synonym"],
      [
        {
          supplied: "Quercus alba",
          matched: "Quercus alba L.",
          accepted: "Quercus alba L.",
          rank: "SPECIES",
          matchType: "EXACT",
          synonym: false,
        },
      ],
    ),
  ],
};
const events = [
  {
    sequence: 1,
    kind: "source",
    createdAt: "2026-10-09",
    payload: { provider: "epa-ecoregions", status: "available", message: "Found." },
  },
];
const summary: Summary = {
  algorithm: "field-cover-area-weighted-v1",
  landAreaM2: 10000,
  sampledAreaM2: 1000,
  sampledFraction: 0.1,
  plotAreasM2: [1000],
  species: [
    {
      taxon: "Festuca rubra",
      stratum: "herb",
      identifications: ["verified"],
      meanPercent: 40,
      minimumPercent: 40,
      maximumPercent: 40,
      measuredPlots: 1,
      assessedAreaM2: 1000,
      assessedSampleFraction: 1,
    },
    {
      taxon: "Carex sp.",
      stratum: "herb",
      identifications: ["tentative"],
      meanPercent: 90,
      minimumPercent: 90,
      maximumPercent: 90,
      measuredPlots: 1,
      assessedAreaM2: 1000,
      assessedSampleFraction: 1,
    },
  ],
  limitations: ["Synthetic observations."],
};
const savedSurvey: Survey = {
  id: "44444444-4444-4444-8444-444444444444",
  landId: land.id,
  name: "Spring survey",
  boundaryRevision: 2,
  observedOn: "2026-05-01",
  observer: "Tester",
  method: "visual-cover",
  design: "purposive",
  assessedStrata: ["herb"],
  methodNotes: "Synthetic",
  recordedBy: "pilot",
  createdAt: "2026-05-02",
  sha256: "abc123",
  stale: false,
  summary,
  plots: [
    {
      label: "Plot 1",
      boundary,
      completeInventory: false,
      samplePoints: null,
      observations: [
        { taxon: "Festuca rubra", stratum: "herb", identification: "verified", percentCover: 40 },
        { taxon: "Carex sp.", stratum: "herb", identification: "tentative", percentCover: 90 },
      ],
    },
  ],
};
const draft: components["schemas"]["SurveyCreate"] = {
  requestKey: "55555555-5555-4555-8555-555555555555",
  name: "Unfinished",
  observedOn: "2025-07-01",
  observer: "",
  boundaryRevision: 1,
  method: "visual-cover",
  design: "purposive",
  assessedStrata: ["herb"],
  methodNotes: "",
  plots: [
    {
      label: "Plot 1",
      boundary,
      observations: [
        { taxon: "", stratum: "herb", identification: "tentative", percentCover: null },
      ],
    },
  ],
};

interface Fixture {
  investigations?: unknown[];
  detail?: Detail;
  surveys?: Survey[];
}
function serve({ investigations = [], detail: shown, surveys = [] }: Fixture = {}) {
  return vi.spyOn(api, "GET").mockImplementation((path: string) =>
    Promise.resolve({
      data: path.endsWith("/events")
        ? events
        : path.includes("/investigations/{")
          ? shown
          : path.endsWith("/investigations")
            ? investigations
            : path.endsWith("/surveys/{survey_id}")
              ? surveys[0]
              : path.endsWith("/surveys")
                ? surveys
                : [],
      response: new Response(),
    }),
  );
}
function mount() {
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <EcologyLens land={land} />
    </QueryClientProvider>,
  );
}
async function fillBasics() {
  fireEvent.change(screen.getByLabelText("Observer", { exact: true }), {
    target: { value: "Test observer" },
  });
  fireEvent.change(screen.getByLabelText("Survey method and limitations"), {
    target: { value: "Synthetic fixture" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Next: map the plots" }));
  await screen.findByRole("group", { name: "Plots" });
}

beforeEach(() => {
  access.canEdit = true;
  scene.areas.pickGround.mockResolvedValue(null);
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  scene.areas.pickGround.mockReset();
  localStorage.clear();
  useLandContext.getState().clear();
  useLandMap.setState({ pins: {}, focus: null });
  useAgentBus.setState({ request: null });
  useLens.getState().setLens("overview");
  useUi.getState().setPanel(null);
});

it("reads the sources in plain words and keeps survey drafts strict", () => {
  expect(ecologyNames(" Quercus   alba \n\n Carex sp. ", "")).toEqual([
    { scientificName: "Quercus alba", kingdom: null },
    { scientificName: "Carex sp.", kingdom: null },
  ]);
  expect(ecologyNames("Quercus alba\nquercus ALBA", "Plantae")).toBeNull();
  expect(
    ecologyNames(Array.from({ length: 21 }, (_, i) => `Taxon ${i}`).join("\n"), ""),
  ).toBeNull();
  expect(titleCase("MEDITERRANEAN CALIFORNIA")).toBe("Mediterranean California");
  expect(titleCase("SOUTHERN AND CENTRAL PLAINS")).toBe("Southern and Central Plains");
  expect(plainSiteName('R077CY036TX Loamy Upland 16-20" PZ')).toBe("Loamy Upland");
  expect(plainSiteName("SHALLOW RIDGE 10-14 p.z.")).toBe("Shallow Ridge");
  expect(plainSiteName("Moist Loamy Hills")).toBe("Moist Loamy Hills");

  const context = readEcology(detail, RUN, events);
  expect(context.regions?.level4).toEqual(["Piedmont Uplands"]);
  expect(context.sites?.samplePoint).toEqual([-77.052, 38.8875]);
  expect(context.species?.ranked[0]).toEqual({ name: "Quercus alba", count: 2 });
  expect(context.species?.years).toEqual([2015, 2021]);
  expect(context.names).toMatchObject({ exact: 1, unsure: 0 });
  expect(ecologyHeadline(context)).toBe(
    "Northern Piedmont, in the Eastern Temperate Forests. 2 species recorded in and around this land.",
  );

  expect(
    parseSurveyDraft(JSON.stringify(draft)).plots[0]?.observations[0]?.percentCover,
  ).toBeNull();
  expect(() =>
    parseSurveyDraft(JSON.stringify({ ...draft, plots: [{ label: "Broken", observations: [] }] })),
  ).toThrow();
  expect(parseSurveyCorners("[[181,40]]")).toEqual([]);
  expect(parseSurveyCorners("[[-77.05,38.88]]")).toEqual([[-77.05, 38.88]]);
});

it("starts with one button and sensible defaults when nothing has been looked up", async () => {
  serve();
  const post = vi.spyOn(api, "POST").mockImplementation((path: string) =>
    Promise.resolve({
      data: path.endsWith("/investigations") ? inv : run,
      response: new Response(),
    }),
  );
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Learn what lives here" }));
  await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
  expect(post.mock.calls[1]?.[1]).toMatchObject({
    body: { kind: "ecology", analysis: DEFAULT_REQUEST },
  });
});

it("keeps viewers read-only", async () => {
  access.canEdit = false;
  serve();
  mount();
  expect(await screen.findByText(/An editor can look this up/)).toBeVisible();
  expect(screen.queryByRole("button", { name: "Learn what lives here" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Record a field survey" })).toBeNull();
});

it("recovers source choices and reuses the request key after a lost queue response", async () => {
  vi.spyOn(api, "GET").mockImplementation((path: string) =>
    Promise.resolve({
      data: path.includes("/investigations/{") ? { ...detail, runs: [] } : [],
      response: new Response(),
    }),
  );
  const post = vi.spyOn(api, "POST").mockImplementation((path: string) => {
    if (path.endsWith("/investigations"))
      return Promise.resolve({ data: inv, response: new Response() });
    return Promise.reject(new Error("Connection lost after queueing"));
  });
  const view = mount();
  fireEvent.click(await screen.findByText("Choose sources and names"));
  fireEvent.change(screen.getByLabelText("Scientific names to review"), {
    target: { value: "Quercus alba" },
  });
  fireEvent.change(screen.getByLabelText("Kingdom hint"), { target: { value: "Plantae" } });
  fireEvent.click(screen.getByLabelText(/EPA ecoregions/));
  fireEvent.click(screen.getByRole("button", { name: "Run with these choices" }));
  await screen.findByRole("alert");
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Run with these choices" })).toBeEnabled(),
  );
  expect(post).toHaveBeenCalledTimes(2);
  const first = post.mock.calls[1]?.[1];
  view.unmount();
  mount();
  fireEvent.click(await screen.findByText("Choose sources and names"));
  expect(screen.getByLabelText(/EPA ecoregions/)).not.toBeChecked();
  expect(screen.getByLabelText("Kingdom hint")).toHaveValue("Plantae");
  expect(screen.getByLabelText("Scientific names to review")).toHaveValue("Quercus alba");
  fireEvent.click(screen.getByRole("button", { name: "Run with these choices" }));
  await waitFor(() => expect(post).toHaveBeenCalledTimes(3));
  expect(post.mock.calls[2]?.[1]).toEqual(first);
});

it("summarises what lives here in a few facts, pins the soil sample, and asks the agent only when asked", async () => {
  serve({ investigations: [inv], detail });
  mount();
  expect(
    await screen.findByRole("heading", {
      name: "Northern Piedmont, in the Eastern Temperate Forests. 2 species recorded in and around this land.",
    }),
  ).toBeVisible();
  const facts = screen.getByRole("list", { name: "What lives here" });
  expect(facts.children).toHaveLength(4);
  expect(within(facts).getByText("Piedmont Uplands")).toBeVisible();
  expect(within(facts).getByText("Loamy Upland")).toBeVisible();
  expect(within(facts).getByText("2 species in 3 records, 2015–2021")).toBeVisible();
  expect(within(facts).getByText("1 of 1 matched exactly")).toBeVisible();
  expect(within(facts).queryByText(/64c|F148XY001MD/)).toBeNull();
  // Species records have no place on the map; only the soil sample point is pinned.
  expect(useLandMap.getState().pins.ecology?.map((pin) => pin.label)).toEqual([
    "Soil sample · Loamy Upland",
  ]);
  expect(
    Object.values(useLandContext.getState().layers).flatMap((layer) => layer.features),
  ).toEqual([]);
  expect(useAgentBus.getState().request).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Ask" }));
  expect(useAgentBus.getState().request?.question).toContain("saved ecological evidence");
  expect(useLandContext.getState().selectedInvestigationId).toBe(inv.id);
  // The raw tables stay in Details.
  fireEvent.click(screen.getByText("Details and sources"));
  fireEvent.click(screen.getByText("Ecoregions (1)"));
  expect(screen.getByRole("table", { name: "Ecoregions" })).toHaveTextContent("64c");
});

it("labels stale results and runs the same lookup again on the current boundary", async () => {
  const stale = { ...inv, boundaryRevision: 1, stale: true };
  serve({ investigations: [stale], detail: { ...detail, investigation: stale } });
  const post = vi.spyOn(api, "POST").mockImplementation((path: string) =>
    Promise.resolve({
      data: path.endsWith("/investigations") ? inv : run,
      response: new Response(),
    }),
  );
  mount();
  expect(await screen.findByText("Made for an earlier boundary.")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Run again" }));
  await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
  expect(post.mock.calls[1]?.[1]).toMatchObject({ body: { analysis: DEFAULT_REQUEST } });
});

it("shows progress while the lookup runs and can stop it", async () => {
  const running = { ...detail, runs: [{ ...run, status: "running" as const }] };
  serve({ investigations: [inv], detail: running });
  const post = vi
    .spyOn(api, "POST")
    .mockResolvedValue({ data: { ...run, status: "cancelled" }, response: new Response() });
  mount();
  expect(await screen.findByText(/1 of 3 sources done/)).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Stop" }));
  await waitFor(() =>
    expect(post).toHaveBeenCalledWith("/api/v1/research/runs/{run_id}/cancel", {
      params: { path: { run_id: RUN } },
    }),
  );
});

it("draws saved surveys as plots and highlights the selected one with its species", async () => {
  serve({ investigations: [inv], detail, surveys: [savedSurvey] });
  mount();
  const row = await screen.findByRole("button", { name: /^Spring survey ·/ });
  expect(useLandContext.getState().layers["ecology-surveys"]?.features).toHaveLength(1);
  fireEvent.click(row);
  expect(useLandContext.getState().selectedSurveyId).toBe(savedSurvey.id);
  await waitFor(() =>
    expect(useLandContext.getState().layers["ecology-survey"]?.features).toHaveLength(1),
  );
  expect(useLandContext.getState().layers["ecology-surveys"]).toBeUndefined();
  expect(useLandMap.getState().pins.ecology?.map((pin) => pin.label)).toContain(
    "Plot 1 · Carex sp. 90%",
  );
  expect(screen.getByRole("list", { name: "Species in Spring survey" })).toHaveTextContent(
    "Festuca rubra",
  );
  expect(scene.camera.flyToRectangle).toHaveBeenCalled();
  fireEvent.click(screen.getByText("Details and sources"));
  expect(screen.getByText(/abc123/)).toBeInTheDocument();
  fireEvent.click(screen.getByText("Choose sources and names"));
  fireEvent.click(screen.getByRole("button", { name: "Use names from the displayed survey" }));
  expect(screen.getByLabelText("Scientific names to review")).toHaveValue(
    "Festuca rubra\nCarex sp.",
  );
  fireEvent.click(screen.getByRole("button", { name: "Ask about this survey" }));
  expect(useAgentBus.getState().request?.question).toContain(savedSurvey.id);
});

it("records plot corners by clicking the map until the plot is finished", async () => {
  serve();
  const points = [
    [-77.054, 38.8872],
    [-77.051, 38.8872],
    [-77.051, 38.8884],
  ];
  for (const [longitude, latitude] of points)
    scene.areas.pickGround.mockResolvedValueOnce({ longitude, latitude, x: 0, y: 0 });
  scene.areas.pickGround.mockResolvedValue(null);
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Record a field survey" }));
  await fillBasics();
  expect(await screen.findByText("3 corners placed")).toBeVisible();
  expect(useLandContext.getState().pointPicker).toBeNull();
  const sketch = useLandContext.getState().layers["ecology-sketch"];
  expect(sketch?.features.map((feature) => feature.geometry.type)).toEqual([
    "Polygon",
    "Point",
    "Point",
    "Point",
  ]);
  fireEvent.click(screen.getByRole("button", { name: "Finish plot (3 corners)" }));
  expect(screen.getByLabelText("Plot label")).toHaveValue("Plot 1");
  expect(useLandContext.getState().layers["ecology-draft"]?.features).toHaveLength(1);
  expect(useLandContext.getState().layers["ecology-sketch"]).toBeUndefined();
});

it("requires explicit measurements and clears them when the method changes", async () => {
  serve();
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Record a field survey" }));
  await fillBasics();
  fireEvent.click(screen.getByRole("button", { name: "Use entire land as the surveyed plot" }));
  fireEvent.click(screen.getByRole("button", { name: "Add species observation" }));
  expect(screen.getByLabelText("Cover (%)")).toHaveValue(null);
  fireEvent.change(screen.getByLabelText("Cover (%)"), { target: { value: "45" } });
  fireEvent.click(screen.getByRole("button", { name: /Basics/ }));
  fireEvent.change(screen.getByLabelText("Method", { exact: true }), {
    target: { value: "point-intercept" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Next: map the plots" }));
  expect(screen.getByLabelText("Points with taxon")).toHaveValue(null);
  expect(screen.getByLabelText("Sampled points")).toHaveValue(null);
  expect(screen.getByLabelText(/Complete inventory/)).not.toBeChecked();
});

it("invalidates a reviewed survey when it changes and recovers the unfinished draft", async () => {
  serve();
  const post = vi.spyOn(api, "POST").mockResolvedValue({ data: summary, response: new Response() });
  const view = mount();
  fireEvent.click(await screen.findByRole("button", { name: "Record a field survey" }));
  await fillBasics();
  fireEvent.click(screen.getByRole("button", { name: "Use entire land as the surveyed plot" }));
  fireEvent.click(screen.getByRole("button", { name: "Review survey" }));
  await screen.findByRole("button", { name: "Save immutable survey" });
  expect(screen.getByRole("list", { name: "Species cover" })).toHaveTextContent("Carex sp.");
  fireEvent.click(screen.getByRole("button", { name: /Basics/ }));
  fireEvent.change(screen.getByLabelText("Survey name"), {
    target: { value: "Revised field survey" },
  });
  fireEvent.click(screen.getByRole("button", { name: /Review/ }));
  expect(screen.queryByRole("button", { name: "Save immutable survey" })).toBeNull();
  await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
  view.unmount();
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Resume survey" }));
  await screen.findByRole("group", { name: "Plots" });
  fireEvent.click(screen.getByRole("button", { name: /Basics/ }));
  expect(screen.getByLabelText("Survey name")).toHaveValue("Revised field survey");
  expect(screen.getByLabelText("Observer", { exact: true })).toHaveValue("Test observer");
});

it("keeps Show once, everywhere under its own heading, taking the map while the lens is open", async () => {
  serve({ investigations: [inv], detail });
  useLens.getState().setLens("ecology");
  useUi.getState().setPanel("land");
  const view = mount();
  const card = await screen.findByTestId("show-once");
  const lens = screen.getByTestId("land-lens-card-ecology");
  expect(lens).toContainElement(card);
  // The lens still leads with its one sentence; Show once sits below, under its own heading.
  await waitFor(() => expect(within(lens).getAllByRole("heading")[0]).toHaveClass("land-headline"));
  expect(within(lens).getByRole("heading", { name: "Show once, everywhere" })).toBeVisible();
  expect(card).toHaveTextContent("Not a species ID");
  expect(await within(card).findByTestId("show-once-prepare")).toBeVisible();
  // The drape and the taps belong to the card while it is on screen (ShowOnceBridge).
  await waitFor(() => expect(useShowOnce.getState().shown).toBe(true));
  view.unmount();
  expect(useShowOnce.getState().shown).toBe(false);
});

it("counts a saved field survey for the whole land: the fog over it is asked again", async () => {
  serve();
  const post = vi.spyOn(api, "POST").mockImplementation((path: string) =>
    Promise.resolve({
      data: path.endsWith("/preview") ? summary : savedSurvey,
      response: new Response(),
    }),
  );
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const fog = ["land-knowledge", "pilot", land.id];
  client.setQueryData(fog, { cells: [] });
  render(
    <QueryClientProvider client={client}>
      <EcologyLens land={land} />
    </QueryClientProvider>,
  );
  fireEvent.click(await screen.findByRole("button", { name: "Record a field survey" }));
  await fillBasics();
  fireEvent.click(screen.getByRole("button", { name: "Use entire land as the surveyed plot" }));
  fireEvent.click(screen.getByRole("button", { name: "Review survey" }));
  expect(client.getQueryState(fog)?.isInvalidated).toBe(false);
  fireEvent.click(await screen.findByRole("button", { name: "Save immutable survey" }));
  await waitFor(() =>
    expect(post).toHaveBeenCalledWith("/api/v1/land/{land_id}/surveys", expect.anything()),
  );
  await waitFor(() => expect(client.getQueryState(fog)?.isInvalidated).toBe(true));
});
