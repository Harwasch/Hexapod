import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { components, LandArea } from "@twin/contracts";
import { footprintAreaM2 } from "@twin/geo";

import { api } from "@/api/client";
import { SolarAssessmentView } from "@/features/land/SolarAssessmentView";
import { useLens } from "@/features/land/surface/lenses";
import { SolarLens } from "@/features/land/surface/lenses/SolarLens";
import {
  parseSolarDraft,
  settingsProblem,
  settingsSummary,
  solarDraftKey,
  solarRequest,
  type SolarAssessment,
} from "@/features/land/surface/lenses/solar/solarDraft";
import {
  economicsDraftKey,
  solarFinance,
  type Scenario,
  type ScenarioResult,
} from "@/features/land/surface/lenses/solar/solarEconomics";
import {
  assessmentHeadline,
  energyLabel,
  paybackWords,
  powerLabel,
  zonePlace,
  zoneSize,
} from "@/features/land/surface/lenses/solar/solarWords";
import { useLandContext } from "@/state/landContext";
import type * as LandIdentity from "@/state/landIdentity";
import { useLandMap } from "@/state/landMap";

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

type RunRead = components["schemas"]["RunRead"];

const boundary = {
  type: "Polygon" as const,
  coordinates: [
    [
      [-77.055, 38.886],
      [-77.045, 38.886],
      [-77.045, 38.891],
      [-77.055, 38.891],
      [-77.055, 38.886],
    ],
  ],
};
const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Public fixture",
  description: "",
  revision: 1,
  areaM2: footprintAreaM2(boundary),
  perimeterM: 2800,
  boundary,
  source: { method: "drawn", label: "Synthetic" },
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};
/** A strip along the south edge of the land. */
const zone = {
  type: "Polygon" as const,
  coordinates: [
    [
      [-77.052, 38.8862],
      [-77.048, 38.8862],
      [-77.048, 38.8872],
      [-77.052, 38.8872],
      [-77.052, 38.8862],
    ],
  ],
};
const MONTHLY = [80, 100, 150, 180, 200, 205, 212, 190, 160, 120, 85, 61];
function assessment(overrides: Partial<SolarAssessment["metadata"]> = {}): SolarAssessment {
  return {
    id: "44444444-4444-4444-8444-444444444444",
    landId: land.id,
    runId: "55555555-5555-4555-8555-555555555555",
    boundaryRevision: 1,
    request: { ...solarRequest(zone, "Drawn on the map; check it on site."), year: 2025 },
    sha256: "a".repeat(64),
    byteSize: 1000,
    createdAt: "2026-10-10T12:00:00Z",
    stale: false,
    metadata: {
      algorithm: "pvlib-hourly",
      modelVersion: "1",
      sourceUrl: "https://power.larc.nasa.gov/api/temporal/hourly/point",
      sourceSha256: "b".repeat(64),
      sourceBytes: 100,
      sourceHeader: {},
      sourceUnits: {},
      sourceEtag: null,
      sourceLastModified: null,
      retrievedAt: "2026-10-10T12:00:00Z",
      attribution: "NASA POWER",
      license: "NASA open data",
      documentationUrl: "https://power.larc.nasa.gov/docs/",
      longitude: -77.05,
      latitude: 38.8867,
      sourceElevationM: 20,
      mappedZoneAreaM2: 24281,
      capacityKwDc: 1200,
      inverterKwAc: 1000,
      expectedHours: 8760,
      validHours: 8760,
      completeYear: true,
      invalidParameterHours: {},
      modeledGenerationKwh: 1_874_000,
      annualGenerationKwh: 1_874_000,
      unshadedGenerationKwh: 1_950_000,
      planeIrradiationKwhM2: 1700,
      unshadedPlaneIrradiationKwhM2: 1770,
      horizonSkyFraction: 0.98,
      peakAcKw: 990,
      monthly: MONTHLY.map((mwh, index) => ({
        month: index + 1,
        expectedHours: 730,
        validHours: 730,
        generationKwh: mwh * 1000,
        unshadedGenerationKwh: mwh * 1040,
        planeIrradiationKwhM2: 140,
        unshadedPlaneIrradiationKwhM2: 146,
      })),
      limitations: ["One historical weather year is not a forecast."],
      ...overrides,
    },
  };
}
const result: ScenarioResult = {
  algorithm: "solar-hourly-cash-flow/1",
  summary: {
    capacityKwDc: 1200,
    firstYearGenerationKwh: 1_874_000,
    netPresentValue: 210_000,
    initialEquity: 1_500_000,
    annualLoanPayment: 0,
    paybackYear: 9,
    lifetimeNetCashFlow: 2_000_000,
    currency: "USD",
  },
  rows: [
    { year: 0, cumulativeCashFlow: -1_500_000, netCashFlow: -1_500_000 },
    { year: 1, cumulativeCashFlow: -1_300_000, netCashFlow: 200_000 },
    { year: 2, cumulativeCashFlow: 100_000, netCashFlow: 1_400_000 },
  ],
  sensitivity: [{ case: "Generation -20%", netPresentValue: 90_000 }],
  limitations: ["Annual end-of-year cash flows."],
};
const INVESTIGATION = "22222222-2222-4222-8222-222222222222";
const investigation = {
  id: INVESTIGATION,
  landId: land.id,
  title: "Hourly solar assessment · 2025",
  question: "Hourly solar assessment · 2025",
  boundaryRevision: 1,
  createdAt: "2026-10-11T10:00:00Z",
  stale: false,
};
function run(status: RunRead["status"], error: string | null = null): RunRead {
  return {
    id: "33333333-3333-4333-8333-333333333333",
    investigationId: INVESTIGATION,
    question: "Calculate hourly generation",
    kind: "solar",
    status,
    budget: {},
    attempt: 1,
    error,
    createdAt: "2026-10-11T10:00:00Z",
    startedAt: null,
    finishedAt: null,
    analysis: { ...solarRequest(zone, "Drawn on the map; check it on site."), year: 2025 },
  };
}

interface Fixture {
  assessments?: SolarAssessment[];
  scenarios?: () => Scenario[];
  investigations?: unknown[];
  runs?: RunRead[];
}
function serve({
  assessments = [],
  scenarios = () => [],
  investigations = [],
  runs = [],
}: Fixture) {
  return vi.spyOn(api, "GET").mockImplementation((path: string) => {
    const data = path.endsWith("/solar-assessments")
      ? assessments
      : path.endsWith("/solar-assessments/{assessment_id}")
        ? assessments[0]
        : path.endsWith("/scenarios")
          ? scenarios()
          : path.endsWith("/investigations")
            ? investigations
            : path.endsWith("/investigations/{investigation_id}")
              ? {
                  investigation,
                  runs,
                  findings: [],
                  evidence: [],
                  artifacts: [],
                  messages: [],
                  page: { offset: 0, limit: 100, total: 0 },
                }
              : undefined;
    if (data === undefined)
      return Promise.resolve({ data: undefined, response: new Response(null, { status: 404 }) });
    return Promise.resolve({ data, response: new Response() });
  });
}
function mount(node = <SolarLens land={land} />) {
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      {node}
    </QueryClientProvider>,
  );
}
const draftKey = solarDraftKey("pilot", land.id);
let pendingPick: ((value: null) => void) | null = null;
let queue: { longitude: number; latitude: number; x: number; y: number }[] = [];

beforeEach(() => {
  access.canEdit = true;
  queue = [];
  pendingPick = null;
  scene.areas.pickGround.mockImplementation(() => {
    const next = queue.shift();
    if (next) return Promise.resolve(next);
    return new Promise((resolve) => {
      pendingPick = resolve;
    });
  });
  scene.areas.cancelPick.mockImplementation(() => {
    pendingPick?.(null);
    pendingPick = null;
  });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  scene.areas.pickGround.mockReset();
  scene.areas.cancelPick.mockReset();
  localStorage.clear();
  useLandContext.getState().clear();
  useLens.getState().setLens("overview");
});

describe("solar model and words", () => {
  it("keeps blank numbers in drafts and rejects broken outlines", () => {
    const draft = solarRequest(land.boundary);
    draft.moduleAreaM2 = Number.NaN;
    expect(parseSolarDraft(JSON.stringify(draft)).moduleAreaM2).toBeNaN();
    expect(() =>
      parseSolarDraft(
        JSON.stringify({ ...draft, arrayZone: { type: "Point", coordinates: [0, 0] } }),
      ),
    ).toThrow();
    expect(() =>
      parseSolarDraft(JSON.stringify({ ...draft, horizon: [{ azimuthDegrees: "bad" }] })),
    ).toThrow();
  });

  it("starts panels at 40% of the array, facing the equator", () => {
    const north = solarRequest(zone);
    expect(north.moduleAreaM2).toBe(Math.round(footprintAreaM2(zone) * 0.4));
    expect(north.azimuthDegrees).toBe(180);
    const south = solarRequest({
      type: "Polygon",
      coordinates: [
        [
          [150, -33],
          [150.01, -33],
          [150.01, -33.01],
          [150, -33],
        ],
      ],
    });
    expect(south.azimuthDegrees).toBe(0);
    expect(settingsSummary(north)).toMatch(
      /20° tilt, facing south · 20% panels over 40% of the area/,
    );
  });

  it("explains a setting that cannot be used before asking the server", () => {
    const request = solarRequest(zone, "Drawn");
    expect(settingsProblem(request)).toBeNull();
    expect(settingsProblem({ ...request, tiltDegrees: 95 })).toMatch(/Tilt above flat must be/);
    expect(settingsProblem({ ...request, moduleAreaM2: Number.NaN })).toBe(
      "Fill in total panel area.",
    );
    expect(settingsProblem({ ...request, moduleAreaM2: 1e6 })).toMatch(/need more room/);
    expect(
      settingsProblem({
        ...request,
        horizon: [
          { azimuthDegrees: 0, elevationDegrees: 5 },
          { azimuthDegrees: 90, elevationDegrees: 5 },
          { azimuthDegrees: 180, elevationDegrees: 5 },
        ],
      }),
    ).toMatch(/four different directions/);
    expect(settingsProblem({ ...request, zoneBasis: " " })).toMatch(/outline came from/);
  });

  it("says sizes and places the way people do", () => {
    expect(powerLabel(1234)).toBe("1.2 MW");
    expect(powerLabel(120.4)).toBe("120 kW");
    expect(energyLabel(1_874_000)).toBe("1,900 MWh");
    expect(energyLabel(8_400)).toBe("8.4 MWh");
    expect(energyLabel(640)).toBe("640 kWh");
    expect(zoneSize(24281)).toBe("6-acre");
    expect(zoneSize(450)).toBe("450 m²");
    expect(zonePlace(zone, land)).toBe("on the south side");
    expect(zonePlace(land.boundary, land)).toBe("covering the whole land");
    expect(assessmentHeadline(assessment(), land)).toBe(
      "A 6-acre array on the south side could make about 1,900 MWh a year.",
    );
    expect(paybackWords(9, 25)).toBe("Pays back in year 9");
    expect(paybackWords(null, 25)).toBe("Does not pay back within 25 years");
  });

  it("links the physics of a complete year and refuses a partial one", () => {
    const saved = assessment();
    const finance = solarFinance(saved);
    expect(finance.annualPlaneIrradiationKwhM2).toBe(1770);
    expect(finance.usableRoofAreaM2).toBe(saved.request.moduleAreaM2);
    expect(finance.shadeLoss).toBe(saved.request.additionalShadeLoss);
    expect(() =>
      solarFinance(assessment({ completeYear: false, annualGenerationKwh: null })),
    ).toThrow(/complete/);
  });
});

describe("the Solar lens", { timeout: 20_000 }, () => {
  it("starts empty with one way in, and keeps viewers read-only", async () => {
    serve({});
    const view = mount();
    expect(await screen.findByText("See what the sun could make here.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Sketch an array" })).toBeEnabled();
    view.unmount();
    access.canEdit = false;
    mount();
    expect(await screen.findByText(/No one has sketched a solar array/)).toBeVisible();
    expect(screen.queryByRole("button", { name: "Sketch an array" })).toBeNull();
  });

  it("draws an array on the map, sizes it at once and runs the hourly model on request", async () => {
    serve({});
    const post = vi.spyOn(api, "POST").mockImplementation((path: string) =>
      Promise.resolve({
        data: path.endsWith("/preview")
          ? {
              mappedZoneAreaM2: footprintAreaM2(zone),
              capacityKwDc: 3100,
              inverterKwAc: 2580,
              longitude: -77.05,
              latitude: 38.8867,
            }
          : path.endsWith("/investigations")
            ? investigation
            : run("queued"),
        response: new Response(),
      }),
    );
    queue = (zone.coordinates[0] ?? []).slice(0, 4).map(([longitude = 0, latitude = 0]) => ({
      longitude,
      latitude,
      x: 0.5,
      y: 0.5,
    }));
    mount();
    fireEvent.click(await screen.findByRole("button", { name: "Sketch an array" }));
    expect(await screen.findByText(/4 corners · 9\.\d acres/)).toBeVisible();
    expect(useLandContext.getState().layers["solar-draft"]?.features).toHaveLength(5);
    expect(useLandContext.getState().pointPicker).toBe(`solar-draft:${land.id}`);
    expect(post).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(scene.areas.cancelPick).toHaveBeenCalled();
    expect(useLandContext.getState().pointPicker).toBeNull();
    expect(
      await screen.findByRole("heading", { name: /9\.\d-acre array on the south side/ }),
    ).toBeVisible();
    const runButton = screen.getByRole("button", { name: "Run the hourly model" });
    await waitFor(() => expect(runButton).toBeEnabled(), { timeout: 4000 });
    expect(post).toHaveBeenCalledTimes(1);
    expect(post.mock.calls[0]?.[0]).toMatch(/preview$/);
    expect(screen.getByText(/About 3\.1 MW of panels, 2\.6 MW of inverters/)).toBeVisible();
    expect(useLandMap.getState().pins.solar?.[0]?.label).toBe("Planned · ≈ 3.1 MW");
    expect(localStorage.getItem(draftKey)).toContain('"hasZone":true');

    fireEvent.click(runButton);
    await waitFor(() => expect(post).toHaveBeenCalledTimes(3));
    const body = (post.mock.calls[2]?.[1] as unknown as { body: Record<string, unknown> }).body;
    expect(body).toMatchObject({ kind: "solar" });
    const analysis = body.analysis as { arrayZone: typeof zone; zoneBasis: string };
    expect(analysis.arrayZone.coordinates[0]).toHaveLength(5);
    expect(analysis.zoneBasis).toBe("Drawn on the map; check it on site.");
    await waitFor(() => expect(localStorage.getItem(draftKey)).toBeNull());
    expect(localStorage.getItem(`${draftKey}:run`)).toBe(INVESTIGATION);
    expect(localStorage.getItem(`${draftKey}:operation`)).toBeNull();
  });

  it("picks up an unfinished array after a reload, blank values included", async () => {
    serve({});
    vi.spyOn(api, "POST").mockResolvedValue({
      data: {
        mappedZoneAreaM2: 1,
        capacityKwDc: 1,
        inverterKwAc: 1,
        longitude: 0,
        latitude: 0,
      },
      response: new Response(),
    });
    localStorage.setItem(
      draftKey,
      JSON.stringify({ revision: 0, request: solarRequest(zone, "Drawn"), points: [] }),
    );
    const view = mount();
    expect(await screen.findByText(/20° tilt, facing south/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Tilt above flat (degrees)"), {
      target: { value: "30" },
    });
    expect(screen.getByText(/30° tilt, facing south/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Total panel area (m²)"), { target: { value: "" } });
    expect(screen.getByText("Fill in total panel area.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Run the hourly model" })).toBeDisabled();
    view.unmount();
    mount();
    expect(await screen.findByLabelText("Tilt above flat (degrees)")).toHaveValue(30);
    expect(screen.getByLabelText("Total panel area (m²)")).toHaveValue(null);
  });

  it("warns when a recovered array was drawn for an earlier outline", async () => {
    serve({});
    localStorage.setItem(
      draftKey,
      JSON.stringify({ revision: 3, request: solarRequest(zone, "Drawn"), points: [] }),
    );
    mount();
    expect(await screen.findByText(/outline changed since you drew this array/)).toBeVisible();
  });

  it("shows a saved array: the sentence, facts tied to the map, the months and its pin", async () => {
    serve({ assessments: [assessment()] });
    mount();
    expect(
      await screen.findByRole("heading", {
        name: "A 6-acre array on the south side could make about 1,900 MWh a year.",
      }),
    ).toBeVisible();
    const facts = screen.getByRole("button", { name: "Size: show on map" });
    expect(facts).toHaveTextContent("1.2 MW of panels on 6 acres");
    expect(screen.getByRole("button", { name: "Yearly energy: show on map" })).toHaveTextContent(
      "About 1,900 MWh with 2025 weather",
    );
    expect(screen.getByText("Horizon and shade cost 3.9% of the energy")).toBeVisible();
    expect(
      screen.getByRole("img", {
        name: "Monthly energy. Most in July, 212 MWh; least in December, 61 MWh.",
      }),
    ).toBeVisible();
    expect(useLandContext.getState().layers["solar-array"]).toBeDefined();
    expect(useLandMap.getState().pins.solar?.[0]?.label).toBe("≈ 1.2 MW · 1,900 MWh/yr");
    expect(screen.getByRole("button", { name: "Work out costs and payback" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Download hourly data and sources" })).toBeEnabled();
  });

  it("is honest about a year with weather gaps", async () => {
    serve({
      assessments: [
        assessment({ completeYear: false, annualGenerationKwh: null, validHours: 8000 }),
      ],
    });
    mount();
    expect(await screen.findByRole("heading", { name: /no full-year total/ })).toBeVisible();
    expect(screen.getByRole("button", { name: "Energy in 2025: show on map" })).toHaveTextContent(
      "1,900 MWh in 8,000 of 8,760 hours",
    );
    expect(screen.getByRole("button", { name: "Work out costs and payback" })).toBeDisabled();
    expect(screen.getByText("Needs a full year of weather.")).toBeVisible();
  });

  it("works out costs and payback, keeps the draft, and saves it as a solar scenario", async () => {
    let saved: Scenario[] = [];
    serve({ assessments: [assessment()], scenarios: () => saved });
    const post = vi.spyOn(api, "POST").mockImplementation((path: string, init: unknown) => {
      const body = (init as { body: Scenario }).body;
      if (path.endsWith("/scenarios")) {
        saved = [
          {
            ...body,
            id: "66666666-6666-4666-8666-666666666666",
            landId: land.id,
            revision: 1,
            result,
            stale: false,
            createdAt: "2026-10-11",
            updatedAt: "2026-10-11",
          },
        ];
        return Promise.resolve({ data: saved[0], response: new Response() });
      }
      return Promise.resolve({ data: result, response: new Response() });
    });
    const view = mount();
    fireEvent.click(await screen.findByRole("button", { name: "Work out costs and payback" }));
    expect(
      screen.getByRole("heading", { name: "Costs and payback for the 1.2 MW array" }),
    ).toBeVisible();
    fireEvent.change(screen.getByLabelText("Installed cost"), { target: { value: "1500000" } });
    expect(await screen.findByText("Pays back in year 9", {}, { timeout: 4000 })).toBeVisible();
    const preview = post.mock.calls.at(-1)?.[1] as unknown as { body: Scenario };
    expect(preview.body.solarAssessmentId).toBe(assessment().id);
    expect(preview.body.inputs).toMatchObject({ kind: "solar", installedCost: 1_500_000 });
    expect(localStorage.getItem(economicsDraftKey("pilot", land.id))).toContain("1500000");

    view.unmount();
    mount();
    expect(await screen.findByLabelText("Installed cost")).toHaveValue(1_500_000);
    const save = screen.getByRole("button", { name: "Save costs" });
    await waitFor(() => expect(save).toBeEnabled(), { timeout: 4000 });
    fireEvent.click(save);
    expect(await screen.findByRole("button", { name: "Payback: show on map" })).toHaveTextContent(
      "Pays back in year 9 · worth $210,000 today",
    );
    expect(localStorage.getItem(economicsDraftKey("pilot", land.id))).toBeNull();
    const create = (post.mock.calls as unknown as [string, unknown][]).find(([path]) =>
      path.endsWith("/scenarios"),
    );
    expect((create?.[1] as { body: { requestKey: string } }).body.requestKey).toMatch(
      /^[a-f0-9-]{36}$/,
    );
  });

  it("shows a running model on the map and lets an editor stop it", async () => {
    serve({ investigations: [investigation], runs: [run("running")] });
    const post = vi.spyOn(api, "POST").mockResolvedValue({
      data: run("cancelled"),
      response: new Response(),
    });
    mount();
    expect(
      await screen.findByRole("heading", { name: /Modeling a 9\.\d-acre array with 2025 weather/ }),
    ).toBeVisible();
    expect(useLandContext.getState().layers["solar-running"]).toBeDefined();
    fireEvent.click(within(screen.getByRole("status")).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
    expect(post.mock.calls[0]?.[0]).toBe("/api/v1/research/runs/{run_id}/cancel");
  });

  it("says when the model failed and offers to try again with the same array", async () => {
    serve({
      investigations: [investigation],
      runs: [run("failed", "NASA POWER was unavailable.")],
    });
    vi.spyOn(api, "POST").mockResolvedValue({
      data: { mappedZoneAreaM2: 1, capacityKwDc: 1, inverterKwAc: 1, longitude: 0, latitude: 0 },
      response: new Response(),
    });
    mount();
    expect(
      await screen.findByText("The hourly model did not finish. NASA POWER was unavailable."),
    ).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText(/Settings from the run that did not finish/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Run the hourly model" })).toBeInTheDocument();
  });

  it("takes all of its map marks away when the lens closes", async () => {
    serve({ assessments: [assessment()] });
    const view = mount();
    await screen.findByRole("heading", { name: /1,900 MWh a year/ });
    expect(useLandContext.getState().layers["solar-array"]).toBeDefined();
    view.unmount();
    expect(
      Object.keys(useLandContext.getState().layers).filter((id) => id.startsWith("solar")),
    ).toEqual([]);
    expect(useLandMap.getState().pins.solar).toBeUndefined();
  });
});

it("keeps the old assessment view working and leads to the Solar lens", async () => {
  serve({ assessments: [assessment()] });
  mount(<SolarAssessmentView id={assessment().id} />);
  expect(await screen.findByText(/could make about 1,900 MWh a year/)).toBeVisible();
  act(() => {
    fireEvent.click(screen.getByRole("button", { name: "Open in Solar" }));
  });
  expect(useLens.getState().lens).toBe("solar");
  expect(useLandContext.getState().selectedSolarId).toBe(assessment().id);
});
