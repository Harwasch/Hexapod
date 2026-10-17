/**
 * The Time Lantern in the strip (features/lantern/TimeLanternControl.tsx) and the catalog it
 * reads (useDatedSources.ts): where it is offered, the rail newest first, the kind chips, the
 * providers that could not answer, the keys, and the words every source carries.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { DatedImageryCatalog, DatedImagerySource, LandArea } from "@twin/contracts";

import { api } from "@/api/client";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import type { LanternFrame } from "@/cesium/lanternTypes";
import { SceneContext, SceneRegistry } from "@/cesium/SceneContext";
import type { SceneEvents } from "@/cesium/types";
import { LanternBridge } from "@/features/lantern/LanternBridge";
import { providersLine } from "@/features/lantern/lanternWords";
import { TimeLanternControl } from "@/features/lantern/TimeLanternControl";
import {
  accuracyLabel,
  formatDay,
  roundBbox,
  toLanternSource,
} from "@/features/lantern/useDatedSources";
import { Emitter } from "@/lib/emitter";
import { useLand } from "@/state/land";
import { useLantern } from "@/state/lantern";
import { initialCamera, useViewer } from "@/state/viewer";

const square = {
  type: "Polygon" as const,
  coordinates: [
    [
      [-120.06, 34.68],
      [-120.02, 34.68],
      [-120.02, 34.72],
      [-120.06, 34.72],
      [-120.06, 34.68],
    ],
  ],
};

function source(over: Partial<DatedImagerySource>): DatedImagerySource {
  return {
    id: "naip-2018",
    provider: "naip-aerial",
    kind: "aerial-photo",
    title: "NAIP 2018 aerial photograph",
    dateLabel: "Flown 15 to 22 July 2018",
    dateMeaning: "Flight dates",
    sortDate: "2018-07-22",
    resolutionM: 0.6,
    mapScale: null,
    bounds: [-120.1, 34.6, -120.0, 34.75],
    tiles: {
      scheme: "web-mercator",
      urlTemplate: "https://planetarycomputer.microsoft.com/x/{z}/{x}/{y}",
      maximumLevel: 18,
      format: "png",
    },
    license: "U.S. public domain (USDA Farm Service Agency)",
    licenseUrl: "https://planetarycomputer.microsoft.com/dataset/naip",
    attribution: "USDA FSA NAIP",
    parts: [
      {
        observedOn: "2018-07-22",
        observedYear: 2018,
        footprint: square,
        recordUrl: "https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip/items/a",
        accuracy: {
          kind: "stated-by-source",
          valueM: 6,
          basis:
            "From 2009 to present all NAIP imagery acquisitions used the +/- 6 meters to ground specification.",
          citationUrl: null,
        },
      },
    ],
    warnings: [],
    ...over,
  };
}

const CATALOG: DatedImageryCatalog = {
  bounds: [-120.06, 34.68, -120.02, 34.72],
  retrievedAt: "2026-10-16T12:00:00Z",
  cached: false,
  sources: [
    source({
      id: "s2-2023",
      kind: "satellite",
      title: "Sentinel-2 2023 true colour",
      dateLabel: "Seen 29 August 2023",
      sortDate: "2023-08-29",
      resolutionM: 10,
    }),
    source({
      id: "naip-2022",
      title: "NAIP 2022 aerial photograph",
      dateLabel: "Flown 18 to 21 May 2022",
      sortDate: "2022-05-21",
      // Two quarter quads, each with its own record.
      parts: [
        { ...source({}).parts[0]!, observedOn: "2022-05-21", observedYear: 2022 },
        {
          ...source({}).parts[0]!,
          observedOn: "2022-05-18",
          observedYear: 2022,
          recordUrl: "https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip/items/b",
        },
      ],
    }),
    source({}),
    source({
      id: "usgs-1919",
      kind: "historical-map",
      title: "USGS 1:62,500 Los Olivos",
      dateLabel: "Map dated 1919",
      dateMeaning:
        "Catalog publication year; surveying, revision and scanning may have different dates.",
      sortDate: "1919-01-01",
      resolutionM: 5.29,
      mapScale: 62_500,
      license: "U.S. public domain",
      tiles: {
        scheme: "geographic",
        urlTemplate: "/api/v1/imagery/usgs-topo/tiles/{z}/{x}/{y}.png?sheet=x",
        maximumLevel: 17,
        format: "png",
      },
      parts: [
        {
          observedOn: null,
          observedYear: 1919,
          footprint: square,
          recordUrl: "https://www.sciencebase.gov/catalog/item/5eb2",
          accuracy: {
            kind: "not-stated",
            valueM: null,
            basis: "Predates the National Map Accuracy Standards (1947); no accuracy is stated.",
            citationUrl: null,
          },
        },
      ],
      datumShift: {
        fromDatum: "NAD27",
        operation: "NAD27 to WGS 84 (6)",
        accuracyM: 7,
        betterAvailable: true,
      },
      warnings: [
        "Shifted from NAD27 to WGS 84 with “NAD27 to WGS 84 (6)”, stated accuracy about 7 m. A more accurate grid-based shift exists but is not installed on this server.",
      ],
    }),
  ],
  providers: [
    {
      provider: "naip-aerial",
      name: "NAIP",
      status: "available",
      summary: "2 found, 2018 to 2022.",
    },
    {
      provider: "sentinel-2-true-colour",
      name: "Sentinel-2",
      status: "available",
      summary: "1 found, 2023.",
    },
    {
      provider: "usgs-historical-topo-sheets",
      name: "USGS maps",
      status: "unavailable",
      summary: "USGS historical topographic maps as map tiles could not be reached (HTTP 503).",
    },
  ],
};

/** The catalog as it reads when the two public tilers are down and USGS has nothing here. */
const OUTAGE: DatedImageryCatalog = {
  ...CATALOG,
  sources: [],
  providers: [
    { provider: "naip-aerial", name: "NAIP", status: "unavailable", summary: "NAIP: HTTP 503." },
    {
      provider: "sentinel-2-true-colour",
      name: "Sentinel-2",
      status: "unavailable",
      summary: "Sentinel-2: timed out.",
    },
    {
      provider: "usgs-historical-topo-sheets",
      name: "USGS maps",
      status: "empty",
      summary: "USGS maps: none found for this area.",
    },
  ],
};

function wrap(children: ReactNode, registry: SceneRegistry = new SceneRegistry()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return (
    <SceneContext.Provider value={registry}>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </SceneContext.Provider>
  );
}

/** A scene with only its events, for the strip to hear the lens from. */
function fakeScene() {
  const events = new Emitter<SceneEvents>();
  const registry = new SceneRegistry();
  registry.set({ events } as unknown as CesiumSceneManager);
  return { events, registry };
}

function frame(over: Partial<LanternFrame>): LanternFrame {
  return {
    mode: "sticky",
    index: 0,
    availability: { ok: true },
    lens: null,
    ground: null,
    outside: false,
    source: null,
    part: null,
    covered: false,
    ...over,
  };
}

beforeEach(() => {
  useViewer.getState().setCamera({ ...initialCamera, altitude: 900 });
  useLantern.setState({
    mode: "off",
    index: 0,
    kinds: { aerial: true, maps: true, satellite: false },
    bboxOverride: [-120.06, 34.68, -120.02, 34.72],
    autoBbox: null,
    availability: { ok: true },
    requested: false,
  });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("where it is offered", () => {
  it("is not shown from high up", () => {
    useViewer.getState().setCamera({ ...initialCamera, altitude: 30_000 });
    render(wrap(<TimeLanternControl />));
    expect(screen.queryByTestId("lantern-control")).not.toBeInTheDocument();
  });

  it("says why it cannot open in the open world", () => {
    useLantern.setState({ availability: { ok: false, reason: "open-world" } });
    render(wrap(<TimeLanternControl />));
    expect(screen.getByRole("button", { name: /Look back/ })).toBeDisabled();
    expect(screen.getByText("Needs the 3D world (Settings › World)")).toBeInTheDocument();
  });
});

describe("open", () => {
  beforeEach(() => {
    vi.spyOn(api, "GET").mockResolvedValue({ data: CATALOG, response: new Response() });
  });

  it("lists the records newest first, without satellite scenes until asked", async () => {
    render(wrap(<TimeLanternControl />));
    await userEvent.click(screen.getByRole("button", { name: /Look back/ }));
    expect(useLantern.getState().mode).toBe("sticky");
    const rail = await screen.findByRole("radiogroup", { name: "Dated imagery, newest first" });
    const radios = within(rail).getAllByRole("radio");
    expect(radios.map((r) => r.getAttribute("aria-label"))).toEqual([
      "Flown 18 to 21 May 2022, NAIP 2022 aerial photograph",
      "Flown 15 to 22 July 2018, NAIP 2018 aerial photograph",
      "Map dated 1919, USGS 1:62,500 Los Olivos",
    ]);
    expect(radios[0]).toHaveAttribute("aria-checked", "true");
    expect(within(rail).getByText("Map 1:62,500")).toBeInTheDocument();
    // The live line says what is shown, with its licence.
    expect(screen.getByTestId("lantern-live")).toHaveTextContent(
      "Flown 18 to 21 May 2022 · NAIP 2022 aerial photograph · U.S. public domain (USDA Farm Service Agency)",
    );
    expect(screen.getByText(/Hold Alt to peek/)).toBeInTheDocument();
  });

  it("filters by kind", async () => {
    render(wrap(<TimeLanternControl />));
    await userEvent.click(screen.getByRole("button", { name: /Look back/ }));
    await screen.findByTestId("lantern-rail");
    await userEvent.click(screen.getByRole("button", { name: "Satellite" }));
    await userEvent.click(screen.getByRole("button", { name: "Maps" }));
    const labels = within(screen.getByTestId("lantern-rail"))
      .getAllByRole("radio")
      .map((r) => r.getAttribute("aria-label"));
    expect(labels).toEqual([
      "Seen 29 August 2023, Sentinel-2 2023 true colour",
      "Flown 18 to 21 May 2022, NAIP 2022 aerial photograph",
      "Flown 15 to 22 July 2018, NAIP 2018 aerial photograph",
    ]);
  });

  it("lists the providers that could not answer, apart from those with nothing here", async () => {
    render(wrap(<TimeLanternControl />));
    await userEvent.click(screen.getByRole("button", { name: /Look back/ }));
    const providers = await screen.findByTestId("lantern-providers");
    expect(providers).toHaveTextContent("1 source could not be reached");
    expect(providers).not.toHaveTextContent("nothing");
    expect(providers).toHaveTextContent("could not be reached (HTTP 503)");
    expect(providersLine(OUTAGE.providers)).toBe(
      "2 sources could not be reached, 1 has nothing here",
    );
    expect(providersLine(OUTAGE.providers.slice(2))).toBe("1 source has nothing to show here");
  });

  it("never reads an outage as nothing found, and asks again", async () => {
    const get = vi.mocked(api.GET);
    get.mockResolvedValue({ data: OUTAGE, response: new Response() });
    render(wrap(<TimeLanternControl />));
    await userEvent.click(screen.getByRole("button", { name: /Look back/ }));
    const empty = await screen.findByTestId("lantern-empty");
    expect(empty).toHaveTextContent(
      "Some sources could not be reached, so this list may be incomplete.",
    );
    expect(empty).not.toHaveTextContent("No dated imagery found");
    expect(screen.getByTestId("lantern-providers")).toHaveTextContent(
      "2 sources could not be reached, 1 has nothing here",
    );
    const asked = get.mock.calls.length;
    get.mockResolvedValue({ data: CATALOG, response: new Response() });
    await userEvent.click(within(empty).getByRole("button", { name: "Try again" }));
    expect(get.mock.calls.length).toBe(asked + 1);
    await screen.findByTestId("lantern-rail");
  });

  it("offers to try again when the list could not be loaded", async () => {
    const get = vi.mocked(api.GET);
    get.mockRejectedValue(new Error("HTTP 502"));
    render(wrap(<TimeLanternControl />));
    await userEvent.click(screen.getByRole("button", { name: /Look back/ }));
    // After the one automatic retry.
    const alert = await screen.findByRole("alert", undefined, { timeout: 4000 });
    get.mockResolvedValue({ data: CATALOG, response: new Response() });
    expect(alert).toHaveTextContent("The dated imagery list could not be loaded (HTTP 502).");
    expect(alert).not.toHaveTextContent("Close the lantern");
    await userEvent.click(within(alert).getByRole("button", { name: "Try again" }));
    await screen.findByTestId("lantern-rail");
  });

  it("says what a record's date means and every caveat the catalog gives", async () => {
    render(wrap(<TimeLanternControl />));
    await userEvent.click(screen.getByRole("button", { name: /Look back/ }));
    await screen.findByTestId("lantern-rail");
    expect(screen.getByTestId("lantern-about")).toHaveTextContent("Flight dates");
    useLantern.getState().select(2);
    const about = await screen.findByRole("region", { name: "About this record" });
    await waitFor(() => expect(about).toHaveTextContent("Catalog publication year"));
    expect(about).toHaveTextContent(
      "Shifted from NAD27 to WGS 84 with “NAD27 to WGS 84 (6)”, stated accuracy about 7 m. A more accurate grid-based shift exists but is not installed on this server.",
    );
    // One record: its page. (Of a source's many, only the one under the lens is linked.)
    expect(within(about).getByRole("link", { name: "Open the record" })).toHaveAttribute(
      "href",
      "https://www.sciencebase.gov/catalog/item/5eb2",
    );
  });

  it("links the record under the lens, and says where to move when the lens is away", async () => {
    const { events, registry } = fakeScene();
    useLand.setState({ active: null });
    render(wrap(<TimeLanternControl />, registry));
    await userEvent.click(screen.getByRole("button", { name: /Look back/ }));
    await screen.findByTestId("lantern-rail");
    // NAIP 2022 has many quarter quads: none is linked until one is under the lens.
    expect(screen.queryByTestId("lantern-record")).not.toBeInTheDocument();
    act(() =>
      events.emit(
        "lantern",
        frame({
          part: {
            dateLabel: "21 May 2022",
            accuracyLabel: "x",
            recordUrl:
              "https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip/items/q",
          },
          covered: true,
        }),
      ),
    );
    expect(screen.getByTestId("lantern-record")).toHaveAttribute(
      "href",
      "https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip/items/q",
    );
    expect(screen.queryByTestId("lantern-outside")).not.toBeInTheDocument();
    act(() => events.emit("lantern", frame({ outside: true })));
    expect(screen.getByTestId("lantern-outside")).toHaveTextContent(
      "Dated imagery is listed for the place you chose. Move over it to look back.",
    );
  });

  it("steps older with the down arrow and ]", async () => {
    render(wrap(<TimeLanternControl />));
    await userEvent.click(screen.getByRole("button", { name: /Look back/ }));
    const rail = await screen.findByTestId("lantern-rail");
    within(rail).getAllByRole("radio")[0]?.focus();
    await userEvent.keyboard("{ArrowDown}");
    expect(useLantern.getState().index).toBe(1);
    await userEvent.keyboard("]");
    expect(useLantern.getState().index).toBe(2);
    await waitFor(() =>
      expect(within(rail).getAllByRole("radio")[2]).toHaveAttribute("aria-checked", "true"),
    );
    expect(screen.getByTestId("lantern-live")).toHaveTextContent("U.S. public domain");
    await userEvent.keyboard("[[");
    expect(useLantern.getState().index).toBe(1);
  });

  it("asks for the area chosen, rounded outward", async () => {
    useLantern.setState({ bboxOverride: [-120.0512, 34.6809, -120.0201, 34.7191] });
    const get = vi.mocked(api.GET);
    render(wrap(<TimeLanternControl />));
    await userEvent.click(screen.getByRole("button", { name: /Look back/ }));
    await screen.findByTestId("lantern-rail");
    expect(get).toHaveBeenCalledWith(
      "/api/v1/imagery/dated",
      expect.objectContaining({
        params: { query: { west: -120.06, south: 34.68, east: -120.02, north: 34.72 } },
      }),
    );
  });
});

describe("the words", () => {
  it("names the accuracy on whose word, and never claims it was checked", () => {
    const naip = toLanternSource(source({}));
    expect(naip.accuracyLabel).toBe(
      "stated ±6 m by its record · not checked against today's 3D world",
    );
    expect(
      accuracyLabel({ kind: "standard", valueM: 12.19, basis: "NMAS", citationUrl: null }),
    ).toBe("NMAS ±12 m (90%) if compliant · not checked against today's 3D world");
    expect(
      accuracyLabel({
        kind: "not-stated",
        valueM: null,
        basis: "Predates the National Map Accuracy Standards (1947); no accuracy is stated.",
        citationUrl: null,
      }),
    ).toBe(
      "Predates the National Map Accuracy Standards (1947); no accuracy is stated · not checked against today's 3D world",
    );
    expect(
      accuracyLabel({ kind: "measured-fit", valueM: 3.4, basis: "x", citationUrl: null }),
    ).toMatch(/^fit RMS 3.4 m/);
  });

  it("puts a map's datum shift beside its own accuracy, for the rim", () => {
    const map = toLanternSource(CATALOG.sources[3]!);
    expect(map.accuracyLabel).toBe(
      "Predates the National Map Accuracy Standards (1947); no accuracy is stated · datum shift ±7 m · not checked against today's 3D world",
    );
    expect(
      accuracyLabel(
        { kind: "standard", valueM: 12.19, basis: "NMAS", citationUrl: null },
        { fromDatum: "NAD27", operation: "x", accuracyM: null, betterAvailable: false },
      ),
    ).toBe(
      "NMAS ±12 m (90%) if compliant · datum shift accuracy not stated · not checked against today's 3D world",
    );
  });

  it("writes a day as the catalog does", () => {
    expect(formatDay("2022-05-18")).toBe("18 May 2022");
    expect(formatDay("2016-06-05T00:00:00Z")).toBe("5 June 2016");
    expect(formatDay("not a day")).toBe("not a day");
    expect(toLanternSource(source({})).parts[0]?.dateLabel).toBe("22 July 2018");
  });

  it("leaves a resolution nobody stated off the rail", async () => {
    vi.spyOn(api, "GET").mockResolvedValue({
      data: { ...CATALOG, sources: [source({ resolutionM: null })] },
      response: new Response(),
    });
    render(wrap(<TimeLanternControl />));
    await userEvent.click(screen.getByRole("button", { name: /Look back/ }));
    const rail = await screen.findByTestId("lantern-rail");
    expect(within(rail).getByRole("radio")).toHaveTextContent(/^2018Aerial$/);
  });

  it("makes the API's own tiles absolute", () => {
    const map = toLanternSource(CATALOG.sources[3]!);
    expect(map.tiles.urlTemplate).toMatch(/\/api\/v1\/imagery\/usgs-topo\/tiles\//);
  });

  it("rounds a saved land's area coarsely before it reaches the public catalog", () => {
    expect(roundBbox([-120.051, 34.681, -120.021, 34.719], 0.05)).toEqual([
      -120.1, 34.65, -120, 34.75,
    ]);
    // Larger than the route takes: the middle of it, at most 0.25° a side.
    const big = roundBbox([-121, 34, -120, 35]);
    expect(big[2] - big[0]).toBeLessThanOrEqual(0.25);
    expect(big[3] - big[1]).toBeLessThanOrEqual(0.25);
  });
});

describe("the area looked back on", () => {
  const land = {
    id: "land-1",
    name: "Oak Bench Ranch",
    boundary: square,
  } as unknown as LandArea;

  afterEach(() => useLand.setState({ active: null }));

  it("is the land selected after a descent, not the descent's area", () => {
    useLand.setState({ active: null });
    render(wrap(<LanternBridge />));
    expect(useLantern.getState().bboxOverride).not.toBeNull();
    act(() => useLand.setState({ active: land }));
    expect(useLantern.getState().bboxOverride).toBeNull();
  });

  it("is a descent's area chosen while a land is selected", () => {
    useLand.setState({ active: land });
    render(wrap(<LanternBridge />));
    act(() => useLantern.getState().setBbox([-120.1, 34.6, -120.08, 34.62]));
    expect(useLantern.getState().bboxOverride).toEqual([-120.1, 34.6, -120.08, 34.62]);
  });
});

describe("panning on around the view", () => {
  afterEach(() => useLand.setState({ active: null }));

  it("re-centres only when the view leaves the area, keeping the list, the layers and the year", async () => {
    useLand.setState({ active: null });
    useLantern.setState({ mode: "sticky", requested: true, bboxOverride: null, autoBbox: null });
    const centre = { longitude: -120.04, latitude: 34.7 };
    const setSources = vi.fn();
    const events = new Emitter<SceneEvents>();
    const registry = new SceneRegistry();
    registry.set({
      events,
      camera: { viewCenter: () => ({ ...centre }) },
      lantern: {
        available: { ok: true },
        setSources,
        setMode: vi.fn(),
        select: vi.fn(),
      },
    } as unknown as CesiumSceneManager);
    let answer: (value: unknown) => void = () => undefined;
    const get = vi
      .spyOn(api, "GET")
      .mockResolvedValueOnce({ data: CATALOG, response: new Response() })
      .mockImplementation(() => new Promise((resolve) => (answer = resolve)) as never);
    render(wrap(<LanternBridge />, registry));
    await waitFor(() => expect(setSources.mock.lastCall?.[0]).toHaveLength(3));
    const listed = setSources.mock.lastCall?.[1] as { west: number; east: number };
    // The operator steps back to an older year.
    act(() => useLantern.getState().select(1));
    const move = (east: number) =>
      act(() => {
        centre.longitude = -120.04 + east / (111_320 * Math.cos((34.7 * Math.PI) / 180));
        useViewer.getState().setCamera({ ...initialCamera, ...centre, altitude: 900 });
      });
    // 800 m on: still inside the area asked about, so no new catalog.
    move(800);
    expect(get).toHaveBeenCalledTimes(1);
    // 2 km on: re-centred, and while the next list loads the lens keeps this one and its area.
    move(2_000);
    await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
    expect(setSources.mock.lastCall?.[0]).toHaveLength(3);
    expect(setSources.mock.lastCall?.[1]).toEqual(listed);
    expect(setSources).not.toHaveBeenCalledWith([], expect.anything());
    expect(useLantern.getState().index).toBe(1);
    // The next list arrives: its own area, and the chosen year is the scene's to keep.
    act(() => answer({ data: CATALOG, response: new Response() }));
    await waitFor(() => expect(setSources.mock.lastCall?.[1]).not.toEqual(listed));
    expect(useLantern.getState().index).toBe(1);
  });
});
