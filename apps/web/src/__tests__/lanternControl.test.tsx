/**
 * The Time Lantern in the strip (features/lantern/TimeLanternControl.tsx) and the catalog it
 * reads (useDatedSources.ts): where it is offered, the rail newest first, the kind chips, the
 * providers that could not answer, the keys, and the words every source carries.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { DatedImageryCatalog, DatedImagerySource } from "@twin/contracts";

import { api } from "@/api/client";
import { TimeLanternControl } from "@/features/lantern/TimeLanternControl";
import { accuracyLabel, roundBbox, toLanternSource } from "@/features/lantern/useDatedSources";
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
    }),
    source({}),
    source({
      id: "usgs-1919",
      kind: "historical-map",
      title: "USGS 1:62,500 Los Olivos",
      dateLabel: "Map dated 1919",
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
      summary:
        "USGS historical topographic maps as map tiles could not be reached (HTTP 503). Try again in a few minutes.",
    },
  ],
};

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
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

  it("lists the providers that could not answer", async () => {
    render(wrap(<TimeLanternControl />));
    await userEvent.click(screen.getByRole("button", { name: /Look back/ }));
    const providers = await screen.findByTestId("lantern-providers");
    expect(providers).toHaveTextContent("1 source has nothing to show here");
    expect(providers).toHaveTextContent("could not be reached (HTTP 503)");
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

  it("makes the API's own tiles absolute and keeps a part's own date", () => {
    const map = toLanternSource(CATALOG.sources[3]!);
    expect(map.tiles.urlTemplate).toMatch(/\/api\/v1\/imagery\/usgs-topo\/tiles\//);
    expect(toLanternSource(source({})).parts[0]?.dateLabel).toMatch(/2018/);
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
