/**
 * "At this point" in the inspector (features/inspector/PlaceFacts.tsx): what public data says
 * at the inspected spot, row by row as each source answers, each cited and labelled, and the
 * way from there into the Land tool (features/land/landEntry.ts).
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { LandArea, LandCreate, PlaceFact, PlaceFactKind } from "@twin/contracts";
import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { env } from "@/app/env";
import { InspectorPanel } from "@/features/inspector/InspectorPanel";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { useSelection, type Selection } from "@/state/selection";
import { useUi } from "@/state/ui";

vi.mock("@/cesium/SceneContext", () => ({ useScene: () => null }));

const LON = -120.72;
const LAT = 35.53;

function source(kind: PlaceFactKind, name = `${kind} source`): PlaceFact["source"] {
  return {
    provider: kind,
    name,
    url: `https://example.test/${kind}?x=${LON}`,
    documentationUrl: "https://example.test/docs",
    license: "U.S. public domain",
    attribution: "An agency",
    resolution: "Point",
  };
}

function fact(kind: PlaceFactKind, over: Partial<PlaceFact> = {}): PlaceFact {
  return {
    kind,
    status: "available",
    label: `${kind} answer`,
    detail: null,
    basis: "modelled-surface",
    vintage: null,
    caveat: `${kind} caveat`,
    longitude: LON,
    latitude: LAT,
    retrievedAt: "2026-10-16T18:00:00Z",
    cached: false,
    source: source(kind),
    data: {},
    ...over,
  };
}

const FACTS: Record<PlaceFactKind, PlaceFact> = {
  elevation: fact("elevation", {
    label: "236.8 m",
    detail: "From a 1 m elevation model",
    vintage: "DEM acquired 2019",
    caveat:
      "Interpolated from the USGS 3D Elevation Program model at this point; not a surveyed height.",
    source: source("elevation", "USGS elevation"),
  }),
  soil: fact("soil", {
    label: "Corducci and Typic Xerofluvents, 0 to 5 percent slopes",
    detail: "Corducci 50% · Typic Xerofluvents 30%",
    basis: "mapped-survey",
  }),
  "land-cover": fact("land-cover", {
    label: "Tree cover (class 10)",
    basis: "classified-imagery",
    vintage: "2021",
  }),
  "flood-zone": fact("flood-zone", {
    status: "empty",
    label: "No mapped zone",
    basis: "regulatory-map",
  }),
  ecoregion: fact("ecoregion", {
    status: "uncovered",
    label: "EPA ecoregions cover the United States only.",
    basis: "regional-framework",
    source: null,
  }),
};

function land(name: string, ring: [number, number][]): LandArea {
  return {
    id: `land-${name}`,
    name,
    description: "",
    revision: 1,
    areaM2: 1000,
    perimeterM: 100,
    createdAt: "2026-10-16",
    updatedAt: "2026-10-16",
    source: { method: "drawn", label: "Fixture" },
    boundary: { type: "Polygon", coordinates: [ring] },
  };
}
const AROUND = land("Oak Bench", [
  [-120.73, 35.52],
  [-120.71, 35.52],
  [-120.71, 35.54],
  [-120.73, 35.54],
  [-120.73, 35.52],
]);
const ELSEWHERE = land("Redmond yard", [
  [-122.14, 47.64],
  [-122.13, 47.64],
  [-122.13, 47.65],
  [-122.14, 47.64],
]);

interface Calls {
  facts: { kind: string; retry: unknown }[];
}

function serve({
  lands = [] as LandArea[],
  facts = FACTS,
  hold = new Set<PlaceFactKind>(),
} = {}): Calls {
  const calls: Calls = { facts: [] };
  vi.spyOn(api, "GET").mockImplementation(((path: string, init?: { params?: never }) => {
    const params = (init?.params ?? {}) as {
      path?: { fact?: PlaceFactKind };
      query?: { retry?: boolean };
    };
    if (path === "/api/v1/research/place/{fact}") {
      const kind = params.path?.fact ?? "elevation";
      calls.facts.push({ kind, retry: params.query?.retry });
      if (hold.has(kind)) return new Promise(() => undefined);
      return Promise.resolve({ data: facts[kind], response: new Response() });
    }
    if (path === "/api/v1/land") return Promise.resolve({ data: lands, response: new Response() });
    return Promise.resolve({ data: [], response: new Response() });
  }) as never);
  return calls;
}

function inspect(kind: Selection["kind"] = "ground") {
  useSelection.getState().setSelection({
    kind,
    title: kind === "site" ? "Back yard" : "Location",
    longitude: LON,
    latitude: LAT,
    height: 230,
    terrainHeight: 229,
    ...(kind === "site" ? { siteId: "site-1" } : {}),
    at: Date.now(),
  });
  useUi.getState().setInspectorOpen(true);
}

function mount(children: ReactNode = <InspectorPanel />) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>,
  );
}

const enabled = env.landExplorationEnabled;
beforeEach(() => {
  env.landExplorationEnabled = true;
  useUi.setState({ activePanel: null, measureMode: null });
});
afterEach(() => {
  cleanup();
  env.landExplorationEnabled = enabled;
  act(() => {
    useSelection.getState().setSelection(null);
    useUi.getState().setInspectorOpen(false);
    useLand.getState().clear();
    useLandContext.getState().clear();
  });
  vi.restoreAllMocks();
});

describe("At this point", () => {
  it("fills five cited rows as each source answers", async () => {
    serve({ hold: new Set(["soil"]) });
    inspect();
    mount();
    const section = await screen.findByTestId("inspector-place");
    expect(section).toHaveTextContent("At this point · looked up live, not saved");
    expect(
      within(section)
        .getAllByRole("term")
        .map((term) => term.textContent),
    ).toEqual(["Elevation", "Soil", "Land cover", "Flood zone", "Ecoregion"]);
    expect(screen.getByTestId("place-soil")).toHaveTextContent("checking USDA…");
    const elevation = screen.getByTestId("place-elevation");
    await waitFor(() => expect(elevation).toHaveTextContent("236.8 m"));
    expect(elevation).toHaveTextContent("Elevation model · DEM acquired 2019 · USGS elevation");
    expect(within(elevation).getByRole("link", { name: /USGS elevation/ })).toHaveAttribute(
      "href",
      `https://example.test/elevation?x=${LON}`,
    );
    expect(within(elevation).getByText("Limits")).toBeInTheDocument();
    expect(elevation).toHaveTextContent("not a surveyed height");
    expect(screen.getByTestId("place-land-cover")).toHaveTextContent(
      "Tree cover (class 10)Satellite classification · 2021",
    );
    expect(screen.getByTestId("place-flood-zone")).toHaveTextContent("No mapped zone");
    expect(screen.getByTestId("place-ecoregion")).toHaveTextContent(
      "EPA ecoregions cover the United States only.",
    );
    expect(screen.getByTestId("place-soil")).toHaveTextContent("checking USDA…");
  });

  it("says why a row failed and retries past the server's cached failure", async () => {
    const calls = serve({
      facts: {
        ...FACTS,
        "flood-zone": fact("flood-zone", {
          status: "unavailable",
          label: "FEMA mapped flood zones could not be retrieved (HTTP 503)",
          source: null,
        }),
      },
    });
    inspect();
    mount();
    const row = await screen.findByTestId("place-flood-zone");
    await waitFor(() =>
      expect(row).toHaveTextContent("FEMA mapped flood zones could not be retrieved (HTTP 503)."),
    );
    expect(row).not.toHaveTextContent("No mapped zone");
    await userEvent.click(within(row).getByRole("button", { name: "Retry flood zone" }));
    await waitFor(() =>
      expect(calls.facts.filter((call) => call.kind === "flood-zone")).toEqual([
        { kind: "flood-zone", retry: false },
        { kind: "flood-zone", retry: true },
      ]),
    );
  });

  it("is not shown for a site, or without land exploration", async () => {
    serve();
    inspect("site");
    mount();
    expect(await screen.findByTestId("inspector-position")).toBeInTheDocument();
    expect(screen.queryByTestId("inspector-place")).not.toBeInTheDocument();
    cleanup();
    env.landExplorationEnabled = false;
    inspect();
    mount();
    expect(await screen.findByTestId("inspector-position")).toBeInTheDocument();
    expect(screen.queryByTestId("inspector-place")).not.toBeInTheDocument();
  });
});

describe("into the Land tool", () => {
  it("outlines the land around a point outside every saved land, never revising one", async () => {
    serve({ lands: [ELSEWHERE] });
    act(() => useLand.getState().select(ELSEWHERE));
    inspect();
    mount();
    await userEvent.click(await screen.findByRole("button", { name: "Outline this land" }));
    expect(useUi.getState().activePanel).toBe("land");
    const state = useLand.getState();
    expect(state.mode).toBe("candidates");
    expect(state.points).toEqual([[LON, LAT]]);
    // A saved land left active would make Save revise it with the new outline.
    expect(state.active).toBeNull();
  });

  it("opens the saved land the point is in, on Discover", async () => {
    serve({ lands: [ELSEWHERE, AROUND] });
    act(() => useLandContext.getState().setSection("actions"));
    inspect();
    mount();
    expect(await screen.findByText("Oak Bench", { selector: "strong" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Research Oak Bench" }));
    expect(useLand.getState().active?.id).toBe(AROUND.id);
    expect(useLandContext.getState().section).toBe("discover");
    expect(useUi.getState().activePanel).toBe("land");
  });

  it("returns to an unsaved outline instead of replacing it", async () => {
    serve({ lands: [AROUND] });
    const draft: LandCreate = {
      name: "Half drawn",
      description: "",
      boundary: AROUND.boundary,
      source: { method: "drawn", label: "Drawn on the map" },
    };
    act(() => useLand.getState().propose(draft));
    inspect();
    mount();
    await userEvent.click(
      await screen.findByRole("button", { name: "Return to your unsaved outline" }),
    );
    expect(useLand.getState().draft).toEqual(draft);
    expect(useLand.getState().mode).toBe("browse");
    expect(useUi.getState().activePanel).toBe("land");
  });
});
