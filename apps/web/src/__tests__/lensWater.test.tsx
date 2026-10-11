import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { WaterLens } from "@/features/land/surface/lenses/WaterLens";
import { useLens } from "@/features/land/surface/lenses";
import {
  coverage,
  floodHeadline,
  floodRisk,
  summarizeFlood,
} from "@/features/land/surface/lenses/water/flood";
import { rainSummary, rainWords } from "@/features/land/surface/lenses/water/rain";
import { readOverviewBundle } from "@/features/land/surface/lenses/overviewBundle";
import { useAgentBus } from "@/features/land/surface/agentBus";
import { useLandContext } from "@/state/landContext";
import { useLandMap } from "@/state/landMap";

import {
  fullDetail,
  investigation,
  land,
  mockApi,
  run,
  RUN_ID,
  SHADED_X,
  sourceEvents,
  WEST_HALF,
} from "./lensWaterSoilFixtures";

function mount() {
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <WaterLens land={land} />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  useLandContext.getState().clear();
  useLandMap.setState({ pins: {}, focus: null });
  useLens.getState().setLens("overview");
});

describe("Water lens", () => {
  it("offers one button that starts the overview bundle, and starts nothing on its own", async () => {
    const { post } = mockApi({ investigations: [] });
    mount();
    const button = await screen.findByRole("button", { name: "Check water, soil and climate" });
    expect(screen.getByText(/Look up flood zones, soil, rainfall and elevation/)).toBeVisible();
    expect(post).not.toHaveBeenCalled();
    fireEvent.click(button);
    await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
    expect(post.mock.calls[0]?.[0]).toBe("/api/v1/land/{land_id}/overview");
    expect(post.mock.calls[0]?.[1]).toMatchObject({ body: { boundaryRevision: land.revision } });
  });

  it("says where the flood zone is, draws it, pins each zone and explains rain and drainage", async () => {
    mockApi({});
    const view = mount();
    const card = await screen.findByRole("region", { name: "Water" });
    expect(
      await within(card).findByRole("heading", {
        name: /^About [\d.,]+ acres in the west are in a mapped flood zone\.$/,
      }),
    ).toBeVisible();
    // The two overlapping AE polygons count once: half the land, not all of it.
    const hazard = within(card).getByRole("button", { name: "Mapped flood zone: show on map" });
    expect(hazard).toHaveTextContent("1% yearly flood chance");
    expect(hazard).toHaveTextContent(/Includes a floodway/);
    expect(
      within(card).getByRole("button", { name: "Moderate flood risk: show on map" }),
    ).toHaveTextContent("0.2% yearly flood chance");
    expect(within(card).getByText(/About 830 mm a year · wettest Nov to Jan/)).toBeVisible();
    expect(
      within(card).getByRole("img", { name: /Rain by month, wettest November/ }),
    ).toBeVisible();
    expect(within(card).getByText("Drains slowly")).toBeVisible();
    expect(within(card).queryByText(/hydgrp|muname|SFHA/)).toBeNull();

    const layers = useLandContext.getState().layers;
    expect(layers["water-flood-zones"]?.features).toHaveLength(2);
    expect(layers["water-flood-moderate"]?.features).toHaveLength(1);
    expect(layers["water-flood-zones"]?.style?.fill).toBeGreaterThan(
      layers["water-flood-moderate"]?.style?.fill ?? 1,
    );
    const pins = useLandMap.getState().pins.water ?? [];
    expect(pins.map((pin) => pin.label)).toEqual([
      expect.stringMatching(/^Zone AE · [\d.]+ ac$/),
      expect.stringMatching(/^Zone X · /),
    ]);

    fireEvent.click(within(card).getByRole("button", { name: "See Soil" }));
    expect(useLens.getState().lens).toBe("soil");

    view.unmount();
    expect(useLandContext.getState().layers["water-flood-zones"]).toBeUndefined();
    expect(useLandMap.getState().pins.water).toBeUndefined();
  });

  it("keeps the FEMA table, sources and follow-up questions in Details", async () => {
    mockApi({});
    mount();
    const card = await screen.findByRole("region", { name: "Water" });
    fireEvent.click(await within(card).findByText("Details and sources"));
    const table = within(card).getByRole("region", { name: "Flood zones on this land" });
    expect(within(table).getAllByRole("button", { name: "Zone AE: show on map" })).toHaveLength(2);
    expect(within(table).getAllByText("11001C").length).toBeGreaterThan(0);
    expect(within(table).getByText("Floodway")).toBeVisible();
    expect(
      within(card).getByRole("region", { name: "Rain and temperature by month" }),
    ).toBeVisible();
    const sources = within(card).getByRole("region", { name: "Sources" });
    expect(within(sources).getByText("Flood zones")).toBeVisible();
    expect(within(sources).getAllByText("Found")).toHaveLength(3);
    fireEvent.click(
      within(card).getByRole("button", { name: "Question about FEMA mapped flood zones" }),
    );
    expect(useAgentBus.getState().request?.question).toBe("Question about FEMA mapped flood zones");
  });

  it("says plainly when FEMA maps no flood zone on the land", async () => {
    const detail = fullDetail();
    mockApi({
      detail: {
        ...detail,
        findings: detail.findings.filter((item) => item.title !== "FEMA mapped flood zones"),
        artifacts: detail.artifacts.filter((item) => !/flood/i.test(item.title)),
      },
      events: sourceEvents({ "fema-flood-zones": "empty" }),
    });
    mount();
    expect(
      await screen.findByRole("heading", { name: "No mapped flood zone on this land." }),
    ).toBeVisible();
    expect(screen.getByText("No flood zone mapped here")).toBeVisible();
    expect(useLandContext.getState().layers["water-flood-zones"]).toBeUndefined();
  });

  it("tells an unreachable flood service apart from no flood zone, and runs the check again", async () => {
    const detail = fullDetail({ runs: [run("partial")] });
    const { post } = mockApi({
      detail: {
        ...detail,
        findings: detail.findings.filter((item) => item.title !== "FEMA mapped flood zones"),
        artifacts: detail.artifacts.filter((item) => !/flood/i.test(item.title)),
      },
      events: sourceEvents({ "fema-flood-zones": "unavailable" }),
    });
    mount();
    expect(
      await screen.findByRole("heading", { name: "Flood zones couldn't be checked." }),
    ).toBeVisible();
    expect(screen.getByText("Some sources didn't respond: flood zones.")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
    expect(post.mock.calls[0]?.[0]).toBe("/api/v1/research/investigations/{investigation_id}/runs");
    expect(post.mock.calls[0]?.[1]).toMatchObject({
      params: { path: { investigation_id: investigation.id } },
      body: { kind: "overview" },
    });
  });

  it("shows progress while the check runs, and can stop it", async () => {
    const { post } = mockApi({
      detail: fullDetail({ runs: [run("running")], findings: [], artifacts: [] }),
      events: sourceEvents({}, ["usgs-elevation", "nasa-power"]),
    });
    mount();
    expect(await screen.findByText("Checking public maps · 2 of 5 done")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    await waitFor(() =>
      expect(post).toHaveBeenCalledWith("/api/v1/research/runs/{run_id}/cancel", {
        params: { path: { run_id: RUN_ID } },
      }),
    );
  });

  it("notes a result made for an earlier outline and runs it for the current one", async () => {
    const old = { ...investigation, boundaryRevision: 1 };
    const { post } = mockApi({
      investigations: [old],
      detail: fullDetail({ investigation: old }),
    });
    mount();
    expect(await screen.findByText("Checked for an earlier outline of this land.")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Run again" }));
    await waitFor(() =>
      expect(post).toHaveBeenCalledWith("/api/v1/land/{land_id}/overview", {
        params: { path: { land_id: land.id } },
        body: { boundaryRevision: land.revision },
      }),
    );
  });
});

describe("flood zones in plain words", () => {
  it("reads FEMA zones as high, coastal, moderate, minimal or not studied", () => {
    const risk = (zone: string, subtype = "", specialFloodHazard: boolean | null = null) =>
      floodRisk({ zone, subtype, specialFloodHazard });
    expect(risk("AE")).toBe("high");
    expect(risk("A99")).toBe("high");
    expect(risk("VE")).toBe("coastal");
    expect(risk("X", "0.2 PCT ANNUAL CHANCE FLOOD HAZARD")).toBe("moderate");
    expect(risk("X", "AREA WITH REDUCED FLOOD RISK DUE TO LEVEE")).toBe("moderate");
    expect(risk("X", "AREA OF MINIMAL FLOOD HAZARD")).toBe("minimal");
    expect(risk("D")).toBe("undetermined");
    expect(risk("AREA NOT INCLUDED")).toBe("other");
    expect(risk("OPEN WATER", "", true)).toBe("high");
  });

  it("counts overlapping polygons once against the land", () => {
    const once = coverage(land.boundary, [WEST_HALF]);
    const twice = coverage(land.boundary, [WEST_HALF, WEST_HALF, SHADED_X]);
    expect(once.fraction).toBeCloseTo(0.5, 2);
    // Half the land plus a strip a tenth of it: the repeated half adds nothing.
    expect(twice.fraction).toBeCloseTo(0.6, 2);
    const area = (geometry: typeof WEST_HALF, areaM2: number, id: string) => ({
      id,
      zone: "AE",
      subtype: "",
      specialFloodHazard: true,
      studyType: "",
      mapId: "",
      recordId: "",
      areaM2,
      geometry,
    });
    const summary = summarizeFlood(
      [area(WEST_HALF, land.areaM2 / 2, "a"), area(WEST_HALF, land.areaM2 / 2, "b")],
      land,
    );
    expect(summary.hazard?.share).toBeCloseTo(0.5, 2);
    expect(summary.hazard?.side).toBe("the west");
    expect(summary.zones).toHaveLength(1);
  });

  it("explains each FEMA status differently", () => {
    const source = (status: "empty" | "uncovered" | "unavailable" | null, message = "") => ({
      provider: "fema-flood-zones" as const,
      status,
      message,
      finding: null,
      evidence: null,
    });
    expect(floodHeadline(source("empty"), null, true)).toBe("No mapped flood zone on this land.");
    expect(
      floodHeadline(source("uncovered", "FEMA NFHL does not cover this location."), null, true),
    ).toBe("Flood maps don't cover this land.");
    expect(
      floodHeadline(
        source("uncovered", "Choose a smaller land area for bounded flood-zone analysis."),
        null,
        true,
      ),
    ).toBe("This land is too large for a flood-zone check.");
    expect(floodHeadline(source("unavailable"), null, true)).toBe(
      "Flood zones couldn't be checked.",
    );
    expect(floodHeadline(source(null), null, false)).toBeNull();
    const minimal = summarizeFlood(
      [
        {
          id: "m",
          zone: "X",
          subtype: "AREA OF MINIMAL FLOOD HAZARD",
          specialFloodHazard: false,
          studyType: "",
          mapId: "",
          recordId: "",
          areaM2: land.areaM2,
          geometry: land.boundary,
        },
      ],
      land,
    );
    expect(minimal.minimalOnly).toBe(true);
    expect(floodHeadline({ ...source(null), status: "available" as const }, minimal, true)).toBe(
      "No mapped flood zone on this land.",
    );
  });
});

describe("rain", () => {
  it("turns NASA POWER daily averages into a yearly total and the wettest season", () => {
    const detail = fullDetail();
    const bundle = readOverviewBundle(detail, RUN_ID, sourceEvents());
    const summary = rainSummary(bundle.climate?.precipitation ?? null);
    expect(summary?.annualMm).toBeCloseTo(825.6, 0);
    expect(summary?.wettest).toEqual({ start: 10, end: 0 });
    expect(summary && rainWords(summary, "imperial")).toBe(
      "About 33 in a year · wettest Nov to Jan",
    );
    expect(bundle.climate?.period).toBe("2001–2020");
    const even = rainSummary({
      title: "Precipitation",
      unit: "mm/day",
      values: Array.from({ length: 12 }, () => 2),
      annual: 2,
    });
    expect(even && rainWords(even, "metric")).toBe("About 730 mm a year · spread through the year");
  });
});
