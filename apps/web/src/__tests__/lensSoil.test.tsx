import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SoilLens } from "@/features/land/surface/lenses/SoilLens";
import {
  drainageShort,
  rainSoak,
  slopeWords,
  soilHeadline,
  soilTexture,
  soilUse,
} from "@/features/land/surface/lenses/soil/soilWords";
import { anchorOf } from "@/features/land/surface/geometryAnchor";
import { useLandContext } from "@/state/landContext";
import { useLandMap } from "@/state/landMap";
import { useSettings } from "@/state/settings";

import {
  ELEVATION_POINT,
  EVIDENCE,
  FINDINGS,
  fullDetail,
  land,
  mockApi,
  SOIL_POINT,
  sourceEvents,
} from "./lensWaterSoilFixtures";

function mount() {
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <SoilLens land={land} />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  useLandContext.getState().clear();
  useLandMap.setState({ pins: {}, focus: null });
  useSettings.setState({ units: "metric" });
});

describe("Soil lens", () => {
  it("shares the one start button with the Water lens", async () => {
    const { post } = mockApi({ investigations: [] });
    mount();
    fireEvent.click(await screen.findByRole("button", { name: "Check water, soil and climate" }));
    await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
    expect(post.mock.calls[0]?.[0]).toBe("/api/v1/land/{land_id}/overview");
  });

  it("describes the soil at its one sample point in plain words and pins that point", async () => {
    mockApi({});
    const view = mount();
    const card = await screen.findByRole("region", { name: "Soil" });
    expect(
      await within(card).findByRole("heading", {
        name: "Silt loam near the middle of the land: drains slowly, best for pasture or hay.",
      }),
    ).toBeVisible();
    expect(within(card).getByRole("button", { name: "Soil: show on map" })).toHaveTextContent(
      "Hagerstown silt loam",
    );
    expect(within(card).getByRole("button", { name: "Drainage: show on map" })).toHaveTextContent(
      "Drains slowly",
    );
    expect(
      within(card).getByRole("button", { name: "When it rains: show on map" }),
    ).toHaveTextContent("Soaks in slowly; some runs off");
    expect(
      within(card).getByRole("button", { name: "Typical slope: show on map" }),
    ).toHaveTextContent("Gentle, about 5%");
    expect(
      within(card).getByText(/Sampled at one point, 57 m above sea level\. Soil can differ/),
    ).toBeVisible();
    expect(within(card).queryByText(/hydgrp|muname|mukey/)).toBeNull();
    expect(useLandMap.getState().pins.soil).toEqual([
      { id: "soil-sample", position: SOIL_POINT, label: "Sampled near here", tone: "accent" },
    ]);

    fireEvent.click(within(card).getByText("Details and sources"));
    const table = within(card).getByRole("region", { name: "Soil components at the sample point" });
    expect(within(table).getByText("Duffield")).toBeVisible();
    expect(within(table).getByText("85%")).toBeVisible();
    expect(within(card).getByText("Group D")).toBeVisible();
    expect(within(card).getByText(/rule of thumb/)).toBeVisible();
    const sources = within(card).getByRole("region", { name: "Sources" });
    expect(within(sources).getByText("Soil")).toBeVisible();
    expect(within(sources).getByText("Elevation")).toBeVisible();

    view.unmount();
    expect(useLandMap.getState().pins.soil).toBeUndefined();
  });

  it("reads the sample point from evidence beyond the first page", async () => {
    const detail = fullDetail();
    const { get } = mockApi({
      detail: { ...detail, evidence: detail.evidence.filter((item) => item.id !== "e-soil") },
      evidence: { "e-soil": EVIDENCE.soils },
    });
    mount();
    await screen.findByRole("heading", { name: /^Silt loam/ });
    await waitFor(() => expect(useLandMap.getState().pins.soil?.[0]?.position).toEqual(SOIL_POINT));
    expect(get).toHaveBeenCalledWith("/api/v1/research/evidence/{evidence_id}", {
      params: { path: { evidence_id: "e-soil" } },
    });
  });

  it("falls back to the elevation point, then the middle of the land, when no point is recorded", async () => {
    const detail = fullDetail();
    mockApi({
      detail: {
        ...detail,
        evidence: detail.evidence.filter((item) => item.id !== "e-soil"),
        findings: detail.findings.filter((item) => item.id !== FINDINGS.soils.id),
      },
    });
    mount();
    await screen.findByRole("heading", { name: /^Silt loam/ });
    expect(useLandMap.getState().pins.soil?.[0]?.position).toEqual(ELEVATION_POINT);
    cleanup();
    mockApi({
      detail: {
        ...detail,
        evidence: [],
        findings: [],
      },
    });
    mount();
    await screen.findByRole("heading", { name: /^Silt loam/ });
    expect(useLandMap.getState().pins.soil?.[0]).toMatchObject({
      position: anchorOf(land.boundary),
      label: "Sampled near here",
    });
  });

  it("says when soil maps do not cover the land", async () => {
    const detail = fullDetail();
    mockApi({
      detail: {
        ...detail,
        findings: detail.findings.filter((item) => item.id !== FINDINGS.soils.id),
        artifacts: detail.artifacts.filter((item) => item.id !== "a-soil"),
      },
      events: sourceEvents({ "usda-soils": "uncovered" }),
    });
    mount();
    expect(
      await screen.findByRole("heading", { name: "Soil maps don't cover this location." }),
    ).toBeVisible();
  });

  it("shows elevation in feet for imperial units", async () => {
    useSettings.setState({ units: "imperial" });
    mockApi({});
    mount();
    expect(await screen.findByText(/Sampled at one point, 188 ft above sea level/)).toBeVisible();
  });
});

describe("soil in plain words", () => {
  it("finds the texture in a map unit name", () => {
    expect(soilTexture("Hagerstown silt loam, 3 to 8 percent slopes")).toBe("silt loam");
    expect(soilTexture("Glenelg gravelly loam, 8 to 15 percent slopes")).toBe("gravelly loam");
    expect(soilTexture("Manor very channery silty clay loam")).toBe(
      "very channery silty clay loam",
    );
    expect(soilTexture("Glenelg-Manor complex, 15 to 25 percent slopes")).toBeNull();
  });

  it("reads drainage, hydrologic group and slope without codes", () => {
    expect(drainageShort("Well drained")).toBe("drains well");
    expect(drainageShort("Somewhat poorly drained")).toBe("drains slowly");
    expect(drainageShort("Very poorly drained")).toBe("stays waterlogged");
    expect(rainSoak("A")).toBe("Soaks in fast; little runs off");
    expect(rainSoak("D")).toBe("Mostly runs off; soaks in very slowly");
    expect(rainSoak("B/D")).toBe("Runs off while waterlogged; soaks in once drained");
    expect(slopeWords(1)).toBe("Nearly flat, about 1%");
    expect(slopeWords(22)).toBe("Steep, about 22%");
    expect(soilUse("Well drained", 3)).toBe("suits most crops");
    expect(soilUse("Well drained", 20)).toBe("best kept in grass or trees");
    expect(soilUse("Poorly drained", 2)).toBe("best for pasture");
  });

  it("writes one honest sentence for odd map units", () => {
    const component = (mapUnit: string) => ({
      mapUnit,
      component: "",
      percent: 100,
      drainage: null,
      hydrologicGroup: null,
      slopePercent: null,
    });
    expect(soilHeadline([component("Water")], "the middle")).toBe(
      "The sample point is mapped as open water.",
    );
    expect(soilHeadline([component("Clay loam, 0 to 2 percent slopes")], "the north")).toBe(
      "Clay loam in the north of the land.",
    );
  });
});
