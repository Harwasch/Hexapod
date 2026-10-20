/**
 * The change drape in the app (features/change, Compare › Dates): offered once both chosen
 * dates measured a height surface, and otherwise says why; under any splat renderer, since it
 * is ground geometry; its legend counts each class, marks low confidence, says it is derived
 * and uncalibrated, the least change it can show, and which date is not survey-grade; a 401
 * asks for the pilot's write token rather than leaving a blank drape; and it ends with its site.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ChangeMap, Epoch, Site } from "@twin/contracts";
import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { queryKeys } from "@/api/queries";
import { CompareDates } from "@/features/compare/CompareDates";
import { ChangeDrapeSlot } from "@/features/change/ChangeDrapeSlot";
import { NEEDS_TOKEN } from "@/lib/changeDrape";
import type { CompareDates as CompareDatesPair } from "@/lib/epochs";
import { useChangeDrape } from "@/state/changeDrape";
import { clearHistory, undo } from "@/state/history";
import { useSettings, type SplatRenderer } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";

import { A1, A2, E1, E2, SITE_ID, epoch, splat, twoEpochSite } from "./epochFixtures";

const layer = vi.hoisted(() => ({
  show: vi.fn(),
  clear: vi.fn(),
  destroy: vi.fn(),
  made: 0,
}));
vi.mock("@/cesium/changeDrape", () => ({
  ChangeDrapeLayer: class {
    constructor() {
      layer.made += 1;
    }
    show = layer.show;
    clear = layer.clear;
    destroy = layer.destroy;
  },
}));

const scene = vi.hoisted(() => ({
  scene: {},
  sites: { setCompareDates: vi.fn((_dates: CompareDatesPair | null) => Promise.resolve()) },
  layers: { setSplit: vi.fn(), setSplitPosition: vi.fn(), setVisible: vi.fn() },
}));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));

const surface = {
  status: "measured",
  assetId: "a",
  url: "https://tiles.example/surface.bin",
  cellM: 1,
  level: 2,
  policy: "surface-v1",
  calibration: "uncalibrated",
  cells: 1280,
  surfacedCells: 1272,
  medianSigmaM: 0.02,
  originHeightM: 0,
  groundWindowM: 5,
};

const S1 = { ...E1, surface } as unknown as Epoch;
const S2 = {
  ...E2,
  surface,
  alignment: { verdict: "coarse", uncertaintyM: 0.062 },
} as unknown as Epoch;
const surfaced = (extra: Partial<Site> = {}) => twoEpochSite({ epochs: [S1, S2], ...extra });

const changeMap = {
  policy: "change-v1",
  derived: true,
  label: "Derived from height surfaces, not observed",
  calibration: "uncalibrated",
  siteId: SITE_ID,
  fromEpochId: E1.id,
  toEpochId: E2.id,
  fromAssetId: A1.id,
  toAssetId: A2.id,
  frame: "tileset",
  level: 2,
  nativeLevel: 2,
  cellM: 1,
  thresholdSigma: 3,
  minDetectableM: 0.229,
  uncertainty: { fromM: 0, toM: 0.062 },
  counts: { unchanged: 1137, raised: 47, lowered: 24, insufficient: 72 },
  lowConfidence: { unchanged: 0, raised: 23, lowered: 0 },
  appearance: {
    status: "not-computed",
    reason:
      "needs the new photos compared with the old epoch at fixed geometry; never splat pixels",
  },
  areaLevel: 2,
  areaCellM: 1,
  areas: [],
} as unknown as ChangeMap;

interface GetResult {
  data?: unknown;
  error?: unknown;
  response: Response;
}
const get = vi.spyOn(api, "GET");
const answer = (data: unknown): GetResult => ({
  data,
  response: new Response(null, { status: 200 }),
});
const refuse = (status: number, code?: string): GetResult => ({
  error: { title: "Refused", status, code },
  response: new Response(null, { status }),
});

let client: QueryClient;

function show(site: Site = surfaced(), renderer: SplatRenderer = "cesium") {
  useSettings.getState().set({ splatRenderer: renderer });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(queryKeys.site(SITE_ID), site);
  return render(
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>
        <CompareDates siteId={SITE_ID} />
        <ChangeDrapeSlot />
      </GlassTooltipProvider>
    </QueryClientProvider>,
  );
}

const changeSwitch = () => screen.getByRole("switch", { name: "Change map" });

beforeEach(() => {
  clearHistory();
  layer.show.mockClear();
  layer.clear.mockClear();
  layer.destroy.mockClear();
  layer.made = 0;
  get.mockReset();
  get.mockResolvedValue(answer(changeMap));
  useSites.setState({
    activeSiteId: SITE_ID,
    representation: { [SITE_ID]: "gaussian-splat" },
    epoch: {},
    compareDates: null,
  });
});

afterEach(() => {
  cleanup();
  useSettings.getState().reset();
  useChangeDrape.setState({ request: null, error: null });
  useUi.setState({ settingsOpen: false });
  useSites.setState({ activeSiteId: null, representation: {}, epoch: {}, compareDates: null });
});

describe("the offer in Compare › Dates", () => {
  it("is made once both dates measured their heights, from the older to the newer", async () => {
    show();
    expect(changeSwitch()).toBeEnabled();
    expect(screen.queryByTestId("change-drape")).toBeNull();
    await userEvent.click(changeSwitch());
    expect(useChangeDrape.getState().request).toEqual({
      siteId: SITE_ID,
      from: E1.id,
      to: E2.id,
    });
    expect(changeSwitch()).toHaveAttribute("aria-checked", "true");
    await userEvent.click(changeSwitch());
    expect(useChangeDrape.getState().request).toBeNull();
    undo();
    expect(useChangeDrape.getState().request).toEqual({ siteId: SITE_ID, from: E1.id, to: E2.id });
  });

  it("says why it is not offered when a date has no height surface", () => {
    show(twoEpochSite({ epochs: [S1, E2] }));
    expect(changeSwitch()).toBeDisabled();
    expect(screen.getByTestId("compare-change-not-offered")).toHaveTextContent(
      "Not offered: Jun 1, 2026 has no height surface measured",
    );
  });

  it("is offered under every splat renderer: it is ground geometry, not splats", () => {
    show(surfaced(), "playcanvas");
    expect(screen.getByRole("switch", { name: "Swipe between dates" })).toBeDisabled();
    expect(changeSwitch()).toBeEnabled();
  });

  it("follows a date chosen while it is on, and ends with the reason if the new date has none", async () => {
    const S3 = epoch("e3", {
      ordinal: 3,
      observedAt: "2026-09-01T15:00:00Z",
      currentAssetId: "a3",
      surface,
    } as unknown as Partial<Epoch>);
    const E4 = epoch("e4", {
      ordinal: 4,
      observedAt: "2026-12-01T15:00:00Z",
      currentAssetId: "a4",
    });
    const A3 = splat("a3", { epochId: "e3", createdAt: "2026-09-02T00:00:00Z" });
    const A4 = splat("a4", { epochId: "e4", createdAt: "2026-12-02T00:00:00Z" });
    show(twoEpochSite({ epochs: [S1, S2, S3, E4], assets: [A1, A2, A3, A4] }));
    // The latest (no surface) against the one before: not offered, and why.
    expect(changeSwitch()).toBeDisabled();
    await userEvent.selectOptions(screen.getByLabelText("Right of the divider"), S2.id);
    // Newer on the left, older on the right: still from the older to the newer.
    expect(screen.getByLabelText("Left of the divider")).toHaveValue(S3.id);
    await userEvent.click(changeSwitch());
    expect(useChangeDrape.getState().request).toEqual({ siteId: SITE_ID, from: S2.id, to: S3.id });
    await userEvent.selectOptions(screen.getByLabelText("Right of the divider"), S1.id);
    expect(useChangeDrape.getState().request).toEqual({ siteId: SITE_ID, from: S1.id, to: S3.id });
    await userEvent.selectOptions(screen.getByLabelText("Left of the divider"), E4.id);
    expect(useChangeDrape.getState().request).toBeNull();
    expect(changeSwitch()).toHaveAttribute("aria-checked", "false");
    expect(screen.getByTestId("compare-change-not-offered")).toHaveTextContent(
      "Not offered: Dec 1, 2026 has no height surface measured",
    );
  });
});

describe("the drape and its legend", () => {
  it("draws the map and keys it: counts, low confidence, derived, the floor, uncalibrated, placement", async () => {
    show();
    await userEvent.click(changeSwitch());
    const legend = await screen.findByTestId("change-drape");
    await waitFor(() => expect(layer.show).toHaveBeenCalledWith(changeMap));
    expect(get).toHaveBeenCalledWith("/api/v1/sites/{site_id}/change", {
      params: { path: { site_id: SITE_ID }, query: { from: E1.id, to: E2.id } },
    });
    expect(screen.getByTestId("change-drape-label")).toHaveTextContent(
      "Derived from height surfaces — not observed · changes under ±0.23 m cannot be seen between these dates",
    );
    expect(legend).toHaveTextContent("Mar 1, 2026 → Jun 1, 2026");
    expect(screen.getByTestId("change-row-raised")).toHaveTextContent(
      /Raised.*47 cells.*23 low confidence, hatched/,
    );
    expect(screen.getByTestId("change-row-lowered")).toHaveTextContent(/Lowered.*24 cells/);
    expect(screen.getByTestId("change-row-lowered")).not.toHaveTextContent(/low confidence/);
    expect(screen.getByTestId("change-row-insufficient")).toHaveTextContent(
      /Not comparable.*72 cells/,
    );
    expect(screen.getByTestId("change-row-unchanged")).toHaveTextContent(
      /Unchanged.*1,137 cells.*counted, not drawn/,
    );
    expect(legend).toHaveTextContent(/uncalibrated/i);
    expect(legend).toHaveTextContent("the 3σ rule and the 5 m ground window");
    expect(screen.getByTestId("change-drape-placement")).toHaveTextContent(
      "Jun 1, 2026: Phone video, aligned visually, ±0.062 m — not survey-grade",
    );
    expect(legend).toHaveTextContent(/Appearance-only change: not computed/);
  });

  it("ends from its own Hide, taking the drape off the ground", async () => {
    show();
    await userEvent.click(changeSwitch());
    await screen.findByTestId("change-drape");
    await userEvent.click(screen.getByRole("button", { name: "Hide change map" }));
    expect(useChangeDrape.getState().request).toBeNull();
    await waitFor(() => expect(screen.queryByTestId("change-drape")).toBeNull());
    expect(layer.destroy).toHaveBeenCalled();
  });

  it("asks for the pilot's write token on a 401, in the legend and in Compare, and loads once given", async () => {
    get.mockResolvedValue(refuse(401));
    show();
    await userEvent.click(changeSwitch());
    expect(await screen.findByTestId("change-drape-error")).toHaveTextContent(NEEDS_TOKEN);
    expect(screen.getByTestId("compare-change-refused")).toHaveTextContent(NEEDS_TOKEN);
    expect(layer.show).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Open Settings" }));
    expect(useUi.getState().settingsOpen).toBe(true);
    get.mockResolvedValue(answer(changeMap));
    act(() => useSettings.getState().set({ writeToken: "pilot" }));
    await waitFor(() => expect(layer.show).toHaveBeenCalledWith(changeMap));
    expect(screen.queryByTestId("change-drape-error")).toBeNull();
    expect(screen.queryByTestId("compare-change-refused")).toBeNull();
  });

  it("says a refusal in words rather than drawing a blank drape", async () => {
    get.mockResolvedValue(refuse(409, "confidence-not-measured"));
    show();
    await userEvent.click(changeSwitch());
    expect(await screen.findByTestId("change-drape-error")).toHaveTextContent(
      /no confidence map, so how far to trust its heights is unknown/,
    );
    expect(layer.show).not.toHaveBeenCalled();
  });

  it("ends when another site becomes the active one", async () => {
    show();
    await userEvent.click(changeSwitch());
    await screen.findByTestId("change-drape");
    act(() => useSites.setState({ activeSiteId: "elsewhere" }));
    await waitFor(() => expect(useChangeDrape.getState().request).toBeNull());
    expect(screen.queryByTestId("change-drape")).toBeNull();
    expect(layer.destroy).toHaveBeenCalled();
  });

  it("renders nothing, and makes no layer, until asked", () => {
    show();
    expect(screen.queryByTestId("change-drape")).toBeNull();
    expect(layer.made).toBe(0);
    expect(get).not.toHaveBeenCalled();
  });
});
