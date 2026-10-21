/**
 * The change drape in the app (features/change, Compare › Dates): offered once both chosen
 * dates measured a height surface, and otherwise says why; under any splat renderer, since it
 * is ground geometry; the ground drawn and keyed, vegetation apart behind its own switch; its
 * legend counts each class, marks low confidence, says it is derived and uncalibrated, the
 * least change each layer can show, and when a date is not survey-grade; a 401
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
import { useViewRule } from "@/state/viewRule";

import { A1, A2, E1, E2, SITE_ID, epoch, splat, twoEpochSite } from "./epochFixtures";

/** The layers made, each by its kind: show/clear/destroy say which layer they were on. */
const layer = vi.hoisted(() => ({
  show: vi.fn(),
  clear: vi.fn(),
  destroy: vi.fn(),
  made: [] as string[],
}));
vi.mock("@/cesium/changeDrape", () => ({
  ChangeDrapeLayer: class {
    kind: string;
    constructor(_scene: unknown, options?: { layer?: string }) {
      this.kind = options?.layer ?? "ground";
      layer.made.push(this.kind);
    }
    show = (map: unknown) => {
      layer.show(this.kind, map);
    };
    clear = () => {
      layer.clear(this.kind);
    };
    destroy = () => {
      layer.destroy(this.kind);
    };
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

const changeLayer = (name: "ground" | "vegetation") => ({
  layer: name,
  label: name === "ground" ? "Ground: the bare earth" : "Vegetation and objects",
  heights: name === "ground" ? "ground" : "dsm",
  minDetectableM: name === "ground" ? 0.489 : 1.089,
  motionM: name === "ground" ? 0 : 0.25,
  horizontalRadiusM: name === "ground" ? 1 : 2,
  driftReachM: name === "ground" ? 3.536 : 0,
  counts:
    name === "ground"
      ? { unchanged: 1084, raised: 0, lowered: 48, insufficient: 148 }
      : { unchanged: 1060, raised: 47, lowered: 24, insufficient: 149 },
  lowConfidence:
    name === "ground"
      ? { unchanged: 562, raised: 0, lowered: 0 }
      : { unchanged: 539, raised: 23, lowered: 0 },
  areaLevel: 2,
  areaCellM: 1,
  areas: [],
});

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
  uncertainty: {
    fromM: 0.03,
    toM: 5,
    fromVerdict: "survey",
    toVerdict: "coarse",
    fromFrameM: 0,
    toFrameM: 0.062,
  },
  ground: changeLayer("ground"),
  vegetation: changeLayer("vegetation"),
  appearance: {
    status: "not-computed",
    reason:
      "needs the new photos compared with the old epoch at fixed geometry; never splat pixels",
  },
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
  layer.made = [];
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
    expect(useChangeDrape.getState().request).toMatchObject({
      siteId: SITE_ID,
      from: E1.id,
      to: E2.id,
    });
    expect(changeSwitch()).toHaveAttribute("aria-checked", "true");
    await userEvent.click(changeSwitch());
    expect(useChangeDrape.getState().request).toBeNull();
    undo();
    expect(useChangeDrape.getState().request).toMatchObject({
      siteId: SITE_ID,
      from: E1.id,
      to: E2.id,
    });
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
    expect(useChangeDrape.getState().request).toMatchObject({
      siteId: SITE_ID,
      from: S2.id,
      to: S3.id,
    });
    await userEvent.selectOptions(screen.getByLabelText("Right of the divider"), S1.id);
    expect(useChangeDrape.getState().request).toMatchObject({
      siteId: SITE_ID,
      from: S1.id,
      to: S3.id,
    });
    await userEvent.selectOptions(screen.getByLabelText("Left of the divider"), E4.id);
    expect(useChangeDrape.getState().request).toBeNull();
    expect(changeSwitch()).toHaveAttribute("aria-checked", "false");
    expect(screen.getByTestId("compare-change-not-offered")).toHaveTextContent(
      "Not offered: Dec 1, 2026 has no height surface measured",
    );
  });
});

describe("its dates", () => {
  const S3 = epoch("e3", {
    ordinal: 3,
    observedAt: "2026-09-01T15:00:00Z",
    currentAssetId: "a3",
    surface,
  } as unknown as Partial<Epoch>);
  const A3 = splat("a3", { epochId: "e3", createdAt: "2026-09-02T00:00:00Z" });
  const three = () => twoEpochSite({ epochs: [S1, S2, S3], assets: [A1, A2, A3] });

  it("are the ones the panel shows when it opens again", async () => {
    const { unmount } = show(three());
    await userEvent.selectOptions(screen.getByLabelText("Left of the divider"), S1.id);
    await userEvent.click(changeSwitch());
    expect(useChangeDrape.getState().request).toMatchObject({
      siteId: SITE_ID,
      from: S1.id,
      to: S3.id,
    });
    // The panel closes (its choice goes with it) and opens again: it shows what is drawn.
    unmount();
    show(three());
    expect(screen.getByLabelText("Left of the divider")).toHaveValue(S1.id);
    expect(screen.getByLabelText("Right of the divider")).toHaveValue(S3.id);
    expect(changeSwitch()).toHaveAttribute("aria-checked", "true");
  });

  it("are the ones the panel shows after an undo brings it back", async () => {
    show(three());
    // The two latest: S2 left, S3 right. On, off, another date, then undo the off.
    await userEvent.click(changeSwitch());
    await userEvent.click(changeSwitch());
    await userEvent.selectOptions(screen.getByLabelText("Left of the divider"), S1.id);
    act(() => {
      undo();
    });
    expect(useChangeDrape.getState().request).toMatchObject({
      siteId: SITE_ID,
      from: S2.id,
      to: S3.id,
    });
    expect(screen.getByLabelText("Left of the divider")).toHaveValue(S2.id);
    expect(screen.getByLabelText("Right of the divider")).toHaveValue(S3.id);
  });

  it("keep the sides the viewer chose, newer on the left", async () => {
    const { unmount } = show(three());
    await userEvent.selectOptions(screen.getByLabelText("Right of the divider"), S1.id);
    // Left S2 (newer), right S1 (older): the map is S1 -> S2, the panel keeps the sides.
    await userEvent.click(changeSwitch());
    expect(useChangeDrape.getState().request).toMatchObject({
      siteId: SITE_ID,
      from: S1.id,
      to: S2.id,
    });
    expect(screen.getByLabelText("Left of the divider")).toHaveValue(S2.id);
    unmount();
    show(three());
    expect(screen.getByLabelText("Left of the divider")).toHaveValue(S2.id);
    expect(screen.getByLabelText("Right of the divider")).toHaveValue(S1.id);
  });
});

describe("beside the swipe and a per-cell rule", () => {
  it("stays on when the swipe goes on and off, and is offered while a rule is on", async () => {
    useViewRule.setState({ active: { siteId: SITE_ID, rule: "latest" } });
    show();
    expect(changeSwitch()).toBeEnabled();
    await userEvent.click(changeSwitch());
    act(() => useSites.getState().setCompareDates(SITE_ID, { left: E1.id, right: E2.id }));
    expect(useChangeDrape.getState().request).toMatchObject({
      siteId: SITE_ID,
      from: E1.id,
      to: E2.id,
    });
    act(() => useSites.getState().setCompareDates(SITE_ID, null));
    expect(useChangeDrape.getState().request).toMatchObject({
      siteId: SITE_ID,
      from: E1.id,
      to: E2.id,
    });
    expect(changeSwitch()).toHaveAttribute("aria-checked", "true");
    act(() => useViewRule.setState({ active: null }));
  });
});

describe("the drape and its legend", () => {
  it("draws the ground and keys it: counts, low confidence, derived, its floor, uncalibrated, verdicts", async () => {
    show();
    await userEvent.click(changeSwitch());
    const legend = await screen.findByTestId("change-drape");
    await waitFor(() => expect(layer.show).toHaveBeenCalledWith("ground", changeMap.ground));
    // The ground only: vegetation is not drawn until asked for.
    expect(layer.made).toEqual(["ground"]);
    expect(get).toHaveBeenCalledWith("/api/v1/sites/{site_id}/change", {
      params: { path: { site_id: SITE_ID }, query: { from: E1.id, to: E2.id } },
    });
    expect(screen.getByTestId("change-drape-label")).toHaveTextContent(
      "Derived from height surfaces — not observed · changes under ±0.49 m cannot be seen between these dates",
    );
    expect(legend).toHaveTextContent("Mar 1, 2026 → Jun 1, 2026");
    expect(screen.getByTestId("change-row-raised")).toHaveTextContent(/Raised.*0 cells/);
    expect(screen.getByTestId("change-row-lowered")).toHaveTextContent(/Lowered.*48 cells/);
    expect(screen.getByTestId("change-row-lowered")).not.toHaveTextContent(/low confidence/);
    expect(screen.getByTestId("change-row-insufficient")).toHaveTextContent(
      /Not comparable.*148 cells/,
    );
    expect(screen.getByTestId("change-row-unchanged")).toHaveTextContent(
      /Unchanged.*1,084 cells.*counted, not drawn/,
    );
    expect(legend).toHaveTextContent("the 3σ rule and the 5 m ground window");
    expect(screen.getByTestId("change-drape-placement")).toHaveTextContent(
      "Not survey-grade: Mar 1, 2026 survey (±0.03 m) · Jun 1, 2026 coarse (±5 m)",
    );
    expect(legend).toHaveTextContent(/Appearance-only change: not computed/);
    // Vegetation is neither keyed nor counted in the ground's legend.
    expect(screen.queryByTestId("change-vegetation")).toBeNull();
    expect(legend).not.toHaveTextContent("47 cells");
  });

  it("draws vegetation apart, behind its own switch, keyed and floored on its own", async () => {
    show();
    await userEvent.click(changeSwitch());
    await screen.findByTestId("change-drape");
    const vegetation = screen.getByRole("switch", { name: /^Vegetation/ });
    expect(vegetation).toHaveAttribute("aria-checked", "false");
    expect(screen.getByTestId("change-drape")).toHaveTextContent(
      "Vegetation: canopy height change, higher threshold, wind and growth not ground change",
    );
    await userEvent.click(vegetation);
    await waitFor(() =>
      expect(layer.show).toHaveBeenCalledWith("vegetation", changeMap.vegetation),
    );
    const section = screen.getByTestId("change-vegetation");
    expect(screen.getByTestId("change-vegetation-label")).toHaveTextContent(
      "changes under ±1.1 m cannot be seen between these dates",
    );
    expect(screen.getByTestId("change-vegetation-row-raised")).toHaveTextContent(
      /Raised.*47 cells.*23 low confidence, hatched/,
    );
    expect(section).toHaveTextContent("±0.25 m of wind sway per date");
    // The ground's own key is untouched by it.
    expect(screen.getByTestId("change-row-raised")).toHaveTextContent(/Raised.*0 cells/);
    await userEvent.click(vegetation);
    await waitFor(() => expect(layer.destroy).toHaveBeenCalledWith("vegetation"));
    expect(screen.queryByTestId("change-vegetation")).toBeNull();
    expect(layer.destroy).not.toHaveBeenCalledWith("ground");
  });

  it("says nothing of survey grade when both dates are survey-grade", async () => {
    get.mockResolvedValue(
      answer({
        ...changeMap,
        uncertainty: { ...changeMap.uncertainty, toM: 0.062, toVerdict: "survey" },
      }),
    );
    show();
    await userEvent.click(changeSwitch());
    await screen.findByTestId("change-drape-label");
    expect(screen.queryByTestId("change-drape-placement")).toBeNull();
  });

  it("ends from its own Hide, taking the drape off the ground", async () => {
    show();
    await userEvent.click(changeSwitch());
    await screen.findByTestId("change-drape");
    await userEvent.click(screen.getByRole("button", { name: "Hide change map" }));
    expect(useChangeDrape.getState().request).toBeNull();
    await waitFor(() => expect(screen.queryByTestId("change-drape")).toBeNull());
    expect(layer.destroy).toHaveBeenCalledWith("ground");
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
    await waitFor(() => expect(layer.show).toHaveBeenCalledWith("ground", changeMap.ground));
    expect(screen.queryByTestId("change-drape-error")).toBeNull();
    expect(screen.queryByTestId("compare-change-refused")).toBeNull();
  });

  it("asks again with a token entered while a request without it is still on its way", async () => {
    let answerFirst: (value: GetResult) => void = () => undefined;
    get.mockImplementationOnce(
      () => new Promise<GetResult>((resolve) => (answerFirst = resolve)) as never,
    );
    show();
    await userEvent.click(changeSwitch());
    await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
    get.mockResolvedValue(answer(changeMap));
    act(() => useSettings.getState().set({ writeToken: "pilot" }));
    // The first request, sent without the token, is refused only now.
    act(() => answerFirst(refuse(401)));
    await waitFor(() => expect(layer.show).toHaveBeenCalledWith("ground", changeMap.ground));
    expect(screen.queryByTestId("change-drape-error")).toBeNull();
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
    expect(layer.destroy).toHaveBeenCalledWith("ground");
  });

  it("renders nothing, and makes no layer, until asked", () => {
    show();
    expect(screen.queryByTestId("change-drape")).toBeNull();
    expect(layer.made).toEqual([]);
    expect(get).not.toHaveBeenCalled();
  });
});
