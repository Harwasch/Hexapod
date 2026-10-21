/**
 * Compare › Dates (features/compare/CompareDates.tsx): two of the active site's dates, one each
 * side of the divider, asked of the scene rather than kept by the control; under another splat
 * renderer disabled, with the reason and the one-click switch; and the divider drawn while the
 * scene compares two dates (CompareSplit).
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Site } from "@twin/contracts";
import { GlassTooltipProvider } from "@twin/ui";

import { queryKeys } from "@/api/queries";
import type { CompareDates as CompareDatesPair } from "@/lib/epochs";
import { CompareSplit } from "@/features/compare/Compare";
import { CompareDates } from "@/features/compare/CompareDates";
import { clearHistory, undo } from "@/state/history";
import { useRendererOverride, useSettings, type SplatRenderer } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useViewRule } from "@/state/viewRule";

import { A1, E1, E2, SITE_ID, epoch, splat, twoEpochSite } from "./epochFixtures";

const scene = vi.hoisted(() => ({
  sites: { setCompareDates: vi.fn((_dates: CompareDatesPair | null) => Promise.resolve()) },
  layers: { setSplit: vi.fn(), setSplitPosition: vi.fn(), setVisible: vi.fn() },
}));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));

function show(site: Site = twoEpochSite(), renderer: SplatRenderer = "cesium") {
  useSettings.getState().set({ splatRenderer: renderer });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(queryKeys.site(SITE_ID), site);
  return render(
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>
        <CompareDates siteId={SITE_ID} />
        <CompareSplit />
      </GlassTooltipProvider>
    </QueryClientProvider>,
  );
}

const BOTH = { left: E1.id, right: E2.id };
const toggle = () => screen.getByRole("switch", { name: "Swipe between dates" });

beforeEach(() => {
  clearHistory();
  scene.sites.setCompareDates.mockClear();
  scene.layers.setSplitPosition.mockClear();
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
  useSites.setState({ activeSiteId: null, representation: {}, epoch: {}, compareDates: null });
});

describe("Compare › Dates", () => {
  it("says that swiping ends a per-cell view rule, while one is on", () => {
    useViewRule.setState({ active: { siteId: SITE_ID, rule: "latest" } });
    show();
    expect(screen.getByTestId("compare-dates-ends-rule")).toHaveTextContent(
      "Swiping ends the per-cell dates (Splat view › Per cell): the two are one or the other.",
    );
    act(() => useViewRule.setState({ active: null }));
    expect(screen.queryByTestId("compare-dates-ends-rule")).toBeNull();
  });

  it("shows nothing for a site with fewer than two dates of the model shown", () => {
    show(twoEpochSite({ epochs: [E1], assets: [A1] }));
    expect(screen.queryByTestId("compare-dates")).toBeNull();
  });

  it("offers the latest against the one before, and asks the scene for the two", async () => {
    show();
    const left = screen.getByLabelText("Left of the divider");
    const right = screen.getByLabelText("Right of the divider");
    expect(left).toHaveValue(E1.id);
    expect(right).toHaveValue(E2.id);
    expect(left).toHaveTextContent("Mar 1, 2026 · Drone");
    // The other side's date cannot be picked for this one.
    expect(screen.getAllByRole("option", { name: /Jun 1, 2026/ })[0]).toBeDisabled();
    expect(screen.queryByTestId("split-slider")).toBeNull();
    await userEvent.click(toggle());
    expect(scene.sites.setCompareDates).toHaveBeenCalledWith(BOTH);
    // The switch follows the scene, not its own click.
    expect(toggle()).toHaveAttribute("aria-checked", "false");
    act(() => useSites.getState().setCompareDates(SITE_ID, BOTH));
    expect(toggle()).toHaveAttribute("aria-checked", "true");
    // The divider is drawn, and it sets where the scene splits.
    expect(screen.getByTestId("split-slider")).toBeVisible();
    expect(scene.layers.setSplitPosition).toHaveBeenCalledWith(0.5);
    await userEvent.click(toggle());
    expect(scene.sites.setCompareDates).toHaveBeenLastCalledWith(null);
    act(() => useSites.getState().setCompareDates(SITE_ID, null));
    expect(screen.queryByTestId("split-slider")).toBeNull();
    // Undo turns it back on; the history is the active site's.
    undo();
    expect(scene.sites.setCompareDates).toHaveBeenLastCalledWith(BOTH);
  });

  it("a date changed while comparing asks for the new pair at once", async () => {
    const E3 = epoch("e3", {
      ordinal: 3,
      observedAt: "2026-09-01T15:00:00Z",
      currentAssetId: "a3",
    });
    const A3 = splat("a3", { epochId: "e3", createdAt: "2026-09-02T00:00:00Z" });
    show(
      twoEpochSite({ epochs: [E1, E2, E3], assets: [A1, ...twoEpochSite().assets.slice(1), A3] }),
    );
    act(() => useSites.getState().setCompareDates(SITE_ID, { left: E2.id, right: E3.id }));
    await userEvent.selectOptions(screen.getByLabelText("Left of the divider"), E1.id);
    expect(scene.sites.setCompareDates).toHaveBeenLastCalledWith({ left: E1.id, right: E3.id });
  });

  it("under PlayCanvas, the default renderer, is offered as under CesiumJS, with no notice", () => {
    show(twoEpochSite(), "playcanvas");
    expect(toggle()).toBeEnabled();
    expect(screen.getByLabelText("Left of the divider")).toBeEnabled();
    expect(screen.queryByTestId("compare-dates-needs-renderer")).toBeNull();
  });

  it("under Spark: disabled, says it needs CesiumJS or PlayCanvas, and switches in one click", async () => {
    show(twoEpochSite(), "spark");
    expect(toggle()).toBeDisabled();
    expect(screen.getByLabelText("Left of the divider")).toBeDisabled();
    expect(screen.getByTestId("compare-dates-needs-renderer")).toHaveTextContent(
      "Needs the CesiumJS or PlayCanvas renderer (Settings › Advanced › Splat renderer)",
    );
    act(() => useRendererOverride.setState({ renderer: "spark" }));
    await userEvent.click(screen.getByRole("button", { name: "Draw with PlayCanvas" }));
    expect(useSettings.getState().splatRenderer).toBe("playcanvas");
    expect(useRendererOverride.getState().renderer).toBeNull();
    await waitFor(() => expect(screen.queryByTestId("compare-dates-needs-renderer")).toBeNull());
    expect(toggle()).toBeEnabled();
  });

  it("under the WebGPU trial: disabled, with the same notice", () => {
    show(twoEpochSite(), "playcanvas-webgpu");
    expect(toggle()).toBeDisabled();
    expect(screen.getByTestId("compare-dates-needs-renderer")).toBeInTheDocument();
  });

  it("undo skips turning it on once a date choice has ended it, rather than doing nothing", async () => {
    // The scene, as it answers: the comparison is what it says (SiteManager.setCompareDates).
    scene.sites.setCompareDates.mockImplementation((dates) => {
      useSites.getState().setCompareDates(SITE_ID, dates);
      return Promise.resolve();
    });
    show();
    await userEvent.click(toggle());
    expect(useSites.getState().compareDates).toMatchObject(BOTH);
    // A date chosen in the capture-date control: the scene ends the comparison itself.
    act(() => useSites.getState().setCompareDates(SITE_ID, null));
    scene.sites.setCompareDates.mockClear();
    // Nothing of it is left to take back: the step is dropped, not replayed as a no-op.
    expect(undo()).toBeNull();
    expect(scene.sites.setCompareDates).not.toHaveBeenCalled();
    scene.sites.setCompareDates.mockImplementation(() => Promise.resolve());
  });

  it("another site's comparison is not this one's", () => {
    show();
    act(() => useSites.getState().setCompareDates("another-site", BOTH));
    expect(toggle()).toHaveAttribute("aria-checked", "false");
    // Ending this site's says nothing of the other's.
    act(() => useSites.getState().setCompareDates(SITE_ID, null));
    expect(useSites.getState().compareDates?.siteId).toBe("another-site");
  });
});
