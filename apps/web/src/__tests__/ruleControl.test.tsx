/**
 * Splat view › Per cell (features/views/RuleControl.tsx in ViewsControl): offered for a site
 * with two dates, the choice handed to the scene and the select showing what the scene says it
 * draws; a date's own analytic views held while a rule is on; the key to a rule (cells per
 * date, a grid-less date's tile bounds flagged, inferred cells); disabled while a Living
 * Survey rig moves the scan; and what happens with Compare › Dates, and why a rule ended, said.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Site } from "@twin/contracts";
import { GlassTooltipProvider } from "@twin/ui";

import { queryKeys } from "@/api/queries";
import { ViewsSlot } from "@/features/views/ViewsSlot";
import { VIEW_RULE_ENDED, VIEW_RULE_TEXT } from "@/features/views/ruleText";
import { useAsCaptured } from "@/state/asCaptured";
import { useLiving } from "@/state/living";
import { useSettings, type SplatRenderer } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useViewMode } from "@/state/viewMode";
import { useViewRule } from "@/state/viewRule";

import { A1, datedSite, E1, E2, SITE_ID, twoEpochSite } from "./epochFixtures";

const scene = vi.hoisted(() => ({ sites: { setViewRule: vi.fn(() => Promise.resolve()) } }));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));

function show(renderer: SplatRenderer = "cesium", site: Site = twoEpochSite()) {
  useSettings.getState().set({ splatRenderer: renderer });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(queryKeys.site(SITE_ID), site);
  return render(
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>
        <ViewsSlot />
      </GlassTooltipProvider>
    </QueryClientProvider>,
  );
}

const ruleOn = (rule: "latest" | "data-age" | "ground-truth" = "latest") =>
  act(() => useViewRule.setState({ active: { siteId: SITE_ID, rule }, ended: null }));

beforeEach(() => {
  scene.sites.setViewRule.mockClear();
  useSites.setState({
    activeSiteId: SITE_ID,
    representation: { [SITE_ID]: "gaussian-splat" },
    epoch: {},
    compareDates: null,
  });
});

afterEach(() => {
  cleanup();
  useAsCaptured.getState().set(false);
  useViewMode.setState({ mode: "photo", cells: {} });
  useViewRule.setState({ active: null, ended: null, legend: null });
  useLiving.getState().reset();
  useSettings.getState().reset();
});

describe("Per cell", () => {
  it("is offered for two dates, off by default, and asks the scene for a rule", async () => {
    show();
    const select = await screen.findByLabelText(VIEW_RULE_TEXT.label);
    expect(select).toHaveValue("");
    expect(Array.from((select as HTMLSelectElement).options).map((o) => o.text)).toEqual([
      "One date",
      "Latest",
      "Ground truth",
      "Highlight invented",
      "Data age",
    ]);
    await userEvent.selectOptions(select, "ground-truth");
    expect(scene.sites.setViewRule).toHaveBeenCalledWith("ground-truth");
    // The select says what the scene draws, not the click.
    expect(select).toHaveValue("");
    ruleOn();
    expect(select).toHaveValue("latest");
    await userEvent.selectOptions(select, "");
    expect(scene.sites.setViewRule).toHaveBeenLastCalledWith(null);
  });

  it("is not offered for one date", async () => {
    show("cesium", twoEpochSite({ epochs: [E1], assets: [A1] }));
    await screen.findByTestId("views");
    expect(screen.queryByTestId("views-rule")).toBeNull();
  });

  it("holds a date's own analytic views while a rule is on, and keys the rule", async () => {
    useViewMode.getState().set("confidence");
    show();
    await screen.findByTestId("views");
    ruleOn();
    expect(screen.getByRole("radio", { name: "Photo" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("radio", { name: "Confidence" })).toHaveAttribute("disabled");
    // Said as it is: the views wait, and Photo is what comes back (the view chosen before is
    // not restored).
    expect(screen.getByTestId("views-rule-holds")).toHaveTextContent(
      "Per-cell dates are on: a date's own Confidence and Data age can be chosen again with One date",
    );
    act(() =>
      useViewRule.setState({
        legend: {
          cellM: 1,
          dates: [
            { epochId: E1.id, coverage: "grid", cells: 120, ageBucket: 3 },
            { epochId: E2.id, coverage: "tile-bounds", cells: 40, ageBucket: 2 },
          ],
          inferredCells: 6,
          left: 0,
          leftOut: [],
        },
      }),
    );
    const rows = screen.getAllByTestId("rule-legend-date").map((r) => r.textContent);
    expect(rows).toEqual([
      "Mar 1, 2026: 120 cells",
      "Jun 1, 2026: 40 cells (tile bounds, overstates what it saw)",
    ]);
    expect(screen.getByTestId("rule-legend-inferred")).toHaveTextContent(
      "Inferred: 6 cells no date saw",
    );
    expect(screen.getByTestId("rule-legend")).toHaveTextContent("1 m cells");
    // Data age: each date's age besides.
    ruleOn("data-age");
    expect(screen.getAllByTestId("rule-legend-date")[1]).toHaveTextContent(
      "Jun 1, 2026: 40 cells (tile bounds, overstates what it saw) · 3–6 months",
    );
  });

  it("says a date dated only by its upload has no known capture time, and shows it no age", async () => {
    const upload = { ...E2, observedAtSource: "created" } as typeof E2;
    show("cesium", twoEpochSite({ epochs: [E1, upload] }));
    await screen.findByTestId("views");
    ruleOn("data-age");
    act(() =>
      useViewRule.setState({
        legend: {
          cellM: 1,
          dates: [
            { epochId: E1.id, coverage: "grid", cells: 120, ageBucket: 3 },
            { epochId: E2.id, coverage: "grid", cells: 4, ageBucket: null },
          ],
          inferredCells: 0,
          left: 0,
          leftOut: [],
        },
      }),
    );
    expect(screen.getAllByTestId("rule-legend-date")[1]?.textContent).toBe(
      "Uploaded Jun 1, 2026, capture time unknown: 4 cells",
    );
  });

  it("names each date it leaves out and why, and says when it is the As of date", async () => {
    show();
    await screen.findByTestId("views");
    ruleOn();
    act(() =>
      useViewRule.setState({
        legend: {
          cellM: 1,
          dates: [{ epochId: E1.id, coverage: "grid", cells: 120, ageBucket: 3 }],
          inferredCells: 0,
          left: 0,
          leftOut: [{ epochId: E2.id, reason: "frame", asOf: true }],
        },
      }),
    );
    expect(screen.getByTestId("rule-legend-left-out")).toHaveTextContent(
      "Jun 1, 2026, the As of date, is not drawn: placed in its own frame, not the site's",
    );
    act(() =>
      useViewRule.setState({
        legend: {
          cellM: 1,
          dates: [{ epochId: E1.id, coverage: "grid", cells: 120, ageBucket: 3 }],
          inferredCells: 0,
          left: 0,
          leftOut: [{ epochId: E2.id, reason: "loading", asOf: false }],
        },
      }),
    );
    expect(screen.getByTestId("rule-legend-left-out")).toHaveTextContent(
      "Jun 1, 2026 is not drawn yet: still loading",
    );
  });

  it("says a rule is drawn as Ground truth while As captured is on, and keys it so", async () => {
    show();
    await screen.findByTestId("views");
    ruleOn("data-age");
    act(() =>
      useViewRule.setState({
        legend: {
          cellM: 1,
          dates: [{ epochId: E1.id, coverage: "grid", cells: 120, ageBucket: 3 }],
          inferredCells: 6,
          left: 0,
          leftOut: [],
        },
      }),
    );
    const row = () => screen.getAllByTestId("rule-legend-date")[0];
    expect(row()).toHaveTextContent("Mar 1, 2026: 120 cells ·");
    expect(screen.getByTestId("rule-legend-inferred")).toBeInTheDocument();
    expect(screen.queryByTestId("views-rule-as-captured")).toBeNull();
    act(() => useAsCaptured.getState().set(true));
    expect(screen.getByTestId("views-rule-as-captured")).toHaveTextContent(
      "Drawn as Ground truth while As captured is on",
    );
    // The key is Ground truth's: no age per date, no inferred cells.
    expect(row()).toHaveTextContent(/^Mar 1, 2026: 120 cells$/);
    expect(screen.queryByTestId("rule-legend-inferred")).toBeNull();
    // The rule chosen is still the viewer's, and comes back with the key.
    expect(screen.getByLabelText(VIEW_RULE_TEXT.label)).toHaveValue("data-age");
    act(() => useAsCaptured.getState().set(false));
    expect(screen.queryByTestId("views-rule-as-captured")).toBeNull();
    expect(row()).toHaveTextContent("Mar 1, 2026: 120 cells ·");
    // Ground truth chosen needs no note: it is drawn as chosen.
    ruleOn("ground-truth");
    act(() => useAsCaptured.getState().set(true));
    expect(screen.queryByTestId("views-rule-as-captured")).toBeNull();
  });

  it("is disabled, and says why, while a Living Survey rig moves the scan", async () => {
    act(() =>
      useLiving.setState({
        status: { ...useLiving.getState().status, sites: [{ siteId: SITE_ID } as never] },
      }),
    );
    show();
    expect(await screen.findByLabelText(VIEW_RULE_TEXT.label)).toBeDisabled();
    expect(screen.getByTestId("views-rule-living")).toHaveTextContent(VIEW_RULE_TEXT.living);
  });

  it("says that choosing one ends Compare › Dates, and why the last one ended", async () => {
    act(() => useSites.setState({ compareDates: { siteId: SITE_ID, left: E1.id, right: E2.id } }));
    show();
    await screen.findByTestId("views-rule");
    expect(screen.getByTestId("views-rule-compare")).toHaveTextContent(VIEW_RULE_TEXT.compare);
    act(() =>
      useViewRule.setState({ active: null, ended: { siteId: SITE_ID, reason: "compare" } }),
    );
    expect(screen.getByTestId("views-rule-ended")).toHaveTextContent(VIEW_RULE_ENDED.compare);
    // Said once: the note before gives way to why the rule ended.
    expect(screen.queryByTestId("views-rule-compare")).toBeNull();
    act(() => useViewRule.setState({ ended: { siteId: SITE_ID, reason: "one-model" } }));
    expect(screen.getByTestId("views-rule-ended")).toHaveTextContent(
      "Per-cell dates ended: one model was asked for by itself (a version, or Retry)",
    );
    act(() => useViewRule.setState({ ended: { siteId: SITE_ID, reason: "living" } }));
    expect(screen.getByTestId("views-rule-ended")).toHaveTextContent(VIEW_RULE_ENDED.living);
    // Another site's rule that ended says nothing here.
    act(() => useViewRule.setState({ ended: { siteId: "elsewhere", reason: "living" } }));
    expect(screen.queryByTestId("views-rule-ended")).toBeNull();
  });

  it("is disabled, and says why, for more than four dates up to As of", async () => {
    const { epochs, assets } = datedSite(5);
    show("cesium", twoEpochSite({ epochs, assets }));
    const select = await screen.findByLabelText(VIEW_RULE_TEXT.label);
    expect(select).toBeDisabled();
    expect(screen.getByTestId("views-rule-dates")).toHaveTextContent(VIEW_RULE_TEXT.tooMany);
    // As of the fourth: four dates, offered.
    act(() => useSites.setState({ epoch: { [SITE_ID]: "e4" } }));
    expect(select).toBeEnabled();
    expect(screen.queryByTestId("views-rule-dates")).toBeNull();
  });

  it("is offered under PlayCanvas, which draws it, with no notice", async () => {
    show("playcanvas");
    const select = await screen.findByLabelText(VIEW_RULE_TEXT.label);
    expect(select).toBeEnabled();
    expect(screen.queryByTestId("views-needs-cesium")).toBeNull();
    await userEvent.selectOptions(select, "latest");
    expect(scene.sites.setViewRule).toHaveBeenCalledWith("latest");
  });

  it("is disabled under a renderer that draws none, whose notice the views already show", async () => {
    for (const renderer of ["spark", "playcanvas-webgpu"] as const) {
      show(renderer);
      expect(await screen.findByLabelText(VIEW_RULE_TEXT.label)).toBeDisabled();
      expect(screen.getByTestId("views-needs-cesium")).toHaveTextContent("Per cell");
      cleanup();
    }
  });
});
