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
import { useLiving } from "@/state/living";
import { useSettings, type SplatRenderer } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useViewMode } from "@/state/viewMode";
import { useViewRule } from "@/state/viewRule";

import { A1, E1, E2, SITE_ID, twoEpochSite } from "./epochFixtures";

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

const ruleOn = (rule: "latest" | "data-age" = "latest") =>
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

  it("is disabled under another renderer, whose notice the views already show", async () => {
    show("playcanvas");
    expect(await screen.findByLabelText(VIEW_RULE_TEXT.label)).toBeDisabled();
    expect(screen.getByTestId("views-needs-cesium")).toBeInTheDocument();
  });
});
