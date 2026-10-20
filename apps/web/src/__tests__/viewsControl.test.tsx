/**
 * The splat view control (features/views): a slot that renders only for a splat site; under
 * CesiumJS the three views and the key to what is drawn; under PlayCanvas and Spark the
 * analytic views disabled, the reason, and a switch that works.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";

import { GlassTooltipProvider } from "@twin/ui";

import { queryKeys } from "@/api/queries";
import { ViewsSlot } from "@/features/views/ViewsSlot";
import { useRendererOverride, useSettings, type SplatRenderer } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useViewMode, type CellViewReport } from "@/state/viewMode";

import { A2, SITE_ID, twoEpochSite } from "./epochFixtures";

function show(
  renderer: SplatRenderer,
  representation: "gaussian-splat" | "mesh" = "gaussian-splat",
) {
  useSettings.getState().set({ splatRenderer: renderer });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(queryKeys.site(SITE_ID), twoEpochSite());
  useSites.setState({
    activeSiteId: SITE_ID,
    representation: { [SITE_ID]: representation },
    epoch: {},
  });
  return render(
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>
        <ViewsSlot />
      </GlassTooltipProvider>
    </QueryClientProvider>,
  );
}

const report = (r: CellViewReport) => act(() => useViewMode.getState().report(A2.id, r));

afterEach(() => {
  cleanup();
  useViewMode.setState({ mode: "photo", cells: {} });
  useSettings.getState().reset();
  useSites.setState({ activeSiteId: null, representation: {}, epoch: {} });
});

describe("the slot", () => {
  it("renders nothing unless a splat scan is the active site's model", () => {
    show("cesium", "mesh");
    expect(screen.queryByTestId("views")).toBeNull();
  });
});

describe("under CesiumJS", () => {
  it("offers all three views and keys the one drawn", async () => {
    show("cesium");
    const group = await screen.findByTestId("views");
    const radios = screen.getAllByRole("radio");
    expect(radios.map((r) => r.textContent)).toEqual(["Photo", "Confidence", "Data age"]);
    expect(radios.every((r) => !r.hasAttribute("disabled"))).toBe(true);
    expect(screen.queryByTestId("views-needs-cesium")).toBeNull();
    expect(screen.queryByTestId("cell-legend")).toBeNull();
    await userEvent.click(screen.getByRole("radio", { name: "Confidence" }));
    expect(useViewMode.getState().mode).toBe("confidence");
    expect(screen.getByTestId("cell-legend")).toHaveTextContent("Loading…");
    report({ kind: "ready", mode: "confidence" });
    expect(screen.getByTestId("cell-legend")).toHaveTextContent(
      /Weak: under half its splats pass the capture checks, or unseen.*Fair.*Strong.*No evidence.*weak-v1/,
    );
    report({ kind: "not-measured" });
    expect(screen.getByTestId("cell-legend")).toHaveTextContent(/^Not measured/);
    expect(group).toHaveAccessibleName("Splat view");
  });

  it("says how old the shown date is, and where newer data is or why it was not compared", async () => {
    show("cesium");
    await userEvent.click(await screen.findByRole("radio", { name: "Data age" }));
    report({
      kind: "ready",
      mode: "data-age",
      ageBucket: 3,
      newer: {
        dates: 2,
        compared: 1,
        cells: 40,
        skipped: ["Sep 1, 2026: placed in another frame"],
      },
    });
    expect(screen.getByTestId("cell-legend-age")).toHaveTextContent("This date: 6–12 months old");
    expect(screen.getByTestId("cell-legend-newer")).toHaveTextContent(
      "Newer data here: switch date",
    );
    expect(screen.getByTestId("cell-legend-skipped")).toHaveTextContent(
      "Not compared: Sep 1, 2026: placed in another frame",
    );
    report({
      kind: "ready",
      mode: "data-age",
      ageBucket: null,
      newer: { dates: 0, compared: 0, cells: 0, skipped: [] },
    });
    expect(screen.getByTestId("cell-legend-age")).toHaveTextContent(
      "This model has no capture date",
    );
    expect(screen.getByTestId("cell-legend-newer")).toHaveTextContent("No later date of this site");
  });

  it("says an age counted from the upload is one", async () => {
    show("cesium");
    await userEvent.click(await screen.findByRole("radio", { name: "Data age" }));
    report({
      kind: "ready",
      mode: "data-age",
      ageBucket: 0,
      ageFromUpload: true,
      newer: { dates: 0, compared: 0, cells: 0, skipped: [] },
    });
    expect(screen.getByTestId("cell-legend-age")).toHaveTextContent(
      "This date: uploaded under a month ago (capture time not recorded)",
    );
  });
});

describe.each(["playcanvas", "spark", "playcanvas-webgpu"] as const)("under %s", (renderer) => {
  it("disables the analytic views, says why, and switches to CesiumJS in one click", async () => {
    useViewMode.getState().set("confidence");
    show(renderer);
    await screen.findByTestId("views");
    expect(screen.getByRole("radio", { name: "Photo" })).not.toHaveAttribute("disabled");
    expect(screen.getByRole("radio", { name: "Confidence" })).toHaveAttribute("disabled");
    expect(screen.getByRole("radio", { name: "Data age" })).toHaveAttribute("disabled");
    // What is drawn is the photo, so that is what the control says.
    expect(screen.getByRole("radio", { name: "Photo" })).toHaveAttribute("aria-checked", "true");
    expect(screen.queryByTestId("cell-legend")).toBeNull();
    expect(screen.getByTestId("views-needs-cesium")).toHaveTextContent(
      "Needs the CesiumJS renderer (Settings › Advanced › Splat renderer)",
    );
    act(() => useRendererOverride.setState({ renderer }));
    await userEvent.click(screen.getByRole("button", { name: "Draw with CesiumJS" }));
    expect(useSettings.getState().splatRenderer).toBe("cesium");
    expect(useRendererOverride.getState().renderer).toBeNull();
    await waitFor(() => expect(screen.queryByTestId("views-needs-cesium")).toBeNull());
    expect(screen.getByRole("radio", { name: "Confidence" })).toHaveAttribute(
      "aria-checked",
      "true",
    );
  });
});
