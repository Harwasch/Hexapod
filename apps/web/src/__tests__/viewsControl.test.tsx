/**
 * The splat view control (features/views): a slot that renders only for a splat site; under
 * CesiumJS, PlayCanvas and Spark the three views and the key to what is drawn, and under the
 * latter two a notice naming only what still needs CesiumJS; under the WebGPU trial the
 * analytic views disabled, the reason, and a switch that works.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
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
    // The first test loads the control's chunk: its transform is slow on a loaded machine.
    const group = await screen.findByTestId("views", {}, { timeout: 10_000 });
    const radios = within(screen.getByTestId("views-modes")).getAllByRole("radio");
    expect(radios.map((r) => r.textContent)).toEqual(["Photo", "Confidence", "Data age"]);
    expect(radios.every((r) => !r.hasAttribute("disabled"))).toBe(true);
    expect(screen.queryByTestId("views-needs-cesium")).toBeNull();
    expect(screen.queryByTestId("cell-legend")).toBeNull();
    await userEvent.click(screen.getByRole("radio", { name: "Confidence" }));
    expect(useViewMode.getState().mode).toBe("confidence");
    // The scan stays the photo until the grid is in: the key says it is coming, so the photo
    // is not read as a scan with no evidence.
    expect(screen.getByTestId("cell-legend")).toHaveTextContent("Loading confidence…");
    report({ kind: "ready", mode: "confidence", weakBelow: 128 });
    expect(screen.getByTestId("cell-legend")).toHaveTextContent(
      /Weak: score under 128 \(under half its splats pass the capture checks\), or unseen.*Fair: score 128–191.*Strong: score 192–255.*No evidence.*weak-v1/,
    );
    // Each class has a pattern of its own besides its colour, as the scan draws it.
    const legend = screen.getByTestId("cell-legend");
    expect(legend.querySelectorAll(".cell-legend__swatch--hatched")).toHaveLength(1);
    expect(legend.querySelectorAll(".cell-legend__swatch--dotted")).toHaveLength(1);
    // The thresholds are the grid's own.
    report({ kind: "ready", mode: "confidence", weakBelow: 100 });
    expect(screen.getByTestId("cell-legend")).toHaveTextContent(
      /Weak: score under 100 .*Fair: score 100–177.*Strong: score 178–255/,
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
    expect(screen.getByTestId("cell-legend")).toHaveTextContent("Loading data age…");
    // Confidence is in, Data age's comparison not yet: still loading, not Confidence's key.
    report({ kind: "ready", mode: "confidence", weakBelow: 128 });
    expect(screen.getByTestId("cell-legend")).toHaveTextContent("Loading data age…");
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

describe.each([
  ["playcanvas", "PlayCanvas"],
  ["spark", "Spark"],
] as const)("under %s", (renderer, name) => {
  it("offers all three views, keys the one drawn, and names only what still needs CesiumJS", async () => {
    useViewMode.getState().set("confidence");
    show(renderer);
    await screen.findByTestId("views");
    for (const view of ["Photo", "Confidence", "Data age"]) {
      expect(screen.getByRole("radio", { name: view })).not.toHaveAttribute("disabled");
    }
    expect(screen.getByRole("radio", { name: "Confidence" })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    report({ kind: "ready", mode: "confidence", weakBelow: 128 });
    const legend = screen.getByTestId("cell-legend");
    expect(legend).toHaveTextContent(/Weak: score under 128 .*Strong: score 192–255.*weak-v1/);
    // Honest about what this renderer draws: one cell level a tile.
    expect(legend).toHaveTextContent(
      `Cell size picked per tile by ${name} (per splat by CesiumJS)`,
    );
    // The notice stays for what CesiumJS alone draws, and no longer names the views.
    const notice = screen.getByTestId("views-needs-cesium");
    expect(notice).toHaveTextContent(
      "Per cell: Needs the CesiumJS renderer (Settings › Advanced › Splat renderer)",
    );
    expect(notice).not.toHaveTextContent(/Confidence|Data age|Sunlight/);
  });
});

describe("under the WebGPU trial", () => {
  const renderer = "playcanvas-webgpu" as const;
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
      "Confidence, Data age and Per cell: Needs the CesiumJS renderer (Settings › Advanced › Splat renderer)",
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
