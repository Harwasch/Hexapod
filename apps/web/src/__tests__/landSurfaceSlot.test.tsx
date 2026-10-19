/**
 * The Land surface's chunk loads with the shell. When that load fails (a network blip, a tab left
 * open across a deploy), the left dock's other panels keep working, and opening Land says to
 * reload rather than replacing the whole dock with "Panel hit a problem".
 */
import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { LandSurfaceSlot } from "@/features/shell/LandSurfaceSlot";
import { recentLogs } from "@/lib/log";
import { useUi } from "@/state/ui";

vi.mock("@/features/land/surface/LandSurface", () => {
  throw new TypeError("Failed to fetch dynamically imported module");
});

afterEach(() => {
  vi.restoreAllMocks();
  act(() => useUi.getState().setPanel(null));
});

it("keeps Layers, Measure and Add when the Land chunk fails, and says to reload Land", async () => {
  // React reports the caught error to the console; the boundary's own log line is the record.
  vi.spyOn(console, "error").mockImplementation(() => undefined);
  render(
    <ErrorBoundary inline label="Panel">
      <p>Layers</p>
      <LandSurfaceSlot />
      <p>Measure</p>
      <p>Add</p>
    </ErrorBoundary>,
  );
  await waitFor(() =>
    expect(recentLogs().at(-1)).toMatchObject({ level: "error", data: { label: "Land surface" } }),
  );
  for (const panel of ["Layers", "Measure", "Add"]) expect(screen.getByText(panel)).toBeVisible();
  expect(screen.queryByText(/hit a problem/)).not.toBeInTheDocument();
  // Nothing in the dock while another panel is open; the reason once Land is.
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  act(() => useUi.getState().setPanel("land"));
  expect(screen.getByRole("alert")).toHaveTextContent(
    "Land could not loadIts code did not download. Reload the page to load it again.",
  );
  expect(screen.getByRole("button", { name: "Reload" })).toBeVisible();
  expect(screen.getByText("Layers")).toBeVisible();
});
