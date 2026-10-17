/**
 * A component loaded on demand whose chunk fails to load (a flaky network, a tab left open
 * across a deploy) fails alone: the quiet boundary AppShell mounts the land bridges and the
 * settings sheet under renders nothing in its place, and everything beside it keeps rendering.
 */
import { render, screen, waitFor } from "@testing-library/react";
import { lazy, Suspense } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { recentLogs } from "@/lib/log";

describe("a quiet error boundary", () => {
  afterEach(() => vi.restoreAllMocks());

  it("renders nothing for a chunk that fails to load, and leaves its siblings alone", async () => {
    // React reports the caught error to the console; the boundary's own log line is the record.
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const Missing = lazy(() =>
      Promise.reject(new TypeError("Failed to fetch dynamically imported module")),
    );
    render(
      <div data-testid="hud">
        <ErrorBoundary quiet label="Land bridges">
          <Suspense fallback={null}>
            <Missing />
          </Suspense>
        </ErrorBoundary>
        <p>Layers</p>
      </div>,
    );
    // The boundary catches the rejected import and says so in the log.
    await waitFor(() =>
      expect(recentLogs().at(-1)).toMatchObject({
        level: "error",
        scope: "ui",
        data: { label: "Land bridges", error: "Failed to fetch dynamically imported module" },
      }),
    );
    expect(screen.getByText("Layers")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByText(/went wrong|hit a problem/)).not.toBeInTheDocument();
  });

  it("renders its children while nothing fails", async () => {
    const Loaded = lazy(() => Promise.resolve({ default: () => <p>Land panel</p> }));
    render(
      <ErrorBoundary quiet label="Land">
        <Suspense fallback={null}>
          <Loaded />
        </Suspense>
      </ErrorBoundary>,
    );
    expect(await screen.findByText("Land panel")).toBeVisible();
  });
});
