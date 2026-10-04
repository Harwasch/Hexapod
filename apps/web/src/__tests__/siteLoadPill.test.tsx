/**
 * The load pill beside Splat / Mesh / Points (SiteLoadStatus) speaks for the site a fly-to is
 * taking the camera to, not only for the active one.
 *
 * A site becomes active only once its record has arrived. Read from the active site alone, a
 * record that failed (a cold API past its deadline, a 5xx, no network) left the camera landing
 * at the summary's pose with no model and nothing on screen saying why -- and the Retry that
 * would fly there again out of reach.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GlassTooltipProvider } from "@twin/ui";

import { SiteLoadStatus } from "@/features/sites/SiteLoadStatus";
import { useSites, type SiteLoad } from "@/state/sites";

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>
  );
}

const FLOWN = "22222222-2222-4222-8222-222222222222";

const failed: SiteLoad = {
  phase: "error",
  progress: 0.05,
  error: "The site's details did not load: The server answered 503.",
  retryable: true,
  attempt: 1,
  flight: true,
  startedAt: 0,
};

describe("the site load pill", () => {
  afterEach(() => {
    useSites.setState({ activeSiteId: null, flightSiteId: null, siteLoads: {} });
    useSites.getState().setSiteLoadRetry(() => undefined);
  });

  it("says a fly-to's failed record, with Retry, before the site was ever active", async () => {
    const retry = vi.fn();
    useSites.getState().setSiteLoadRetry(retry);
    useSites.setState({
      activeSiteId: null,
      flightSiteId: FLOWN,
      siteLoads: { [FLOWN]: { ...failed, phase: "details", error: null, retryable: false } },
    });
    render(wrap(<SiteLoadStatus />));
    // While the record is on its way: loading, from the click.
    expect(screen.getByTestId("site-load")).toHaveTextContent("Loading 3D model");
    act(() => useSites.getState().setSiteLoad(FLOWN, failed));
    const pill = screen.getByTestId("site-load");
    expect(pill).toHaveTextContent("Couldn’t load the 3D model");
    await userEvent.setup().click(screen.getByTestId("site-load-retry"));
    expect(retry).toHaveBeenCalledWith(FLOWN);
  });

  it("speaks for the flight's site over the site still active from before", () => {
    useSites.setState({
      activeSiteId: "11111111-1111-4111-8111-111111111111",
      flightSiteId: FLOWN,
      siteLoads: { [FLOWN]: failed },
    });
    render(wrap(<SiteLoadStatus />));
    expect(screen.getByTestId("site-load")).toHaveTextContent("Couldn’t load the 3D model");
    // Once the camera has left it, the active site speaks again (here: nothing to say).
    act(() => useSites.getState().setFlightSite(null));
    expect(screen.queryByTestId("site-load")).not.toBeInTheDocument();
  });
});
