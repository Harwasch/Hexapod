/**
 * The controls beside Splat / Mesh describe the model the scene shows -- on a site with two
 * dated captures, the shown epoch's -- not the site's first asset of the representation.
 *
 * Methods (other methods' objects, fill, motion) and the objects search are handed the asset
 * whose sidecars they read. They used to be handed `site.assets.find(representation)`, the
 * first listed: with the latest of two epochs drawn, they followed the older one.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { queryKeys } from "@/api/queries";
import { RepresentationSwitcher } from "@/features/sites/RepresentationSwitcher";
import { SiteLoadStatus } from "@/features/sites/SiteLoadStatus";
import { useSites } from "@/state/sites";

import { A1, A2, E1, E2, SITE_ID, twoEpochSite } from "./epochFixtures";

vi.mock("@/cesium/SceneContext", () => ({ useScene: () => null }));
// Stand-ins that say which asset they were handed; the real modules are not edited.
vi.mock("@/features/sites/InstanceSearch", () => ({
  InstanceSearch: ({ assetId }: { assetId: string }) => (
    <span data-testid="search-asset">{assetId}</span>
  ),
}));
vi.mock("@/features/sites/CompareMethods", () => ({
  CompareMethods: ({ assetId }: { assetId: string }) => (
    <span data-testid="methods-asset">{assetId}</span>
  ),
}));

function show(ui: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(queryKeys.site(SITE_ID), twoEpochSite());
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

beforeEach(() => {
  useSites.setState({
    activeSiteId: SITE_ID,
    nearSiteId: SITE_ID,
    inViewSiteId: SITE_ID,
    flightSiteId: null,
    representation: { [SITE_ID]: "gaussian-splat" },
    epoch: {},
    assets: {},
    siteLoads: {},
  });
});

afterEach(() => cleanup());

describe("the controls follow the shown epoch", () => {
  it("Methods and the objects search are handed the shown epoch's asset", () => {
    show(<RepresentationSwitcher />);
    expect(screen.getByTestId("search-asset")).toHaveTextContent(A2.id);
    expect(screen.getByTestId("methods-asset")).toHaveTextContent(A2.id);
    act(() => useSites.setState({ epoch: { [SITE_ID]: E1.id } }));
    expect(screen.getByTestId("search-asset")).toHaveTextContent(A1.id);
    expect(screen.getByTestId("methods-asset")).toHaveTextContent(A1.id);
    act(() => useSites.setState({ epoch: { [SITE_ID]: E2.id } }));
    expect(screen.getByTestId("search-asset")).toHaveTextContent(A2.id);
  });

  it("the load pill speaks for the shown epoch's model", () => {
    // The older date failed; the latest, which is shown, is still loading.
    useSites.setState({
      assets: {
        [A1.id]: {
          loadState: "error",
          error: "gone",
          progress: { pending: 0, processing: 0 },
          screenSpaceError: null,
          memoryMb: 0,
          boundingRadiusM: null,
        },
        [A2.id]: {
          loadState: "loading",
          error: null,
          progress: { pending: 0, processing: 0 },
          screenSpaceError: null,
          memoryMb: 0,
          boundingRadiusM: null,
        },
      },
    });
    show(<SiteLoadStatus />);
    expect(screen.getByTestId("site-load")).toHaveTextContent("Loading 3D model");
    act(() => useSites.setState({ epoch: { [SITE_ID]: E1.id } }));
    expect(screen.getByTestId("site-load")).toHaveTextContent("Couldn’t load the 3D model");
  });
});
