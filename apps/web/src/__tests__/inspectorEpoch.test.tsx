/**
 * Date shown, in the Inspector (features/inspector/EpochFacts.tsx): which of a site's dated
 * captures is on screen, observed or filled in, what captured it, how it is aligned and how it
 * is placed -- and nothing new for a site before epochs.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Site } from "@twin/contracts";
import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { queryKeys } from "@/api/queries";
import { InspectorPanel } from "@/features/inspector/InspectorPanel";
import { useSelection } from "@/state/selection";
import { useSites } from "@/state/sites";
import { useToasts } from "@/state/toasts";
import { useUi } from "@/state/ui";

import { A1, A2, E1, E2, SITE_ID, twoEpochSite } from "./epochFixtures";

const scene = vi.hoisted(() => {
  const listeners = new Map<string, Set<() => void>>();
  return {
    sites: {
      sharesPlacement: vi.fn((): boolean | null => true),
      previewScale: vi.fn(),
      loadedAssets: vi.fn((): { assetId: string; tileset: { root: { extras?: unknown } } }[] => []),
    },
    camera: { flyTo: vi.fn() },
    selection: { clear: vi.fn() },
    events: {
      on: (name: string, listener: () => void) => {
        const set = listeners.get(name) ?? new Set();
        listeners.set(name, set);
        set.add(listener);
        return () => set.delete(listener);
      },
      emit: (name: string) => listeners.get(name)?.forEach((listener) => listener()),
    },
  };
});
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));
const rescanWeak = vi.hoisted(() => vi.fn((..._args: unknown[]) => Promise.resolve()));
vi.mock("@/features/rescan/rescanWeak", () => ({ rescanWeak }));

/** The anchor rests on its own measured ground. */
const clamped = {
  ...A1,
  renderConfig: {
    ...A1.renderConfig,
    clampToGround: true,
    groundSamples: [{ lon: 0, lat: 0, height: 0 }],
  },
};

function inspect(site: Site, epochId: string | null = null) {
  vi.spyOn(api, "GET").mockImplementation((() =>
    Promise.resolve({ data: [], response: new Response() })) as never);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(queryKeys.site(SITE_ID), site);
  useSites.setState({
    activeSiteId: SITE_ID,
    representation: { [SITE_ID]: "gaussian-splat" },
    epoch: epochId ? { [SITE_ID]: epochId } : {},
    compareDates: null,
  });
  useSelection.getState().setSelection({
    kind: "site",
    title: site.name,
    siteId: SITE_ID,
    longitude: -123.88,
    latitude: 46.13,
    height: 10,
    terrainHeight: 4,
    observedAt: "2026-06-01T15:00:00Z",
    at: Date.now(),
  });
  useUi.getState().setInspectorOpen(true);
  return render(
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>
        <InspectorPanel />
      </GlassTooltipProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  scene.sites.sharesPlacement.mockReturnValue(true);
  scene.sites.loadedAssets.mockReturnValue([]);
});

afterEach(() => {
  cleanup();
  act(() => {
    useSelection.getState().setSelection(null);
    useUi.getState().setInspectorOpen(false);
  });
  vi.restoreAllMocks();
});

describe("Date shown", () => {
  it("states the shown epoch's date, source, platform and that it is not aligned yet", () => {
    inspect(twoEpochSite({ assets: [clamped, A2] }));
    expect(screen.getByTestId("inspector-epoch")).toHaveAccessibleName("Date shown: Yard");
    expect(screen.getByTestId("inspector-epoch-date")).toHaveTextContent(
      /^Jun 1, 2026, 03:00\sPM UTC$/,
    );
    expect(screen.getByTestId("inspector-epoch-source")).toHaveTextContent("Observed (captured)");
    expect(screen.getByTestId("inspector-epoch-platform")).toHaveTextContent("Phone video");
    expect(screen.getByTestId("inspector-epoch-alignment")).toHaveTextContent(
      "Not aligned yet: placed by its own GPS",
    );
    expect(screen.getByTestId("inspector-epoch-placement")).toHaveTextContent(
      "Shares the site frame's placement, rested on the Mar 1, 2026 capture's ground",
    );
    // One date on the card: the selection's Observed row gives way to Date shown.
    expect(screen.queryByText("Observed")).toBeNull();
  });

  it("the frame-defining epoch says it defines the frame", () => {
    inspect(twoEpochSite({ assets: [clamped, A2] }), E1.id);
    expect(screen.getByTestId("inspector-epoch-date")).toHaveTextContent(/^Mar 1, 2026/);
    expect(screen.getByTestId("inspector-epoch-platform")).toHaveTextContent("Drone");
    expect(screen.getByTestId("inspector-epoch-alignment")).toHaveTextContent(
      "Defines the site frame",
    );
    expect(screen.getByTestId("inspector-epoch-placement")).toHaveTextContent(
      "Rests on its own measured ground; every date of this site shares this placement",
    );
  });

  it("a date the scene placed on its own says so", () => {
    scene.sites.sharesPlacement.mockReturnValue(false);
    inspect(twoEpochSite({ assets: [clamped, A2] }));
    expect(scene.sites.sharesPlacement).toHaveBeenCalledWith(SITE_ID, A2.id);
    expect(screen.getByTestId("inspector-epoch-placement")).toHaveTextContent(
      "Placed on its own, not with the placement the site's other dates share",
    );
  });

  it("a date whose placement is not decided yet does not claim one, and follows the scene", () => {
    scene.sites.sharesPlacement.mockReturnValue(null);
    inspect(twoEpochSite({ assets: [clamped, A2] }));
    const placement = screen.getByTestId("inspector-epoch-placement");
    expect(placement).toHaveTextContent(
      "Not placed yet: waiting for the Mar 1, 2026 capture, whose placement every date shares",
    );
    // The scene decides (its anchor loaded, the frames agree) and says so.
    scene.sites.sharesPlacement.mockReturnValue(true);
    act(() => scene.events.emit("tilesets"));
    expect(placement).toHaveTextContent(
      "Shares the site frame's placement, rested on the Mar 1, 2026 capture's ground",
    );
  });

  it("an upload date is said to be one", () => {
    inspect(twoEpochSite({ epochs: [E1, { ...E2, observedAtSource: "created" }] }));
    expect(screen.getByTestId("inspector-epoch-date")).toHaveTextContent(
      "· upload date; capture time not recorded",
    );
  });

  it("while two dates are compared, describes each by its side of the divider", () => {
    inspect(twoEpochSite({ assets: [clamped, A2] }));
    act(() => useSites.getState().setCompareDates(SITE_ID, { left: E1.id, right: E2.id }));
    expect(screen.queryByTestId("inspector-epoch")).toBeNull();
    const left = screen.getByTestId("inspector-epoch-left");
    const right = screen.getByTestId("inspector-epoch-right");
    expect(left).toHaveTextContent("Left of the divider · Yard");
    expect(left).toHaveTextContent("Mar 1, 2026");
    expect(left).toHaveTextContent("Defines the site frame");
    expect(right).toHaveTextContent("Right of the divider · Yard");
    expect(right).toHaveTextContent("Jun 1, 2026");
    expect(right).toHaveTextContent("Phone video");
    act(() => useSites.getState().setCompareDates(SITE_ID, null));
    expect(screen.getByTestId("inspector-epoch")).toHaveTextContent("Date shown · Yard");
  });

  it("a site without epochs shows no Date shown section", () => {
    const legacy = { ...A1, epochId: null, observedAt: "2025-05-05T00:00:00Z" };
    inspect(twoEpochSite({ epochs: [], assets: [legacy] }));
    expect(screen.queryByTestId("inspector-epoch")).toBeNull();
    // The card's own Observed row stays, as before.
    expect(screen.getByText("Observed")).toBeInTheDocument();
  });
});

describe("Confidence, in Date shown", () => {
  /** The design's example summary (M4-6 C2). */
  const confidence = {
    format: "hexapod.cellgrid",
    version: 1,
    policy: "weak-v1",
    summary: {
      cells: 5600,
      scored: 4800,
      weak: 620,
      unseen: 80,
      weakShare: 0.129,
      reasons: { views: 300, spread: 210, gsd: 40, accuracy: 0, unseen: 80, alignment: 0 },
    },
  };
  const loaded = (extras: unknown) =>
    scene.sites.loadedAssets.mockReturnValue([{ assetId: A2.id, tileset: { root: { extras } } }]);

  it("says how much of the shown model is weak, and mostly why", () => {
    loaded({ confidence });
    inspect(twoEpochSite({ assets: [clamped, A2] }));
    expect(screen.getByTestId("inspector-epoch-confidence")).toHaveTextContent(
      "13% weak: views 6%, spread 4%, unseen 2% · weak-v1",
    );
  });

  it("says a model without capture evidence was not measured, and waits for one to load", () => {
    loaded({ viewCones: {} });
    inspect(twoEpochSite({ assets: [clamped, A2] }));
    expect(screen.getByTestId("inspector-epoch-confidence")).toHaveTextContent(
      /^Not measured: this model came without capture evidence/,
    );
    cleanup();
    scene.sites.loadedAssets.mockReturnValue([]);
    inspect(twoEpochSite({ assets: [clamped, A2] }));
    expect(screen.getByTestId("inspector-epoch-confidence")).toHaveTextContent(
      "Known once the model has loaded",
    );
  });

  it("says when nothing was weak", () => {
    loaded({ confidence: { ...confidence, summary: { scored: 10, weak: 0, weakShare: 0 } } });
    inspect(twoEpochSite({ assets: [clamped, A2] }));
    expect(screen.getByTestId("inspector-epoch-confidence")).toHaveTextContent(
      "Nothing weak · weak-v1",
    );
  });
});

describe("Plan a re-scan, in Date shown", () => {
  it("plans a re-scan of the shown date's weak areas, whatever draws the splats", async () => {
    // Nothing loaded in the scene (another renderer, or still loading): offered all the same.
    inspect(twoEpochSite({ assets: [clamped, A2] }));
    const button = screen.getByRole("button", { name: "Plan a re-scan of weak areas" });
    fireEvent.click(button);
    expect(button).toBeDisabled();
    await waitFor(() => expect(rescanWeak).toHaveBeenCalledTimes(1));
    const [passedScene, site, epochId] = rescanWeak.mock.calls[0] ?? [];
    expect(passedScene).toBe(scene);
    expect((site as Site).id).toBe(SITE_ID);
    expect(epochId).toBe(E2.id);
    await waitFor(() => expect(button).toBeEnabled());
  });

  it("says so when the re-scan flow cannot be loaded", async () => {
    useToasts.setState({ items: [] });
    // A chunk gone after a deploy rejects the import, as the flow failing does here.
    rescanWeak.mockRejectedValueOnce(new Error("Failed to fetch dynamically imported module"));
    inspect(twoEpochSite({ assets: [clamped, A2] }));
    const button = screen.getByRole("button", { name: "Plan a re-scan of weak areas" });
    fireEvent.click(button);
    await waitFor(() =>
      expect(useToasts.getState().items.map((t) => t.title)).toEqual([
        "Could not plan the re-scan",
      ]),
    );
    expect(button).toBeEnabled();
  });

  it("asks for the date on screen", async () => {
    rescanWeak.mockClear();
    inspect(twoEpochSite({ assets: [clamped, A2] }), E1.id);
    fireEvent.click(screen.getByRole("button", { name: "Plan a re-scan of weak areas" }));
    await waitFor(() => expect(rescanWeak).toHaveBeenCalledTimes(1));
    expect(rescanWeak.mock.calls[0]?.[2]).toBe(E1.id);
  });
});
