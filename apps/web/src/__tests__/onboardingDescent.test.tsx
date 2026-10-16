/**
 * "Where is your land?" on the welcome card (features/descent/DescentForm.tsx) and the card the
 * Descent lands on (DescentArrival.tsx). The geocoder is asked once, on submit, never per key
 * (Nominatim forbids autocomplete); coordinates need no geocoder; and "Outline this land" goes
 * through the Land tool's own entry points, never a bare useLand.propose that would revise the
 * land already selected.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type * as Osm from "@/missions/osm";

import type { LandArea } from "@twin/contracts";
import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { env } from "@/app/env";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { sceneRegistry } from "@/cesium/SceneContext";
import type { GeocodeResult } from "@/cesium/types";
import { DescentArrival } from "@/features/descent/DescentArrival";
import { DESCENT_LAYER_ID, stopDescent } from "@/features/descent/descent";
import { OnboardingCard } from "@/features/onboarding/OnboardingCard";
import type { OsmCandidate } from "@/missions/osm";
import { useDescent } from "@/state/descent";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { useLantern } from "@/state/lantern";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";
import { useViewer } from "@/state/viewer";

vi.mock("@/missions/osm", async (importOriginal) => ({
  ...(await importOriginal<typeof Osm>()),
  fetchOsmContaining: () => new Promise(() => undefined),
}));

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>
  );
}

const PLACE = (label: string, longitude = -120.04, latitude = 34.698): GeocodeResult => ({
  label,
  destination: { kind: "point", longitude, latitude },
});

function fakeScene(search: (query: string) => Promise<GeocodeResult[]>) {
  const handle = { active: true, retarget: vi.fn(), cancel: vi.fn() };
  const scene = {
    geocoder: {
      name: "Test",
      attribution: "Search results © Test geocoder",
      search: vi.fn(search),
    },
    sites: { flyTo: vi.fn(() => Promise.resolve()) },
    areas: { cancelPick: vi.fn(), edit: vi.fn() },
    camera: {
      flyHome: vi.fn(),
      flyToRectangle: vi.fn(),
      glide: vi.fn(() => handle),
      groundHeightAt: vi.fn(() => new Promise(() => undefined)),
      rectangleArrival: vi.fn(() => ({
        longitude: 0,
        latitude: 0,
        height: 800,
        heading: 0,
        pitch: -38,
      })),
      footprintArrival: vi.fn(),
    },
  };
  sceneRegistry.set(scene as unknown as CesiumSceneManager);
  return scene;
}

const FIELD: OsmCandidate = {
  id: "osm-w42",
  name: "Creek pasture",
  acres: 412.4,
  footprint: {
    type: "Polygon",
    coordinates: [
      [
        [-120.05, 34.69],
        [-120.03, 34.69],
        [-120.03, 34.71],
        [-120.05, 34.69],
      ],
    ],
  },
};

beforeEach(() => {
  vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
  useViewer.getState().setStatus("ready");
  useSettings.getState().set({ onboardingDismissed: false });
  useDescent.getState().clear();
  useLand.getState().clear();
  useLandContext.getState().clear();
  useUi.getState().setPanel(null);
});
afterEach(() => {
  cleanup();
  stopDescent();
  sceneRegistry.set(null);
  useSettings.getState().reset();
  vi.restoreAllMocks();
});

describe("the place field on the welcome card", () => {
  it("asks the geocoder once, on submit, never while typing", async () => {
    const scene = fakeScene(() => Promise.resolve([PLACE("Oak Bench Road")]));
    render(wrap(<OnboardingCard />));
    const input = await screen.findByLabelText("Where do you want to go?");
    await userEvent.type(input, "Oak Bench Road, Los Olivos");
    expect(scene.geocoder.search).not.toHaveBeenCalled();
    expect(screen.getByText("Search results © Test geocoder")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Take me there" }));
    expect(scene.geocoder.search).toHaveBeenCalledTimes(1);
    expect(scene.geocoder.search).toHaveBeenCalledWith(
      "Oak Bench Road, Los Olivos",
      expect.any(AbortSignal),
    );
    // One place: straight there, and the welcome card goes.
    await waitFor(() => expect(scene.camera.glide).toHaveBeenCalledTimes(1));
    expect(useSettings.getState().onboardingDismissed).toBe(true);
    expect(useDescent.getState().target?.label).toBe("Oak Bench Road");
  });

  it("offers several places as choices, each flying there", async () => {
    const scene = fakeScene(() =>
      Promise.resolve([PLACE("Los Olivos, CA"), PLACE("Los Olivos, Mexico", -100, 20)]),
    );
    render(wrap(<OnboardingCard />));
    await userEvent.type(await screen.findByTestId("descent-input"), "Los Olivos");
    await userEvent.click(screen.getByTestId("descent-submit"));
    const choices = await screen.findByTestId("descent-choices");
    expect(choices).toHaveTextContent("Los Olivos, CA");
    expect(scene.camera.glide).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Los Olivos, Mexico" }));
    expect(scene.camera.glide).toHaveBeenCalledTimes(1);
    expect(useDescent.getState().target).toMatchObject({ longitude: -100, latitude: 20 });
  });

  it("says when nothing was found, in place", async () => {
    fakeScene(() => Promise.resolve([]));
    render(wrap(<OnboardingCard />));
    await userEvent.type(await screen.findByTestId("descent-input"), "Nowhere at all");
    await userEvent.click(screen.getByTestId("descent-submit"));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "No place found for “Nowhere at all”",
    );
  });

  it("flies to coordinates without asking the geocoder", async () => {
    const scene = fakeScene(() => Promise.resolve([]));
    render(wrap(<OnboardingCard />));
    await userEvent.type(await screen.findByTestId("descent-input"), "34.698, -120.04");
    await userEvent.click(screen.getByTestId("descent-submit"));
    expect(scene.geocoder.search).not.toHaveBeenCalled();
    expect(scene.camera.glide).toHaveBeenCalledTimes(1);
    expect(useDescent.getState().target).toMatchObject({
      latitude: 34.698,
      longitude: -120.04,
      attribution: "Coordinates you entered",
    });
  });

  it("keeps Explore Earth and the demo ranch", async () => {
    const scene = fakeScene(() => Promise.resolve([]));
    render(wrap(<OnboardingCard />));
    await userEvent.click(await screen.findByTestId("onboarding-explore"));
    expect(scene.camera.flyHome).toHaveBeenCalled();
    expect(useSettings.getState().onboardingDismissed).toBe(true);
    cleanup();
    useSettings.getState().set({ onboardingDismissed: false });
    render(wrap(<OnboardingCard />));
    const demo = await screen.findByRole("button", { name: "Open the demo ranch" });
    await waitFor(() => expect(demo).toBeEnabled());
    await userEvent.click(demo);
    expect(scene.sites.flyTo).toHaveBeenCalled();
  });
});

describe("the arrival card", () => {
  const arrive = (mapped: OsmCandidate | null) => {
    useDescent.getState().start({
      label: "34.69800° N, 120.04000° W",
      longitude: -120.04,
      latitude: 34.698,
      attribution: "Coordinates you entered",
    });
    useDescent.getState().setMapped(mapped, mapped ? "found" : "none");
    useDescent.getState().arrive();
  };

  it("outlines the mapped area as a new draft through the Land tool's entry", async () => {
    const enabled = env.landExplorationEnabled;
    env.landExplorationEnabled = true;
    try {
      const scene = fakeScene(() => Promise.resolve([]));
      // A saved land is selected: the outline must not become a revision of it.
      useLand.getState().select({ id: "land-1", name: "Saved" } as LandArea);
      useLandContext.getState().setLayer({ id: DESCENT_LAYER_ID, title: "x", features: [] });
      render(wrap(<DescentArrival />));
      act(() => arrive(FIELD));
      expect(screen.getByTestId("descent-mapped")).toHaveTextContent(
        "Mapped area: Creek pasture · 412 acres, from OpenStreetMap. Not a legal parcel.",
      );
      await userEvent.click(screen.getByRole("button", { name: "Outline this land" }));
      expect(useUi.getState().activePanel).toBe("land");
      const land = useLand.getState();
      expect(land.active).toBeNull();
      expect(land.draft?.name).toBe("Creek pasture");
      expect(land.draft?.source).toMatchObject({
        method: "mapped-feature",
        meaning: "physical-feature",
        label: "OpenStreetMap feature",
      });
      expect(scene.areas.cancelPick).toHaveBeenCalled();
      expect(useLandContext.getState().layers[DESCENT_LAYER_ID]).toBeUndefined();
      expect(useDescent.getState().phase).toBe("idle");
    } finally {
      env.landExplorationEnabled = enabled;
    }
  });

  it("starts drawing where nothing is mapped", async () => {
    const enabled = env.landExplorationEnabled;
    env.landExplorationEnabled = true;
    try {
      fakeScene(() => Promise.resolve([]));
      render(wrap(<DescentArrival />));
      act(() => arrive(null));
      expect(screen.getByTestId("descent-mapped")).toHaveTextContent(
        "No mapped boundary here; you can draw one.",
      );
      await userEvent.click(screen.getByRole("button", { name: "Outline this land" }));
      expect(useLand.getState().mode).toBe("draw");
      expect(useLand.getState().draft).toBeNull();
      expect(useUi.getState().activePanel).toBe("land");
    } finally {
      env.landExplorationEnabled = enabled;
    }
  });

  it("has no Land wording without the Land tool, and looks back in time when it can", async () => {
    fakeScene(() => Promise.resolve([]));
    useLantern.getState().setAvailability({ ok: true });
    render(wrap(<DescentArrival />));
    act(() => arrive(null));
    expect(screen.queryByRole("button", { name: "Outline this land" })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Look back in time" }));
    expect(useLantern.getState().mode).toBe("sticky");
    const bbox = useLantern.getState().bboxOverride;
    expect(bbox?.[0]).toBeLessThan(-120.04);
    expect(bbox?.[2]).toBeGreaterThan(-120.04);
    useLantern.getState().setMode("off");
    useLantern.getState().setAvailability({ ok: false, reason: "open-world" });
    await userEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByTestId("descent-arrival")).not.toBeInTheDocument();
  });
});
