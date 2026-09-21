import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { InspectorPanel } from "@/features/inspector/InspectorPanel";
import { DevPanel } from "@/features/dev/DevPanel";
import { SimulatedBadge } from "@/features/living/SimulatedBadge";
import { LayersPanel } from "@/features/layers/LayersPanel";
import { SettingsSheet } from "@/features/settings/SettingsSheet";
import { OnboardingCard } from "@/features/onboarding/OnboardingCard";
import { SetupNotices } from "@/features/notices/SetupNotices";
import { CommandBar } from "@/features/mission/CommandBar";
import { ToolRail } from "@/features/shell/ToolRail";
import {
  DEFAULT_WIND_STRENGTH,
  useLiving,
  type LivingSiteStatus,
  type LivingSurveyStatus,
} from "@/state/living";
import { useSelection } from "@/state/selection";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";
import { useViewer } from "@/state/viewer";

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>
  );
}

/** The fixture's own rig note, verbatim from `data/tiles/synthetic-tree/source/rig.json`. */
const RIG_NOTE = "synthetic tree, 6.0 m, 33 nodes (tools/captures/synthetic_tree.py)";

/** A published Living Survey status, shaped exactly as `LivingSurveyManager` publishes it. */
function livingStatus(
  animating: boolean,
  over: Partial<LivingSiteStatus> = {},
): LivingSurveyStatus {
  const site: LivingSiteStatus = {
    siteId: "builtin-demo-site",
    siteSlug: "synthetic-tree",
    assetId: "builtin-demo-splat",
    phase: "ready",
    numSplats: 2000,
    displaced: animating,
    rigSourceNote: RIG_NOTE,
    maxDisplacementM: 0.197,
    sortStaleness: 9.8,
    ...over,
  };
  return {
    wind: { strength: animating ? DEFAULT_WIND_STRENGTH : 0, bearingDeg: 250 },
    animating,
    sites: [site],
  };
}

beforeEach(() => {
  useUi.setState({ activePanel: null, inspectorOpen: false });
  useSelection.setState({ selection: null });
  useSites.setState({ activeSiteId: null });
  useSettings.getState().reset();
  useLiving.getState().reset();
});

describe("ToolRail + panels", () => {
  it("opens the layers panel and shows the built-in catalog when the API is offline", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("Failed to fetch"));
    render(
      wrap(
        <>
          <ToolRail />
          <LayersPanel />
        </>,
      ),
    );
    await userEvent.click(screen.getByRole("button", { name: "Layers" }));
    expect(await screen.findByTestId("layers-panel")).toBeInTheDocument();
    expect(await screen.findByText("Built-in catalog (API offline)")).toBeInTheDocument();
    expect(await screen.findByTestId("layer-card-openstreetmap")).toBeInTheDocument();
    await userEvent.type(screen.getByTestId("layers-filter"), "terrain");
    await waitFor(() =>
      expect(screen.queryByTestId("layer-card-openstreetmap")).not.toBeInTheDocument(),
    );
    expect(screen.getByTestId("layer-card-cesium-world-terrain")).toBeInTheDocument();
    vi.restoreAllMocks();
  });
});

describe("InspectorPanel", () => {
  it("opens when a selection exists and shows position and attribution", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    render(wrap(<InspectorPanel />));
    expect(screen.queryByTestId("inspector")).not.toBeInTheDocument();
    useSelection.getState().setSelection({
      kind: "ground",
      title: "Location",
      longitude: -122.138,
      latitude: 47.644,
      height: 120,
      terrainHeight: 118.5,
      attribution: [{ text: "Cesium World Terrain", organization: null, url: null }],
      at: Date.now(),
    });
    useUi.getState().setInspectorOpen(true);
    expect(await screen.findByTestId("inspector")).toBeInTheDocument();
    expect(screen.getByTestId("inspector-position")).toHaveTextContent("47.64400° N, 122.13800° W");
    expect(screen.getByText("Cesium World Terrain")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Close panel" }));
    await waitFor(() => expect(screen.queryByTestId("inspector")).not.toBeInTheDocument());
    vi.restoreAllMocks();
  });
});

describe("CommandBar + Onboarding", () => {
  it("renders altitude in the selected unit system", () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useViewer.getState().setCamera({
      ...useViewer.getState().camera,
      altitude: 1500,
      scaleBand: "city",
      metersPerPixel: 2,
    });
    const { rerender } = render(wrap(<CommandBar />));
    expect(screen.getByTestId("status-altitude")).toHaveTextContent("1.5 km");
    useSettings.getState().set({ units: "imperial" });
    rerender(wrap(<CommandBar />));
    expect(screen.getByTestId("status-altitude")).toHaveTextContent("ft");
  });

  it("onboarding appears once the viewer is ready and stays dismissed", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useViewer.getState().setStatus("ready");
    render(wrap(<OnboardingCard />));
    expect(await screen.findByTestId("onboarding")).toBeInTheDocument();
    await userEvent.click(screen.getByTestId("onboarding-explore"));
    await waitFor(() => expect(screen.queryByTestId("onboarding")).not.toBeInTheDocument());
    expect(useSettings.getState().onboardingDismissed).toBe(true);
    vi.restoreAllMocks();
  });
});

describe("SetupNotices", () => {
  it("lets the evaluation-token hint be dismissed for good, and keeps a rejected token loud", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useViewer.getState().setTokenState("default");
    const { unmount } = render(wrap(<SetupNotices />));
    expect(await screen.findByTestId("notice-default-token")).toBeInTheDocument();

    // A real, labelled, keyboard-reachable control — not a click handler on a div.
    const dismiss = screen.getByRole("button", {
      name: "Dismiss the evaluation ion token notice",
    });
    dismiss.focus();
    await userEvent.keyboard("{Enter}");
    await waitFor(() =>
      expect(screen.queryByTestId("notice-default-token")).not.toBeInTheDocument(),
    );
    expect(useSettings.getState().ionTokenNoticeDismissed).toBe(true);
    // It survives a reload because it rides the same persisted settings as `onboardingDismissed`.
    await waitFor(() =>
      expect(
        JSON.parse(window.localStorage.getItem("twin.settings.v1") ?? "{}") as {
          state?: { ionTokenNoticeDismissed?: boolean };
        },
      ).toMatchObject({ state: { ionTokenNoticeDismissed: true } }),
    );

    // Dismissed means dismissed: the setting is persisted, so a later mount stays quiet.
    unmount();
    const remount = render(wrap(<SetupNotices />));
    await waitFor(() =>
      expect(screen.queryByTestId("notice-default-token")).not.toBeInTheDocument(),
    );
    remount.unmount();

    // A rejected token is a fault, not a setup note: it is never dismissible.
    useViewer.getState().setTokenState("invalid");
    render(wrap(<SetupNotices />));
    expect(await screen.findByTestId("notice-token")).toBeInTheDocument();
    expect(screen.queryByTestId("notice-default-token-dismiss")).not.toBeInTheDocument();
    useViewer.getState().setTokenState("unknown");
    vi.restoreAllMocks();
  });
});

describe("The Simulated badge", () => {
  it("is absent while nothing moves and present, named, while something does", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    const { rerender } = render(wrap(<SimulatedBadge />));
    expect(screen.queryByTestId("simulated-badge")).not.toBeInTheDocument();

    useLiving.getState().setStatus(livingStatus(true));
    rerender(wrap(<SimulatedBadge />));
    const badge = await screen.findByTestId("simulated-badge");
    expect(badge).toHaveTextContent("Simulated motion");
    // Names what is moving, from the catalog, so a viewer knows which thing on screen is modelled.
    await waitFor(() => expect(badge).toHaveTextContent("Cesium Gaussian splat demo"));
    // It is announced, not merely coloured: jsx-a11y aside, an amber border means nothing to a
    // screen reader and nothing to a person who cannot see the difference.
    expect(badge).toHaveAttribute("role", "status");

    useLiving.getState().setStatus(livingStatus(false));
    rerender(wrap(<SimulatedBadge />));
    await waitFor(() => expect(screen.queryByTestId("simulated-badge")).not.toBeInTheDocument());
    vi.restoreAllMocks();
  });

  it("stays silent while a rig is still attaching, and never overclaims the geometry", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useLiving.getState().setStatus(livingStatus(true, { phase: "waiting", displaced: false }));
    const { rerender } = render(wrap(<SimulatedBadge />));
    expect(screen.queryByTestId("simulated-badge")).not.toBeInTheDocument();

    useLiving.getState().setStatus(livingStatus(true));
    rerender(wrap(<SimulatedBadge />));
    const text = (await screen.findByTestId("simulated-badge")).textContent ?? "";
    // The three claims this badge must never make. It cannot tell a scanned tree from a
    // procedural one, so it says nothing about the geometry beyond that it is untouched.
    expect(text).not.toMatch(/m\/s/);
    expect(text).not.toMatch(/wind from/i);
    expect(text).not.toMatch(/measured|surveyed|scanned/i);
    expect(text).toMatch(/Modelled wind/);
    expect(text).toMatch(/never altered/);
    vi.restoreAllMocks();
  });
});

describe("DevPanel: the one place sort staleness is printed", () => {
  it("prints the yardstick the ratio was divided by, and says it is not this tileset's", () => {
    useSettings.getState().set({ devToolsOpen: true });
    useLiving.getState().setStatus(livingStatus(true));
    render(wrap(<DevPanel />));

    const line = screen.getByTestId("dev-living");
    expect(line).toHaveTextContent("9.8 splat radii");
    // Without this clause the number reads as a measurement of the splats on screen. The
    // denominator is a fixed reference from a different capture; the fixture's own median is
    // five times coarser, so the figure is pessimistic by about that factor.
    expect(line).toHaveTextContent("against a 2 cm reference gaussian");
    expect(line).toHaveTextContent("not this tileset's own median");
  });

  it("says so plainly when nothing is rigged", () => {
    useSettings.getState().set({ devToolsOpen: true });
    render(wrap(<DevPanel />));
    expect(screen.getByTestId("dev-living")).toHaveTextContent("no rigged site loaded");
  });
});

describe("InspectorPanel: observed and simulated", () => {
  it("separates what was recorded from what is modelled, for a living site", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useLiving.getState().setStatus(livingStatus(true));
    useSelection.getState().setSelection({
      kind: "site",
      title: "Cesium Gaussian splat demo",
      longitude: -122.138,
      latitude: 47.644,
      height: 120,
      terrainHeight: 118.5,
      siteId: "builtin-demo-site",
      at: Date.now(),
    });
    useUi.getState().setInspectorOpen(true);
    render(wrap(<InspectorPanel />));

    const section = await screen.findByTestId("inspector-living");
    // The catalog holds no capture date and no resolution for this asset. That absence is
    // stated, not filled in — and not spun into a claim that it was generated either.
    expect(screen.getByTestId("inspector-geometry")).toHaveTextContent(
      "No capture date or resolution recorded",
    );
    expect(screen.getByTestId("inspector-geometry")).toHaveTextContent(
      "Sample asset referenced by the official CesiumJS Sandcastle.",
    );
    // The motion carries the rig's own provenance string, verbatim.
    expect(screen.getByTestId("inspector-motion")).toHaveTextContent(`Simulated · ${RIG_NOTE}`);
    expect(screen.getByTestId("inspector-motion")).toHaveTextContent("Modelled wind is moving it");
    expect(section.textContent).not.toMatch(/m\/s/);
    expect(section.textContent).not.toMatch(/wind from/i);
    // And the top of the panel says so at a glance, in words.
    expect(screen.getAllByText("Simulated motion").length).toBeGreaterThan(0);
    vi.restoreAllMocks();
  });

  it("says the tree is standing still when the wind is off, and vanishes for other sites", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useLiving.getState().setStatus(livingStatus(false));
    useSelection.getState().setSelection({
      kind: "site",
      title: "Cesium Gaussian splat demo",
      longitude: -122.138,
      latitude: 47.644,
      height: 120,
      terrainHeight: 118.5,
      siteId: "builtin-demo-site",
      at: Date.now(),
    });
    useUi.getState().setInspectorOpen(true);
    const { rerender } = render(wrap(<InspectorPanel />));
    expect(await screen.findByTestId("inspector-motion")).toHaveTextContent(
      "Wind is off, so it stands exactly as it was loaded.",
    );
    expect(screen.queryByText("Simulated motion")).not.toBeInTheDocument();

    // A site with no rig attached gets no section at all — the split is about the living
    // survey, not a disclaimer stamped on everything.
    useSelection.getState().setSelection({
      kind: "site",
      title: "Somewhere else",
      longitude: -122.138,
      latitude: 47.644,
      height: 120,
      terrainHeight: 118.5,
      siteId: "some-other-site",
      at: Date.now(),
    });
    rerender(wrap(<InspectorPanel />));
    await waitFor(() => expect(screen.queryByTestId("inspector-living")).not.toBeInTheDocument());
    vi.restoreAllMocks();
  });

  it("is still reachable when the click lands on the ground, because splats cannot be picked", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useLiving.getState().setStatus(livingStatus(true));
    // A click on a swaying tree picks nothing: the splat primitive skips the pick pass, so the
    // ray reaches the terrain behind it and the selection carries no site at all.
    useSites.getState().setActiveSite("builtin-demo-site");
    useSelection.getState().setSelection({
      kind: "ground",
      title: "Location",
      longitude: -122.138,
      latitude: 47.644,
      height: 120,
      terrainHeight: 118.5,
      at: Date.now(),
    });
    useUi.getState().setInspectorOpen(true);
    render(wrap(<InspectorPanel />));

    // The section still appears, and names the site it is about so it cannot be mistaken for a
    // statement about the patch of ground that was clicked.
    const section = await screen.findByTestId("inspector-living");
    expect(section).toHaveTextContent("Measured and simulated · Cesium Gaussian splat demo");
    expect(screen.getByTestId("inspector-motion")).toHaveTextContent(RIG_NOTE);
    vi.restoreAllMocks();
  });
});

describe("SettingsSheet: the wind control", () => {
  it("starts calm, opens at the default strength, and never calls the wind a speed", async () => {
    useUi.setState({ settingsOpen: true });
    render(wrap(<SettingsSheet />));

    const wind = screen.getByRole("switch", { name: "Simulated wind" });
    expect(wind).toHaveAttribute("aria-checked", "false");
    expect(screen.queryByRole("slider", { name: "Wind strength" })).not.toBeInTheDocument();

    await userEvent.click(wind);
    expect(useLiving.getState().wind.strength).toBe(DEFAULT_WIND_STRENGTH);
    expect(await screen.findByRole("slider", { name: "Wind strength" })).toBeInTheDocument();

    // The two claims this control must never make: that the scale is a wind speed, and that
    // the bearing says where the wind comes from.
    const living = screen.getByRole("region", { name: "Living survey" });
    expect(living).toHaveTextContent("not a wind speed");
    expect(living).toHaveTextContent("Blowing towards");
    expect(living).not.toHaveTextContent(/wind from/i);
    // "Explore speed" elsewhere in the sheet is a real m/s; nothing in this section may be.
    expect(living.textContent).not.toMatch(/m\/s/);
    // Nor may it call the geometry measured: every rigged site today is a procedural fixture,
    // and which sites are captures is the Inspector's question to answer.
    expect(living.textContent).not.toMatch(/measured|surveyed|scanned/i);
  });

  it("reports the displacement in metres and no longer quotes staleness in splat radii", () => {
    useLiving.getState().setStatus(livingStatus(true));
    useLiving.getState().setWind({ strength: DEFAULT_WIND_STRENGTH });
    useUi.setState({ settingsOpen: true });
    render(wrap(<SettingsSheet />));

    const living = screen.getByRole("region", { name: "Living survey" });
    // Metres are a property of this rig at this wind, and belong here.
    expect(living).toHaveTextContent("by up to 0.20 m");
    // "Splat radii" is not. `sortStaleness` divides by REFERENCE_GAUSSIAN_SCALE_M — a different
    // capture's median, five times finer than the fixture's own — so the ratio reads as a
    // measurement of what is on screen while being nothing of the kind. It is printed only in
    // the developer panel, with its yardstick beside it.
    expect(living.textContent).not.toMatch(/radii|staleness/i);
  });

  it("is held calm, and disabled, while reduced motion is on", () => {
    useSettings.getState().set({ reducedMotion: true });
    useLiving.getState().setWind({ strength: DEFAULT_WIND_STRENGTH });
    useUi.setState({ settingsOpen: true });
    render(wrap(<SettingsSheet />));

    const wind = screen.getByRole("switch", { name: "Simulated wind" });
    expect(wind).toBeDisabled();
    expect(screen.getByTestId("settings-sheet")).toHaveTextContent("Held calm by reduced motion");
    // The person's choice is kept; only the scene is forced calm (by SceneBridge).
    expect(useLiving.getState().wind.strength).toBe(DEFAULT_WIND_STRENGTH);
    expect(screen.queryByRole("slider", { name: "Wind strength" })).not.toBeInTheDocument();
  });
});
