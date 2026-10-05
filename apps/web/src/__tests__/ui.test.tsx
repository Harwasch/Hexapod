import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MotionConfig } from "motion/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, onTestFinished, vi } from "vitest";

import { GlassTooltipProvider } from "@twin/ui";

import type { Site, SiteAsset } from "@twin/contracts";

import { api } from "@/api/client";
import { env } from "@/app/env";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { sceneRegistry } from "@/cesium/SceneContext";
import { InspectorPanel } from "@/features/inspector/InspectorPanel";
import { DevPanel } from "@/features/dev/DevPanel";
import { SimulatedBadge } from "@/features/living/SimulatedBadge";
import { LayersPanel } from "@/features/layers/LayersPanel";
import { SettingsSheet } from "@/features/settings/SettingsSheet";
import { OnboardingCard } from "@/features/onboarding/OnboardingCard";
import { DevReadouts } from "@/features/shell/DevReadouts";
import { MapCorner } from "@/features/shell/MapCorner";
import { StatusLine } from "@/features/mission/StatusLine";
import { ToolRail } from "@/features/shell/ToolRail";
import { PHONE_MEDIA } from "@/lib/media";
import {
  DEFAULT_WIND_STRENGTH,
  LIVING_SURVEY_IDLE,
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
    motionEvidence: null,
    maxDisplacementM: 0.197,
    sortStaleness: 9.8,
    motionPath: "gpu",
    cpuReason: null,
    motionMs: null,
    applyMs: null,
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

describe("Developer readouts + Onboarding", () => {
  it("are off by default and render altitude in the selected unit system when on", () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useViewer.getState().setCamera({
      ...useViewer.getState().camera,
      altitude: 1500,
      scaleBand: "city",
      metersPerPixel: 2,
    });
    const { rerender } = render(wrap(<DevReadouts />));
    // An operator reads the map, not its telemetry: nothing until Settings › Advanced asks.
    expect(screen.queryByTestId("status-bar")).not.toBeInTheDocument();
    useSettings.getState().set({ devReadouts: true });
    rerender(wrap(<DevReadouts />));
    expect(screen.getByTestId("status-altitude")).toHaveTextContent("1.5 km");
    useSettings.getState().set({ units: "imperial" });
    rerender(wrap(<DevReadouts />));
    expect(screen.getByTestId("status-altitude")).toHaveTextContent("ft");
    vi.restoreAllMocks();
  });

  it("name the splat renderer's API, its latest frame rate, and why WebGPU fell back", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useSettings.getState().set({ devReadouts: true, splatRenderer: "playcanvas-webgpu" });
    useSites.setState({ activeSiteId: "site", representation: { site: "gaussian-splat" } });
    const status = {
      kind: "playcanvas-webgpu",
      active: true,
      api: "webgl2",
      notice: "WebGPU device lost: GPU process restarted",
      meter: { fps: 31.6, p95Ms: 48.2, cpuMs: 4.04, frames: 40, live: false },
    };
    sceneRegistry.set({ scanRendererStatus: status } as unknown as CesiumSceneManager);
    render(wrap(<DevReadouts />));
    expect(await screen.findByTestId("status-splat-renderer")).toHaveTextContent(
      "PlayCanvas · WebGL2 (WebGPU unavailable)",
    );
    expect(screen.getByTestId("status-splat-meter")).toHaveTextContent(
      "last move 32 fps · p95 48 ms · draw 4.0 ms",
    );
    expect(screen.getByTestId("status-splat-notice")).toHaveTextContent(
      "WebGPU device lost: GPU process restarted",
    );
    sceneRegistry.set(null);
    useSites.setState({ activeSiteId: null, representation: {} });
    useSettings.getState().reset();
    vi.restoreAllMocks();
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

  it("onboarding goes the moment it is dismissed under reduced motion, not a frame later", () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useViewer.getState().setStatus("ready");
    // As App.tsx sets it when the app's own switch is on.
    render(
      wrap(
        <MotionConfig reducedMotion="always">
          <OnboardingCard />
        </MotionConfig>,
      ),
    );
    // No entrance either: it is there at once.
    expect(screen.getByTestId("onboarding")).toBeInTheDocument();
    // Gone in the click's own render (fireEvent is synchronous, and wrapped in act). An exit
    // animation waits for frames, and a globe flying off on a slow GPU draws one every few
    // seconds.
    fireEvent.click(screen.getByTestId("onboarding-explore"));
    expect(screen.queryByTestId("onboarding")).not.toBeInTheDocument();
    expect(useSettings.getState().onboardingDismissed).toBe(true);
  });
});

describe("Setup and connection notices", () => {
  it("lets the evaluation-token hint be dismissed for good, and keeps a rejected token loud", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useViewer.getState().setTokenState("default");
    // The hint is for whoever deployed this build, so it rides the developer readouts.
    const { unmount: unmountQuiet } = render(wrap(<DevReadouts />));
    expect(screen.queryByTestId("notice-default-token")).not.toBeInTheDocument();
    unmountQuiet();
    useSettings.getState().set({ devReadouts: true });
    const { unmount } = render(wrap(<DevReadouts />));
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
    const remount = render(wrap(<DevReadouts />));
    await waitFor(() =>
      expect(screen.queryByTestId("notice-default-token")).not.toBeInTheDocument(),
    );
    remount.unmount();

    // A rejected token is a fault, not a setup note: it is on the status line for everyone,
    // readouts or not, and never dismissible.
    useSettings.getState().set({ devReadouts: false });
    useViewer.getState().setTokenState("invalid");
    render(wrap(<StatusLine />));
    expect(await screen.findByTestId("notice-token")).toHaveTextContent("Map key rejected");
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

/** A site whose splat asset records how it was placed, as the worker registers one. */
function placedSite(provenance: NonNullable<SiteAsset["provenance"]>): Site {
  const asset = {
    id: "placed-splat",
    siteId: "placed-site",
    provider: "3d-tiles-url",
    name: "Capture splat",
    representation: "gaussian-splat",
    source: { type: "3d-tiles-url", url: "https://example.invalid/splat/tileset.json" },
    footprint: null,
    observedAt: null,
    validFrom: null,
    validTo: null,
    resolution: null,
    crs: null,
    license: null,
    attribution: [],
    provenance,
    renderConfig: { clipsWorld: true, clipFootprint: "catalog", heightOffsetM: 0 },
    defaultVisible: true,
    createdAt: "2026-01-01T00:00:00Z",
    updatedAt: "2026-01-01T00:00:00Z",
  } as unknown as SiteAsset;
  return {
    id: "placed-site",
    slug: "placed",
    name: "Placed capture",
    description: null,
    boundary: { type: "Polygon", coordinates: [] },
    centroid: { longitude: 0, latitude: 0, height: 0 },
    areaM2: 1,
    thumbnailUrl: null,
    metadata: {},
    attribution: [],
    license: null,
    assets: [asset],
    cameraBookmarks: [],
    createdAt: "2026-01-01T00:00:00Z",
    updatedAt: "2026-01-01T00:00:00Z",
  } as unknown as Site;
}

async function placementRow(provenance: NonNullable<SiteAsset["provenance"]>): Promise<string> {
  vi.spyOn(api, "GET").mockResolvedValue({
    data: placedSite(provenance),
    error: undefined,
    response: new Response(null, { status: 200 }),
  });
  // Not animating, and no rigged sites: the placement row is about a capture's recorded
  // georeference, and nothing about it should depend on whether the wind is blowing.
  useLiving.getState().setStatus(LIVING_SURVEY_IDLE);
  useSelection.getState().setSelection({
    kind: "site",
    title: "Placed capture",
    longitude: -82.6966,
    latitude: 28.0389,
    height: 22.5,
    terrainHeight: 20,
    siteId: "placed-site",
    at: Date.now(),
  });
  useUi.getState().setInspectorOpen(true);
  render(wrap(<InspectorPanel />));
  const row = await screen.findByTestId("inspector-placement");
  return row.textContent ?? "";
}

describe("InspectorPanel: how the capture was placed", () => {
  it("does not let a hand placement look like an alignment", async () => {
    const byHand = await placementRow({
      georefMethod: "manual",
      scaleSource: "unresolved",
      uncertaintyM: 10,
    });
    expect(byHand).toContain("Placed by hand");
    expect(byHand).toContain("10");
    expect(byHand).toContain("scale unresolved");
    // The one that matters most is said at the top of the panel too: a reconstruction with
    // no metric scale is one whose measurements mean nothing.
    expect(screen.getAllByText("Scale unresolved").length).toBeGreaterThan(0);
    vi.restoreAllMocks();
  });

  it("says so, differently, when the capture was aligned to EXIF GPS", async () => {
    const aligned = await placementRow({
      georefMethod: "exif-gps",
      scaleSource: "exif-gps",
      uncertaintyM: 5,
    });
    expect(aligned).toContain("Aligned to EXIF GPS");
    expect(aligned).toContain("metric scale from EXIF GPS");
    expect(screen.queryByText("Scale unresolved")).not.toBeInTheDocument();
    vi.restoreAllMocks();
  });

  it("calls an estimated scale an estimate, softly, rather than unresolved", async () => {
    const estimated = await placementRow({
      georefMethod: "exif-gps",
      scaleSource: "camera-height-estimate",
      uncertaintyM: 10,
      scaleUncertaintyPct: 22.4,
    });
    expect(estimated).toContain("scale estimated from camera height (±22%)");
    expect(screen.getAllByText("Scale estimated").length).toBeGreaterThan(0);
    expect(screen.queryByText("Scale unresolved")).not.toBeInTheDocument();
    vi.restoreAllMocks();
  });
});

describe("InspectorPanel: observed and simulated", () => {
  it("separates what was recorded from what is modelled, for a living site", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    // The offline demo site has a splat only when one is configured.
    const configured = env.defaultSplatAssetId;
    env.defaultSplatAssetId = 4547222;
    onTestFinished(() => {
      env.defaultSplatAssetId = configured;
    });
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
    expect(screen.getByTestId("inspector-geometry")).toHaveTextContent("VITE_DEFAULT_*_ASSET_ID");
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

  it("offers Motion on GPU, on by default, and shows each site's path and cost", async () => {
    useLiving.getState().setStatus(livingStatus(true, { motionMs: 0.42, applyMs: 0.05 }));
    useViewer.setState((s) => ({
      performance: { ...s.performance, rendering: true, fps: 58.6, frameTimeMs: 17.1 },
    }));
    useUi.setState({ settingsOpen: true });
    const { rerender } = render(wrap(<SettingsSheet />));

    const gpu = screen.getByRole("switch", { name: "Motion on GPU" });
    expect(gpu).toHaveAttribute("aria-checked", "true");
    const readout = screen.getByTestId("living-motion-readout");
    expect(screen.getByTestId("living-motion-site")).toHaveTextContent(
      "synthetic-tree: GPU · motion 0.42 ms/frame (write 0.05)",
    );
    expect(screen.getByTestId("living-frame-time")).toHaveTextContent(
      "Scene frame 17.1 ms (59 fps)",
    );
    // Still nothing in this section that reads as a measurement or a wind speed.
    const living = screen.getByRole("region", { name: "Living survey" });
    expect(living.textContent).not.toMatch(/measured|surveyed|scanned|m\/s/i);

    await userEvent.click(gpu);
    expect(useSettings.getState().livingGpuMotion).toBe(false);

    // What the scene then reports: the CPU path, and why.
    useLiving.getState().setStatus(
      livingStatus(true, {
        motionPath: "cpu",
        cpuReason: "switched-off",
        motionMs: null,
        applyMs: null,
      }),
    );
    rerender(wrap(<SettingsSheet />));
    expect(readout).toHaveTextContent(
      "synthetic-tree: CPU (switched off) · no animated frames yet",
    );
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

describe("MapCorner", () => {
  /** `matchMedia` with a phone switch, and the change events a rotation would fire. */
  function fakeMedia(phone: { matches: boolean }) {
    const listeners = new Set<() => void>();
    vi.spyOn(window, "matchMedia").mockImplementation(
      (query: string) =>
        ({
          get matches() {
            return query === PHONE_MEDIA && phone.matches;
          },
          media: query,
          addEventListener: (_: string, listener: () => void) => listeners.add(listener),
          removeEventListener: (_: string, listener: () => void) => listeners.delete(listener),
        }) as unknown as MediaQueryList,
    );
    return (matches: boolean) =>
      act(() => {
        phone.matches = matches;
        for (const listener of listeners) listener();
      });
  }

  it("moves Cesium's credit container out of the pill into a strip of its own on a phone", () => {
    // The widget's own container, where CesiumSceneManager leaves it.
    const viewport = document.createElement("div");
    const credits = document.createElement("div");
    credits.className = "cesium-viewer-bottom";
    viewport.appendChild(credits);
    document.body.appendChild(viewport);
    sceneRegistry.set({ viewer: { creditContainer: credits } } as unknown as CesiumSceneManager);
    const rotate = fakeMedia({ matches: false });

    const { unmount } = render(wrap(<MapCorner />));
    // Wide: in the pill, beside the compass.
    expect(screen.getByTestId("map-corner")).toContainElement(credits);
    expect(credits.parentElement).toBe(screen.getByTestId("credits"));

    // Phone: out of the pill into the strip, the same element (moved, never copied), and no
    // button to fold it behind — the providers' terms want it on screen.
    rotate(true);
    const strip = screen.getByTestId("credits");
    expect(strip).toHaveClass("credit-slot--strip");
    expect(screen.getByTestId("map-corner")).not.toContainElement(strip);
    expect(credits.parentElement).toBe(strip);
    expect(document.querySelectorAll(".cesium-viewer-bottom")).toHaveLength(1);
    expect(screen.queryByRole("button", { name: /credits/i })).not.toBeInTheDocument();

    rotate(false);
    expect(screen.getByTestId("map-corner")).toContainElement(credits);

    // Unmounted, the container goes home so the viewer can tear down what it built.
    unmount();
    expect(credits.parentElement).toBe(viewport);
    viewport.remove();
    sceneRegistry.set(null);
    vi.restoreAllMocks();
  });
});
