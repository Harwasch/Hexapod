import { act, cleanup, render, waitFor } from "@testing-library/react";
import { JulianDate } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { weatherFeed } from "@/lib/weather/openMeteo";
import { forgetRigProbes } from "@/lib/weather/rigProbe";
import { useAsCaptured } from "@/state/asCaptured";
import { useInstances } from "@/state/instances";
import { useLivingLab } from "@/state/livingLab";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";
import { DEFAULT_MANUAL_WIND, useWeatherWind } from "@/state/weatherWind";

import { WeatherWindDriver } from "./WeatherWindDriver";

const { source, scene } = vi.hoisted(() => {
  const source = { site: null as unknown, time: null as unknown };
  const scene = {
    viewer: {
      clock: {
        get currentTime() {
          return source.time;
        },
        onTick: { addEventListener: () => () => undefined },
      },
    },
  };
  return { source, scene };
});
vi.mock("@/api/queries", () => ({ useSite: () => ({ data: source.site }) }));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));
vi.mock("@/lib/weather/openMeteo", () => ({ WEATHER_FLAG: true, weatherFeed: vi.fn() }));

const SITE = {
  id: "s",
  name: "Oak Bench",
  centroid: { latitude: 45, longitude: -93 },
  assets: [
    {
      id: "a",
      name: "Grove",
      representation: "gaussian-splat",
      renderConfig: {},
      source: { type: "3d-tiles-url", url: "https://tiles.example/grove/tileset.json" },
    },
  ],
};
const head = vi.fn<typeof fetch>();

beforeEach(() => {
  forgetRigProbes();
  head.mockReset();
  head.mockResolvedValue(new Response(null, { status: 404 }));
  vi.stubGlobal("fetch", head);
  vi.mocked(weatherFeed).mockReset();
  source.site = SITE;
  source.time = JulianDate.fromIso8601("2024-11-19T04:00Z");
  useSites.setState({ activeSiteId: "s" });
  useInstances.setState({ assets: { a: {} as never } });
  useAsCaptured.setState({ on: false });
  useSettings.setState({ reducedMotion: false, splatRenderer: "playcanvas" });
  useLivingLab.setState({
    weather: null,
    motion: "off",
    articulationUrl: "",
    articulationStatus: "No articulation loaded",
    weatherSidecarUrl: "",
  });
  useWeatherWind.setState({
    enabled: false,
    mode: "manual",
    manual: DEFAULT_MANUAL_WIND,
    live: null,
    scan: null,
    reason: "",
  });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("weather wind switch", () => {
  it("does nothing while off: no probe, no request, the renderer untouched", () => {
    render(<WeatherWindDriver />);
    expect(head).not.toHaveBeenCalled();
    expect(weatherFeed).not.toHaveBeenCalled();
    expect(useLivingLab.getState().motion).toBe("off");
  });

  it("moves a segmented scan at Tier A with the site's own location", async () => {
    render(<WeatherWindDriver />);
    act(() => useWeatherWind.getState().set({ enabled: true }));
    await waitFor(() => expect(useLivingLab.getState().motion).toBe("articulated"));
    const lab = useLivingLab.getState();
    const url = new URL(lab.articulationUrl);
    expect(url.pathname).toBe("/grove/instances.json");
    expect(url.searchParams.get("tier-a")).toBe("1");
    expect(JSON.parse(url.searchParams.get("weather") ?? "")).toMatchObject({
      meanMps: 2.5,
      latitude: 45,
      longitude: -93,
    });
    expect(useWeatherWind.getState().scan).toEqual({ assetId: "a", name: "Grove", tier: null });
  });

  it("uses a baked rig beside the tiles when one is published", async () => {
    head.mockResolvedValue(
      new Response(null, { status: 200, headers: { "content-type": "application/json" } }),
    );
    render(<WeatherWindDriver />);
    act(() => useWeatherWind.getState().set({ enabled: true }));
    await waitFor(() =>
      expect(useLivingLab.getState().articulationUrl).toContain("/grove/articulation.json"),
    );
  });

  it("does not take a host's HTML fallback page for a rig", async () => {
    head.mockResolvedValue(
      new Response(null, { status: 200, headers: { "content-type": "text/html" } }),
    );
    render(<WeatherWindDriver />);
    act(() => useWeatherWind.getState().set({ enabled: true }));
    await waitFor(() =>
      expect(useLivingLab.getState().articulationUrl).toContain("/grove/instances.json"),
    );
  });

  it("As captured turns it fully off, and so does reduced motion", async () => {
    render(<WeatherWindDriver />);
    act(() => useWeatherWind.getState().set({ enabled: true }));
    await waitFor(() => expect(useLivingLab.getState().motion).toBe("articulated"));
    act(() => useAsCaptured.setState({ on: true }));
    expect(useLivingLab.getState().motion).toBe("off");
    expect(useLivingLab.getState().weather).toBeNull();
    act(() => useAsCaptured.setState({ on: false }));
    await waitFor(() => expect(useLivingLab.getState().motion).toBe("articulated"));
    act(() => {
      useSettings.setState({ reducedMotion: true });
    });
    expect(useLivingLab.getState().motion).toBe("off");
  });

  it("asks for no weather under As captured even in live mode", () => {
    useAsCaptured.setState({ on: true });
    useWeatherWind.setState({ enabled: true, mode: "live" });
    render(<WeatherWindDriver />);
    expect(weatherFeed).not.toHaveBeenCalled();
    expect(head).not.toHaveBeenCalled();
  });

  it("reads the tier once the renderer loads, and lets go when there is no vegetation", async () => {
    render(<WeatherWindDriver />);
    act(() => useWeatherWind.getState().set({ enabled: true }));
    await waitFor(() => expect(useLivingLab.getState().motion).toBe("articulated"));
    const tiers = await import("@/cesium/scanView/livingArticulation");
    const tier = vi.spyOn(tiers, "weatherArticulationTier").mockReturnValue("off");
    act(() => useLivingLab.getState().set({ articulationStatus: "Simulated · 1 nodes" }));
    expect(useWeatherWind.getState().scan?.tier).toBe("off");
    expect(useLivingLab.getState().motion).toBe("off");
    // A new wind does not load the same empty scan again.
    act(() =>
      useWeatherWind
        .getState()
        .set({ manual: { ...DEFAULT_MANUAL_WIND, meanMps: 9, gustMps: 14 } }),
    );
    expect(useLivingLab.getState().motion).toBe("off");
    tier.mockRestore();
  });

  it("says why when the renderer cannot load the scan's data", async () => {
    render(<WeatherWindDriver />);
    act(() => useWeatherWind.getState().set({ enabled: true }));
    await waitFor(() => expect(useLivingLab.getState().motion).toBe("articulated"));
    act(() => useLivingLab.getState().set({ articulationStatus: "Error: Articulation HTTP 503" }));
    expect(useLivingLab.getState().motion).toBe("off");
    expect(useWeatherWind.getState().reason).toBe("wind off · Error: Articulation HTTP 503");
  });

  it("live mode follows the feed's hour and stays still until it arrives", async () => {
    const hour = {
      meanMps: 8.51,
      gustMps: 15.4,
      directionDeg: 97,
      latitude: 45,
      longitude: -93,
      time: "2024-11-19T04:00Z",
    };
    vi.mocked(weatherFeed).mockResolvedValue({
      wind: hour,
      note: "Simulated from weather at 45.00000, -93.00000",
    });
    useWeatherWind.setState({ mode: "live" });
    render(<WeatherWindDriver />);
    act(() => useWeatherWind.getState().set({ enabled: true }));
    expect(weatherFeed).toHaveBeenCalledWith(45, -93, new Date("2024-11-19T04:00Z"));
    await waitFor(() => expect(useLivingLab.getState().weather).toEqual(hour));
    expect(useWeatherWind.getState().feedNote).toBe("Simulated from weather at Oak Bench");
  });
});
