import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { JulianDate } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LivingWeatherControls } from "./LivingWeatherControls";
import { weatherFeed, type WeatherResult } from "@/lib/weather/openMeteo";
import { useLivingLab } from "@/state/livingLab";
import { useLivingWeather } from "@/state/livingWeather";
const { source, scene } = vi.hoisted(() => {
  const source = {
    site: null as {
      name: string;
      centroid: { latitude: number; longitude: number };
      assets: [];
    } | null,
    time: null as unknown,
    ticks: [] as (() => void)[],
  };
  const scene = {
    viewer: {
      clock: {
        get currentTime() {
          return source.time;
        },
        onTick: {
          addEventListener: (fn: () => void) => {
            source.ticks.push(fn);
            return () => {
              source.ticks = source.ticks.filter((f) => f !== fn);
            };
          },
        },
      },
    },
  };
  return { source, scene };
});
vi.mock("@/api/queries", () => ({ useSite: () => ({ data: source.site }) }));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));
vi.mock("@/lib/weather/openMeteo", () => ({ WEATHER_FLAG: true, weatherFeed: vi.fn() }));
const hour = {
  meanMps: 8.51,
  gustMps: 15.4,
  directionDeg: 97,
  latitude: 44.944403,
  longitude: -93.425903,
  time: "2024-11-19T04:00Z",
};
beforeEach(() => {
  source.site = null;
  source.ticks = [];
  source.time = JulianDate.fromIso8601(hour.time);
  vi.mocked(weatherFeed).mockReset();
  useLivingLab.setState({
    weather: null,
    articulationUrl: "",
    weatherSidecarUrl: "",
    motion: "off",
    strength: 1,
  });
  useLivingWeather.setState({ mode: "live", note: "", wind: hour });
});
afterEach(() => {
  cleanup();
});
describe("live weather follows place and map hour", () => {
  it("stays off without a site and does not rewrite state every clock tick", () => {
    render(<LivingWeatherControls />);
    expect(screen.getByText("No weather, wind off · select a site")).toBeVisible();
    const changed = vi.fn(),
      remove = useLivingWeather.subscribe(changed);
    act(() => {
      for (let i = 0; i < 10; i++) for (const tick of source.ticks) tick();
    });
    expect(changed).not.toHaveBeenCalled();
    expect(weatherFeed).not.toHaveBeenCalled();
    remove();
  });
  it("ignores an old response when the map advances to another hour", async () => {
    source.site = {
      name: "Minnetonka",
      centroid: { latitude: hour.latitude, longitude: hour.longitude },
      assets: [],
    };
    const pending: ((r: WeatherResult) => void)[] = [];
    vi.mocked(weatherFeed).mockImplementation(
      () => new Promise((resolve) => pending.push(resolve)),
    );
    render(<LivingWeatherControls />);
    act(() => {
      source.time = JulianDate.fromIso8601("2024-11-19T05:00Z");
      for (const tick of source.ticks) tick();
    });
    expect(weatherFeed).toHaveBeenCalledTimes(2);
    await act(async () => {
      pending[0]?.({ wind: hour, note: "old hour" });
      await Promise.resolve();
    });
    expect(useLivingLab.getState().weather).toBeNull();
    const next = { ...hour, time: "2024-11-19T05:00Z", meanMps: 10 };
    await act(async () => {
      pending[1]?.({ wind: next, note: "Simulated from weather at Minnetonka, 2024-11-19T05:00Z" });
      await Promise.resolve();
    });
    await waitFor(() => {
      expect(useLivingLab.getState().weather?.meanMps).toBe(10);
    });
    expect(
      screen.getByText(/Simulated from weather at Minnetonka, 2024-11-19T05:00Z/),
    ).toBeVisible();
  });
});
