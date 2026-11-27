import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type * as EnvModule from "@/app/env";
import { env } from "@/app/env";
import { HIDDEN_AS_CAPTURED, useAsCaptured } from "@/state/asCaptured";
import { useSettings } from "@/state/settings";
import { useWeatherWind } from "@/state/weatherWind";

import { WeatherWindSection } from "./WeatherWindSection";

vi.mock("@/app/env", async (original) => {
  const actual = await original<typeof EnvModule>();
  return { ...actual, env: { ...actual.env, livingWeather: true } };
});

beforeEach(() => {
  env.livingWeather = true;
  useAsCaptured.setState({ on: false });
  useSettings.setState({ reducedMotion: false });
  useWeatherWind.setState({ enabled: false, mode: "live", scan: null, reason: "" });
});
afterEach(cleanup);

describe("Settings › Weather wind", () => {
  it("is absent from a build without VITE_LIVING_WEATHER", () => {
    env.livingWeather = false;
    render(<WeatherWindSection />);
    expect(screen.queryByTestId("weather-wind")).toBeNull();
  });

  it("starts off and says Simulated", () => {
    render(<WeatherWindSection />);
    expect(screen.getByText("Weather wind · Simulated")).toBeVisible();
    expect(screen.getByRole("switch", { name: "Weather wind · Simulated" })).not.toBeChecked();
    expect(screen.queryByLabelText("Weather wind source")).toBeNull();
  });

  it("names the scan and its tier while on", () => {
    useWeatherWind.setState({ enabled: true, scan: { assetId: "a", name: "Grove", tier: "A" } });
    render(<WeatherWindSection />);
    expect(screen.getByTestId("weather-wind-status")).toHaveTextContent(
      "Grove · Tier A · whole plants sway",
    );
  });

  it("is held off and disabled under As captured", () => {
    useAsCaptured.setState({ on: true });
    useWeatherWind.setState({ enabled: true });
    render(<WeatherWindSection />);
    expect(screen.getByRole("switch", { name: "Weather wind · Simulated" })).toBeDisabled();
    expect(screen.getByTestId("weather-wind-status")).toHaveTextContent(HIDDEN_AS_CAPTURED);
  });
});
