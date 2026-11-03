import { create } from "zustand";

import type { WeatherWind } from "@twin/world";

import type { WeatherWindTier } from "@/lib/weather/weatherWindPlan";

/**
 * Weather wind (docs/LIVING_WEATHER_WIND.md): the active scan's plants move with the hour's
 * wind, **Simulated**. Shown only in a build with `VITE_LIVING_WEATHER=1`, and off at the start
 * of every session, like the Living Survey's wind (state/living.ts): the first thing a person
 * sees is the measurement. Never persisted.
 */
export interface WeatherWindState {
  /** The viewer's switch. */
  enabled: boolean;
  /** `live`: the site's hourly weather at the map's time (Open-Meteo); `manual`: the sliders. */
  mode: "live" | "manual";
  manual: WeatherWind;
  /** The hour the live feed returned; null while loading, on failure, or off. */
  live: WeatherWind | null;
  /** What the live feed says (place and hour, or why there is no weather). */
  feedNote: string;
  /** The scan being moved, and how (null: none). */
  scan: { assetId: string; name: string; tier: WeatherWindTier | null } | null;
  /** Why nothing moves, when nothing does. */
  reason: string;
  set: (patch: Partial<Omit<WeatherWindState, "set">>) => void;
}

/** A light breeze at Minnetonka, the research site; the location is replaced by the site's. */
export const DEFAULT_MANUAL_WIND: WeatherWind = {
  meanMps: 2.5,
  gustMps: 4,
  directionDeg: 270,
  latitude: 44.944403,
  longitude: -93.425903,
};

export const useWeatherWind = create<WeatherWindState>()((set) => ({
  enabled: false,
  mode: "live",
  manual: DEFAULT_MANUAL_WIND,
  live: null,
  feedNote: "",
  scan: null,
  reason: "",
  set: (patch) => set(patch),
}));

/** The wind input in force, or null (live with no hour yet). */
export function weatherInput(s: Pick<WeatherWindState, "mode" | "manual" | "live">) {
  return s.mode === "manual" ? s.manual : s.live;
}

/** Whether plants are moving now: what the Simulated badge hangs off. */
export function weatherWindMoving(
  s: Pick<WeatherWindState, "enabled" | "mode" | "manual" | "live" | "scan">,
  blocked: boolean,
): boolean {
  const tier = s.scan?.tier;
  return (
    s.enabled &&
    !blocked &&
    weatherInput(s) !== null &&
    (tier === "A" || tier === "B" || tier === "C")
  );
}
