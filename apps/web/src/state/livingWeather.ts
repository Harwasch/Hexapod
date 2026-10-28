import { create } from "zustand";
import type { WeatherWind } from "@twin/world";
import { useLivingLab } from "./livingLab";
/** Lab only. A single scene field; query configuration also invalidates fixed-clock pose cache. */
export function setLabWeather(wind: WeatherWind | null, articulationUrl?: string): void {
  const lab = useLivingLab.getState();
  const explicit = lab.articulationUrl.includes("tier-a=1") ? "" : lab.articulationUrl;
  const base = articulationUrl ?? (explicit || lab.weatherSidecarUrl);
  if (base && wind) {
    const url = new URL(base, location.href);
    url.searchParams.delete("breeze");
    url.searchParams.set("weather", JSON.stringify(wind));
    lab.set({
      motion: "articulated",
      strength: 1,
      articulationUrl: url.toString(),
      weather: wind,
      water: false,
    });
  } else
    lab.set({
      motion: wind ? "gust" : "off",
      weather: wind,
      water: false,
      ...(wind ? { strength: 1 } : {}),
    });
}
interface WeatherState {
  mode: "manual" | "live";
  wind: WeatherWind;
  note: string;
  set: (patch: Partial<Omit<WeatherState, "set">>) => void;
}
export const useLivingWeather = create<WeatherState>()((set) => ({
  mode: "manual",
  wind: { meanMps: 2.5, gustMps: 4, directionDeg: 270, latitude: 44.944403, longitude: -93.425903 },
  note: "Simulated · manual weather inputs",
  set: (patch) => set(patch),
}));
