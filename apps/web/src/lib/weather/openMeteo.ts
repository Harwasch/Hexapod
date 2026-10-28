import type { WeatherWind } from "@twin/world";
export interface WeatherHour extends WeatherWind {
  gustMps: number;
  time: string;
}
export interface WeatherResult {
  wind: WeatherHour | null;
  note: string;
}
export const WEATHER_FLAG = import.meta.env.VITE_LIVING_WEATHER === "1";
/** All hours UTC; never reinterpret server times in the browser's local zone. */
export function parseWeather(value: unknown, latitude: number, longitude: number): WeatherHour[] {
  const d = value as {
    hourly?: Record<string, unknown[]>;
    hourly_units?: Record<string, string>;
    utc_offset_seconds?: number;
  };
  const h = d?.hourly,
    units = d?.hourly_units;
  if (
    !h ||
    d.utc_offset_seconds !== 0 ||
    units?.wind_speed_10m !== "m/s" ||
    units.wind_gusts_10m !== "m/s" ||
    units.wind_direction_10m !== "°"
  )
    throw new Error("Unexpected weather units/timezone");
  const times = h.time;
  if (
    !Array.isArray(times) ||
    ![h.wind_speed_10m, h.wind_gusts_10m, h.wind_direction_10m].every(
      (a) => Array.isArray(a) && a.length === times.length,
    )
  )
    throw new Error("Incomplete hourly weather");
  return times.map((time, i) => {
    const meanMps = h.wind_speed_10m?.[i],
      gustMps = h.wind_gusts_10m?.[i],
      directionDeg = h.wind_direction_10m?.[i];
    if (
      typeof time !== "string" ||
      !/^\d{4}-\d\d-\d\dT\d\d:00$/.test(time) ||
      !Number.isFinite(Date.parse(time + "Z")) ||
      typeof meanMps !== "number" ||
      typeof gustMps !== "number" ||
      typeof directionDeg !== "number" ||
      ![meanMps, gustMps, directionDeg].every(Number.isFinite) ||
      meanMps < 0 ||
      gustMps < 0 ||
      directionDeg < 0 ||
      directionDeg > 360
    )
      throw new Error("Missing/invalid weather hour");
    return {
      time: time + "Z",
      meanMps,
      gustMps: Math.max(meanMps, gustMps),
      directionDeg,
      latitude,
      longitude,
    };
  });
}
/** Injected fetch and clock keep tests offline. Bounded in-memory cache, requests deduplicated.
 * Forecast expires after 15 minutes; past daily archives after 24 hours.
 */
export function createWeatherFeed(get: typeof fetch = fetch, now = () => Date.now()) {
  const cache = new Map<string, { expires: number; request: Promise<WeatherHour[]> }>();
  return async (
    latitude: number,
    longitude: number,
    at: Date,
    enabled = WEATHER_FLAG,
  ): Promise<WeatherResult> => {
    if (!enabled)
      return { wind: null, note: "No weather, wind off · enable VITE_LIVING_WEATHER=1" };
    try {
      if (
        !Number.isFinite(latitude) ||
        Math.abs(latitude) > 90 ||
        !Number.isFinite(longitude) ||
        Math.abs(longitude) > 180 ||
        !Number.isFinite(at.getTime())
      )
        throw new Error("Invalid place/time");
      const day = at.toISOString().slice(0, 10),
        hour = at.toISOString().slice(0, 13) + ":00Z";
      const past = at.getTime() < now() - 5 * 86400000;
      const url = new URL(
        past
          ? "https://archive-api.open-meteo.com/v1/archive"
          : "https://api.open-meteo.com/v1/forecast",
      );
      url.search = new URLSearchParams({
        latitude: String(latitude),
        longitude: String(longitude),
        start_date: day,
        end_date: day,
        hourly: "wind_speed_10m,wind_gusts_10m,wind_direction_10m",
        wind_speed_unit: "ms",
        timezone: "UTC",
      }).toString();
      const key = url.toString();
      let entry = cache.get(key);
      if (!entry || entry.expires <= now()) {
        if (cache.size >= 64) cache.delete(cache.keys().next().value ?? "");
        const request = (async () => {
          const r = await get(key, { signal: AbortSignal.timeout(12000) });
          if (!r.ok) throw new Error(`Weather HTTP ${r.status}`);
          return parseWeather(await r.json(), latitude, longitude);
        })();
        entry = { expires: now() + (past ? 86400000 : 900000), request };
        cache.set(key, entry);
        void request.catch(() => {
          cache.delete(key);
        });
      }
      const wind = (await entry.request).find((h) => h.time === hour);
      if (!wind) throw new Error("Requested hour unavailable");
      return {
        wind,
        note: `Simulated from weather at ${latitude.toFixed(5)}, ${longitude.toFixed(5)}, ${wind.time} · Open-Meteo`,
      };
    } catch (error) {
      return {
        wind: null,
        note: `No weather, wind off · ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  };
}
export const weatherFeed = createWeatherFeed();
