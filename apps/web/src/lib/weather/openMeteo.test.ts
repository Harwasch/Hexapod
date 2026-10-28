import { describe, expect, it, vi } from "vitest";
import fixture from "./fixtures/minnetonka-hours.json";
import { createWeatherFeed, parseWeather } from "./openMeteo";
const lat = 44.944403,
  lon = -93.425903,
  date = new Date("2024-12-04T21:00Z");
describe("Open-Meteo offline fixtures", () => {
  it("preserves m/s and FROM bearing", () => {
    const h = parseWeather(fixture, lat, lon);
    expect(h[3]).toMatchObject({
      meanMps: 11.49,
      gustMps: 22.9,
      directionDeg: 316,
      time: date.toISOString().replace(":00.000", ""),
    });
  });
  it("deduplicates hourly archive requests and expires cache", async () => {
    const get = vi
      .fn<typeof fetch>()
      .mockImplementation(() => Promise.resolve(new Response(JSON.stringify(fixture))));
    let now = Date.parse("2026-10-28");
    const feed = createWeatherFeed(get, () => now);
    const [a, b] = await Promise.all([feed(lat, lon, date, true), feed(lat, lon, date, true)]);
    expect(a.wind).toEqual(b.wind);
    expect(a.wind?.gustMps).toBe(22.9);
    await feed(lat, lon, new Date("2024-12-04T22:00Z"), true);
    expect(get).toHaveBeenCalledTimes(1);
    expect(get.mock.calls[0]?.[0]).toEqual(expect.stringContaining("archive-api.open-meteo.com"));
    now += 86400001;
    expect((await feed(lat, lon, date, true)).wind?.gustMps).toBe(22.9);
    expect(get).toHaveBeenCalledTimes(2);
  });
  it("uses forecast for now and does not request data while flag is off", async () => {
    const get = vi
      .fn<typeof fetch>()
      .mockImplementation(() => Promise.resolve(new Response(JSON.stringify(fixture))));
    const feed = createWeatherFeed(get, () => date.getTime());
    expect((await feed(lat, lon, date, false)).wind).toBeNull();
    expect(get).not.toHaveBeenCalled();
    await feed(lat, lon, date, true);
    expect(get.mock.calls[0]?.[0]).toEqual(
      expect.stringContaining("api.open-meteo.com/v1/forecast"),
    );
  });
  it("fails soft on bad units, missing values, HTTP failures and missing hours", async () => {
    for (const value of [
      { ...fixture, hourly_units: { wind_speed_10m: "km/h" } },
      { ...fixture, hourly: { ...fixture.hourly, wind_gusts_10m: [null] } },
    ])
      expect(() => parseWeather(value, lat, lon)).toThrow();
    const feed = createWeatherFeed(vi.fn<typeof fetch>().mockRejectedValue(new Error("offline")));
    expect(await feed(lat, lon, date, true)).toEqual({
      wind: null,
      note: "No weather, wind off · offline",
    });
  });
  it("turns wind off on a missing hour or an HTTP error", async () => {
    const missing = createWeatherFeed(
      vi
        .fn<typeof fetch>()
        .mockImplementation(() => Promise.resolve(new Response(JSON.stringify(fixture)))),
    );
    expect((await missing(lat, lon, new Date("2024-12-04T22:00Z"), true)).note).toContain(
      "Requested hour unavailable",
    );
    const bad = createWeatherFeed(
      vi.fn<typeof fetch>().mockResolvedValue(new Response("unavailable", { status: 503 })),
    );
    expect(await bad(lat, lon, date, true)).toEqual({
      wind: null,
      note: "No weather, wind off · Weather HTTP 503",
    });
  });
});
