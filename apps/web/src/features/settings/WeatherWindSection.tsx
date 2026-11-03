import { GlassSegmentedControl, GlassSlider, GlassSwitch } from "@twin/ui";

import { env } from "@/app/env";
import { TIER_TEXT } from "@/lib/weather/weatherWindPlan";
import { HIDDEN_AS_CAPTURED, useAsCaptured } from "@/state/asCaptured";
import { useSettings } from "@/state/settings";
import { useWeatherWind } from "@/state/weatherWind";

const SLIDERS = [
  { key: "meanMps", label: "Mean at 10 m", unit: "m/s", min: 0, max: 25, step: 0.1 },
  { key: "gustMps", label: "Gusts at 10 m", unit: "m/s", min: 0, max: 40, step: 0.1 },
  { key: "directionDeg", label: "Coming from", unit: "°", min: 0, max: 359, step: 1 },
] as const;

/** What the switch's hint says: the scan and tier moving, or why nothing is. */
function statusLine(
  s: ReturnType<typeof useWeatherWind.getState>,
  captured: boolean,
  reduced: boolean,
): string {
  if (captured) return HIDDEN_AS_CAPTURED;
  if (reduced) return "Held calm by reduced motion";
  if (!s.enabled) return "Plants move with the hour's real wind. Modelled; off";
  if (s.reason) return `Nothing moves: ${s.reason}`;
  if (!s.scan) return "Looking for a scan…";
  if (s.scan.tier === null) return `${s.scan.name} · loading`;
  return `${s.scan.name} · ${TIER_TEXT[s.scan.tier]}`;
}

/**
 * Settings › Living survey › Weather wind (docs/LIVING_WEATHER_WIND.md). Only in a build with
 * `VITE_LIVING_WEATHER=1`; the switch starts off every session.
 */
export function WeatherWindSection() {
  const s = useWeatherWind();
  const captured = useAsCaptured((a) => a.on);
  const reduced = useSettings((a) => a.reducedMotion);
  if (!env.livingWeather) return null;
  const on = s.enabled && !captured && !reduced;
  return (
    <div data-testid="weather-wind">
      <div className="setting">
        <div>
          <div className="setting__label" id="weather-wind-label">
            Weather wind · Simulated
          </div>
          <div className="setting__hint" role="status" data-testid="weather-wind-status">
            {statusLine(s, captured, reduced)}
          </div>
        </div>
        <GlassSwitch
          aria-labelledby="weather-wind-label"
          checked={on}
          disabled={captured || reduced}
          onCheckedChange={(enabled) => s.set({ enabled })}
        />
      </div>
      {on && (
        <>
          <div className="setting">
            <div>
              <div className="setting__label" id="weather-wind-source">
                Wind from
              </div>
              {s.mode === "live" && <div className="setting__hint">{s.feedNote}</div>}
            </div>
            <GlassSegmentedControl
              aria-label="Weather wind source"
              value={s.mode}
              onValueChange={(mode) => s.set({ mode })}
              options={[
                { value: "live", label: "Weather" },
                { value: "manual", label: "Manual" },
              ]}
            />
          </div>
          {s.mode === "manual" &&
            SLIDERS.map(({ key, label, unit, min, max, step }) => (
              <div className="setting" key={key}>
                <div>
                  <div className="setting__label">{label}</div>
                  <div className="setting__hint">
                    {(s.manual[key] ?? 0).toFixed(step < 1 ? 1 : 0)} {unit}
                  </div>
                </div>
                <div style={{ width: "9rem" }}>
                  <GlassSlider
                    aria-label={`Weather wind ${label.toLowerCase()}`}
                    value={s.manual[key] ?? 0}
                    min={min}
                    max={max}
                    step={step}
                    onValueChange={(value) => {
                      const manual = { ...s.manual, [key]: value };
                      // Gusts are never below the mean.
                      if (key === "meanMps")
                        manual.gustMps = Math.max(value, manual.gustMps ?? value);
                      if (key === "gustMps") manual.meanMps = Math.min(value, manual.meanMps);
                      s.set({ manual });
                    }}
                  />
                </div>
              </div>
            ))}
        </>
      )}
    </div>
  );
}
