import { CloudSun } from "lucide-react";

import { GlassPanel } from "@twin/ui";

import { TIER_TEXT } from "@/lib/weather/weatherWindPlan";
import { useAsCaptured } from "@/state/asCaptured";
import { useSettings } from "@/state/settings";
import { useWeatherWind, weatherInput, weatherWindMoving } from "@/state/weatherWind";

/**
 * On screen for as long as weather wind moves a scan, like the Living Survey's badge
 * (SimulatedBadge.tsx): the motion is modelled from the hour's wind, never a recording.
 */
export function WeatherWindBadge() {
  const state = useWeatherWind();
  const captured = useAsCaptured((s) => s.on);
  const reduced = useSettings((s) => s.reducedMotion);
  if (!weatherWindMoving(state, captured || reduced) || !state.scan?.tier) return null;
  const wind = weatherInput(state);
  const speed = wind
    ? `${wind.meanMps.toFixed(1)} m/s, gusts ${(wind.gustMps ?? wind.meanMps).toFixed(1)}`
    : "";
  return (
    <GlassPanel
      strong
      compact
      className="living-badge"
      role="status"
      aria-label="Simulated weather wind"
      data-testid="weather-wind-badge"
    >
      <CloudSun className="notice__icon" size={16} aria-hidden="true" />
      <div className="notice__text">
        <strong>Simulated motion · weather wind</strong>
        <span>
          {TIER_TEXT[state.scan.tier]} on {state.scan.name}, from{" "}
          {state.mode === "live" ? "the hour's weather" : "manual wind"} ({speed}). The data
          underneath is never altered.
        </span>
      </div>
    </GlassPanel>
  );
}
