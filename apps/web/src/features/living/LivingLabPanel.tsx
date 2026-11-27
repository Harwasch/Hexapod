import { useState } from "react";
import { LivingWeatherControls } from "./LivingWeatherControls";
import { setLabWeather, useLivingWeather } from "@/state/livingWeather";
import { Waves } from "lucide-react";

import { GlassPanel } from "@twin/ui";

import { useAsCaptured } from "@/state/asCaptured";
import { useLivingLab, type LivingMotionArm } from "@/state/livingLab";

function articulationAddress(value: string): URL | null {
  try {
    return new URL(value || "/", window.location.href);
  } catch {
    // A sidecar URL may be incomplete while the user types it.
    return null;
  }
}

/**
 * The living lab's developer toggle (`?living=1`): water and wind prototypes drawn on the scan
 * under PlayCanvas (state/livingLab.ts). While anything is on it says Simulated, and As
 * captured turns everything off without touching the choices here.
 */
export function LivingLabPanel() {
  const lab = useLivingLab();
  const [weatherControls, setWeatherControls] = useState(false);
  const captured = useAsCaptured((s) => s.on);
  if (!lab.open) return null;
  const on = !captured && (lab.water || lab.motion !== "off");
  const slider = (
    label: string,
    key:
      "strength" | "windBearingDeg" | "waterLevel" | "flowBearingDeg" | "greenness" | "flatGround",
    min: number,
    max: number,
    step: number,
  ) => (
    <label className="living-lab__row">
      <span>
        {label} <output>{lab[key].toFixed(step < 0.1 ? 2 : 0)}</output>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={lab[key]}
        data-testid={`living-lab-${key}`}
        onChange={(e) => lab.set({ [key]: Number(e.target.value) })}
      />
    </label>
  );
  return (
    <GlassPanel strong compact className="living-badge living-lab" data-testid="living-lab">
      <Waves className="notice__icon" size={16} aria-hidden="true" />
      <div className="notice__text">
        <strong>Living lab{on ? " · Simulated" : ""}</strong>
        <span>
          {captured
            ? "Hidden while As captured is on."
            : on
              ? "Invented water and wind, drawn per frame. The data underneath is never altered."
              : "Prototype water and wind. Off."}
        </span>
        <label className="living-lab__row">
          <input
            type="checkbox"
            checked={lab.water}
            data-testid="living-lab-water"
            onChange={(e) => lab.set({ water: e.target.checked })}
          />{" "}
          Water
        </label>
        <label className="living-lab__row">
          Motion{" "}
          <select
            value={lab.motion}
            data-testid="living-lab-motion"
            onChange={(e) => lab.set({ motion: e.target.value as LivingMotionArm })}
          >
            <option value="off">Off</option>
            <option value="gust">Gust field + shimmer</option>
            <option value="clumps">Resonant clumps</option>
            <option value="spectra">Voxel spectra</option>
            <option value="articulated">Articulated branch spectra</option>
          </select>
        </label>
        <label className="living-lab__row">
          <input
            type="checkbox"
            checked={lab.mask}
            onChange={(e) => lab.set({ mask: e.target.checked })}
          />{" "}
          Show classes
        </label>
        {lab.motion === "spectra" && (
          <label className="living-lab__row">
            Spectra sidecar
            <input
              aria-label="Spectra sidecar URL"
              value={lab.spectraUrl}
              onChange={(e) => lab.set({ spectraUrl: e.target.value })}
            />
            <span role="status">{lab.spectraStatus}</span>
          </label>
        )}
        {lab.motion === "articulated" && !weatherControls && (
          <label className="living-lab__row">
            Simulated breeze
            <select
              aria-label="Breeze preset"
              value={
                articulationAddress(lab.articulationUrl)?.searchParams.get("breeze") ?? "round3"
              }
              onChange={(e) => {
                const url = articulationAddress(lab.articulationUrl);
                if (!url) return;
                if (e.target.value === "round3") url.searchParams.delete("breeze");
                else url.searchParams.set("breeze", e.target.value);
                lab.set({ articulationUrl: url.toString() });
              }}
            >
              <option value="round3">Round 3 reference</option>
              <option value="calm">Calm</option>
              <option value="breezy">Breezy</option>
              <option value="windy">Windy</option>
            </select>
          </label>
        )}
        {lab.motion === "articulated" && (
          <label className="living-lab__row">
            Articulation sidecar
            <input
              aria-label="Articulation sidecar URL"
              value={lab.articulationUrl}
              onChange={(e) => lab.set({ articulationUrl: e.target.value })}
            />
            <span role="status">{lab.articulationStatus}</span>
          </label>
        )}
        <label className="living-lab__row">
          <input
            type="checkbox"
            checked={weatherControls}
            onChange={(e) => {
              setWeatherControls(e.target.checked);
              setLabWeather(e.target.checked ? useLivingWeather.getState().wind : null);
            }}
          />{" "}
          Weather-driven wind
        </label>
        {weatherControls && <LivingWeatherControls />}
        {!weatherControls && slider("Strength", "strength", 0, 3, 0.05)}
        {!weatherControls && slider("Wind to (°)", "windBearingDeg", 0, 360, 5)}
        {slider("Greenness", "greenness", -1, 0.4, 0.01)}
        {slider("Water level (m)", "waterLevel", -5, 5, 0.01)}
        {slider("Flow to (°)", "flowBearingDeg", 0, 360, 5)}
      </div>
    </GlassPanel>
  );
}
