import { useState } from "react";
import type { components } from "@twin/contracts";

import { vegetationPeriods } from "./vegetationPeriods";

type Period = components["schemas"]["VegetationPeriod"];
function defaultMonths() {
  const now = new Date();
  const latest = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() - 1, 1));
  const previous = new Date(Date.UTC(latest.getUTCFullYear() - 1, latest.getUTCMonth(), 1));
  return [previous.toISOString().slice(0, 7), latest.toISOString().slice(0, 7)];
}
export function VegetationStart({
  disabled,
  busy,
  onStart,
}: {
  disabled: boolean;
  busy: boolean;
  onStart: (periods: Period[]) => void;
}) {
  const [months, setMonths] = useState(defaultMonths);
  const periods = vegetationPeriods(months);
  return (
    <div className="land-terrain-start land-vegetation-start">
      <strong>Vegetation through time</strong>
      <p className="land-footnote">
        Compare dated Sentinel-2 vegetation signals. Each month selects one acquisition after
        checking cloud and quality masks. This does not measure species cover.
      </p>
      <details>
        <summary>Choose observation months</summary>
        <p className="land-footnote">
          Use the same season across years for a more useful comparison. Up to six months; missing
          or cloudy observations stay visible as gaps.
        </p>
        {months.map((month, index) => (
          <div className="land-vegetation-month" key={index}>
            <label>
              Observation month {index + 1}
              <input
                type="month"
                min="2015-07"
                max={new Date().toISOString().slice(0, 7)}
                value={month}
                disabled={busy}
                onChange={(e) =>
                  setMonths(months.map((v, i) => (i === index ? e.target.value : v)))
                }
              />
            </label>
            <button
              type="button"
              aria-label={`Remove observation month ${index + 1}`}
              disabled={busy || months.length === 1}
              onClick={() => setMonths(months.filter((_, i) => i !== index))}
            >
              Remove
            </button>
          </div>
        ))}
        <button
          type="button"
          disabled={busy || months.length >= 6}
          onClick={() => setMonths([...months, ""])}
        >
          Add observation month
        </button>
        {!periods && <p role="status">Choose distinct months from July 2015 through this month.</p>}
      </details>
      <p className="land-footnote">
        {periods?.map((p) => p.startDate.slice(0, 7)).join(" → ") ?? "Choose dates above"} · 20 m
        minimum analysis grid
      </p>
      <button
        type="button"
        disabled={disabled || !periods}
        onClick={() => {
          if (periods) onStart(periods);
        }}
      >
        {busy ? "Starting…" : "Compare vegetation"}
      </button>
    </div>
  );
}
