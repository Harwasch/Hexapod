import { useState } from "react";

import type { SolarAssessment } from "./solarDraft";
import { bestAndWorst, energyLabel, MONTHS } from "./solarWords";

const SLOT = 24;
const BAR = 14;
const BASE = 80;
const TALL = 56;
const RADIUS = 4;

/** A column rounded at its data end and square on the baseline. */
function barPath(x: number, top: number): string {
  const r = Math.min(RADIUS, (BASE - top) / 2, BAR / 2);
  return `M${x},${BASE} V${top + r} Q${x},${top} ${x + r},${top} H${x + BAR - r} Q${x + BAR},${top} ${x + BAR},${top + r} V${BASE} Z`;
}

/**
 * Twelve columns of modeled energy, one per month. The best and the worst month carry their
 * value; hovering a month reads it out. The exact numbers are in the table under Details.
 */
export function SolarMonthly({ assessment }: { assessment: SolarAssessment }) {
  const [hover, setHover] = useState<number | null>(null);
  const months = assessment.metadata.monthly;
  const extremes = bestAndWorst(assessment);
  const maximum = Math.max(1, ...months.map((month) => month.generationKwh ?? 0));
  const partial = months.some((month) => month.validHours < month.expectedHours);
  const hovered = hover === null ? null : months[hover];
  const label = extremes
    ? `Monthly energy. Most in ${MONTHS[extremes.best.month - 1]}, ${energyLabel(extremes.best.kwh, 3)}; least in ${MONTHS[extremes.worst.month - 1]}, ${energyLabel(extremes.worst.kwh, 3)}.`
    : "Monthly energy.";
  return (
    <figure className="land-solar-chart" onMouseLeave={() => setHover(null)}>
      <svg viewBox={`0 0 ${SLOT * 12} 100`} role="img" aria-label={label}>
        <line className="land-solar-chart__base" x1="0" x2={SLOT * 12} y1={BASE} y2={BASE} />
        {months.map((month, index) => {
          const x = index * SLOT + (SLOT - BAR) / 2;
          const kwh = month.generationKwh;
          const top = kwh == null ? BASE : BASE - Math.max(1.5, (kwh / maximum) * TALL);
          const extreme =
            extremes &&
            (extremes.best.month === month.month || extremes.worst.month === month.month);
          const anchor = index === 0 ? "start" : index === 11 ? "end" : "middle";
          const textX = index === 0 ? x : index === 11 ? x + BAR : x + BAR / 2;
          return (
            <g key={month.month}>
              {kwh == null ? (
                <line
                  className="land-solar-chart__gap"
                  x1={x + 3}
                  x2={x + BAR - 3}
                  y1={BASE - 2}
                  y2={BASE - 2}
                />
              ) : (
                <path
                  className="land-solar-chart__bar"
                  d={barPath(x, top)}
                  opacity={
                    month.validHours < month.expectedHours ? 0.45 : hover === index ? 1 : 0.88
                  }
                />
              )}
              {extreme && kwh != null && (
                <text className="land-solar-chart__value" x={textX} y={top - 5} textAnchor={anchor}>
                  {energyLabel(kwh, 3)}
                </text>
              )}
              <text className="land-solar-chart__month" x={x + BAR / 2} y={95} textAnchor="middle">
                {MONTHS[index]?.[0]}
              </text>
              <rect
                className="land-solar-chart__hit"
                x={index * SLOT}
                y={0}
                width={SLOT}
                height={100}
                onMouseEnter={() => setHover(index)}
              >
                <title>
                  {MONTHS[index]}: {kwh == null ? "no weather data" : energyLabel(kwh, 3)}
                </title>
              </rect>
            </g>
          );
        })}
      </svg>
      <figcaption className="land-muted" aria-hidden="true">
        {hovered
          ? `${MONTHS[hovered.month - 1]} · ${hovered.generationKwh == null ? "no weather data" : energyLabel(hovered.generationKwh, 3)}${hovered.validHours < hovered.expectedHours ? ` · ${hovered.validHours} of ${hovered.expectedHours} hours` : ""}`
          : `Energy each month, ${assessment.request.year} weather${partial ? " · paler months miss some hours" : ""}`}
      </figcaption>
    </figure>
  );
}

/** The monthly numbers behind the chart. */
export function SolarMonthlyTable({ assessment }: { assessment: SolarAssessment }) {
  return (
    // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- Wide tables need keyboard scrolling.
    <div className="land-solar-table" role="region" aria-label="Monthly values" tabIndex={0}>
      <table>
        <thead>
          <tr>
            <th scope="col">Month</th>
            <th scope="col">Energy (kWh)</th>
            <th scope="col">Open sky (kWh)</th>
            <th scope="col">Weather hours</th>
          </tr>
        </thead>
        <tbody>
          {assessment.metadata.monthly.map((month) => (
            <tr key={month.month}>
              <th scope="row">{MONTHS[month.month - 1]}</th>
              <td>
                {month.generationKwh == null
                  ? "No data"
                  : Math.round(month.generationKwh).toLocaleString("en-US")}
              </td>
              <td>
                {month.unshadedGenerationKwh == null
                  ? "No data"
                  : Math.round(month.unshadedGenerationKwh).toLocaleString("en-US")}
              </td>
              <td>
                {month.validHours} of {month.expectedHours}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
