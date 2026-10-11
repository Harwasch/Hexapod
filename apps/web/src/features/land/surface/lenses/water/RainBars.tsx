import type { UnitSystem } from "@twin/geo";

import { inWettest, MONTH_NAMES, rainAmount, type RainSummary } from "./rain";

const WIDTH = 84;
const HEIGHT = 24;
const BAR = 5;
const GAP = (WIDTH - 12 * BAR) / 11;

/** A bar with a rounded top and a square foot on the baseline. */
function barPath(x: number, top: number): string {
  const r = Math.min(1.5, (HEIGHT - top) / 2);
  return `M${x},${HEIGHT}V${top + r}Q${x},${top} ${x + r},${top}H${x + BAR - r}Q${x + BAR},${top} ${x + BAR},${top + r}V${HEIGHT}Z`;
}

/**
 * Twelve tiny bars of monthly rain, January to December, the wettest three months in the
 * lens colour. Each bar names its month and amount on hover; the table in Details has them all.
 */
export function RainBars({ summary, units }: { summary: RainSummary; units: UnitSystem }) {
  const most = Math.max(...summary.monthlyMm.map((value) => value ?? 0), 0);
  const season = summary.wettest
    ? `wettest ${MONTH_NAMES[summary.wettest.start] ?? ""} to ${MONTH_NAMES[summary.wettest.end] ?? ""}`
    : "spread through the year";
  return (
    <svg
      className="water-bars"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      width={WIDTH}
      height={HEIGHT}
      role="img"
      aria-label={`Rain by month, ${season}`}
    >
      {summary.monthlyMm.map((value, index) => {
        if (value === null) return null;
        const height = most > 0 ? Math.max(1.5, (value / most) * HEIGHT) : 1.5;
        return (
          <path
            key={MONTH_NAMES[index]}
            d={barPath(index * (BAR + GAP), HEIGHT - height)}
            className={inWettest(summary, index) ? "is-wet" : undefined}
          >
            <title>{`${MONTH_NAMES[index] ?? ""}: ${rainAmount(value, units)}`}</title>
          </path>
        );
      })}
    </svg>
  );
}
