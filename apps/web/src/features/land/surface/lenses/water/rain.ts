import type { UnitSystem } from "@twin/geo";

import type { MonthlySeries } from "../overviewBundle";

export const MONTH_NAMES = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];
const SHORT = MONTH_NAMES.map((name) => name.slice(0, 3));
/** Average days per month, so a daily rate becomes a monthly amount (February averages leap years). */
const DAYS = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

export interface RainSummary {
  /** Rain and snow (as water) per month, in millimetres; null where the source has none. */
  monthlyMm: (number | null)[];
  annualMm: number | null;
  /** The wettest three months in a row, as month indexes (0 = January); null when even. */
  wettest: { start: number; end: number } | null;
}

/**
 * Monthly and yearly rain from NASA POWER's corrected precipitation. POWER publishes it as a
 * daily average ("mm/day") per month; the monthly amount is that times the month's days.
 */
export function rainSummary(series: MonthlySeries | null): RainSummary | null {
  if (!series) return null;
  const perDay = /day|d-1|\/d\b/i.test(series.unit) || !series.unit;
  const monthlyMm = series.values.map((value, index) =>
    value === null ? null : Math.max(0, perDay ? value * (DAYS[index] ?? 30.4) : value),
  );
  const known = monthlyMm.filter((value): value is number => value !== null);
  if (!known.length) return null;
  const annualMm =
    known.length === 12
      ? known.reduce((sum, value) => sum + value, 0)
      : series.annual !== null
        ? series.annual * (perDay ? 365.25 : 1)
        : null;
  const most = Math.max(...known);
  const least = Math.min(...known);
  // Rain spread fairly evenly through the year has no season worth naming.
  const even = known.length < 12 || (least > 0 && most / least < 1.6);
  let wettest: RainSummary["wettest"] = null;
  if (!even) {
    let best = -1;
    for (let start = 0; start < 12; start++) {
      const total = [0, 1, 2].reduce(
        (sum, offset) => sum + (monthlyMm[(start + offset) % 12] ?? 0),
        0,
      );
      if (total > best) {
        best = total;
        wettest = { start, end: (start + 2) % 12 };
      }
    }
  }
  return { monthlyMm, annualMm, wettest };
}

/** True when month `index` falls in the wettest stretch. */
export function inWettest(summary: RainSummary, index: number): boolean {
  return summary.wettest ? (index - summary.wettest.start + 12) % 12 < 3 : false;
}

/** A rain amount in the person's units: "1,020 mm" or "40 in". */
export function rainAmount(mm: number, units: UnitSystem): string {
  if (units === "imperial") {
    const inches = mm / 25.4;
    return `${inches.toLocaleString(undefined, { maximumFractionDigits: inches < 10 ? 1 : 0 })} in`;
  }
  const rounded = mm >= 100 ? Math.round(mm / 10) * 10 : Math.round(mm);
  return `${rounded.toLocaleString()} mm`;
}

/** "About 1,020 mm a year · wettest Nov to Jan" */
export function rainWords(summary: RainSummary, units: UnitSystem): string {
  const season = summary.wettest
    ? `wettest ${SHORT[summary.wettest.start] ?? ""} to ${SHORT[summary.wettest.end] ?? ""}`
    : "spread through the year";
  if (summary.annualMm === null) return season.charAt(0).toUpperCase() + season.slice(1);
  return `About ${rainAmount(summary.annualMm, units)} a year · ${season}`;
}

/** A temperature in the person's units. */
export function temperature(celsius: number, units: UnitSystem): string {
  return units === "imperial"
    ? `${Math.round((celsius * 9) / 5 + 32)} °F`
    : `${celsius.toLocaleString(undefined, { maximumFractionDigits: 1 })} °C`;
}
