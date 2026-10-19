import type { Footprint, LandArea } from "@twin/contracts";
import { boundsOf, footprintAreaM2 } from "@twin/geo";

import { anchorOf } from "../../geometryAnchor";
import type { SolarAssessment } from "./solarDraft";

const ACRE_M2 = 4046.8564224;

export const MONTHS = [
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

/** Rounds to a number of significant digits (1,874 → 1,900 at two). */
export function roundSignificant(value: number, digits = 2): number {
  if (!Number.isFinite(value) || value === 0) return 0;
  const scale = 10 ** (digits - Math.ceil(Math.log10(Math.abs(value))));
  return Math.round(value * scale) / scale;
}

function grouped(value: number, fraction = 0): string {
  return value.toLocaleString("en-US", { maximumFractionDigits: fraction });
}

/** Power the way people say it: "120 kW", "1.2 MW". */
export function powerLabel(kw: number): string {
  if (!Number.isFinite(kw)) return "unknown power";
  if (kw >= 1000) return `${grouped(roundSignificant(kw / 1000, 2), 1)} MW`;
  if (kw >= 10) return `${grouped(roundSignificant(kw, 2))} kW`;
  return `${grouped(kw, 1)} kW`;
}

/** Energy the way people say it: "840 kWh", "8.4 MWh", "1,900 MWh", "24 GWh". */
export function energyLabel(kwh: number, digits = 2): string {
  if (!Number.isFinite(kwh)) return "unknown energy";
  if (kwh >= 1e7) return `${grouped(roundSignificant(kwh / 1e6, digits), 1)} GWh`;
  if (kwh >= 1000) {
    const mwh = kwh / 1000;
    return `${grouped(mwh >= 10 ? roundSignificant(mwh, digits) : mwh, mwh >= 10 ? 0 : 1)} MWh`;
  }
  return `${grouped(kwh >= 10 ? roundSignificant(kwh, digits) : kwh, kwh >= 10 ? 0 : 1)} kWh`;
}

/** An array's size as an adjective: "6.1-acre", "450 m²". */
export function zoneSize(squareMetres: number): string {
  const acres = squareMetres / ACRE_M2;
  if (acres < 0.25) return `${grouped(roundSignificant(squareMetres, 2))} m²`;
  return `${grouped(acres >= 10 ? Math.round(acres) : Number(acres.toFixed(1)), 1)}-acre`;
}

/** An array's size as a measure: "6.1 acres", "450 m²". */
export function zoneArea(squareMetres: number): string {
  const acres = squareMetres / ACRE_M2;
  if (acres < 0.25) return `${grouped(roundSignificant(squareMetres, 2))} m²`;
  const shown = acres >= 10 ? Math.round(acres) : Number(acres.toFixed(1));
  return `${grouped(shown, 1)} ${shown === 1 ? "acre" : "acres"}`;
}

const SIDES = [
  "on the east side",
  "in the northeast",
  "on the north side",
  "in the northwest",
  "on the west side",
  "in the southwest",
  "on the south side",
  "in the southeast",
];

/** Where an array sits on the land, in words: "on the south side", "covering the whole land". */
export function zonePlace(zone: Footprint, land: LandArea): string {
  const area = footprintAreaM2(zone);
  if (land.areaM2 > 0 && area >= land.areaM2 * 0.9) return "covering the whole land";
  const { west, south, east, north } = boundsOf(land.boundary);
  const [x, y] = anchorOf(zone);
  const dx = (x - (west + east) / 2) / Math.max(1e-9, (east - west) / 2);
  const dy = (y - (south + north) / 2) / Math.max(1e-9, (north - south) / 2);
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 0.34) return "in the middle of the land";
  const sector = Math.round(Math.atan2(dy, dx) / (Math.PI / 4));
  return SIDES[((sector % 8) + 8) % 8] ?? "on this land";
}

/** The best and the worst month with data, or null when fewer than two months have any. */
export function bestAndWorst(assessment: SolarAssessment) {
  const months = assessment.metadata.monthly
    .filter((month) => month.generationKwh != null)
    .map((month) => ({ month: month.month, kwh: month.generationKwh ?? 0 }));
  if (months.length < 2) return null;
  const sorted = [...months].sort((a, b) => b.kwh - a.kwh);
  const best = sorted[0];
  const worst = sorted.at(-1);
  return best && worst ? { best, worst } : null;
}

/** The share of the open-sky energy lost to the horizon and extra shading (0–1). */
export function shadingLoss(assessment: SolarAssessment): number | null {
  const { modeledGenerationKwh, unshadedGenerationKwh } = assessment.metadata;
  if (!(unshadedGenerationKwh > 0)) return null;
  return Math.max(0, 1 - modeledGenerationKwh / unshadedGenerationKwh);
}

/** "July 212 MWh" for a month number and its energy. */
export function monthEnergy(month: number, kwh: number): string {
  return `${MONTHS[month - 1] ?? `Month ${month}`} ${energyLabel(kwh)}`;
}

/** The plain sentence a saved assessment leads with. */
export function assessmentHeadline(assessment: SolarAssessment, land: LandArea | null): string {
  const { metadata, request } = assessment;
  const size = zoneSize(metadata.mappedZoneAreaM2);
  const array = land ? `A ${size} array ${zonePlace(request.arrayZone, land)}` : `A ${size} array`;
  if (metadata.completeYear && metadata.annualGenerationKwh != null)
    return `${array} could make about ${energyLabel(metadata.annualGenerationKwh)} a year.`;
  return `${array} could hold ${powerLabel(metadata.capacityKwDc)} of panels. The ${request.year} weather has gaps, so there is no full-year total.`;
}

/** How much the horizon and extra shade cost, in words. */
export function shadingWords(assessment: SolarAssessment): string {
  const loss = shadingLoss(assessment);
  const { request } = assessment;
  if (loss === null) return "Not worked out";
  if (!request.horizon?.length && request.additionalShadeLoss === 0 && loss < 0.001)
    return "Open horizon assumed; no shade counted";
  const percent = loss * 100;
  return `Horizon and shade cost ${percent < 10 ? Number(percent.toFixed(1)) : Math.round(percent)}% of the energy`;
}

/** The sentence for an array still being planned. */
export function draftHeadline(zone: Footprint, capacityKw: number, land: LandArea): string {
  const size = zoneSize(footprintAreaM2(zone));
  const place = zonePlace(zone, land);
  if (!Number.isFinite(capacityKw) || capacityKw <= 0) return `A ${size} array ${place}.`;
  return `A ${size} array ${place} could hold about ${powerLabel(capacityKw)} of panels.`;
}

/** The pin on a saved array: "≈ 1.2 MW · 1,900 MWh/yr". */
export function assessmentPin(assessment: SolarAssessment): string {
  const { metadata } = assessment;
  const power = `≈ ${powerLabel(metadata.capacityKwDc)}`;
  return metadata.completeYear && metadata.annualGenerationKwh != null
    ? `${power} · ${energyLabel(metadata.annualGenerationKwh)}/yr`
    : `${power} · part year`;
}

/** Money in the scenario's currency, falling back to a plain number for unknown codes. */
export function moneyLabel(value: number, currency: string): string {
  if (!Number.isFinite(value)) return "—";
  try {
    return value.toLocaleString("en-US", {
      style: "currency",
      currency,
      maximumFractionDigits: Math.abs(value) >= 100 ? 0 : 2,
    });
  } catch {
    return `${grouped(value, 2)} ${currency}`;
  }
}

/** "Pays back in year 9" or "Does not pay back within 25 years". */
export function paybackWords(payback: unknown, years: number): string {
  if (typeof payback === "number" && Number.isFinite(payback))
    return payback <= 0 ? "Pays for itself from the start" : `Pays back in year ${payback}`;
  return `Does not pay back within ${years} years`;
}

/** A date as "3 Oct 2026". */
export function dateLabel(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}
