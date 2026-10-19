import type { components, Footprint } from "@twin/contracts";
import { boundsOf, footprintAreaM2 } from "@twin/geo";

import { importBoundary } from "../../../geometry";

export type SolarRequest = components["schemas"]["SolarRequest"];
export type SolarAssessment = components["schemas"]["SolarAssessmentRead"];
export type SolarPreview = components["schemas"]["SolarPreview"];
export type SolarMounting = SolarRequest["mounting"];
export type HorizonPoint = components["schemas"]["SolarHorizonPoint"];

/** The share of an array area the panels cover by default: spaced ground-mounted rows. */
export const PANEL_COVER = 0.4;

/** The newest complete calendar year of hourly weather the server accepts. */
export function lastWeatherYear(now: Date = new Date()): number {
  return now.getUTCFullYear() - 1;
}

/** Panel face area for an array area, at the default cover; at least 1 m². */
export function panelAreaFor(zone: Footprint): number {
  return Math.min(1e7, Math.max(1, Math.round(footprintAreaM2(zone) * PANEL_COVER)));
}

/** Panels face the equator: south in the northern hemisphere, north in the southern. */
function equatorFacing(zone: Footprint): number {
  const { south, north } = boundsOf(zone);
  return (south + north) / 2 < 0 ? 0 : 180;
}

/**
 * A complete request with planning defaults for an array area. Every value is an editable
 * assumption; the panel area starts at {@link PANEL_COVER} of the drawn area.
 */
export function solarRequest(zone: Footprint, zoneBasis = ""): SolarRequest {
  return {
    dataset: "nasa-power-hourly-solar",
    arrayZone: zone,
    zoneBasis,
    year: lastWeatherYear(),
    moduleAreaM2: panelAreaFor(zone),
    moduleEfficiency: 0.2,
    tiltDegrees: 20,
    azimuthDegrees: equatorFacing(zone),
    mounting: "open_rack_glass_glass",
    temperatureCoefficient: -0.0035,
    dcAcRatio: 1.2,
    inverterEfficiency: 0.96,
    systemLoss: 0.14,
    additionalShadeLoss: 0,
    albedo: 0.2,
    iamB: 0.05,
    horizon: [],
    horizonBasis: "Open horizon assumed; local obstructions have not been measured.",
    assumptions:
      "Planning values. Verify equipment, panel layout, roof suitability and local shading.",
  };
}

/** Moves a request onto a new array area, resetting the panel area to fit it. */
export function withZone(request: SolarRequest, zone: Footprint, zoneBasis: string): SolarRequest {
  return { ...request, arrayZone: zone, zoneBasis, moduleAreaM2: panelAreaFor(zone) };
}

type NumericKey = {
  [K in keyof SolarRequest]-?: SolarRequest[K] extends number ? K : never;
}[keyof SolarRequest];

export interface SolarField {
  key: NumericKey;
  label: string;
  /** Shown values are percentages of the stored fraction. */
  percent?: boolean;
  min: number;
  max: number;
  /** The bound itself is not allowed. */
  above?: boolean;
  below?: boolean;
  hint?: string;
  unit: string;
}

/** The system settings, in the order a person thinks about them. Bounds match the server. */
export function solarFields(now: Date = new Date()): SolarField[] {
  return [
    {
      key: "year",
      label: "Weather year",
      min: 2001,
      max: lastWeatherYear(now),
      unit: "",
      hint: "A past year of NASA hourly weather.",
    },
    {
      key: "moduleAreaM2",
      label: "Total panel area (m²)",
      min: 0,
      above: true,
      max: 1e7,
      unit: " m²",
      hint: `Starts at ${PANEL_COVER * 100}% of the array area, typical for spaced rows on the ground.`,
    },
    {
      key: "moduleEfficiency",
      label: "Panel efficiency (%)",
      percent: true,
      min: 0,
      above: true,
      max: 0.5,
      unit: "%",
    },
    {
      key: "tiltDegrees",
      label: "Tilt above flat (degrees)",
      min: 0,
      max: 85,
      unit: "°",
    },
    {
      key: "azimuthDegrees",
      label: "Facing, clockwise from north (degrees)",
      min: 0,
      max: 360,
      below: true,
      unit: "°",
      hint: "North is 0, east 90, south 180, west 270.",
    },
    {
      key: "dcAcRatio",
      label: "Panel to inverter size ratio",
      min: 0.5,
      max: 2,
      unit: "",
    },
    {
      key: "inverterEfficiency",
      label: "Inverter efficiency (%)",
      percent: true,
      min: 0.8,
      max: 1,
      unit: "%",
    },
    {
      key: "temperatureCoefficient",
      label: "Power change per °C of heat (%)",
      percent: true,
      min: -0.01,
      max: 0,
      unit: "%",
    },
    {
      key: "systemLoss",
      label: "Wiring, dirt and other losses (%)",
      percent: true,
      min: 0,
      max: 1,
      unit: "%",
    },
    {
      key: "additionalShadeLoss",
      label: "Extra shading (%)",
      percent: true,
      min: 0,
      max: 1,
      unit: "%",
      hint: "Shade from trees or buildings the horizon does not capture.",
    },
    {
      key: "albedo",
      label: "Ground reflectance (%)",
      percent: true,
      min: 0,
      max: 1,
      unit: "%",
    },
    {
      key: "iamB",
      label: "Glancing-light loss (ASHRAE b)",
      min: 0,
      max: 0.2,
      unit: "",
    },
  ];
}

export const MOUNTINGS: { value: SolarMounting; label: string }[] = [
  { value: "open_rack_glass_glass", label: "Open rack, glass on both sides" },
  { value: "close_mount_glass_glass", label: "Close to a roof, glass on both sides" },
  { value: "insulated_back_glass_polymer", label: "Insulated back, glass and polymer" },
];

/** A shown value from a stored one (percentages are stored as fractions). */
export function shownValue(field: SolarField, value: number): number | "" {
  return Number.isFinite(value) ? Number((value * (field.percent ? 100 : 1)).toPrecision(10)) : "";
}

/** A stored value from what was typed; blank stays blank (NaN) so a draft can keep it. */
export function storedValue(field: SolarField, text: string): number {
  return text.trim() === "" ? Number.NaN : Number(text) / (field.percent ? 100 : 1);
}

function horizonProblem(horizon: HorizonPoint[]): string | null {
  if (!horizon.length) return null;
  if (
    horizon.some(
      (point) =>
        !Number.isFinite(point.azimuthDegrees) ||
        !Number.isFinite(point.elevationDegrees) ||
        point.azimuthDegrees < 0 ||
        point.azimuthDegrees >= 360 ||
        point.elevationDegrees < 0 ||
        point.elevationDegrees > 90,
    )
  )
    return "Horizon directions run from 0 to 359 degrees and heights from 0 to 90 degrees.";
  const azimuths = horizon.map((point) => point.azimuthDegrees).sort((a, b) => a - b);
  if (azimuths.length < 4 || new Set(azimuths).size !== azimuths.length)
    return "A horizon needs at least four different directions.";
  const first = azimuths[0] ?? 0;
  const gaps = azimuths.map((azimuth, index) => (azimuths[index + 1] ?? first + 360) - azimuth);
  if (Math.max(...gaps) > 90) return "Horizon directions must be no more than 90 degrees apart.";
  return null;
}

/**
 * The first thing that stops the request from being checked, in words, or null when every
 * setting is usable. Mirrors the server's rules so a bad value never reaches it.
 */
export function settingsProblem(request: SolarRequest, now: Date = new Date()): string | null {
  for (const field of solarFields(now)) {
    const value = request[field.key];
    if (typeof value !== "number" || !Number.isFinite(value)) return `Fill in ${lower(field)}.`;
    const tooLow = field.above ? value <= field.min : value < field.min;
    const tooHigh = field.below ? value >= field.max : value > field.max;
    if (tooLow || tooHigh) {
      const scale = field.percent ? 100 : 1;
      const min = Number((field.min * scale).toPrecision(6));
      const max = Number((field.max * scale).toPrecision(6));
      if (field.key === "year") return `Choose a weather year from ${min} to ${max}.`;
      return `${sentence(field)} must be ${field.above ? "above" : "at least"} ${min} and ${field.below ? "below" : "at most"} ${max}.`;
    }
  }
  if (request.year % 1 !== 0) return "Choose a whole weather year.";
  const horizon = horizonProblem(request.horizon ?? []);
  if (horizon) return horizon;
  if (!request.zoneBasis.trim()) return "Say where the array outline came from.";
  if (!request.horizonBasis.trim()) return "Say where the horizon came from.";
  if (!request.assumptions.trim()) return "Add a note about the equipment and other assumptions.";
  const area = footprintAreaM2(request.arrayZone);
  if (request.moduleAreaM2 * Math.cos((request.tiltDegrees * Math.PI) / 180) > area * 1.001)
    return "The panels need more room than the array area. Lower the panel area or draw a bigger array.";
  return null;
}

function lower(field: SolarField): string {
  return field.label.replace(/ \(.*\)$/, "").replace(/^./, (letter) => letter.toLowerCase());
}

function sentence(field: SolarField): string {
  return field.label.replace(/ \(.*\)$/, "");
}

/** The compass word for a facing direction. */
export function compass(azimuth: number): string {
  const words = [
    "north",
    "northeast",
    "east",
    "southeast",
    "south",
    "southwest",
    "west",
    "northwest",
  ];
  return words[((Math.round(azimuth / 45) % 8) + 8) % 8] ?? "south";
}

function percent(value: number): string {
  return `${Number((value * 100).toPrecision(3))}%`;
}

/** Every setting in one line: "2025 weather · 20° tilt, facing south · …". */
export function settingsSummary(request: SolarRequest): string {
  const area = footprintAreaM2(request.arrayZone);
  const parts = [
    Number.isFinite(request.year) ? `${request.year} weather` : "No weather year",
    Number.isFinite(request.tiltDegrees) && Number.isFinite(request.azimuthDegrees)
      ? `${Number(request.tiltDegrees.toPrecision(3))}° tilt, facing ${compass(request.azimuthDegrees)}`
      : "Tilt or facing missing",
    Number.isFinite(request.moduleEfficiency) && Number.isFinite(request.moduleAreaM2) && area > 0
      ? `${percent(request.moduleEfficiency)} panels over ${percent(request.moduleAreaM2 / area)} of the area`
      : "Panel size missing",
    Number.isFinite(request.systemLoss)
      ? `${percent(request.systemLoss)} losses`
      : "Losses missing",
    request.horizon?.length ? `${request.horizon.length}-point horizon` : "open horizon",
  ];
  return parts.join(" · ");
}

/** Same as the old study's, so an in-flight run from before still matches its draft. */
export function requestSignature(request: SolarRequest, revision: number): string {
  return JSON.stringify({ draft: request, revision });
}

export const solarDraftKey = (scope: string, landId: string) =>
  `living-world-land-draft:${encodeURIComponent(scope)}:solar:${landId}`;

/** What a browser keeps of an unfinished array: the request and any corners being placed. */
export interface StoredSolarDraft {
  revision: number;
  request: SolarRequest;
  points: number[][];
  /** False while the first outline is still being drawn (older drafts always had a zone). */
  hasZone: boolean;
}

/** Reads one stored request, keeping blank numbers blank. Throws on anything malformed. */
export function parseSolarDraft(raw: string): SolarRequest {
  if (raw.length > 2_000_000) throw new Error("Solar drafts are limited to 2 MB.");
  const value: unknown = JSON.parse(raw);
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Invalid solar draft.");
  const input = value as Record<string, unknown>;
  const zone = importBoundary(JSON.stringify(input.arrayZone));
  const defaults = solarRequest(zone);
  for (const [key, expected] of Object.entries(defaults)) {
    if (key === "arrayZone" || key === "horizon") continue;
    if (typeof expected === "number" && input[key] === null) {
      input[key] = Number.NaN;
      continue;
    }
    if (
      typeof input[key] !== typeof expected ||
      (typeof expected === "number" && !Number.isFinite(input[key]))
    )
      throw new Error(`Invalid solar field: ${key}.`);
  }
  if (
    !Array.isArray(input.horizon) ||
    input.horizon.length > 360 ||
    !input.horizon.every((point: unknown) => {
      if (!point || typeof point !== "object") return false;
      const { azimuthDegrees, elevationDegrees } = point as Record<string, unknown>;
      return (
        typeof azimuthDegrees === "number" &&
        Number.isFinite(azimuthDegrees) &&
        typeof elevationDegrees === "number" &&
        Number.isFinite(elevationDegrees)
      );
    })
  )
    throw new Error("Invalid horizon points.");
  return Object.fromEntries(
    Object.keys(defaults).map((key) => [key, key === "arrayZone" ? zone : input[key]]),
  ) as SolarRequest;
}

function parseCorners(value: unknown): number[][] {
  if (
    Array.isArray(value) &&
    value.length <= 1000 &&
    value.every(
      (point) =>
        Array.isArray(point) &&
        point.length === 2 &&
        typeof point[0] === "number" &&
        typeof point[1] === "number" &&
        Number.isFinite(point[0]) &&
        Number.isFinite(point[1]) &&
        Math.abs(point[0]) <= 180 &&
        Math.abs(point[1]) <= 90,
    )
  )
    return value as number[][];
  return [];
}

/** Reads the whole stored draft (`…:solar:<landId>`). Throws when it cannot be used. */
export function parseStoredDraft(raw: string): StoredSolarDraft {
  const value: unknown = JSON.parse(raw);
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("The saved array sketch is damaged.");
  const stored = value as Record<string, unknown>;
  const request = parseSolarDraft(JSON.stringify(stored.request));
  return {
    revision: typeof stored.revision === "number" ? stored.revision : 0,
    request,
    points: parseCorners(stored.points),
    hasZone: stored.hasZone !== false,
  };
}

/** The pending run, so a retried start reuses its investigation and request key. */
export interface PendingRun {
  signature: string;
  id: string;
  key: string;
}

export function parsePendingRun(raw: string | null): PendingRun | null {
  if (!raw) return null;
  try {
    const value: unknown = JSON.parse(raw);
    if (
      value &&
      typeof value === "object" &&
      "signature" in value &&
      "id" in value &&
      "key" in value &&
      typeof value.signature === "string" &&
      typeof value.id === "string" &&
      typeof value.key === "string"
    )
      return { signature: value.signature, id: value.id, key: value.key };
  } catch {
    /* Malformed browser state cannot authorize a different server request. */
  }
  return null;
}

/** The array request a solar run was started with, when the run carries one. */
export function runRequest(analysis: unknown): SolarRequest | null {
  if (!analysis || typeof analysis !== "object" || !("arrayZone" in analysis)) return null;
  return analysis as SolarRequest;
}

/** The map target of a saved assessment's array, for facts and lists that point at it. */
export function zoneTarget(assessment: SolarAssessment) {
  return { key: `solar-zone-${assessment.id}`, geometry: assessment.request.arrayZone };
}
