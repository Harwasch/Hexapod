import type { components, Footprint } from "@twin/contracts";
import { importBoundary } from "./geometry";
import { solarDefaults } from "./scenarioDefaults";
export type SolarRequest = components["schemas"]["SolarRequest"];
export type SolarAssessment = components["schemas"]["SolarAssessmentRead"];
export const physicalFields = new Set([
  "usableRoofAreaM2",
  "moduleEfficiency",
  "annualPlaneIrradiationKwhM2",
  "tiltDegrees",
  "azimuthDegrees",
  "shadeLoss",
  "systemLoss",
]);
export function solarRequest(boundary: Footprint): SolarRequest {
  return {
    dataset: "nasa-power-hourly-solar",
    arrayZone: boundary,
    zoneBasis: "",
    year: new Date().getUTCFullYear() - 1,
    moduleAreaM2: 100,
    moduleEfficiency: 0.2,
    tiltDegrees: 20,
    azimuthDegrees: 180,
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
export function solarFinance(a: SolarAssessment) {
  if (!a.metadata.completeYear || a.metadata.annualGenerationKwh == null)
    throw new Error("A complete weather year is required for annual economics.");
  const p = a.request;
  return {
    ...solarDefaults(),
    usableRoofAreaM2: p.moduleAreaM2,
    moduleEfficiency: p.moduleEfficiency,
    annualPlaneIrradiationKwhM2: a.metadata.unshadedPlaneIrradiationKwhM2,
    tiltDegrees: p.tiltDegrees,
    azimuthDegrees: p.azimuthDegrees,
    shadeLoss: p.additionalShadeLoss,
    systemLoss: p.systemLoss,
    irradiationBasis: `NASA POWER ${p.year}; saved hourly AC generation includes orientation, temperature, horizon and inverter effects.`,
    assumptions: p.assumptions,
  };
}
export function parseSolarDraft(raw: string): SolarRequest {
  if (raw.length > 2_000_000) throw new Error("Solar drafts are limited to 2 MB.");
  const v: unknown = JSON.parse(raw);
  if (!v || typeof v !== "object" || Array.isArray(v)) throw new Error("Invalid solar draft.");
  const input = v as Record<string, unknown>;
  const boundary = importBoundary(JSON.stringify(input.arrayZone));
  const defaults = solarRequest(boundary);
  for (const [key, value] of Object.entries(defaults)) {
    if (key === "arrayZone" || key === "horizon") continue;
    if (typeof value === "number" && input[key] === null) {
      input[key] = Number.NaN;
      continue;
    }
    if (
      typeof input[key] !== typeof value ||
      (typeof value === "number" && !Number.isFinite(input[key]))
    )
      throw new Error(`Invalid solar field: ${key}.`);
  }
  if (
    !Array.isArray(input.horizon) ||
    input.horizon.length > 360 ||
    !input.horizon.every((p: unknown) => {
      if (!p || typeof p !== "object") return false;
      const point = p as Record<string, unknown>;
      return (
        typeof point.azimuthDegrees === "number" &&
        Number.isFinite(point.azimuthDegrees) &&
        typeof point.elevationDegrees === "number" &&
        Number.isFinite(point.elevationDegrees)
      );
    })
  )
    throw new Error("Invalid horizon points.");
  return Object.fromEntries(
    Object.keys(defaults).map((key) => [key, key === "arrayZone" ? boundary : input[key]]),
  ) as SolarRequest;
}
