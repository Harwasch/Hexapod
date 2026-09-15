/** Camera-scale heuristics shared by the status bar and the adaptive quality controller. */

export type ScaleBand = "planet" | "continent" | "region" | "city" | "site" | "object" | "detail";

/** Classifies a camera altitude above the surface into a semantic scale band. */
export function scaleBandForAltitude(altitudeM: number): ScaleBand {
  if (altitudeM > 4_000_000) return "planet";
  if (altitudeM > 800_000) return "continent";
  if (altitudeM > 60_000) return "region";
  if (altitudeM > 3_000) return "city";
  if (altitudeM > 150) return "site";
  if (altitudeM > 5) return "object";
  return "detail";
}

export const SCALE_BAND_LABELS: Record<ScaleBand, string> = {
  planet: "Planet",
  continent: "Continent",
  region: "Region",
  city: "City",
  site: "Site",
  object: "Object",
  detail: "Detail",
};

/**
 * Approximate ground metres per screen pixel for a perspective camera looking at a
 * surface `distanceM` away with the given vertical field of view.
 */
export function metersPerPixel(
  distanceM: number,
  fovYRadians: number,
  viewportHeightPx: number,
): number {
  if (!(distanceM > 0) || !(viewportHeightPx > 0)) return Number.NaN;
  return (2 * distanceM * Math.tan(fovYRadians / 2)) / viewportHeightPx;
}
