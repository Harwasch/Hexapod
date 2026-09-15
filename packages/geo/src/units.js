/** Unit systems and formatting helpers. All internal math is metric. */
const FEET_PER_METER = 3.280839895;
const MILES_PER_METER = 0.000621371192;
const SQFT_PER_M2 = 10.7639104;
const ACRES_PER_M2 = 0.000247105381;
const HECTARES_PER_M2 = 0.0001;
function round(value, digits) {
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  });
}
/** Formats a length. Picks a sensible sub-unit so centimetres and kilometres both read well. */
export function formatLength(meters, units = "metric") {
  if (!Number.isFinite(meters)) return "—";
  const abs = Math.abs(meters);
  if (units === "imperial") {
    const feet = abs * FEET_PER_METER;
    if (feet < 1) return `${round(feet * 12, 2)} in`;
    if (feet < 5280) return `${round(feet, feet < 100 ? 2 : 0)} ft`;
    return `${round(abs * MILES_PER_METER, 2)} mi`;
  }
  if (abs < 0.01) return `${round(abs * 1000, 1)} mm`;
  if (abs < 1) return `${round(abs * 100, 1)} cm`;
  if (abs < 1000) return `${round(abs, abs < 100 ? 2 : 1)} m`;
  return `${round(abs / 1000, abs < 100_000 ? 2 : 0)} km`;
}
/** Formats an altitude / height value with a coarser precision than lengths. */
export function formatAltitude(meters, units = "metric") {
  if (!Number.isFinite(meters)) return "—";
  if (units === "imperial") {
    const feet = meters * FEET_PER_METER;
    if (Math.abs(feet) >= 5280 * 10) return `${round(meters * MILES_PER_METER, 0)} mi`;
    return `${round(feet, Math.abs(feet) < 10 ? 1 : 0)} ft`;
  }
  if (Math.abs(meters) >= 10_000) return `${round(meters / 1000, 0)} km`;
  if (Math.abs(meters) >= 1000) return `${round(meters / 1000, 1)} km`;
  return `${round(meters, Math.abs(meters) < 10 ? 1 : 0)} m`;
}
export function formatArea(squareMeters, units = "metric") {
  if (!Number.isFinite(squareMeters)) return "—";
  const abs = Math.abs(squareMeters);
  if (units === "imperial") {
    if (abs < 4047 / 4) return `${round(abs * SQFT_PER_M2, 0)} sq ft`;
    return `${round(abs * ACRES_PER_M2, abs < 40_470 ? 2 : 1)} ac`;
  }
  if (abs < 10_000) return `${round(abs, abs < 100 ? 1 : 0)} m²`;
  if (abs < 1_000_000) return `${round(abs * HECTARES_PER_M2, 2)} ha`;
  return `${round(abs / 1_000_000, 2)} km²`;
}
/** Ground resolution per screen pixel, e.g. "3.2 cm/px". */
export function formatResolution(metersPerPixel, units = "metric") {
  if (!Number.isFinite(metersPerPixel) || metersPerPixel <= 0) return "—";
  return `${formatLength(metersPerPixel, units)}/px`;
}
export function toFeet(meters) {
  return meters * FEET_PER_METER;
}
export function toAcres(squareMeters) {
  return squareMeters * ACRES_PER_M2;
}
