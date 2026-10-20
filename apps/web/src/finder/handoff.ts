/**
 * What the site finder (finder.html) hands back to the page that opened it, the landing page's
 * request form, over `postMessage`. Plain data and a validator, no React and no CesiumJS: the
 * landing page imports this module, and must stay free of both.
 */
import type { Footprint } from "@twin/contracts";

/** Marks the finder's messages among whatever else a window receives. */
export const FINDER_SOURCE = "twin-finder";

/** The land the visitor chose: parcels picked from county records, or an outline drawn. */
export interface FoundLand {
  method: "parcels" | "drawn";
  /** One parcel's polygon, several parcels as one MultiPolygon, or the drawn outline. */
  boundary: Footprint;
  /** The parcels' candidate ids (`ca-parcel/<county fips>/<parcel number>`); none when drawn. */
  parcelIds: string[];
  label: string;
  areaM2: number;
  /** Whose records the parcels are, as their sources ask to be credited. */
  attributions: string[];
}

export type FinderMessage =
  | { source: typeof FINDER_SOURCE; type: "found"; land: FoundLand }
  | { source: typeof FINDER_SOURCE; type: "close" };

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;
const isStrings = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === "string");

function isFootprint(value: unknown): value is Footprint {
  return (
    isRecord(value) &&
    (value.type === "Polygon" || value.type === "MultiPolygon") &&
    Array.isArray(value.coordinates)
  );
}

/** A finder message, or null for anything else (another sender, another shape). */
export function readFinderMessage(data: unknown): FinderMessage | null {
  if (!isRecord(data) || data.source !== FINDER_SOURCE) return null;
  if (data.type === "close") return { source: FINDER_SOURCE, type: "close" };
  if (data.type !== "found" || !isRecord(data.land)) return null;
  const land = data.land;
  if (
    (land.method !== "parcels" && land.method !== "drawn") ||
    !isFootprint(land.boundary) ||
    !isStrings(land.parcelIds) ||
    typeof land.label !== "string" ||
    typeof land.areaM2 !== "number" ||
    !Number.isFinite(land.areaM2) ||
    !isStrings(land.attributions)
  )
    return null;
  return {
    source: FINDER_SOURCE,
    type: "found",
    land: {
      method: land.method,
      boundary: land.boundary,
      parcelIds: land.parcelIds,
      label: land.label,
      areaM2: land.areaM2,
      attributions: land.attributions,
    },
  };
}
