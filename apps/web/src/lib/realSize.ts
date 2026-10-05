/**
 * Setting a scan's real size (features/inspector/RealSize.tsx, docs/DATA_MODEL.md "Runtime
 * scale"): which scans can be resized, and the arithmetic from a length measured on one to the
 * scale it should be drawn at, exactly as `PUT /assets/{id}/scale` checks it.
 *
 * Pure: numbers and records in, numbers and request bodies out.
 */

import type { AssetScaleUpdate, Site, SiteAsset } from "@twin/contracts";

/** The scales the API takes: a hundredth to a hundred times the model as registered. */
export const MIN_SCALE = 0.01;
export const MAX_SCALE = 100;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteIn(value: unknown, limit: number): boolean {
  return typeof value === "number" && Number.isFinite(value) && Math.abs(value) <= limit;
}

/** Creation order, as the API's `_registered_splat` orders a site's splats: time, then id. */
function byCreation(a: SiteAsset, b: SiteAsset): number {
  const at = Date.parse(a.createdAt) - Date.parse(b.createdAt);
  if (Number.isFinite(at) && at !== 0) return at;
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

/**
 * The asset of a site that `PUT /assets/{id}/scale` resizes, or null when it has none: the splat
 * its pipeline run registered -- the site's first gaussian splat by creation, served as a 3D
 * Tiles URL -- on a site whose metadata records the run (`captureId`) and the origin its tiles
 * are placed at (`registration.georef`). That is the API's own rule (apps/api assets.py
 * `_placed`): anything else, a seeded or hand-made asset, a mesh, a Cesium ion asset, is a 409
 * `not_scalable`, so the tool is never offered for one.
 */
export function scalableAsset(site: Site | null | undefined): SiteAsset | null {
  if (!site) return null;
  const metadata: unknown = site.metadata;
  if (!isRecord(metadata) || !metadata.captureId) return null;
  const registration = metadata.registration;
  const georef = isRecord(registration) ? registration.georef : undefined;
  if (!isRecord(georef) || !isFiniteIn(georef.lat, 90) || !isFiniteIn(georef.lon, 180)) return null;
  const first = site.assets
    .filter((asset) => asset.representation === "gaussian-splat")
    .sort(byCreation)[0];
  if (first?.provider !== "3d-tiles-url" || first.source.type !== "3d-tiles-url") return null;
  return first;
}

/** The catalog's runtime scale for an asset: `renderConfig.scale`, 1 when it has none. */
export function assetScale(asset: Pick<SiteAsset, "renderConfig">): number {
  const scale = asset.renderConfig.scale;
  return typeof scale === "number" && Number.isFinite(scale) && scale > 0 ? scale : 1;
}

/** Whether a scale is one the API takes. */
export function isScale(value: number): boolean {
  return Number.isFinite(value) && value >= MIN_SCALE && value <= MAX_SCALE;
}

/**
 * The scale a measured length says: a length `measuredM` long as drawn at `atScale` is really
 * `trueM`, so the model should be drawn `trueM / measuredM` times as big as it was then. The
 * API's `implied_scale`, term for term, so a body built from these numbers always agrees with
 * the scale sent beside them.
 */
export function scaleFromLength(measuredM: number, trueM: number, atScale: number): number {
  return (atScale * trueM) / measuredM;
}

/** A length measured on the scan and what it really is: the evidence for a measured scale. */
export interface MeasuredLength {
  /** As drawn when it was measured, in metres. */
  readonly measuredM: number;
  readonly trueM: number;
  /** The scale the scan was drawn at when it was measured (a preview counts). */
  readonly atScale: number;
}

/** `PUT /assets/{id}/scale` for a measured length: the scale it says, and the numbers. */
export function measuredScaleBody(length: MeasuredLength): AssetScaleUpdate {
  return {
    scale: scaleFromLength(length.measuredM, length.trueM, length.atScale),
    evidence: {
      method: "measured-length",
      measuredLengthM: length.measuredM,
      trueLengthM: length.trueM,
      measuredAtScale: length.atScale,
    },
  };
}

/** `PUT /assets/{id}/scale` for a factor typed in. */
export function directScaleBody(scale: number): AssetScaleUpdate {
  return { scale, evidence: { method: "direct" } };
}

/** `PUT /assets/{id}/scale` back to the model as registered. */
export const RESET_SCALE_BODY: AssetScaleUpdate = { reset: true };

const LENGTH_UNITS: Record<string, number> = {
  "": 1,
  m: 1,
  cm: 0.01,
  mm: 0.001,
  km: 1000,
  ft: 0.3048,
  "'": 0.3048,
  in: 0.0254,
  '"': 0.0254,
};

/**
 * A length as somebody types it, in metres: a number, in `bare` -- metres, or feet for somebody
 * who reads lengths in feet -- unless it says otherwise (`1.8`, `180 cm`, `6 ft`, `72"`). A
 * decimal comma is read as a point. Null for anything else, and for a length not above zero.
 */
export function parseLength(text: string, bare: "m" | "ft" = "m"): number | null {
  const match = /^\s*(\d+(?:[.,]\d*)?|[.,]\d+)\s*(m|cm|mm|km|ft|in|'|")?\s*$/i.exec(text);
  if (!match) return null;
  const value = Number.parseFloat((match[1] ?? "").replace(",", "."));
  const unit = LENGTH_UNITS[(match[2] ?? bare).toLowerCase()];
  if (unit === undefined || !Number.isFinite(value) || value <= 0) return null;
  return value * unit;
}

/** A scale as somebody types it: `0.25`, `×0.25`, `x0.25`, `25%`. Null unless one the API takes. */
export function parseScale(text: string): number | null {
  const match = /^\s*[×x*]?\s*(\d+(?:[.,]\d*)?|[.,]\d+)\s*(%)?\s*$/i.exec(text);
  if (!match) return null;
  const value = Number.parseFloat((match[1] ?? "").replace(",", ".")) / (match[2] ? 100 : 1);
  return isScale(value) ? value : null;
}

/** A scale for reading: four significant figures, no trailing zeros (`×0.2427`, `×1`). */
export function formatScale(scale: number): string {
  return `×${Number(scale.toPrecision(4)).toString()}`;
}
