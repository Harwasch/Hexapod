/**
 * What a site's geometry is, in the catalog's own words.
 *
 * The Living Survey puts modelled motion on top of geometry the console did not model, and the
 * UI has to keep those two apart. This is the "what it is" half: it reads the catalog metadata
 * a capture already carries — capture date, ground sample distance, the pipeline that produced
 * it — and turns it into a sentence that claims **only what the catalog records**.
 *
 * Three rules, all of them about not overclaiming:
 *
 * 1. **"Measured capture" is earned, not assumed.** It is said only when the catalog holds a
 *    capture date or a resolution figure. A site with neither gets "No capture date or
 *    resolution recorded" — which is what is actually known about it — rather than a label
 *    asserting it was, or was not, surveyed.
 * 2. **Resolution is stated as what it is.** Ground sample distance is a property of the
 *    imagery a capture was reconstructed from, not of the reconstruction, so it is printed as
 *    "per pixel of source imagery". A splat does not resolve detail at its source GSD.
 * 3. **The pipeline note is carried verbatim.** The capture archive says of the
 *    synthetic tree "procedural; no capture, no reconstruction", and that sentence must reach
 *    the screen unedited. It is the one thing standing between a generated fixture and a viewer
 *    who assumes everything in a survey console was scanned — so no classifier of ours gets to
 *    paraphrase it away.
 */

import type { Provenance, Site, SiteAsset } from "@twin/contracts";
import { formatLength, type UnitSystem } from "@twin/geo";

import { formatDate } from "./format";

export interface GeometryProvenance {
  /** True only when the catalog records a capture date or a resolution figure. */
  readonly measured: boolean;
  /** One line: what this geometry is, with the date and resolution the catalog holds. */
  readonly summary: string;
  /** The pipeline that produced it, in the catalog's own words. Empty when unrecorded. */
  readonly note: string;
}

/**
 * A capture date as the catalog stores it, without inventing precision.
 *
 * `metadata.captured` is free text from the capture manifest and is often just a year. Passing
 * "2016" through a date formatter would print "Jan 1, 2016" — a day and a month nobody
 * recorded — so anything that is not a full timestamp is shown exactly as written.
 */
export function formatCaptureDate(value: string | null | undefined): string | null {
  if (typeof value !== "string") return null;
  const raw = value.trim();
  if (raw === "") return null;
  if (!/^\d{4}-\d{2}-\d{2}/.test(raw)) return raw;
  const formatted = formatDate(raw);
  return formatted === "—" ? raw : formatted;
}

/** The resolution figure a catalog asset carries, phrased as the thing it actually measures. */
export function formatResolution(asset: SiteAsset | undefined, units: UnitSystem): string | null {
  const resolution = asset?.resolution;
  if (!resolution) return null;
  const gsd = resolution.groundSampleDistanceM;
  if (typeof gsd === "number" && Number.isFinite(gsd) && gsd > 0)
    return `about ${formatLength(gsd, units)} per pixel of source imagery`;
  const spacing = resolution.pointSpacingM;
  if (typeof spacing === "number" && Number.isFinite(spacing) && spacing > 0)
    return `points about ${formatLength(spacing, units)} apart`;
  return null;
}

/** Reads the free-text `captured` field the capture pipeline writes into site metadata. */
function capturedFromMetadata(site: Site | null | undefined): string | null {
  const value = site?.metadata?.captured;
  return typeof value === "string" ? value : null;
}

/** Reads the pipeline note, preferring the asset's own provenance over the site's. */
function pipelineNote(site: Site | null | undefined, asset: SiteAsset | undefined): string {
  const fromAsset = asset?.provenance?.notes;
  if (typeof fromAsset === "string" && fromAsset.trim() !== "") return fromAsset.trim();
  const fromSite = site?.metadata?.pipeline;
  if (typeof fromSite === "string" && fromSite.trim() !== "") return fromSite.trim();
  return "";
}

/**
 * What the catalog knows about the geometry of one asset of one site.
 *
 * `assetId` is the deformed asset when there is one, so the resolution quoted belongs to the
 * representation actually on screen rather than to whichever asset happens to be first.
 */
export function geometryProvenance(
  site: Site | null | undefined,
  assetId: string | null | undefined,
  units: UnitSystem,
): GeometryProvenance {
  const asset = assetId ? site?.assets.find((a) => a.id === assetId) : undefined;
  const date = formatCaptureDate(asset?.observedAt ?? capturedFromMetadata(site));
  const resolution = formatResolution(asset, units);
  const note = pipelineNote(site, asset);
  if (date === null && resolution === null)
    return { measured: false, summary: "No capture date or resolution recorded", note };
  const parts = [
    "Measured capture",
    date ?? "capture date not recorded",
    resolution ?? "resolution not recorded",
  ];
  return { measured: true, summary: parts.join(" · "), note };
}

/**
 * How a capture was placed on the globe, and whether its size means anything.
 *
 * The "where it is" half, and the third thing the inspector has to keep apart from the other
 * two: a capture somebody dropped on a map at plus or minus ten metres and one aligned to EXIF
 * GPS are both "measured capture" by `geometryProvenance`, and they are not the same claim.
 *
 * The same two rules as above apply. Nothing is said that the catalog does not record — an
 * asset with no georeference provenance produces `null` and the inspector shows no row at all,
 * rather than a row reading "unknown" that looks like a finding. And `scaleSource:
 * "unresolved"` is printed, loudly, because it is the one that matters most and reads like an
 * absence: a reconstruction from images alone has no metric scale, so a length measured off it
 * — or a modal frequency derived from it — is meaningless rather than merely imprecise.
 */
export interface PlacementProvenance {
  /** One line: how it was placed, how well, and where its scale came from. */
  readonly summary: string;
  /** True only when the placement came from a measurement rather than from a person. */
  readonly measured: boolean;
  /** True when the reconstruction's metric scale was never resolved. */
  readonly scaleUnresolved: boolean;
}

const GEOREF_LABEL: Record<NonNullable<Provenance["georefMethod"]>, string> = {
  // "Located", not "Aligned" — the difference is whether a similarity was ever solved for,
  // and the honest tell for that is `scaleSource`: `exif_gps` sets it to `exif-gps` only on
  // the branch that ran `colmap model_aligner`, and leaves it `unresolved` when all it had
  // was a coordinate (an iPhone video's container location, or frames with no pose model).
  // Calling that "Aligned to EXIF GPS" would be the overclaim this module exists to
  // prevent, so that one label is chosen from both fields rather than from the method.
  "exif-gps": "Located by EXIF GPS",
  arkit: "Aligned to ARKit poses",
  manual: "Placed by hand",
  none: "Not georeferenced",
};

const SCALE_LABEL: Record<NonNullable<Provenance["scaleSource"]>, string> = {
  arkit: "metric scale from ARKit",
  "exif-gps": "metric scale from EXIF GPS",
  manual: "scale set by hand",
  unresolved: "scale unresolved",
};

/** The first asset of a site that records how it was placed, or undefined. */
function placedAsset(
  site: Site | null | undefined,
  assetId?: string | null,
): SiteAsset | undefined {
  const assets = site?.assets ?? [];
  const preferred = assetId ? assets.find((a) => a.id === assetId) : undefined;
  if (preferred?.provenance?.georefMethod) return preferred;
  return assets.find((a) => a.provenance?.georefMethod);
}

export function placementProvenance(
  site: Site | null | undefined,
  assetId: string | null | undefined,
  units: UnitSystem,
): PlacementProvenance | null {
  const provenance = placedAsset(site, assetId)?.provenance;
  const method = provenance?.georefMethod;
  if (!method) return null;
  const uncertainty = provenance?.uncertaintyM;
  const scale = provenance?.scaleSource ?? "unresolved";
  const aligned = method === "exif-gps" && scale === "exif-gps";
  const parts = [
    aligned ? "Aligned to EXIF GPS" : GEOREF_LABEL[method],
    typeof uncertainty === "number" && Number.isFinite(uncertainty)
      ? `±${formatLength(uncertainty, units)}`
      : "uncertainty not recorded",
    SCALE_LABEL[scale],
  ];
  return {
    summary: parts.join(" · "),
    measured: aligned || method === "arkit",
    scaleUnresolved: scale === "unresolved",
  };
}
