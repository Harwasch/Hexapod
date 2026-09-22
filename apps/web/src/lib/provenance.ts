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

import type { Site, SiteAsset } from "@twin/contracts";
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
