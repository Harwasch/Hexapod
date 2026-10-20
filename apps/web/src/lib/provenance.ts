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
 *
 * `camera-height-estimate` is neither: the pipeline sized the capture by how high a handheld
 * phone is, so it is roughly the right size and nothing about it was measured. It is said as
 * an estimate, with its ±% whenever the catalog carries one — "estimated" with no figure would
 * read as nearly measured — and it is not the "unresolved" warning, which would no longer be
 * true.
 *
 * `manual` is a scale somebody set by hand (`PUT /assets/{id}/scale`, the Set real size tool),
 * and the asset's `renderConfig.scaleEvidence` says how: from a length measured on the scan and
 * its true length ("scaled by hand from a measured length"), or a factor typed in ("scale set
 * by hand"). The first is a measurement, of one length; the second is somebody's word.
 */
export interface PlacementProvenance {
  /** One line: how it was placed, how well, and where its scale came from. */
  readonly summary: string;
  /** True only when the placement came from a measurement rather than from a person. */
  readonly measured: boolean;
  /** True when the reconstruction's metric scale was never resolved. */
  readonly scaleUnresolved: boolean;
  /** True when the scale is an estimate (from the camera's height), not a measurement. */
  readonly scaleEstimated: boolean;
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
  // M3b: its frames were registered into an earlier epoch of the site and placed by that.
  // Only as good as the epoch it was aligned to, which the ± says; never "measured".
  "visual-epoch": "Aligned visually to an earlier scan",
  none: "Not georeferenced",
};

const SCALE_LABEL: Record<NonNullable<Provenance["scaleSource"]>, string> = {
  arkit: "metric scale from ARKit",
  "exif-gps": "metric scale from EXIF GPS",
  manual: "scale set by hand",
  "camera-height-estimate": "scale estimated from camera height",
  // The earlier scan's measured scale, carried by the alignment. The worker writes this only
  // when that scan's scale was measured; an estimated or unresolved one is inherited as such.
  "visual-epoch": "metric scale from an earlier scan",
  unresolved: "scale unresolved",
};

/** A hand-set scale found from a measured length, rather than typed in. */
const MEASURED_BY_HAND_LABEL = "scaled by hand from a measured length";

/**
 * The scale part of the summary: its label, an estimate's ±% when the catalog has one, and for
 * a scale set by hand, whether a length was measured for it (`scaleEvidence.method`).
 */
function scaleLabel(asset: SiteAsset | undefined): string {
  const provenance = asset?.provenance;
  const scale = provenance?.scaleSource ?? "unresolved";
  const pct = provenance?.scaleUncertaintyPct;
  if (scale === "camera-height-estimate" && typeof pct === "number" && Number.isFinite(pct))
    return `${SCALE_LABEL[scale]} (±${Math.round(pct)}%)`;
  if (scale === "manual" && asset?.renderConfig.scaleEvidence?.method === "measured-length")
    return MEASURED_BY_HAND_LABEL;
  return SCALE_LABEL[scale];
}

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

/** The three verdicts an epoch's alignment records (decision D4), in words that never round up. */
const VERDICT_WORDS = {
  survey: "survey-grade",
  coarse: "coarse, not survey-grade",
  unaligned: "unaligned",
} as const;
type Verdict = keyof typeof VERDICT_WORDS;

/** What the web reads of `epochs.alignment` (D6): its label, verdict and uncertainty. */
interface EpochVerdict {
  readonly label: string;
  readonly verdict: Verdict;
  readonly uncertaintyM: number | null;
}

/**
 * The recorded alignment of the epoch an asset belongs to, when it carries a verdict and a
 * label; read defensively, since `alignment` is an open object and an M2-era epoch has none.
 */
function epochVerdict(site: Site | null | undefined, asset: SiteAsset): EpochVerdict | undefined {
  const listed: unknown = (site as { epochs?: unknown } | null | undefined)?.epochs;
  if (!Array.isArray(listed)) return undefined;
  const epoch = (listed as Record<string, unknown>[]).find(
    (e) => (asset.epochId != null && e.id === asset.epochId) || e.currentAssetId === asset.id,
  );
  const alignment = epoch?.alignment;
  if (!alignment || typeof alignment !== "object") return undefined;
  const { label, verdict, uncertaintyM } = alignment as Record<string, unknown>;
  if (typeof label !== "string" || !label) return undefined;
  if (typeof verdict !== "string" || !(verdict in VERDICT_WORDS)) return undefined;
  return {
    label,
    verdict: verdict as Verdict,
    uncertaintyM:
      typeof uncertaintyM === "number" && Number.isFinite(uncertaintyM) ? uncertaintyM : null,
  };
}

/** The label less its trailing "±X m", which the summary prints in the viewer's units. */
function withoutFigure(label: string): string {
  return label.replace(/\s*±\s*[\d.eE+-]+\s*m$/, "").replace(/,\s*$/, "");
}

export function placementProvenance(
  site: Site | null | undefined,
  assetId: string | null | undefined,
  units: UnitSystem,
): PlacementProvenance | null {
  const asset = placedAsset(site, assetId);
  const provenance = asset?.provenance;
  const method = provenance?.georefMethod;
  if (!asset || !method) return null;
  const scale = provenance?.scaleSource ?? "unresolved";
  const aligned = method === "exif-gps" && scale === "exif-gps";
  // The epoch's verdict outranks the asset's method (the R2 review of #52): an epoch that
  // failed its visual alignment keeps its EXIF similarity, and "Aligned to EXIF GPS" alone
  // would hide that it is unaligned; a phone's coarse alignment must say phone and coarse.
  const recorded = epochVerdict(site, asset);
  const uncertainty = recorded?.uncertaintyM ?? provenance?.uncertaintyM;
  const how = recorded
    ? `${withoutFigure(recorded.label)} (${VERDICT_WORDS[recorded.verdict]})`
    : aligned
      ? "Aligned to EXIF GPS"
      : GEOREF_LABEL[method];
  const parts = [
    how,
    typeof uncertainty === "number" && Number.isFinite(uncertainty)
      ? `±${formatLength(uncertainty, units)}`
      : "uncertainty not recorded",
    scaleLabel(asset),
  ];
  const fromMeasurement = aligned || method === "arkit";
  return {
    summary: parts.join(" · "),
    measured: recorded
      ? recorded.verdict === "survey" || (recorded.verdict === "coarse" && fromMeasurement)
      : fromMeasurement,
    scaleUnresolved: scale === "unresolved",
    scaleEstimated: scale === "camera-height-estimate",
  };
}
