import type { components } from "@twin/contracts";
import { toFeet, type UnitSystem } from "@twin/geo";

/**
 * The Overview lens's words and numbers, worked out from the saved rasters alone: no model, no
 * network. Everything a card says about the terrain, the land cover and the greenness comes
 * from here, so the same data always reads the same way.
 */

export type Raster = components["schemas"]["LandRasterRead"];
export type RasterBand = components["schemas"]["RasterBand"];
export type RasterClass = components["schemas"]["RasterClass"];
export type Timeline = components["schemas"]["VegetationTimeline"];
export type Observation = components["schemas"]["VegetationObservation"];
export type Period = components["schemas"]["VegetationPeriod"];
type RasterRequest = components["schemas"]["RasterRequest"];
type Dataset = NonNullable<RasterRequest["dataset"]>;

export type OverviewKey = "terrain" | "cover" | "vegetation";

export interface OverviewSpec {
  key: OverviewKey;
  /** The investigation title the run is saved under (shared with the old research tab). */
  title: string;
  dataset: Dataset;
  resolutionM: number;
  question: string;
  /** What the person sees, in a few words. */
  name: string;
  start: string;
  running: string;
  failed: string;
  source: string;
}

export const OVERVIEW_SPECS: Record<OverviewKey, OverviewSpec> = {
  terrain: {
    key: "terrain",
    title: "Surface elevation and slope",
    dataset: "cop-dem-glo-30",
    resolutionM: 30,
    question: "Analyze surface elevation and slope inside this land.",
    name: "Terrain",
    start: "Read the terrain",
    running: "Reading the terrain…",
    failed: "The terrain could not be read.",
    source: "Copernicus GLO-30 elevation model",
  },
  cover: {
    key: "cover",
    title: "Land cover in 2021",
    dataset: "esa-worldcover-2021",
    resolutionM: 10,
    question: "Analyze the broad land-cover classes mapped in 2021 inside this land.",
    name: "Land cover",
    start: "Map land cover",
    running: "Mapping land cover…",
    failed: "Land cover could not be mapped.",
    source: "ESA WorldCover 2021",
  },
  vegetation: {
    key: "vegetation",
    title: "Vegetation through time",
    dataset: "sentinel-2-ndvi",
    resolutionM: 20,
    question: "Compare dated vegetation signals on common clear cells inside this land.",
    name: "Vegetation",
    start: "Compare greenness with last year",
    running: "Comparing greenness…",
    failed: "Greenness could not be compared.",
    source: "Sentinel-2 satellite imagery",
  },
};

export const OVERVIEW_KEYS: readonly OverviewKey[] = ["terrain", "cover", "vegetation"];

export function rasterRequest(spec: OverviewSpec, periods?: Period[]): RasterRequest {
  return {
    dataset: spec.dataset,
    resolutionM: spec.resolutionM,
    maxDimension: 512,
    ...(spec.dataset === "sentinel-2-ndvi" && periods?.length ? { periods } : {}),
  };
}

/** The months a vegetation comparison starts with: last month, and the same month a year before. */
export function defaultMonths(now = new Date()): string[] {
  const latest = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() - 1, 1));
  const previous = new Date(Date.UTC(latest.getUTCFullYear() - 1, latest.getUTCMonth(), 1));
  return [previous.toISOString().slice(0, 7), latest.toISOString().slice(0, 7)];
}

/** The newest raster of a dataset: the analysis's own, else one for this outline, else any. */
export function pickRaster(
  rows: readonly Raster[],
  dataset: Dataset,
  preferredId: string | null,
  revision: number,
): Raster | null {
  // Rows come from the server newest first; anything without a request is not a raster row.
  const mine = rows.filter((row) => row?.request?.dataset === dataset);
  return (
    (preferredId ? mine.find((row) => row.id === preferredId) : undefined) ??
    mine.find((row) => row.boundaryRevision === revision) ??
    mine[0] ??
    null
  );
}

export type RasterKind = "terrain" | "cover" | "vegetation";

/** Which analysis a raster came from, read from its request (or its bands, for old rows). */
export function rasterKind(raster: Raster): RasterKind {
  if (raster.metadata.vegetation || raster.request?.dataset === "sentinel-2-ndvi")
    return "vegetation";
  if (
    raster.request?.dataset === "esa-worldcover-2021" ||
    raster.metadata.bands[0]?.palette === "categorical"
  )
    return "cover";
  return "terrain";
}

export function isStale(raster: Raster, revision: number): boolean {
  return raster.stale || raster.boundaryRevision !== revision;
}

export function bandOf(raster: Raster | null, index: number): RasterBand | null {
  return raster?.metadata.bands.find((band) => band.index === index) ?? null;
}

// ---- Slope -----------------------------------------------------------------

/** The model reports slope in degrees; people read it as percent (rise over run). */
export function slopePercent(degrees: number): number {
  return Math.tan((Math.min(Math.max(degrees, 0), 89.9) * Math.PI) / 180) * 100;
}

export type SlopeWord = "flat" | "gentle" | "moderate" | "steep";

/** Flat under 1%, gentle under 5%, moderate up to 15%, steep beyond. */
export function slopeWord(percent: number): SlopeWord {
  if (percent < 1) return "flat";
  if (percent < 5) return "gentle";
  if (percent <= 15) return "moderate";
  return "steep";
}

export const SLOPE_SCALE = "Flat under 1%, gentle 1–5%, moderate 5–15%, steep over 15%.";

export function percentLabel(percent: number): string {
  return `${percent < 1 ? percent.toFixed(1) : Math.round(percent).toLocaleString()}%`;
}

export interface SlopeSummary {
  word: SlopeWord;
  typical: number;
  upTo: number | null;
}

/** The typical slope is the median cell; "up to" is the 95th percentile, so one spike is ignored. */
export function slopeSummary(band: RasterBand | null): SlopeSummary | null {
  if (!band?.validCells) return null;
  const median = band.percentiles["50"] ?? band.mean;
  if (median == null) return null;
  const typical = slopePercent(median);
  const high = band.percentiles["95"] ?? band.maximum;
  return {
    word: slopeWord(typical),
    typical,
    upTo: high == null ? null : slopePercent(high),
  };
}

export function capitalise(text: string): string {
  return `${text[0]?.toUpperCase() ?? ""}${text.slice(1)}`;
}

export function slopeValue(summary: SlopeSummary): string {
  const typical = `${capitalise(summary.word)}, typically ${percentLabel(summary.typical)}`;
  return summary.upTo != null && Math.round(summary.upTo) > Math.round(summary.typical)
    ? `${typical} · up to ${percentLabel(summary.upTo)}`
    : typical;
}

function slopeHeadline(word: SlopeWord): string {
  if (word === "flat") return "Nearly flat";
  if (word === "gentle") return "Mostly gentle slopes";
  if (word === "moderate") return "Moderate slopes";
  return "Steep ground";
}

// ---- Height ------------------------------------------------------------------

export function heightUnit(units: UnitSystem): string {
  return units === "imperial" ? "ft" : "m";
}

/** Whole metres or feet, sign kept (land can lie below sea level). */
export function heightValue(metres: number, units: UnitSystem): number {
  return Math.round(units === "imperial" ? toFeet(metres) : metres);
}

export function heightLabel(metres: number, units: UnitSystem): string {
  return `${heightValue(metres, units).toLocaleString()} ${heightUnit(units)}`;
}

export function elevationPhrase(minimum: number, maximum: number, units: UnitSystem): string {
  const low = heightValue(minimum, units);
  const high = heightValue(maximum, units);
  const unit = heightUnit(units);
  const size = (value: number) => Math.abs(value).toLocaleString();
  if (high < 0)
    return low === high
      ? `about ${size(low)} ${unit} below sea level`
      : `${size(high)}–${size(low)} ${unit} below sea level`;
  if (low < 0) return `from ${size(low)} ${unit} below to ${size(high)} ${unit} above sea level`;
  return low === high
    ? `about ${size(low)} ${unit} above sea level`
    : `${size(low)}–${size(high)} ${unit} above sea level`;
}

// ---- Land cover ------------------------------------------------------------

const PLAIN_CLASS: Record<number, string> = {
  10: "Tree cover",
  20: "Shrubland",
  30: "Grassland",
  40: "Cropland",
  50: "Built-up",
  60: "Bare ground",
  70: "Snow and ice",
  80: "Open water",
  90: "Wetland",
  95: "Mangroves",
  100: "Moss and lichen",
};

const IN_A_SENTENCE: Record<number, string> = { 50: "built-up land" };

/** The class name people use ("Open water", not "Permanent water bodies"). */
export function className(item: RasterClass): string {
  return PLAIN_CLASS[item.code] ?? item.label;
}

function classInSentence(item: RasterClass): string {
  return IN_A_SENTENCE[item.code] ?? className(item).toLowerCase();
}

/** Classes found on the land, largest first. */
export function presentClasses(band: RasterBand | null): RasterClass[] {
  return [...(band?.classes ?? [])]
    .filter((item) => item.cells > 0)
    .sort((a, b) => b.cells - a.cells);
}

export function shareLabel(fraction: number | null): string {
  const value = fraction ?? 0;
  if (value > 0 && value < 0.005) return "<1%";
  return `${Math.round(value * 100)}%`;
}

/** The two or three classes worth naming: at least 5% of the land each, always the largest. */
export function mainClasses(classes: readonly RasterClass[]): RasterClass[] {
  return classes.filter((item, index) => index === 0 || (item.fraction ?? 0) >= 0.05).slice(0, 3);
}

export function coverPhrase(classes: readonly RasterClass[]): string | null {
  const [first, second] = classes;
  if (!first) return null;
  const share = first.fraction ?? 0;
  const name = classInSentence(first);
  if (share >= 0.9) return `almost all ${name}`;
  if (share >= 0.72) return `mainly ${name}`;
  if (share >= 0.6) return `about two-thirds ${name}`;
  if (share >= 0.4) return `about half ${name}`;
  if (share >= 0.28 || !second) return `about a third ${name}`;
  return `a mix of ${name} and ${classInSentence(second)}`;
}

// ---- Greenness -------------------------------------------------------------

export function monthLabel(date: string): string {
  const [year = 2000, month = 1] = date.slice(0, 7).split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, 1)).toLocaleDateString(undefined, {
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  });
}

/** Under 0.03 is no change; under 0.08 slight; under 0.2 plain; beyond that, much. */
export const GREENNESS_SCALE =
  "Greenness is NDVI, from −1 to 1. Change under 0.03 reads as no change, under 0.08 as slight, 0.2 or more as much.";

function changeWords(change: number): string {
  const size = Math.abs(change);
  if (size < 0.03) return "About as green as";
  const word = change > 0 ? "greener" : "less green";
  if (size < 0.08) return `Slightly ${word} than`;
  if (size < 0.2) return `${capitalise(word)} than`;
  return `Much ${word} than`;
}

function levelWords(ndvi: number): string {
  if (ndvi < 0.1) return "Mostly bare or built";
  if (ndvi < 0.3) return "Sparse plant cover";
  if (ndvi < 0.5) return "Moderately green";
  return "Lush and green";
}

export interface Greenness {
  /** The sentence for the fact: "Slightly greener than in July 2024". */
  value: string;
  /** One short caveat. */
  note: string | null;
}

export function greenness(series: Timeline | null | undefined): Greenness | null {
  if (!series?.observations.length) return null;
  const observed = series.observations.filter((item) => item.validCells > 0);
  const first = series.observations[0];
  const last = series.observations.at(-1);
  if (!first || !last) return null;
  if (!observed.length)
    return { value: "No clear satellite view in those months", note: "Try other months." };
  if (series.observations.length === 1) {
    const only = observed[0];
    if (only?.meanNdvi == null) return null;
    return {
      value: `${levelWords(only.meanNdvi)} in ${monthLabel(only.period.startDate)}`,
      note: null,
    };
  }
  if (series.meanChange == null || !series.changeCells)
    return { value: "Clouds hid the comparison", note: "Try other months in Details." };
  const coverage = series.commonCoverageFraction;
  return {
    value: `${changeWords(series.meanChange)} in ${monthLabel(first.period.startDate)}`,
    note:
      coverage != null && coverage < 0.95
        ? `Compared on the ${shareLabel(coverage)} of the land seen clearly both times`
        : null,
  };
}

/** What the map shows for vegetation by default: the change, else the latest clear month. */
export function defaultVegetationBand(raster: Raster): number {
  const series = raster.metadata.vegetation;
  if (series?.changeBand != null && series.changeCells > 0) return series.changeBand;
  const latest = [...(series?.observations ?? [])].reverse().find((item) => item.validCells > 0);
  return latest?.band ?? raster.metadata.bands[0]?.index ?? 1;
}

// ---- Headline --------------------------------------------------------------

export interface OverviewData {
  terrain: Raster | null;
  cover: Raster | null;
  vegetation: Raster | null;
}

/** One sentence: slope, height, then what covers the land ("Mostly gentle slopes, 62–78 m…"). */
export function overviewHeadline(data: OverviewData, units: UnitSystem): string | null {
  const parts: string[] = [];
  if (data.terrain) {
    const elevation = bandOf(data.terrain, 1);
    const slope = slopeSummary(bandOf(data.terrain, 2));
    if (slope) parts.push(slopeHeadline(slope.word));
    if (elevation?.validCells && elevation.minimum != null && elevation.maximum != null)
      parts.push(elevationPhrase(elevation.minimum, elevation.maximum, units));
  }
  const cover = data.cover ? coverPhrase(presentClasses(bandOf(data.cover, 1))) : null;
  if (cover) parts.push(cover);
  if (!parts.length) {
    const green = greenness(data.vegetation?.metadata.vegetation);
    if (green) return `${green.value}.`;
    if (data.terrain || data.cover)
      return "The maps have no samples inside this land. It may be smaller than their grid.";
    return null;
  }
  return `${capitalise(parts.join(", "))}.`;
}
