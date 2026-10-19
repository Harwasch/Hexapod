import type { components, RasterDataset } from "@twin/contracts";
import { toFeet, type UnitSystem } from "@twin/geo";

import { RASTER_RUNS, rasterRequest as runRequest } from "../../../rasterRuns";

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

/**
 * The Overview's analyses: three a step at a time on the card (terrain, then land cover, then
 * greenness), and two on demand in Details (bare-earth terrain, US only, and the seasonal
 * signatures Show once compares).
 */
export type OverviewKey = "terrain" | "cover" | "vegetation" | "bareEarth" | "signature";

export interface OverviewSpec {
  key: OverviewKey;
  /** The investigation title the run is saved under (rasterRuns.ts, one table for all). */
  title: string;
  dataset: RasterDataset;
  resolutionM: number;
  question: string;
  /** The run's time budget in seconds (rasterRuns.ts). */
  maxSeconds: number;
  /** What the person sees, in a few words. */
  name: string;
  start: string;
  running: string;
  /** While the run waits for the research worker. */
  queued: string;
  failed: string;
  source: string;
}

/** The request, title and budget of a dataset's run, from the one table every start uses. */
function runOf(dataset: RasterDataset) {
  const { title, question, resolutionM, maxSeconds } = RASTER_RUNS[dataset];
  return { dataset, title, question, resolutionM, maxSeconds };
}

export const OVERVIEW_SPECS: Record<OverviewKey, OverviewSpec> = {
  terrain: {
    key: "terrain",
    ...runOf("cop-dem-glo-30"),
    name: "Terrain",
    start: "Read the terrain",
    running: "Reading the terrain…",
    queued: "Waiting to read the terrain…",
    failed: "The terrain could not be read.",
    source: "Copernicus GLO-30 elevation model",
  },
  cover: {
    key: "cover",
    ...runOf("esa-worldcover-2021"),
    name: "Land cover",
    start: "Map land cover",
    running: "Mapping land cover…",
    queued: "Waiting to map land cover…",
    failed: "Land cover could not be mapped.",
    source: "ESA WorldCover 2021",
  },
  vegetation: {
    key: "vegetation",
    ...runOf("sentinel-2-ndvi"),
    name: "Vegetation",
    start: "Compare greenness with last year",
    running: "Comparing greenness…",
    queued: "Waiting to compare greenness…",
    failed: "Greenness could not be compared.",
    source: "Sentinel-2 satellite imagery",
  },
  bareEarth: {
    key: "bareEarth",
    ...runOf("usgs-3dep-13"),
    name: "Bare-earth terrain",
    start: "Analyze bare-earth terrain",
    running: "Reading the bare-earth terrain…",
    queued: "Waiting to read the bare-earth terrain…",
    failed: "The bare-earth terrain could not be read.",
    source: "USGS 3DEP 1/3 arc-second bare-earth model",
  },
  signature: {
    key: "signature",
    ...runOf("sentinel-2-signature"),
    name: "Seasonal signatures",
    start: "Prepare 10 m signatures",
    running: "Preparing the seasonal signatures…",
    queued: "Waiting to prepare the seasonal signatures…",
    failed: "The seasonal signatures could not be prepared.",
    source: "Sentinel-2 seasonal scenes",
  },
};

/** Said beside "Analyze bare-earth terrain" on land USGS 3DEP does not cover. */
export const OUTSIDE_US =
  "This land is outside the United States, where USGS 3DEP has no data. Its terrain comes from the worldwide Copernicus model instead.";

/** The card's analyses, one next step at a time. */
export const OVERVIEW_KEYS = ["terrain", "cover", "vegetation"] as const;
/** The analyses Details offers on demand. */
export const ON_DEMAND_KEYS = ["bareEarth", "signature"] as const;
export const ALL_OVERVIEW_KEYS: readonly OverviewKey[] = [...OVERVIEW_KEYS, ...ON_DEMAND_KEYS];

/** A run's request: months only for the analyses that read dated scenes. */
export function rasterRequest(spec: OverviewSpec, periods?: Period[]): RasterRequest {
  const dated = spec.dataset === "sentinel-2-ndvi" || spec.dataset === "sentinel-2-signature";
  return runRequest(spec.dataset, dated && periods?.length ? periods : undefined);
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
  dataset: RasterDataset,
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

export type RasterKind = "terrain" | "cover" | "vegetation" | "signature";

/** Which analysis a raster came from, read from its request (or its bands, for old rows). */
export function rasterKind(raster: Raster): RasterKind {
  // A signature carries a vegetation timeline too (its dated NDVI bands): ask it first.
  if (raster.metadata.signature || raster.request?.dataset === "sentinel-2-signature")
    return "signature";
  if (raster.metadata.vegetation || raster.request?.dataset === "sentinel-2-ndvi")
    return "vegetation";
  if (
    raster.request?.dataset === "esa-worldcover-2021" ||
    raster.metadata.bands[0]?.palette === "categorical"
  )
    return "cover";
  return "terrain";
}

const SOURCE_BY_DATASET: Record<RasterDataset, string> = {
  "cop-dem-glo-30": "Copernicus GLO-30 surface model",
  "esa-worldcover-2021": "ESA WorldCover 2021",
  "sentinel-2-ndvi": "Sentinel-2",
  "sentinel-2-signature": "Sentinel-2 seasonal signatures",
  "usgs-3dep-13": "USGS 3DEP 1/3 arc-second bare-earth model",
};

const SOURCE_BY_KIND: Record<RasterKind, string> = {
  terrain: SOURCE_BY_DATASET["cop-dem-glo-30"],
  cover: SOURCE_BY_DATASET["esa-worldcover-2021"],
  vegetation: SOURCE_BY_DATASET["sentinel-2-ndvi"],
  signature: SOURCE_BY_DATASET["sentinel-2-signature"],
};

/** The model or imagery a raster was made from, by name. */
export function rasterSource(raster: Raster): string {
  const dataset = raster.request?.dataset;
  return dataset ? SOURCE_BY_DATASET[dataset] : SOURCE_BY_KIND[rasterKind(raster)];
}

export function isBareEarth(raster: Raster | null | undefined): boolean {
  return raster?.request?.dataset === "usgs-3dep-13";
}

/**
 * The bands a raster can put on the map. A signature holds five bands per season; the map shows
 * its dated NDVI bands, and the reflectance bands stay in the download.
 */
export function mapBands(raster: Raster): RasterBand[] {
  const ndvi = raster.metadata.signature?.features
    .filter((feature) => feature.key === "ndvi")
    .map((feature) => feature.band);
  return ndvi
    ? raster.metadata.bands.filter((band) => ndvi.includes(band.index))
    : raster.metadata.bands;
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

/** The newest saved raster of each analysis on this land, by analysis. */
export interface OverviewData {
  /** Copernicus GLO-30: the surface model, worldwide. */
  terrain: Raster | null;
  cover: Raster | null;
  vegetation: Raster | null;
  /** USGS 3DEP: bare earth at 10 m, United States only. */
  bareEarth: Raster | null;
  signature: Raster | null;
}

/**
 * The terrain the card reads: bare earth (USGS 3DEP) where this land has it, since it leaves
 * out trees and buildings; else the worldwide Copernicus surface model. One made for the
 * current outline wins over one made for an earlier outline.
 */
export function terrainOf(data: Pick<OverviewData, "terrain" | "bareEarth">): Raster | null {
  const { terrain, bareEarth } = data;
  if (bareEarth && !bareEarth.stale) return bareEarth;
  if (terrain && !terrain.stale) return terrain;
  return bareEarth ?? terrain;
}

/** Which analysis the card's terrain came from. */
export function terrainKey(data: OverviewData): "terrain" | "bareEarth" {
  const ground = terrainOf(data);
  return ground && ground === data.bareEarth ? "bareEarth" : "terrain";
}

/** Where a terrain number came from, in a few words: "Bare earth · USGS 3DEP, 10 m". */
export function terrainNote(raster: Raster): string {
  const grid = `${Math.round(raster.metadata.resolutionM).toLocaleString()} m`;
  return isBareEarth(raster)
    ? `Bare earth · USGS 3DEP, ${grid}`
    : `Surface incl. trees and buildings · Copernicus, ${grid}`;
}

/** One sentence: slope, height, then what covers the land ("Mostly gentle slopes, 62–78 m…"). */
export function overviewHeadline(data: OverviewData, units: UnitSystem): string | null {
  const parts: string[] = [];
  const ground = terrainOf(data);
  if (ground) {
    const elevation = bandOf(ground, 1);
    const slope = slopeSummary(bandOf(ground, 2));
    if (slope) parts.push(slopeHeadline(slope.word));
    if (elevation?.validCells && elevation.minimum != null && elevation.maximum != null)
      parts.push(elevationPhrase(elevation.minimum, elevation.maximum, units));
  }
  const cover = data.cover ? coverPhrase(presentClasses(bandOf(data.cover, 1))) : null;
  if (cover) parts.push(cover);
  if (!parts.length) {
    const green = greenness(data.vegetation?.metadata.vegetation);
    if (green) return `${green.value}.`;
    if (ground || data.cover)
      return "The maps have no samples inside this land. It may be smaller than their grid.";
    return null;
  }
  return `${capitalise(parts.join(", "))}.`;
}
