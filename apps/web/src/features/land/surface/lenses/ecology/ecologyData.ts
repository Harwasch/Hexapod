import type {
  components,
  InvestigationDetail,
  LandEvidence,
  LandFinding,
  ResearchArtifact,
} from "@twin/contracts";

/**
 * Reading the ecological lookups in plain words. Everything here is deterministic: it turns
 * the saved tables, evidence and source events of one ecology run into the few things a person
 * wants to know (which ecoregion, which plant community the soil suggests, which species have
 * been recorded), and keeps the raw rows for Details.
 */

export type EcologyRequest = components["schemas"]["EcologyRequest"];
type SurveySummary = components["schemas"]["SurveySummary"];
type SurveyRead = components["schemas"]["SurveyRead"];
type SurveyPlot = components["schemas"]["SurveyPlot"];
export type TableRow = Record<string, string | number | boolean | null>;
/** The part of a research event this lens reads. */
export interface SourceEvent {
  kind: string;
  payload: Record<string, unknown>;
}
export interface TableColumn {
  key: string;
  label: string;
  unit?: string | null;
}

/** The investigation title every ecology lookup gets; saved runs are found by it. */
export const ECOLOGY_TITLE = "Ecological context and name review";

export const PROVIDERS = {
  regions: "epa-ecoregions",
  sites: "usda-ecological-sites",
  species: "gbif-occurrences",
  names: "col-taxonomy",
} as const;

/** What each source is, the way a person would say it. */
export const SOURCE_LABELS: Record<string, string> = {
  [PROVIDERS.regions]: "EPA ecoregions",
  [PROVIDERS.sites]: "USDA ecological sites",
  [PROVIDERS.species]: "GBIF species records",
  [PROVIDERS.names]: "Catalogue of Life name check",
};

export const STATUS_LABELS: Record<string, string> = {
  available: "Found",
  empty: "Nothing found",
  uncovered: "Not covered here",
  unavailable: "Could not be reached",
};

/** What "Learn what lives here" asks for: the three context sources, no name review. */
export const DEFAULT_REQUEST: EcologyRequest = {
  dataset: "ecological-context",
  taxa: [],
  includeEcoregions: true,
  includeEcologicalSites: true,
  includeOccurrences: true,
};

export function isEcologyRequest(value: unknown): value is EcologyRequest {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return record.dataset === "ecological-context" || "includeEcoregions" in record;
}

/** Scientific names to review, one per line: at most 20 distinct names under 201 characters. */
export function ecologyNames(text: string, kingdom: string): EcologyRequest["taxa"] | null {
  const names = text
    .split(/\r?\n/)
    .map((line) => line.trim().replace(/\s+/g, " "))
    .filter(Boolean);
  if (
    names.length > 20 ||
    names.some((name) => name.length > 200) ||
    new Set(names.map((name) => name.toLowerCase())).size !== names.length
  )
    return null;
  return names.map((scientificName) => ({ scientificName, kingdom: kingdom || null }));
}

const SMALL_WORDS = new Set(["and", "of", "the", "in", "on", "or", "to", "a"]);

/** "EASTERN TEMPERATE FORESTS" → "Eastern Temperate Forests"; mixed case is left alone. */
export function titleCase(value: string): string {
  const text = value.trim();
  if (text !== text.toUpperCase()) return text;
  return text
    .toLowerCase()
    .split(/\s+/)
    .map((word, index) =>
      index > 0 && SMALL_WORDS.has(word) ? word : word.replace(/^[a-z]/, (c) => c.toUpperCase()),
    )
    .join(" ");
}

/**
 * An ecological site's name without its catalogue code or rainfall band:
 * `R077CY036TX Loamy Upland 16-20" PZ` → "Loamy Upland".
 */
export function plainSiteName(raw: string): string {
  const name = raw
    .replace(/^[RF]\d{3}[A-Z]{2}\d{3}[A-Z]{2}\s*[-:–]?\s*/, "")
    .replace(
      /[\s,]+\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?\s*(["”″]|in\.?|inches)?\s*(p\.?\s*z\.?)?\s*$/i,
      "",
    )
    .trim();
  return titleCase(name || raw.trim());
}

function text(value: unknown): string | null {
  if (typeof value === "string") return value.trim() || null;
  if (typeof value === "number") return String(value);
  return null;
}

function distinct(values: (string | null)[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    if (!value) continue;
    const key = value.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(value);
  }
  return out;
}

function parseExcerpt(evidence: LandEvidence | undefined): Record<string, unknown> | null {
  if (!evidence) return null;
  try {
    const value: unknown = JSON.parse(evidence.excerpt);
    return value && typeof value === "object" && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

function yearOf(value: unknown): number | null {
  const match = typeof value === "string" ? /^(\d{4})/.exec(value.trim()) : null;
  return match ? Number(match[1]) : null;
}

export interface SourceStatus {
  provider: string;
  status: string;
  message: string;
}

export interface EcologyTable {
  artifact: ResearchArtifact;
  columns: TableColumn[];
  rows: TableRow[];
}

export interface EcologyContext {
  runId: string;
  regions: {
    table: EcologyTable;
    level4: string[];
    level3: string[];
    level1: string[];
  } | null;
  sites: {
    table: EcologyTable;
    /** Plain names, the largest soil component's first. */
    names: string[];
    samplePoint: [number, number] | null;
  } | null;
  species: {
    table: EcologyTable;
    records: number;
    /** Species by how often they were recorded, most first. */
    ranked: { name: string; count: number }[];
    years: [number, number] | null;
  } | null;
  names: {
    tables: EcologyTable[];
    rows: TableRow[];
    exact: number;
    unsure: number;
  } | null;
  /** The last word from each source, by provider (name checks keep the latest). */
  sources: Record<string, SourceStatus>;
  findings: LandFinding[];
  artifacts: ResearchArtifact[];
  evidence: LandEvidence[];
}

/** Source events (`kind: "source"`) by provider, the newest winning. */
export function sourceStatuses(events: SourceEvent[] | undefined): Record<string, SourceStatus> {
  const out: Record<string, SourceStatus> = {};
  for (const event of events ?? []) {
    if (event.kind !== "source") continue;
    const provider = text(event.payload.provider);
    if (!provider) continue;
    out[provider] = {
      provider,
      status: text(event.payload.status) ?? "unknown",
      message: text(event.payload.message) ?? "",
    };
  }
  return out;
}

/** Everything one ecology run found, read out of its saved investigation. */
export function readEcology(
  detail: InvestigationDetail,
  runId: string,
  events?: SourceEvent[],
): EcologyContext {
  const evidence = detail.evidence.filter((item) => item.runId === runId);
  const byId = new Map(evidence.map((item) => [item.id, item]));
  const providerOf = (ids: string[]) =>
    ids.map((id) => byId.get(id)?.provider).find((value): value is string => Boolean(value)) ??
    null;
  const artifacts = detail.artifacts.filter((item) => item.runId === runId);
  const findings = detail.findings.filter((item) => item.runId === runId);
  const tables = (provider: string): EcologyTable[] =>
    artifacts.flatMap((artifact) =>
      artifact.output.kind === "table" && providerOf(artifact.evidenceIds) === provider
        ? [
            {
              artifact,
              columns: artifact.output.columns.map((column) => ({
                key: column.key,
                label: column.label,
                unit: column.unit ?? null,
              })),
              rows: artifact.output.rows,
            },
          ]
        : [],
    );

  const regionTable = tables(PROVIDERS.regions)[0];
  const regions = regionTable
    ? {
        table: regionTable,
        level4: distinct(regionTable.rows.map((row) => text(row.US_L4NAME))),
        level3: distinct(regionTable.rows.map((row) => text(row.US_L3NAME))),
        level1: distinct(regionTable.rows.map((row) => text(row.NA_L1NAME))).map(titleCase),
      }
    : null;

  const siteTable = tables(PROVIDERS.sites)[0];
  let samplePoint: [number, number] | null = null;
  const siteData = parseExcerpt(evidence.find((item) => item.provider === PROVIDERS.sites));
  const point = siteData?.samplePoint;
  if (
    Array.isArray(point) &&
    typeof point[0] === "number" &&
    typeof point[1] === "number" &&
    Number.isFinite(point[0]) &&
    Number.isFinite(point[1])
  )
    samplePoint = [point[0], point[1]];
  const sites = siteTable
    ? {
        table: siteTable,
        names: distinct(
          siteTable.rows.map((row) => {
            const name = text(row.ecoclassname);
            return name ? plainSiteName(name) : null;
          }),
        ),
        samplePoint,
      }
    : null;

  const speciesTable = tables(PROVIDERS.species)[0];
  let species: EcologyContext["species"] = null;
  if (speciesTable) {
    const counts = new Map<string, number>();
    let first: number | null = null;
    let last: number | null = null;
    for (const row of speciesTable.rows) {
      const name = text(row.species);
      if (name) counts.set(name, (counts.get(name) ?? 0) + 1);
      const year = yearOf(row.date);
      if (year !== null) {
        first = first === null ? year : Math.min(first, year);
        last = last === null ? year : Math.max(last, year);
      }
    }
    species = {
      table: speciesTable,
      records: speciesTable.rows.length,
      ranked: [...counts.entries()]
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name)),
      years: first !== null && last !== null ? [first, last] : null,
    };
  }

  const nameTables = tables(PROVIDERS.names);
  const nameRows = nameTables.flatMap((table) => table.rows);
  const exact = nameRows.filter((row) => row.matchType === "EXACT").length;
  const names = nameTables.length
    ? { tables: nameTables, rows: nameRows, exact, unsure: nameRows.length - exact }
    : null;

  return {
    runId,
    regions,
    sites,
    species,
    names,
    sources: sourceStatuses(events),
    findings,
    artifacts,
    evidence,
  };
}

function list(items: string[]): string {
  if (items.length <= 1) return items[0] ?? "";
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`;
}

/** "Triassic Lowlands", "A and B", or "A and 2 more". */
export function shortList(items: string[], max = 2): string {
  if (items.length <= max) return list(items);
  return `${items.slice(0, max - 1).join(", ")} and ${items.length - max + 1} more`;
}

/** Continental regions named like places take no article: "Mediterranean California". */
function withArticle(name: string): string {
  return name.endsWith("California") ? name : `the ${name}`;
}

/** Where the land sits, as a caption: "Northern Piedmont, in the Eastern Temperate Forests." */
export function regionSentence(regions: EcologyContext["regions"]): string | null {
  if (!regions) return null;
  const [first, second] = regions.level3;
  if (!first) return null;
  const realm = regions.level1.length === 1 ? regions.level1[0] : undefined;
  if (!second) return realm ? `${first}, in ${withArticle(realm)}.` : `${first}.`;
  if (regions.level3.length === 2) return `Where the ${first} meets the ${second}.`;
  return `Where ${regions.level3.length} ecoregions meet, including the ${first}.`;
}

/** The headline: where the land sits and what has been recorded on it, in plain words. */
export function ecologyHeadline(context: EcologyContext, requested?: EcologyRequest): string {
  const parts: string[] = [];
  const region = regionSentence(context.regions);
  if (region) parts.push(region);
  const recorded = context.species?.ranked.length ?? 0;
  if (recorded > 0)
    parts.push(`${recorded.toLocaleString()} species recorded in and around this land.`);
  else if (context.sources[PROVIDERS.species]?.status === "empty" || (context.species && !recorded))
    parts.push("No public species records turned up here.");
  if (!parts.length && context.sites?.names[0])
    parts.push(`The soil at one sample point suits ${context.sites.names[0]} plant communities.`);
  if (!parts.length && context.names?.rows.length)
    parts.push(
      `${context.names.exact} of ${context.names.rows.length} names matched the Catalogue of Life exactly.`,
    );
  if (!parts.length)
    parts.push(
      requested && !requested.includeEcoregions && !requested.includeOccurrences
        ? "Nothing was found for these choices."
        : "Nothing was found for this land.",
    );
  return parts.join(" ");
}

/** "2008–2024", or one year. */
export function yearRange(years: [number, number] | null): string | null {
  if (!years) return null;
  return years[0] === years[1] ? String(years[0]) : `${years[0]}–${years[1]}`;
}

/** Sources the run asked for that had nothing to give here, with their own reason. */
export function missingSources(context: EcologyContext): SourceStatus[] {
  return Object.values(context.sources).filter(
    (source) =>
      source.provider !== PROVIDERS.names &&
      (source.status === "uncovered" || source.status === "unavailable"),
  );
}

/** How many sources a run will look up, for its progress line. */
export function plannedSources(request: EcologyRequest | null | undefined): number {
  const value = request ?? DEFAULT_REQUEST;
  return (
    (value.includeEcoregions ? 1 : 0) +
    (value.includeEcologicalSites ? 1 : 0) +
    (value.includeOccurrences ? 1 : 0) +
    (value.taxa?.length ?? 0)
  );
}

/** Source events seen so far (each name check counts once). */
export function completedSources(events: SourceEvent[] | undefined): number {
  return (events ?? []).filter((event) => event.kind === "source").length;
}

/* ---- Field surveys -------------------------------------------------------- */

/** A percentage people read at a glance: "40%", "0.5%". */
export function percent(value: number): string {
  if (value > 0 && value < 1) return `${value.toFixed(1)}%`;
  return `${Math.round(value)}%`;
}

/** The species with the most cover in a survey's summary. */
export function topSpecies(summary: SurveySummary, count = 1) {
  return [...summary.species]
    .sort((a, b) => b.meanPercent - a.meanPercent || a.taxon.localeCompare(b.taxon))
    .slice(0, count);
}

/** The species with the most cover in one plot, as recorded. */
export function plotTopSpecies(
  plot: SurveyPlot,
  method: SurveyRead["method"],
): { taxon: string; cover: number } | null {
  let best: { taxon: string; cover: number } | null = null;
  for (const observation of plot.observations) {
    const cover =
      method === "visual-cover"
        ? observation.percentCover
        : observation.hits != null && plot.samplePoints
          ? (100 * observation.hits) / plot.samplePoints
          : null;
    if (cover == null || !observation.taxon.trim()) continue;
    if (!best || cover > best.cover) best = { taxon: observation.taxon.trim(), cover };
  }
  return best;
}

/** Every taxon recorded in a survey or draft, once each. */
export function surveyTaxa(plots: SurveyPlot[]): string[] {
  return distinct(
    plots.flatMap((plot) => plot.observations.map((observation) => observation.taxon.trim())),
  );
}

export const METHOD_LABELS: Record<SurveyRead["method"], string> = {
  "visual-cover": "Visual percent cover",
  "point-intercept": "Point intercept",
};

export const DESIGN_LABELS: Record<SurveyRead["design"], string> = {
  purposive: "Purposive plots",
  random: "Random plots",
  systematic: "Systematic plots",
  census: "Whole-boundary census",
};

export const IDENTIFICATION_LABELS: Record<string, string> = {
  tentative: "Tentative",
  verified: "Verified",
  unidentified: "Unidentified",
};

/** "12 Jun 2025" from an ISO date, without shifting the day across time zones. */
export function formatDay(value: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(value);
  if (!match) return value;
  const date = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])));
  return date.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}
