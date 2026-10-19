import { useEffect, useMemo, useRef, useState } from "react";
import { useQueries, useQueryClient, type UseQueryResult } from "@tanstack/react-query";

import type {
  Footprint,
  InvestigationDetail,
  LandArea,
  LandEvidence,
  LandFinding,
  ResearchArtifact,
  ResearchEvent,
  ResearchRun,
} from "@twin/contracts";
import { boundsOf } from "@twin/geo";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import type { LandPoint } from "@/state/land";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";

import { cancelResearch } from "../../researchApi";
import { useResearchProgress } from "../../useResearchProgress";
import { OVERVIEW_TITLE, useLandAnalysis, useStartAnalysis } from "../useLandAnalysis";

/**
 * The overview bundle, read as plain data for the Water and Soil lenses (and anyone else who
 * wants it). One deterministic run per boundary revision queries five public sources at the
 * land's representative point or box; this turns the run's artifacts, evidence and source
 * events into typed values, and says plainly what each source returned.
 */

export type OverviewProvider =
  "usgs-elevation" | "nasa-power" | "gbif-occurrences" | "usda-soils" | "fema-flood-zones";

export const OVERVIEW_PROVIDERS: readonly OverviewProvider[] = [
  "usgs-elevation",
  "nasa-power",
  "gbif-occurrences",
  "usda-soils",
  "fema-flood-zones",
];

/** The question the server gives every overview run (`services/research.py`). */
const OVERVIEW_QUESTION = "What is useful and interesting about this land?";

/** What a source returned: data, nothing mapped there, no coverage, or no answer. */
export type SourceStatus = "available" | "empty" | "uncovered" | "unavailable";
const STATUSES = new Set<string>(["available", "empty", "uncovered", "unavailable"]);

interface SourceSpec {
  /** What it is, in a few words. */
  name: string;
  /** Who publishes it. */
  publisher: string;
  license: string;
  attribution: string;
  documentation: string;
}

/** Mirrors `research/providers` so licences show even when the evidence page is not loaded. */
export const SOURCE_SPECS: Record<OverviewProvider, SourceSpec> = {
  "usgs-elevation": {
    name: "Elevation",
    publisher: "USGS 3D Elevation Program",
    license: "U.S. public domain",
    attribution: "U.S. Geological Survey, 3D Elevation Program",
    documentation: "https://epqs.nationalmap.gov/v1/docs",
  },
  "nasa-power": {
    name: "Climate",
    publisher: "NASA POWER",
    license: "NASA open data; acknowledge POWER and its contributing data sources",
    attribution: "NASA POWER Project, NASA Langley Research Center",
    documentation: "https://power.larc.nasa.gov/docs/services/api/temporal/climatology/",
  },
  "gbif-occurrences": {
    name: "Species records",
    publisher: "GBIF",
    license: "Record-specific CC0 or CC BY; other licenses excluded",
    attribution: "GBIF and the original occurrence data publishers",
    documentation: "https://techdocs.gbif.org/en/openapi/v1/occurrence",
  },
  "usda-soils": {
    name: "Soil",
    publisher: "USDA soil survey (SSURGO)",
    license: "U.S. public domain",
    attribution: "USDA Natural Resources Conservation Service, Soil Data Access",
    documentation: "https://sdmdataaccess.sc.egov.usda.gov/WebServiceHelp.aspx",
  },
  "fema-flood-zones": {
    name: "Flood zones",
    publisher: "FEMA National Flood Hazard Layer",
    license: "U.S. public domain",
    attribution: "Federal Emergency Management Agency, National Flood Hazard Layer",
    documentation: "https://www.fema.gov/flood-maps/national-flood-hazard-layer",
  },
};

/** Finding titles are the source names in `research/providers`; they name the provider. */
const FINDING_PROVIDERS: Record<string, OverviewProvider> = {
  "USGS elevation": "usgs-elevation",
  "NASA POWER climate and solar resource": "nasa-power",
  "GBIF biodiversity observations": "gbif-occurrences",
  "USDA mapped soil components": "usda-soils",
  "FEMA mapped flood zones": "fema-flood-zones",
};

export interface SourceState {
  provider: OverviewProvider;
  /** Null while the run has not reached the source, or when no report survives. */
  status: SourceStatus | null;
  /** The source's own one-line report from the run. */
  message: string | null;
  finding: LandFinding | null;
  evidence: LandEvidence | null;
}

type Polygonal = Footprint;

export interface FloodArea {
  id: string;
  /** FEMA zone code, e.g. "AE" or "X". */
  zone: string;
  /** FEMA zone subtype, e.g. "FLOODWAY" or "0.2 PCT ANNUAL CHANCE FLOOD HAZARD". */
  subtype: string;
  /** FEMA's special flood hazard area flag; null when FEMA did not say. */
  specialFloodHazard: boolean | null;
  studyType: string;
  mapId: string;
  recordId: string;
  /** This polygon's area inside the land. Polygons can overlap: never sum these. */
  areaM2: number;
  geometry: Polygonal;
}

export interface FloodData {
  areas: FloodArea[];
  /** FEMA returned its record limit, so some areas may be missing; null when unknown. */
  truncated: boolean | null;
  methods: string[];
  legend: string | null;
}

export interface SoilComponent {
  mapUnit: string;
  component: string;
  /** Share of the map unit, 0–100. */
  percent: number | null;
  drainage: string | null;
  hydrologicGroup: string | null;
  slopePercent: number | null;
}

export interface SoilData {
  components: SoilComponent[];
  /** Where the soil survey was sampled, when the evidence says. */
  samplePoint: LandPoint | null;
  method: string | null;
}

export interface MonthlySeries {
  title: string;
  unit: string;
  /** January to December; null where the source has no value. */
  values: (number | null)[];
  /** The source's annual figure (same unit), when it publishes one. */
  annual: number | null;
}

export interface ClimateData {
  precipitation: MonthlySeries | null;
  temperature: MonthlySeries | null;
  /** The averaging period, e.g. "2001–2020". */
  period: string | null;
  method: string | null;
}

export interface ElevationData {
  valueM: number;
  point: LandPoint | null;
}

export interface OverviewBundle {
  runId: string;
  sources: Record<OverviewProvider, SourceState>;
  flood: FloodData | null;
  soils: SoilData | null;
  climate: ClimateData | null;
  elevation: ElevationData | null;
}

const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function parseJson(text: string | undefined): Record<string, unknown> | null {
  if (!text) return null;
  try {
    return record(JSON.parse(text));
  } catch {
    return null;
  }
}

function finite(value: unknown): number | null {
  const number = typeof value === "string" && value.trim() ? Number(value) : value;
  return typeof number === "number" && Number.isFinite(number) ? number : null;
}

function text(value: unknown): string {
  return typeof value === "string" ? value : typeof value === "number" ? String(value) : "";
}

function point(value: unknown): LandPoint | null {
  if (!Array.isArray(value)) return null;
  const lon = finite(value[0]);
  const lat = finite(value[1]);
  return lon !== null && lat !== null && Math.abs(lon) <= 180 && Math.abs(lat) <= 90
    ? [lon, lat]
    : null;
}

function isPolygonal(geometry: { type: string }): geometry is Polygonal {
  return geometry.type === "Polygon" || geometry.type === "MultiPolygon";
}

/** The run whose results a lens shows: the newest finished one, else the newest. */
export function shownRun(runs: readonly ResearchRun[]): ResearchRun | null {
  const finished = runs.filter((run) => run.status === "succeeded" || run.status === "partial");
  return finished.at(-1) ?? runs.at(-1) ?? null;
}

/** Provider of each evidence record in the run, from the evidence page and the findings. */
function evidenceProviders(
  detail: InvestigationDetail,
  runId: string,
  extra: readonly LandEvidence[],
): Map<string, OverviewProvider> {
  const providers = new Map<string, OverviewProvider>();
  for (const finding of detail.findings) {
    const provider = FINDING_PROVIDERS[finding.title];
    if (finding.runId !== runId || !provider) continue;
    for (const id of finding.evidenceIds) providers.set(id, provider);
  }
  for (const item of [...detail.evidence, ...extra])
    if (item.runId === runId && (OVERVIEW_PROVIDERS as readonly string[]).includes(item.provider))
      providers.set(item.id, item.provider as OverviewProvider);
  return providers;
}

function artifactProvider(
  artifact: ResearchArtifact,
  providers: Map<string, OverviewProvider>,
): OverviewProvider | null {
  for (const id of artifact.evidenceIds) {
    const provider = providers.get(id);
    if (provider) return provider;
  }
  // The overview's titles are fixed (`research/outputs.py`); they stand in when evidence is paged.
  if (/flood/i.test(artifact.title)) return "fema-flood-zones";
  if (/soil/i.test(artifact.title)) return "usda-soils";
  if (artifact.output.kind === "chart") return "nasa-power";
  return null;
}

/** The evidence ids a lens needs read in full: one per overview source, from its finding. */
export function overviewEvidenceIds(detail: InvestigationDetail, runId: string): string[] {
  return detail.findings
    .filter((finding) => finding.runId === runId && FINDING_PROVIDERS[finding.title])
    .flatMap((finding) => finding.evidenceIds.slice(0, 1));
}

function readFlood(
  artifacts: readonly ResearchArtifact[],
  evidence: LandEvidence | null,
): FloodData | null {
  const map = artifacts.find((item) => item.output.kind === "map");
  const table = artifacts.find((item) => item.output.kind === "table");
  if (map?.output.kind !== "map") return null;
  const rows = table?.output.kind === "table" ? table.output.rows : [];
  const aligned = rows.length === map.output.features.length;
  const areas: FloodArea[] = [];
  map.output.features.forEach((feature, index) => {
    if (!isPolygonal(feature.geometry)) return;
    const row = aligned ? rows[index] : undefined;
    const label = /^Zone\s+(.+?)\s*·\s*(.*)$/.exec(feature.label);
    const flag = row?.specialFloodHazard;
    areas.push({
      id: `${map.id}:${index}`,
      zone: text(row?.zone) || (label?.[1] ?? "Unknown"),
      subtype: text(row?.subtype) || (label?.[2] ?? ""),
      specialFloodHazard:
        flag === true || flag === "T" ? true : flag === false || flag === "F" ? false : null,
      studyType: text(row?.studyType),
      mapId: text(row?.mapId),
      recordId: text(row?.recordId),
      areaM2: Math.max(0, finite(row?.intersectedAreaM2) ?? finite(feature.value) ?? 0),
      geometry: feature.geometry,
    });
  });
  const snapshot = parseJson(evidence?.excerpt);
  const truncated =
    typeof snapshot?.truncated === "boolean"
      ? snapshot.truncated
      : areas.length >= 100
        ? true
        : null;
  return {
    areas,
    truncated,
    methods: [map.method, ...(table ? [table.method] : [])],
    legend: map.output.legend || null,
  };
}

function readSoils(
  artifacts: readonly ResearchArtifact[],
  evidence: LandEvidence | null,
): SoilData | null {
  const table = artifacts.find((item) => item.output.kind === "table");
  if (table?.output.kind !== "table") return null;
  const components = table.output.rows
    .map((row): SoilComponent => ({
      mapUnit: text(row.muname),
      component: text(row.compname),
      percent: finite(row.comppct_r),
      drainage: text(row.drainagecl) || null,
      hydrologicGroup: text(row.hydgrp) || null,
      slopePercent: finite(row.slope_r),
    }))
    .sort((a, b) => (b.percent ?? -1) - (a.percent ?? -1));
  const snapshot = parseJson(evidence?.excerpt);
  let samplePoint = point(snapshot?.samplePoint);
  if (!samplePoint && evidence) {
    // A long excerpt is cut at 30k characters; the point is near its end.
    const match = /"samplePoint":\s*\[\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\]/.exec(evidence.excerpt);
    samplePoint = match ? point([match[1], match[2]]) : null;
  }
  return { components, samplePoint, method: table.method };
}

function chartSeries(artifact: ResearchArtifact): MonthlySeries | null {
  if (artifact.output.kind !== "chart") return null;
  const series = artifact.output.series[0];
  if (!series) return null;
  const labels = artifact.output.labels.map((label) => label.slice(0, 3).toUpperCase());
  const values = MONTHS.map((month, index) => {
    const at = labels.indexOf(month);
    return finite(series.values[at >= 0 ? at : index]);
  });
  return { title: artifact.title, unit: artifact.output.unit, values, annual: null };
}

function readClimate(
  artifacts: readonly ResearchArtifact[],
  evidence: LandEvidence | null,
): ClimateData | null {
  const snapshot = parseJson(evidence?.excerpt);
  const parameters = record(record(snapshot?.properties)?.parameter);
  const definitions = record(snapshot?.parameters);
  const header = record(snapshot?.header);
  const fromSnapshot = (key: string): MonthlySeries | null => {
    const values = record(parameters?.[key]);
    if (!values) return null;
    const definition = record(definitions?.[key]);
    const fill = finite(header?.fill_value);
    const clean = (value: unknown) => {
      const number = finite(value);
      return number === null || number === fill ? null : number;
    };
    return {
      title: text(definition?.longname) || key,
      unit: text(definition?.units),
      values: MONTHS.map((month) => clean(values[month])),
      annual: clean(values.ANN),
    };
  };
  const charts = artifacts.filter((item) => item.output.kind === "chart");
  const find = (pattern: RegExp) =>
    charts.find((item) =>
      pattern.test(`${item.title} ${item.output.kind === "chart" ? item.output.yLabel : ""}`),
    );
  const merge = (chart: ResearchArtifact | undefined, key: string): MonthlySeries | null => {
    const fromChart = chart ? chartSeries(chart) : null;
    const raw = fromSnapshot(key);
    if (!fromChart) return raw;
    const unit = fromChart.unit ? fromChart.unit : (raw?.unit ?? "");
    return { ...fromChart, unit, annual: raw?.annual ?? null };
  };
  const precipitation = merge(find(/precip/i), "PRECTOTCORR");
  const temperature = merge(find(/temperature/i), "T2M");
  if (!precipitation && !temperature) return null;
  const start = text(header?.start);
  const end = text(header?.end);
  const period =
    start && end ? `${start.slice(0, 4)}–${end.slice(0, 4)}` : text(header?.range) || null;
  return { precipitation, temperature, period, method: charts[0]?.method ?? null };
}

function readElevation(
  evidence: LandEvidence | null,
  finding: LandFinding | null,
): ElevationData | null {
  const snapshot = parseJson(evidence?.excerpt);
  const location = record(snapshot?.location);
  const fromSnapshot = finite(snapshot?.value);
  const fromSummary = finding
    ? finite(/(-?[\d,]+(?:\.\d+)?)\s*m elevation/.exec(finding.summary)?.[1]?.replace(/,/g, ""))
    : null;
  const valueM = fromSnapshot !== null && fromSnapshot > -999_999 ? fromSnapshot : fromSummary;
  if (valueM === null) return null;
  return { valueM, point: point([location?.x, location?.y]) };
}

/**
 * Reads one overview run into plain data. Pure: the lens hook feeds it the investigation
 * detail, the run's events and any evidence records fetched beyond the detail page.
 */
export function readOverviewBundle(
  detail: InvestigationDetail,
  runId: string,
  events: readonly ResearchEvent[],
  extraEvidence: readonly LandEvidence[] = [],
): OverviewBundle {
  const providers = evidenceProviders(detail, runId, extraEvidence);
  const evidenceById = new Map<string, LandEvidence>();
  for (const item of [...detail.evidence, ...extraEvidence])
    if (item.runId === runId) evidenceById.set(item.id, item);
  const artifacts = new Map<OverviewProvider, ResearchArtifact[]>();
  for (const artifact of detail.artifacts) {
    if (artifact.runId !== runId) continue;
    const provider = artifactProvider(artifact, providers);
    if (provider) artifacts.set(provider, [...(artifacts.get(provider) ?? []), artifact]);
  }
  const reports = new Map<OverviewProvider, { status: SourceStatus | null; message: string }>();
  for (const event of events) {
    if (event.kind !== "source") continue;
    const provider = text(event.payload.provider);
    if (!(OVERVIEW_PROVIDERS as readonly string[]).includes(provider)) continue;
    const status = text(event.payload.status);
    reports.set(provider as OverviewProvider, {
      status: STATUSES.has(status) ? (status as SourceStatus) : null,
      message: text(event.payload.message),
    });
  }
  const sources = Object.fromEntries(
    OVERVIEW_PROVIDERS.map((provider) => {
      const finding =
        detail.findings.find(
          (item) => item.runId === runId && FINDING_PROVIDERS[item.title] === provider,
        ) ?? null;
      const evidenceId =
        finding?.evidenceIds[0] ?? artifacts.get(provider)?.[0]?.evidenceIds[0] ?? null;
      const evidence =
        (evidenceId ? evidenceById.get(evidenceId) : undefined) ??
        [...evidenceById.values()].find((item) => item.provider === provider) ??
        null;
      const report = reports.get(provider);
      const found = finding !== null || (artifacts.get(provider)?.length ?? 0) > 0;
      const state: SourceState = {
        provider,
        status: report?.status ?? (found ? "available" : null),
        message: report && report.message.length > 0 ? report.message : null,
        finding,
        evidence,
      };
      return [provider, state];
    }),
  ) as Record<OverviewProvider, SourceState>;
  return {
    runId,
    sources,
    flood: readFlood(artifacts.get("fema-flood-zones") ?? [], sources["fema-flood-zones"].evidence),
    soils: readSoils(artifacts.get("usda-soils") ?? [], sources["usda-soils"].evidence),
    climate: readClimate(artifacts.get("nasa-power") ?? [], sources["nasa-power"].evidence),
    elevation: readElevation(sources["usgs-elevation"].evidence, sources["usgs-elevation"].finding),
  };
}

const COMPASS = [
  "east",
  "northeast",
  "north",
  "northwest",
  "west",
  "southwest",
  "south",
  "southeast",
];

/**
 * Where a spot lies on the land, in words: "the middle" when it is close to the centre,
 * else a compass side ("the southwest"). Offsets are scaled by the land's own extent.
 */
export function sideOfLand(spot: LandPoint, boundary: Footprint): string {
  const b = boundsOf(boundary);
  const midLon = (b.west + b.east) / 2;
  const midLat = (b.south + b.north) / 2;
  const halfWidth = Math.max(1e-9, (b.east - b.west) / 2);
  const halfHeight = Math.max(1e-9, (b.north - b.south) / 2);
  const x = (spot[0] - midLon) / halfWidth;
  const y = (spot[1] - midLat) / halfHeight;
  if (Math.hypot(x, y) < 0.35) return "the middle";
  const sector = Math.round(Math.atan2(y, x) / (Math.PI / 4));
  return `the ${COMPASS[(sector + 8) % 8] ?? "middle"}`;
}

const NO_EVENTS: readonly ResearchEvent[] = [];

function combineEvidence(results: UseQueryResult<LandEvidence>[]): LandEvidence[] {
  return results.flatMap((result) => (result.data ? [result.data] : []));
}

/**
 * The overview bundle for a lens: the saved analysis and its state, the shown run read into
 * plain data, live progress while it runs, and the actions a person can take (start it, try
 * a failed run again, stop it). Nothing starts on its own.
 */
export function useOverviewBundle(land: LandArea) {
  const analysis = useLandAnalysis(land, OVERVIEW_TITLE);
  const starter = useStartAnalysis(land);
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit() && ready;
  const cache = useQueryClient();
  const detail = analysis.detail;
  const latest = analysis.run;
  const shown = shownRun(detail?.runs ?? []);
  const shownLive = shown?.status === "queued" || shown?.status === "running";
  const shownEvents = useResearchProgress(shown?.id, shownLive);
  const latestEvents = useResearchProgress(
    latest && latest.id !== shown?.id ? latest.id : undefined,
    analysis.running,
  );

  const loaded = new Set(detail?.evidence.map((item) => item.id) ?? []);
  const missing =
    detail && shown ? overviewEvidenceIds(detail, shown.id).filter((id) => !loaded.has(id)) : [];
  const extraEvidence = useQueries({
    queries: missing.map((id) => ({
      queryKey: ["land-research", scope, "evidence", id],
      queryFn: () =>
        unwrap(
          api.GET("/api/v1/research/evidence/{evidence_id}", {
            params: { path: { evidence_id: id } },
          }),
        ),
      enabled: ready,
      retry: false,
      staleTime: Infinity,
    })),
    combine: combineEvidence,
  });
  // Evidence records never change; keep one array per set of ids so the bundle (and the map
  // layers built from it) keep their identity between renders.
  const extraKey = extraEvidence.map((item) => item.id).join(",");
  const extra = useMemo(() => extraEvidence, [extraKey]);

  const events = shownEvents.data ?? NO_EVENTS;
  const bundle = useMemo(
    () => (detail && shown ? readOverviewBundle(detail, shown.id, events, extra) : null),
    [detail, shown, events, extra],
  );

  const progressEvents =
    (latest && latest.id !== shown?.id ? latestEvents.data : shownEvents.data) ?? NO_EVENTS;
  const done = new Set(
    progressEvents
      .filter((event) => event.kind === "source")
      .map((event) => text(event.payload.provider))
      .filter((provider) => (OVERVIEW_PROVIDERS as readonly string[]).includes(provider)),
  ).size;

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A retried request after a lost response reuses its key, so the worker runs it once.
  const retryKey = useRef<{ after: string; key: string } | null>(null);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const refresh = () => cache.invalidateQueries({ queryKey: ["land-research", scope] });
  const act = async (work: () => Promise<unknown>) => {
    if (!canEdit || busy || starter.busy) return;
    setBusy(true);
    setError(null);
    try {
      await work();
      await refresh();
    } catch (cause) {
      if (alive.current) setError(describeError(cause));
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  /** Runs the bundle again on the same boundary, after a failure or an unreachable source. */
  const retry = () =>
    act(async () => {
      const investigation = analysis.investigation;
      if (!investigation || analysis.running) return;
      const after = latest?.id ?? "none";
      const key = retryKey.current?.after === after ? retryKey.current.key : crypto.randomUUID();
      retryKey.current = { after, key };
      await unwrap(
        api.POST("/api/v1/research/investigations/{investigation_id}/runs", {
          params: { path: { investigation_id: investigation.id } },
          body: { kind: "overview", question: OVERVIEW_QUESTION, requestKey: key },
        }),
      );
      retryKey.current = null;
    });

  /** Stops the running check; whatever it already found stays. */
  const stop = () =>
    act(async () => {
      if (latest && analysis.running) await cancelResearch(latest.id);
    });

  return {
    loading: analysis.loading,
    loadError: analysis.error,
    investigation: analysis.investigation,
    /** The newest run, whatever its state. */
    latest,
    /** The run whose results are shown (the newest finished one). */
    shown,
    running: analysis.running,
    failed: analysis.failed,
    stale: analysis.stale,
    bundle,
    progress: { done, total: OVERVIEW_PROVIDERS.length },
    eventsMessage: shownEvents.message ?? null,
    canEdit,
    busy: busy || starter.busy,
    error: error ?? starter.error,
    /** Starts the bundle for the current boundary (also the way to refresh a stale one). */
    start: () => void starter.startOverview(),
    retry: () => void retry(),
    stop: () => void stop(),
    reload: () => void refresh(),
  };
}

export type OverviewBundleState = ReturnType<typeof useOverviewBundle>;
