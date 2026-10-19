/**
 * Shared fixtures for the Water and Soil lens tests: one saved overview bundle on a square of
 * land, shaped like the API returns it (`research/outputs.py`, `providers/us_land.py`).
 */
import { vi } from "vitest";

import type {
  InvestigationDetail,
  LandArea,
  LandEvidence,
  LandFinding,
  ResearchArtifact,
  ResearchEvent,
  ResearchRun,
} from "@twin/contracts";
import { footprintAreaM2 } from "@twin/geo";

import { api } from "@/api/client";

const WEST = -77.06;
const EAST = -77.05;
const SOUTH = 38.88;
const NORTH = 38.89;
const MID = (WEST + EAST) / 2;

const square = (w: number, s: number, e: number, n: number) => ({
  type: "Polygon" as const,
  coordinates: [
    [
      [w, s],
      [e, s],
      [e, n],
      [w, n],
      [w, s],
    ],
  ],
});

const boundary = square(WEST, SOUTH, EAST, NORTH);

export const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Creek field",
  description: "",
  revision: 2,
  areaM2: footprintAreaM2(boundary),
  perimeterM: 4000,
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
  source: { method: "drawn", label: "Fixture", meaning: "study-area" },
  boundary,
};

export const investigation = {
  id: "22222222-2222-4222-8222-222222222222",
  landId: land.id,
  title: "Land overview",
  question: "What is useful and interesting about this land?",
  boundaryRevision: 2,
  createdAt: "2026-10-10T10:00:00Z",
  stale: false,
};

export const RUN_ID = "33333333-3333-4333-8333-333333333333";

export function run(status: ResearchRun["status"], id = RUN_ID): ResearchRun {
  return {
    id,
    investigationId: investigation.id,
    question: investigation.question,
    kind: "overview",
    status,
    budget: {},
    attempt: 1,
    error: status === "failed" ? "The worker stopped." : null,
    createdAt: "2026-10-10T10:00:00Z",
    startedAt: "2026-10-10T10:00:01Z",
    finishedAt: status === "running" ? null : "2026-10-10T10:01:00Z",
  };
}

function evidence(id: string, provider: string, excerpt: unknown): LandEvidence {
  return {
    id,
    runId: RUN_ID,
    provider,
    title: `${provider} record`,
    url: `https://example.org/${provider}`,
    license: "U.S. public domain",
    attribution: `${provider} attribution`,
    retrievedAt: "2026-10-10T10:00:30Z",
    excerpt: JSON.stringify(excerpt),
    spatialRelevance: "within",
    relevanceNote: "A representative interior point was sampled.",
  };
}

export const SOIL_POINT: [number, number] = [-77.0551, 38.8852];
export const ELEVATION_POINT: [number, number] = [-77.0553, 38.8854];

export const EVIDENCE = {
  elevation: evidence("e-elev", "usgs-elevation", {
    location: { x: ELEVATION_POINT[0], y: ELEVATION_POINT[1] },
    value: "57.3",
  }),
  climate: evidence("e-climate", "nasa-power", {
    header: { start: "20010101", end: "20201231", fill_value: -999 },
    parameters: {
      PRECTOTCORR: { longname: "Precipitation Corrected", units: "mm/day" },
      T2M: { longname: "Temperature at 2 Meters", units: "C" },
    },
    properties: {
      parameter: {
        PRECTOTCORR: { JAN: 4, ANN: 2.9 },
        T2M: { JAN: 1.2, ANN: 13.4 },
      },
    },
  }),
  soils: evidence("e-soil", "usda-soils", {
    query: "SELECT …",
    records: [],
    samplePoint: SOIL_POINT,
    truncated: false,
  }),
  fema: evidence("e-fema", "fema-flood-zones", { invalidFeaturesSkipped: 0, truncated: false }),
};

function finding(title: string, id: string, evidenceId: string, summary: string): LandFinding {
  return {
    id,
    runId: RUN_ID,
    title,
    summary,
    category: "physical",
    evidenceIds: [evidenceId],
    confidence: "supported",
    uncertainty: "",
    disposition: "visible",
    suggestedQuestions: [`Question about ${title}`],
  };
}

export const FINDINGS = {
  elevation: finding(
    "USGS elevation",
    "f-elev",
    "e-elev",
    "USGS estimates 57.3 m elevation at a sample point inside the land.",
  ),
  climate: finding("NASA POWER climate and solar resource", "f-climate", "e-climate", "Climate."),
  soils: finding("USDA mapped soil components", "f-soil", "e-soil", "Mapped soils."),
  fema: finding("FEMA mapped flood zones", "f-fema", "e-fema", "Zones AE, X."),
};

/** Wet winters, dry summers: monthly daily averages in mm/day. */
const RAIN = [4, 3.6, 3.2, 2.4, 1.6, 0.8, 0.4, 0.5, 1, 2, 3.5, 4.2];
const TEMPERATURE = [1.2, 2.5, 6.8, 12.4, 17.6, 22.3, 24.9, 24.1, 20.2, 13.8, 8.1, 3.0];
const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

function chart(id: string, title: string, unit: string, values: number[]): ResearchArtifact {
  return {
    id,
    runId: RUN_ID,
    title,
    method: "NASA POWER monthly climatology at a representative point.",
    evidenceIds: ["e-climate"],
    output: {
      kind: "chart",
      chartType: "bar",
      xLabel: "Month",
      yLabel: title,
      unit,
      labels: MONTHS,
      series: [{ label: title, values }],
    },
  };
}

export const WEST_HALF = square(WEST, SOUTH, MID, NORTH);
export const SHADED_X = square(MID, SOUTH, EAST, SOUTH + 0.002);
const MINIMAL_X = square(MID, SOUTH + 0.002, EAST, NORTH);

const half = land.areaM2 / 2;
const ZONES = [
  // The same AE polygon twice, as overlapping FEMA panels return it: it must not count twice.
  { zone: "AE", subtype: "", sfha: "T", geometry: WEST_HALF, area: half },
  { zone: "AE", subtype: "FLOODWAY", sfha: "T", geometry: WEST_HALF, area: half },
  {
    zone: "X",
    subtype: "0.2 PCT ANNUAL CHANCE FLOOD HAZARD",
    sfha: "F",
    geometry: SHADED_X,
    area: footprintAreaM2(SHADED_X),
  },
  {
    zone: "X",
    subtype: "AREA OF MINIMAL FLOOD HAZARD",
    sfha: "F",
    geometry: MINIMAL_X,
    area: footprintAreaM2(MINIMAL_X),
  },
];

export function floodArtifacts(zones = ZONES): ResearchArtifact[] {
  return [
    {
      id: "a-flood-map",
      runId: RUN_ID,
      title: "Flood zones intersecting the land",
      method: "Returned FEMA NFHL polygons clipped to the pinned land boundary.",
      evidenceIds: ["e-fema"],
      output: {
        kind: "map",
        unit: "m²",
        legend: "Mapped zone intersections; not a forecast.",
        features: zones.map((zone) => ({
          label: `Zone ${zone.zone} · ${zone.subtype}`,
          geometry: zone.geometry,
          value: zone.area,
        })),
      },
    },
    {
      id: "a-flood-table",
      runId: RUN_ID,
      title: "Flood-map intersection records",
      method: "Each row is one returned mapped polygon clipped to the land.",
      evidenceIds: ["e-fema"],
      output: {
        kind: "table",
        columns: [
          { key: "zone", label: "Zone" },
          { key: "subtype", label: "Zone description" },
        ],
        rows: zones.map((zone, index) => ({
          zone: zone.zone,
          subtype: zone.subtype,
          specialFloodHazard: zone.sfha,
          studyType: "NP",
          recordId: `FLD-${index}`,
          mapId: "11001C",
          intersectedAreaM2: zone.area,
        })),
      },
    },
  ];
}

export function soilArtifact(
  rows: Record<string, string | number | null>[] = [
    {
      mukey: "123",
      muname: "Hagerstown silt loam, 3 to 8 percent slopes",
      compname: "Hagerstown",
      comppct_r: 85,
      drainagecl: "Somewhat poorly drained",
      hydgrp: "C",
      slope_r: 5,
    },
    {
      mukey: "123",
      muname: "Hagerstown silt loam, 3 to 8 percent slopes",
      compname: "Duffield",
      comppct_r: 10,
      drainagecl: "Well drained",
      hydgrp: "B",
      slope_r: 4,
    },
  ],
): ResearchArtifact {
  return {
    id: "a-soil",
    runId: RUN_ID,
    title: "Soil map-unit components at the sample point",
    method: "Major components of the soil survey map unit at a representative point.",
    evidenceIds: ["e-soil"],
    output: { kind: "table", columns: [], rows },
  };
}

export function sourceEvents(
  statuses: Partial<Record<string, string>> = {},
  only?: string[],
): ResearchEvent[] {
  const providers = only ?? [
    "usgs-elevation",
    "nasa-power",
    "gbif-occurrences",
    "usda-soils",
    "fema-flood-zones",
  ];
  return providers.map((provider, index) => ({
    sequence: index + 2,
    kind: "source",
    createdAt: "2026-10-10T10:00:10Z",
    payload: {
      provider,
      status: statuses[provider] ?? "available",
      message: `${provider} report`,
    },
  }));
}

export function fullDetail(overrides: Partial<InvestigationDetail> = {}): InvestigationDetail {
  return {
    investigation,
    page: { offset: 0, limit: 100, totals: {} },
    runs: [run("succeeded")],
    evidence: Object.values(EVIDENCE),
    findings: Object.values(FINDINGS),
    artifacts: [
      chart("a-rain", "Precipitation Corrected", "mm/day", RAIN),
      chart("a-temp", "Temperature at 2 Meters", "C", TEMPERATURE),
      soilArtifact(),
      ...floodArtifacts(),
    ],
    messages: [],
    ...overrides,
  };
}

export interface ApiFixture {
  investigations?: (typeof investigation)[];
  detail?: InvestigationDetail;
  events?: ResearchEvent[];
  evidence?: Record<string, LandEvidence>;
}

const ok = (data: unknown) => Promise.resolve({ data, response: new Response("{}") });

/** Answers the lens's GETs from the fixture and records every POST. */
export function mockApi(fixture: ApiFixture) {
  const get = vi.spyOn(api, "GET").mockImplementation((path: string, init?: unknown) => {
    if (path === "/api/v1/land/{land_id}/investigations")
      return ok(fixture.investigations ?? [investigation]);
    if (path === "/api/v1/research/investigations/{investigation_id}")
      return ok(fixture.detail ?? fullDetail());
    if (path === "/api/v1/research/runs/{run_id}/events") {
      const after =
        (init as { params?: { query?: { after?: number } } } | undefined)?.params?.query?.after ??
        0;
      return ok((fixture.events ?? sourceEvents()).filter((event) => event.sequence > after));
    }
    if (path === "/api/v1/research/evidence/{evidence_id}") {
      const id = (init as { params: { path: { evidence_id: string } } }).params.path.evidence_id;
      return ok(fixture.evidence?.[id]);
    }
    if (path === "/api/v1/research/runs/{run_id}/stream")
      return Promise.resolve({ data: undefined, response: new Response(null, { status: 404 }) });
    return ok([]);
  });
  const post = vi.spyOn(api, "POST").mockImplementation((path: string) => {
    if (path === "/api/v1/land/{land_id}/overview")
      return ok({ investigation, run: run("queued") });
    if (path === "/api/v1/research/runs/{run_id}/cancel") return ok(run("cancelled"));
    return ok(run("queued", "44444444-4444-4444-8444-444444444444"));
  });
  return { get, post };
}
