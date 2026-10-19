import type { Page } from "@playwright/test";

import type {
  components,
  BoundaryRevision,
  Footprint,
  LandArea,
  LandCreate,
  LandRevise,
} from "@twin/contracts";

import { workspaceMissions } from "./fixtures";

type Action = components["schemas"]["LandActionRead"];
type ActionDraft = components["schemas"]["LandActionCreate"];
type PlaceFact = components["schemas"]["PlaceFact"];

const ACRE_M2 = 4046.8564224;

/** A 1 x 1 transparent PNG. */
const CLEAR_TILE =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

/**
 * What a fixture land has had analysed: a USGS 3DEP bare-earth elevation (a gentle west half, a
 * 40 % slope over the east half) or four seasons of Sentinel-2 signatures.
 */
export type FixtureRaster = "dem" | "signature";

/** The seasons the signature fixture saw, one scene each. */
const SCENES = ["2025-01-15", "2025-04-05", "2025-07-24", "2025-10-17"].map((day, window) => ({
  window,
  period: { startDate: `${day.slice(0, 8)}01`, endDate: `${day.slice(0, 8)}30` },
  sceneId: `S2_FIXTURE_${window}`,
  acquiredAt: `${day}T18:55:00Z`,
}));

/** The Fog of Knowing's one cited record: a dated inspection of a trough. */
const FOG_RECORD = {
  key: "inspection:1",
  kind: "inspection",
  title: "Inspection: North trough, good",
  classId: "field",
  countedClassId: "field",
  classNote: null,
  resolutionM: null,
  observedAt: "2026-10-15T00:00:00Z",
  observedLabel: "inspected 2026-10-15",
  recordedAt: "2026-10-15T00:00:00Z",
  ageDays: 1,
  peakCredit: 0.845,
  arithmetic: "0.85 x 0.5^(1/120) = 0.84",
  located: true,
  positionalUncertaintyM: null,
  cells: 1,
  provenance: {
    provider: "field inspection",
    license: "Workspace record",
    attribution: "Recorded in this workspace's inventory",
    url: null,
  },
  link: { kind: "inventory", id: "feature-1", detail: {} },
};

const FOG_POLICY = {
  version: "fog-v1",
  classes: [
    {
      id: "field",
      label: "Field inspection",
      quality: 0.85,
      halfLifeDays: 120,
      maxResolutionM: null,
      example: "A person's dated inspection of a mapped feature",
    },
  ],
  undatedFactor: 0.25,
  fieldReachM: { Point: 15, LineString: 5 },
  foggyThreshold: 0.5,
  formula: "Fog density = 1 - the best credit among the records that cover a cell.",
};

const FOG_NOT_COUNTED = [
  { title: "Google Photorealistic 3D Tiles (the basemap)", reason: "No capture date." },
];

const NHD_SOURCE = {
  provider: "usgs-nhd",
  name: "USGS National Hydrography Dataset (NHD) flowlines",
  url: "https://example.test/nhd",
  linkUrl: "https://example.test/nhd",
  documentationUrl: "https://example.test/docs",
  license: "Public domain (U.S. Government work)",
  attribution: "U.S. Geological Survey",
  resolution: "1:24,000 (Flowline - Large Scale); mapped channels",
};

export function rectangle(west: number, south: number, east: number, north: number) {
  return [
    [
      [west, south],
      [east, south],
      [east, north],
      [west, north],
      [west, south],
    ],
  ];
}

const base64 = (bytes: ArrayBufferLike | Uint8Array) =>
  Buffer.from(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)).toString("base64");

/**
 * The Land API every journey here shares, answered in the page so nothing reaches a server or
 * an agency: saved land and its boundary history, the inspector's point facts, the research
 * worker's status (online, no agent model), a land as mission ground, the Fog of Knowing (full
 * fog but one inspected cell), creeks (none mapped), the workable ground for a mower (the west
 * half; the east half too steep), the analyses `rasters` lists, and plans (Land actions) with
 * their approval and the private mission scheduling one makes. What the app asked is recorded
 * for the journeys' assertions.
 */
export async function landApi(page: Page, { rasters = [] }: { rasters?: FixtureRaster[] } = {}) {
  const saved: LandArea[] = [];
  const history: BoundaryRevision[] = [];
  const asked = {
    place: [] as string[],
    workable: [] as Record<string, unknown>[],
    scouted: [] as Record<string, unknown>[],
    similar: 0,
    zones: [] as Record<string, unknown>[],
  };
  const actions = {
    current: null as Action | null,
    versions: [] as Action[],
    requests: [] as string[],
    /** The bodies of "Schedule mission". */
    scheduled: [] as Record<string, unknown>[],
  };
  /** The saved land's bounds, the frame every grid here is laid on. */
  const frame = () => {
    const ring = (saved[0]?.boundary as { coordinates: number[][][] } | undefined)
      ?.coordinates[0] ?? [[0, 0]];
    const lons = ring.map((p) => p[0] ?? 0);
    const lats = ring.map((p) => p[1] ?? 0);
    return {
      west: Math.min(...lons),
      south: Math.min(...lats),
      east: Math.max(...lons),
      north: Math.max(...lats),
    };
  };

  await page.route("**/api/v1/research/place/*", async (route) => {
    const url = new URL(route.request().url());
    const kind = url.pathname.split("/").at(-1) as PlaceFact["kind"];
    asked.place.push(kind);
    const fact: PlaceFact = {
      kind,
      status: kind === "flood-zone" ? "empty" : "available",
      label: kind === "elevation" ? "236.8 m" : kind === "flood-zone" ? "No mapped zone" : kind,
      detail: null,
      basis: "modelled-surface",
      vintage: kind === "elevation" ? "DEM acquired 2019" : null,
      caveat: "Fixture caveat",
      longitude: Number(url.searchParams.get("longitude")),
      latitude: Number(url.searchParams.get("latitude")),
      retrievedAt: "2026-10-16T00:00:00Z",
      cached: false,
      source: {
        provider: kind,
        name: `${kind} fixture`,
        url: "https://example.test/query",
        linkUrl: "https://example.test/query",
        documentationUrl: "https://example.test/docs",
        license: "Public domain",
        attribution: "Fixture",
        resolution: "Point",
      },
      data: {},
    };
    await route.fulfill({ json: fact });
  });

  await page.route("**/api/v1/research/status", async (route) => {
    await route.fulfill({
      json: {
        model: null,
        modelConfigured: false,
        queue: { queued: 0, running: 0, oldestQueuedAt: null, oldestQueuedSeconds: null },
        worker: {
          state: "online",
          busy: false,
          lastSeenAt: "2026-10-16T00:00:00Z",
          secondsSinceSeen: 12,
          checkInEverySeconds: 30,
        },
        workerRequired: true,
      },
    });
  });

  // The analyses as the API lists them: a record, its grid's metadata and each band's numbers.
  const rasterRecords = () => {
    const { west, south, east, north } = frame();
    const record = (id: string, dataset: string, metadata: Record<string, unknown>) => ({
      id,
      runId: `run-${id}`,
      landId: saved[0]?.id,
      boundaryRevision: saved[0]?.revision ?? 1,
      request: { dataset, resolutionM: 10, maxDimension: 512 },
      sha256: "a".repeat(64),
      byteSize: 4096,
      createdAt: "2026-10-16T00:00:00Z",
      stale: false,
      metadata: {
        algorithm: `${dataset}-v1`,
        method: "Fixture",
        bounds: [west, south, east, north],
        crs: "EPSG:4326",
        resolutionM: 10,
        width: 4,
        height: 4,
        coverageFraction: 1,
        sampledAreaM2: 1600,
        downloadedBytes: 4096,
        warnings: [],
        sources: [],
        ...metadata,
      },
    });
    const band = (index: number, name: string, unit: string, low: number, high: number) => ({
      index,
      name,
      unit,
      validCells: 16,
      minimum: low,
      maximum: high,
      mean: (low + high) / 2,
      standardDeviation: (high - low) / 4,
      percentiles: { "5": low, "50": (low + high) / 2, "95": high },
      histogramEdges: [low, (low + high) / 2, high],
      histogramCounts: [8, 8],
      palette: unit === "degrees" ? "magma" : "viridis",
    });
    const keys = ["green", "red", "nir", "swir16", "ndvi"] as const;
    const labels = ["Green (B03)", "Red (B04)", "NIR (B08)", "SWIR (B11)", "NDVI"];
    return rasters.map((kind) =>
      kind === "dem"
        ? record("raster-dem", "usgs-3dep-13", {
            boundaryCells: 1200,
            validCells: 1200,
            bands: [
              band(1, "Bare-earth elevation", "m", 100, 235),
              band(2, "Bare-earth slope", "degrees", 2.9, 21.8),
            ],
          })
        : record("raster-1", "sentinel-2-signature", {
            boundaryCells: 16,
            validCells: 16,
            bands: SCENES.flatMap(({ acquiredAt }, window) =>
              keys.map((key, i) =>
                band(
                  window * 5 + i + 1,
                  `${acquiredAt.slice(0, 10)} ${labels[i]}`,
                  key === "ndvi" ? "NDVI" : "reflectance",
                  0.1,
                  0.7,
                ),
              ),
            ),
            vegetation: {
              observations: SCENES.map(({ period, sceneId, acquiredAt }, window) => ({
                period,
                band: window * 5 + 5,
                sceneId,
                acquiredAt,
                candidateCount: 4,
                catalogTruncated: false,
                qualityCandidatesExamined: 2,
                validCells: 16,
                coverageFraction: 1,
                meanNdvi: 0.64,
                commonMeanNdvi: 0.64,
                qualityCounts: { Vegetation: 16 },
                explanation: "One selected acquisition.",
              })),
              commonCells: 16,
              commonCoverageFraction: 1,
              changeBand: null,
              changeCells: 0,
              meanChange: null,
            },
            signature: {
              features: SCENES.flatMap(({ acquiredAt }, window) =>
                keys.map((key, i) => ({
                  band: window * 5 + i + 1,
                  window,
                  key,
                  name: `${acquiredAt.slice(0, 10)} ${labels[i]}`,
                  commonMean: 0.3,
                  commonStandardDeviation: 0.05,
                })),
              ),
              commonCells: 16,
              commonCoverageFraction: 1,
              standardization: "z-score per feature",
              similarity: "cosine of standardized vectors",
            },
          }),
    );
  };

  // The elevation over the land: gentle (5 %) over the west half, 40 % over the east half.
  const demGrid = () => {
    const { west, south, east, north } = frame();
    const columns = 40;
    const rows = 30;
    const cellM =
      ((east - west) * 111_320 * Math.cos((((south + north) / 2) * Math.PI) / 180)) / columns;
    const values = new Float32Array(columns * rows);
    for (let row = 0; row < rows; row++)
      for (let column = 0; column < columns; column++)
        values[row * columns + column] =
          column < columns / 2
            ? 100 + column * cellM * 0.05
            : 100 + (columns / 2) * cellM * 0.05 + (column - columns / 2) * cellM * 0.4;
    return {
      west,
      south,
      east,
      north,
      columns,
      rows,
      cellM,
      encoding: "f32",
      scale: 1,
      offset: 0,
      nodata: null,
      data: base64(values.buffer),
    };
  };

  // One place's seasons, as the signature compare reads them.
  const seedOf = (longitude: number, latitude: number) => ({
    longitude,
    latitude,
    cellLongitude: longitude,
    cellLatitude: latitude,
    windows: [0.69, 0.64, 0.58, 0.65].map((ndvi, window) => ({
      window,
      acquiredAt: SCENES[window]?.acquiredAt,
      green: 0.05,
      red: 0.04,
      nir: 0.3,
      swir16: 0.2,
      ndvi,
    })),
  });

  // A part of the land's frame, by fractions of its width and height.
  const box = (w: number, s: number, e: number, n: number) => {
    const f = frame();
    const lon = (x: number) => f.west + (f.east - f.west) * x;
    const lat = (y: number) => f.south + (f.north - f.south) * y;
    return { type: "Polygon", coordinates: rectangle(lon(w), lat(s), lon(e), lat(n)) };
  };

  // What scheduling an approved action files: a private mission on the action's ground.
  const missionOf = (action: Action, body: { projectId: string; note: string }) => {
    const work = `LA-${action.id.replace(/-/g, "").slice(0, 8)}-r${action.revision}`;
    const duration = Math.max(...action.steps.map((step) => step.startDay + step.days));
    const end = new Date(`${action.startDate}T00:00:00Z`);
    end.setUTCDate(end.getUTCDate() + duration - 1);
    const estimates = {
      acres: (saved[0]?.areaM2 ?? 0) / ACRE_M2,
      machineHours: 0,
      calendarDays: duration,
    };
    return {
      id: "33333333-3333-4333-8333-333333333333",
      projectId: body.projectId,
      siteId: null,
      title: action.title,
      objective: action.objective,
      goal: action.objective,
      cadence: "once",
      startDate: action.startDate,
      endDate: end.toISOString().slice(0, 10),
      zoneIds: [work],
      machineIds: [],
      areas: [
        {
          id: work,
          name: `Approved work area · ${action.title}`,
          footprint: action.effectiveBoundary,
          attribution: `Approved land action revision ${action.revision} on boundary revision ${action.boundaryRevision}`,
        },
      ],
      steps: action.steps.map((step) => ({
        title: step.title,
        detail: step.detail ?? "",
        machineIds: step.machineIds ?? [],
        zoneId: work,
        when: `Day ${step.startDay + 1}`,
        startDay: step.startDay,
        days: step.days,
      })),
      estimates,
      assumptions: ["Machine hours are not estimated in this land action."],
      risks: [],
      questions: [],
      source: "rules",
      model: null,
      status: "scheduled",
      landId: action.landId,
      boundaryRevision: action.boundaryRevision,
      private: true,
      landActionId: action.id,
      revision: 1,
      revisions: [
        {
          revision: 1,
          note: body.note,
          createdAt: "2026-10-09T00:00:00Z",
          title: action.title,
          machineIds: [],
          zoneIds: [work],
          estimates,
        },
      ],
      createdAt: "2026-10-09T00:00:00Z",
      updatedAt: "2026-10-09T00:00:00Z",
    };
  };

  await page.route("**/api/v1/land{,/**,?*}", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();

    // Plans on the land (Land actions), their approval, and the mission scheduling files.
    if (/\/actions(\/|$)/.test(path)) {
      let { current: action } = actions;
      const { versions, requests } = actions;
      let result: unknown;
      if (method === "POST" && path.endsWith("/approve") && action) {
        requests.push("approve");
        action = {
          ...action,
          status: "approved",
          approvedBy: "operator",
          approvedAt: "2026-10-09T00:00:00Z",
          approvalNote: "Reviewed",
        };
        versions[0] = action;
        result = action;
      } else if (path.endsWith("/mission") && action) {
        if (method === "POST") {
          requests.push("schedule");
          const body = request.postDataJSON() as { projectId: string; note: string };
          actions.scheduled.push(body);
          const mission = missionOf(action, body);
          workspaceMissions(page).push(mission);
          action = { ...action, status: "scheduled", missionId: mission.id };
          versions[0] = action;
        }
        result = workspaceMissions(page).find((mission) => mission.id === action?.missionId);
      } else if (method === "POST" || method === "PUT") {
        requests.push(method === "POST" ? "draft" : "revise");
        const payload = request.postDataJSON() as ActionDraft;
        action = {
          ...payload,
          id: "22222222-2222-4222-8222-222222222222",
          landId: saved[0]?.id ?? "",
          revision: versions.length + 1,
          status: "draft",
          staleReasons: [],
          totalKnownCost: 0,
          uncostedSteps: 1,
          effectiveBoundary: saved[0]!.boundary,
          approvedBy: null,
          approvedAt: null,
          approvalNote: null,
          missionId: null,
          createdAt: "2026-10-09T00:00:00Z",
          updatedAt: "2026-10-09T00:00:00Z",
        };
        versions.unshift(action);
        result = action;
      } else
        result = path.endsWith("/revisions")
          ? versions
          : path.endsWith("/actions")
            ? action
              ? [action]
              : []
            : action;
      actions.current = action;
      await route.fulfill({ json: result });
      return;
    }

    if (path.endsWith("/ground")) {
      // The land as mission ground: its boundary only (no inventory in this fixture).
      const land = saved[0];
      await route.fulfill({
        status: land ? 200 : 404,
        json: land
          ? {
              landId: land.id,
              name: land.name,
              boundaryRevision: land.revision,
              skipped: 0,
              items: [
                {
                  zoneId: `L-${land.id.replace(/-/g, "").slice(0, 8)}`,
                  kind: "boundary",
                  featureId: null,
                  revision: land.revision,
                  name: land.name,
                  category: null,
                  status: null,
                  footprint: land.boundary,
                  line: null,
                  areaM2: land.areaM2,
                  lengthM: null,
                  bufferM: null,
                  source: `Saved boundary, revision ${land.revision} · ${land.source.label} · study area, not a verified property boundary`,
                },
              ],
            }
          : { title: "Not found", status: 404 },
      });
      return;
    }

    // The Fog of Knowing: thick fog over a 4 x 4 grid but the one inspected cell.
    const fogFrame = () => ({ ...frame(), columns: 4, rows: 4, cellM: 20 });
    if (path.endsWith("/knowledge/cell")) {
      await route.fulfill({
        json: {
          longitude: Number(url.searchParams.get("longitude")),
          latitude: Number(url.searchParams.get("latitude")),
          column: 1,
          row: 1,
          inside: true,
          density: 0.156,
          cellM: 20,
          bounds: null,
          records: [
            { record: FOG_RECORD, credit: 0.845, best: true, arithmetic: FOG_RECORD.arithmetic },
          ],
          notCounted: FOG_NOT_COUNTED,
          policyVersion: "fog-v1",
        },
      });
      return;
    }
    if (path.endsWith("/knowledge/scout-areas")) {
      asked.scouted.push(request.postDataJSON() as Record<string, unknown>);
      await route.fulfill({
        json: {
          areas: [
            {
              footprint: box(0.1, 0.1, 0.6, 0.6),
              areaM2: 25000,
              cells: 60,
              meanDensity: 1,
              bestClassCounts: { none: 60 },
              reason:
                "Foggiest 6 ac within one battery, nearest your view; no stored evidence covers it.",
            },
          ],
          threshold: 0.5,
          budgetM2: 259000,
          truncatedByBudget: true,
          forecast:
            "Forecast, not evidence: flown at 5 cm per pixel or finer and processed into a capture, these cells would read as centimetre evidence.",
          policyVersion: "fog-v1",
        },
      });
      return;
    }
    if (path.endsWith("/knowledge")) {
      const density = new Uint8Array(16).fill(250);
      density[5] = 39;
      const best = new Uint8Array(32).fill(255);
      best[10] = 0;
      best[11] = 0;
      await route.fulfill({
        json: {
          landId: saved[0]?.id,
          boundaryRevision: 1,
          asOf: "2026-10-16T00:00:00Z",
          density: {
            ...fogFrame(),
            encoding: "u8",
            scale: 0.004,
            offset: 0,
            nodata: 255,
            data: base64(density),
          },
          best: {
            ...fogFrame(),
            encoding: "u16",
            scale: 1,
            offset: 0,
            nodata: 65535,
            data: base64(best),
          },
          records: [FOG_RECORD],
          summary: {
            insideCells: 16,
            byBestClass: { field: 1 / 16, none: 15 / 16 },
            foggyFraction: 15 / 16,
            medianDensity: 1,
          },
          policy: FOG_POLICY,
          notCounted: FOG_NOT_COUNTED,
        },
      });
      return;
    }

    if (path.endsWith("/hydrography")) {
      await route.fulfill({
        json: {
          landId: saved[0]?.id,
          boundaryRevision: 1,
          status: "empty",
          message: "The USGS NHD maps no streams or ditches within 100 m of this land.",
          bufferM: 100,
          features: [],
          truncated: false,
          retrievedAt: "2026-10-16T00:00:00Z",
          source: NHD_SOURCE,
        },
      });
      return;
    }

    if (path.endsWith("/workable-ground")) {
      asked.workable.push(request.postDataJSON() as Record<string, unknown>);
      const { west, south, east, north } = frame();
      const middle = (west + east) / 2;
      await route.fulfill({
        json: {
          landId: saved[0]?.id,
          boundaryRevision: 1,
          workable: { type: "MultiPolygon", coordinates: [rectangle(west, south, middle, north)] },
          footprintM2: 40 * ACRE_M2,
          workableM2: 20 * ACRE_M2,
          removed: {
            steepM2: 20 * ACRE_M2,
            steepEdgeM2: 0,
            creekBufferM2: 0,
            sliversM2: 0,
            slivers: 0,
            noDataM2: 0,
          },
          leftOut: [
            {
              reason: "steep",
              label: "Left out: too steep",
              areaM2: 20 * ACRE_M2,
              geometry: {
                type: "MultiPolygon",
                coordinates: [rectangle(middle, south, east, north)],
              },
            },
          ],
          limits: {
            machineClass: "mower",
            label: "Robotic brush mower (assumed)",
            maxSlopePct: 25,
            creekBufferM: 15,
          },
          elevation: {
            rasterId: "raster-dem",
            dataset: "usgs-3dep-13",
            label: "USGS 3DEP 1/3″ bare-earth DEM, 10 m",
            published: "2025-08-27",
            resolutionM: 10,
            edgeM: 5,
            stale: false,
            fallback: false,
          },
          creeks: {
            status: "empty",
            label: "USGS National Hydrography Dataset (NHD) flowlines, 1:24,000",
            counted: 0,
            ephemeralIncluded: false,
            message: "The USGS NHD maps no streams or ditches within 100 m of this land.",
          },
          sources: ["Slope: USGS 3DEP 1/3″ bare-earth DEM, 10 m, raster raster-d"],
          assumptions: [],
          warnings: [],
        },
      });
      return;
    }

    // The analyses: their list, a DEM's grid, and Show once's compare and areas.
    const zonesRoute = /^\/api\/v1\/land\/rasters\/[^/]+\/similar\/zones$/.exec(path);
    if (zonesRoute) {
      const body = request.postDataJSON() as Record<string, unknown>;
      asked.zones.push(body);
      await route.fulfill({
        json: {
          rasterId: "raster-1",
          landId: saved[0]?.id,
          boundaryRevision: 1,
          stale: false,
          threshold: body.threshold,
          seed: seedOf(Number(body.longitude), Number(body.latitude)),
          zones: [
            {
              footprint: box(0, 0.5, 0.5, 1),
              areaM2: 16_000,
              cells: 4,
              meanSimilarity: 0.99,
              patches: 1,
            },
          ],
          remainder: null,
          dropped: { patches: 0, areaM2: 0 },
          unoutlined: { patches: 0, areaM2: 0 },
          protectedCells: 2,
          matchedCells: 4,
          cellAreaM2: 100,
          nativeResolutionM: 10,
          scenes: SCENES,
          method: "cosine",
          interpretation: "Similar seasonal satellite signature at 10 m. Not a species ID.",
          attribution: "Contains modified Copernicus Sentinel data",
        },
      });
      return;
    }
    if (/^\/api\/v1\/land\/rasters\/[^/]+\/similar$/.test(path)) {
      // One tap's similarity on a 4 x 4 grid over the land: a 4-cell patch north-west, a
      // 2-cell patch south-east, the rest unlike.
      asked.similar += 1;
      const scores = [250, 250, 0, 0, 250, 250, 0, 0, 0, 0, 0, 0, 0, 0, 240, 240];
      await route.fulfill({
        json: {
          rasterId: "raster-1",
          landId: saved[0]?.id,
          boundaryRevision: 1,
          stale: false,
          seed: seedOf(
            Number(url.searchParams.get("longitude")),
            Number(url.searchParams.get("latitude")),
          ),
          grid: {
            ...frame(),
            columns: 4,
            rows: 4,
            cellM: 10,
            encoding: "u8",
            scale: 0.004,
            offset: 0,
            nodata: 255,
            data: base64(Uint8Array.from(scores)),
          },
          cellAreaM2: 400,
          nativeResolutionM: 10,
          comparedCells: 16,
          boundaryCells: 16,
          scenes: SCENES,
          method: "cosine",
          interpretation: "Similar seasonal satellite signature at 10 m. Not a species ID.",
          warnings: [],
          attribution: "Contains modified Copernicus Sentinel data",
        },
      });
      return;
    }
    if (/^\/api\/v1\/land\/rasters\/[^/]+\/grid$/.test(path)) {
      await route.fulfill({ json: demGrid() });
      return;
    }
    const rasterRoute = /^\/api\/v1\/land\/rasters\/([^/]+)$/.exec(path);
    if (rasterRoute) {
      const raster = rasterRecords().find((item) => item.id === rasterRoute[1]);
      await route.fulfill(
        raster ? { json: raster } : { status: 404, json: { title: "Not found", status: 404 } },
      );
      return;
    }
    if (/^\/api\/v1\/land\/[^/]+\/rasters$/.test(path)) {
      await route.fulfill({ json: rasterRecords() });
      return;
    }
    if (/^\/api\/v1\/land\/rasters\/[^/]+\/tiles\//.test(path)) {
      // A map tile of an analysis: clear, as nothing here is measured.
      await route.fulfill({ contentType: "image/png", body: Buffer.from(CLEAR_TILE, "base64") });
      return;
    }
    if (path.startsWith("/api/v1/land/rasters/")) {
      await route.fulfill({ status: 404, json: { title: "Not found", status: 404 } });
      return;
    }

    if (
      path.endsWith("/investigations") ||
      path.endsWith("/scenarios") ||
      path.endsWith("/surveys") ||
      path.endsWith("/solar-assessments") ||
      path.endsWith("/features") ||
      path.endsWith("/features/imports") ||
      path.endsWith("/documents") ||
      path.endsWith("/documents/links")
    ) {
      await route.fulfill({ status: 200, contentType: "application/json", body: "[]" });
      return;
    }
    if (path.endsWith("/overview")) {
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ title: "Research unavailable in selection fixture", status: 503 }),
      });
      return;
    }
    if (path.endsWith("/import")) {
      const part = request.postData()?.split("\r\n\r\n")[1]?.split("\r\n--")[0];
      const boundary = JSON.parse(part ?? "{}") as Footprint;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          status: "ready",
          boundary: { type: "MultiPolygon", coordinates: [boundary.coordinates] },
          sourceCrs: "EPSG:4326",
          targetCrs: "EPSG:4326",
          layers: [],
          warnings: [],
        }),
      });
      return;
    }
    let body: unknown = saved;
    if (method === "POST" || method === "PUT") {
      const payload = request.postDataJSON() as LandCreate | LandRevise;
      const land: LandArea = {
        ...payload,
        description: payload.description ?? "",
        id: "11111111-1111-4111-8111-111111111111",
        revision: history.length + 1,
        areaM2: 100000,
        perimeterM: 1300,
        createdAt: "2026-10-09T00:00:00Z",
        updatedAt: "2026-10-09T00:00:00Z",
      };
      history.unshift({
        revision: land.revision,
        boundary: land.boundary,
        source: land.source,
        note: "Saved boundary",
        createdAt: land.updatedAt,
      });
      saved[0] = land;
      body = land;
    } else if (path.endsWith("/revisions")) body = history;
    await route.fulfill({
      status: method === "POST" ? 201 : 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });
  return { saved, history, asked, actions, frame };
}
