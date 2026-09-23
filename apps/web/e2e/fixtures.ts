import { test as base, expect, type Page } from "@playwright/test";

/** Mirrors the API seed closely enough for the UI; keeps e2e independent of PostGIS. */
export const demoSite = {
  id: "11111111-1111-4111-8111-111111111111",
  slug: "cesium-splat-demo",
  name: "Cesium Gaussian splat demo",
  description: "Public 3D Gaussian splat tileset with hierarchical level of detail.",
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-122.143272, 47.635222],
        [-122.120532, 47.635222],
        [-122.120526, 47.660476],
        [-122.143272, 47.660476],
        [-122.143272, 47.635222],
      ],
    ],
  },
  centroid: { longitude: -122.13810992689156, latitude: 47.644519699638366, height: 120 },
  areaM2: 4_797_797,
  thumbnailUrl: null,
  metadata: { quality: { resolutionDescription: "Sub-decimetre splat detail (visual)" } },
  attribution: [
    { text: "Cesium sample data", organization: "Cesium GS, Inc.", url: "https://cesium.com/" },
  ],
  license: {
    name: "Cesium ion sample asset",
    spdxId: null,
    url: "https://cesium.com/legal/terms-of-service/",
    requiresAttribution: true,
    notes: null,
  },
  assets: [
    {
      id: "22222222-2222-4222-8222-222222222222",
      siteId: "11111111-1111-4111-8111-111111111111",
      provider: "cesium-ion",
      name: "Gaussian splat (LOD)",
      representation: "gaussian-splat",
      source: { type: "cesium-ion", assetId: 4547222 },
      footprint: null,
      observedAt: null,
      validFrom: null,
      validTo: null,
      resolution: null,
      crs: null,
      license: null,
      attribution: [
        { text: "Cesium sample data", organization: "Cesium GS, Inc.", url: "https://cesium.com/" },
      ],
      provenance: null,
      renderConfig: {
        maximumScreenSpaceError: 16,
        pointCloudShading: null,
        clipsWorld: true,
        clipFootprint: "tileset",
        heightOffsetM: 0,
      },
      defaultVisible: true,
      createdAt: "2026-09-15T00:00:00Z",
      updatedAt: "2026-09-15T00:00:00Z",
    },
    {
      id: "33333333-3333-4333-8333-333333333333",
      siteId: "11111111-1111-4111-8111-111111111111",
      provider: "3d-tiles-url",
      name: "Mesh (test)",
      representation: "mesh",
      source: { type: "3d-tiles-url", url: "https://example.invalid/mesh/tileset.json" },
      footprint: null,
      observedAt: "2025-05-01T00:00:00Z",
      validFrom: null,
      validTo: null,
      resolution: null,
      crs: null,
      license: null,
      attribution: [],
      provenance: null,
      renderConfig: {
        maximumScreenSpaceError: 16,
        pointCloudShading: null,
        clipsWorld: true,
        clipFootprint: "catalog",
        heightOffsetM: 0,
      },
      defaultVisible: false,
      createdAt: "2026-09-15T00:00:00Z",
      updatedAt: "2026-09-15T00:00:00Z",
    },
  ],
  cameraBookmarks: [
    {
      id: "44444444-4444-4444-8444-444444444444",
      siteId: "11111111-1111-4111-8111-111111111111",
      name: "Overview",
      longitude: -122.14406,
      latitude: 47.64523,
      height: 331.3,
      heading: 100,
      pitch: -25,
      roll: 0,
      isDefault: true,
      createdAt: "2026-09-15T00:00:00Z",
    },
  ],
  createdAt: "2026-09-15T00:00:00Z",
  updatedAt: "2026-09-15T00:00:00Z",
};

export const demoSummary = {
  id: demoSite.id,
  slug: demoSite.slug,
  name: demoSite.name,
  description: demoSite.description,
  centroid: demoSite.centroid,
  areaM2: demoSite.areaM2,
  thumbnailUrl: null,
  representations: ["gaussian-splat", "mesh"],
  latestObservedAt: "2025-05-01T00:00:00Z",
  quality: {
    resolutionDescription: "Sub-decimetre splat detail (visual)",
    groundSampleDistanceM: null,
  },
  createdAt: demoSite.createdAt,
  updatedAt: demoSite.updatedAt,
};

const baseLayer = {
  description: null,
  temporalExtent: null,
  observedAt: null,
  legend: null,
  provenance: null,
  builtin: true,
  createdAt: "2026-09-15T00:00:00Z",
  updatedAt: "2026-09-15T00:00:00Z",
  render: {
    opacity: 1,
    minimumAltitudeM: null,
    maximumAltitudeM: null,
    maximumScreenSpaceError: null,
    exclusiveGroup: null,
  },
  spatialExtent: { west: -180, south: -90, east: 180, north: 90 },
  coverage: "Global",
  resolution: null,
  license: {
    name: "ODbL 1.0",
    spdxId: "ODbL-1.0",
    url: "https://opendatacommons.org/licenses/odbl/",
    requiresAttribution: true,
    notes: null,
  },
};

export const layers = [
  {
    ...baseLayer,
    id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    slug: "openstreetmap",
    name: "OpenStreetMap",
    description: "Community-maintained street map raster tiles.",
    category: "imagery",
    sourceType: "xyz",
    source: {
      type: "xyz",
      urlTemplate: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      minimumLevel: 0,
      maximumLevel: 19,
      subdomains: null,
      tileWidth: 256,
    },
    render: { ...baseLayer.render, exclusiveGroup: "basemap" },
    attribution: [
      {
        text: "© OpenStreetMap contributors",
        organization: "OpenStreetMap Foundation",
        url: "https://www.openstreetmap.org/copyright",
      },
    ],
    defaultVisible: false,
  },
  {
    ...baseLayer,
    id: "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
    slug: "esa-worldcover-2021",
    name: "ESA WorldCover 2021",
    description: "Global 10 m land-cover map.",
    category: "land-cover",
    sourceType: "wms",
    source: {
      type: "wms",
      url: "https://services.terrascope.be/wms/v2",
      layers: "WORLDCOVER_2021_MAP",
      parameters: { transparent: "true" },
    },
    observedAt: "2021-01-01T00:00:00Z",
    resolution: "10 m",
    legend: {
      title: "Land cover class",
      entries: [{ label: "Tree cover", color: "#006400" }],
      imageUrl: null,
    },
    attribution: [
      {
        text: "© ESA WorldCover project 2021",
        organization: "ESA",
        url: "https://esa-worldcover.org",
      },
    ],
    license: {
      name: "CC BY 4.0",
      spdxId: "CC-BY-4.0",
      url: "https://creativecommons.org/licenses/by/4.0/",
      requiresAttribution: true,
      notes: null,
    },
    defaultVisible: false,
  },
];

export interface MockOptions {
  /** Simulate the catalog API being down. */
  apiDown?: boolean;
  onCreateSite?: (body: unknown) => void;
  /** When set, every mutating request must carry `Authorization: Bearer <token>` or get a 401. */
  writeToken?: string;
  /**
   * Bytes per part. Tiny on purpose: 4 bytes against a 12-byte file is three parts, so a
   * handful of bytes exercises the same window-walking a 12 GB video would.
   */
  partSize?: number;
  /** Parts per presigned window — the mock's `PRESIGN_WINDOW_PARTS`. */
  windowParts?: number;
  /** Fail this part's first PUT, so the retry path has something real to resume from. */
  failPartOnce?: number;
  /**
   * Let a worker claim the queued job and walk its stages, one per poll of the job list.
   *
   * A7 built the real worker, so this is now a stand-in for something that exists rather
   * than for something that does not. Left off, the job stays `not-started` — which is
   * still an honest state: it is what a queued job looks like before any worker is free
   * to claim it.
   */
  workerRuns?: boolean;
  /** The stage the simulated worker fails on, so the retry-from-stage path has a target. */
  workerFailsAt?: string;
}

/** What the capture mock saw, so a test can assert on the wire rather than on the DOM alone. */
export interface CaptureMockState {
  captures: Record<string, unknown>[];
  jobs: Record<string, unknown>[];
  /** One entry per part PUT: which part, how many bytes, and whether it carried auth. */
  partPuts: { partNumber: number; bytes: number; authorization: string | null }[];
  /** `firstPartNumber` of every window presigned after the first one. */
  presigns: number[];
  /** The ordered parts sent to `.../complete`. */
  completed: { partNumber: number; etag: string }[][];
  /** Mutating requests that were refused for want of the write token. */
  unauthorized: string[];
  /** The capture id of every handoff minted, in order. */
  handoffs: string[];
}

const STORAGE_PREFIX = "/__storage";

const STAGES = ["normalize", "pose", "train", "package", "register"];

/**
 * A stand-in for the worker A7 has not built yet: one stage per poll of the job list.
 *
 * Only used by tests that ask for it. Left alone, a queued job stays `not-started`,
 * because that is exactly what happens against the real API today.
 */
function advanceJobs(state: CaptureMockState, poll: number, failAt?: string): void {
  for (const job of state.jobs) {
    if (job.status === "complete" || job.status === "error" || job.status === "cancelled") continue;
    // Counted from the poll the job was last queued at, not from the panel's first one:
    // a job is queued when it is created, and again when somebody retries it.
    const done = Math.max(0, poll - Number(job.createdPoll) - 1);
    // Stages before `resumeFrom` were completed by an earlier run and are not run again,
    // which is what "retry from a stage" means on the worker's side.
    const resumeFrom = Number(job.resumeFrom ?? 0);
    const reached = Math.min(resumeFrom + done, STAGES.length);
    // The failure is a first-run fact: a retry that failed again at the same stage would
    // never reach the thing the retry test is about.
    const failing = failAt && !job.retried ? STAGES.indexOf(failAt) : -1;
    const stopped = failing >= 0 && reached > failing;
    // A stopped run got as far as the stage before the one that failed.
    const settled = stopped ? failing : reached;
    job.status = stopped
      ? "error"
      : done === 0 && resumeFrom === 0
        ? "not-started"
        : reached >= STAGES.length
          ? "complete"
          : "in-progress";
    if (stopped) job.error = `stage '${failAt}' failed and will not be retried again`;
    const shown = stopped ? failing + 1 : Math.min(reached + 1, STAGES.length);
    job.steps = STAGES.slice(0, shown).map((stageId, index) => ({
      id: `${String(job.id)}-${stageId}`,
      jobId: job.id,
      stageId,
      impl: "stub",
      ordinal: index,
      attempt: 1,
      status: index < settled ? "complete" : stopped && index === failing ? "error" : "in-progress",
      startedAt: "2026-09-20T00:00:00Z",
      finishedAt: index < settled ? "2026-09-20T00:00:12Z" : null,
      preemptedAt: null,
      checkpointKey: null,
      // Logs are in object storage; the panel reads them through
      // GET /jobs/{id}/steps/{id}/log, which this mock answers below.
      logKey: `runs/${String(job.id)}/${stageId}/log.txt`,
      metrics: {},
      artifacts: [],
      createdAt: "2026-09-20T00:00:00Z",
      updatedAt: "2026-09-20T00:00:00Z",
    }));
    if (job.status === "complete") {
      job.finishedAt = "2026-09-20T00:01:00Z";
      job.durationS = 60;
      const capture = state.captures.find((c) => String(c.id) === String(job.captureId));
      if (capture) {
        capture.status = "complete";
        capture.siteId = demoSite.id;
      }
    }
  }
}

/** The worker noticing a cancel: the stage that was running stops where it is. */
function cancelJob(job: Record<string, unknown>): void {
  job.status = "cancelled";
  job.finishedAt = "2026-09-20T00:00:30Z";
  for (const step of (job.steps ?? []) as Record<string, unknown>[]) {
    if (step.status === "in-progress" || step.status === "not-started") step.status = "cancelled";
  }
}

/** Re-queue a job from one of its stages, keeping the steps before it. */
function retryJob(job: Record<string, unknown>, poll: number, fromStage: string | null): void {
  const steps = (job.steps ?? []) as Record<string, unknown>[];
  const target = fromStage
    ? steps.findIndex((step) => step.stageId === fromStage)
    : steps.findIndex((step) => step.status !== "complete");
  const from = target < 0 ? 0 : target;
  job.steps = steps.slice(0, from);
  job.resumeFrom = from;
  job.createdPoll = poll;
  job.retried = true;
  job.status = "not-started";
  job.error = null;
  job.finishedAt = null;
}

export async function mockApi(page: Page, options: MockOptions = {}): Promise<CaptureMockState> {
  // Plans approved during a test live here so list, revise and status round-trip.
  const mockPlans: Record<string, unknown>[] = [];
  const partSize = options.partSize ?? 4;
  const windowParts = options.windowParts ?? 2;
  const state: CaptureMockState = {
    captures: [],
    jobs: [],
    partPuts: [],
    presigns: [],
    completed: [],
    unauthorized: [],
    handoffs: [],
  };
  // How many times the job list has been polled: the simulated worker advances one stage
  // per poll, the way `mockPlans` mutates across calls.
  let jobPolls = 0;

  /**
   * One window of presigned parts — never the whole upload.
   *
   * The real API hands out 32 at a time (`PRESIGN_WINDOW_PARTS`) because a 12 GB video
   * is 1536 parts of URL JSON whose tail would expire before a phone reached it. The
   * mock shrinks both numbers so a 12-byte file walks the same path.
   */
  const uploadWindow = (fileId: string, partsTotal: number, firstPartNumber: number) => {
    const last = Math.min(firstPartNumber + windowParts - 1, partsTotal);
    return {
      uploadId: `mpu-${fileId}`,
      storageKey: `captures/source/${fileId}`,
      partSize,
      partsTotal,
      parts: Array.from({ length: Math.max(0, last - firstPartNumber + 1) }, (_, i) => ({
        partNumber: firstPartNumber + i,
        url: `${STORAGE_PREFIX}/${fileId}/${firstPartNumber + i}`,
      })),
      expiresIn: 3600,
      nextPartNumber: last >= partsTotal ? null : last + 1,
    };
  };

  // Parts go to "object storage", which here is a same-origin path so the test is about
  // the uploader rather than about CORS. The ETag header is the load-bearing part: the
  // real bucket has to expose it, and without it completion is impossible.
  const failedOnce = new Set<number>();
  await page.route(`**${STORAGE_PREFIX}/**`, async (route) => {
    const request = route.request();
    const partNumber = Number(new URL(request.url()).pathname.split("/").pop());
    state.partPuts.push({
      partNumber,
      bytes: (request.postDataBuffer() ?? Buffer.alloc(0)).length,
      authorization: await request.headerValue("authorization"),
    });
    if (options.failPartOnce === partNumber && !failedOnce.has(partNumber)) {
      failedOnce.add(partNumber);
      await route.fulfill({ status: 500, body: "storage said no" });
      return;
    }
    await route.fulfill({ status: 200, headers: { ETag: `"etag-${partNumber}"` }, body: "" });
  });
  // OpenStreetMap feature lookup for "find on the map": one closed water way in any view.
  await page.route("https://overpass-api.de/**", (route) => {
    // "What contains this point" gets a field around the point; a bbox query gets the lake.
    const data = decodeURIComponent(route.request().postData() ?? "");
    const isIn = /is_in\(([-\d.]+),([-\d.]+)\)/.exec(data);
    const elements = isIn
      ? [
          {
            type: "way",
            id: 7,
            tags: { name: "Test Field", landuse: "farmland" },
            geometry: (() => {
              const lat = Number(isIn[1]);
              const lon = Number(isIn[2]);
              const d = 0.003;
              return [
                { lat: lat - d, lon: lon - d },
                { lat: lat - d, lon: lon + d },
                { lat: lat + d, lon: lon + d },
                { lat: lat + d, lon: lon - d },
                { lat: lat - d, lon: lon - d },
              ];
            })(),
          },
        ]
      : [
          {
            type: "way",
            id: 4242,
            tags: { name: "Test Lake", natural: "water" },
            geometry: [
              { lat: 36.597, lon: -119.905 },
              { lat: 36.597, lon: -119.895 },
              { lat: 36.603, lon: -119.895 },
              { lat: 36.603, lon: -119.905 },
              { lat: 36.597, lon: -119.905 },
            ],
          },
        ];
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ elements }),
    });
  });
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    if (options.apiDown) {
      await route.abort("connectionrefused");
      return;
    }
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

    // The write gate A3 put on every mutating endpoint. With no token configured the API
    // leaves writes open, which is why the default here configures none.
    const mutating = request.method() !== "GET";
    if (options.writeToken && mutating) {
      const header = await request.headerValue("authorization");
      if (header !== `Bearer ${options.writeToken}`) {
        state.unauthorized.push(`${request.method()} ${path}`);
        return json(
          {
            title: "Unauthorized",
            status: 401,
            detail: "This endpoint needs the write token.",
          },
          401,
        );
      }
    }

    const handoffRoute = /^\/api\/v1\/captures\/([^/]+)\/handoff$/.exec(path);
    if (handoffRoute && request.method() === "POST") {
      const captureId = String(handoffRoute[1]);
      if (!state.captures.some((c) => String(c.id) === captureId))
        return json({ title: "Not found", status: 404 }, 404);
      state.handoffs.push(captureId);
      const expiresAt = new Date(Date.now() + 600_000).toISOString();
      return json(
        {
          captureId,
          token: `v1.mock.${captureId}`,
          url: `https://twin.example/upload.html#v1.mock.${captureId}`,
          expiresAt,
          expiresIn: 600,
          renewableUntil: expiresAt,
          qrSvg: `<svg xmlns="http://www.w3.org/2000/svg" data-capture="${captureId}"></svg>`,
        },
        201,
      );
    }

    const captureRoute =
      /^\/api\/v1\/captures(?:\/([^/]+))?(?:\/files(?:\/([^/]+))?(\/parts|\/complete|\/abort)?|(\/process))?$/.exec(
        path,
      );
    if (captureRoute) {
      const [, captureId, fileId, fileAction, process] = captureRoute;
      const capture = state.captures.find((c) => String(c.id) === captureId) as
        | (Record<string, unknown> & { files: Record<string, unknown>[]; status: string })
        | undefined;
      if (!captureId) {
        if (request.method() === "GET") return json(state.captures);
        const body = request.postDataJSON() as { name: string; kind: string };
        const created = {
          id: `capture-${state.captures.length + 1}`,
          slug: `capture-${state.captures.length + 1}`,
          name: body.name,
          kind: body.kind,
          description: null,
          status: "awaiting-files",
          siteId: null,
          device: null,
          sensor: null,
          capturedAt: null,
          temporalExtent: null,
          georefMethod: null,
          scaleSource: null,
          uncertaintyM: null,
          license: null,
          provenance: null,
          attribution: [],
          metadata: (request.postDataJSON() as { metadata?: unknown }).metadata ?? {},
          files: [] as Record<string, unknown>[],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        };
        state.captures.unshift(created);
        return json(created, 201);
      }
      if (!capture) return json({ title: "Not found", status: 404 }, 404);
      if (process) {
        const body = request.postDataJSON() as { recipe: string };
        const job = {
          id: `job-${state.jobs.length + 1}`,
          // Bookkeeping for the simulated worker; the API has no such field and the UI
          // never reads it. `resumeFrom` is how a retry keeps the stages already done.
          createdPoll: jobPolls,
          resumeFrom: 0,
          captureId: String(capture.id),
          recipe: body.recipe,
          recipeVersion: "0.1.0",
          params: {},
          status: "not-started",
          provider: null,
          tier: null,
          error: null,
          claimedBy: null,
          claimedAt: null,
          leaseExpiresAt: null,
          finishedAt: null,
          durationS: null,
          costUsd: null,
          steps: [] as Record<string, unknown>[],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        };
        state.jobs.unshift(job);
        return json(job, 202);
      }
      if (fileId && fileAction === "/complete") {
        const body = request.postDataJSON() as { parts: { partNumber: number; etag: string }[] };
        state.completed.push(body.parts);
        const file = capture.files.find((f) => String(f.id) === fileId)!;
        Object.assign(file, {
          status: "complete",
          uploadId: null,
          partsCompleted: body.parts.length,
          partsTotal: body.parts.length,
        });
        capture.status = "not-started";
        return json(file);
      }
      if (fileId && fileAction === "/parts") {
        const body = request.postDataJSON() as { firstPartNumber: number };
        state.presigns.push(body.firstPartNumber);
        const file = capture.files.find((f) => String(f.id) === fileId)!;
        return json(uploadWindow(String(file.id), Number(file.partsTotal), body.firstPartNumber));
      }
      if (fileId && fileAction === "/abort") {
        const file = capture.files.find((f) => String(f.id) === fileId)!;
        Object.assign(file, { status: "aborted", uploadId: null });
        return json(file);
      }
      if (request.method() === "POST") {
        const body = request.postDataJSON() as {
          filename: string;
          bytes: number | null;
          contentType: string | null;
        };
        const id = `file-${capture.files.length + 1}`;
        const partsTotal = Math.max(1, Math.ceil((body.bytes ?? partSize) / partSize));
        const file = {
          id,
          captureId: capture.id,
          filename: body.filename,
          contentType: body.contentType,
          bytes: body.bytes,
          checksum: null,
          storageKey: `captures/${String(capture.id)}/source/${id}/${body.filename}`,
          status: "in-progress",
          uploadId: `mpu-${id}`,
          partsCompleted: 0,
          partsTotal,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        };
        capture.files.push(file);
        return json({ file, upload: uploadWindow(id, partsTotal, 1) }, 201);
      }
      return json({
        ...capture,
        jobs: state.jobs.filter((j) => String(j.captureId) === String(capture.id)),
      });
    }

    if (path === "/api/v1/jobs" && request.method() === "GET") {
      jobPolls += 1;
      if (options.workerRuns) advanceJobs(state, jobPolls, options.workerFailsAt);
      return json(state.jobs);
    }

    const jobRoute = /^\/api\/v1\/jobs\/([^/]+)(?:\/(cancel|retry)|\/steps\/([^/]+)\/log)$/.exec(
      path,
    );
    if (jobRoute) {
      const [, jobId, action, stepId] = jobRoute;
      const job = state.jobs.find((j) => String(j.id) === jobId);
      if (!job) return json({ title: "Not found", status: 404 }, 404);
      if (action === "cancel") {
        cancelJob(job);
        return json(job);
      }
      if (action === "retry") {
        const body = request.postDataJSON() as { fromStage: string | null };
        retryJob(job, jobPolls, body.fromStage);
        return json(job);
      }
      const step = ((job.steps ?? []) as Record<string, unknown>[]).find(
        (s) => String(s.id) === stepId,
      );
      if (!step?.logKey) return json({ title: "Not found", status: 404 }, 404);
      return json({
        stepId: step.id,
        stageId: step.stageId,
        logKey: step.logKey,
        text: `$ run ${String(step.stageId)}\nwrote 1 artifact\n`,
      });
    }

    if (path === "/api/v1/health")
      return json({
        status: "ok",
        database: true,
        postgisVersion: "3.4",
        objectStorage: "none",
        objectStorageAvailable: false,
      });
    if (path === "/api/v1/sites" && request.method() === "GET") return json([demoSummary]);
    if (path === "/api/v1/sites" && request.method() === "POST") {
      const body = request.postDataJSON() as { name: string; boundary: unknown };
      options.onCreateSite?.(body);
      if (typeof body.name !== "string" || body.name.length < 2) {
        return json(
          {
            title: "Validation error",
            status: 422,
            errors: [{ loc: ["body", "name"], msg: "too short", type: "value_error" }],
          },
          422,
        );
      }
      return json(
        {
          ...demoSite,
          id: "55555555-5555-4555-8555-555555555555",
          slug: "new-site",
          name: body.name,
          assets: [],
          cameraBookmarks: [],
        },
        201,
      );
    }
    if (path === `/api/v1/sites/${demoSite.id}`) return json(demoSite);
    if (path === "/api/v1/layers" && request.method() === "GET") return json(layers);
    if (path === "/api/v1/layers" && request.method() === "POST") {
      const body = request.postDataJSON() as { name: string; source: { type: string } };
      return json(
        {
          ...layers[0],
          id: "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
          slug: "new-layer",
          name: body.name,
          sourceType: body.source.type,
          source: body.source,
          builtin: false,
          defaultVisible: false,
        },
        201,
      );
    }
    if (path === "/api/v1/ion/status")
      return json({
        configured: false,
        apiBase: "https://api.cesium.com",
        reconstruction: {
          monitorJobs: false,
          registerAssets: true,
          createJobs: false,
          createJobsReason: "sourceType for photo inputs is not documented",
        },
      });
    if (path === "/api/v1/plans" && request.method() === "GET") return json(mockPlans);
    if (path === "/api/v1/plans" && request.method() === "POST") {
      const body = request.postDataJSON() as Record<string, unknown>;
      const now = new Date().toISOString();
      const record = {
        ...body,
        id: `plan-${mockPlans.length + 1}`,
        status: "scheduled",
        revision: 1,
        revisions: [
          {
            revision: 1,
            note: "Approved",
            createdAt: now,
            title: body.title,
            machineIds: body.machineIds,
            zoneIds: body.zoneIds,
            estimates: body.estimates,
          },
        ],
        createdAt: now,
        updatedAt: now,
      };
      mockPlans.push(record);
      return json(record, 201);
    }
    const planMatch = /^\/api\/v1\/plans\/([^/]+)(\/status)?$/.exec(path);
    if (planMatch) {
      const index = mockPlans.findIndex((p) => p.id === planMatch[1]);
      if (index < 0) return json({ title: "Not found", status: 404 }, 404);
      const current = mockPlans[index] as Record<string, unknown> & {
        revision: number;
        revisions: unknown[];
      };
      if (planMatch[2] && request.method() === "PATCH") {
        const { status } = request.postDataJSON() as { status: string };
        mockPlans[index] = { ...current, status };
        return json(mockPlans[index]);
      }
      if (request.method() === "PUT") {
        const body = request.postDataJSON() as Record<string, unknown> & { note?: string };
        // The API forbids unknown fields; a revision never carries identity.
        if ("projectId" in body || "siteId" in body)
          return json({ title: "Validation error", status: 422 }, 422);
        const revision = current.revision + 1;
        mockPlans[index] = {
          ...current,
          ...body,
          revision,
          revisions: [
            ...current.revisions,
            {
              revision,
              note: body.note ?? "Revised",
              createdAt: new Date().toISOString(),
              title: body.title,
              machineIds: body.machineIds,
              zoneIds: body.zoneIds,
              estimates: body.estimates,
            },
          ],
        };
        return json(mockPlans[index]);
      }
      if (request.method() === "DELETE") {
        mockPlans.splice(index, 1);
        return route.fulfill({ status: 204 });
      }
      return json(current);
    }
    if (path === "/api/v1/agent/status")
      return json({ configured: false, provider: "rules", model: null });
    if (path === "/api/v1/agent/plan-draft" && request.method() === "POST") {
      const body = request.postDataJSON() as {
        goal: string;
        refinement?: string;
        preferredZoneIds?: string[];
        preferredMachineIds?: string[];
        answers?: Record<string, string | number | boolean>;
      };
      const answers = body.answers ?? {};
      const clarifications = [
        ...("deadline" in answers
          ? []
          : [
              {
                id: "deadline",
                question: "When should this be done?",
                kind: "range",
                options: [],
                min: 1,
                max: 90,
                step: 1,
                unit: "days",
                default: 14,
                why: "Sets the end date.",
              },
            ]),
        ...("passes" in answers
          ? []
          : [
              {
                id: "passes",
                question: "One pass, or a follow-up pass?",
                kind: "choice",
                options: [
                  { value: "one", label: "Single pass" },
                  { value: "two", label: "Two passes" },
                ],
                min: null,
                max: null,
                step: null,
                unit: null,
                default: "one",
                why: "A second pass doubles the machine-hours.",
              },
            ]),
      ];
      const hours = answers.passes === "two" ? 294 : 147;
      const zoneIds = body.preferredZoneIds?.length
        ? body.preferredZoneIds
        : ((body as { zones?: { id: string }[] }).zones ?? []).some((z) => z.id === "Z-14")
          ? ["Z-14"]
          : ((body as { zones?: { id: string }[] }).zones ?? []).slice(0, 1).map((z) => z.id);
      const machineIds = body.preferredMachineIds?.length ? body.preferredMachineIds : ["TR-04"];
      for (const id of body.refinement?.match(/TR-\d{2}/g) ?? [])
        if (!machineIds.includes(id)) machineIds.push(id);
      return json({
        title: body.goal,
        objective: body.goal,
        zoneIds,
        machineIds,
        cadence: /weekly/i.test(body.goal) ? "weekly" : "once",
        startDate: "2026-09-17",
        endDate: /weekly/i.test(body.goal) ? null : "2026-09-24",
        estimates: { acres: 220, machineHours: hours, calendarDays: 7 },
        steps: [
          {
            title: "Survey pass",
            detail: "",
            machineIds,
            zoneId: zoneIds[0],
            when: "Day 1",
            startDay: 0,
            days: 1,
          },
          {
            title: `Treat ${zoneIds[0]}`,
            detail: "Mow",
            machineIds,
            zoneId: zoneIds[0],
            when: "Day 2-6",
            startDay: 1,
            days: 5,
          },
        ],
        assumptions: ["1.5 acres per machine-hour (mock)"],
        clarifications,
        risks: [],
        questions: [],
        source: "rules",
        model: null,
        note: "Rule-based draft: no model is configured (mock).",
      });
    }
    return json({ title: "Not found", status: 404 }, 404);
  });
  // Keep the globe offline-friendly: block the heavy external tile traffic so runs are deterministic.
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.route(/https:\/\/tile\.openstreetmap\.org\/.*/, (route) => route.abort());
  await page.route(/https:\/\/services\.terrascope\.be\/.*/, (route) => route.abort());
  await page.route(
    /https:\/\/(dev\.virtualearth|ecn\.t[0-9]\.tiles\.virtualearth)\.net\/.*/,
    (route) => route.abort(),
  );
  return state;
}

export const test = base.extend<{ app: Page }>({
  app: async ({ page }, use) => {
    await mockApi(page);
    await page.addInitScript(() => {
      window.localStorage.setItem(
        "twin.settings.v1",
        JSON.stringify({
          state: { onboardingDismissed: false, quality: "performance" },
          version: 1,
        }),
      );
    });
    await page.goto("/");
    await expect(page.getByTestId("cesium-viewport")).toBeVisible();
    await expect(page.locator("canvas").first()).toBeVisible();
    await expect(page.getByTestId("status-bar")).toContainText("Alt");
    await use(page);
  },
});

export { expect };
