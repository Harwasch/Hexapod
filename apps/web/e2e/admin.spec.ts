/**
 * The data console, driven against a mocked API.
 *
 * `admin.html` is a separate Vite entry, so nothing in `app.spec.ts` loads a line of it —
 * these are the only tests that run this page at all. The last one asserts the property
 * the separate entry exists for: that a full-page data browser does not pull in a 3D
 * globe, which `phoneUpload.spec.ts` asserts the same way for the phone page.
 */
import { expect, test, type Page } from "@playwright/test";

const CAPTURE_A = "11111111-1111-4111-8111-111111111111";
const CAPTURE_B = "11111111-1111-4111-8111-111111111112";
const GHOST_RUN = "99999999-9999-4999-8999-999999999999";

const captures = [
  {
    id: CAPTURE_A,
    slug: "back-paddock",
    name: "Back paddock",
    description: null,
    status: "complete",
    kind: "gaussian-splat",
    device: "iPhone 15 Pro",
    sensor: "Scaniverse",
    capturedAt: "2026-09-18T09:00:00Z",
    temporalExtent: null,
    georefMethod: "manual",
    scaleSource: "unresolved",
    uncertaintyM: 10,
    metadata: {},
    attribution: [],
    license: null,
    provenance: null,
    siteId: "33333333-3333-4333-8333-333333333333",
    files: [
      {
        id: "44444444-4444-4444-8444-444444444444",
        captureId: CAPTURE_A,
        filename: "scan.ply",
        contentType: "application/octet-stream",
        bytes: 2_400_000,
        checksum: "sha256:3f9a",
        storageKey: `captures/${CAPTURE_A}/source/scan.ply`,
        status: "complete",
        uploadId: null,
        partsTotal: 1,
        partsCompleted: 1,
        createdAt: "2026-09-18T09:05:00Z",
        updatedAt: "2026-09-18T09:06:00Z",
      },
    ],
    createdAt: "2026-09-18T09:05:00Z",
    updatedAt: "2026-09-18T09:10:00Z",
  },
  {
    id: CAPTURE_B,
    slug: "north-gate",
    name: "North gate",
    description: null,
    status: "not-started",
    kind: "video",
    device: null,
    sensor: null,
    capturedAt: null,
    temporalExtent: null,
    georefMethod: null,
    scaleSource: null,
    uncertaintyM: null,
    metadata: {},
    attribution: [],
    license: null,
    provenance: null,
    siteId: null,
    files: [],
    createdAt: "2026-09-19T09:05:00Z",
    updatedAt: "2026-09-19T09:05:00Z",
  },
];

function step(jobId: string, stageId: string, seconds: number) {
  return {
    id: `${jobId}-${stageId}`,
    jobId,
    stageId,
    ordinal: 1,
    impl: "splat_tiles",
    status: "complete",
    startedAt: "2026-09-20T10:00:00Z",
    finishedAt: new Date(Date.parse("2026-09-20T10:00:00Z") + seconds * 1000).toISOString(),
    metrics: {},
    logKey: `runs/${jobId}/${stageId}/log.txt`,
    attempt: 1,
    checkpointKey: null,
    preemptedAt: null,
    artifacts: [],
    createdAt: "2026-09-20T10:00:00Z",
    updatedAt: "2026-09-20T10:01:00Z",
  };
}

function run(id: string, maxGaussians: number, seconds: number) {
  return {
    id,
    captureId: CAPTURE_A,
    recipe: "splat-ingest",
    recipeVersion: "2",
    params: { package: { max_gaussians: maxGaussians } },
    status: "complete",
    provider: "modal",
    tier: "l4",
    finishedAt: "2026-09-20T10:02:00Z",
    durationS: seconds,
    costUsd: "0.12",
    error: null,
    claimedBy: "worker-1",
    claimedAt: "2026-09-20T10:00:00Z",
    leaseExpiresAt: null,
    steps: [step(id, "package", seconds)],
    createdAt: "2026-09-20T09:59:00Z",
    updatedAt: "2026-09-20T10:02:00Z",
  };
}

const RUN_FULL = "55555555-5555-4555-8555-555555555551";
const RUN_SMALL = "55555555-5555-4555-8555-555555555552";
const jobs = [run(RUN_FULL, 400000, 30), run(RUN_SMALL, 50000, 12)];

const artifacts = [
  {
    id: "66666666-6666-4666-8666-666666666661",
    jobStepId: `${RUN_FULL}-package`,
    kind: "3d-tiles",
    storageKey: `runs/${RUN_FULL}/package/splat`,
    bytes: 175_000,
    checksum: "sha256:aa11",
    contentType: "application/json",
    createdAt: "2026-09-20T10:01:00Z",
    updatedAt: "2026-09-20T10:01:00Z",
    stageId: "package",
    impl: "splat_tiles",
    jobId: RUN_FULL,
    recipe: "splat-ingest",
    captureId: CAPTURE_A,
    captureName: "Back paddock",
    references: [
      {
        kind: "site-asset",
        siteId: "33333333-3333-4333-8333-333333333333",
        siteSlug: "back-paddock",
        label: "Back paddock splat",
      },
    ],
  },
  {
    id: "66666666-6666-4666-8666-666666666662",
    jobStepId: `${RUN_SMALL}-package`,
    kind: "3d-tiles",
    storageKey: `runs/${RUN_SMALL}/package/splat`,
    bytes: 42_000,
    checksum: "sha256:bb22",
    contentType: "application/json",
    createdAt: "2026-09-20T10:02:00Z",
    updatedAt: "2026-09-20T10:02:00Z",
    stageId: "package",
    impl: "splat_tiles",
    jobId: RUN_SMALL,
    recipe: "splat-ingest",
    captureId: CAPTURE_A,
    captureName: "Back paddock",
    references: [],
  },
];

const reconciliation = {
  prefixes: ["captures/", "runs/"],
  scanned: 5,
  truncated: false,
  matched: 4,
  bytesScanned: 220_000,
  bytesOrphaned: 64,
  orphans: [
    {
      key: `runs/${GHOST_RUN}/train/point_cloud.ply`,
      bytes: 64,
      lastModified: "2026-09-19T22:14:00Z",
      etag: "d41d8cd9",
      reason: `under runs/${GHOST_RUN}: no artifact, log or checkpoint row claims it`,
    },
  ],
  rowsChecked: 3,
  missing: [
    {
      kind: "artifact",
      id: "66666666-6666-4666-8666-666666666662",
      key: `runs/${RUN_SMALL}/package/splat`,
      label: "package · 3d-tiles",
      captureId: CAPTURE_A,
      jobId: RUN_SMALL,
    },
  ],
  checkedAt: "2026-09-21T08:00:00Z",
};

const catalogue = {
  recipes: [
    {
      name: "splat-ingest",
      version: "2",
      description: "Package an already-reconstructed splat and register it as a site.",
      inputs: ["upload"],
      stages: [
        { id: "normalize", impl: "ingest_splat", params: {}, gpu: null },
        {
          id: "package",
          impl: "splat_tiles",
          params: { max_gaussians: 400000, opacity_min: 0.02, geometric_error: 2.0 },
          gpu: null,
        },
      ],
    },
  ],
  providers: [
    {
      name: "modal",
      label: "Modal",
      tiers: ["l4", "a100"],
      usdPerHourA100: 2.5,
      usdPerHour: { l4: 0.7992, a100: 2.5 },
      interruptible: false,
      note: "Per-second billing, scale to zero. The reliable default.",
    },
  ],
};

/** The back paddock's splat, whose objects a republish could not carry onto new tiles. */
const assets = [
  {
    id: "66666666-6666-4666-8666-666666666666",
    siteId: "33333333-3333-4333-8333-333333333333",
    provider: "3d-tiles-url",
    name: "Back paddock splat",
    representation: "gaussian-splat",
    source: { type: "3d-tiles-url", url: "https://tiles.example.com/runs/x/tileset.json" },
    footprint: null,
    observedAt: null,
    validFrom: null,
    validTo: null,
    resolution: null,
    crs: null,
    license: null,
    attribution: [],
    provenance: null,
    renderConfig: {},
    defaultVisible: true,
    sidecarFlags: [
      {
        kind: "instances",
        action: "Objects need re-segmenting",
        reason: "the run published new tiles whose positions are not all ones it was bound to",
        jobId: RUN_FULL,
        flaggedAt: "2026-09-20T10:05:00Z",
      },
    ],
    createdAt: "2026-09-18T09:10:00Z",
    updatedAt: "2026-09-20T10:05:00Z",
  },
];

/** Every read the console makes, answered from the fixtures above. */
async function mockApi(page: Page, launched: { body: unknown }[]) {
  await page.route("**/api/v1/captures?**", async (route) => {
    await route.fulfill({ json: captures });
  });
  await page.route("**/api/v1/jobs?**", async (route) => {
    await route.fulfill({ json: jobs });
  });
  await page.route("**/api/v1/artifacts?**", async (route) => {
    await route.fulfill({ json: artifacts });
  });
  await page.route("**/api/v1/sites", async (route) => {
    await route.fulfill({
      json: [
        { id: "33333333-3333-4333-8333-333333333333", slug: "back-paddock", name: "Back paddock" },
      ],
    });
  });
  await page.route("**/api/v1/assets", async (route) => {
    await route.fulfill({ json: assets });
  });
  await page.route("**/api/v1/storage/reconciliation**", async (route) => {
    await route.fulfill({ json: reconciliation });
  });
  await page.route("**/api/v1/recipes", async (route) => {
    await route.fulfill({ json: catalogue });
  });
  await page.route("**/api/v1/captures/*/process", async (route) => {
    launched.push({ body: route.request().postDataJSON() as unknown });
    await route.fulfill({ json: run("77777777-7777-4777-8777-777777777777", 120000, 8) });
  });
}

test.describe("the data console", () => {
  test("lists every capture with its files, bytes and checksums", async ({ page }) => {
    await mockApi(page, []);
    await page.goto("/admin.html");

    await expect(page.getByRole("heading", { name: "Data console" })).toBeVisible();
    await expect(page.getByTestId("capture-row")).toHaveCount(2);
    await expect(page.getByTestId("captures-table")).toContainText("Back paddock");
    await expect(page.getByTestId("captures-table")).toContainText("Scaniverse");
    // People's words for the API's values: the kind and the state, not "gaussian-splat".
    const rows = page.getByTestId("capture-row");
    await expect(rows.first()).toContainText("Splat");
    await expect(rows.first()).toContainText("Done");
    await expect(rows.nth(1)).toContainText("Ready");
    await expect(page.getByTestId("captures-table")).not.toContainText("gaussian-splat");
    await expect(page.getByTestId("captures-table")).not.toContainText("not-started");

    // The source files are one click down, with the checksum and the storage key that
    // reconciliation is about.
    await page.getByRole("button", { name: "Show Back paddock" }).click();
    await expect(page.getByText("sha256:3f9a")).toBeVisible();
    await expect(page.getByText(`captures/${CAPTURE_A}/source/scan.ply`)).toBeVisible();
  });

  test("says which scan lost its objects to a republish, beside the capture", async ({ page }) => {
    await mockApi(page, []);
    await page.goto("/admin.html");

    const card = page.getByTestId("flagged-assets");
    await expect(card).toContainText("Back paddock splat");
    await expect(card).toContainText("Objects need re-segmenting");
    const row = page.getByTestId("capture-row").first();
    await expect(row.getByTestId("sidecar-flag")).toHaveText("Objects need re-segmenting");
    await expect(row.getByTestId("sidecar-flag")).toHaveAttribute("title", /positions/);
    await expect(page.getByTestId("capture-row").nth(1).getByTestId("sidecar-flag")).toHaveCount(0);
  });

  test("reconciles storage against the database in both directions", async ({ page }) => {
    await mockApi(page, []);
    await page.goto("/admin.html");

    // Not on a timer: it walks a bucket, so it is asked for.
    await page.getByTestId("reconcile").click();

    const orphans = page.getByTestId("orphans");
    await expect(orphans).toContainText(`runs/${GHOST_RUN}/train/point_cloud.ply`);
    await expect(orphans).toContainText("no artifact, log or checkpoint row claims it");

    const missing = page.getByTestId("missing");
    await expect(missing).toContainText(`runs/${RUN_SMALL}/package/splat`);
    await expect(missing).toContainText("package · 3d-tiles");
  });

  test("compares two runs of one capture side by side", async ({ page }) => {
    await mockApi(page, []);
    await page.goto("/admin.html#/runs");

    await expect(page.getByTestId("run-row")).toHaveCount(2);
    await expect(page.getByTestId("run-status").first()).toHaveText("Done");
    await page.getByRole("checkbox").first().check();
    await page.getByRole("checkbox").nth(1).check();

    const comparison = page.getByTestId("comparison");
    await expect(comparison).toBeVisible();
    // The parameter that differs, in two columns, next to each other.
    await expect(comparison.getByRole("row", { name: /package · max_gaussians/ })).toContainText(
      "400000",
    );
    await expect(comparison.getByRole("row", { name: /package · max_gaussians/ })).toContainText(
      "50000",
    );
    // And the things that did not differ are on the same screen, so the comparison is
    // controlled rather than assumed: same recipe, same version, same capture.
    await expect(comparison.getByRole("row", { name: /Recipe/ })).toContainText("splat-ingest v2");
    await expect(page.getByTestId("cross-capture")).toHaveCount(0);
  });

  test("launches a run with an overridden parameter", async ({ page }) => {
    const launched: { body: unknown }[] = [];
    await mockApi(page, launched);
    await page.goto("/admin.html");

    await page.getByTestId("run-back-paddock").click();
    await expect(page.getByTestId("new-run-sheet")).toBeVisible();

    // The recipe's own default is the placeholder; typing over it is the override.
    const field = page.getByTestId("param-package-max_gaussians");
    await expect(field).toHaveAttribute("placeholder", "400000");
    await field.fill("120000");
    await page.getByTestId("run-provider").selectOption("modal");
    await page.getByTestId("run-tier").selectOption("l4");

    await page.getByTestId("launch").click();

    await expect.poll(() => launched.length).toBe(1);
    expect(launched[0]?.body).toEqual({
      recipe: "splat-ingest",
      // Only the changed parameter: the untouched ones stay the recipe's, so two runs
      // that differ in one value do not look like two runs that differ in three.
      params: { package: { max_gaussians: 120000 } },
      provider: "modal",
      tier: "l4",
    });
  });

  test("shows the estimate from the plan's own cost table", async ({ page }) => {
    await mockApi(page, []);
    await page.goto("/admin.html");
    await page.getByTestId("run-back-paddock").click();
    // splat-ingest declares no GPU stage, so there is no GPU time to pay for.
    await expect(page.getByTestId("estimate")).toContainText("under a minute");
    await expect(page.getByTestId("estimate")).toContainText("no GPU time");
  });

  test("marks the artifact nothing points at as the cleanup list", async ({ page }) => {
    await mockApi(page, []);
    await page.goto("/admin.html#/outputs");

    await expect(page.getByTestId("output-row")).toHaveCount(2);
    await expect(page.getByTestId("output-row").first()).toContainText("3D Tiles");
    await page.getByTestId("outputs-filter").selectOption("unreferenced");
    await expect(page.getByTestId("output-row")).toHaveCount(1);
    await expect(page.getByTestId("outputs-table")).toContainText(
      `runs/${RUN_SMALL}/package/splat`,
    );
  });

  test("shares the product's header: back to the globe, across to the scans", async ({ page }) => {
    await mockApi(page, []);
    await page.goto("/admin.html");
    const bar = page.getByTestId("product-bar");
    await expect(bar.getByRole("link", { name: "Globe" })).toHaveAttribute("href", "/");
    await expect(bar.getByRole("link", { name: "Scans" })).toHaveAttribute("href", "/view.html");
    await expect(bar.getByRole("link", { name: "Data console" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    // The shared tokens and fonts: the same face the globe sets its words in.
    const font = await bar.evaluate((el) => getComputedStyle(el).fontFamily);
    expect(font).toContain("Instrument Sans");
  });

  test("the page does not load CesiumJS", async ({ page }) => {
    const requests: string[] = [];
    page.on("request", (request) => requests.push(request.url()));

    await mockApi(page, []);
    await page.goto("/admin.html");
    await expect(page.getByRole("heading", { name: "Data console" })).toBeVisible();
    await expect(page.getByTestId("captures-table")).toBeVisible();

    // The entire reason admin.html is a third entry: a table of runs has no use for a
    // globe, and the globe is the single biggest thing in this repository's bundle.
    //
    // Every request, not only scripts: the engine's stylesheet, a worker or a file under
    // CESIUM_BASE_URL would each mean the console reached the engine. What this test cannot
    // see is chunking: the dev server serves modules one by one, so it proves the console's
    // *source* imports no CesiumJS, not that the *build* keeps it out. That failed once (a
    // shared `tslib` inside `cesium-*.js` made the built page modulepreload all of it) while
    // this test passed; `scripts/check-bundle.mjs` reads dist/ in CI and covers that half.
    expect(requests.filter((url) => /cesium/i.test(new URL(url).pathname))).toEqual([]);
  });
});
