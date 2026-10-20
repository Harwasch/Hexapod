/**
 * Photo / Confidence / Data age in the app itself, under each splat renderer.
 *
 * A site with two dated captures of the synthetic yard: E1, the drone flight of Mar 1 that made
 * the site's frame, and E2, a phone video of Jun 1 (the yard's `tileset.json?epoch=2`). The
 * route gives each root an `extras.confidence` (hexapod.cellgrid v1) and serves its grid, built
 * by the TypeScript fixture encoder: E1 weak west of the yard's middle, E2 strong everywhere.
 *
 * Under CesiumJS: the three views, the key to what is drawn, the colour part in the shown
 * model's primitive (read through `window.__twin`) and out of it again on Photo, Data age's
 * "newer data here" on the older date. Under PlayCanvas and Spark: the analytic views disabled
 * with the reason, and "Draw with CesiumJS" really switching. The tool rail is as it was.
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { gzipSync } from "node:zlib";

import type { Page } from "@playwright/test";

import { encodeConfidenceGrid, REASON_BITS } from "../src/dev/cellGridFixture";

import { expect, mockApi, test } from "./fixtures";

const TILES = resolve(process.cwd(), "../../data/tiles");
const EVIDENCE = process.env.W4A_EVIDENCE_DIR;
const YARD = "/fixture-tiles/synthetic-yard/splat/tileset.json";
const SITE_ID = "99999999-9999-4999-8999-999999999999";
const E1 = "99999999-9999-4999-8999-000000000001";
const E2 = "99999999-9999-4999-8999-000000000002";
const A1 = "99999999-9999-4999-8999-0000000000a1";
const A2 = "99999999-9999-4999-8999-0000000000a2";

/** The yard's root transform's origin (synthetic_yard.py); the yard is 32 x 24 m east/north. */
const LON = -123.88;
const LAT = 46.1335;
const east = (m: number) => m / (111_320 * Math.cos((LAT * Math.PI) / 180));
const north = (m: number) => m / 111_320;
const MID = { lon: LON + east(16), lat: LAT + north(12) };

const ring = (half: number): [number, number][] => [
  [MID.lon - east(half), MID.lat - north(half)],
  [MID.lon + east(half), MID.lat - north(half)],
  [MID.lon + east(half), MID.lat + north(half)],
  [MID.lon - east(half), MID.lat + north(half)],
  [MID.lon - east(half), MID.lat - north(half)],
];

const CELLS = Array.from({ length: 16 }, (_, i) => {
  const r = 14 * Math.sqrt((i + 0.5) / 16);
  const a = i * 2.399963;
  return { lon: MID.lon + east(r * Math.cos(a)), lat: MID.lat + north(r * Math.sin(a)), height: 0 };
});

const footprint = { type: "MultiPolygon", coordinates: [[ring(40)]] };

function epoch(id: string, ordinal: number, observedAt: string, platform: string, asset: string) {
  return {
    id,
    siteId: SITE_ID,
    ordinal,
    captureId: null,
    jobId: null,
    observedAt,
    observedAtSource: "captured-at",
    footprint,
    platform,
    status: "published",
    joinMethod: ordinal === 1 ? "new-site" : "spatial",
    alignment: null,
    quality: null,
    currentAssetId: asset,
    createdAt: observedAt,
    updatedAt: observedAt,
  };
}

function asset(id: string, epochId: string, url: string, createdAt: string) {
  return {
    id,
    siteId: SITE_ID,
    provider: "3d-tiles-url",
    name: "Gaussian splat",
    representation: "gaussian-splat",
    source: { type: "3d-tiles-url", url },
    footprint: null,
    observedAt: createdAt,
    validFrom: null,
    validTo: null,
    resolution: null,
    crs: null,
    license: null,
    attribution: [],
    provenance: null,
    epochId,
    sourceKind: "observed",
    sidecarFlags: [],
    renderConfig: {
      maximumScreenSpaceError: 16,
      pointCloudShading: null,
      clipsWorld: true,
      clipFootprint: "catalog",
      heightOffsetM: 0,
      clampToGround: true,
      groundSamples: CELLS,
    },
    defaultVisible: true,
    createdAt,
    updatedAt: createdAt,
  };
}

const site = {
  id: SITE_ID,
  slug: "yard-confidence",
  name: "Yard, confidence",
  description: null,
  boundary: { type: "Polygon", coordinates: [ring(40)] },
  centroid: { longitude: MID.lon, latitude: MID.lat, height: null },
  areaM2: 6_400,
  thumbnailUrl: null,
  metadata: {},
  attribution: [],
  license: null,
  assets: [
    asset(A1, E1, YARD, "2026-03-02T00:00:00Z"),
    asset(A2, E2, `${YARD}?epoch=2`, "2026-06-02T00:00:00Z"),
  ],
  epochs: [
    epoch(E1, 1, "2026-03-01T15:00:00Z", "drone", A1),
    epoch(E2, 2, "2026-06-01T15:00:00Z", "phone-video", A2),
  ],
  frame: null,
  cameraBookmarks: [],
  createdAt: "2026-03-02T00:00:00Z",
  updatedAt: "2026-06-02T00:00:00Z",
};

const summary = {
  id: SITE_ID,
  slug: site.slug,
  name: site.name,
  description: null,
  centroid: site.centroid,
  areaM2: site.areaM2,
  thumbnailUrl: null,
  representations: ["gaussian-splat"],
  latestObservedAt: "2026-06-01T15:00:00Z",
  quality: null,
  createdAt: site.createdAt,
  updatedAt: site.updatedAt,
};

/** The two dates' grids: 1 m cells over the yard and a 2 m margin. */
async function grids() {
  const over = { level: 2, i0: -2, j0: -2, ni: 36, nj: 28, coarser: 8 };
  const march = await encodeConfidenceGrid({
    ...over,
    uri: "confidence-e1.bin",
    cell: (i) => (i < 16 ? { score: 60, reasons: REASON_BITS.views } : { score: 240 }),
  });
  const june = await encodeConfidenceGrid({
    ...over,
    uri: "confidence-e2.bin",
    cell: () => ({ score: 230 }),
  });
  return { march, june };
}

/** Routes the catalog (this site only) and the yard, each date's root given its grid. */
async function route(page: Page): Promise<void> {
  const { march, june } = await grids();
  const yard = JSON.parse(
    readFileSync(resolve(TILES, "synthetic-yard/splat/tileset.json"), "utf8"),
  ) as {
    root: { extras: Record<string, unknown> };
  };
  const withGrid = (meta: unknown) =>
    JSON.stringify({
      ...yard,
      root: { ...yard.root, extras: { ...yard.root.extras, confidence: meta } },
    });
  await mockApi(page);
  await page.route("**/api/v1/sites", (r) =>
    r.request().method() === "GET" ? r.fulfill({ status: 200, json: [summary] }) : r.fallback(),
  );
  await page.route(`**/api/v1/sites/${SITE_ID}`, (r) => r.fulfill({ status: 200, json: site }));
  await page.route("**/fixture-tiles/**", (r) => {
    const url = new URL(r.request().url());
    const relative = url.pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return r.abort();
    const json = { status: 200, contentType: "application/json" };
    const bin = { status: 200, contentType: "application/octet-stream" };
    if (relative === "synthetic-yard/splat/tileset.json") {
      const second = url.searchParams.get("epoch") === "2";
      return r.fulfill({ ...json, body: withGrid(second ? june.meta : march.meta) });
    }
    if (relative === "synthetic-yard/splat/confidence-e1.bin")
      return r.fulfill({ ...bin, body: gzipSync(march.planes, { level: 9 }) });
    if (relative === "synthetic-yard/splat/confidence-e2.bin")
      return r.fulfill({ ...bin, body: gzipSync(june.planes, { level: 9 }) });
    const file = resolve(TILES, relative);
    if (!existsSync(file)) return r.fulfill({ status: 404, body: "" });
    return r.fulfill({ ...(relative.endsWith(".json") ? json : bin), body: readFileSync(file) });
  });
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (r) => r.abort());
}

/** The colour parts in each loaded date's primitive, and which date is drawn. */
const parts = (page: Page): Promise<{ shown: string | null; parts: Record<string, string[]> }> =>
  page.evaluate(`(() => {
    const twin = window.__twin;
    twin.scene.requestRender();
    const parts = {};
    for (const a of twin.sites.loadedAssets()) {
      const chain = a.tileset.gaussianSplatPrimitive?.vertexColor;
      parts[a.assetId] = (chain?.parts ?? []).map((p) => p.colorFunction);
    }
    return { shown: twin.sites.shownAsset()?.id ?? null, parts };
  })()`);

async function arrive(page: Page, renderer: string): Promise<void> {
  await page.setViewportSize({ width: 1280, height: 760 });
  await route(page);
  await page.goto(`/?renderer=${renderer}`);
  await page.waitForFunction(() => "__twin" in window && Boolean(window.__twin), undefined, {
    timeout: 120_000,
  });
  await page.getByTestId("onboarding-explore").click();
  await page.waitForFunction(`window.__twin.sites.summaries.length === 1`, undefined, {
    timeout: 60_000,
  });
  await page.evaluate(`void window.__twin.sites.flyTo("${SITE_ID}")`);
  await page.waitForFunction(
    `window.__twin.sites.flight?.state === "landed" && !window.__twin.camera.gliding && window.__twin.sites.activeSite?.id === "${SITE_ID}"`,
    undefined,
    { timeout: 240_000 },
  );
}

async function shot(page: Page, name: string): Promise<void> {
  await page.evaluate(`window.__twin.scene.requestRender()`);
  await page.waitForTimeout(1_500);
  const path = EVIDENCE ? `${EVIDENCE}/${name}.png` : test.info().outputPath(`${name}.png`);
  await page.screenshot({ path });
  await test.info().attach(name, { path, contentType: "image/png" });
}

async function railUnchanged(page: Page): Promise<void> {
  await expect(page.getByRole("toolbar", { name: "Tools" }).getByRole("button")).toHaveText([
    ...(process.env.VITE_ENABLE_LAND_EXPLORATION === "true" ? ["Land"] : []),
    "Layers",
    "Measure",
    "Add",
    "Settings",
  ]);
}

const poll = { timeout: 120_000, intervals: [250, 500, 1000] };

test("under cesium, Confidence and Data age tint the shown date, keyed, and Photo undoes it", async ({
  page,
}) => {
  test.setTimeout(360_000);
  await arrive(page, "cesium");
  const views = page.getByTestId("views");
  await expect(views).toBeVisible();
  const radio = (name: string) => views.getByRole("radio", { name });
  for (const name of ["Photo", "Confidence", "Data age"]) await expect(radio(name)).toBeEnabled();
  await expect(radio("Photo")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("views-needs-cesium")).toHaveCount(0);
  await expect.poll(async () => (await parts(page)).shown, poll).toBe(A2);
  expect((await parts(page)).parts[A2] ?? []).not.toContain("splatCellViewColor");

  // Confidence: the key, and the part in the shown date's primitive.
  await radio("Confidence").click();
  const legend = page.getByTestId("cell-legend");
  await expect(legend).toContainText(
    "Weak: score under 128 (under half its splats pass the capture checks)",
    { timeout: 60_000 },
  );
  await expect(legend).toContainText("Strong: score 192–255");
  await expect(legend).toContainText("weak-v1");
  await expect
    .poll(async () => (await parts(page)).parts[A2] ?? [], poll)
    .toContain("splatCellViewColor");
  await shot(page, "cesium-confidence-jun");

  // Data age on the latest date: its age, and no later date.
  await radio("Data age").click();
  await expect(page.getByTestId("cell-legend-age")).toContainText(/This date: .* old/, {
    timeout: 60_000,
  });
  await expect(page.getByTestId("cell-legend-newer")).toHaveText("No later date of this site");
  await shot(page, "cesium-data-age-jun");

  // The older date: June has evidence everywhere March does, so "newer data here".
  await page
    .getByTestId("timeline")
    .getByRole("radio", { name: /Mar 1, 2026/ })
    .click();
  await expect.poll(async () => (await parts(page)).shown, poll).toBe(A1);
  await expect(page.getByTestId("cell-legend-newer")).toHaveText("Newer data here: switch date", {
    timeout: 60_000,
  });
  await expect
    .poll(async () => (await parts(page)).parts[A1] ?? [], poll)
    .toContain("splatCellViewColor");
  await shot(page, "cesium-data-age-mar");
  await radio("Confidence").click();
  await expect(legend).toContainText("Strong");
  await shot(page, "cesium-confidence-mar");

  // Photo: out of every primitive's chain again.
  await radio("Photo").click();
  await expect(legend).toHaveCount(0);
  await expect
    .poll(async () => Object.values((await parts(page)).parts).flat(), poll)
    .not.toContain("splatCellViewColor");
  await railUnchanged(page);
});

for (const renderer of ["playcanvas", "spark"] as const) {
  test(`under ${renderer}, the analytic views say they need CesiumJS and switch to it in one click`, async ({
    page,
  }) => {
    test.setTimeout(360_000);
    await arrive(page, renderer);
    const views = page.getByTestId("views");
    await expect(views).toBeVisible();
    const radio = (name: string) => views.getByRole("radio", { name });
    await expect(radio("Photo")).toBeEnabled();
    await expect(radio("Photo")).toHaveAttribute("aria-checked", "true");
    await expect(radio("Confidence")).toBeDisabled();
    await expect(radio("Data age")).toBeDisabled();
    const notice = page.getByTestId("views-needs-cesium");
    await expect(notice).toContainText(
      "Needs the CesiumJS renderer (Settings › Advanced › Splat renderer)",
    );
    await expect(page.getByTestId("cell-legend")).toHaveCount(0);
    await shot(page, `${renderer}-notice`);
    await railUnchanged(page);

    // One click: CesiumJS draws the scan, the views are offered, and Confidence works.
    await notice.getByRole("button", { name: "Draw with CesiumJS" }).click();
    await expect(notice).toHaveCount(0);
    await expect
      .poll(() => page.evaluate(`window.__twin.scanRendererStatus.active`), poll)
      .toBe(false);
    const saved = await page.evaluate(() => localStorage.getItem("twin.settings.v1") ?? "");
    expect(saved).toContain('"splatRenderer":"cesium"');
    await expect(radio("Confidence")).toBeEnabled();
    await radio("Confidence").click();
    await expect(page.getByTestId("cell-legend")).toContainText("weak-v1", { timeout: 60_000 });
    await expect
      .poll(async () => (await parts(page)).parts[A2] ?? [], poll)
      .toContain("splatCellViewColor");
    await shot(page, `${renderer}-switched-confidence`);
    await railUnchanged(page);
  });
}
