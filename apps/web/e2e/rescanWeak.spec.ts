/**
 * "Plan a re-scan of weak areas" in the app itself, under the default renderer (PlayCanvas)
 * and CesiumJS: the flow reads only the API and mission zones, so it must not depend on what
 * draws the splats.
 *
 * A site with one dated capture of the synthetic yard. The weak-areas route is mocked: two
 * patches over the yard, an oblique and a nadir one, as the API outlines them (weak-v1). From
 * the Inspector's Date shown section the button asks for them, and the plan composer opens
 * on the site's project with them as its ground, drafting a survey. The tool rail is as it
 * was.
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { Page } from "@playwright/test";

import { expect, mockApi, test } from "./fixtures";

const TILES = resolve(process.cwd(), "../../data/tiles");
const EVIDENCE = process.env.W4B_EVIDENCE_DIR;
const YARD = "/fixture-tiles/synthetic-yard/splat/tileset.json";
const SITE_ID = "99999999-9999-4999-8999-999999999998";
const E1 = "99999999-9999-4999-8999-000000000011";
const A1 = "99999999-9999-4999-8999-0000000000b1";

/** The yard's root transform's origin (synthetic_yard.py); the yard is 32 x 24 m east/north. */
const LON = -123.88;
const LAT = 46.1335;
const east = (m: number) => m / (111_320 * Math.cos((LAT * Math.PI) / 180));
const north = (m: number) => m / 111_320;
const MID = { lon: LON + east(16), lat: LAT + north(12) };

/** A square `half` metres either side of a point `dx`, `dy` metres from the yard's middle. */
const ring = (half: number, dx = 0, dy = 0): [number, number][] => {
  const lon = MID.lon + east(dx);
  const lat = MID.lat + north(dy);
  return [
    [lon - east(half), lat - north(half)],
    [lon + east(half), lat - north(half)],
    [lon + east(half), lat + north(half)],
    [lon - east(half), lat + north(half)],
    [lon - east(half), lat - north(half)],
  ];
};

const CELLS = Array.from({ length: 16 }, (_, i) => {
  const r = 14 * Math.sqrt((i + 0.5) / 16);
  const a = i * 2.399963;
  return { lon: MID.lon + east(r * Math.cos(a)), lat: MID.lat + north(r * Math.sin(a)), height: 0 };
});

const site = {
  id: SITE_ID,
  slug: "yard-rescan",
  name: "Yard, re-scan",
  description: null,
  boundary: { type: "Polygon", coordinates: [ring(40)] },
  centroid: { longitude: MID.lon, latitude: MID.lat, height: null },
  areaM2: 6_400,
  thumbnailUrl: null,
  metadata: {},
  attribution: [],
  license: null,
  assets: [
    {
      id: A1,
      siteId: SITE_ID,
      provider: "3d-tiles-url",
      name: "Gaussian splat",
      representation: "gaussian-splat",
      source: { type: "3d-tiles-url", url: YARD },
      footprint: null,
      observedAt: "2026-03-02T00:00:00Z",
      validFrom: null,
      validTo: null,
      resolution: null,
      crs: null,
      license: null,
      attribution: [],
      provenance: null,
      epochId: E1,
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
      createdAt: "2026-03-02T00:00:00Z",
      updatedAt: "2026-03-02T00:00:00Z",
    },
  ],
  epochs: [
    {
      id: E1,
      siteId: SITE_ID,
      ordinal: 1,
      captureId: null,
      jobId: null,
      observedAt: "2026-03-01T15:00:00Z",
      observedAtSource: "captured-at",
      footprint: { type: "MultiPolygon", coordinates: [[ring(40)]] },
      platform: "drone",
      status: "published",
      joinMethod: "new-site",
      alignment: null,
      quality: null,
      currentAssetId: A1,
      createdAt: "2026-03-01T15:00:00Z",
      updatedAt: "2026-03-01T15:00:00Z",
    },
  ],
  frame: null,
  cameraBookmarks: [],
  createdAt: "2026-03-02T00:00:00Z",
  updatedAt: "2026-03-02T00:00:00Z",
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
  latestObservedAt: "2026-03-01T15:00:00Z",
  quality: null,
  createdAt: site.createdAt,
  updatedAt: site.updatedAt,
};

/** What the API would answer for the yard's weak west half (shape: WeakAreasRead). */
const weakAreas = {
  epochId: E1,
  assetId: A1,
  policy: "weak-v1",
  cellM: 1,
  level: 2,
  weakCells: 320,
  totalCells: 896,
  truncatedByBudget: false,
  heuristic: "Patterns are UX rules of thumb, not measurements.",
  areas: [
    {
      footprint: { type: "Polygon", coordinates: [ring(6, -8, 0)] },
      areaM2: 144,
      cells: 144,
      meanScore: 60,
      reasons: { views: 120, spread: 24 },
      reliefM: 1.5,
      diameterM: 17,
      pattern: "oblique",
      targetGsdM: null,
      gimbalPitchDeg: -45,
      hint: "Seen from too few views: fly two crossing passes with the camera tilted",
    },
    {
      footprint: { type: "Polygon", coordinates: [ring(4, -10, 9)] },
      areaM2: 64,
      cells: 64,
      meanScore: 90,
      reasons: { gsd: 64 },
      reliefM: 0.5,
      diameterM: 11,
      pattern: "nadir",
      targetGsdM: 0.01,
      gimbalPitchDeg: -90,
      hint: "Too coarse: fly lower for 1 cm per pixel",
    },
  ],
};

/** Routes the catalog (this site only), the yard, and the weak areas; records what is asked. */
async function route(page: Page): Promise<unknown[]> {
  const asked: unknown[] = [];
  await mockApi(page);
  await page.route("**/api/v1/sites", (r) =>
    r.request().method() === "GET" ? r.fulfill({ status: 200, json: [summary] }) : r.fallback(),
  );
  await page.route(`**/api/v1/sites/${SITE_ID}`, (r) => r.fulfill({ status: 200, json: site }));
  await page.route(`**/api/v1/sites/${SITE_ID}/epochs/${E1}/weak-areas`, (r) => {
    if (r.request().method() !== "POST") return r.fallback();
    asked.push(r.request().postDataJSON());
    return r.fulfill({ status: 200, json: weakAreas });
  });
  await page.route("**/fixture-tiles/**", (r) => {
    const url = new URL(r.request().url());
    const relative = url.pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return r.abort();
    const file = resolve(TILES, relative);
    if (!existsSync(file)) return r.fulfill({ status: 404, body: "" });
    const contentType = relative.endsWith(".json")
      ? "application/json"
      : "application/octet-stream";
    return r.fulfill({ status: 200, contentType, body: readFileSync(file) });
  });
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (r) => r.abort());
  return asked;
}

async function arrive(page: Page, renderer: string): Promise<unknown[]> {
  await page.setViewportSize({ width: 1280, height: 760 });
  const asked = await route(page);
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
  return asked;
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

for (const renderer of ["playcanvas", "cesium"] as const) {
  test(`under ${renderer}, Date shown plans a re-scan of the weak areas in the composer`, async ({
    page,
  }) => {
    test.setTimeout(360_000);
    const asked = await arrive(page, renderer);

    // What's here, on the yard's building (its collision solid picks as the site): the
    // site's card and its Date shown section.
    await page.mouse.click(528, 215, { button: "right" });
    await page
      .getByRole("menu", { name: "Map" })
      .getByRole("menuitem", { name: "What's here" })
      .click();
    const epoch = page.getByTestId("inspector-epoch");
    await expect(epoch).toBeVisible({ timeout: 30_000 });
    await epoch.getByRole("button", { name: "Plan a re-scan of weak areas" }).click();

    // The composer, on the site's project, with the two patches as its ground.
    const card = page.getByTestId("plan-card");
    await expect(card).toBeVisible({ timeout: 30_000 });
    await expect(card).toContainText("Re-scan the weak ground from the air at", {
      timeout: 30_000,
    });
    // The note counts both patches; the ground row names the first (largest) of them.
    await expect(card).toContainText("2 weak areas");
    await expect(card).toContainText("chosen by Confidence map weak-v1");
    await expect(card.getByTestId("plan-ground")).toContainText("Weak area 1");
    // Each patch flies its own height (the nadir one lower, for its detail): said, not hidden.
    await expect(card.getByTestId("scout-mixed")).toBeVisible();
    expect(asked).toHaveLength(1);
    expect(asked[0]).toMatchObject({ maxAreas: 3 });
    await shot(page, `${renderer}-rescan-composer`);
    await railUnchanged(page);
  });
}
