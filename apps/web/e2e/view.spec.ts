/**
 * The scan viewer (view.html): a finished capture on its own, rendered with Spark.
 *
 * The render test serves the repository's committed synthetic-tree tile, so the path it
 * exercises is the real one: tileset.json -> splat.glb -> the SPZ inside it -> Spark.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test } from "@playwright/test";

const SITE = "d446c3d1-40c6-4659-991f-c80aedf6d706";
const TILES = resolve(import.meta.dirname, "../../../data/tiles/synthetic-tree/splat");
const TILESET = "https://tiles.example/sites/synthetic-tree/splat/tileset.json";

const capture = {
  id: "c1",
  name: "Garden tree",
  siteId: SITE,
  createdAt: "2026-09-24T15:00:00Z",
  metadata: { origin: "phone-key" },
  files: [],
  status: "complete",
};

test.use({ viewport: { width: 390, height: 844 } });

test.beforeEach(async ({ page }) => {
  await page.route(
    (url) => url.pathname === "/api/v1/captures",
    (route) => route.fulfill({ json: [capture] }),
  );
  await page.route(
    (url) => url.pathname === "/api/v1/sites",
    (route) =>
      route.fulfill({
        json: [
          {
            id: SITE,
            name: "Synthetic tree",
            representations: ["gaussian-splat"],
            thumbnailUrl: null,
            createdAt: "2026-09-24T15:00:00Z",
          },
          // A seeded demo site with no capture behind it: not one of your scans.
          {
            id: "demo",
            name: "Demo",
            representations: ["gaussian-splat"],
            thumbnailUrl: null,
            createdAt: "2026-09-20T00:00:00Z",
          },
        ],
      }),
  );
  await page.route(
    (url) => url.pathname === `/api/v1/sites/${SITE}`,
    (route) =>
      route.fulfill({
        json: {
          id: SITE,
          name: "Garden tree",
          createdAt: "2026-09-24T15:00:00Z",
          assets: [
            { representation: "gaussian-splat", source: { type: "3d-tiles-url", url: TILESET } },
          ],
        },
      }),
  );
  await page.route("https://tiles.example/**", async (route) => {
    const name = new URL(route.request().url()).pathname.split("/").pop() ?? "";
    await route.fulfill({
      body: readFileSync(resolve(TILES, name)),
      headers: { "access-control-allow-origin": "*" },
    });
  });
});

test("the gallery lists your scans and not the demo sites", async ({ page }) => {
  await page.goto("/view.html");
  await expect(page.locator("#gallery-status")).toContainText("1 scan");
  const card = page.locator(".card");
  await expect(card).toHaveCount(1);
  await expect(card).toContainText("Garden tree");
  await expect(card).toHaveAttribute("href", `#${SITE}`);
});

test("a scan opens on its own and renders", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`/view.html#${SITE}`);
  await expect(page.locator("#scan-name")).toHaveText("Garden tree");
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn", { timeout: 30_000 });
  await expect(page.locator("#viewer canvas")).toBeVisible();
  expect(errors).toEqual([]);
});

/** A coverage_enu.ply as tools/pipeline/quality.py writes it: points, tier, colour. */
function coveragePly(points: [number, number, number, number][]): Buffer {
  const header = Buffer.from(
    [
      "ply",
      "format binary_little_endian 1.0",
      `element vertex ${String(points.length)}`,
      "property float x",
      "property float y",
      "property float z",
      "property uchar red",
      "property uchar green",
      "property uchar blue",
      "property uchar tier",
      "end_header",
      "",
    ].join("\n"),
  );
  const body = Buffer.alloc(points.length * 16);
  points.forEach(([x, y, z, tier], i) => {
    body.writeFloatLE(x, i * 16);
    body.writeFloatLE(y, i * 16 + 4);
    body.writeFloatLE(z, i * 16 + 8);
    body.writeUInt8(tier === 2 ? 64 : 200, i * 16 + 12);
    body.writeUInt8(150, i * 16 + 13);
    body.writeUInt8(100, i * 16 + 14);
    body.writeUInt8(tier, i * 16 + 15);
  });
  return Buffer.concat([header, body]);
}

test("a scan with no coverage offers no Coverage toggle", async ({ page }) => {
  await page.goto(`/view.html#${SITE}`);
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn", { timeout: 30_000 });
  await expect(page.getByRole("button", { name: "Coverage" })).toBeHidden();
});

test("Coverage lays the quality bar's cloud and the camera path over the scan", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const coverageUrl = "https://tiles.example/runs/job/place/coverage_enu.ply";
  await page.route(
    (url) => url.pathname === `/api/v1/sites/${SITE}`,
    (route) =>
      route.fulfill({
        json: {
          id: SITE,
          name: "Garden tree",
          createdAt: "2026-09-24T15:00:00Z",
          metadata: { captureId: "c1", coverageUrl, keepVerifiedPct: 47.6 },
          assets: [
            { representation: "gaussian-splat", source: { type: "3d-tiles-url", url: TILESET } },
          ],
        },
      }),
  );
  let fetched = 0;
  await page.route(coverageUrl, async (route) => {
    fetched += 1;
    await route.fulfill({
      body: coveragePly([
        [0, 0, 1, 2],
        [0.5, 0, 1, 2],
        [1, 1, 1, 1],
        [3, 3, 0, 0],
        [4, 0, 1.5, 3],
        [0, 4, 1.5, 3],
      ]),
      headers: { "access-control-allow-origin": "*" },
    });
  });

  await page.goto(`/view.html#${SITE}`);
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn", { timeout: 30_000 });
  const toggle = page.getByRole("button", { name: "Coverage" });
  await expect(toggle).toBeVisible();
  // Not fetched until asked for: it is megabytes a phone need not spend unasked.
  expect(fetched).toBe(0);

  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-pressed", "true");
  const legend = page.locator("#coverage-legend");
  await expect(legend).toBeVisible();
  await expect(legend).toContainText("Camera path");
  await expect(legend).toContainText("2 kept · 1 context · 1 dropped");
  await expect(legend).toContainText("48% of the scene verified by held-out frames");

  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-pressed", "false");
  await expect(legend).toBeHidden();
  await toggle.click();
  await expect(legend).toBeVisible();
  expect(fetched).toBe(1);
  expect(errors).toEqual([]);
});

test("the viewer does not load CesiumJS", async ({ page }) => {
  const scripts: string[] = [];
  page.on("request", (request) => {
    if (request.resourceType() === "script") scripts.push(request.url());
  });
  await page.goto(`/view.html#${SITE}`);
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn", { timeout: 30_000 });
  expect(scripts.filter((url) => /cesium/i.test(url))).toEqual([]);
});
