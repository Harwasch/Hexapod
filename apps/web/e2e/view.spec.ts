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

test("the viewer does not load CesiumJS", async ({ page }) => {
  const scripts: string[] = [];
  page.on("request", (request) => {
    if (request.resourceType() === "script") scripts.push(request.url());
  });
  await page.goto(`/view.html#${SITE}`);
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn", { timeout: 30_000 });
  expect(scripts.filter((url) => /cesium/i.test(url))).toEqual([]);
});
