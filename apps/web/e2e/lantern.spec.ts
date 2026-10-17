/**
 * The Time Lantern (cesium/TimeLantern.ts, features/lantern): dated imagery under the cursor,
 * the rim's words, the wheel through time, Esc.
 *
 * e2e has no Google tiles (the network to Cesium ion is cut), so the photorealistic world is
 * entered without its tileset: the clipping goes into its photorealistic mode, as LayerManager
 * would put it, and the lens opens over the globe it keeps. The catalog and the tiles are
 * fixtures: a NAIP year and a 1919 USGS sheet over the demo ranch.
 */
import type { Page } from "@playwright/test";

import { expect, mockApi, test } from "./fixtures";

/** A 1x1 PNG, for every tile asked for. */
const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "base64",
);
const LON = -120.04;
const LAT = 34.698;
const square = (d: number) => ({
  type: "Polygon",
  coordinates: [
    [
      [LON - d, LAT - d],
      [LON + d, LAT - d],
      [LON + d, LAT + d],
      [LON - d, LAT + d],
      [LON - d, LAT - d],
    ],
  ],
});

const CATALOG = {
  bounds: [-120.06, 34.68, -120.02, 34.72],
  retrievedAt: "2026-10-16T12:00:00Z",
  cached: false,
  sources: [
    {
      id: "naip-2016",
      provider: "naip-aerial",
      kind: "aerial-photo",
      title: "NAIP 2016 aerial photograph",
      dateLabel: "Flown 15 June 2016",
      dateMeaning: "Flight dates of the photographs in this area.",
      sortDate: "2016-06-15",
      resolutionM: 0.6,
      mapScale: null,
      bounds: [LON - 0.05, LAT - 0.05, LON + 0.05, LAT + 0.05],
      tiles: {
        scheme: "web-mercator",
        urlTemplate:
          "https://planetarycomputer.microsoft.com/api/data/v1/mosaic/abc/tiles/WebMercatorQuad/{z}/{x}/{y}@1x?collection=naip",
        maximumLevel: 18,
        format: "png",
      },
      license: "U.S. public domain (USDA Farm Service Agency)",
      licenseUrl: "https://planetarycomputer.microsoft.com/dataset/naip",
      attribution: "USDA FSA National Agriculture Imagery Program",
      parts: [
        {
          observedOn: "2016-06-15",
          observedYear: 2016,
          footprint: square(0.05),
          recordUrl: "https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip/items/a",
          accuracy: {
            kind: "stated-by-source",
            valueM: 6,
            basis:
              "From 2009 to present all NAIP imagery acquisitions used the +/- 6 meters to ground specification.",
            citationUrl: null,
          },
        },
      ],
      warnings: [],
    },
    {
      id: "usgs-1919",
      provider: "usgs-historical-topo-sheets",
      kind: "historical-map",
      title: "USGS 1:62,500 Los Olivos",
      dateLabel: "Map dated 1919",
      dateMeaning:
        "Catalog publication year; surveying, revision and scanning may have different dates.",
      sortDate: "1919-01-01",
      resolutionM: 5.29,
      mapScale: 62500,
      bounds: [-120.25, 34.5, -120.0, 34.75],
      tiles: {
        scheme: "geographic",
        urlTemplate:
          "/api/v1/imagery/usgs-topo/tiles/{z}/{x}/{y}.png?sheet=CA%2FCA_Los%20Olivos_1_1919_62500_geo&w=-120.25&s=34.5&e=-120&n=34.75",
        maximumLevel: 17,
        format: "png",
      },
      license: "U.S. public domain",
      licenseUrl: null,
      attribution: "U.S. Geological Survey, Historical Topographic Map Collection",
      parts: [
        {
          observedOn: null,
          observedYear: 1919,
          footprint: square(0.1),
          recordUrl: "https://www.sciencebase.gov/catalog/item/x",
          accuracy: {
            kind: "not-stated",
            valueM: null,
            basis: "Predates the National Map Accuracy Standards (1947); no accuracy is stated.",
            citationUrl: null,
          },
        },
      ],
      warnings: [],
    },
  ],
  providers: [
    { provider: "naip-aerial", name: "NAIP", status: "available", summary: "1 found, 2016." },
    {
      provider: "usgs-historical-topo-sheets",
      name: "USGS maps",
      status: "available",
      summary: "1 found, 1919.",
    },
  ],
};

async function open(page: Page, errors: string[]): Promise<void> {
  page.on("console", (message) => {
    // The fixtures cut the network to Cesium ion and map servers on purpose.
    if (message.type() === "error" && !message.text().startsWith("Failed to load resource"))
      errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  await mockApi(page);
  await page.route("**/api/v1/imagery/dated**", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(CATALOG) }),
  );
  await page.route("https://planetarycomputer.microsoft.com/**", (route) =>
    route.fulfill({ status: 200, contentType: "image/png", body: PNG }),
  );
  await page.route("**/api/v1/imagery/usgs-topo/**", (route) =>
    route.fulfill({ status: 200, contentType: "image/png", body: PNG }),
  );
  await page.addInitScript(() => {
    window.localStorage.setItem(
      "twin.settings.v1",
      JSON.stringify({ state: { onboardingDismissed: true, quality: "performance" }, version: 1 }),
    );
  });
  await page.goto("/");
  await page.waitForFunction(() => "__twin" in window && Boolean(window.__twin));
  // Over the ranch, low, in the photorealistic world (without its tiles: see above).
  await page.evaluate(`(() => {
    const twin = window.__twin;
    twin.camera.cancelFlight();
    twin.camera.setView(${LON}, ${LAT - 0.008}, 900, 0, -55);
    twin.clipping.setWorldMode(true);
    twin.events.emit("world", "Google Photorealistic 3D Tiles");
  })()`);
}

test("shows dated imagery under the cursor, and steps back to the 1919 map", async ({ page }) => {
  const errors: string[] = [];
  await open(page, errors);
  const rail = page.getByRole("navigation", { name: "Tools" }).getByRole("button");
  const railButtons = await rail.count();

  const toggle = page.getByTestId("lantern-toggle");
  await expect(toggle).toBeEnabled({ timeout: 30_000 });
  await toggle.click();
  await expect(page.getByTestId("lantern-rail")).toBeVisible();
  const box = await page.locator("canvas").first().boundingBox();
  if (!box) throw new Error("no canvas");
  await page.mouse.move(box.x + box.width / 2, box.y + box.height * 0.45);
  await page.mouse.move(box.x + box.width / 2 + 4, box.y + box.height * 0.45);

  const rim = page.getByTestId("lantern-rim");
  await expect(rim).toHaveAttribute("data-visible", "true", { timeout: 30_000 });
  await expect(page.getByTestId("lantern-rim-top")).toContainText("NAIP");
  await expect(page.getByTestId("lantern-rim-bottom")).toContainText("public domain");
  await expect(page.getByTestId("lantern-rim-bottom")).toContainText("stated ±6 m by its record");
  await expect(rim).toContainText("Not checked against today's 3D world");

  // Alt and the wheel, over the map: older.
  await page.keyboard.down("Alt");
  await page.mouse.wheel(0, 120);
  await page.keyboard.up("Alt");
  await expect(page.getByTestId("lantern-rim-top")).toContainText("Map dated 1919", {
    timeout: 15_000,
  });
  await expect(page.getByTestId("lantern-rim-bottom")).toContainText(
    "Predates the National Map Accuracy Standards",
  );
  await expect(page.getByTestId("lantern-live")).toContainText("USGS 1:62,500 Los Olivos");

  await page.keyboard.press("Escape");
  await expect(rim).toHaveAttribute("data-visible", "false");
  await expect(toggle).toHaveAttribute("aria-pressed", "false");

  // The strip took the lantern; the rail of tools is as it was.
  await expect(rail).toHaveCount(railButtons);
  expect(errors).toEqual([]);
});

test("is offered only in the 3D world, and says why not elsewhere", async ({ page }) => {
  const errors: string[] = [];
  await open(page, errors);
  await page.evaluate(`(() => {
    window.__twin.clipping.setWorldMode(false);
    window.__twin.events.emit("world", "Open world");
  })()`);
  await expect(page.getByTestId("lantern-toggle")).toBeDisabled();
  await expect(page.getByTestId("lantern-control")).toContainText(
    "Needs the 3D world (Settings › World)",
  );
  await page.keyboard.press("y");
  await expect(page.getByText("The Time Lantern is not available here")).toBeVisible();
});
