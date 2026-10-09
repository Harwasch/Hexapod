import type { Page } from "@playwright/test";

import type { BoundaryRevision, LandArea, LandCreate, LandRevise } from "@twin/contracts";

import { expect, test } from "./fixtures";

test.skip(process.env.VITE_ENABLE_LAND_EXPLORATION !== "true", "Land preview is opt-in.");

async function landApi(page: Page) {
  const saved: LandArea[] = [];
  const history: BoundaryRevision[] = [];
  await page.route("**/api/v1/land{,/**,?*}", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path.endsWith("/investigations")) {
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
    let body: unknown = saved;
    if (request.method() === "POST" || request.method() === "PUT") {
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
      status: request.method() === "POST" ? 201 : 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });
  return { saved, history };
}

test("draw, save, inspect elsewhere, edit, and reopen a land area without a mission", async ({
  app,
}) => {
  const records = await landApi(app);
  await app.getByRole("button", { name: "Explore Earth", exact: true }).click();
  await app.getByTestId("tool-land").click();
  await app.evaluate(() => {
    const twin = (
      window as unknown as {
        __twin: {
          camera: {
            cancelFlight(): void;
            setView(lon: number, lat: number, height: number, heading: number, pitch: number): void;
          };
        };
      }
    ).__twin;
    twin.camera.cancelFlight();
    twin.camera.setView(-119.9, 36.6, 1500, 0, -90);
  });
  await app.getByRole("button", { name: /Draw a boundary/ }).click();
  for (const [x, y] of [
    [650, 300],
    [1000, 300],
    [1000, 550],
    [650, 550],
  ]) {
    await app.mouse.click(x!, y!);
  }
  await expect(app.getByText(/4 points placed/)).toBeVisible();
  await app.getByRole("button", { name: "Review boundary", exact: true }).click();
  await app.getByLabel("Name this land").fill("West field");
  await app.getByRole("button", { name: "Save this land", exact: true }).click();
  await expect(app.getByRole("heading", { name: "West field", exact: true })).toBeVisible();
  expect(records.saved[0]?.boundary.type).toBe("Polygon");
  expect(records.saved[0]?.source.method).toBe("drawn");
  await app.mouse.click(1100, 650);
  await expect(app.getByRole("heading", { name: "West field", exact: true })).toBeVisible();
  // Inspect the ground through the land overlay; the area remains active.
  await app.mouse.click(800, 425, { button: "right" });
  await app.getByRole("menuitem", { name: "What's here" }).click();
  await expect(app.getByRole("heading", { name: "Location", exact: true })).toBeVisible();
  await expect(app.getByRole("heading", { name: "West field", exact: true })).toBeVisible();
  await app.getByRole("button", { name: "Edit boundary", exact: true }).click();
  await app.getByLabel("Notes", { exact: true }).fill("Survey before restoration");
  await app.getByRole("button", { name: "Save revision", exact: true }).click();
  await expect.poll(() => records.history.length).toBe(2);
  await app.reload();
  await app.getByTestId("tool-land").click();
  await app.getByRole("button", { name: /West field/ }).click();
  await expect(app.getByText("Survey before restoration", { exact: true })).toBeVisible();
  await app.getByRole("button", { name: "History", exact: true }).click();
  await expect(app.getByText("Revision 2", { exact: true })).toBeVisible();
});

test("import with an exclusion and preserve the map on a phone", async ({ app }) => {
  const records = await landApi(app);
  await app.setViewportSize({ width: 390, height: 844 });
  await app.getByRole("button", { name: "Explore Earth", exact: true }).click();
  await app.getByTestId("phone-tab-more").click();
  await app.getByTestId("more-land").click();
  const boundary = {
    type: "Polygon",
    coordinates: [
      [
        [-122.14, 47.64],
        [-122.13, 47.64],
        [-122.13, 47.65],
        [-122.14, 47.65],
        [-122.14, 47.64],
      ],
      [
        [-122.138, 47.642],
        [-122.132, 47.642],
        [-122.132, 47.648],
        [-122.138, 47.642],
      ],
    ],
  };
  await app.getByLabel("Import GeoJSON boundary").setInputFiles({
    name: "Creek reserve.geojson",
    mimeType: "application/geo+json",
    buffer: Buffer.from(JSON.stringify(boundary)),
  });
  await app.getByRole("button", { name: "Save this land", exact: true }).click();
  await expect(app.getByRole("heading", { name: "Creek reserve", exact: true })).toBeVisible();
  expect(records.saved[0]?.boundary.coordinates).toEqual([boundary.coordinates]);
  expect(await app.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await app.getByRole("button", { name: "Close panel", exact: true }).click();
  const outlines = await app.evaluate(() => {
    const twin = (
      window as unknown as { __twin: { viewer: { entities: { values: { id: string }[] } } } }
    ).__twin;
    return twin.viewer.entities.values.filter((entity) =>
      entity.id.startsWith("land-boundary:ring:"),
    ).length;
  });
  expect(outlines).toBe(2);
});
