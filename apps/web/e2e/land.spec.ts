import type { Page } from "@playwright/test";

import type {
  components,
  BoundaryRevision,
  Footprint,
  LandArea,
  LandCreate,
  LandRevise,
} from "@twin/contracts";

import { expect, test } from "./fixtures";

test.skip(process.env.VITE_ENABLE_LAND_EXPLORATION !== "true", "Land preview is opt-in.");

async function landApi(page: Page) {
  const saved: LandArea[] = [];
  const history: BoundaryRevision[] = [];
  await page.route("**/api/v1/land{,/**,?*}", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (
      path.endsWith("/investigations") ||
      path.endsWith("/scenarios") ||
      path.endsWith("/features") ||
      path.endsWith("/actions")
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
  await app.getByLabel("Import land boundary").setInputFiles({
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

test("select a mapped transmission line and preview a described corridor", async ({ app }) => {
  await landApi(app);
  const coordinates = [
    [-122.14, 47.64],
    [-122.14, 47.645],
    [-122.139, 47.65],
  ];
  await app.route("**/api/v1/land/selection/candidates", async (route) => {
    await route.fulfill({
      json: {
        status: "available",
        message: "One mapped line found.",
        truncated: false,
        candidates: [
          {
            id: "osm:way:9",
            label: "Transmission line 9",
            geometry: { type: "LineString", coordinates },
            source: {
              method: "mapped-feature",
              meaning: "physical-feature",
              label: "OpenStreetMap line",
            },
            distanceM: 0,
            properties: {},
          },
        ],
      },
    });
  });
  await app.route("**/api/v1/land/selection/interpret", async (route) => {
    expect((route.request().postDataJSON() as { selectedIds: string[] }).selectedIds).toEqual([
      "osm:way:9",
    ]);
    await route.fulfill({
      json: {
        operation: "corridor",
        candidateIds: ["osm:way:9"],
        widthM: 30.48,
        cap: "flat",
        explanation: "100 feet total width along the selected line",
        provider: "local",
      },
    });
  });
  await app.route("**/api/v1/land/corridor", async (route) => {
    const body = route.request().postDataJSON() as {
      widthM: number;
      coordinates: number[][];
      cap: string;
    };
    expect(body.widthM).toBe(30.48);
    expect(body.coordinates).toEqual(coordinates);
    expect(body.cap).toBe("flat");
    await route.fulfill({
      json: {
        boundary: {
          type: "Polygon",
          coordinates: [
            [
              [-122.141, 47.64],
              [-122.138, 47.64],
              [-122.138, 47.65],
              [-122.141, 47.65],
              [-122.141, 47.64],
            ],
          ],
        },
        areaM2: 30000,
        perimeterM: 2100,
      },
    });
  });
  await app.getByRole("button", { name: "Explore Earth", exact: true }).click();
  await app.getByTestId("tool-land").click();
  await app.getByRole("button", { name: /Find parcels, lines or buildings/ }).click();
  await app.getByRole("combobox", { name: "Find", exact: true }).selectOption("line");
  await app.mouse.click(900, 400);
  const candidate = app.getByRole("button", { name: /Transmission line 9/ });
  await candidate.click();
  await expect(candidate).toHaveAttribute("aria-pressed", "true");
  await app
    .getByLabel("Or describe your selection")
    .fill("100 feet wide along this transmission line, flat ends");
  await app.getByRole("button", { name: "Preview my instruction" }).click();
  await expect(app.getByLabel("Name this land")).toHaveValue("Transmission line 9");
  await expect(app.getByText("Corridor along Transmission line 9", { exact: true })).toBeVisible();
});

test("review and schedule an action, then preserve that mission when revising", async ({ app }) => {
  const records = await landApi(app);
  const boundary: Footprint = {
    type: "Polygon",
    coordinates: [
      [
        [-122.136, 47.644],
        [-122.134, 47.644],
        [-122.134, 47.646],
        [-122.136, 47.646],
        [-122.136, 47.644],
      ],
    ],
  };
  records.saved.push({
    id: "11111111-1111-4111-8111-111111111111",
    name: "Action field",
    description: "",
    boundary,
    source: { method: "drawn", label: "Fixture" },
    revision: 1,
    areaM2: 10000,
    perimeterM: 400,
    createdAt: "2026-10-09T00:00:00Z",
    updatedAt: "2026-10-09T00:00:00Z",
  });
  type Action = components["schemas"]["LandActionRead"];
  type Draft = components["schemas"]["LandActionCreate"];
  let action: Action | null = null;
  const versions: Action[] = [];
  const requests: string[] = [];
  await app.route("**/api/v1/land/*/actions{,/**,?*}", async (route) => {
    const request = route.request(),
      path = new URL(request.url()).pathname;
    let result: unknown;
    if (request.method() === "POST" && path.endsWith("/approve") && action) {
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
      if (request.method() === "POST") {
        requests.push("schedule");
        action = {
          ...action,
          status: "scheduled",
          missionId: "33333333-3333-4333-8333-333333333333",
        };
        versions[0] = action;
      }
      result = {
        id: action.missionId,
        status: "scheduled",
        startDate: action.startDate,
        endDate: action.startDate,
      };
    } else if (request.method() === "POST" || request.method() === "PUT") {
      requests.push(request.method() === "POST" ? "draft" : "revise");
      const payload = request.postDataJSON() as Draft;
      action = {
        ...payload,
        id: "22222222-2222-4222-8222-222222222222",
        landId: records.saved[0]?.id ?? "",
        revision: versions.length + 1,
        status: "draft",
        staleReasons: [],
        totalKnownCost: 0,
        uncostedSteps: 1,
        effectiveBoundary: boundary,
        approvedBy: null,
        approvedAt: null,
        approvalNote: null,
        missionId: null,
        createdAt: "2026-10-09T00:00:00Z",
        updatedAt: "2026-10-09T00:00:00Z",
      };
      versions.unshift(action);
      result = action;
    } else result = path.endsWith("/revisions") ? versions : action ? [action] : [];
    await route.fulfill({ json: result });
  });
  await app.getByRole("button", { name: "Explore Earth", exact: true }).click();
  await app.getByTestId("tool-land").click();
  await app.getByRole("button", { name: /Action field/ }).click();
  await app.getByRole("tab", { name: "Actions", exact: true }).click();
  const actions = app.getByRole("region", { name: "Land actions" });
  await actions.getByRole("button", { name: "Plan an action", exact: true }).click();
  await actions.getByLabel("Action title", { exact: true }).fill("Baseline survey");
  await actions.getByLabel("Outcome to achieve", { exact: true }).fill("Measure native cover");
  await actions.getByLabel("Step 1 name", { exact: true }).fill("Survey quadrats");
  await actions
    .getByLabel("Step 1 success measure", { exact: true })
    .fill("Five recorded quadrats");
  await actions.getByRole("button", { name: "Save action draft", exact: true }).click();
  await expect(actions.getByRole("article", { name: "Action review" })).toBeVisible();
  expect(requests).toEqual(["draft"]);
  await actions.getByLabel("Review note", { exact: true }).fill("Reviewed work area and access");
  await actions.getByRole("button", { name: "Approve revision 1", exact: true }).click();
  await actions.getByLabel("Scheduling note", { exact: true }).fill("Schedule the reviewed survey");
  expect(requests).toEqual(["draft", "approve"]);
  await actions.getByRole("button", { name: "Schedule approved action", exact: true }).click();
  await expect(actions.getByText(/Mission scheduled for/)).toBeVisible();
  await actions.getByRole("button", { name: "Revise action", exact: true }).click();
  await actions.getByLabel("Action title", { exact: true }).fill("Survey and follow-up");
  await actions.getByRole("button", { name: "Save action revision", exact: true }).click();
  await expect(
    actions.getByRole("button", { name: "Approve revision 2", exact: true }),
  ).toBeDisabled();
  expect(versions[1]?.status).toBe("scheduled");
  expect(versions[0]?.missionId).toBeNull();
  expect(requests).toEqual(["draft", "approve", "schedule", "revise"]);
});
