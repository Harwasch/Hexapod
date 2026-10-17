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

/** The inspector's point facts, answered here so "What's here" never reaches an agency. */
async function placeApi(page: Page) {
  const asked: string[] = [];
  await page.route("**/api/v1/research/place/*", async (route) => {
    const url = new URL(route.request().url());
    const kind = url.pathname.split("/").at(-1) as components["schemas"]["PlaceFact"]["kind"];
    asked.push(kind);
    const fact: components["schemas"]["PlaceFact"] = {
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
  return asked;
}

async function landApi(page: Page) {
  await placeApi(page);
  const saved: LandArea[] = [];
  const history: BoundaryRevision[] = [];
  await page.route("**/api/v1/land{,/**,?*}", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (
      path.endsWith("/investigations") ||
      path.endsWith("/scenarios") ||
      path.endsWith("/surveys") ||
      path.endsWith("/solar-assessments") ||
      path.endsWith("/features") ||
      path.endsWith("/features/imports") ||
      path.endsWith("/actions") ||
      path.endsWith("/documents") ||
      path.endsWith("/documents/links")
    ) {
      await route.fulfill({ status: 200, contentType: "application/json", body: "[]" });
      return;
    }
    if (path.endsWith("/ground")) {
      // The land as mission ground: its boundary only (no inventory in this fixture).
      const land = saved[0];
      await route.fulfill({
        status: land ? 200 : 404,
        contentType: "application/json",
        body: JSON.stringify(
          land
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
        ),
      });
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
  // The point is inside the saved land, so the inspector offers to research it.
  const place = app.getByTestId("inspector-place");
  await expect(place.getByTestId("place-elevation")).toContainText("236.8 m");
  await expect(place.getByRole("button", { name: "Research West field" })).toBeVisible();
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

test("ask what is here, then outline the land around that point", async ({ app }) => {
  await landApi(app);
  const asked = await placeApi(app);
  let searched: number[] | null = null;
  await app.route("**/api/v1/land/selection/candidates", async (route) => {
    searched = (route.request().postDataJSON() as { point: { coordinates: number[] } }).point
      .coordinates;
    await route.fulfill({
      json: {
        status: "available",
        message: "One mapped field found.",
        truncated: false,
        candidates: [
          {
            id: "osm:way:7",
            label: "Creek field",
            geometry: {
              type: "Polygon",
              coordinates: [
                [
                  [-119.905, 36.595],
                  [-119.895, 36.595],
                  [-119.895, 36.605],
                  [-119.905, 36.605],
                  [-119.905, 36.595],
                ],
              ],
            },
            source: {
              method: "mapped-feature",
              meaning: "physical-feature",
              label: "OpenStreetMap field",
            },
            distanceM: 0,
            properties: {},
          },
        ],
      },
    });
  });
  await app.getByRole("button", { name: "Explore Earth", exact: true }).click();
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
  await app.mouse.click(800, 425, { button: "right" });
  await app.getByRole("menuitem", { name: "What's here" }).click();
  const place = app.getByTestId("inspector-place");
  await expect(place).toContainText("At this point · looked up live, not saved");
  await expect(place.getByTestId("place-elevation")).toContainText("236.8 m");
  await expect(place.getByTestId("place-flood-zone")).toContainText("No mapped zone");
  expect(new Set(asked)).toEqual(
    new Set(["elevation", "soil", "land-cover", "flood-zone", "ecoregion"]),
  );
  await place.getByRole("button", { name: "Outline this land" }).click();
  await expect(app.getByRole("button", { name: /Creek field/ })).toBeVisible();
  expect(searched).not.toBeNull();
});

test("plan work on a saved land: the land is the ground, and the plan stays private", async ({
  app,
}) => {
  const records = await landApi(app);
  const missions: Record<string, unknown>[] = [];
  const publicPlans: Record<string, unknown>[] = [];
  const publicLandReads: string[] = [];
  app.on("request", (request) => {
    const url = new URL(request.url());
    const path = url.pathname;
    // The public list is never asked about a Land's project.
    if (path === "/api/v1/plans" && url.searchParams.get("projectId")?.startsWith("land:"))
      publicLandReads.push(request.url());
    if (request.method() !== "POST") return;
    if (path === "/api/v1/workspaces/missions")
      missions.push(request.postDataJSON() as Record<string, unknown>);
    if (path === "/api/v1/plans")
      publicPlans.push(request.postDataJSON() as Record<string, unknown>);
  });
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
  await app.getByRole("button", { name: "Review boundary", exact: true }).click();
  await app.getByLabel("Name this land").fill("West field");
  await app.getByRole("button", { name: "Save this land", exact: true }).click();
  await expect(app.getByRole("heading", { name: "West field", exact: true })).toBeVisible();
  await app.getByTestId("land-plan-work").click();
  const plans = app.getByTestId("plans-panel");
  await expect(plans.getByTestId("plan-ground")).toContainText("West field");
  await expect(plans.getByTestId("plan-ground")).toContainText("saved boundary revision 1");
  await expect(plans.getByTestId("plan-awaiting-ground")).toHaveCount(0);
  await expect(app.getByTestId("project-card")).toContainText("West field");
  await app.getByTestId("command-input").fill("monitor the fence every week");
  await app.getByTestId("command-input").press("Enter");
  await expect(plans.getByTestId("plan-review")).toBeVisible({ timeout: 30_000 });
  // No feature named "fence" on this land: the land's own boundary, never a rough square.
  await expect(plans.getByTestId("plan-ground")).toContainText("West field");
  await expect(plans.getByTestId("plan-ground")).not.toContainText("Rough area");
  await expect(plans.getByTestId("plan-awaiting-ground")).toHaveCount(0);
  await plans.getByTestId("plan-approve").dispatchEvent("click");
  await expect(plans.getByTestId("plan-detail")).toContainText("Private to workspace");
  expect(missions).toHaveLength(1);
  expect(missions[0]).toMatchObject({
    projectId: `land:${records.saved[0]?.id}`,
    landId: records.saved[0]?.id,
    boundaryRevision: 1,
  });
  expect(publicPlans).toEqual([]);
  expect(publicLandReads).toEqual([]);
  // The browser keeps none of it.
  const stored = await app.evaluate(() => localStorage.getItem("twin.mission.v1") ?? "");
  expect(stored).not.toContain(records.saved[0]?.id ?? "missing");
});

/** The Fog of Knowing over the saved land: full fog but one inspected cell, from fixtures. */
async function knowledgeApi(page: Page, land: () => LandArea | undefined) {
  const scouted: Record<string, unknown>[] = [];
  const frame = () => {
    const ring = (land()?.boundary as { coordinates: number[][][] }).coordinates[0] ?? [];
    const lons = ring.map((p) => p[0] ?? 0);
    const lats = ring.map((p) => p[1] ?? 0);
    return {
      west: Math.min(...lons),
      south: Math.min(...lats),
      east: Math.max(...lons),
      north: Math.max(...lats),
      columns: 4,
      rows: 4,
      cellM: 20,
    };
  };
  const record = {
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
  const policy = {
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
  const notCounted = [
    { title: "Google Photorealistic 3D Tiles (the basemap)", reason: "No capture date." },
  ];
  await page.route("**/api/v1/land/*/knowledge*", async (route) => {
    const density = new Uint8Array(16).fill(250);
    density[5] = 39;
    const best = new Uint8Array(32).fill(255);
    best[10] = 0;
    best[11] = 0;
    const base64 = (bytes: Uint8Array) => Buffer.from(bytes).toString("base64");
    await route.fulfill({
      json: {
        landId: land()?.id,
        boundaryRevision: 1,
        asOf: "2026-10-16T00:00:00Z",
        density: {
          ...frame(),
          encoding: "u8",
          scale: 0.004,
          offset: 0,
          nodata: 255,
          data: base64(density),
        },
        best: {
          ...frame(),
          encoding: "u16",
          scale: 1,
          offset: 0,
          nodata: 65535,
          data: base64(best),
        },
        records: [record],
        summary: {
          insideCells: 16,
          byBestClass: { field: 1 / 16, none: 15 / 16 },
          foggyFraction: 15 / 16,
          medianDensity: 1,
        },
        policy,
        notCounted,
      },
    });
  });
  await page.route("**/api/v1/land/*/knowledge/cell*", async (route) => {
    const url = new URL(route.request().url());
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
        records: [{ record, credit: 0.845, best: true, arithmetic: record.arithmetic }],
        notCounted,
        policyVersion: "fog-v1",
      },
    });
  });
  await page.route("**/api/v1/land/*/knowledge/scout-areas", async (route) => {
    scouted.push(route.request().postDataJSON() as Record<string, unknown>);
    const f = frame();
    const w = f.west + (f.east - f.west) * 0.1;
    const e = f.west + (f.east - f.west) * 0.6;
    const s = f.south + (f.north - f.south) * 0.1;
    const n = f.south + (f.north - f.south) * 0.6;
    await route.fulfill({
      json: {
        areas: [
          {
            footprint: {
              type: "Polygon",
              coordinates: [
                [
                  [w, s],
                  [e, s],
                  [e, n],
                  [w, n],
                  [w, s],
                ],
              ],
            },
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
  });
  return scouted;
}

test("show the fog of knowing, read a cell, and draft a scout flight with its DJI route", async ({
  app,
}) => {
  const records = await landApi(app);
  const scouted = await knowledgeApi(app, () => records.saved[0]);
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
  await app.getByRole("button", { name: "Review boundary", exact: true }).click();
  await app.getByLabel("Name this land").fill("West field");
  await app.getByRole("button", { name: "Save this land", exact: true }).click();
  await expect(app.getByRole("heading", { name: "West field", exact: true })).toBeVisible();
  await app.getByTestId("tool-land").click();
  const legend = app.getByTestId("knowledge-legend");
  await legend.getByLabel("Show the fog of knowing").click();
  await expect(app.getByTestId("knowledge-foggy")).toHaveText("94% foggy");
  await legend.getByTestId("knowledge-details").click();
  const details = app.getByTestId("knowledge-legend-details");
  await expect(details).toContainText("Fog density = 1 - the best credit");
  await expect(details).toContainText("computed from 1 cited record · fog-v1");
  await expect(details).toContainText("Not counted (1)");
  await legend.getByTestId("knowledge-details").click();
  await expect(details).toHaveCount(0);
  // The map menu asks about a point inside the land.
  await app.mouse.click(800, 425, { button: "right" });
  await expect(app.getByRole("menu")).toBeVisible();
  await app.getByRole("menuitem", { name: "What do we know here" }).click();
  const card = app.getByTestId("knowledge-cell");
  await expect(card).toContainText("Inspection: North trough, good");
  await expect(card).toContainText("0.85 x 0.5^(1/120) = 0.84");
  await expect(card).toContainText("Sets this cell");
  await card.getByLabel("Close panel").click();
  // Lift the fog: a scout plan over the foggiest ground, with its DJI route.
  await legend.getByTestId("knowledge-details").click();
  await app.getByTestId("knowledge-lift").click();
  const plans = app.getByTestId("plans-panel");
  await expect(plans.getByTestId("plan-review")).toBeVisible({ timeout: 30_000 });
  await expect(plans.getByTestId("plan-ground")).toContainText("Fog patch 1");
  const scout = plans.getByTestId("scout-export");
  await expect(scout).toContainText("DJI Mavic 3 Enterprise wide camera (assumed)");
  await expect(scout).toContainText("Draft, not flight-tested");
  expect(scouted[0]).toMatchObject({ threshold: 0.5, maxAreas: 3, cellM: 20 });
  const [download] = await Promise.all([
    app.waitForEvent("download"),
    scout.getByTestId("scout-download").click(),
  ]);
  expect(download.suggestedFilename()).toMatch(/^scout-west-field-\d{4}-\d{2}-\d{2}\.kmz$/);
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
    } else
      result = path.endsWith("/revisions")
        ? versions
        : path.endsWith("/actions")
          ? action
            ? [action]
            : []
          : action;
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

for (const kind of ["boundary", "corridor"] as const) {
  test(`recover an unfinished ${kind} after reload and continue tracing`, async ({ app }) => {
    await landApi(app);
    await app.getByRole("button", { name: "Explore Earth", exact: true }).click();
    await app.getByTestId("tool-land").click();
    await app.evaluate(() => {
      const twin = (
        window as unknown as {
          __twin: {
            camera: {
              cancelFlight(): void;
              setView(
                lon: number,
                lat: number,
                height: number,
                heading: number,
                pitch: number,
              ): void;
            };
          };
        }
      ).__twin;
      twin.camera.cancelFlight();
      twin.camera.setView(-77.05, 38.888, 1500, 0, -90);
    });
    await app
      .getByRole("button", { name: kind === "boundary" ? /Draw a boundary/ : /Trace a corridor/ })
      .click();
    if (kind === "corridor") {
      await app.getByLabel("Total corridor width", { exact: true }).fill("150");
      await app.getByRole("combobox", { name: "Width units", exact: true }).selectOption("m");
      await expect(app.getByLabel("Total corridor width", { exact: true })).toHaveValue("45.72");
    }
    await app.mouse.click(650, 300);
    await app.mouse.click(1000, 300);
    await expect(app.getByText(/2 points placed/)).toBeVisible();
    const points = await app.evaluate(
      () =>
        (
          JSON.parse(localStorage.getItem("living-world-land-draft:pilot")!) as {
            sketch: { points: number[][] };
          }
        ).sketch.points,
    );
    await app.reload();
    // The shared fixture intentionally resets onboarding on every document load.
    await app.getByRole("button", { name: "Explore Earth", exact: true }).click();
    await app.getByTestId("tool-land").click();
    await expect(app.getByText(/2 drawn points are stored/)).toBeVisible();
    await app.getByRole("button", { name: "Resume drawing", exact: true }).click();
    await expect(app.getByText(/2 points placed/)).toBeVisible();
    expect(
      await app.evaluate(
        () =>
          (
            JSON.parse(localStorage.getItem("living-world-land-draft:pilot")!) as {
              sketch: { points: number[][] };
            }
          ).sketch.points,
      ),
    ).toEqual(points);
    if (kind === "corridor") {
      await expect(app.getByLabel("Total corridor width", { exact: true })).toHaveValue("45.72");
      await expect(app.getByRole("combobox", { name: "Width units", exact: true })).toHaveValue(
        "m",
      );
    }
    await expect
      .poll(() =>
        app.evaluate(() =>
          Boolean(
            (window as unknown as { __twin: { viewer: { camera: { _currentFlight?: unknown } } } })
              .__twin.viewer.camera._currentFlight,
          ),
        ),
      )
      .toBe(false);
    await app.mouse.click(1000, 550);
    await expect(app.getByText(/3 points placed/)).toBeVisible();
    await app.getByRole("button", { name: "Last point", exact: true }).click();
    await expect(app.getByText(/2 points placed/)).toBeVisible();
    await app.getByRole("button", { name: "Cancel", exact: true }).click();
    expect(
      await app.evaluate(() => localStorage.getItem("living-world-land-draft:pilot")),
    ).toBeNull();
  });
}

test("Escape cancels a ground pick before closing the land workspace", async ({ app }) => {
  await landApi(app);
  await app.getByRole("button", { name: "Explore Earth", exact: true }).click();
  await app.getByTestId("tool-land").click();
  await app.evaluate(() => {
    const twin = (window as unknown as { __twin: { areas: { pickGround(): Promise<unknown> } } })
      .__twin;
    void twin.areas.pickGround();
  });
  await expect(app.locator(".cesium-widget canvas")).toHaveCSS("cursor", "crosshair");
  await app.keyboard.press("Escape");
  await expect(app.locator(".cesium-widget canvas")).not.toHaveCSS("cursor", "crosshair");
  await expect(app.getByTestId("land-panel")).toBeVisible();
  await app.keyboard.press("Escape");
  await expect(app.getByTestId("land-panel")).not.toBeVisible();
});
