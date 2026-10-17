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

/**
 * A saved land's bare-earth elevation (gentle west half, a 40 % slope over the east half), no
 * mapped creeks, and the API's workable ground for it: the west half, the east half left out.
 */
async function workableApi(page: Page, land: () => LandArea | undefined) {
  const asked: Record<string, unknown>[] = [];
  const bounds = () => {
    const ring = (land()?.boundary as { coordinates: number[][][] }).coordinates[0] ?? [];
    const lons = ring.map((p) => p[0] ?? 0);
    const lats = ring.map((p) => p[1] ?? 0);
    return {
      west: Math.min(...lons),
      south: Math.min(...lats),
      east: Math.max(...lons),
      north: Math.max(...lats),
    };
  };
  const rectangle = (west: number, south: number, east: number, north: number) => [
    [
      [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
      ],
    ],
  ];
  await page.route("**/api/v1/land/*/rasters*", async (route) => {
    await route.fulfill({
      json: [
        {
          id: "raster-dem",
          runId: "run-0",
          landId: land()?.id,
          boundaryRevision: land()?.revision ?? 1,
          request: { dataset: "usgs-3dep-13", resolutionM: 10, maxDimension: 512, periods: [] },
          metadata: { resolutionM: 10, boundaryCells: 1200, warnings: [] },
          createdAt: "2026-10-16T00:00:00Z",
          stale: false,
        },
      ],
    });
  });
  await page.route("**/api/v1/land/rasters/*/grid*", async (route) => {
    const { west, south, east, north } = bounds();
    const columns = 40;
    const rows = 30;
    const cellM =
      ((east - west) * 111_320 * Math.cos((((south + north) / 2) * Math.PI) / 180)) / columns;
    const values = new Float32Array(columns * rows);
    for (let row = 0; row < rows; row++)
      for (let column = 0; column < columns; column++)
        values[row * columns + column] =
          column < columns / 2
            ? 100 + column * cellM * 0.05
            : 100 + (columns / 2) * cellM * 0.05 + (column - columns / 2) * cellM * 0.4;
    await route.fulfill({
      json: {
        west,
        south,
        east,
        north,
        columns,
        rows,
        cellM,
        encoding: "f32",
        scale: 1,
        offset: 0,
        nodata: null,
        data: Buffer.from(values.buffer).toString("base64"),
      },
    });
  });
  await page.route("**/api/v1/land/*/hydrography", async (route) => {
    await route.fulfill({
      json: {
        landId: land()?.id,
        boundaryRevision: 1,
        status: "empty",
        message: "The USGS NHD maps no streams or ditches within 100 m of this land.",
        bufferM: 100,
        features: [],
        truncated: false,
        retrievedAt: "2026-10-16T00:00:00Z",
        source: {
          provider: "usgs-nhd",
          name: "USGS National Hydrography Dataset (NHD) flowlines",
          url: "https://example.test/nhd",
          linkUrl: "https://example.test/nhd",
          documentationUrl: "https://example.test/docs",
          license: "Public domain (U.S. Government work)",
          attribution: "U.S. Geological Survey",
          resolution: "1:24,000 (Flowline - Large Scale); mapped channels",
        },
      },
    });
  });
  await page.route("**/api/v1/land/*/workable-ground", async (route) => {
    asked.push(route.request().postDataJSON() as Record<string, unknown>);
    const { west, south, east, north } = bounds();
    const middle = (west + east) / 2;
    const acre = 4046.8564224;
    await route.fulfill({
      json: {
        landId: land()?.id,
        boundaryRevision: 1,
        workable: { type: "MultiPolygon", coordinates: rectangle(west, south, middle, north) },
        footprintM2: 40 * acre,
        workableM2: 20 * acre,
        removed: {
          steepM2: 20 * acre,
          steepEdgeM2: 0,
          creekBufferM2: 0,
          sliversM2: 0,
          slivers: 0,
          noDataM2: 0,
        },
        leftOut: [
          {
            reason: "steep",
            label: "Left out: too steep",
            areaM2: 20 * acre,
            geometry: { type: "MultiPolygon", coordinates: rectangle(middle, south, east, north) },
          },
        ],
        limits: {
          machineClass: "mower",
          label: "Robotic brush mower (assumed)",
          maxSlopePct: 25,
          creekBufferM: 15,
        },
        elevation: {
          rasterId: "raster-dem",
          dataset: "usgs-3dep-13",
          label: "USGS 3DEP 1/3″ bare-earth DEM, 10 m",
          published: "2025-08-27",
          resolutionM: 10,
          edgeM: 5,
          stale: false,
          fallback: false,
        },
        creeks: {
          status: "empty",
          label: "USGS National Hydrography Dataset (NHD) flowlines, 1:24,000",
          counted: 0,
          ephemeralIncluded: false,
          message: "The USGS NHD maps no streams or ditches within 100 m of this land.",
        },
        sources: ["Slope: USGS 3DEP 1/3″ bare-earth DEM, 10 m, raster raster-d"],
        assumptions: [],
        warnings: [],
      },
    });
  });
  return asked;
}

test("rehearse a plan on a saved land, and work only the ground it can", async ({ app }) => {
  const records = await landApi(app);
  const asked = await workableApi(app, () => records.saved[0]);
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
  await app.getByLabel("Name this land").fill("Bench pasture");
  await app.getByRole("button", { name: "Save this land", exact: true }).click();
  await expect(app.getByRole("heading", { name: "Bench pasture", exact: true })).toBeVisible();
  await app.getByTestId("land-plan-work").click();
  const plans = app.getByTestId("plans-panel");
  await expect(plans.getByTestId("plan-ground")).toContainText("Bench pasture");
  await expect(app.getByTestId("project-card")).toContainText("Bench pasture");
  await app.getByTestId("command-input").fill("Mow this land");
  await app.getByTestId("command-input").press("Enter");
  await expect(plans.getByTestId("plan-review")).toBeVisible({ timeout: 30_000 });
  await plans.getByTestId("plan-rehearse").dispatchEvent("click");
  await expect(plans.getByTestId("rehearsal-clock")).toBeVisible({ timeout: 45_000 });
  await plans.getByTestId("rehearsal-skip").dispatchEvent("click");
  // The east half is over the mower's assumed 25 %: hands, grouped by rule.
  await expect(plans.getByTestId("rehearsal-hand-group").first()).toContainText(/Too steep/);
  const before = Number(
    /(\d+) hands?/.exec((await plans.getByTestId("plan-approve").textContent()) ?? "")?.[1],
  );
  expect(before).toBeGreaterThan(0);
  const offer = plans.getByTestId("rehearsal-workable-apply").last();
  await expect(offer).toContainText("Work only the ground it can");
  await expect(offer).toContainText("−20 ac too steep → 20 ac");
  expect(asked[0]).toMatchObject({
    machineClass: "mower",
    demRasterId: "raster-dem",
    includeEphemeralCreeks: false,
  });
  await offer.dispatchEvent("click");
  // The plan is redrafted on the workable half, which is private and derived from the land, the
  // steep half is hatched, and the new draft is rehearsed again on its own.
  await expect(plans.getByTestId("plan-ground")).toContainText("Bench pasture, workable", {
    timeout: 30_000,
  });
  await expect(plans.getByTestId("plan-ground")).toContainText(
    "Workable ground for a robotic brush mower: slope ≤ 25 % (assumed) on USGS 3DEP 1/3″ DEM, published 2025-08-27",
  );
  await expect(plans.getByTestId("rehearsal-after")).toContainText(
    "Rehearsed again on the workable ground of Bench pasture",
    { timeout: 45_000 },
  );
  await expect
    .poll(async () =>
      app.evaluate(
        () =>
          (
            (
              window as unknown as {
                __twin: { viewer: { entities: { values: { id: string }[] } } };
              }
            ).__twin.viewer.entities.values ?? []
          ).filter((e) => e.id.startsWith("rehearsal:left-out:")).length,
      ),
    )
    .toBeGreaterThan(0);
  const after = Number(
    /(\d+) hands?/.exec((await plans.getByTestId("plan-approve").textContent()) ?? "")?.[1] ?? 0,
  );
  expect(after).toBeLessThan(before);
  // Closing the card takes the hatch with it.
  await plans.getByText("All plans").dispatchEvent("click");
  await expect
    .poll(async () =>
      app.evaluate(
        () =>
          (
            (
              window as unknown as {
                __twin: { viewer: { entities: { values: { id: string }[] } } };
              }
            ).__twin.viewer.entities.values ?? []
          ).filter((e) => e.id.startsWith("rehearsal:")).length,
      ),
    )
    .toBe(0);
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
  // The strip reads in time order: dates, then the Time Lantern, then the fog.
  await expect(app.getByTestId("lantern-control")).toBeVisible();
  await expect(legend).toBeVisible();
  const stripOrder = await app
    .locator(".hud-strip")
    .evaluate((strip) =>
      [
        ...strip.querySelectorAll(
          '[data-testid="lantern-control"], [data-testid="knowledge-legend"]',
        ),
      ].map((node) => node.getAttribute("data-testid")),
    );
  expect(stripOrder).toEqual(["lantern-control", "knowledge-legend"]);
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
  // A new height redrafts the plan at the detail it gives, and the plan says so.
  await expect(plans.getByLabel("Plan title")).toHaveValue(/1\.7 cm per pixel/);
  await scout.getByRole("button", { name: "120 m", exact: true }).click();
  await expect(scout.getByRole("button", { name: "120 m", exact: true })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(scout).toContainText("3.4 cm per pixel over flat ground");
  await expect(plans.getByLabel("Plan title")).toHaveValue(/3\.4 cm per pixel/, {
    timeout: 30_000,
  });
  const [download] = await Promise.all([
    app.waitForEvent("download"),
    scout.getByTestId("scout-download").click(),
  ]);
  expect(download.suggestedFilename()).toMatch(/^scout-west-field-\d{4}-\d{2}-\d{2}\.kmz$/);
});

/**
 * Show once, everywhere: the land's signatures, one tap's similarity on a 4 x 4 grid over the
 * land (a 4-cell patch north-west, a 2-cell patch south-east, the rest unlike) and its areas.
 */
async function signatureApi(page: Page, land: () => LandArea | undefined) {
  const asked: { similar: number; zones: Record<string, unknown>[] } = { similar: 0, zones: [] };
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
    };
  };
  const scenes = ["2025-01-15", "2025-04-05", "2025-07-24", "2025-10-17"].map((day, window) => ({
    window,
    period: { startDate: `${day.slice(0, 8)}01`, endDate: `${day.slice(0, 8)}30` },
    sceneId: `S2_FIXTURE_${window}`,
    acquiredAt: `${day}T18:55:00Z`,
  }));
  await page.route("**/api/v1/land/*/rasters*", async (route) => {
    await route.fulfill({
      json: [
        {
          id: "raster-1",
          runId: "run-0",
          landId: land()?.id,
          boundaryRevision: land()?.revision ?? 1,
          request: { dataset: "sentinel-2-signature", resolutionM: 10, maxDimension: 512 },
          metadata: {
            resolutionM: 10,
            boundaryCells: 16,
            warnings: [],
            vegetation: { observations: scenes.map(({ acquiredAt }) => ({ acquiredAt })) },
            signature: { commonCells: 16 },
          },
          stale: false,
        },
      ],
    });
  });
  const seedOf = (longitude: number, latitude: number) => ({
    longitude,
    latitude,
    cellLongitude: longitude,
    cellLatitude: latitude,
    windows: [0.69, 0.64, 0.58, 0.65].map((ndvi, window) => ({
      window,
      acquiredAt: scenes[window]?.acquiredAt,
      green: 0.05,
      red: 0.04,
      nir: 0.3,
      swir16: 0.2,
      ndvi,
    })),
  });
  await page.route("**/api/v1/land/rasters/*/similar*", async (route) => {
    asked.similar += 1;
    const url = new URL(route.request().url());
    const scores = [250, 250, 0, 0, 250, 250, 0, 0, 0, 0, 0, 0, 0, 0, 240, 240];
    await route.fulfill({
      json: {
        rasterId: "raster-1",
        landId: land()?.id,
        boundaryRevision: 1,
        stale: false,
        seed: seedOf(
          Number(url.searchParams.get("longitude")),
          Number(url.searchParams.get("latitude")),
        ),
        grid: {
          ...frame(),
          cellM: 10,
          encoding: "u8",
          scale: 0.004,
          offset: 0,
          nodata: 255,
          data: Buffer.from(Uint8Array.from(scores)).toString("base64"),
        },
        cellAreaM2: 400,
        nativeResolutionM: 10,
        comparedCells: 16,
        boundaryCells: 16,
        scenes,
        method: "cosine",
        interpretation: "Similar seasonal satellite signature at 10 m. Not a species ID.",
        warnings: [],
        attribution: "Contains modified Copernicus Sentinel data",
      },
    });
  });
  await page.route("**/api/v1/land/rasters/*/similar/zones", async (route) => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    asked.zones.push(body);
    const f = frame();
    const box = (w: number, s: number, e: number, n: number) => ({
      type: "Polygon",
      coordinates: [
        [
          [f.west + (f.east - f.west) * w, f.south + (f.north - f.south) * s],
          [f.west + (f.east - f.west) * e, f.south + (f.north - f.south) * s],
          [f.west + (f.east - f.west) * e, f.south + (f.north - f.south) * n],
          [f.west + (f.east - f.west) * w, f.south + (f.north - f.south) * n],
          [f.west + (f.east - f.west) * w, f.south + (f.north - f.south) * s],
        ],
      ],
    });
    await route.fulfill({
      json: {
        rasterId: "raster-1",
        landId: land()?.id,
        boundaryRevision: 1,
        stale: false,
        threshold: body.threshold,
        seed: seedOf(Number(body.longitude), Number(body.latitude)),
        zones: [
          {
            footprint: box(0, 0.5, 0.5, 1),
            areaM2: 16_000,
            cells: 4,
            meanSimilarity: 0.99,
            patches: 1,
          },
        ],
        remainder: null,
        dropped: { patches: 0, areaM2: 0 },
        unoutlined: { patches: 0, areaM2: 0 },
        protectedCells: 2,
        matchedCells: 4,
        cellAreaM2: 100,
        nativeResolutionM: 10,
        scenes,
        method: "cosine",
        interpretation: "Similar seasonal satellite signature at 10 m. Not a species ID.",
        attribution: "Contains modified Copernicus Sentinel data",
      },
    });
  });
  return { asked, frame };
}

/** Clicks the map at a point on the ground (the canvas position Cesium gives for it). */
async function tapGround(page: Page, longitude: number, latitude: number) {
  const at = await page.evaluate(
    ([lon, lat]) => {
      const viewer = (
        window as unknown as {
          __twin: {
            viewer: {
              camera: { position: { constructor: unknown } };
              scene: { cartesianToCanvasCoordinates(p: unknown): { x: number; y: number } };
            };
          };
        }
      ).__twin.viewer;
      const C3 = viewer.camera.position.constructor as {
        fromDegrees(lon: number, lat: number, height: number): unknown;
      };
      const p = viewer.scene.cartesianToCanvasCoordinates(C3.fromDegrees(lon!, lat!, 0));
      return { x: p.x, y: p.y };
    },
    [longitude, latitude],
  );
  const box = await page.locator(".cesium-widget canvas").boundingBox();
  await page.mouse.click((box?.x ?? 0) + at.x, (box?.y ?? 0) + at.y);
}

test("show once: tap a place, slide how alike, protect a patch and make it the job", async ({
  app,
}) => {
  const records = await landApi(app);
  const { asked, frame } = await signatureApi(app, () => records.saved[0]);
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
  await app.getByRole("tab", { name: "Ecology" }).click();
  const card = app.getByTestId("show-once");
  await expect(card).toContainText("Not a species ID");
  await expect(app.getByTestId("show-once-scenes")).toContainText(
    "Scenes Jan 15, Apr 5, Jul 24, Oct 17 2025",
  );
  await app.getByTestId("show-once-start").click();
  await expect(app.locator(".cesium-widget canvas")).toHaveCSS("cursor", "crosshair");
  // A cell of the north-west patch: the dimensions of the land are the grid's.
  const f = frame();
  const cell = (row: number, col: number) =>
    [
      f.west + ((col + 0.5) * (f.east - f.west)) / 4,
      f.north - ((row + 0.5) * (f.north - f.south)) / 4,
    ] as const;
  await tapGround(app, ...cell(0, 0));
  const tally = app.getByTestId("show-once-tally");
  await expect(tally).toContainText("6 cells · ≈0.6 ac");
  await expect(app.getByTestId("show-once-seed")).toContainText("Jan 15 0.69");
  // Stricter: the 240s drop out; looser, back.
  const slider = app.getByRole("slider", { name: "How alike" });
  await slider.focus();
  await app.keyboard.press("End");
  await expect(app.getByTestId("show-once-threshold")).toHaveText("99%");
  await expect(tally).toContainText("4 cells");
  await app.keyboard.press("Home");
  await expect(app.getByTestId("show-once-threshold")).toHaveText("30%");
  await expect(tally).toContainText("6 cells");
  for (let i = 0; i < 5; i++) await app.keyboard.press("PageUp");
  await expect(app.getByTestId("show-once-threshold")).toHaveText("80%");
  // Protect the south-east patch.
  await app.getByTestId("show-once-mode-protect").click();
  await tapGround(app, ...cell(3, 3));
  await expect(app.getByTestId("show-once-protected")).toContainText("2 cells");
  await expect(tally).toContainText("4 cells");
  expect(asked.similar).toBe(1);
  await app.getByTestId("show-once-make-job").click();
  const plans = app.getByTestId("plans-panel");
  await expect(plans.getByTestId("plan-ground")).toContainText("Like-here patch 1");
  await expect(app.getByTestId("show-once-done")).toContainText("Drafted as a job: 1 area");
  expect(asked.zones[0]).toMatchObject({ threshold: 0.8 });
  // The protected patch travels as one row run: row 3, columns 2 and 3.
  const [run] = asked.zones[0]?.protected as number[][];
  expect(asked.zones[0]?.protected).toHaveLength(1);
  expect(run?.[0]).toBeCloseTo(f.west + ((f.east - f.west) * 2) / 4, 9);
  expect(run?.[3]).toBeCloseTo(f.south + (f.north - f.south) / 4, 9);
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

test.describe("on a touch phone", () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });

  /** Where the drawn boundary is on screen, and where the Land sheet starts. */
  async function boundaryOnScreen(page: Page) {
    return page.evaluate(() => {
      interface Twin {
        viewer: {
          entities: {
            values: {
              id: string;
              polyline?: { positions: { getValue(time: unknown): unknown[] } };
            }[];
          };
          clock: { currentTime: unknown };
        };
        scene: { cartesianToCanvasCoordinates(p: unknown): { x: number; y: number } | undefined };
      }
      const twin = (window as unknown as { __twin: Twin }).__twin;
      const ring = twin.viewer.entities.values.find((e) => e.id === "land-boundary:ring:0:0");
      const points = (ring?.polyline?.positions.getValue(twin.viewer.clock.currentTime) ?? []).map(
        (p) => twin.scene.cartesianToCanvasCoordinates(p) ?? null,
      );
      const sheet = document.querySelector('[data-testid="land-panel"]')?.getBoundingClientRect();
      return { points, sheetTop: sheet?.top ?? 0 };
    });
  }

  test("says tap, reviews the boundary above the sheet, and reopens at the overview", async ({
    app,
  }) => {
    await landApi(app);
    await app.getByRole("button", { name: "Explore Earth", exact: true }).click();
    await app.evaluate(() => {
      const twin = (
        window as unknown as {
          __twin: {
            camera: {
              cancelFlight(): void;
              setView(lon: number, lat: number, h: number, hd: number, p: number): void;
            };
          };
        }
      ).__twin;
      twin.camera.cancelFlight();
      twin.camera.setView(-119.9, 36.6, 1500, 0, -90);
    });
    await app.getByTestId("phone-tab-more").click();
    await app.getByTestId("more-land").click();
    // A finger taps: the copy says so.
    await expect(app.getByText("Tap around your land on the map", { exact: true })).toBeVisible();
    await app.getByRole("button", { name: /Draw a boundary/ }).click();
    await expect(
      app.getByText("0 points placed. Tap the map to add the first point."),
    ).toBeVisible();
    // Above the drawing sheet, reaching down to where the taller review sheet will be.
    for (const [x, y] of [
      [60, 150],
      [330, 150],
      [330, 300],
      [60, 300],
    ]) {
      await app.mouse.click(x!, y!);
    }
    await expect(app.getByText(/4 points placed/)).toBeVisible();
    await app.getByRole("button", { name: "Review boundary", exact: true }).click();
    await expect(app.getByText("Review your boundary", { exact: true })).toBeVisible();
    // The camera frames the boundary above the sheet that covers the lower screen.
    await expect
      .poll(
        async () => {
          const { points, sheetTop } = await boundaryOnScreen(app);
          return (
            points.length >= 4 &&
            sheetTop > 0 &&
            points.every((p) => p !== null && p.y < sheetTop && p.y > 0 && p.x > 0 && p.x < 390)
          );
        },
        { timeout: 20_000 },
      )
      .toBe(true);
    await app.getByLabel("Name this land").fill("Phone field");
    await app.getByRole("button", { name: "Save this land", exact: true }).click();
    const title = app.getByRole("heading", { name: "Phone field", exact: true });
    await expect(title).toBeVisible();
    // Down to the list at the panel's foot, and the land opened again from it: the panel
    // shows its overview, not the list it was opened from.
    const row = app.locator(".land-library button", { hasText: "Phone field" });
    await row.scrollIntoViewIfNeeded();
    await expect(title).not.toBeInViewport();
    await row.click();
    await expect(app.getByText("Land overview", { exact: true })).toBeInViewport();
  });
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
  // Save on an empty draft lists what to fix, sends nothing and puts the cursor on the first.
  await actions.getByRole("button", { name: "Save action draft", exact: true }).click();
  await expect(actions.getByRole("alert")).toContainText("Before saving, fix 4 things:");
  await expect(actions.getByLabel("Action title", { exact: true })).toBeFocused();
  expect(requests).toEqual([]);
  await actions.getByLabel("Action title", { exact: true }).fill("Baseline survey");
  await actions.getByLabel("Outcome to achieve", { exact: true }).fill("Measure native cover");
  await actions.getByLabel("Step 1 name", { exact: true }).fill("Survey quadrats");
  await actions
    .getByLabel("Step 1 success measure", { exact: true })
    .fill("Five recorded quadrats");
  await actions.getByRole("button", { name: "Save action draft", exact: true }).click();
  await expect(actions.getByRole("article", { name: "Action review" })).toBeVisible();
  expect(requests).toEqual(["draft"]);
  await expect(
    actions.getByRole("button", { name: "Approve revision 1", exact: true }),
  ).toBeDisabled();
  await expect(actions.getByRole("list", { name: "Before approving" })).toHaveText(
    "Add a review note. It is stored with the approval.",
  );
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
