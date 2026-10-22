import type { Page } from "@playwright/test";

import type { Footprint } from "@twin/contracts";

import { expect, RANCH_NAME, RANCH_SLUG, scanSite, test } from "./fixtures";
import { landApi, rectangle } from "./landApi";

test.skip(process.env.VITE_ENABLE_LAND_EXPLORATION !== "true", "Land preview is opt-in.");

interface Twin {
  camera: {
    cancelFlight(): void;
    setView(lon: number, lat: number, height: number, heading: number, pitch: number): void;
  };
  viewer: {
    camera: { position: { x: number; y: number; z: number }; _currentFlight?: unknown };
    entities: { values: { id: string }[]; getById(id: string): unknown };
    scene: { cartesianToCanvasCoordinates(p: unknown): { x: number; y: number } | undefined };
  };
}

/** Looks straight down on a point from 1.5 km, with no flight in progress. */
async function lookDown(page: Page, longitude: number, latitude: number): Promise<void> {
  await page.evaluate(
    ([lon, lat]) => {
      const twin = (window as unknown as { __twin: Twin }).__twin;
      twin.camera.cancelFlight();
      twin.camera.setView(lon!, lat!, 1500, 0, -90);
    },
    [longitude, latitude],
  );
}

/**
 * Waits for the camera to come to rest: no flight, and the same pose on two reads half a second
 * apart. Flights (a land framed beside its cards) run on real time.
 */
async function cameraAtRest(page: Page): Promise<void> {
  let last = "";
  await expect
    .poll(
      async () => {
        const pose = await page.evaluate(() => {
          const camera = (window as unknown as { __twin: Twin }).__twin.viewer.camera;
          const { x, y, z } = camera.position;
          return `${[x, y, z].map(Math.round).join(",")}:${camera._currentFlight ? 1 : 0}`;
        });
        const still = pose === last && pose.endsWith(":0");
        last = pose;
        return still;
      },
      { intervals: [500], timeout: 30_000 },
    )
    .toBe(true);
}

/** Clicks the map at a point on the ground (the canvas position Cesium gives for it). */
async function tapGround(
  page: Page,
  longitude: number,
  latitude: number,
  options: { button?: "left" | "right" } = {},
): Promise<void> {
  const at = await page.evaluate(
    ([lon, lat]) => {
      const viewer = (window as unknown as { __twin: Twin }).__twin.viewer;
      const C3 = viewer.camera.position.constructor as unknown as {
        fromDegrees(lon: number, lat: number, height: number): unknown;
      };
      const p = viewer.scene.cartesianToCanvasCoordinates(C3.fromDegrees(lon!, lat!, 0));
      return { x: p?.x ?? 0, y: p?.y ?? 0 };
    },
    [longitude, latitude],
  );
  const box = await page.locator(".cesium-widget canvas").boundingBox();
  await page.mouse.click((box?.x ?? 0) + at.x, (box?.y ?? 0) + at.y, options);
}

/** How many of the map's entities have an id starting so. */
function entitiesNamed(page: Page, prefix: string): Promise<number> {
  return page.evaluate(
    (start) =>
      ((window as unknown as { __twin: Twin }).__twin.viewer.entities.values ?? []).filter(
        (entity) => entity.id.startsWith(start),
      ).length,
    prefix,
  );
}

/**
 * From the open world: draws a land of four corners on the map, names it, saves it, and waits
 * for its chip. The camera comes to rest over it.
 */
async function saveDrawnLand(app: Page, name: string): Promise<void> {
  await app.getByRole("button", { name: "Explore Earth", exact: true }).click();
  await app.getByTestId("tool-land").click();
  await lookDown(app, -119.9, 36.6);
  await app.getByRole("button", { name: "Draw on the map", exact: true }).click();
  for (const [x, y] of [
    [650, 300],
    [1000, 300],
    [1000, 550],
    [650, 550],
  ]) {
    await app.mouse.click(x!, y!);
  }
  const drawbar = app.getByTestId("land-drawbar");
  await expect(drawbar).toContainText("4 corners");
  await drawbar.getByRole("button", { name: "Done", exact: true }).click();
  await app.getByLabel("Name this land").fill(name);
  await app.getByRole("button", { name: "Save this land", exact: true }).click();
  await expect(app.getByTestId("land-chip")).toContainText(name);
  await expect(app.getByRole("heading", { name, exact: true })).toBeVisible();
  await cameraAtRest(app);
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
  await app.getByRole("button", { name: "Draw on the map", exact: true }).click();
  for (const [x, y] of [
    [650, 300],
    [1000, 300],
    [1000, 550],
    [650, 550],
  ]) {
    await app.mouse.click(x!, y!);
  }
  const drawbar = app.getByTestId("land-drawbar");
  await expect(drawbar).toContainText("4 corners");
  await drawbar.getByRole("button", { name: "Done", exact: true }).click();
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
  await app.getByRole("button", { name: "Land options", exact: true }).click();
  await app.getByRole("menuitem", { name: "Edit boundary", exact: true }).click();
  await app.getByText("Notes and source", { exact: true }).click();
  await app.getByLabel("Notes", { exact: true }).fill("Survey before restoration");
  await app.getByRole("button", { name: "Save revision", exact: true }).click();
  await expect.poll(() => records.history.length).toBe(2);
  await app.reload();
  await app.getByTestId("tool-land").click();
  await app.getByRole("button", { name: /West field/ }).click();
  await expect(app.getByText("Survey before restoration", { exact: true })).toBeVisible();
  await app.getByRole("button", { name: "Land options", exact: true }).click();
  await app.getByRole("menuitem", { name: "Boundary history", exact: true }).click();
  await expect(app.getByText(/^Revision 2 ·/)).toBeVisible();
});

test("ask what is here, then outline the land around that point", async ({ app }) => {
  const { asked } = await landApi(app);
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
              coordinates: rectangle(-119.905, 36.595, -119.895, 36.605),
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
  await lookDown(app, -119.9, 36.6);
  await app.mouse.click(800, 425, { button: "right" });
  await app.getByRole("menuitem", { name: "What's here" }).click();
  const place = app.getByTestId("inspector-place");
  await expect(place).toContainText("At this point · looked up live, not saved");
  await expect(place.getByTestId("place-elevation")).toContainText("236.8 m");
  await expect(place.getByTestId("place-flood-zone")).toContainText("No mapped zone");
  expect(new Set(asked.place)).toEqual(
    new Set(["elevation", "soil", "land-cover", "flood-zone", "ecoregion"]),
  );
  // Land opens on its parcel picker, already looking around that point.
  await place.getByRole("button", { name: "Outline this land" }).click();
  const picker = app.getByTestId("land-parcel-picker");
  await expect(picker.getByRole("checkbox", { name: "Creek field" })).toBeVisible();
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
  await saveDrawnLand(app, "West field");
  await app.getByTestId("land-lens-plans").click();
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

test("rehearse a plan on a saved land, and work only the ground it can", async ({ app }) => {
  const { asked } = await landApi(app, { rasters: ["dem"] });
  await saveDrawnLand(app, "Bench pasture");
  await app.getByTestId("land-lens-plans").click();
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
  expect(asked.workable[0]).toMatchObject({
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
  // Ground derived from the land, not the saved land itself; NHD answered, so it is named.
  await expect(plans.getByTestId("plan-ground")).toContainText("GROUND · DERIVED FROM YOUR LAND");
  await expect(plans.getByTestId("plan-ground")).toContainText(
    "Derived from USGS 3DEP 1/3″ bare-earth DEM, 10 m and USGS NHD creeks; the limits are assumed.",
  );
  await expect(plans.getByTestId("plan-ground")).toContainText(
    "Workable ground for a robotic brush mower: slope ≤ 25 % (assumed), 5 m clear of steeper cells, on USGS 3DEP 1/3″ DEM, published 2025-08-27",
  );
  await expect(plans.getByTestId("rehearsal-after")).toContainText(
    "Rehearsed again on the workable ground of Bench pasture",
    { timeout: 45_000 },
  );
  await expect.poll(() => entitiesNamed(app, "rehearsal:left-out:")).toBeGreaterThan(0);
  const after = Number(
    /(\d+) hands?/.exec((await plans.getByTestId("plan-approve").textContent()) ?? "")?.[1] ?? 0,
  );
  expect(after).toBeLessThan(before);
  // Closing the card takes the hatch with it.
  await plans.getByText("All plans").dispatchEvent("click");
  await expect.poll(() => entitiesNamed(app, "rehearsal:")).toBe(0);
});

test("show the fog of knowing, read a cell, and draft a scout flight with its DJI route", async ({
  app,
}) => {
  const { asked, frame } = await landApi(app);
  await saveDrawnLand(app, "West field");
  // Land closed, the land stays selected: its fog is in the strip.
  await app.getByTestId("tool-land").click();
  await expect(app.getByTestId("land-panel")).toHaveCount(0);
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
  const f = frame();
  await tapGround(app, (f.west + f.east) / 2, (f.south + f.north) / 2, { button: "right" });
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
  expect(asked.scouted[0]).toMatchObject({ threshold: 0.5, maxAreas: 3, cellM: 20 });
  // Drone missions are a stub until there is a drone (VITE_ENABLE_DRONE_EXPORT, off).
  if (process.env.VITE_ENABLE_DRONE_EXPORT !== "true") {
    await expect(scout.getByTestId("scout-drone-stub")).toHaveText(
      "Drone missions: not available yet",
    );
    await expect(scout.getByTestId("scout-download")).toHaveCount(0);
    return;
  }
  await expect(scout).toContainText("DJI Mavic 3 Enterprise wide camera (assumed)");
  await expect(scout).toContainText("Draft, not flight-tested");
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

/** Whether Show once's drape is on the map: the tapped cell's outline, and the tap cursor. */
function signatureDrape(app: Page): Promise<{ outline: boolean; cursor: string }> {
  return app.evaluate(() => {
    const twin = (window as unknown as { __twin: Twin }).__twin;
    const canvas = document.querySelector<HTMLElement>(".cesium-widget canvas");
    return {
      outline: twin.viewer.entities.getById("grid-outline:signature") !== undefined,
      cursor: canvas?.style.cursor ?? "",
    };
  });
}

/** Saves a drawn land "West field" with seasonal signatures and opens its Ecology lens. */
async function openEcologyOnNewLand(app: Page) {
  await saveDrawnLand(app, "West field");
  await app.getByTestId("land-lens-ecology").click();
  await expect(app.getByTestId("land-lens-card-ecology")).toBeVisible();
  await expect(app.getByTestId("show-once")).toBeVisible();
}

test("show once: tap a place, slide how alike, protect a patch and make it the job", async ({
  app,
}) => {
  const { asked, frame } = await landApi(app, { rasters: ["signature"] });
  await openEcologyOnNewLand(app);
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
  // The session is over: its drape and taps leave the map to the plan card.
  await expect.poll(() => signatureDrape(app)).toEqual({ outline: false, cursor: "" });
  expect(asked.zones[0]).toMatchObject({ threshold: 0.8 });
  // The protected patch travels as one row run: row 3, columns 2 and 3.
  const [run] = asked.zones[0]?.protected as number[][];
  expect(asked.zones[0]?.protected).toHaveLength(1);
  expect(run?.[0]).toBeCloseTo(f.west + ((f.east - f.west) * 2) / 4, 9);
  expect(run?.[3]).toBeCloseTo(f.south + (f.north - f.south) / 4, 9);
  // The plan card took the map; back in the Ecology lens, the card says what became of it.
  await app.getByTestId("tool-land").click();
  await expect(app.getByTestId("show-once-done")).toContainText("Drafted as a job: 1 area");
});

test("show once ends with a switch of site: the drape and the taps go with it", async ({ app }) => {
  const { frame } = await landApi(app, { rasters: ["signature"] });
  await openEcologyOnNewLand(app);
  await app.getByTestId("show-once-start").click();
  const f = frame();
  await tapGround(app, f.west + (f.east - f.west) / 8, f.north - (f.north - f.south) / 8);
  await expect(app.getByTestId("show-once-tally")).toContainText("6 cells");
  await expect.poll(() => signatureDrape(app)).toEqual({ outline: true, cursor: "crosshair" });
  await app.getByRole("button", { name: /switch site/ }).click();
  await app.getByTestId(`site-row-${RANCH_SLUG}`).click();
  await expect(app.getByTestId("project-card")).toContainText(RANCH_NAME, { timeout: 30_000 });
  // The land stays selected, its card offering to start again; nothing of it is left lit.
  await expect(app.getByTestId("show-once-start")).toBeVisible();
  await expect(app.getByTestId("show-once-tally")).toHaveCount(0);
  await expect.poll(() => signatureDrape(app)).toEqual({ outline: false, cursor: "" });
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
  await app.getByRole("button", { name: "Close land", exact: true }).click();
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
      interface PhoneTwin {
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
      const twin = (window as unknown as { __twin: PhoneTwin }).__twin;
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
    await lookDown(app, -119.9, 36.6);
    await app.getByTestId("phone-tab-more").click();
    await app.getByTestId("more-land").click();
    await app.getByRole("button", { name: "Draw on the map", exact: true }).click();
    // A finger taps: the copy says so.
    const drawbar = app.getByTestId("land-drawbar");
    await expect(drawbar).toContainText("Tap the corners of your land");
    // Above the draw bar, reaching down to where the taller review sheet will be.
    for (const [x, y] of [
      [60, 150],
      [330, 150],
      [330, 300],
      [60, 300],
    ]) {
      await app.mouse.click(x!, y!);
    }
    await expect(drawbar).toContainText("4 corners");
    await drawbar.getByRole("button", { name: "Done", exact: true }).click();
    await expect(app.getByTestId("land-review")).toContainText("New land");
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
    const chip = app.getByTestId("land-chip");
    await expect(chip.getByRole("heading", { name: "Phone field", exact: true })).toBeVisible();
    // Left on another lens, then opened again from the saved list at the foot of the start
    // card: the sheet shows the land's chip and its Overview, not the list or the last lens.
    await app.getByTestId("land-lens-plans").click();
    await expect(app.getByTestId("land-lens-card-plans")).toBeVisible();
    await app.getByRole("button", { name: "Land options", exact: true }).click();
    await app.getByRole("menuitem", { name: "Outline new land", exact: true }).click();
    const row = app
      .getByRole("region", { name: "Saved land" })
      .getByRole("button", { name: /Phone field/ });
    await row.scrollIntoViewIfNeeded();
    await row.click();
    await expect(chip).toBeInViewport();
    await expect(app.getByTestId("land-lens-card-overview")).toBeInViewport();
    await expect(app.getByTestId("land-lens-overview")).toHaveAttribute("aria-pressed", "true");
    await expect(app.getByTestId("land-lens-card-plans")).toHaveCount(0);
    // The agent shares the sheet folded to its one line, so the land stays in sight.
    await expect(app.getByTestId("land-agent-blocked")).toBeInViewport();
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
  await app.getByRole("button", { name: "Pick parcels", exact: true }).click();
  await app.getByRole("radio", { name: "Roads & rivers", exact: true }).click();
  await app.mouse.click(900, 400);
  const candidate = app.getByRole("checkbox", { name: "Transmission line 9" });
  await candidate.check();
  await expect(candidate).toBeChecked();
  await app.getByText("Describe it instead", { exact: true }).click();
  await app
    .getByLabel("Describe your selection")
    .fill("100 feet wide along this transmission line, flat ends");
  await app.getByRole("button", { name: "Use my description" }).click();
  await expect(app.getByLabel("Name this land")).toHaveValue("Transmission line 9");
  await app.getByText("Notes and source", { exact: true }).click();
  await expect(app.getByText("Corridor along Transmission line 9", { exact: true })).toBeVisible();
});

test("review and schedule an action, then preserve that mission when revising", async ({ app }) => {
  const records = await landApi(app);
  const boundary: Footprint = {
    type: "Polygon",
    coordinates: rectangle(-122.136, 47.644, -122.134, 47.646),
  };
  const landId = "11111111-1111-4111-8111-111111111111";
  records.saved.push({
    id: landId,
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
  const { versions, requests } = records.actions;
  await app.getByRole("button", { name: "Explore Earth", exact: true }).click();
  await app.getByTestId("tool-land").click();
  await app.getByRole("button", { name: /Action field/ }).click();
  await app.getByTestId("land-lens-plans").click();
  const plans = app.getByRole("region", { name: "Plans", exact: true });
  await plans.getByRole("button", { name: "Make a plan", exact: true }).click();
  // Save on an empty plan lists what to fix, sends nothing and puts the cursor on the first.
  await plans.getByRole("button", { name: "Save plan", exact: true }).click();
  await expect(plans.getByTestId("plans-draft-problems")).toContainText(
    /^Before saving, fix \d+ things:/,
  );
  await expect(plans.getByLabel("Plan name", { exact: true })).toBeFocused();
  expect(requests).toEqual([]);
  await plans.getByLabel("Plan name", { exact: true }).fill("Baseline survey");
  await plans.getByLabel("Goal", { exact: true }).fill("Measure native cover");
  await plans.getByLabel("Step 1 name", { exact: true }).fill("Survey quadrats");
  await plans.getByLabel("Step 1 success measure", { exact: true }).fill("Five recorded quadrats");
  await plans.getByRole("button", { name: "Save plan", exact: true }).click();
  await expect(plans.getByRole("region", { name: "Plan: Baseline survey" })).toBeVisible();
  expect(requests).toEqual(["draft"]);
  await plans.getByRole("button", { name: "Review and approve", exact: true }).click();
  // Approval waits for the review note, and says so beside the button.
  await expect(
    plans.getByRole("button", { name: "Approve revision 1", exact: true }),
  ).toBeDisabled();
  await expect(plans.getByRole("list", { name: "Before approving" })).toHaveText(
    "Add a review note. It is stored with the approval.",
  );
  await plans.getByLabel("Review note", { exact: true }).fill("Reviewed work area and access");
  await plans.getByRole("button", { name: "Approve revision 1", exact: true }).click();
  await plans.getByRole("button", { name: "Schedule as a mission", exact: true }).click();
  await plans.getByLabel("Scheduling note", { exact: true }).fill("Schedule the reviewed survey");
  expect(requests).toEqual(["draft", "approve"]);
  // The land lies in the scan site the map opened on, so the site's mission project keeps its
  // missions together; this land's own plans are the other choice.
  const under = plans.getByRole("combobox", { name: "Schedule under", exact: true });
  const siteProject = `site-${scanSite.id}`;
  await expect(under.locator("option")).toHaveText([
    "Action field · this land's plans",
    `${scanSite.name} · current mission project`,
  ]);
  await expect(under).toHaveValue(siteProject);
  await plans.getByRole("button", { name: "Schedule mission", exact: true }).click();
  await expect(plans.getByText(/Mission scheduled for/)).toBeVisible();
  await expect(plans.getByTestId("land-mission-status")).toContainText(
    `under the mission project ${scanSite.name}. Status: scheduled.`,
  );
  expect(records.actions.scheduled).toEqual([
    expect.objectContaining({ projectId: siteProject, note: "Schedule the reviewed survey" }),
  ]);
  // The mission is in Plan, private to the workspace, and leads back to the action in Land.
  await plans.getByTestId("land-mission-open-plan").click();
  const drawer = app.getByTestId("plans-panel");
  const mission = drawer.getByTestId("plan-detail");
  await expect(mission).toContainText("Baseline survey");
  await expect(mission.getByTestId("plan-land-chips")).toContainText("From land action");
  await expect(mission.getByTestId("plan-land-chips")).toContainText("Private to workspace");
  await mission.getByTestId("plan-revise-in-land").click();
  await expect(plans.getByTestId("land-mission-status")).toBeVisible();
  await plans.getByRole("button", { name: "Revise plan", exact: true }).click();
  await plans.getByLabel("Plan name", { exact: true }).fill("Survey and follow-up");
  await plans.getByRole("button", { name: "Save changes", exact: true }).click();
  await plans.getByRole("button", { name: "Review and approve", exact: true }).click();
  await expect(
    plans.getByRole("button", { name: "Approve revision 2", exact: true }),
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
    if (kind === "boundary")
      await app.getByRole("button", { name: "Draw on the map", exact: true }).click();
    else {
      await app.getByText("More ways", { exact: true }).click();
      await app.getByRole("button", { name: /Trace a corridor/ }).click();
    }
    const drawbar = app.getByTestId("land-drawbar");
    if (kind === "corridor") {
      await app.getByLabel("Total corridor width", { exact: true }).fill("150");
      await app.getByRole("combobox", { name: "Width units", exact: true }).selectOption("m");
      await expect(app.getByLabel("Total corridor width", { exact: true })).toHaveValue("45.72");
    }
    await app.mouse.click(650, 300);
    await app.mouse.click(1000, 300);
    await expect(drawbar).toContainText(/2 (corners|points)/);
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
    await expect(drawbar).toContainText(/2 (corners|points)/);
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
    await expect(drawbar).toContainText(/3 (corners|points)/);
    await drawbar.getByRole("button", { name: "Undo", exact: true }).click();
    await expect(drawbar).toContainText(/2 (corners|points)/);
    await drawbar.getByRole("button", { name: "Cancel", exact: true }).click();
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
