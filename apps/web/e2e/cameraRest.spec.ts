/**
 * Zoom close to a scan and stay still: the camera does not move (cameraOwnership.ts).
 *
 * On production a close-up the wheel had just made was undone a moment later, three ways: a
 * finer terrain tile landed under the camera and CesiumJS's terrain collision lifted the camera
 * onto it; the floor check after the camera came to rest found a surface above it (a coarse
 * world tile) and glided it up; and a fly-to's late, better pose could still settle a camera
 * nobody had moved. Here the person flies to a placed scan, zooms in with the wheel and lets
 * go, and then all three happen to the resting camera -- the terrain under it rises to within
 * the zoom floor, the drawn surface is put above it and the floor check runs, and a better
 * arrival pose comes in -- while frames are drawn for several seconds. The camera must not
 * move by a millimetre. Before the person touches it, the same drawn surface still lifts the
 * app's own landing, so the test's surface is one the floor check sees.
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, mockApi, test } from "./fixtures";

const TILES = resolve(process.cwd(), "../../data/tiles");
/** Where `synthetic_tree.py` placed the tree (its tileset's root transform). */
const LON = -82.6966;
const LAT = 28.0389;
const SITE_ID = "77777777-7777-4777-8777-777777777777";
const ASSET_ID = "88888888-8888-4888-8888-888888888888";

const east = (m: number) => m / (111_320 * Math.cos((LAT * Math.PI) / 180));
const north = (m: number) => m / 111_320;
const ring = (half: number): [number, number][] => [
  [LON - east(half), LAT - north(half)],
  [LON + east(half), LAT - north(half)],
  [LON + east(half), LAT + north(half)],
  [LON - east(half), LAT + north(half)],
  [LON - east(half), LAT - north(half)],
];

const placedSite = {
  id: SITE_ID,
  slug: "tree-scan",
  name: "Tree scan",
  description: null,
  boundary: { type: "Polygon", coordinates: [ring(40)] },
  centroid: { longitude: LON, latitude: LAT, height: null },
  areaM2: 6_400,
  thumbnailUrl: null,
  metadata: {},
  attribution: [],
  license: null,
  assets: [
    {
      id: ASSET_ID,
      siteId: SITE_ID,
      provider: "3d-tiles-url",
      name: "Gaussian splat",
      representation: "gaussian-splat",
      source: { type: "3d-tiles-url", url: "/fixture-tiles/synthetic-tree-lod/tileset.json" },
      footprint: null,
      observedAt: null,
      validFrom: null,
      validTo: null,
      resolution: null,
      crs: null,
      license: null,
      attribution: [],
      provenance: null,
      renderConfig: {
        maximumScreenSpaceError: 16,
        pointCloudShading: null,
        clipsWorld: true,
        clipFootprint: "catalog",
        heightOffsetM: 0,
        clampToGround: true,
      },
      defaultVisible: true,
      createdAt: "2026-10-01T00:00:00Z",
      updatedAt: "2026-10-01T00:00:00Z",
    },
  ],
  cameraBookmarks: [],
  createdAt: "2026-10-01T00:00:00Z",
  updatedAt: "2026-10-01T00:00:00Z",
};

const placedSummary = {
  id: SITE_ID,
  slug: placedSite.slug,
  name: placedSite.name,
  description: null,
  centroid: placedSite.centroid,
  areaM2: placedSite.areaM2,
  thumbnailUrl: null,
  representations: ["gaussian-splat"],
  latestObservedAt: null,
  quality: null,
  createdAt: placedSite.createdAt,
  updatedAt: placedSite.updatedAt,
};

/** Camera position (Earth-fixed) and heading, as the test reads it. */
interface Pose {
  x: number;
  y: number;
  z: number;
  height: number;
  heading: number;
}

test("zoom close and stay still: the camera does not move", async ({ page }) => {
  test.setTimeout(420_000);
  await page.setViewportSize({ width: 960, height: 600 });
  await mockApi(page);
  await page.route("**/api/v1/sites", (route) =>
    route.request().method() === "GET"
      ? route.fulfill({ status: 200, json: [placedSummary] })
      : route.fallback(),
  );
  await page.route(`**/api/v1/sites/${SITE_ID}`, (route) =>
    route.fulfill({ status: 200, json: placedSite }),
  );
  await page.route("**/fixture-tiles/**", (route) => {
    const relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    const file = resolve(TILES, relative);
    if (!existsSync(file)) return route.fulfill({ status: 404, body: "" });
    return route.fulfill({
      status: 200,
      contentType: relative.endsWith(".json") ? "application/json" : "application/octet-stream",
      body: readFileSync(file),
    });
  });

  // The default renderer: the scan is drawn over the globe and CesiumJS's own wheel zoom and
  // terrain collision move the camera, as on production.
  await page.goto("/?renderer=playcanvas");
  await page.waitForFunction(() => "__twin" in window && Boolean(window.__twin), undefined, {
    timeout: 120_000,
  });
  await page.getByTestId("onboarding-explore").click();
  await page.waitForFunction(
    `window.__twin.sites.summaries.length === 1 && !window.__twin.camera.isMoving`,
    undefined,
    { timeout: 60_000 },
  );

  // Fly there, as the switcher does, and land.
  await page.evaluate(`void window.__twin.sites.flyTo("${SITE_ID}")`);
  await page.waitForFunction(
    `window.__twin.sites.flight?.state === "landed" && !window.__twin.camera.gliding`,
    undefined,
    { timeout: 300_000 },
  );
  expect(await page.evaluate(`window.__twin.camera.userHasCamera`)).toBe(false);

  // What the camera does at rest, and what happens to a resting camera after a zoom on
  // production: the terrain under it rises to within the zoom floor, the floor check finds the
  // drawn surface above it, and a better arrival pose arrives.
  await page.evaluate(`(() => {
    const twin = window.__twin;
    const scene = twin.scene;
    const camera = twin.viewer.camera;
    window.__rest = {
      frames: [],
      record() {
        const p = camera.positionWC;
        this.frames.push({
          x: p.x, y: p.y, z: p.z,
          height: camera.positionCartographic.height,
          heading: camera.heading,
        });
      },
      /** The drawn surface the floor check reads, put this far above the camera. */
      surfaceAbove(metres) {
        const at = camera.positionCartographic.height + metres;
        scene.sampleHeight = () => at;
        scene.sampleHeightSupported = true;
      },
      /** A finer terrain tile under the camera, its top this far below the camera. */
      terrainBelow(metres) {
        const top = camera.positionCartographic.height - metres;
        scene.globe.getHeight = () => top;
        scene._globeHeightDirty = true;
      },
    };
    scene.postRender.addEventListener(() => window.__rest.record());
  })()`);

  // Control: the app's own landing, untouched, is still eased off a surface above it.
  const landedHeight = await page.evaluate<number>(`(() => {
    const twin = window.__twin;
    window.__rest.surfaceAbove(0.3);
    twin.viewer.camera.moveEnd.raiseEvent();
    return twin.viewer.camera.positionCartographic.height;
  })()`);
  await page.waitForFunction(
    `window.__twin.viewer.camera.positionCartographic.height > ${landedHeight + 0.5} && !window.__twin.camera.gliding`,
    undefined,
    { timeout: 60_000, polling: 250 },
  );

  // The person zooms in with the wheel over the scan, and lets go.
  const box = await page.locator("canvas").first().boundingBox();
  if (!box) throw new Error("no canvas");
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  const before = await page.evaluate<number>(
    `window.__twin.viewer.camera.positionCartographic.height`,
  );
  for (let notch = 0; notch < 6; notch++) {
    await page.mouse.wheel(0, -200);
    await page.waitForTimeout(300);
  }
  expect(await page.evaluate(`window.__twin.camera.userHasCamera`)).toBe(true);
  // At rest: the gesture and its inertia are over, nothing moved for two seconds.
  await page.waitForFunction(
    `(() => {
      const twin = window.__twin;
      const p = twin.viewer.camera.positionWC;
      const now = performance.now();
      const last = (window.__still ??= { x: p.x, y: p.y, z: p.z, since: now });
      if (Math.hypot(p.x - last.x, p.y - last.y, p.z - last.z) > 1e-4) {
        window.__still = { x: p.x, y: p.y, z: p.z, since: now };
        twin.scene.requestRender();
        return false;
      }
      twin.scene.requestRender();
      return now - last.since > 2000;
    })()`,
    undefined,
    { timeout: 120_000, polling: 200 },
  );
  const zoomed = await page.evaluate<number>(
    `window.__twin.viewer.camera.positionCartographic.height`,
  );
  expect(zoomed).toBeLessThan(before);

  // Now everything that used to move a resting close-up happens to it, and frames are drawn.
  const rest = await page.evaluate<Pose>(`(() => {
    const twin = window.__twin;
    const rest = window.__rest;
    rest.frames.length = 0;
    rest.record();
    // A finer terrain tile under the close-up, 0.2 m below it: within the 0.6 m zoom floor.
    rest.terrainBelow(0.2);
    // The floor check, as after any move: a drawn surface 2 m above the camera.
    rest.surfaceAbove(2);
    twin.viewer.camera.moveEnd.raiseEvent();
    // The model's late clamp: a better arrival pose, 5 m east.
    const flight = twin.sites.flight;
    const pose = flight.pose;
    twin.sites.steer(flight.serial, {
      ...pose,
      longitude: pose.longitude + 5 / (111320 * Math.cos(pose.latitude * Math.PI / 180)),
    });
    window.__ticker = setInterval(() => twin.scene.requestRender(), 100);
    return rest.frames[0];
  })()`);
  await page.waitForTimeout(6_000);
  const frames = await page.evaluate<Pose[]>(`(() => {
    clearInterval(window.__ticker);
    return window.__rest.frames;
  })()`);

  expect(frames.length).toBeGreaterThan(5);
  let drift = 0;
  let turn = 0;
  for (const frame of frames) {
    drift = Math.max(drift, Math.hypot(frame.x - rest.x, frame.y - rest.y, frame.z - rest.z));
    turn = Math.max(turn, Math.abs(frame.heading - rest.heading));
  }
  expect(drift).toBeLessThan(0.001);
  expect(turn).toBeLessThan(1e-6);
  expect(await page.evaluate(`window.__twin.camera.gliding`)).toBe(false);
});
