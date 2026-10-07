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

import type { Page } from "@playwright/test";

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

/** The catalog with the placed tree scan in it, and the scan's tiles from `data/tiles`. */
async function serveTreeScan(page: Page): Promise<void> {
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
}

/** What the test has done to the page that outlives a reload (the welcome card's dismissal). */
interface Visit {
  explored: boolean;
}

/** Opens the app, flies to the tree scan, and waits until it lands and the scan is drawn. */
async function arrive(page: Page, visit: Visit): Promise<void> {
  // The default renderer: the scan is drawn over the globe and CesiumJS's own wheel zoom and
  // terrain collision move the camera, as on production.
  await page.goto("/?renderer=playcanvas");
  // Marks this document: a reload replaces it (`zoomCloseAndStayStill`'s caller).
  await page.evaluate(`window.__cameraRestPage = true`);
  await page.waitForFunction(() => "__twin" in window && Boolean(window.__twin), undefined, {
    timeout: 120_000,
  });
  // A frame drawn by the app this page loaded.
  await page.waitForFunction(
    `(() => {
      const scene = window.__twin.scene;
      scene.requestRender();
      return scene.frameState.frameNumber > 0;
    })()`,
    undefined,
    { timeout: 60_000, polling: 250 },
  );
  // The welcome card, the first time: its dismissal is remembered across a reload.
  if (!visit.explored) {
    await page.getByTestId("onboarding-explore").click();
    visit.explored = true;
  }
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
  // The scan drawn, its tiles in: by then the renderer, its tile worker and the decoder have
  // all loaded, and starting them (seconds of main thread on software GL) is behind us.
  await page.waitForFunction(
    `(() => {
      const twin = window.__twin;
      twin.scene.requestRender();
      const scan = twin.scanRendererStatus;
      return scan.active && scan.tiles > 0 && scan.frames > 0 && scan.loading === 0;
    })()`,
    undefined,
    { timeout: 120_000, polling: 250 },
  );
}

/**
 * Waits until the camera has held still for `ms` while frames are drawn (CesiumJS raises
 * `moveEnd`, and with it the floor check, only from a drawn frame).
 */
async function waitForStill(page: Page, ms: number): Promise<void> {
  await page.evaluate(`window.__still = undefined`);
  await page.waitForFunction(
    `(() => {
      const twin = window.__twin;
      const p = twin.viewer.camera.positionWC;
      const now = performance.now();
      twin.scene.requestRender();
      const last = (window.__still ??= { x: p.x, y: p.y, z: p.z, since: now });
      if (
        twin.camera.isMoving ||
        twin.camera.gliding ||
        Math.hypot(p.x - last.x, p.y - last.y, p.z - last.z) > 1e-4
      ) {
        window.__still = { x: p.x, y: p.y, z: p.z, since: now };
        return false;
      }
      return now - last.since > ${ms};
    })()`,
    undefined,
    { timeout: 120_000, polling: 200 },
  );
}

/** The whole scenario, on a freshly opened page. */
async function zoomCloseAndStayStill(page: Page, visit: Visit): Promise<void> {
  await arrive(page, visit);
  expect(await page.evaluate(`window.__twin.camera.userHasCamera`)).toBe(false);

  // What the camera does at rest, and what happens to a resting camera after a zoom on
  // production: the terrain under it rises to within the zoom floor, the floor check finds the
  // drawn surface above it, and a better arrival pose arrives.
  await page.evaluate(`(() => {
    const twin = window.__twin;
    const scene = twin.scene;
    const camera = twin.viewer.camera;
    const real = {
      sampleHeight: scene.sampleHeight,
      supported: Object.getOwnPropertyDescriptor(scene, "sampleHeightSupported"),
      raycast: twin.collider.raycast,
    };
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
        // CesiumJS's is a getter (whether the GPU has depth textures); this sample always works.
        Object.defineProperty(scene, "sampleHeightSupported", { value: true, configurable: true });
      },
      /**
       * No scanned solids straight under the camera, as for a scan packaged without them (the
       * Pumpkin's): the floor check reads the drawn surface instead.
       */
      noSolidsBelow() {
        twin.collider.raycast = () => null;
      },
      /** The real drawn surface and solids again. */
      restore() {
        scene.sampleHeight = real.sampleHeight;
        if (real.supported) Object.defineProperty(scene, "sampleHeightSupported", real.supported);
        else delete scene.sampleHeightSupported;
        twin.collider.raycast = real.raycast;
      },
      /**
       * A finer terrain tile under the camera, its top this far below the camera, there long
       * enough that CesiumJS's collision believes it (it acts on a camera nobody moves only
       * once the height under it has held steady).
       */
      terrainBelow(metres) {
        const top = camera.positionCartographic.height - metres;
        scene.globe.getHeight = () => top;
        scene._globeHeightDirty = true;
        scene.screenSpaceCameraController._lastGlobeHeight = top;
      },
      /**
       * Ends a move every two seconds, as the camera's own moveEnd would, until 'stop'. The
       * floor check runs at most once in 1.5 s, so a single moveEnd can land inside that
       * window -- behind the landing's own check, when a stall on software GL delayed that one
       * by seconds -- and be ignored; the next one is not.
       */
      endMoves() {
        camera.moveEnd.raiseEvent();
        this.mover = setInterval(() => camera.moveEnd.raiseEvent(), 2000);
      },
      stop() {
        clearInterval(this.mover);
      },
    };
    scene.postRender.addEventListener(() => window.__rest.record());
  })()`);

  // Control: the app's own landing, untouched, is still eased off a surface above it.
  await waitForStill(page, 1_000);
  const landedHeight = await page.evaluate<number>(`(() => {
    const rest = window.__rest;
    rest.noSolidsBelow();
    rest.surfaceAbove(1);
    rest.endMoves();
    return window.__twin.viewer.camera.positionCartographic.height;
  })()`);
  await page.waitForFunction(
    `(() => {
      const twin = window.__twin;
      twin.scene.requestRender();
      return twin.viewer.camera.positionCartographic.height > ${landedHeight + 1} &&
        !twin.camera.gliding;
    })()`,
    undefined,
    { timeout: 60_000, polling: 250 },
  );
  await page.evaluate(`(() => {
    window.__rest.stop();
    window.__rest.restore();
  })()`);
  await waitForStill(page, 1_000);

  // The person zooms in with the wheel over the scan, and lets go.
  const box = await page.locator("canvas").first().boundingBox();
  if (!box) throw new Error("no canvas");
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  const before = await page.evaluate<number>(
    `window.__twin.viewer.camera.positionCartographic.height`,
  );
  // A few notches, and a few more if a slow software-GL frame swallowed the first ones.
  for (let round = 0; round < 3; round++) {
    for (let notch = 0; notch < 6; notch++) {
      await page.mouse.wheel(0, -200);
      await page.waitForTimeout(300);
    }
    await page.waitForTimeout(1_000);
    const now = await page.evaluate<number>(
      `window.__twin.viewer.camera.positionCartographic.height`,
    );
    if (now < before - 0.01) break;
  }
  expect(await page.evaluate(`window.__twin.camera.userHasCamera`)).toBe(true);
  // At rest: the gesture and its inertia are over, nothing moved for two seconds.
  await waitForStill(page, 2_000);
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
    rest.noSolidsBelow();
    // A finer terrain tile under the close-up, 0.2 m below it: within the 0.6 m zoom floor.
    rest.terrainBelow(0.2);
    // The floor check, as after any move: a drawn surface 2 m above the camera.
    rest.surfaceAbove(2);
    rest.endMoves();
    // The model's late clamp: a better arrival pose, 5 m east. Its own guard refuses it here
    // too, since the zoom moved the camera from where it landed; a hand that has not moved it
    // yet is siteFlight.test.ts's case.
    const flight = twin.sites.flight;
    const pose = flight.pose;
    twin.sites.steer(flight.serial, {
      ...pose,
      longitude: pose.longitude + 5 / (111320 * Math.cos(pose.latitude * Math.PI / 180)),
    });
    window.__ticker = setInterval(() => twin.scene.requestRender(), 100);
    return rest.frames[0];
  })()`);
  // Six seconds (three moves ended) and at least ten frames drawn, however slowly software GL
  // draws them.
  await page.waitForTimeout(6_000);
  await page.waitForFunction(`window.__rest.frames.length >= 10`, undefined, {
    timeout: 120_000,
    polling: 250,
  });
  const frames = await page.evaluate<Pose[]>(`(() => {
    clearInterval(window.__ticker);
    window.__rest.stop();
    return window.__rest.frames;
  })()`);

  expect(frames.length).toBeGreaterThanOrEqual(10);
  let drift = 0;
  let turn = 0;
  for (const frame of frames) {
    drift = Math.max(drift, Math.hypot(frame.x - rest.x, frame.y - rest.y, frame.z - rest.z));
    turn = Math.max(turn, Math.abs(frame.heading - rest.heading));
  }
  expect(drift).toBeLessThan(0.001);
  expect(turn).toBeLessThan(1e-6);
  expect(await page.evaluate(`window.__twin.camera.gliding`)).toBe(false);
}

test("zoom close and stay still: the camera does not move", async ({ page }) => {
  test.setTimeout(600_000);
  await page.setViewportSize({ width: 960, height: 600 });
  await serveTreeScan(page);

  // A reload under the test resets the app and everything the test set up in it. The dev
  // server used to do that once on a cold start, when the scan's tile worker first imported a
  // package it had not bundled (vite.config.ts, `optimizeDeps`); should anything like it
  // happen again, the scenario starts over on a fresh page instead of failing on a page that
  // no longer has the app in it. Any other failure is the test's verdict.
  const visit: Visit = { explored: false };
  for (let attempt = 1; ; attempt += 1) {
    try {
      await zoomCloseAndStayStill(page, visit);
      return;
    } catch (error) {
      const samePage = await page
        .evaluate<boolean>(`window.__cameraRestPage === true`)
        .catch(() => false);
      if (samePage || attempt >= 2) throw error;
      console.warn(`cameraRest: the page reloaded under attempt ${attempt}; starting over`);
    }
  }
});
