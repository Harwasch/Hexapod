/**
 * A site fly-to from the whole Earth down to a placed scan, in the app itself (SiteManager.flyTo,
 * CameraController.glide).
 *
 * On production this flight dived thirty kilometres into the earth -- its first aim was the
 * globe's placeholder height for a tile it had not loaded -- landed there in the black, then
 * climbed kilometres out of the ground and came back down to the scan, re-pointed twice more on
 * the way. Here the catalog has a phone-style scan (no catalog height, clamped to the ground,
 * the record slow to answer), and the camera is recorded every frame from the click to rest.
 *
 * Software GL draws a few frames a second, so the glide runs on a clock that steps 1/60 s per
 * frame (`setGlideClock`): the path is measured at 60 fps whatever the machine, with every
 * correction (the record, the terrain, the model resting on the ground) arriving whenever it
 * really does. Asserted: one flight, no second take-off; no frame kicks the camera off its path
 * by more than a hundredth of its height; never under the ground; the site picked is the one
 * the switcher names; and the frame it lands on is not black.
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

/** Degrees for metres east and north of the scan. */
const east = (m: number) => m / (111_320 * Math.cos((LAT * Math.PI) / 180));
const north = (m: number) => m / 111_320;

const ring = (half: number): [number, number][] => [
  [LON - east(half), LAT - north(half)],
  [LON + east(half), LAT - north(half)],
  [LON + east(half), LAT + north(half)],
  [LON - east(half), LAT + north(half)],
  [LON - east(half), LAT - north(half)],
];

/** The capture's own ground, as the pipeline measures it: a few cells around the tree. */
const CELLS = Array.from({ length: 16 }, (_, i) => {
  const r = 6 * Math.sqrt((i + 0.5) / 16);
  const a = i * 2.399963;
  return { lon: LON + east(r * Math.cos(a)), lat: LAT + north(r * Math.sin(a)), height: 0 };
});

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
        groundSamples: CELLS,
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

interface Frame {
  x: number;
  y: number;
  z: number;
  height: number;
  ground: number | null;
}

test("a fly-to from orbit to a placed scan is one smooth glide, above ground, onto a lit frame", async ({
  page,
}) => {
  test.setTimeout(240_000);
  await mockApi(page);
  // The catalog this test needs, over the mock's: registered after it, so asked first.
  await page.route("**/api/v1/sites", (route) =>
    route.request().method() === "GET"
      ? route.fulfill({ status: 200, json: [placedSummary] })
      : route.fallback(),
  );
  // The record answers late, as the API does from a cold start: the flight leaves without it.
  await page.route(`**/api/v1/sites/${SITE_ID}`, async (route) => {
    await new Promise((done) => setTimeout(done, 1_500));
    await route.fulfill({ status: 200, json: placedSite });
  });
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

  // CesiumJS draws the scan: what is measured is the camera, not a second renderer.
  await page.goto("/?renderer=cesium");
  await page.waitForFunction(() => "__twin" in window && Boolean(window.__twin), undefined, {
    timeout: 120_000,
  });
  await page.getByTestId("onboarding-explore").click();
  // The catalog is in, and the camera has come back to rest over the whole Earth.
  await page.waitForFunction(
    `window.__twin.sites.summaries.length === 1 && !window.__twin.camera.isMoving`,
    undefined,
    { timeout: 60_000 },
  );

  await page.evaluate(`(() => {
    const twin = window.__twin;
    let virtual = performance.now();
    twin.scene.preUpdate.addEventListener(() => { virtual += 1000 / 60; });
    twin.camera.setGlideClock(() => virtual);
    const record = (window.__flight = { frames: [], flightsBefore: twin.camera.flights });
    twin.scene.postRender.addEventListener(() => {
      if (!twin.camera.gliding) return;
      const camera = twin.viewer.camera;
      const c = camera.positionCartographic;
      const p = camera.positionWC;
      const ground = twin.scene.globe.getHeight(c);
      record.frames.push({ x: p.x, y: p.y, z: p.z, height: c.height, ground: ground ?? null });
    });
  })()`);

  await page.keyboard.press("s");
  await page.getByTestId("site-row-tree-scan").click();
  // Landed, and the model rests where it will: no glide under way any more.
  await page.waitForFunction(
    `window.__twin.sites.flight?.state === "landed" && !window.__twin.camera.gliding && window.__twin.sites.activeSite?.id === "${SITE_ID}"`,
    undefined,
    { timeout: 180_000 },
  );

  const { frames, flights } = await page.evaluate<{ frames: Frame[]; flights: number }>(`({
    frames: window.__flight.frames,
    flights: window.__twin.camera.flights - window.__flight.flightsBefore,
  })`);
  expect(frames.length).toBeGreaterThan(60);
  // One glide from the click; at most one more, from rest, if the model came to rest only
  // after landing -- never a climb back out of the ground.
  expect(flights).toBeLessThanOrEqual(2);

  // No frame kicks the camera off its path: the second difference of its position, against
  // its height above the ground, as glide.test.ts measures it (a restart from rest kicks by
  // several hundredths).
  let worst = 0;
  for (let i = 1; i + 1 < frames.length; i++) {
    const [a, b, c] = [frames[i - 1], frames[i], frames[i + 1]];
    if (!a || !b || !c) continue;
    const kick = Math.hypot(a.x + c.x - 2 * b.x, a.y + c.y - 2 * b.y, a.z + c.z - 2 * b.z);
    const ground = b.ground !== null && b.ground > -500 && b.ground < 9_000 ? b.ground : 0;
    worst = Math.max(worst, kick / Math.max(b.height - ground, 1));
  }
  expect(worst).toBeLessThan(0.01);

  // Never under the ground the globe has, and down from orbit without climbing again.
  for (const frame of frames) {
    const ground = frame.ground !== null && frame.ground > -500 ? frame.ground : 0;
    expect(frame.height).toBeGreaterThan(ground);
  }
  const lowest = frames.reduce((low, f, i) => (f.height < (frames[low]?.height ?? 0) ? i : low), 0);
  const climb =
    Math.max(...frames.slice(lowest).map((f) => f.height)) - (frames[lowest]?.height ?? 0);
  expect(climb).toBeLessThan(50);

  // The switcher names the scan flown to.
  await expect(page.getByTestId("project-title")).toContainText("Tree scan");

  // The frame it lands on is lit: the globe and the scan, not the inside of the earth.
  const png = (await page.screenshot()).toString("base64");
  const light = await page.evaluate(async (data) => {
    const image = new Image();
    image.src = `data:image/png;base64,${data}`;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext("2d");
    if (!context) return { mean: 0, dark: 1 };
    context.drawImage(image, 0, 0);
    // The middle of the view, clear of the HUD.
    const w = Math.floor(image.width / 2);
    const h = Math.floor(image.height / 2);
    const pixels = context.getImageData(w / 2, h / 2, w, h).data;
    let sum = 0;
    let dark = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      const l =
        0.2126 * (pixels[i] ?? 0) + 0.7152 * (pixels[i + 1] ?? 0) + 0.0722 * (pixels[i + 2] ?? 0);
      sum += l;
      if (l < 12) dark += 1;
    }
    const n = pixels.length / 4;
    return { mean: sum / n, dark: dark / n };
  }, png);
  expect(light.dark).toBeLessThan(0.5);
  expect(light.mean).toBeGreaterThan(20);
});
