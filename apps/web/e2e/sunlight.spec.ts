/**
 * Sunlight in the app itself (ADR 0019 "Sunlight"), under each splat renderer.
 *
 * A site with two dated captures of the synthetic yard: E1 at local midnight on Mar 1 (the sun
 * well below the horizon at 46° N) and E2 mid-morning on Jun 1. Under CesiumJS, through
 * `window.__twin` and what is on screen:
 *
 * - Off: nothing in the shown model's colour chain, no badge;
 * - Now: the relight part in the chain, and the badge "Simulated lighting: the sun (now) over
 *   shadows baked in at capture, {the capture's date and time} UTC";
 * - At capture: the scene clock stands at the shown date's capture time, still, the badge says
 *   "(at capture)"; the March date moves it to March's midnight and the badge says "Sun below
 *   horizon";
 * - As captured on: no part, no badge, the clock given back (running, today), the control
 *   saying why; off again: held again;
 * - Off: the clock given back, running, today.
 *
 * Under PlayCanvas (the default renderer) the same, drawn by PlayCanvas's per-frame hook (what
 * the renderer reads out: state/sunReport.ts): Now relights and says so, At capture holds
 * the clock and gives it back, As captured takes the relight off. Under Spark: the Sun control
 * disabled, saying it needs CesiumJS or PlayCanvas, and "Draw with PlayCanvas" really
 * switching, after which Now relights. With the sun on, a switch from CesiumJS to PlayCanvas and
 * back hands the relight over each way and keeps the badge. The tool rail is as it was.
 * Screenshots go to `W6B_EVIDENCE_DIR` when set.
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { Page } from "@playwright/test";

import { expect, mockApi, test } from "./fixtures";

const TILES = resolve(process.cwd(), "../../data/tiles");
const EVIDENCE = process.env.W6B_EVIDENCE_DIR;
const YARD = "/fixture-tiles/synthetic-yard/splat/tileset.json";
const SITE_ID = "66666666-6666-4666-8666-666666666666";
const E1 = "66666666-6666-4666-8666-000000000001";
const E2 = "66666666-6666-4666-8666-000000000002";
const A1 = "66666666-6666-4666-8666-0000000000a1";
const A2 = "66666666-6666-4666-8666-0000000000a2";
/** Local midnight at 123.88° W, and 10:15 local solar time. */
const MARCH = "2026-03-01T08:15:00Z";
const JUNE = "2026-06-01T18:30:00Z";

const LON = -123.88;
const LAT = 46.1335;
const east = (m: number) => m / (111_320 * Math.cos((LAT * Math.PI) / 180));
const north = (m: number) => m / 111_320;
const MID = { lon: LON + east(16), lat: LAT + north(12) };
const ring = (half: number): [number, number][] => [
  [MID.lon - east(half), MID.lat - north(half)],
  [MID.lon + east(half), MID.lat - north(half)],
  [MID.lon + east(half), MID.lat + north(half)],
  [MID.lon - east(half), MID.lat + north(half)],
  [MID.lon - east(half), MID.lat - north(half)],
];
const CELLS = Array.from({ length: 16 }, (_, i) => {
  const r = 14 * Math.sqrt((i + 0.5) / 16);
  const a = i * 2.399963;
  return { lon: MID.lon + east(r * Math.cos(a)), lat: MID.lat + north(r * Math.sin(a)), height: 0 };
});
const footprint = { type: "MultiPolygon", coordinates: [[ring(40)]] };

function epoch(id: string, ordinal: number, observedAt: string, platform: string, asset: string) {
  return {
    id,
    siteId: SITE_ID,
    ordinal,
    captureId: null,
    jobId: null,
    observedAt,
    observedAtSource: "captured-at",
    footprint,
    platform,
    status: "published",
    joinMethod: ordinal === 1 ? "new-site" : "spatial",
    alignment: null,
    quality: null,
    currentAssetId: asset,
    createdAt: observedAt,
    updatedAt: observedAt,
  };
}

function asset(id: string, epochId: string, url: string, createdAt: string) {
  return {
    id,
    siteId: SITE_ID,
    provider: "3d-tiles-url",
    name: "Gaussian splat",
    representation: "gaussian-splat",
    source: { type: "3d-tiles-url", url },
    footprint: null,
    observedAt: createdAt,
    validFrom: null,
    validTo: null,
    resolution: null,
    crs: null,
    license: null,
    attribution: [],
    provenance: null,
    epochId,
    sourceKind: "observed",
    sidecarFlags: [],
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
    createdAt,
    updatedAt: createdAt,
  };
}

const site = {
  id: SITE_ID,
  slug: "yard-sunlight",
  name: "Yard, sunlight",
  description: null,
  boundary: { type: "Polygon", coordinates: [ring(40)] },
  centroid: { longitude: MID.lon, latitude: MID.lat, height: null },
  areaM2: 6_400,
  thumbnailUrl: null,
  metadata: {},
  attribution: [],
  license: null,
  assets: [
    asset(A1, E1, YARD, "2026-03-02T00:00:00Z"),
    asset(A2, E2, `${YARD}?epoch=2`, "2026-06-02T00:00:00Z"),
  ],
  epochs: [epoch(E1, 1, MARCH, "drone", A1), epoch(E2, 2, JUNE, "phone-video", A2)],
  frame: null,
  cameraBookmarks: [],
  createdAt: "2026-03-02T00:00:00Z",
  updatedAt: "2026-06-02T00:00:00Z",
};

const summary = {
  id: SITE_ID,
  slug: site.slug,
  name: site.name,
  description: null,
  centroid: site.centroid,
  areaM2: site.areaM2,
  thumbnailUrl: null,
  representations: ["gaussian-splat"],
  latestObservedAt: JUNE,
  quality: null,
  createdAt: site.createdAt,
  updatedAt: site.updatedAt,
};

async function route(page: Page): Promise<void> {
  await mockApi(page);
  await page.route("**/api/v1/sites", (r) =>
    r.request().method() === "GET" ? r.fulfill({ status: 200, json: [summary] }) : r.fallback(),
  );
  await page.route(`**/api/v1/sites/${SITE_ID}`, (r) => r.fulfill({ status: 200, json: site }));
  await page.route("**/fixture-tiles/**", (r) => {
    const relative = new URL(r.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return r.abort();
    const file = resolve(TILES, relative);
    if (!existsSync(file)) return r.fulfill({ status: 404, body: "" });
    return r.fulfill({
      status: 200,
      contentType: relative.endsWith(".json") ? "application/json" : "application/octet-stream",
      body: readFileSync(file),
    });
  });
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (r) => r.abort());
}

interface SceneState {
  shown: string | null;
  parts: Record<string, string[]>;
  clock: string;
  animating: boolean;
}

/** The colour parts in each loaded date's primitive, the shown date, and the scene clock. */
const state = (page: Page): Promise<SceneState> =>
  page.evaluate(`(() => {
    const twin = window.__twin;
    twin.scene.requestRender();
    const parts = {};
    for (const a of twin.sites.loadedAssets()) {
      const chain = a.tileset.gaussianSplatPrimitive?.vertexColor;
      parts[a.assetId] = (chain?.parts ?? []).map((p) => p.colorFunction);
    }
    const clock = twin.viewer.clock;
    const time = clock.currentTime;
    return {
      shown: twin.sites.shownAsset()?.id ?? null,
      parts,
      clock: time.constructor.toIso8601(time, 0),
      animating: clock.shouldAnimate,
    };
  })()`);

/** Seconds between the scene clock and the page's own wall clock. */
const clockLag = (page: Page): Promise<number> =>
  page.evaluate(`(() => {
    const time = window.__twin.viewer.clock.currentTime;
    return Math.abs(time.constructor.toDate(time).getTime() - Date.now()) / 1000;
  })()`);

async function arrive(page: Page, renderer: string): Promise<void> {
  await page.setViewportSize({ width: 1280, height: 760 });
  await route(page);
  await page.goto(`/?renderer=${renderer}`);
  await page.waitForFunction(() => "__twin" in window && Boolean(window.__twin), undefined, {
    timeout: 120_000,
  });
  await page.getByTestId("onboarding-explore").click();
  await page.waitForFunction(`window.__twin.sites.summaries.length === 1`, undefined, {
    timeout: 60_000,
  });
  await page.evaluate(`void window.__twin.sites.flyTo("${SITE_ID}")`);
  await page.waitForFunction(
    `window.__twin.sites.flight?.state === "landed" && !window.__twin.camera.gliding && window.__twin.sites.activeSite?.id === "${SITE_ID}"`,
    undefined,
    { timeout: 240_000 },
  );
}

async function shot(page: Page, name: string): Promise<void> {
  await page.evaluate(`window.__twin.scene.requestRender()`);
  await page.waitForTimeout(1_500);
  const path = EVIDENCE ? `${EVIDENCE}/${name}.png` : test.info().outputPath(`${name}.png`);
  await page.screenshot({ path });
  await test.info().attach(name, { path, contentType: "image/png" });
}

async function railUnchanged(page: Page): Promise<void> {
  await expect(page.getByRole("toolbar", { name: "Tools" }).getByRole("button")).toHaveText([
    ...(process.env.VITE_ENABLE_LAND_EXPLORATION === "true" ? ["Land"] : []),
    "Layers",
    "Measure",
    "Add",
    "Settings",
  ]);
}

const poll = { timeout: 120_000, intervals: [250, 500, 1000] };
const RELIT = "splatSunlightColor";

test("under cesium, the sun relights the shown date, says so, and At capture holds and gives back the clock", async ({
  page,
}) => {
  test.setTimeout(420_000);
  await arrive(page, "cesium");
  const sun = page.getByTestId("views-sun");
  await expect(sun).toBeVisible();
  const radio = (name: string) => sun.getByRole("radio", { name });
  for (const name of ["Off", "Now", "At capture"]) await expect(radio(name)).toBeEnabled();
  await expect(radio("Off")).toHaveAttribute("aria-checked", "true");
  await expect.poll(async () => (await state(page)).shown, poll).toBe(A2);
  expect((await state(page)).parts[A2] ?? []).not.toContain(RELIT);
  await expect(page.getByTestId("relit-badge")).toHaveCount(0);
  await shot(page, "app-off");

  // Now: relit, and said to be, over June's capture.
  await radio("Now").click();
  await expect.poll(async () => (await state(page)).parts[A2] ?? [], poll).toContain(RELIT);
  const badge = page.getByTestId("relit-badge");
  await expect(badge).toContainText(
    /Simulated lighting: the sun \(now\) over shadows baked in at capture, .*Jun.*1, 2026.* UTC/,
    { timeout: 60_000 },
  );
  expect((await state(page)).animating).toBe(true);
  await shot(page, "app-now");

  // At capture: the clock stands at June's capture time.
  await radio("At capture").click();
  await expect.poll(async () => (await state(page)).clock, poll).toBe("2026-06-01T18:30:00Z");
  expect((await state(page)).animating).toBe(false);
  await expect(badge).toContainText("Simulated lighting: the sun (at capture)");
  await expect(page.getByTestId("relit-night")).toHaveCount(0);
  await shot(page, "app-at-capture-jun");

  // March: its own capture time, at midnight -- the sun below the horizon, and said.
  await page
    .getByTestId("timeline")
    .getByRole("radio", { name: /Mar 1, 2026/ })
    .click();
  await expect.poll(async () => (await state(page)).shown, poll).toBe(A1);
  await expect.poll(async () => (await state(page)).clock, poll).toBe("2026-03-01T08:15:00Z");
  await expect.poll(async () => (await state(page)).parts[A1] ?? [], poll).toContain(RELIT);
  await expect(page.getByTestId("relit-night")).toHaveText("Sun below horizon", {
    timeout: 60_000,
  });
  await shot(page, "app-at-capture-mar-night");

  // As captured: nothing relit, nothing said, the clock given back; the control says why.
  await page.getByRole("switch", { name: "As captured" }).click();
  await expect.poll(async () => (await state(page)).parts[A1] ?? [], poll).not.toContain(RELIT);
  await expect(badge).toHaveCount(0);
  await expect(page.getByTestId("views-sun-as-captured")).toHaveText("Off while As captured is on");
  await expect(radio("Off")).toHaveAttribute("aria-checked", "true");
  await expect(radio("Now")).toBeDisabled();
  expect((await state(page)).animating).toBe(true);
  expect(await clockLag(page)).toBeLessThan(120);
  await shot(page, "app-as-captured");

  // Off again: At capture comes back, held at March's time.
  await page.getByRole("switch", { name: "As captured" }).click();
  await expect.poll(async () => (await state(page)).clock, poll).toBe("2026-03-01T08:15:00Z");
  await expect.poll(async () => (await state(page)).parts[A1] ?? [], poll).toContain(RELIT);
  await expect(radio("At capture")).toHaveAttribute("aria-checked", "true");

  // The sun off: no part, no badge, the clock running at today again.
  await radio("Off").click();
  await expect
    .poll(async () => Object.values((await state(page)).parts).flat(), poll)
    .not.toContain(RELIT);
  await expect(badge).toHaveCount(0);
  expect((await state(page)).animating).toBe(true);
  expect(await clockLag(page)).toBeLessThan(120);
  await railUnchanged(page);
});

/** What PlayCanvas, drawing `assetId`, says of its sunlight hook (state/sunReport.ts). */
const hook = (
  page: Page,
  assetId: string,
): Promise<{ installed: boolean; frames: number } | null> =>
  page.evaluate(async (id) => {
    const link = "/src/state/sunReport.ts";
    const module = (await import(/* @vite-ignore */ link)) as {
      scanSunReading(id: string): { installed: boolean; frames: number } | null;
    };
    return module.scanSunReading(id);
  }, assetId);

const installed = async (page: Page, assetId: string) =>
  (await hook(page, assetId))?.installed ?? false;

test("under playcanvas, the sun relights the shown date by PlayCanvas's hook, says so, and At capture holds the clock", async ({
  page,
}) => {
  test.setTimeout(420_000);
  await arrive(page, "playcanvas");
  await expect
    .poll(() => page.evaluate(`window.__twin.scanRendererStatus.active`), poll)
    .toBe(true);
  const sun = page.getByTestId("views-sun");
  await expect(sun).toBeVisible();
  const radio = (name: string) => sun.getByRole("radio", { name });
  for (const name of ["Off", "Now", "At capture"]) await expect(radio(name)).toBeEnabled();
  await expect(page.getByTestId("views-sun-needs")).toHaveCount(0);
  // The shared notice is about Per cell alone now.
  await expect(page.getByTestId("views-needs-cesium")).not.toContainText("Sunlight");
  expect(await installed(page, A2)).toBe(false);
  await expect(page.getByTestId("relit-badge")).toHaveCount(0);

  // Now: PlayCanvas's hook in, and the badge over June's capture.
  await radio("Now").click();
  await expect.poll(() => installed(page, A2), poll).toBe(true);
  const badge = page.getByTestId("relit-badge");
  await expect(badge).toContainText(
    /Simulated lighting: the sun \(now\) over shadows baked in at capture, .*Jun.*1, 2026.* UTC/,
    { timeout: 60_000 },
  );
  // CesiumJS's own part is not on its hidden tileset.
  expect((await state(page)).parts[A2] ?? []).not.toContain(RELIT);
  await shot(page, "app-playcanvas-now");

  // At capture: the clock stands at June's capture time.
  await radio("At capture").click();
  await expect.poll(async () => (await state(page)).clock, poll).toBe("2026-06-01T18:30:00Z");
  expect((await state(page)).animating).toBe(false);
  await expect(badge).toContainText("Simulated lighting: the sun (at capture)");
  await shot(page, "app-playcanvas-at-capture-jun");

  // As captured: the hook out, nothing said, the clock given back.
  await page.getByRole("switch", { name: "As captured" }).click();
  await expect.poll(() => installed(page, A2), poll).toBe(false);
  await expect(badge).toHaveCount(0);
  await expect(page.getByTestId("views-sun-as-captured")).toHaveText("Off while As captured is on");
  expect((await state(page)).animating).toBe(true);
  expect(await clockLag(page)).toBeLessThan(120);
  await page.getByRole("switch", { name: "As captured" }).click();
  await expect.poll(() => installed(page, A2), poll).toBe(true);
  await expect.poll(async () => (await state(page)).clock, poll).toBe("2026-06-01T18:30:00Z");

  // Off: the hook out, the clock running at today again.
  await radio("Off").click();
  await expect.poll(() => installed(page, A2), poll).toBe(false);
  await expect(badge).toHaveCount(0);
  expect((await state(page)).animating).toBe(true);
  expect(await clockLag(page)).toBeLessThan(120);
  await railUnchanged(page);
});

test("under spark, the Sun says it needs CesiumJS or PlayCanvas and switches to PlayCanvas in one click", async ({
  page,
}) => {
  test.setTimeout(360_000);
  await arrive(page, "spark");
  const sun = page.getByTestId("views-sun");
  await expect(sun).toBeVisible();
  const radio = (name: string) => sun.getByRole("radio", { name });
  for (const name of ["Off", "Now", "At capture"]) await expect(radio(name)).toBeDisabled();
  await expect(radio("Off")).toHaveAttribute("aria-checked", "true");
  const needs = page.getByTestId("views-sun-needs");
  await expect(needs).toContainText(
    "Needs the CesiumJS or PlayCanvas renderer (Settings › Advanced › Splat renderer)",
  );
  await expect(page.getByTestId("relit-badge")).toHaveCount(0);
  await shot(page, "app-spark-notice");
  await railUnchanged(page);

  await needs.getByRole("button", { name: "Draw with PlayCanvas" }).click();
  await expect(needs).toHaveCount(0);
  await expect
    .poll(() => page.evaluate(`window.__twin.scanRendererStatus.kind`), poll)
    .toBe("playcanvas");
  await expect(radio("Now")).toBeEnabled();
  await radio("Now").click();
  await expect.poll(() => installed(page, A2), poll).toBe(true);
  await expect(page.getByTestId("relit-badge")).toContainText("Simulated lighting: the sun (now)", {
    timeout: 60_000,
  });
  await shot(page, "app-spark-switched-now");
  await railUnchanged(page);
});

/** Chooses the splat renderer as Settings › Advanced does (state/settings.ts). */
const chooseRenderer = (page: Page, renderer: string): Promise<void> =>
  page.evaluate(async (kind) => {
    const link = "/src/state/settings.ts";
    const module = (await import(/* @vite-ignore */ link)) as {
      useSettings: { getState(): { set(patch: { splatRenderer: string }): void } };
    };
    module.useSettings.getState().set({ splatRenderer: kind });
  }, renderer);

test("with the sun on, switching CesiumJS to PlayCanvas and back keeps the relight on whichever draws", async ({
  page,
}) => {
  test.setTimeout(420_000);
  await arrive(page, "cesium");
  const sun = page.getByTestId("views-sun");
  await expect(sun).toBeVisible();
  await sun.getByRole("radio", { name: "Now" }).click();
  await expect.poll(async () => (await state(page)).parts[A2] ?? [], poll).toContain(RELIT);
  const badge = page.getByTestId("relit-badge");
  await expect(badge).toContainText("Simulated lighting: the sun (now)", { timeout: 60_000 });

  // PlayCanvas: its hook in, CesiumJS's part out of the hidden tileset, the badge still said.
  await chooseRenderer(page, "playcanvas");
  await expect
    .poll(() => page.evaluate(`window.__twin.scanRendererStatus.active`), poll)
    .toBe(true);
  await expect.poll(() => installed(page, A2), poll).toBe(true);
  await expect.poll(async () => (await state(page)).parts[A2] ?? [], poll).not.toContain(RELIT);
  await expect(badge).toContainText("Simulated lighting: the sun (now)", { timeout: 60_000 });
  await expect(sun.getByRole("radio", { name: "Now" })).toHaveAttribute("aria-checked", "true");
  await shot(page, "app-switch-playcanvas");

  // Back to CesiumJS: its part again, PlayCanvas's session gone with its hook.
  await chooseRenderer(page, "cesium");
  await expect
    .poll(() => page.evaluate(`window.__twin.scanRendererStatus.active`), poll)
    .toBe(false);
  await expect.poll(() => installed(page, A2), poll).toBe(false);
  await expect.poll(async () => (await state(page)).parts[A2] ?? [], poll).toContain(RELIT);
  await expect(badge).toContainText("Simulated lighting: the sun (now)", { timeout: 60_000 });
  await shot(page, "app-switch-cesium-again");
  await railUnchanged(page);
});
