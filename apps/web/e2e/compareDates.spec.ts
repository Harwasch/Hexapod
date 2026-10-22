/**
 * Compare › Dates in the app itself: two dated captures of one site, one each side of the
 * divider, under CesiumJS; and under PlayCanvas the control disabled, with the reason and the
 * one-click switch.
 *
 * E1 is the synthetic yard as a drone flew it on Mar 1 (the capture that made the site's
 * frame); E2 is a phone video of Jun 1, whose tileset is the yard's `tileset.json?epoch=2`
 * rewritten in the route to the committed synthetic tree's tile under the yard's root transform
 * (as epochSlider.spec.ts does), so the two dates look different on screen. Asserted, through
 * `window.__twin`: both dates drawn, E1 split LEFT and E2 RIGHT, one model matrix; the divider
 * moves the scene's split; turning the comparison off draws the one date shown before, whole.
 * And by pixels: left of the divider looks like E1 alone, right of it like E2 alone, and moving
 * the divider moves where E2 begins. The tool rail is as it was.
 *
 * Headless GL is SwiftShader, and splats are sorted afresh each view: halves are compared by
 * which single date they are nearer to, never against a stored image.
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { Page } from "@playwright/test";

import { expect, mockApi, test } from "./fixtures";

const TILES = resolve(process.cwd(), "../../data/tiles");
const EVIDENCE = process.env.W5A_EVIDENCE_DIR;
const YARD = "/fixture-tiles/synthetic-yard/splat/tileset.json";
const SITE_ID = "99999999-9999-4999-8999-999999999999";
const E1 = "99999999-9999-4999-8999-000000000001";
const E2 = "99999999-9999-4999-8999-000000000002";
const A1 = "99999999-9999-4999-8999-0000000000a1";
const A2 = "99999999-9999-4999-8999-0000000000a2";
/** Cesium's SplitDirection: LEFT -1, NONE 0, RIGHT 1. */
const LEFT = -1;
const NONE = 0;
const RIGHT = 1;

/** The yard's root transform's origin (synthetic_yard.py); the yard is 32 x 24 m east/north. */
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

/** The capture's own ground, as the pipeline measures it: cells about the yard, at 0 m. */
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
  slug: "yard-dates",
  name: "Yard, two dates",
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
  epochs: [
    epoch(E1, 1, "2026-03-01T15:00:00Z", "drone", A1),
    epoch(E2, 2, "2026-06-01T15:00:00Z", "phone-video", A2),
  ],
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
  latestObservedAt: "2026-06-01T15:00:00Z",
  quality: null,
  createdAt: site.createdAt,
  updatedAt: site.updatedAt,
};

/** E2's tileset: the yard's own root transform around the synthetic tree's one tile. */
function secondDate(): string {
  const yard = JSON.parse(
    readFileSync(resolve(TILES, "synthetic-yard/splat/tileset.json"), "utf8"),
  ) as { root: { transform: number[] } };
  const tree = JSON.parse(
    readFileSync(resolve(TILES, "synthetic-tree/splat/tileset.json"), "utf8"),
  ) as {
    root: {
      content: { uri: string };
      extras: { viewCones: { uri: string }; collision: { uri: string } };
    } & Record<string, unknown>;
  } & Record<string, unknown>;
  const base = "/fixture-tiles/synthetic-tree/splat/";
  const extras = tree.root.extras;
  return JSON.stringify({
    ...tree,
    root: {
      ...tree.root,
      transform: yard.root.transform,
      content: { uri: base + tree.root.content.uri },
      extras: {
        ...extras,
        viewCones: { ...extras.viewCones, uri: base + extras.viewCones.uri },
        collision: { ...extras.collision, uri: base + extras.collision.uri },
      },
    },
  });
}

async function route(page: Page): Promise<void> {
  await mockApi(page);
  await page.route("**/api/v1/sites", (r) =>
    r.request().method() === "GET" ? r.fulfill({ status: 200, json: [summary] }) : r.fallback(),
  );
  await page.route(`**/api/v1/sites/${SITE_ID}`, (r) => r.fulfill({ status: 200, json: site }));
  const second = secondDate();
  await page.route("**/fixture-tiles/**", (r) => {
    const url = new URL(r.request().url());
    const relative = url.pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return r.abort();
    if (relative === "synthetic-yard/splat/tileset.json" && url.searchParams.get("epoch") === "2")
      return r.fulfill({ status: 200, contentType: "application/json", body: second });
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
  compare: { left: string; right: string } | null;
  splitPosition: number;
  loaded: { id: string; shown: boolean; split: number; m: number[] }[];
}

const state = (page: Page): Promise<SceneState> =>
  page.evaluate(`(() => {
    const twin = window.__twin;
    twin.scene.requestRender();
    return {
      shown: twin.sites.shownAsset()?.id ?? null,
      compare: twin.sites.compareDates(),
      splitPosition: twin.scene.splitPosition,
      loaded: twin.sites.loadedAssets().map((a) => ({
        id: a.assetId,
        shown: a.shown,
        split: a.tileset.splitDirection,
        m: Array.from({ length: 16 }, (_, i) => a.tileset.modelMatrix[i]),
      })),
    };
  })()`);

/** Each loaded date's [shown, split], by asset id. */
async function drawnAs(page: Page): Promise<Record<string, [boolean, number]>> {
  const s = await state(page);
  return Object.fromEntries(s.loaded.map((a) => [a.id, [a.shown, a.split]]));
}

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

/** A screenshot once the scene has drawn a few frames at rest; saved as evidence when named. */
async function settledShot(page: Page, name?: string): Promise<Buffer> {
  await page.evaluate(`window.__twin.scene.requestRender()`);
  await page.waitForTimeout(3_000);
  await page.evaluate(`window.__twin.scene.requestRender()`);
  await page.waitForTimeout(1_000);
  if (!name) return page.screenshot();
  const path = EVIDENCE ? `${EVIDENCE}/${name}.png` : test.info().outputPath(`${name}.png`);
  const shot = await page.screenshot({ path });
  await test.info().attach(name, { path, contentType: "image/png" });
  return shot;
}

/**
 * The share of pixels that differ (summed RGB difference over 48) between two screenshots,
 * within the band of columns [x0, x1) and rows [0.08, 0.55) of the view, as fractions of it:
 * under the top bar and above the HUD strip, where the camera's arrival puts both dates.
 */
async function differ(page: Page, a: Buffer, b: Buffer, x0: number, x1: number): Promise<number> {
  return page.evaluate(
    async ([first, second, from, to]) => {
      const read = async (data: string) => {
        const image = new Image();
        image.src = `data:image/png;base64,${data}`;
        await image.decode();
        const canvas = document.createElement("canvas");
        canvas.width = image.width;
        canvas.height = image.height;
        const context = canvas.getContext("2d");
        if (!context) return null;
        context.drawImage(image, 0, 0);
        const x = Math.floor(image.width * from);
        const w = Math.floor(image.width * to) - x;
        const y = Math.floor(image.height * 0.08);
        const h = Math.floor(image.height * 0.47);
        return context.getImageData(x, y, w, h).data;
      };
      const [p, q] = [await read(first), await read(second)];
      if (!p || p.length !== q?.length) return 1;
      let count = 0;
      for (let i = 0; i < p.length; i += 4) {
        const d =
          Math.abs((p[i] ?? 0) - (q[i] ?? 0)) +
          Math.abs((p[i + 1] ?? 0) - (q[i + 1] ?? 0)) +
          Math.abs((p[i + 2] ?? 0) - (q[i + 2] ?? 0));
        if (d > 48) count += 1;
      }
      return count / (p.length / 4);
    },
    [a.toString("base64"), b.toString("base64"), x0, x1] as const,
  );
}

/**
 * Which date's solids answer a pick through each point of a grid over the view (the collider,
 * as Measure and selection ask it), split by the side of the divider the point is on.
 */
const sidePicks = (
  page: Page,
): Promise<{ left: string[]; right: string[]; leftUnsided: string[] }> =>
  page.evaluate(`(() => {
    const twin = window.__twin;
    const canvas = twin.viewer.canvas;
    const divider = twin.scene.splitPosition * canvas.clientWidth;
    const byTileset = new Map(twin.sites.loadedAssets().map((a) => [a.tileset, a.assetId]));
    const out = { left: [], right: [], leftUnsided: [] };
    for (let i = 1; i < 20; i++) {
      for (let j = 1; j < 10; j++) {
        const x = (canvas.clientWidth * i) / 20;
        const y = (canvas.clientHeight * j) / 10;
        if (Math.abs(x - divider) < 4) continue;
        const ray = twin.viewer.camera.getPickRay({ x, y });
        const hit = ray && twin.collider.raycast(ray, undefined, { x, y });
        const id = hit ? byTileset.get(hit.tileset) : undefined;
        if (id) (x < divider ? out.left : out.right).push(id);
        // The same ray with no point on screen, as every pick was before: what it would say.
        const unsided = ray && twin.collider.raycast(ray);
        const was = unsided ? byTileset.get(unsided.tileset) : undefined;
        if (was && x < divider) out.leftUnsided.push(was);
      }
    }
    const unique = (list) => [...new Set(list)];
    return { left: unique(out.left), right: unique(out.right), leftUnsided: unique(out.leftUnsided) };
  })()`);

async function railUnchanged(page: Page): Promise<void> {
  await expect(page.getByRole("toolbar", { name: "Tools" }).getByRole("button")).toHaveText([
    ...(process.env.VITE_ENABLE_LAND_EXPLORATION === "true" ? ["Land"] : []),
    "Layers",
    "Measure",
    "Add",
    "Settings",
  ]);
}

/** Layers › Compare, open. */
async function openCompare(page: Page) {
  await page.getByTestId("tool-layers").click();
  await page.getByTestId("layers-mode").getByRole("radio", { name: "Compare" }).click();
  const dates = page.getByTestId("compare-dates");
  await expect(dates).toBeVisible({ timeout: 30_000 });
  return dates;
}

async function closePanel(page: Page) {
  await page.getByTestId("layers-panel").getByRole("button", { name: "Close panel" }).click();
  await expect(page.getByTestId("layers-panel")).toHaveCount(0);
}

const poll = { timeout: 120_000, intervals: [250, 500, 1000] };

test("under cesium, Compare › Dates draws two dates either side of the divider", async ({
  page,
}) => {
  test.setTimeout(420_000);
  await arrive(page, "cesium");
  const timeline = page.getByTestId("timeline");
  const mar = timeline.getByRole("radio", { name: /Mar 1, 2026/ });
  const jun = timeline.getByRole("radio", { name: /Jun 1, 2026/ });

  // Each date alone, from where the camera landed: what each half should look like.
  await expect.poll(async () => (await state(page)).shown, poll).toBe(A2);
  await expect
    .poll(async () => (await state(page)).loaded.map((a) => a.id).sort(), poll)
    .toEqual([A1, A2].sort());
  const juneAlone = await settledShot(page, "jun-alone");
  await mar.click();
  await expect.poll(() => drawnAs(page), poll).toEqual({ [A1]: [true, NONE], [A2]: [false, NONE] });
  const marchAlone = await settledShot(page, "mar-alone");
  await jun.click();
  await expect.poll(() => drawnAs(page), poll).toEqual({ [A1]: [false, NONE], [A2]: [true, NONE] });

  // Compare › Dates: the latest against the one before, and on.
  const dates = await openCompare(page);
  await expect(dates.getByLabel("Left of the divider")).toHaveValue(E1);
  await expect(dates.getByLabel("Right of the divider")).toHaveValue(E2);
  await expect(page.getByTestId("compare-dates-needs-cesium")).toHaveCount(0);
  const toggle = dates.getByRole("switch", { name: "Swipe between dates" });
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-checked", "true");
  await expect.poll(() => drawnAs(page), poll).toEqual({ [A1]: [true, LEFT], [A2]: [true, RIGHT] });
  const both = await state(page);
  expect(both.compare).toEqual({ left: E1, right: E2 });
  // The single date is kept for after; both are drawn with one model matrix.
  expect(both.shown).toBe(A2);
  const [m1, m2] = [A1, A2].map((id) => both.loaded.find((a) => a.id === id)?.m);
  expect(m1).toBeDefined();
  expect(m2).toEqual(m1);
  // The divider is drawn, and the scene splits where it is.
  const slider = page.getByTestId("split-slider");
  await expect(slider).toBeVisible();
  expect(both.splitPosition).toBeCloseTo(0.5, 5);
  // The capture-date control checks neither and says which side each is on.
  await expect(mar).toHaveAttribute("aria-checked", "false");
  await expect(jun).toHaveAttribute("aria-checked", "false");
  await expect(mar).toContainText("left of divider");
  await expect(jun).toContainText("right of divider");

  // It outlives the panel. By pixels: each half looks like its own date alone.
  await closePanel(page);
  await expect(slider).toBeVisible();
  const compared = await settledShot(page, "compare-at-half");
  const halves = {
    leftVsMar: await differ(page, compared, marchAlone, 0.08, 0.47),
    leftVsJun: await differ(page, compared, juneAlone, 0.08, 0.47),
    rightVsJun: await differ(page, compared, juneAlone, 0.53, 0.92),
    rightVsMar: await differ(page, compared, marchAlone, 0.53, 0.92),
    // How different the two dates are in each half: the check means nothing without it.
    leftDates: await differ(page, marchAlone, juneAlone, 0.08, 0.47),
    rightDates: await differ(page, marchAlone, juneAlone, 0.53, 0.92),
  };
  test
    .info()
    .annotations.push({ type: "pixels, divider at 0.5", description: JSON.stringify(halves) });
  expect(halves.leftDates).toBeGreaterThan(0.02);
  expect(halves.rightDates).toBeGreaterThan(0.02);
  expect(halves.leftVsMar).toBeLessThan(halves.leftVsJun / 2);
  expect(halves.rightVsJun).toBeLessThan(halves.rightVsMar / 2);

  // A pick at a point on screen (Measure, selection, the orbit pivot) answers from the date
  // drawn there: across a grid of points, never the other side's date.
  const picks = await sidePicks(page);
  test.info().annotations.push({ type: "picks by side", description: JSON.stringify(picks) });
  expect(picks.left).not.toContain(A2);
  expect(picks.right).not.toContain(A1);
  expect(picks.left).toContain(A1);
  expect(picks.right).toContain(A2);
  // The check bites: without the point on screen, June's solids answered left of the divider.
  expect(picks.leftUnsided).toContain(A2);

  // The divider moves the split: ten steps left, to 0.3, and June now begins there.
  await slider.focus();
  for (let i = 0; i < 10; i++) await page.keyboard.press("ArrowLeft");
  await expect.poll(async () => (await state(page)).splitPosition, poll).toBeCloseTo(0.3, 5);
  const moved = await settledShot(page, "compare-at-0.3");
  const band = {
    vsJun: await differ(page, moved, juneAlone, 0.33, 0.47),
    vsMar: await differ(page, moved, marchAlone, 0.33, 0.47),
    dates: await differ(page, marchAlone, juneAlone, 0.33, 0.47),
  };
  test.info().annotations.push({
    type: "pixels, 0.33-0.47, divider at 0.3",
    description: JSON.stringify(band),
  });
  expect(band.dates).toBeGreaterThan(0.02);
  expect(band.vsJun).toBeLessThan(band.vsMar / 2);

  // Off: the one date shown before, whole, and no divider.
  const again = await openCompare(page);
  await again.getByRole("switch", { name: "Swipe between dates" }).click();
  await expect.poll(() => drawnAs(page), poll).toEqual({ [A1]: [false, NONE], [A2]: [true, NONE] });
  expect((await state(page)).compare).toBeNull();
  await expect(page.getByTestId("split-slider")).toHaveCount(0);
  await expect(jun).toHaveAttribute("aria-checked", "true");
  await closePanel(page);
  await settledShot(page, "compare-off");
  await railUnchanged(page);
});

test("under playcanvas, Compare › Dates says it needs CesiumJS and switches to it in one click", async ({
  page,
}) => {
  test.setTimeout(360_000);
  await arrive(page, "playcanvas");
  const dates = await openCompare(page);
  const toggle = dates.getByRole("switch", { name: "Swipe between dates" });
  await expect(toggle).toBeDisabled();
  await expect(dates.getByLabel("Left of the divider")).toBeDisabled();
  const notice = page.getByTestId("compare-dates-needs-cesium");
  await expect(notice).toContainText(
    "Needs the CesiumJS renderer (Settings › Advanced › Splat renderer)",
  );
  await settledShot(page, "playcanvas-notice");
  await railUnchanged(page);

  // One click: CesiumJS draws the scan, and the comparison can be turned on.
  await notice.getByRole("button", { name: "Draw with CesiumJS" }).click();
  await expect(notice).toHaveCount(0);
  await expect
    .poll(() => page.evaluate(`window.__twin.scanRendererStatus.active`), poll)
    .toBe(false);
  await expect(toggle).toBeEnabled();
  await toggle.click();
  await expect.poll(() => drawnAs(page), poll).toEqual({ [A1]: [true, LEFT], [A2]: [true, RIGHT] });
  await closePanel(page);
  await settledShot(page, "playcanvas-switched-compare");
  await railUnchanged(page);
});
