/**
 * The change drape in the app itself (features/change, ADR 0018): Compare › Dates offers the
 * change map between two dated captures of the yard once both measured a height surface; the
 * ground layer is drawn on the ground and outlined over the scan under CesiumJS, PlayCanvas and
 * Spark alike, vegetation apart behind its own switch; the legend keys each layer on its own
 * and says it is derived, uncalibrated, the least change each layer can show, and when a date
 * is not survey-grade; and the write token it is behind is asked for, not a blank drape.
 *
 * The `/change` answers are the API's own for the committed fixture (e2e/fixtures/changeYard.py
 * runs the API's analysis and response code over data/tiles/synthetic-yard-surfaces and
 * synthetic-yard-epochs, placed in the yard's frame): the ground layer finds the 1 m pond dug
 * and nothing else; vegetation finds the shed and the stack raised (the stack where e1 was
 * weak) and the spoil heap gone. changeYardCoarse.json is the same pair with the later date a
 * coarse ±5 m placement: nothing claimed under 15 m, and the legend says not survey-grade.
 *
 * Headless GL is SwiftShader: the drape is asserted through `window.__twin` (the ground
 * primitives it tags) and the overlay's SVG, and, by pixels, as each layer's colours appearing
 * on screen from above where they were not before.
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { Page, Route } from "@playwright/test";

import { expect, mockApi, test } from "./fixtures";

const TILES = resolve(process.cwd(), "../../data/tiles");
const EVIDENCE = process.env.W5C_EVIDENCE_DIR;
const YARD = "/fixture-tiles/synthetic-yard/splat/tileset.json";
const SITE_ID = "99999999-9999-4999-8999-999999999999";
const E1 = "99999999-9999-4999-8999-000000000001";
const E2 = "99999999-9999-4999-8999-000000000002";
const A1 = "99999999-9999-4999-8999-0000000000a1";
const A2 = "99999999-9999-4999-8999-0000000000a2";
const TOKEN = "pilot-token";

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

const CELLS = Array.from({ length: 16 }, (_, i) => {
  const r = 14 * Math.sqrt((i + 0.5) / 16);
  const a = i * 2.399963;
  return { lon: MID.lon + east(r * Math.cos(a)), lat: MID.lat + north(r * Math.sin(a)), height: 0 };
});

const footprint = { type: "MultiPolygon", coordinates: [[ring(40)]] };

const surfaceOf = (assetId: string) => ({
  status: "measured",
  assetId,
  url: `https://tiles.example/${assetId}/surface.bin`,
  cellM: 1,
  level: 2,
  policy: "surface-v1",
  calibration: "uncalibrated",
  cells: 1280,
  surfacedCells: 1272,
  medianSigmaM: 0.02,
  originHeightM: 0,
  groundWindowM: 5,
});

function epoch(
  id: string,
  ordinal: number,
  observedAt: string,
  platform: string,
  asset: string,
  surface: boolean,
  coarse = false,
) {
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
    // The re-scan surveyed (RTK), as the API's own test registers it; or a phone date aligned
    // coarsely at ±5 m.
    alignment:
      ordinal === 1
        ? null
        : coarse
          ? { verdict: "coarse", uncertaintyM: 5, label: "Phone video, aligned visually, ±5 m" }
          : { verdict: "survey", uncertaintyM: 0.062, label: "RTK drone, survey tier, ±0.062 m" },
    quality: null,
    // A map is answered only for dates with a confidence grid too (else 409
    // confidence-not-measured): each of these has one, the phone date capped by ±0.062 m.
    confidence: {
      status: "measured",
      assetId: asset,
      url: `https://tiles.example/${asset}/confidence.bin`,
      cellM: 1,
      level: 2,
      policy: "weak-v1",
      weakShare: 0.02,
      weakCells: 23,
      scoredCells: 1272,
      unseenCells: 0,
      reasons: {},
      alignmentCap: {
        uncertaintyM: ordinal === 1 ? 0 : 0.062,
        factor: 1,
        priorEpochs: ordinal - 1,
        source: ordinal === 1 ? "site-frame" : "epoch_alignment.json",
      },
    },
    surface: surface ? surfaceOf(asset) : null,
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

function site(surfaces: boolean, coarse = false) {
  return {
    id: SITE_ID,
    slug: "yard-change",
    name: "Yard, what changed",
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
      epoch(E1, 1, "2026-03-01T15:00:00Z", "drone", A1, surfaces),
      epoch(
        E2,
        2,
        "2026-06-01T15:00:00Z",
        coarse ? "phone-video" : "rtk-drone",
        A2,
        surfaces,
        coarse,
      ),
    ],
    frame: null,
    cameraBookmarks: [],
    createdAt: "2026-03-02T00:00:00Z",
    updatedAt: "2026-06-02T00:00:00Z",
  };
}

const summary = {
  id: SITE_ID,
  slug: "yard-change",
  name: "Yard, what changed",
  description: null,
  centroid: { longitude: MID.lon, latitude: MID.lat, height: null },
  areaM2: 6_400,
  thumbnailUrl: null,
  representations: ["gaussian-splat"],
  latestObservedAt: "2026-06-01T15:00:00Z",
  quality: null,
  createdAt: "2026-03-02T00:00:00Z",
  updatedAt: "2026-06-02T00:00:00Z",
};

// ---- The change map: the API's own answers for the committed fixture ----------------------

const FIXTURES = resolve(process.cwd(), "e2e/fixtures");
const answerFor = (name: "changeYard" | "changeYardCoarse"): unknown =>
  JSON.parse(readFileSync(resolve(FIXTURES, `${name}.json`), "utf8"));

// ---- The app ---------------------------------------------------------------------------------

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

interface ChangeCalls {
  asked: { from: string | null; to: string | null; auth: string | null }[];
}

async function route(
  page: Page,
  options: { surfaces?: boolean; token?: boolean; coarse?: boolean } = {},
): Promise<ChangeCalls> {
  const calls: ChangeCalls = { asked: [] };
  const map = answerFor(options.coarse ? "changeYardCoarse" : "changeYard");
  await mockApi(page);
  await page.route("**/api/v1/sites", (r) =>
    r.request().method() === "GET" ? r.fulfill({ status: 200, json: [summary] }) : r.fallback(),
  );
  await page.route(`**/api/v1/sites/${SITE_ID}`, (r) =>
    r.fulfill({ status: 200, json: site(options.surfaces ?? true, options.coarse) }),
  );
  await page.route(`**/api/v1/sites/${SITE_ID}/change?*`, (r: Route) => {
    const url = new URL(r.request().url());
    const auth = r.request().headers().authorization ?? null;
    calls.asked.push({ from: url.searchParams.get("from"), to: url.searchParams.get("to"), auth });
    if (options.token && auth !== `Bearer ${TOKEN}`)
      return r.fulfill({
        status: 401,
        json: { type: "about:blank", title: "Unauthorized", status: 401, detail: "write token" },
      });
    return r.fulfill({ status: 200, json: map });
  });
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
  return calls;
}

async function arrive(page: Page, renderer: string): Promise<void> {
  await page.setViewportSize({ width: 1280, height: 760 });
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

/** Straight down over the yard from `heightM`, north up: the whole of the drape in view. */
async function lookDown(page: Page, heightM: number): Promise<void> {
  const [a, e2] = [6_378_137, 6.69437999014e-3];
  const [phi, lambda] = [(MID.lat * Math.PI) / 180, (MID.lon * Math.PI) / 180];
  const n = a / Math.sqrt(1 - e2 * Math.sin(phi) ** 2);
  const [x, y, z] = [
    (n + heightM) * Math.cos(phi) * Math.cos(lambda),
    (n + heightM) * Math.cos(phi) * Math.sin(lambda),
    (n * (1 - e2) + heightM) * Math.sin(phi),
  ];
  await page.evaluate(`(() => {
    const twin = window.__twin;
    const camera = twin.viewer.camera;
    const Cartesian3 = camera.position.constructor;
    camera.setView({
      destination: new Cartesian3(${x}, ${y}, ${z}),
      orientation: { heading: 0, pitch: -Math.PI / 2, roll: 0 },
    });
    twin.scene.requestRender();
  })()`);
}

interface DrapeTag {
  layer: string;
  cls: string;
  lowConfidence: boolean;
  polygons: number;
  hatched: boolean;
  ready: boolean;
  show: boolean;
}

/** The ground primitives the drape drew, as it tagged them. */
const drawn = (page: Page): Promise<DrapeTag[]> =>
  page.evaluate(`(() => {
    const ground = window.__twin.scene.groundPrimitives;
    const out = [];
    for (let i = 0; i < ground.length; i++) {
      const p = ground.get(i);
      if (p && p.changeDrape) out.push({ ...p.changeDrape, ready: Boolean(p.ready), show: p.show });
    }
    return out;
  })()`);

/** What is drawn of one layer: each primitive's class and confidence, in drawing order. */
const layerDrawn = async (page: Page, layer: string) =>
  (await drawn(page))
    .filter((d) => d.layer === layer)
    .map(({ cls, lowConfidence, hatched }) => ({ cls, lowConfidence, hatched }));

/** One layer's outlines over the scan: each class's path, and whether it draws anything. */
const overlay = (
  page: Page,
  layer: string,
): Promise<{ cls: string; low: string; drawn: boolean }[]> =>
  page.evaluate(`[...document.querySelectorAll('svg[data-change-overlay="${layer}"] path')].map((p) => ({
    cls: p.getAttribute("data-class"),
    low: p.getAttribute("data-low-confidence"),
    drawn: (p.getAttribute("d") ?? "").length > 0,
  }))`);

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

interface Shares {
  /** The ground's lowered blue (#3a9cf0): the pond. */
  pond: number;
  /** Vegetation's raised yellow (#f2c94c): the shed and the stack. */
  grown: number;
}

/**
 * The share of the view's pixels near a layer's colour (summed RGB distance under 90), over
 * the view clear of the HUD and the legend.
 */
async function drapePixels(page: Page, shot: Buffer): Promise<Shares> {
  return page.evaluate(async (data) => {
    const image = new Image();
    image.src = `data:image/png;base64,${data}`;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext("2d");
    if (!context) return { pond: 0, grown: 0 };
    context.drawImage(image, 0, 0);
    const [x, y] = [Math.floor(image.width * 0.1), Math.floor(image.height * 0.1)];
    const [w, h] = [Math.floor(image.width * 0.64), Math.floor(image.height * 0.65)];
    const px = context.getImageData(x, y, w, h).data;
    const near = (r: number, g: number, b: number, i: number) =>
      Math.abs((px[i] ?? 0) - r) + Math.abs((px[i + 1] ?? 0) - g) + Math.abs((px[i + 2] ?? 0) - b) <
      90;
    let pond = 0;
    let grown = 0;
    for (let i = 0; i < px.length; i += 4) {
      const [r, g, b] = [px[i] ?? 0, px[i + 1] ?? 0, px[i + 2] ?? 0];
      if (b > r + 60 && near(80, 150, 220, i)) pond += 1;
      if (r > b + 80 && g > b + 60 && near(230, 200, 90, i)) grown += 1;
    }
    const total = px.length / 4;
    return { pond: pond / total, grown: grown / total };
  }, shot.toString("base64"));
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

/** The ground: the pond lowered, and the not-comparable margins. */
const GROUND = [
  { cls: "lowered", lowConfidence: false, hatched: false },
  { cls: "insufficient", lowConfidence: false, hatched: false },
];
/** Vegetation: the shed and the stack (weak, hatched) raised, the spoil heap lowered. */
const VEGETATION = [
  { cls: "raised", lowConfidence: false, hatched: false },
  { cls: "raised", lowConfidence: true, hatched: true },
  { cls: "lowered", lowConfidence: false, hatched: false },
  { cls: "insufficient", lowConfidence: false, hatched: false },
];
const outlined = (layer: typeof GROUND) =>
  layer.map(({ cls, lowConfidence }) => ({ cls, low: String(lowConfidence), drawn: true }));

for (const renderer of ["cesium", "playcanvas", "spark"] as const) {
  test(`under ${renderer}, Compare › Dates drapes the ground's change, and vegetation apart`, async ({
    page,
  }) => {
    test.setTimeout(420_000);
    const calls = await route(page);
    await arrive(page, renderer);
    // The drone date: the yard, its ground covered in splats.
    await page
      .getByTestId("timeline")
      .getByRole("radio", { name: /Mar 1, 2026/ })
      .click();
    await expect
      .poll(() => page.evaluate(`window.__twin.sites.shownAsset()?.id ?? null`), poll)
      .toBe(A1);
    await lookDown(page, 70);
    const before = await settledShot(page, `${renderer}-before`);

    const dates = await openCompare(page);
    const change = dates.getByRole("switch", { name: "Change map" });
    await expect(change).toBeEnabled();
    await change.click();
    await expect(change).toHaveAttribute("aria-checked", "true");
    await expect.poll(() => calls.asked.at(-1), poll).toMatchObject({ from: E1, to: E2 });

    // The ground only, on the ground and over the scan, whatever draws the splats.
    await expect.poll(() => layerDrawn(page, "ground"), poll).toEqual(GROUND);
    await expect
      .poll(async () => (await drawn(page)).every((d) => d.ready && d.show), poll)
      .toBe(true);
    expect(await layerDrawn(page, "vegetation")).toEqual([]);
    await expect.poll(() => overlay(page, "ground"), poll).toEqual(outlined(GROUND));
    await expect(page.locator('svg[data-change-overlay="vegetation"]')).toHaveCount(0);

    // The ground's key: the pond, its floor, uncalibrated; nothing of vegetation in it.
    const legend = page.getByTestId("change-drape");
    await expect(legend).toBeVisible();
    await expect(page.getByTestId("change-drape-label")).toHaveText(
      "Derived from height surfaces — not observed · changes under ±0.49 m cannot be seen between these dates",
    );
    await expect(page.getByTestId("change-row-raised")).toContainText("0 cells");
    await expect(page.getByTestId("change-row-lowered")).toContainText("48 cells");
    await expect(page.getByTestId("change-row-insufficient")).toContainText("148 cells");
    await expect(page.getByTestId("change-row-unchanged")).toContainText(
      "1,084 cells · counted, not drawn",
    );
    await expect(legend).toContainText("Uncalibrated: the 3σ rule and the 5 m ground window");
    // Both dates surveyed: nothing said of survey grade.
    await expect(page.getByTestId("change-drape-placement")).toHaveCount(0);
    await expect(page.getByTestId("change-vegetation")).toHaveCount(0);

    // It outlives the panel; from above, the pond's blue where there was none.
    await closePanel(page);
    await expect(legend).toBeVisible();
    await lookDown(page, 70);
    const ground = await settledShot(page, `${renderer}-drape`);

    // Vegetation, apart, behind its own switch: its own key, its own floor, its own colours.
    const vegetation = legend.getByRole("switch", { name: /^Vegetation: canopy height change/ });
    await expect(vegetation).toHaveAttribute("aria-checked", "false");
    await vegetation.click();
    await expect.poll(() => layerDrawn(page, "vegetation"), poll).toEqual(VEGETATION);
    await expect.poll(() => overlay(page, "vegetation"), poll).toEqual(outlined(VEGETATION));
    await expect(page.getByTestId("change-vegetation-label")).toContainText(
      "changes under ±1.1 m cannot be seen between these dates",
    );
    await expect(page.getByTestId("change-vegetation-row-raised")).toContainText("47 cells");
    await expect(page.getByTestId("change-vegetation-row-raised")).toContainText(
      "23 low confidence, hatched",
    );
    await expect(page.getByTestId("change-vegetation-row-lowered")).toContainText("24 cells");
    await expect(page.getByTestId("change-vegetation")).toContainText(
      "±0.25 m of wind sway per date",
    );
    // The ground's key is untouched by it.
    await expect(page.getByTestId("change-row-lowered")).toContainText("48 cells");
    await lookDown(page, 70);
    const both = await settledShot(page, `${renderer}-vegetation`);

    const off = await drapePixels(page, before);
    const on = await drapePixels(page, ground);
    const withVegetation = await drapePixels(page, both);
    test.info().annotations.push({
      type: "drape pixels",
      description: JSON.stringify({ off, on, withVegetation }),
    });
    expect(on.pond).toBeGreaterThan(off.pond + 0.001);
    // Vegetation's yellow appears only with its switch on.
    expect(on.grown).toBeLessThan(off.grown + 0.0005);
    expect(withVegetation.grown).toBeGreaterThan(on.grown + 0.001);

    // Vegetation off: its drawing goes, the ground's stays.
    await vegetation.click();
    await expect.poll(() => layerDrawn(page, "vegetation"), poll).toEqual([]);
    await expect(page.locator('svg[data-change-overlay="vegetation"]')).toHaveCount(0);
    expect(await layerDrawn(page, "ground")).toEqual(GROUND);

    // Hide: off the ground, and the legend gone.
    await legend.getByRole("button", { name: "Hide change map" }).click();
    await expect(legend).toHaveCount(0);
    await expect.poll(async () => (await drawn(page)).length, poll).toBe(0);
    await expect(page.locator("svg[data-change-overlay]")).toHaveCount(0);
    await railUnchanged(page);
  });
}

test("a coarse date: nothing claimed under 15 m, and the legend says not survey-grade", async ({
  page,
}) => {
  test.setTimeout(300_000);
  await route(page, { coarse: true });
  await arrive(page, "cesium");
  const dates = await openCompare(page);
  await dates.getByRole("switch", { name: "Change map" }).click();
  await expect(page.getByTestId("change-drape-label")).toHaveText(
    "Derived from height surfaces — not observed · changes under ±16 m cannot be seen between these dates",
  );
  await expect(page.getByTestId("change-drape-placement")).toHaveText(
    "Not survey-grade: Mar 1, 2026 survey (±0.03 m) · Jun 1, 2026 coarse (±5 m)",
  );
  await expect(page.getByTestId("change-row-lowered")).toContainText("0 cells");
  await settledShot(page, "coarse");
  await railUnchanged(page);
});

test("the change map asks for the pilot's write token, then draws once it is given", async ({
  page,
}) => {
  test.setTimeout(360_000);
  const calls = await route(page, { token: true });
  await arrive(page, "cesium");
  const dates = await openCompare(page);
  await dates.getByRole("switch", { name: "Change map" }).click();
  const error = page.getByTestId("change-drape-error");
  await expect(error).toContainText("needs the pilot's write token");
  await expect(page.getByTestId("compare-change-refused")).toContainText(
    "needs the pilot's write token",
  );
  expect(calls.asked.at(-1)?.auth).toBeNull();
  expect(await drawn(page)).toEqual([]);
  await settledShot(page, "needs-token");

  await error.getByRole("button", { name: "Open Settings" }).click();
  await page.getByTestId("settings-write-token").fill(TOKEN);
  await expect.poll(() => calls.asked.at(-1)?.auth ?? null, poll).toBe(`Bearer ${TOKEN}`);
  await expect(page.getByTestId("change-drape-label")).toBeVisible({ timeout: 30_000 });
  await expect(error).toHaveCount(0);
  await expect.poll(() => layerDrawn(page, "ground"), poll).toEqual(GROUND);
  await railUnchanged(page);
});

test("dates with no height surface: not offered, and Compare says why", async ({ page }) => {
  test.setTimeout(300_000);
  const calls = await route(page, { surfaces: false });
  await arrive(page, "cesium");
  const dates = await openCompare(page);
  await expect(dates.getByRole("switch", { name: "Change map" })).toBeDisabled();
  await expect(page.getByTestId("compare-change-not-offered")).toHaveText(
    "Not offered: neither Mar 1, 2026 nor Jun 1, 2026 has a height surface measured",
  );
  await settledShot(page, "not-offered");
  expect(calls.asked).toEqual([]);
  await railUnchanged(page);
});
