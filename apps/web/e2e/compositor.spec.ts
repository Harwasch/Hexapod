/**
 * Splat view › Per cell in the app itself (ADR 0017 "Per-cell view rules"): the synthetic yard's
 * two dates (as compareDates.spec.ts builds them) drawn as one under a rule, under CesiumJS and
 * under PlayCanvas, the app's default; and under Spark the control disabled, with the reason
 * and the one-click switch.
 *
 * Asserted, through `window.__twin` and the page: off by default; a rule draws both dates,
 * the later one's primitive drawn by the frame anchor's (one sort), the capture-date control
 * reads "As of", the key names each date with its coverage (neither has a confidence grid, so
 * both are flagged as tile bounds), and a date's own analytic views are held; turning Compare ›
 * Dates on ends the rule and the control says why, choosing a rule again ends the comparison;
 * One date draws the date chosen alone; the tool rail is as it was. Under PlayCanvas: CesiumJS's
 * copies stay hidden while PlayCanvas streams the other date into its one app, its tiles ruled
 * by their own source (what the renderer reads out: scanView/scanRule.ts), the key the same, At
 * capture off while both dates are drawn; One date lets the other date go. That each cell's
 * pixels come from one date is compositorScene.spec.ts's, by pixel counts in a bare engine.
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { Page } from "@playwright/test";

import { expect, mockApi, test } from "./fixtures";

const TILES = resolve(process.cwd(), "../../data/tiles");
const EVIDENCE = process.env.W5B_EVIDENCE_DIR;
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
  slug: "yard-per-cell",
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
  /** Whether the later date's primitive is drawn by the frame anchor's (a rule's companion). */
  drawnByAnchor: boolean;
  /** The anchor's primitive's companions, by asset id where they are a date. */
  companions: (string | null)[];
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
      ...(() => {
        const assets = twin.sites.loadedAssets();
        const of = (id) => assets.find((a) => a.assetId === id)?.tileset;
        const anchor = of("${A1}")?.gaussianSplatPrimitive;
        const later = of("${A2}")?.gaussianSplatPrimitive;
        const byTileset = new Map(assets.map((a) => [a.tileset, a.assetId]));
        return {
          drawnByAnchor: Boolean(anchor) && later?.drawnBy === anchor,
          companions: (anchor?.companions ?? []).map((t) => byTileset.get(t) ?? null),
        };
      })(),
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

async function railUnchanged(page: Page): Promise<void> {
  await expect(page.getByRole("toolbar", { name: "Tools" }).getByRole("button")).toHaveText([
    ...(process.env.VITE_ENABLE_LAND_EXPLORATION === "true" ? ["Land"] : []),
    "Layers",
    "Measure",
    "Add",
    "Settings",
  ]);
}

async function shot(page: Page, name: string): Promise<void> {
  await page.evaluate(`window.__twin.scene.requestRender()`);
  await page.waitForTimeout(2_000);
  const path = EVIDENCE ? `${EVIDENCE}/${name}.png` : test.info().outputPath(`${name}.png`);
  await page.screenshot({ path });
  await test.info().attach(name, { path, contentType: "image/png" });
}

const poll = { timeout: 120_000, intervals: [250, 500, 1000] };

test("under cesium, Per cell draws the site's dates as one, and gives way to Compare › Dates", async ({
  page,
}) => {
  test.setTimeout(420_000);
  await arrive(page, "cesium");
  await expect.poll(async () => (await state(page)).shown, poll).toBe(A2);
  const rule = page.getByTestId("views-rule").getByLabel("Per cell");
  await expect(rule).toBeVisible({ timeout: 60_000 });
  await expect(rule).toHaveValue("");
  await expect(rule).toBeEnabled();
  await expect(page.getByTestId("timeline")).not.toContainText("As of");

  // Latest: both dates drawn, the later one by the frame anchor's primitive, in its sort.
  await rule.selectOption("latest");
  await expect(rule).toHaveValue("latest");
  await expect.poll(() => drawnAs(page), poll).toEqual({ [A1]: [true, NONE], [A2]: [true, NONE] });
  await expect.poll(async () => (await state(page)).drawnByAnchor, poll).toBe(true);
  expect((await state(page)).companions).toContain(A2);
  // The date chosen is what the rule is as of.
  await expect(page.getByTestId("timeline")).toContainText("As of");
  await expect(page.getByTestId("timeline-epochs")).toHaveAttribute("aria-label", "As of");
  // The key: each date, flagged as tile bounds (neither carries a confidence grid).
  const legend = page.getByTestId("rule-legend");
  await expect(legend).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("rule-legend-date")).toHaveCount(2, { timeout: 60_000 });
  await expect(page.getByTestId("rule-legend-date").first()).toContainText(
    /Mar 1, 2026: \d+ cells \(tile bounds, overstates what it saw\)/,
    { timeout: 60_000 },
  );
  await expect(page.getByTestId("rule-legend-date").nth(1)).toContainText(/Jun 1, 2026: \d+ cells/);
  // A date's own Confidence and Data age wait for One date.
  await expect(page.getByTestId("views-rule-holds")).toBeVisible();
  await expect(
    page.getByTestId("views-modes").getByRole("radio", { name: "Confidence" }),
  ).toBeDisabled();
  await shot(page, "per-cell-latest");

  // As of March: June leaves.
  await page
    .getByTestId("timeline")
    .getByRole("radio", { name: /Mar 1, 2026/ })
    .click();
  await expect.poll(() => drawnAs(page), poll).toEqual({ [A1]: [true, NONE], [A2]: [false, NONE] });
  await expect(rule).toHaveValue("latest");
  await page
    .getByTestId("timeline")
    .getByRole("radio", { name: /Jun 1, 2026/ })
    .click();
  await expect.poll(() => drawnAs(page), poll).toEqual({ [A1]: [true, NONE], [A2]: [true, NONE] });

  // Data age and Highlight invented are the same draw, other colours.
  await rule.selectOption("data-age");
  await expect(rule).toHaveValue("data-age");
  await expect(page.getByTestId("rule-legend-date").first()).toContainText("months");
  await shot(page, "per-cell-data-age");

  // Compare › Dates ends the rule, and the control says why.
  await page.getByTestId("tool-layers").click();
  await page.getByTestId("layers-mode").getByRole("radio", { name: "Compare" }).click();
  const dates = page.getByTestId("compare-dates");
  await expect(dates).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("compare-dates-ends-rule")).toBeVisible();
  await dates.getByRole("switch", { name: "Swipe between dates" }).click();
  await expect.poll(() => drawnAs(page), poll).toEqual({ [A1]: [true, LEFT], [A2]: [true, RIGHT] });
  await page.getByTestId("layers-panel").getByRole("button", { name: "Close panel" }).click();
  await expect(rule).toHaveValue("");
  await expect(page.getByTestId("views-rule-ended")).toContainText(
    "Per-cell dates ended: Compare › Dates was turned on",
  );
  // Said once: the "ends Compare › Dates" note gives way to why the rule ended.
  await expect(page.getByTestId("views-rule-compare")).toHaveCount(0);
  await expect(page.getByTestId("timeline")).not.toContainText("As of");
  await shot(page, "per-cell-ended-by-compare");

  // And a rule ends the comparison.
  await rule.selectOption("ground-truth");
  await expect.poll(async () => (await state(page)).compare, poll).toBeNull();
  await expect.poll(() => drawnAs(page), poll).toEqual({ [A1]: [true, NONE], [A2]: [true, NONE] });
  await expect(page.getByTestId("split-slider")).toHaveCount(0);

  // One date: the date chosen alone, drawn by its own primitive.
  await rule.selectOption("");
  await expect.poll(() => drawnAs(page), poll).toEqual({ [A1]: [false, NONE], [A2]: [true, NONE] });
  await expect.poll(async () => (await state(page)).drawnByAnchor, poll).toBe(false);
  await expect(page.getByTestId("rule-legend")).toHaveCount(0);
  await shot(page, "per-cell-off");
  await railUnchanged(page);
});

/** What PlayCanvas, drawing asset `id`, reads out of the rule it draws (scanView/scanRule.ts). */
const ruleReading = (page: Page, id: string) =>
  page.evaluate(async (asset) => {
    const link = "/src/cesium/scanView/scanRule.ts";
    const module = (await import(/* @vite-ignore */ link)) as {
      scanRuleReading(id: string): {
        on: boolean;
        sources: Record<number, number>;
        dates: { dates: number; tiles: number };
      } | null;
    };
    return module.scanRuleReading(asset);
  }, id);

test("under playcanvas, Per cell draws the site's dates in PlayCanvas's one sort", async ({
  page,
}) => {
  test.setTimeout(420_000);
  await arrive(page, "playcanvas");
  await expect.poll(async () => (await state(page)).shown, poll).toBe(A2);
  const rule = page.getByTestId("views-rule").getByLabel("Per cell");
  await expect(rule).toBeVisible({ timeout: 60_000 });
  await expect(rule).toBeEnabled();
  await expect(page.getByTestId("views-needs-cesium")).toHaveCount(0);

  await rule.selectOption("latest");
  await expect(rule).toHaveValue("latest");
  // CesiumJS draws neither date: PlayCanvas, drawing June, streams March beside it.
  await expect
    .poll(() => drawnAs(page), poll)
    .toEqual({ [A1]: [false, NONE], [A2]: [false, NONE] });
  await expect
    .poll(async () => {
      const r = await ruleReading(page, A2);
      return r && { on: r.on, dates: r.dates.dates, drawn: r.dates.tiles > 0 };
    }, poll)
    .toEqual({ on: true, dates: 1, drawn: true });
  // Each tile ruled by its own date: June's own (source 1) and March's (0), none unnamed.
  const reading = await ruleReading(page, A2);
  expect(reading?.sources[0]).toBeGreaterThan(0);
  expect(reading?.sources[1]).toBeGreaterThan(0);
  expect(reading?.sources[255]).toBeUndefined();
  await expect(page.getByTestId("timeline")).toContainText("As of");
  await expect(page.getByTestId("rule-legend-date")).toHaveCount(2, { timeout: 60_000 });
  await expect(page.getByTestId("rule-legend-date").first()).toContainText(
    /Mar 1, 2026: \d+ cells \(tile bounds, overstates what it saw\)/,
    { timeout: 60_000 },
  );
  await expect(page.getByTestId("views-rule-holds")).toBeVisible();
  // At capture is off while more than one date is drawn: one clock is not each date's.
  await expect(
    page.getByTestId("views-sun").getByRole("radio", { name: "At capture" }),
  ).toBeDisabled();
  await shot(page, "per-cell-playcanvas-latest");

  await rule.selectOption("data-age");
  await expect(page.getByTestId("rule-legend-date").first()).toContainText("months");
  await shot(page, "per-cell-playcanvas-data-age");

  // One date: March is let go; June is drawn alone again.
  await rule.selectOption("");
  await expect
    .poll(async () => {
      const r = await ruleReading(page, A2);
      return r && { on: r.on, dates: r.dates.dates };
    }, poll)
    .toEqual({ on: false, dates: 0 });
  await expect(page.getByTestId("rule-legend")).toHaveCount(0);
  await expect(
    page.getByTestId("views-sun").getByRole("radio", { name: "At capture" }),
  ).toBeEnabled();
  await railUnchanged(page);
});

test("under spark, Per cell says it needs CesiumJS and switches to it in one click", async ({
  page,
}) => {
  test.setTimeout(360_000);
  await arrive(page, "spark");
  const rule = page.getByTestId("views-rule").getByLabel("Per cell");
  await expect(rule).toBeVisible({ timeout: 60_000 });
  await expect(rule).toBeDisabled();
  const notice = page.getByTestId("views-needs-cesium");
  await expect(notice).toContainText(
    "Per cell: Needs the CesiumJS renderer (Settings › Advanced › Splat renderer)",
  );
  await shot(page, "per-cell-spark-notice");
  await railUnchanged(page);
  await notice.getByRole("button", { name: "Draw with CesiumJS" }).click();
  await expect(notice).toHaveCount(0);
  await expect(rule).toBeEnabled();
  await rule.selectOption("latest");
  await expect.poll(() => drawnAs(page), poll).toEqual({ [A1]: [true, NONE], [A2]: [true, NONE] });
  await shot(page, "per-cell-spark-switched");
  await railUnchanged(page);
});
