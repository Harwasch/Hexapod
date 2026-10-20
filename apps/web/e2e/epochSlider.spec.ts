/**
 * The capture-date control on a site with two dated captures (epochs), in the app itself,
 * under each splat renderer: PlayCanvas (the default), Spark and CesiumJS.
 *
 * E1 is the synthetic yard as a drone flew it on Mar 1 (the capture that made the site's
 * frame); E2 is a phone video of Jun 1. So that the two dates look different on screen without
 * a new fixture, E2's tileset is the yard's `tileset.json?epoch=2` rewritten in the route: the
 * yard's root transform (one site frame, which the shared placement requires) around the
 * committed synthetic tree's tile. Asserted: the control lists both, the latest checked; the
 * scene shows the date chosen and hides the other, keeping both tilesets; both are drawn with
 * one model matrix; the screen changes with the date; and the tool rail is as it was.
 *
 * Headless GL is SwiftShader: what is drawn is checked by pixels changing, not by looks.
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { Page } from "@playwright/test";

import { expect, mockApi, test } from "./fixtures";

const TILES = resolve(process.cwd(), "../../data/tiles");
const YARD = "/fixture-tiles/synthetic-yard/splat/tileset.json";
const SITE_ID = "99999999-9999-4999-8999-999999999999";
const E1 = "99999999-9999-4999-8999-000000000001";
const E2 = "99999999-9999-4999-8999-000000000002";
const A1 = "99999999-9999-4999-8999-0000000000a1";
const A2 = "99999999-9999-4999-8999-0000000000a2";

/** The yard's root transform's origin (synthetic_yard.py); the yard is 32 x 24 m east/north. */
const LON = -123.88;
const LAT = 46.1335;
const east = (m: number) => m / (111_320 * Math.cos((LAT * Math.PI) / 180));
const north = (m: number) => m / 111_320;
/** The yard's middle. */
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
  // Listed oldest first, both default-visible: the rule from before epochs would show E1.
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

/**
 * E2's tileset: the yard's own root transform around the synthetic tree's one tile, every URI
 * absolute so it resolves to the tree, not beside the yard.
 */
function secondDate(): string {
  const yard = JSON.parse(
    readFileSync(resolve(TILES, "synthetic-yard/splat/tileset.json"), "utf8"),
  ) as {
    root: { transform: number[] };
  };
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

/** Routes the catalog (this site only) and the fixture tiles; counts each tile request. */
async function route(page: Page): Promise<Map<string, number>> {
  const requests = new Map<string, number>();
  await mockApi(page);
  await page.route("**/api/v1/sites", (r) =>
    r.request().method() === "GET" ? r.fulfill({ status: 200, json: [summary] }) : r.fallback(),
  );
  await page.route(`**/api/v1/sites/${SITE_ID}`, (r) => r.fulfill({ status: 200, json: site }));
  const second = secondDate();
  await page.route("**/fixture-tiles/**", (r) => {
    const url = new URL(r.request().url());
    const key = url.pathname + url.search;
    requests.set(key, (requests.get(key) ?? 0) + 1);
    const relative = url.pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return r.abort();
    if (relative === "synthetic-yard/splat/tileset.json" && url.searchParams.get("epoch") === "2")
      return r.fulfill({ status: 200, contentType: "application/json", body: second });
    const file = resolve(TILES, relative);
    // Neither scan has a PlayCanvas streamed package beside it (`sog/`): the probe finds none.
    if (!existsSync(file)) return r.fulfill({ status: 404, body: "" });
    return r.fulfill({
      status: 200,
      contentType: relative.endsWith(".json") ? "application/json" : "application/octet-stream",
      body: readFileSync(file),
    });
  });
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (r) => r.abort());
  return requests;
}

interface SceneState {
  shown: string | null;
  target: string | null;
  scan: { active: boolean; frames: number };
  loaded: { id: string; shown: boolean; marked: boolean; m: number[] }[];
}

const state = (page: Page): Promise<SceneState> =>
  page.evaluate(`(() => {
    const twin = window.__twin;
    twin.scene.requestRender();
    return {
      shown: twin.sites.shownAsset()?.id ?? null,
      target: twin.sites.scanTarget()?.assetId ?? null,
      scan: { active: twin.scanRendererStatus.active, frames: twin.scanRendererStatus.frames },
      loaded: twin.sites.loadedAssets().map((a) => ({
        id: a.assetId,
        shown: a.shown,
        marked: a.tileset.__epochMark === a.assetId,
        m: Array.from({ length: 16 }, (_, i) => a.tileset.modelMatrix[i]),
      })),
    };
  })()`);

/** The share of pixels in the middle of the view that differ between two screenshots. */
async function changed(page: Page, a: Buffer, b: Buffer): Promise<number> {
  return page.evaluate(
    async ([first, second]) => {
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
        // The middle half of the view, clear of the HUD.
        const w = Math.floor(image.width / 2);
        const h = Math.floor(image.height / 2);
        return context.getImageData(Math.floor(w / 2), Math.floor(h / 2), w, h).data;
      };
      const [x, y] = [await read(first), await read(second)];
      if (x?.length !== y?.length || !x || !y) return 0;
      let differ = 0;
      for (let i = 0; i < x.length; i += 4) {
        const d =
          Math.abs((x[i] ?? 0) - (y[i] ?? 0)) +
          Math.abs((x[i + 1] ?? 0) - (y[i + 1] ?? 0)) +
          Math.abs((x[i + 2] ?? 0) - (y[i + 2] ?? 0));
        if (d > 48) differ += 1;
      }
      return differ / (x.length / 4);
    },
    [a.toString("base64"), b.toString("base64")] as const,
  );
}

/** Waits for the date `assetId` to be what is drawn, by whichever renderer draws it. */
async function drawn(page: Page, renderer: string, assetId: string, other: string) {
  const poll = { timeout: 120_000, intervals: [250, 500, 1000] };
  await expect.poll(async () => (await state(page)).shown, poll).toBe(assetId);
  if (renderer === "cesium") {
    await expect
      .poll(async () => {
        const s = await state(page);
        const of = (id: string) => s.loaded.find((a) => a.id === id)?.shown;
        return [of(assetId), of(other)];
      }, poll)
      .toEqual([true, false]);
  } else {
    await expect
      .poll(async () => {
        const s = await state(page);
        return s.target === assetId && s.scan.active && s.loaded.every((a) => !a.shown);
      }, poll)
      .toBe(true);
  }
}

/** A screenshot once the scene has drawn a few frames at rest. */
async function settledShot(page: Page): Promise<Buffer> {
  await page.evaluate(`window.__twin.scene.requestRender()`);
  await page.waitForTimeout(3_000);
  await page.evaluate(`window.__twin.scene.requestRender()`);
  await page.waitForTimeout(1_000);
  return page.screenshot();
}

for (const renderer of ["playcanvas", "spark", "cesium"] as const) {
  test(`under ${renderer}, the capture date switches between a site's two epochs`, async ({
    page,
  }) => {
    test.setTimeout(300_000);
    await page.setViewportSize({ width: 960, height: 600 });
    const requests = await route(page);
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

    // The control: both dates, by date, platform and alignment, the latest checked.
    const timeline = page.getByTestId("timeline");
    await expect(timeline).toBeVisible();
    const radios = timeline.getByRole("radio");
    await expect(radios).toHaveCount(2);
    const mar = timeline.getByRole("radio", {
      name: /Mar 1, 2026.*Drone.*Defines the site frame/,
    });
    const jun = timeline.getByRole("radio", {
      name: /Jun 1, 2026.*Phone video.*Not aligned yet/,
    });
    await expect(jun).toHaveAttribute("aria-checked", "true");
    await expect(mar).toHaveAttribute("aria-checked", "false");

    // The latest is drawn, the frame's own date loaded beside it (hidden), both with one
    // model matrix -- the anchor's, rested on its ground.
    await drawn(page, renderer, A2, A1);
    await expect
      .poll(async () => (await state(page)).loaded.map((a) => a.id).sort(), { timeout: 60_000 })
      .toEqual([A1, A2].sort());
    const placed = await state(page);
    const [m1, m2] = [A1, A2].map((id) => placed.loaded.find((a) => a.id === id)?.m);
    expect(m1).toBeDefined();
    expect(m2).toEqual(m1);
    // Mark both tilesets: a switch that reloaded one would come back unmarked.
    await page.evaluate(`window.__twin.sites.loadedAssets().forEach((a) => {
      a.tileset.__epochMark = a.assetId;
    })`);
    const june = await settledShot(page);
    const junePath = test.info().outputPath(`epoch-slider-${renderer}-jun.png`);
    await page.screenshot({ path: junePath });
    await test
      .info()
      .attach(`Jun 1 under ${renderer}`, { path: junePath, contentType: "image/png" });

    // Mar: the drone flight is drawn, the phone video hidden.
    await mar.click();
    await drawn(page, renderer, A1, A2);
    await expect(mar).toHaveAttribute("aria-checked", "true");
    await expect(jun).toHaveAttribute("aria-checked", "false");
    const march = await settledShot(page);
    const marchPath = test.info().outputPath(`epoch-slider-${renderer}-mar.png`);
    await page.screenshot({ path: marchPath });
    await test
      .info()
      .attach(`Mar 1 under ${renderer}`, { path: marchPath, contentType: "image/png" });
    // The screen changed with the date: the yard is not the tree.
    const share = await changed(page, june, march);
    test.info().annotations.push({ type: "pixels changed", description: share.toFixed(4) });
    expect(share).toBeGreaterThan(0.01);

    // Back to Jun: the same two tilesets, kept, not loaded again.
    await jun.click();
    await drawn(page, renderer, A2, A1);
    const back = await state(page);
    expect(back.loaded.map((a) => a.id).sort()).toEqual([A1, A2].sort());
    expect(back.loaded.every((a) => a.marked)).toBe(true);
    expect(requests.get(YARD)).toBeGreaterThan(0);
    expect(requests.get(`${YARD}?epoch=2`)).toBeGreaterThan(0);
    test.info().annotations.push({
      type: "tileset.json requests",
      description: JSON.stringify({ e1: requests.get(YARD), e2: requests.get(`${YARD}?epoch=2`) }),
    });

    // No new tool-rail button.
    await expect(page.getByRole("toolbar", { name: "Tools" }).getByRole("button")).toHaveText([
      ...(process.env.VITE_ENABLE_LAND_EXPLORATION === "true" ? ["Land"] : []),
      "Layers",
      "Measure",
      "Add",
      "Settings",
    ]);
  });
}
