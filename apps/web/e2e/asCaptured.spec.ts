/**
 * As captured in the app itself (state/asCaptured.ts, ADR 0019 "Living map, scoped"): one
 * switch under which only what was measured is drawn (plans stay, as annotations).
 *
 * The site is the synthetic yard with what a viewer can have drawn over its measured splats:
 * Today's inferred layer (the fixture mound), fill methods to compare (`variants/`, the hedge
 * method declaring a `supersedes` file that hides the big tree's measured splats while it is
 * shown, as variants.spec.ts builds it), the yard's forest rig (`rig.json`) with the wind
 * blowing, so the Simulated motion badge is up. Inferred is set to Highlight.
 *
 * What must hold, through `window.__twin` and the stores the app runs on:
 *
 * - turning it on (the switch in the Splat view control) leaves no inferred tileset drawn --
 *   by the scan's primitive as a companion (no rig: the scan stays incremental) or by its own
 *   (with the rig) -- and no superseded splat hidden;
 *   the Simulated motion badge goes because nothing animates (it is not told); the chip says
 *   so; the Inferred and Fill controls are disabled and say why;
 * - the viewer's own choices are untouched -- the Inferred style, the wind, the fill picked --
 *   and turning it off (the chip's button) draws all of it again;
 * - `?asCaptured=1` starts on: nothing inferred is drawn, and the wind moves nothing; the
 *   switch keeps the address in step;
 * - under PlayCanvas, the renderer that draws the scan's layers itself (scanView/scanLayers.ts)
 *   takes them off screen, by its own status and by pixels;
 * - on the demo ranch, nothing the simulated fleet did is drawn (machines, tracks, zones'
 *   progress), while the zones' outlines and names stay as the plan's annotations;
 * - the tool rail is as it was.
 *
 * Screenshots go to `AS_CAPTURED_EVIDENCE_DIR` when set. Headless GL is SwiftShader: pixels are
 * counted, never compared with a stored image.
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { Page } from "@playwright/test";

import { expect, mockApi, test } from "./fixtures";

const TILES = resolve(process.cwd(), "../../data/tiles");
const EVIDENCE = process.env.AS_CAPTURED_EVIDENCE_DIR;
const TILESET = "synthetic-yard/splat/tileset.json";
const SUPERSEDES = "synthetic-yard/variants/fill/hedge/supersedes.json";
const SITE_ID = "77777777-7777-4777-8777-777777777777";
const ASSET = "77777777-7777-4777-8777-0000000000a1";
/** The yard's big tree (instance 1): what the hedge method supersedes here. */
const TREE = 1;

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

const asset = {
  id: ASSET,
  siteId: SITE_ID,
  provider: "3d-tiles-url",
  name: "Gaussian splat",
  representation: "gaussian-splat",
  source: { type: "3d-tiles-url", url: `/fixture-tiles/${TILESET}` },
  footprint: null,
  observedAt: "2026-03-02T00:00:00Z",
  validFrom: null,
  validTo: null,
  resolution: null,
  crs: null,
  license: null,
  attribution: [],
  provenance: null,
  epochId: null,
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
    // The yard's forest rig (tools/captures/scene_plants.py): the wind moves its plants.
    rigUrl: "rig.json",
  },
  defaultVisible: true,
  createdAt: "2026-03-02T00:00:00Z",
  updatedAt: "2026-03-02T00:00:00Z",
};

const site = {
  id: SITE_ID,
  slug: "yard-as-captured",
  name: "Yard, as captured",
  description: null,
  boundary: { type: "Polygon", coordinates: [ring(40)] },
  centroid: { longitude: MID.lon, latitude: MID.lat, height: null },
  areaM2: 6_400,
  thumbnailUrl: null,
  metadata: {},
  attribution: [],
  license: null,
  assets: [asset],
  epochs: [],
  frame: null,
  cameraBookmarks: [],
  createdAt: asset.createdAt,
  updatedAt: asset.updatedAt,
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
  latestObservedAt: null,
  quality: null,
  createdAt: site.createdAt,
  updatedAt: site.updatedAt,
};

const read = (relative: string): Record<string, unknown> =>
  JSON.parse(readFileSync(resolve(TILES, relative), "utf-8")) as Record<string, unknown>;

/** A `supersedes.json` flagging every splat of instance `id` and below (variants.spec.ts). */
function supersedesOf(
  doc: { instances: { id: number; parent?: number | null }[]; tiles: Record<string, number[]> },
  id: number,
) {
  const wanted = new Set([id]);
  for (let grew = true; grew;) {
    grew = false;
    for (const i of doc.instances)
      if (i.parent != null && wanted.has(i.parent) && !wanted.has(i.id)) {
        wanted.add(i.id);
        grew = true;
      }
  }
  let superseded = 0;
  const tiles: Record<string, number[]> = {};
  for (const [checksum, runs] of Object.entries(doc.tiles)) {
    const out: number[] = [];
    for (let k = 0; k + 1 < runs.length; k += 2) {
      const flag = wanted.has(runs[k] ?? 0) ? 1 : 0;
      const count = runs[k + 1] ?? 0;
      if (flag) superseded += count;
      const last = out.length - 2;
      if (last >= 0 && out[last] === flag) out[last + 1] = (out[last + 1] ?? 0) + count;
      else out.push(flag, count);
    }
    tiles[checksum] = out;
  }
  return { superseded, tiles };
}

async function route(
  page: Page,
  inferredStyle: "show" | "highlight" = "highlight",
  { rig = true }: { rig?: boolean } = {},
) {
  await mockApi(page);
  await page.addInitScript((style) => {
    window.localStorage.setItem(
      "twin.settings.v1",
      JSON.stringify({ state: { inferredStyle: style, quality: "performance" }, version: 3 }),
    );
  }, inferredStyle);
  await page.route("**/api/v1/sites", (r) =>
    r.request().method() === "GET" ? r.fulfill({ status: 200, json: [summary] }) : r.fallback(),
  );
  // A Living Survey scan is CesiumJS's to draw under any renderer (its motion is CesiumJS's
  // shader): without the rig, PlayCanvas draws the yard and its layers.
  const served = rig
    ? site
    : {
        ...site,
        assets: [{ ...asset, renderConfig: { ...asset.renderConfig, rigUrl: undefined } }],
      };
  await page.route(`**/api/v1/sites/${SITE_ID}`, (r) => r.fulfill({ status: 200, json: served }));
  const variants = (read("synthetic-yard/variants/variants.json").variants ?? {}) as Record<
    string,
    Record<string, unknown>[]
  >;
  variants.fill = (variants.fill ?? []).map((entry) =>
    entry.name === "hedge"
      ? { ...entry, supersedes: "../variants/fill/hedge/supersedes.json" }
      : entry,
  );
  const mound = variants.fill.find((entry) => entry.name === "mound");
  const instancesDoc = read("synthetic-yard/instances/instances.json") as {
    instances: { id: number; parent?: number | null }[];
    tiles: Record<string, number[]>;
  };
  const skins = (read("synthetic-yard/skin/skin.json").skins ?? []) as unknown[];
  await page.route("**/fixture-tiles/**", (r) => {
    const relative = new URL(r.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return r.abort();
    if (relative === TILESET) {
      const tileset = read(TILESET) as { root: { extras?: Record<string, unknown> } };
      tileset.root.extras = {
        ...tileset.root.extras,
        instances: { uri: "../instances/instances.json", count: instancesDoc.instances.length },
        skin: { uri: "../skin/skin.json", count: skins.length },
        // Today's inferred layer: the mound method's, as a publish would carry one.
        inferredLayers: mound?.inferredLayers ?? [],
        variants,
        nativeLod: false,
      };
      return r.fulfill({ status: 200, json: tileset });
    }
    if (relative === SUPERSEDES)
      return r.fulfill({ status: 200, json: supersedesOf(instancesDoc, TREE) });
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

async function arrive(page: Page, query: string): Promise<void> {
  await page.setViewportSize({ width: 1280, height: 760 });
  await page.goto(`/?${query}`);
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

interface Drawn {
  /** Inferred tilesets in the scene, shown, and on the scan's primitive's companion list. */
  layers: number;
  shown: number;
  companions: number;
  /** Assets whose fill supersedes measured splats now (state/supersedes.ts). */
  superseding: string[];
  inferredStyle: string;
  wind: number;
  asCaptured: boolean;
  animating: boolean;
  fill: string | null;
}

/**
 * Runs `body` in the page with the app's own stores in scope: the dev server serves each module
 * at one URL, so importing it here is importing the instance the app uses. A string, so the
 * test's transpiler leaves the dynamic imports alone (fixtures.ts `inApp`).
 */
function inApp<T>(page: Page, body: string): Promise<T> {
  return page.evaluate(`(async () => {
    const { useSupersedes } = await import("/src/state/supersedes.ts");
    const { useSettings } = await import("/src/state/settings.ts");
    const { useLiving } = await import("/src/state/living.ts");
    const { useAsCaptured } = await import("/src/state/asCaptured.ts");
    const { useVariants } = await import("/src/state/variants.ts");
    ${body}
  })()`);
}

/** What the scene draws that was not measured, and the viewer's own choices. */
const drawn = (page: Page): Promise<Drawn> =>
  inApp(
    page,
    `
    const twin = window.__twin;
    twin.scene.requestRender();
    const layers = [];
    for (let i = 0; i < twin.scene.primitives.length; i += 1) {
      const p = twin.scene.primitives.get(i);
      const url = p?.resource?.url;
      if (typeof url === "string" && url.includes("/variants/fill/")) layers.push(p);
    }
    const scan = twin.sites.loadedAssets().find((a) => a.assetId === "${ASSET}")?.tileset;
    const primitive = scan?.gaussianSplatPrimitive;
    const host = primitive?.drawnBy ?? primitive;
    return {
      layers: layers.length,
      shown: layers.filter((l) => l.show).length,
      companions: (host?.companions ?? []).filter((t) => layers.includes(t)).length,
      superseding: Object.keys(useSupersedes.getState().active),
      inferredStyle: useSettings.getState().inferredStyle,
      wind: useLiving.getState().wind.strength,
      asCaptured: useAsCaptured.getState().on,
      animating: useLiving.getState().status.animating,
      fill: useVariants.getState().picks["${ASSET}"]?.fill ?? null,
    };`,
  );

const setWind = (page: Page, strength: number): Promise<void> =>
  inApp(page, `useLiving.getState().setWind({ strength: ${String(strength)}, bearingDeg: 250 });`);

const pickFill = (page: Page, name: string): Promise<void> =>
  inApp(page, `useVariants.getState().pick("${ASSET}", "fill", ${JSON.stringify(name)});`);

const setAsCaptured = (page: Page, on: boolean): Promise<void> =>
  inApp(page, `useAsCaptured.getState().set(${String(on)});`);

async function shot(page: Page, name: string): Promise<Buffer> {
  await page.evaluate(`window.__twin.scene.requestRender()`);
  await page.waitForTimeout(2_500);
  await page.evaluate(`window.__twin.scene.requestRender()`);
  await page.waitForTimeout(800);
  const path = EVIDENCE ? `${EVIDENCE}/${name}.png` : test.info().outputPath(`${name}.png`);
  const image = await page.screenshot({ path });
  await test.info().attach(name, { path, contentType: "image/png" });
  return image;
}

/** The share of pixels that differ (summed RGB over 48) in the view's middle band. */
async function differ(page: Page, a: Buffer, b: Buffer): Promise<number> {
  return page.evaluate(
    async ([first, second]) => {
      const load = async (data: string) => {
        const image = new Image();
        image.src = `data:image/png;base64,${data}`;
        await image.decode();
        const canvas = document.createElement("canvas");
        canvas.width = image.width;
        canvas.height = image.height;
        const context = canvas.getContext("2d");
        if (!context) return null;
        context.drawImage(image, 0, 0);
        return context.getImageData(
          0,
          Math.floor(image.height * 0.08),
          image.width,
          Math.floor(image.height * 0.6),
        ).data;
      };
      const [p, q] = [await load(first), await load(second)];
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
    [a.toString("base64"), b.toString("base64")] as const,
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

const poll = { timeout: 180_000, intervals: [250, 500, 1000] };

test("under cesium, a layer the scan's primitive draws leaves its companion list, and comes back", async ({
  page,
}) => {
  test.setTimeout(480_000);
  await route(page, "highlight", { rig: false });
  await arrive(page, "renderer=cesium");
  // Today's layer, then the hedge method's with its swap: drawn by the scan's own primitive.
  await expect
    .poll(async () => {
      const d = await drawn(page);
      return [d.shown, d.companions];
    }, poll)
    .toEqual([1, 1]);
  await pickFill(page, "hedge");
  await expect
    .poll(async () => {
      const d = await drawn(page);
      return [d.shown, d.companions, d.superseding];
    }, poll)
    .toEqual([1, 1, [ASSET]]);
  const before = await drawn(page);
  test.info().annotations.push({ type: "before", description: JSON.stringify(before) });
  const withFill = await shot(page, "cesium-companion-before");
  await page.getByTestId("views").getByRole("switch", { name: "As captured" }).click();
  // The address follows the switch, so a reload or a copied link shows the same.
  await expect.poll(() => new URL(page.url()).searchParams.get("asCaptured")).toBe("1");
  await expect
    .poll(async () => {
      const d = await drawn(page);
      return { shown: d.shown, companions: d.companions, superseding: d.superseding };
    }, poll)
    .toEqual({ shown: 0, companions: 0, superseding: [] });
  const on = await drawn(page);
  test.info().annotations.push({ type: "on", description: JSON.stringify(on) });
  expect(on).toMatchObject({ inferredStyle: "highlight", fill: "hedge", asCaptured: true });
  const asCaptured = await shot(page, "cesium-companion-as-captured");
  const changed = await differ(page, withFill, asCaptured);
  test.info().annotations.push({ type: "pixels changed", description: String(changed) });
  expect(changed).toBeGreaterThan(0.001);
  await page.getByTestId("as-captured-chip").getByRole("button", { name: "Turn off" }).click();
  await expect
    .poll(async () => {
      const d = await drawn(page);
      return [d.shown, d.companions, d.superseding];
    }, poll)
    .toEqual([1, 1, [ASSET]]);
  await railUnchanged(page);
});

test("under cesium with the rig in the wind, the badge goes because nothing moves, and all comes back", async ({
  page,
}) => {
  test.setTimeout(600_000);
  await route(page);
  await arrive(page, "renderer=cesium");
  const badge = page.getByTestId("simulated-badge");
  const chip = page.getByTestId("as-captured-chip");

  // Before: Today's inferred layer drawn, then the hedge method with its supersedes swap; the
  // wind blowing on the yard's plants, and the badge up.
  await expect.poll(async () => (await drawn(page)).shown, poll).toBe(1);
  await pickFill(page, "hedge");
  await expect
    .poll(async () => {
      const d = await drawn(page);
      return [d.shown, d.superseding];
    }, poll)
    .toEqual([1, [ASSET]]);
  await setWind(page, 0.3);
  await expect(badge).toBeVisible({ timeout: 180_000 });
  await expect(badge).toContainText("Simulated motion");
  const before = await drawn(page);
  test.info().annotations.push({ type: "before", description: JSON.stringify(before) });
  expect(before).toMatchObject({ inferredStyle: "highlight", asCaptured: false, fill: "hedge" });
  // A rigged scan's primitive leaves incremental mode, so the layer draws itself here (shown,
  // on no companion list); the companion path is the test before this one.
  expect(before).toMatchObject({ shown: 1, companions: 0 });
  expect(before.animating).toBe(true);
  const withFill = await shot(page, "cesium-before");
  await railUnchanged(page);

  // On, from the Splat view control.
  const views = page.getByTestId("views");
  const toggle = views.getByRole("switch", { name: "As captured" });
  await expect(toggle).toHaveAttribute("aria-checked", "false");
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-checked", "true");
  await expect(chip).toBeVisible();
  await expect(chip).toContainText(
    "As captured — only what was measured is drawn; plans stay as annotations",
  );
  await expect
    .poll(async () => {
      const d = await drawn(page);
      return { shown: d.shown, companions: d.companions, superseding: d.superseding };
    }, poll)
    .toEqual({ shown: 0, companions: 0, superseding: [] });
  // The badge goes because nothing animates: it is never told.
  await expect.poll(async () => (await drawn(page)).animating, poll).toBe(false);
  await expect(badge).toHaveCount(0);
  const on = await drawn(page);
  test.info().annotations.push({ type: "on", description: JSON.stringify(on) });
  // The viewer's own choices are untouched.
  expect(on).toMatchObject({ inferredStyle: "highlight", wind: 0.3, fill: "hedge" });
  // The controls of what it hides say why.
  const inferred = page.getByTestId("inferred-style");
  await expect(inferred).toContainText("Hidden while As captured is on");
  for (const radio of await inferred.getByRole("radio").all()) await expect(radio).toBeDisabled();
  const asCaptured = await shot(page, "cesium-as-captured");
  const changed = await differ(page, withFill, asCaptured);
  test.info().annotations.push({ type: "pixels changed", description: String(changed) });
  expect(changed).toBeGreaterThan(0.001);
  await railUnchanged(page);

  // Off, from the chip: all of it back, as it was.
  await chip.getByRole("button", { name: "Turn off" }).click();
  await expect(chip).toHaveCount(0);
  await expect(toggle).toHaveAttribute("aria-checked", "false");
  await expect
    .poll(async () => {
      const d = await drawn(page);
      return [d.shown, d.superseding, d.animating];
    }, poll)
    .toEqual([1, [ASSET], true]);
  await expect(badge).toBeVisible();
  const after = await drawn(page);
  expect(after).toMatchObject({ inferredStyle: "highlight", wind: 0.3, fill: "hedge" });
  await expect(inferred).not.toContainText("Hidden while As captured is on");
  await shot(page, "cesium-after");
  await railUnchanged(page);
});

test("?asCaptured=1 starts on: nothing inferred is drawn, and the wind does not move the yard", async ({
  page,
}) => {
  test.setTimeout(480_000);
  await route(page, "show");
  await arrive(page, "renderer=cesium&asCaptured=1");
  const chip = page.getByTestId("as-captured-chip");
  await expect(chip).toBeVisible();
  await setWind(page, 0.3);
  const toggle = page.getByTestId("views").getByRole("switch", { name: "As captured" });
  await expect(toggle).toHaveAttribute("aria-checked", "true");
  // The hedge method picked (the Fill row is held while on; a pick kept from before is this):
  // its layer is loaded, and its supersedes file is what would hide the tree.
  await expect.poll(() => page.evaluate(`window.__twin.sites.loadedAssets().length`), poll).toBe(1);
  await pickFill(page, "hedge");
  await expect.poll(async () => (await drawn(page)).layers, poll).toBe(1);
  await page.waitForTimeout(3_000);
  const start = await drawn(page);
  test.info().annotations.push({ type: "start", description: JSON.stringify(start) });
  expect(start).toMatchObject({
    layers: 1,
    shown: 0,
    companions: 0,
    superseding: [],
    asCaptured: true,
    fill: "hedge",
  });
  expect(start.animating).toBe(false);
  await expect(page.getByTestId("simulated-badge")).toHaveCount(0);
  await shot(page, "cesium-url-start");
  await railUnchanged(page);
  // And off: the layer the viewer's Show asks for, its swap, and the badge; the address says
  // so too, replaced in place (no history entry to step back into it).
  const depth = await page.evaluate(() => window.history.length);
  await chip.getByRole("button", { name: "Turn off" }).click();
  await expect.poll(() => new URL(page.url()).searchParams.get("asCaptured")).toBeNull();
  expect(new URL(page.url()).searchParams.get("renderer")).toBe("cesium");
  expect(await page.evaluate(() => window.history.length)).toBe(depth);
  await expect
    .poll(async () => {
      const d = await drawn(page);
      return [d.shown, d.superseding];
    }, poll)
    .toEqual([1, [ASSET]]);
  await expect(page.getByTestId("simulated-badge")).toBeVisible({ timeout: 180_000 });
});

test("under playcanvas, the renderer that draws the scan's layers takes them off screen", async ({
  page,
}) => {
  test.setTimeout(480_000);
  await route(page, "highlight", { rig: false });
  await arrive(page, "renderer=playcanvas");
  interface Layers {
    active: boolean;
    layers: { wanted: number; drawn: number; tiles: number; style: string } | null;
  }
  const layers = (): Promise<Layers> =>
    page.evaluate(`(() => {
      const s = window.__twin.scanRendererStatus;
      return { active: s.active, layers: s.layers };
    })()`);
  await expect
    .poll(async () => {
      const s = await layers();
      return s.active && (s.layers?.tiles ?? 0) > 0;
    }, poll)
    .toBe(true);
  const before = await layers();
  test.info().annotations.push({ type: "before", description: JSON.stringify(before) });
  expect(before.layers).toMatchObject({ wanted: 1, drawn: 1, style: "highlight" });
  const shown = await shot(page, "playcanvas-before");
  await setAsCaptured(page, true);
  await expect
    .poll(async () => (await layers()).layers, poll)
    .toMatchObject({ wanted: 0, drawn: 0, tiles: 0, style: "hide" });
  // CesiumJS's own copy stays hidden under it: nothing inferred is drawn twice or at all.
  expect((await drawn(page)).shown).toBe(0);
  const hidden = await shot(page, "playcanvas-as-captured");
  const changed = await differ(page, shown, hidden);
  test.info().annotations.push({ type: "pixels changed", description: String(changed) });
  expect(changed).toBeGreaterThan(0.001);
  await page.getByTestId("as-captured-chip").getByRole("button", { name: "Turn off" }).click();
  await expect
    .poll(async () => (await layers()).layers, poll)
    .toMatchObject({ wanted: 1, drawn: 1, style: "highlight" });
  await shot(page, "playcanvas-after");
  await railUnchanged(page);
});

test("on the demo ranch, nothing the simulated fleet did is drawn; the plan's zones stay", async ({
  app,
}) => {
  await app.getByTestId("onboarding-demo").click();
  await expect(app.getByTestId("project-card")).toBeVisible({ timeout: 30_000 });
  const markers = app.locator('[data-testid^="machine-marker-"]');
  const chips = app.locator('[data-testid^="zone-chip-"]');
  const progress = app.locator('[data-testid^="zone-chip-"] .mc-chip__short');
  await expect(markers).toHaveCount(6);
  const zoneChips = await chips.count();
  expect(zoneChips).toBeGreaterThan(0);
  await expect(progress).toHaveCount(zoneChips);
  /** Each track's show, and each zone outline: solid (treated by the fleet) or dashed. */
  const drawn = (): Promise<{ tracks: boolean[]; solidZones: number; zones: number }> =>
    app.evaluate(`(() => {
      const values = window.__twin.viewer.entities.values;
      const zones = values.filter((e) => String(e.id).startsWith("mission:zone:") && e.polyline);
      return {
        tracks: values.filter((e) => String(e.id).startsWith("mission:track")).map((e) => e.show),
        zones: zones.length,
        solidZones: zones.filter((e) => e.polyline.material.constructor.name !== "PolylineDashMaterialProperty").length,
      };
    })()`);
  const before = await drawn();
  test.info().annotations.push({ type: "ranch before", description: JSON.stringify(before) });
  expect(before.tracks.length).toBeGreaterThan(0);
  expect(before.tracks.every(Boolean)).toBe(true);
  expect(before.solidZones).toBeGreaterThan(0);
  // Ground polylines are batched and built over a few frames: shots wait for them.
  await shot(app, "ranch-before");
  await setAsCaptured(app, true);
  await expect(markers).toHaveCount(0);
  // Each zone keeps its outline and its name; its simulated status and progress go.
  await expect(chips).toHaveCount(zoneChips);
  await expect(progress).toHaveCount(0);
  const on = await drawn();
  test.info().annotations.push({ type: "ranch as captured", description: JSON.stringify(on) });
  expect(on.tracks.some(Boolean)).toBe(false);
  expect(on.zones).toBe(before.zones);
  expect(on.solidZones).toBe(0);
  const chip = app.getByTestId("as-captured-chip");
  await expect(chip).toContainText(
    "As captured — only what was measured is drawn; plans stay as annotations",
  );
  await shot(app, "ranch-as-captured");
  await chip.getByRole("button", { name: "Turn off" }).click();
  await expect(markers).toHaveCount(6);
  await expect(progress).toHaveCount(zoneChips);
  const after = await drawn();
  expect(after.tracks.every(Boolean)).toBe(true);
  expect(after.solidZones).toBe(before.solidZones);
  await railUnchanged(app);
});
