/**
 * A splat scan seen from afar (cesium/farView.ts), in a real `CesiumSceneManager` under each
 * splat renderer: drawn up close (its site engaged), still drawn -- small -- from 400 m, where
 * the site is long disengaged and the scan used to vanish, and gone from 5 km. A dedicated
 * renderer draws near and far as one session: the overlay canvas from up close is the one
 * drawing from 400 m, so zooming out reloads nothing and never flashes.
 *
 * The scan is the synthetic LOD tree, some 8 m in radius, on a 960 x 600 view: about 33 px
 * across from 400 m (it is drawn from 10 px), under 3 px from 5 km (hidden under 6 px).
 * Headless GL is SwiftShader: what is checked is what is drawn and where, not how it looks.
 *
 * The `@webgpu` test holds PlayCanvas on WebGPU (docs/WEBGPU_TRIAL.md) to the same, in the
 * Playwright project with software WebGPU.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

import { webgpuAdapter } from "./webgpu";

const TILES = resolve(process.cwd(), "../../data/tiles");
/** Where `synthetic_tree.py` places the tree. */
const LONGITUDE = -82.6966;
const LATITUDE = 28.0389;

interface State {
  distanceM: number;
  tilesetShown: boolean;
  target: { key: string; far: boolean } | null;
  scan: {
    active: boolean;
    far: boolean;
    frames: number;
    error: string | null;
    api: string | null;
    budget: number;
  };
  overlays: number;
  marked: boolean;
}

async function open(page: Page): Promise<void> {
  await page.route("**/fixture-tiles/**", (route) => {
    const relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    const file = resolve(TILES, relative);
    // The tree has no PlayCanvas streamed package beside it (`sog/`): the probe finds none.
    if (!existsSync(file)) return route.fulfill({ status: 404, body: "" });
    return route.fulfill({
      status: 200,
      contentType: relative.endsWith(".json") ? "application/json" : "application/octet-stream",
      body: readFileSync(file),
    });
  });
  const html = `<!doctype html><html><head><meta charset="utf-8" /><style>
html, body { margin: 0; height: 100%; background: #10141a; }
#v, #v .cesium-widget, #v .cesium-widget > canvas:first-child { width: 100vw; height: 100vh; display: block; }
</style></head><body>
<div id="v" style="position:relative;width:100vw;height:100vh"></div>
<script type="module">
const h = await import("/src/dev/farViewHarness.ts");
window.__far = await h.startFarViewHarness({ container: document.getElementById("v"),
  tilesetUrl: new URL("/fixture-tiles/synthetic-tree-lod/tileset.json", location.href).toString(),
  longitude: ${String(LONGITUDE)}, latitude: ${String(LATITUDE)} });
</script></body></html>`;
  await page.route("**/__scan-far-view", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: html }),
  );
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.goto("/__scan-far-view");
  await page.waitForFunction(() => "__far" in window, undefined, { timeout: 180_000 });
}

const state = (page: Page): Promise<State> => page.evaluate(`window.__far.state()`);

/**
 * Pixels that are not background within `radius` CSS px of where the scan's centre is drawn.
 * The background is the patch's commonest colour: the scene's colour grade tints it.
 */
async function litNearScan(page: Page, radius = 60): Promise<number> {
  const centre: [number, number] | null = await page.evaluate(`window.__far.centre()`);
  if (!centre) return 0;
  const viewport = page.viewportSize() ?? { width: 960, height: 600 };
  const x = Math.max(0, Math.round(centre[0] - radius));
  const y = Math.max(0, Math.round(centre[1] - radius));
  const width = Math.min(viewport.width - x, radius * 2);
  const height = Math.min(viewport.height - y, radius * 2);
  if (width <= 0 || height <= 0) return 0;
  const png = (await page.screenshot({ clip: { x, y, width, height } })).toString("base64");
  return page.evaluate(async (data) => {
    const image = new Image();
    image.src = `data:image/png;base64,${data}`;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext("2d");
    if (!context) return 0;
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, image.width, image.height).data;
    const counts = new Map<number, number>();
    for (let i = 0; i < pixels.length; i += 4) {
      const key = ((pixels[i] ?? 0) << 16) | ((pixels[i + 1] ?? 0) << 8) | (pixels[i + 2] ?? 0);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    let background = 0;
    let most = 0;
    for (const [key, count] of counts) {
      if (count > most) [background, most] = [key, count];
    }
    let lit = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      const d =
        Math.abs((pixels[i] ?? 0) - ((background >> 16) & 255)) +
        Math.abs((pixels[i + 1] ?? 0) - ((background >> 8) & 255)) +
        Math.abs((pixels[i + 2] ?? 0) - (background & 255));
      if (d > 48) lit += 1;
    }
    return lit;
  }, png);
}

async function view(page: Page, rangeM: number, pitchDeg: number): Promise<void> {
  await page.evaluate(`window.__far.view(${String(rangeM)}, ${String(pitchDeg)})`);
}

/** Whether the scan is drawn, by whichever renderer draws it. */
function drawn(kind: string, s: State): boolean {
  return kind === "cesium" ? s.tilesetShown : s.scan.active && s.target !== null;
}

for (const kind of ["cesium", "playcanvas", "spark", "playcanvas-webgpu"] as const) {
  const webgpu = kind === "playcanvas-webgpu";
  const dedicated = kind !== "cesium";
  test(
    `${kind}: a scan is drawn small from 400 m and gone from 5 km`,
    { tag: webgpu ? "@webgpu" : [] },
    async ({ page }, testInfo) => {
      test.setTimeout(300_000);
      await page.setViewportSize({ width: 960, height: 600 });
      await open(page);
      if (webgpu) test.skip((await webgpuAdapter(page)) === null, "no WebGPU adapter");
      await page.evaluate(`window.__far.use(${JSON.stringify(kind)})`);
      const poll = { timeout: 90_000, intervals: [250, 500, 1000] };

      // Loaded from 30 km up: past the far view's limit, so not drawn at all.
      expect(drawn(kind, await state(page))).toBe(false);

      // Up close: the site engages, and the scan is drawn as before.
      await view(page, 40, 30);
      await expect
        .poll(async () => {
          const s = await state(page);
          return drawn(kind, s) && (!dedicated || (s.target?.far === false && s.scan.frames > 0));
        }, poll)
        .toBe(true);
      const near = await state(page);
      if (dedicated) expect(await page.evaluate(`window.__far.mark()`)).toBe(true);

      // 400 m off and 260 m up: disengaged (about 105 m up hands a scan this size back), but
      // drawn from afar -- by the same overlay session, with a quarter of its budget.
      await view(page, 400, 40);
      await expect
        .poll(async () => {
          const s = await state(page);
          return drawn(kind, s) && (!dedicated || (s.target?.far === true && s.scan.far));
        }, poll)
        .toBe(true);
      const far = await state(page);
      expect(far.distanceM).toBeGreaterThan(390);
      if (dedicated) {
        expect(far.scan.error).toBeNull();
        expect(far.overlays).toBe(1);
        expect(far.marked).toBe(true);
        expect(far.scan.budget).toBe(Math.round(near.scan.budget * 0.25));
        if (webgpu) expect(far.scan.api).toBe("webgpu");
      }
      // And on screen, where the scan is: a small patch of it.
      await expect.poll(() => litNearScan(page), poll).toBeGreaterThan(20);
      const litFar = await litNearScan(page);

      // 5 km off: a few pixels at most, so not drawn; a dedicated renderer's session ends.
      await view(page, 5000, 40);
      await expect.poll(async () => drawn(kind, await state(page)), poll).toBe(false);
      const gone = await state(page);
      expect(gone.distanceM).toBeGreaterThan(4900);
      if (dedicated) {
        expect(gone.target).toBeNull();
        await expect.poll(async () => (await state(page)).overlays, poll).toBe(0);
      }
      await expect.poll(() => litNearScan(page), poll).toBeLessThanOrEqual(2);
      const litGone = await litNearScan(page);

      // Back to 400 m: drawn again.
      await view(page, 400, 40);
      await expect.poll(async () => drawn(kind, await state(page)), poll).toBe(true);
      await expect.poll(() => litNearScan(page), poll).toBeGreaterThan(20);

      const result = { kind, near, far, litFar, gone, litGone };
      writeFileSync(testInfo.outputPath("far-view.json"), JSON.stringify(result, null, 1));
      console.info(JSON.stringify(result));
    },
  );
}
