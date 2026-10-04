/**
 * What the splat overlay (cesium/scanView) costs when nothing moves. The app renders the globe
 * on demand (request-render mode), so a still view used to cost nothing -- except the overlay,
 * which redrew the same frame every display frame. Here the yard fixture is drawn by each
 * dedicated renderer with the globe in request-render mode, as in the app, and the overlay's
 * own counters (`window.__twinStats`, scanView/stats.ts) are read across five seconds of rest:
 * once the scan has loaded, nothing should be drawn at all.
 *
 * The second test is the overlay's resolution: under the performance preset the globe renders
 * one device pixel per CSS pixel, and on a 2x display the overlay's canvas must too.
 *
 * The `@webgpu` variants hold PlayCanvas on WebGPU (docs/WEBGPU_TRIAL.md) to the same rule, in
 * the Playwright project with software WebGPU: there it sorts on the GPU in the frame that
 * draws, so what asks for frames (no sort results; the one confirming frame per batch of tiles,
 * playcanvasBackend.ts) differs, and the rest must be as still.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

import { webgpuAdapter } from "./webgpu";

const TILES = resolve(process.cwd(), "../../data/tiles");
const NATIVE = "synthetic-yard/splat/sog/";

interface Stats {
  overlayDraws: number;
  overlayLoopTicks: number;
  overlayWakes: Record<string, number>;
  overlayCanvas: { width: number; height: number; pixelRatio: number };
}

async function open(page: Page, options: { native: boolean }): Promise<void> {
  await page.route("**/fixture-tiles/**", (route) => {
    let relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    if (relative.startsWith(NATIVE)) {
      if (!options.native) return route.fulfill({ status: 404, body: "" });
      relative = `synthetic-yard-sog/${relative.slice(NATIVE.length)}`;
    }
    const contentType = relative.endsWith(".json")
      ? "application/json"
      : relative.endsWith(".webp")
        ? "image/webp"
        : "model/gltf-binary";
    return route.fulfill({
      status: 200,
      contentType,
      body: readFileSync(resolve(TILES, relative)),
    });
  });
  const html = `<!doctype html><html><head><style>
#v .cesium-widget, #v .cesium-widget > canvas:first-child { width: 100vw; height: 100vh; display: block; }
</style></head><body style="margin:0;background:#10141a">
<div id="v" style="position:relative;width:100vw;height:100vh"></div>
<script type="module">
const h = await import("/src/dev/scanRendererHarness.ts");
window.__scan = await h.startScanRendererHarness({ container: document.getElementById("v"),
  tilesetUrl: "/fixture-tiles/synthetic-yard/splat/tileset.json", rangeM: 45,
  requestRenderMode: true });
</script></body></html>`;
  await page.route("**/__scan-overlay-idle", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: html }),
  );
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.goto("/__scan-overlay-idle");
  await page.waitForFunction(() => "__scan" in window, undefined, { timeout: 180_000 });
}

const stats = (page: Page): Promise<Stats> =>
  page.evaluate(
    () =>
      JSON.parse(
        JSON.stringify((window as unknown as { __twinStats: Stats }).__twinStats),
      ) as Stats,
  );

/** Draws and loop ticks over `ms` of a still camera. */
async function rest(page: Page, ms: number): Promise<{ draws: number; ticks: number }> {
  const before = await stats(page);
  await page.waitForTimeout(ms);
  const after = await stats(page);
  return {
    draws: after.overlayDraws - before.overlayDraws,
    ticks: after.overlayLoopTicks - before.overlayLoopTicks,
  };
}

interface Status {
  tiles: number;
  error: string | null;
  native: boolean;
  api: string | null;
  meter: { fps: number; p95Ms: number; cpuMs: number; frames: number } | null;
}

for (const [kind, native] of [
  ["playcanvas", false],
  ["spark", false],
  ["playcanvas", true],
  ["playcanvas-webgpu", false],
  ["playcanvas-webgpu", true],
] as const) {
  const webgpu = kind === "playcanvas-webgpu";
  test(
    `${kind}${native ? " (native)" : ""} draws nothing once a still scan has loaded`,
    { tag: webgpu ? "@webgpu" : [] },
    async ({ page }, testInfo) => {
      test.setTimeout(300_000);
      await page.setViewportSize({ width: 960, height: 600 });
      await open(page, { native });
      if (webgpu) test.skip((await webgpuAdapter(page)) === null, "no WebGPU adapter");
      const status: Status = await page.evaluate(
        `window.__scan.use(${JSON.stringify(kind)}, ${native ? 20 : 90})`,
      );
      expect(status.error).toBeNull();
      expect(status.native).toBe(native);
      // The trial is measured on WebGPU, not on its WebGL2 fallback.
      if (webgpu) expect(status.api).toBe("webgpu");
      // Whatever was still settling (fades, the last sorts) has a few seconds to finish.
      await page.waitForTimeout(3000);
      const idle = await rest(page, 5000);
      // A move wakes it: frames are drawn while the camera turns, and stop again after.
      const beforeMove = (await stats(page)).overlayDraws;
      const turnStart = Date.now();
      await page.evaluate(`window.__scan.orbit(20, 30)`);
      const turnMs = Date.now() - turnStart;
      const moved = (await stats(page)).overlayDraws - beforeMove;
      // The turn's frames are what the developer readouts' frame meter reads.
      const current: Status = await page.evaluate(`window.__scan.status()`);
      const meter = current.meter;
      await page.waitForTimeout(3000);
      const after = await rest(page, 5000);
      const result = {
        kind,
        native,
        status,
        idle,
        moved,
        turnMs,
        meter,
        after,
        stats: await stats(page),
      };
      writeFileSync(testInfo.outputPath("idle.json"), JSON.stringify(result, null, 1));
      console.info(JSON.stringify(result));
      expect(moved).toBeGreaterThan(5);
      // PlayCanvas, on either API, is what the meter compares in person. (Spark's turn is
      // recorded, not held to it: under SwiftShader on a loaded runner its frames can come
      // further apart than the meter's gap between gestures, and then there is no reading.)
      if (kind !== "spark") {
        expect(meter?.frames).toBeGreaterThanOrEqual(8);
        expect(meter?.fps).toBeGreaterThan(0);
      }
      expect(idle.draws).toBeLessThanOrEqual(2);
      expect(after.draws).toBeLessThanOrEqual(2);
      // PlayCanvas's own update loop pauses too once nothing is loading or sorting.
      if (kind !== "spark") {
        expect(idle.ticks).toBeLessThanOrEqual(30);
        expect(after.ticks).toBeLessThanOrEqual(30);
      }
    },
  );
}

test("playcanvas pauses again after a tile is evicted at rest", async ({ page }) => {
  // A disposed tile's resource waits a few rendered frames before it is destroyed, and
  // PlayCanvas's loop does not pause while one waits. A tile evicted at rest used to get the
  // one frame it was evicted in: the resource waited for ever, and the loop ticked every
  // display frame for as long as the page stayed open (playcanvasBackend.ts).
  test.setTimeout(300_000);
  await page.setViewportSize({ width: 960, height: 600 });
  await open(page, { native: false });
  // Each plan records its streamer, so the test can reach the tile cache.
  await page.evaluate(`import("/src/view/stream.ts").then((m) => {
    const update = m.TileStreamer.prototype.update;
    m.TileStreamer.prototype.update = function (view) {
      window.__streamer = this;
      return update.call(this, view);
    };
  })`);
  const status: Status = await page.evaluate(`window.__scan.use("playcanvas", 90)`);
  expect(status.error).toBeNull();
  await page.waitForTimeout(3000);
  expect((await rest(page, 2000)).ticks).toBeLessThanOrEqual(30);
  // At rest, a tile's load lands after the view abandoned it, and the streamer disposes it at
  // once (view/stream.ts, `start`): a real PlayCanvas resource, made and let go of with
  // nothing moving. Then the globe draws one frame for something else (its resolution).
  const disposed: string = await page.evaluate(`(async () => {
    const streamer = window.__streamer;
    const [tile] = [...streamer.loaded.keys()];
    const mesh = await streamer.host.load(tile, new AbortController().signal);
    streamer.host.dispose(mesh);
    return tile.uri;
  })()`);
  expect(disposed).toMatch(/\.glb$/);
  await page.evaluate(`window.__scan.setResolution(false, 0.9)`);
  await page.waitForTimeout(3000);
  const after = await rest(page, 5000);
  console.info(JSON.stringify({ disposed, after }));
  expect(after.draws).toBeLessThanOrEqual(2);
  expect(after.ticks).toBeLessThanOrEqual(30);
});

test("a renderer that throws while drawing is retired, and the globe keeps rendering", async ({
  page,
}) => {
  // The overlay draws inside the globe's `postRender`, which CesiumJS raises outside the try
  // that turns a render error into `renderError`: a throw there used to stop CesiumJS's render
  // loop for good, with no recovery (overlayFrames.ts).
  test.setTimeout(300_000);
  await page.setViewportSize({ width: 960, height: 600 });
  await open(page, { native: false });
  const status: Status = await page.evaluate(`window.__scan.use("playcanvas", 90)`);
  expect(status.error).toBeNull();
  expect(await page.evaluate(`window.__scan.globeRenders()`)).toBe(true);
  // The tile planner, which runs inside the overlay's frame, breaks; a turn re-plans.
  await page.evaluate(`import("/src/view/stream.ts").then((m) => {
    m.TileStreamer.prototype.update = () => { throw new Error("the planner broke"); };
  })`);
  await page.evaluate(`window.__scan.orbit(20, 20)`);
  await expect
    .poll(() => page.evaluate(`window.__scan.status().active`), { timeout: 30_000 })
    .toBe(false);
  const after: Status = await page.evaluate(`window.__scan.status()`);
  expect(after.error).toContain("the planner broke");
  expect(await page.locator("canvas[data-scan-renderer]").count()).toBe(0);
  // The globe renders on: every requested frame arrives, moving or not.
  expect(await page.evaluate(`window.__scan.globeRenders()`)).toBe(true);
  await page.evaluate(`window.__scan.orbit(10, 10)`);
  expect(await page.evaluate(`window.__scan.globeRenders()`)).toBe(true);
});

test.describe("on a 2x display", () => {
  test.use({ deviceScaleFactor: 2 });

  test("the overlay renders at the globe's resolution", async ({ page }) => {
    test.setTimeout(300_000);
    await page.setViewportSize({ width: 800, height: 500 });
    await open(page, { native: false });
    // Performance: CSS pixels (Cesium's browser-recommended resolution).
    await page.evaluate(`window.__scan.setResolution(true, 1)`);
    await page.evaluate(`window.__scan.use("playcanvas", 90)`);
    await page.waitForTimeout(1500);
    const performance = (await stats(page)).overlayCanvas;
    // Balanced at a ladder step: device pixels times 0.65.
    await page.evaluate(`window.__scan.setResolution(false, 0.65)`);
    await page.waitForTimeout(1500);
    const ladder = (await stats(page)).overlayCanvas;
    // Ultra at rest: every device pixel.
    await page.evaluate(`window.__scan.setResolution(false, 1)`);
    await page.waitForTimeout(1500);
    const ultra = (await stats(page)).overlayCanvas;
    console.info(JSON.stringify({ performance, ladder, ultra }));
    expect(performance.pixelRatio).toBeCloseTo(1, 2);
    expect(performance.width).toBe(800);
    expect(ladder.pixelRatio).toBeCloseTo(1.3, 2);
    expect(ultra.pixelRatio).toBeCloseTo(2, 2);
    expect(ultra.width).toBe(1600);
  });
});
