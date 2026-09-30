/**
 * What streaming costs a person looking and walking around inside a splat scan
 * (dev/navigationHarness.ts): looking away and back, and walking straight ahead.
 *
 * The yard fixture runs every time and checks the one behaviour that must hold: a view
 * looked back at seconds later is drawn again at once, from what is already on the GPU, with
 * no tile fetched twice. `NAV_PERF=1 NAV_PERF_TILESET=<url>` measures a real scan instead and
 * only reports (SwiftShader: counts and main-thread time transfer, frame rates do not).
 */

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");
const REAL = process.env.NAV_PERF ? (process.env.NAV_PERF_TILESET ?? "") : "";

interface Traffic {
  readbacks: number;
  bufferAllocations: number;
  bufferUploadMB: number;
  textureUploadMB: number;
  tileRequests: number;
  tileRefetches: number;
  sorts: number;
}
interface LookReport extends Traffic {
  settledA: number;
  restoreMs: number | null;
  restoreFrames: number | null;
  firstFrameShare: number;
  awayShare: number;
}
interface WalkReport extends Traffic {
  updateMs: number[];
  intervalMs: number[];
}

async function open(page: Page, tilesetUrl: string, skipDraw = false): Promise<void> {
  await page.route("**/fixture-tiles/**", (route) => {
    const relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    return route.fulfill({
      status: 200,
      contentType: relative.endsWith(".json") ? "application/json" : "model/gltf-binary",
      body: readFileSync(resolve(TILES, relative)),
    });
  });
  const html = `<!doctype html><html><body style="margin:0"><div id="v" style="width:100vw;height:100vh"></div>
<script type="module">
const h = await import("/src/dev/navigationHarness.ts");
window.__nav = await h.startNavigationHarness({ container: document.getElementById("v"), tilesetUrl: ${JSON.stringify(tilesetUrl)}, motionFirst: true, skipDraw: ${String(skipDraw)} });
</script></body></html>`;
  await page.route("**/__nav-stream", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: html }),
  );
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.goto("/__nav-stream");
  await page.waitForFunction(() => "__nav" in window, undefined, { timeout: 180_000 });
}

const summary = (walk: WalkReport) => {
  const sorted = [...walk.updateMs].sort((a, b) => a - b);
  const at = (q: number): number =>
    Math.round((sorted[Math.floor(q * (sorted.length - 1))] ?? 0) * 10) / 10;
  const { updateMs: _u, intervalMs: _i, ...traffic } = walk;
  return { ...traffic, frames: walk.intervalMs.length, updateP50: at(0.5), updateP95: at(0.95) };
};

test("a view looked back at is drawn again from what is resident, fetching nothing twice", async ({
  page,
}, testInfo) => {
  test.setTimeout(600_000);
  await open(page, "/fixture-tiles/synthetic-yard/splat/tileset.json");
  const look: LookReport = await page.evaluate(`window.__nav.lookAround(20, 1.6)`);
  const walk: WalkReport = await page.evaluate(`window.__nav.walk(1.4, 6, 1.6)`);
  const report = { look, walk: summary(walk) };
  writeFileSync(testInfo.outputPath("yard.json"), JSON.stringify(report, null, 1));
  console.info(JSON.stringify(report, null, 1));
  expect(look.settledA).toBeGreaterThan(0);
  expect(look.tileRefetches).toBe(0);
  expect(look.firstFrameShare).toBeGreaterThanOrEqual(0.95);
  expect(walk.readbacks).toBe(0);
});

test.describe("a real scan", () => {
  test.skip(!REAL, "opt-in: NAV_PERF=1 NAV_PERF_TILESET=<url>");
  test("look around and walk", async ({ browser }, testInfo) => {
    test.setTimeout(1_800_000);
    const page = await browser.newPage({ viewport: { width: 640, height: 400 } });
    // The public bucket answers CORS for the deployed origin only: fetched here instead.
    await page.route(/\.r2\.dev\//, async (route) => {
      const response = await route.fetch();
      await route.fulfill({
        response,
        headers: { ...response.headers(), "access-control-allow-origin": "*" },
      });
    });
    await open(page, REAL, true);
    const look: LookReport = await page.evaluate(`window.__nav.lookAround(30, 1.6)`);
    const walk: WalkReport = await page.evaluate(`window.__nav.walk(1.4, 12, 1.6)`);
    const report = { look, walk: summary(walk) };
    writeFileSync(testInfo.outputPath("real.json"), JSON.stringify(report, null, 1));
    console.info(JSON.stringify(report, null, 1));
  });
});
