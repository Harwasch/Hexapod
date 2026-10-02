/**
 * What streaming a large splat costs a moving camera, with and without the motion gate
 * (cesium/splatMotionGate.ts): the main thread's splat work during a scripted orbit.
 *
 * Opt-in (`NAV_PERF=1`, with `NAV_PERF_TILESET` the URL of a large level-of-detail splat
 * tileset): it is a measurement, not a check, and it streams hundreds of megabytes. Under
 * SwiftShader the frame *rate* says little (the GPU is emulated on the CPU); the CPU profile's
 * splat functions -- SPZ decode, snapshot aggregation, texture generation and upload -- and
 * the per-frame update time are what transfer.
 *
 * Measured on the Fort Clatsop site (22.6M gaussians, 514 tiles), 20 s orbit at 25 m:
 * splat work on the main thread 2,240 ms before, 430 ms after; update p95 22.6 -> 7.6 ms,
 * worst 43.6 -> 19.4 ms.
 */

import { writeFileSync } from "node:fs";

import { test, type Page } from "@playwright/test";

const TILESET = process.env.NAV_PERF_TILESET ?? "";

/** Profile functions that are splat streaming work on the main thread. */
const SPLAT_WORK =
  /generateSplatTexture|processGeneratedSplatTextureData|aggregateAttributeValues|transformTile|processSpz|texImage2D|wasm-function|worker\.onmessage|GaussianSplatPrimitive\.update|multiplyByPoint/;

interface Report {
  updateMs: number[];
  intervalMs: number[];
  rebuilds: number;
  tilesLoaded: number;
  drawn: number;
}

async function measure(page: Page, motionFirst: boolean) {
  const html = `<!doctype html><html><body style="margin:0"><div id="v" style="width:100vw;height:100vh"></div>
<script type="module">
const h = await import("/src/dev/navigationHarness.ts");
window.__nav = await h.startNavigationHarness({ container: document.getElementById("v"), tilesetUrl: ${JSON.stringify(TILESET)}, motionFirst: ${String(motionFirst)} });
</script></body></html>`;
  await page.route("**/__nav-perf", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: html }),
  );
  await page.goto("/__nav-perf");
  await page.waitForFunction(() => "__nav" in window, undefined, { timeout: 180_000 });
  await page.evaluate(`window.__nav.settle(60)`);
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Profiler.enable");
  await cdp.send("Profiler.setSamplingInterval", { interval: 500 });
  await cdp.send("Profiler.start");
  const report: Report = await page.evaluate(() =>
    (
      window as unknown as { __nav: { orbit(range: number, s: number): Promise<Report> } }
    ).__nav.orbit(25, 20),
  );
  const { profile } = await cdp.send("Profiler.stop");
  const byId = new Map(profile.nodes.map((node) => [node.id, node]));
  let splatMs = 0;
  profile.samples?.forEach((id, i) => {
    const name = byId.get(id)?.callFrame.functionName ?? "";
    if (SPLAT_WORK.test(name)) splatMs += (profile.timeDeltas?.[i] ?? 0) / 1000;
  });
  const sorted = [...report.updateMs].sort((a, b) => a - b);
  const at = (q: number): number => sorted[Math.floor(q * (sorted.length - 1))] ?? 0;
  return {
    motionFirst,
    splatMainThreadMs: Math.round(splatMs),
    updateP50: at(0.5),
    updateP95: at(0.95),
    updateMax: at(1),
    rebuilds: report.rebuilds,
    tilesLoaded: report.tilesLoaded,
    drawn: report.drawn,
  };
}

test.describe("navigation cost of a large splat", () => {
  test.skip(!process.env.NAV_PERF || !TILESET, "opt-in: NAV_PERF=1 NAV_PERF_TILESET=<url>");

  test("orbit with and without the motion gate", async ({ browser }, testInfo) => {
    test.setTimeout(900_000);
    const results = [];
    for (const motionFirst of [false, true]) {
      const page = await browser.newPage({ viewport: { width: 480, height: 320 } });
      results.push(await measure(page, motionFirst));
      await page.close();
    }
    writeFileSync(testInfo.outputPath("navigation.json"), JSON.stringify(results, null, 1));
    console.info(JSON.stringify(results, null, 1));
  });
});
