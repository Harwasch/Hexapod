/**
 * Many plants in one splat tileset, in a real CesiumJS: the committed synthetic yard
 * (`data/tiles/synthetic-yard/splat`: three leafy trees, five shrubs, two snags, a building and
 * a lawn, packed as a level-of-detail tileset) with the forest rig, motion sidecar and plant
 * binding `tools/captures/scene_plants.py` wrote beside its tiles — on both motion paths, under
 * Living Mode.
 *
 * What only a browser can answer: that the engine's real snapshot aggregation, tile order and
 * REPLACE swaps line up with the per-tile plant binding; that the plants move on screen; and
 * that every gaussian the binding calls static — the building, the lawn, the path — is drawn
 * from its canonical bytes throughout, checked splat by splat over the real snapshot
 * (`SplatDeformer.staticAudit`: on the CPU path the words that went to the texture, on the GPU
 * path the shader's transcription over the motion texture it was given). The arithmetic behind
 * each is unit tested (splatDeformerForest.test.ts).
 */

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");

function harnessHtml(gpu: boolean): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Living Survey yard harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/livingSurveyHarness.ts");
      window.__living = await harness.startLivingSurveyHarness({
        container: document.getElementById("viewer"),
        tilesetUrl: "/fixture-tiles/synthetic-yard/splat/tileset.json",
        rigUrl: "/fixture-tiles/synthetic-yard/splat/rig.json",
        maximumScreenSpaceError: 16,
        rangeRadii: 1.6,
        gpu: ${String(gpu)},
        living: true,
      });
    </script>
  </body>
</html>`;
}

async function openHarness(page: Page, gpu: boolean): Promise<void> {
  await page.route("**/fixture-tiles/**", (route) => {
    const relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    return route.fulfill({
      status: 200,
      contentType: relative.endsWith(".json") ? "application/json" : "model/gltf-binary",
      body: readFileSync(resolve(TILES, relative)),
    });
  });
  await page.route("**/__living-yard", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml(gpu) }),
  );
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.goto("/__living-yard");
  await page.waitForFunction(() => "__living" in window, undefined, { timeout: 60_000 });
}

interface Status {
  phase: string;
  reason?: string;
  motion: string;
  numSplats: number;
  numSplatsLoaded: number;
  tiles: number;
  selectedTiles: number;
  tileBindings: number;
  rederivations: number;
  uploads: number;
  displaced: boolean;
  plants: number;
  staticSplats: number;
  lastBoundTiles: number;
  lastBoundSplats: number;
  lastBindMs: number;
  lastDeriveMs: number;
}

interface Audit {
  motion: string;
  staticSplats: number;
  staticMoved: number;
  staticUnpinned: number;
  plantMoved: number;
  plantSplats: number;
}

type Harness = Record<string, (...args: unknown[]) => unknown>;

async function call<T>(page: Page, method: string, ...args: unknown[]): Promise<T> {
  return (await page.evaluate(
    async ([name, rest]) => {
      const harness = (window as unknown as { __living: Harness }).__living;
      const fn = harness[name];
      if (fn === undefined) throw new Error(`no harness method ${name}`);
      return await fn(...rest);
    },
    [method, args] as const,
  )) as T;
}

const WIND = { strength: 1, bearingDeg: 250 };
const CALM = { strength: 0, bearingDeg: 250 };

for (const gpu of [true, false]) {
  const path = gpu ? "GPU" : "CPU";
  test(`a yard of plants moves on the ${path} path and nothing else does`, async ({
    page,
  }, testInfo) => {
    test.setTimeout(420_000);
    await openHarness(page, gpu);

    const near = await call<Status>(page, "view", 30, 0, CALM, 150_000);
    expect(near.phase).toBe("ready");
    expect(near.reason).toBeUndefined();
    expect(near.motion).toBe(gpu ? "gpu" : "cpu");
    expect(near.plants).toBe(10);
    expect(near.tiles).toBeGreaterThan(1);
    expect(near.tiles).toBe(near.selectedTiles);
    expect(near.numSplats).toBe(near.numSplatsLoaded);
    expect(near.staticSplats).toBeGreaterThan(0);
    expect(near.staticSplats).toBeLessThan(near.numSplats);
    await call(page, "capture", "rest");
    await page.screenshot({ path: testInfo.outputPath(`${path}-rest.png`) });

    const audits: Audit[] = [];
    for (let frame = 0; frame < 4; frame += 1) {
      await call<Status>(page, "step", 4 + frame * 0.05, WIND);
      const audit = await call<Audit>(page, "staticAudit");
      audits.push(audit);
      expect(audit.motion).toBe(gpu ? "gpu" : "cpu");
      expect(audit.staticSplats).toBe(near.staticSplats);
      // Bit for bit, every one of them, every frame.
      expect(audit.staticMoved).toBe(0);
      expect(audit.staticUnpinned).toBe(0);
      expect(audit.plantMoved).toBeGreaterThan(0);
    }
    const blown = await call<Status>(page, "status");
    expect(blown.displaced).toBe(true);
    await call(page, "capture", "wind");
    await page.screenshot({ path: testInfo.outputPath(`${path}-wind.png`) });
    const moved = await call<{ meanAbs: number; changed: number }>(page, "diff", "rest", "wind");
    expect(moved.changed).toBeGreaterThan(0.0005);

    // Far: REPLACE hands the view to merged parents, bound by the same binding.
    const far = await call<Status>(page, "view", 120, 4.5, WIND, 150_000);
    expect(far.phase).toBe("ready");
    expect(far.rederivations).toBeGreaterThan(near.rederivations);
    const farAudit = await call<Audit>(page, "staticAudit");
    expect(farAudit.staticMoved).toBe(0);

    const timing = await call<{ mean: number; max: number }>(page, "measureApply", 30, 5, WIND);
    await call<Status>(page, "view", 30, 6, CALM, 150_000);
    const calm = await call<Status>(page, "step", 6.1, CALM);
    expect(calm.displaced).toBe(false);
    const record = { near, blown, far, calm, audits, farAudit, moved, timing };
    writeFileSync(testInfo.outputPath("statuses.json"), JSON.stringify(record, null, 1));
    await testInfo.attach("statuses", {
      body: JSON.stringify(record, null, 1),
      contentType: "application/json",
    });
  });
}
