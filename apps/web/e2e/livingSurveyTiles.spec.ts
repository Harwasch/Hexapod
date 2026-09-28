/**
 * The Living Survey on a level-of-detail splat tileset, in a real CesiumJS: the committed
 * REPLACE fixture (`data/tiles/synthetic-tree-lod`, merged parents over the synthetic tree's
 * 12,000 gaussians) with its stamped rig, moving on both motion paths.
 *
 * What only a browser can answer: that the engine's real snapshot aggregation, its tile order
 * and its REPLACE swaps line up with the deformer's per-tile bookkeeping; that motion survives
 * the camera changing which tiles are drawn; and that the patched vertex shader compiles and
 * displaces. The arithmetic behind each of those is unit tested (splatDeformerTiles.test.ts).
 *
 * Headless GL is SwiftShader. Pixels here are evidence that the paths run and that motion
 * reaches the screen — frames are compared for *difference*, never judged for looks.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");

function harnessHtml(gpu: boolean): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Living Survey tiles harness</title>
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
        tilesetUrl: "/fixture-tiles/synthetic-tree-lod/tileset.json",
        rigUrl: "/fixture-tiles/synthetic-tree-lod/rig.json",
        maximumScreenSpaceError: 16,
        gpu: ${String(gpu)},
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
  await page.route("**/__living-tiles", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml(gpu) }),
  );
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.goto("/__living-tiles");
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
  lastUploadWords: number;
  displaced: boolean;
  bakeResidualM: number;
  lastDeriveMs: number;
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

for (const gpu of [false, true]) {
  const path = gpu ? "GPU" : "CPU";
  test(`a REPLACE tileset moves on the ${path} path, through tile swaps, and comes back`, async ({
    page,
  }, testInfo) => {
    test.setTimeout(420_000);
    await openHarness(page, gpu);

    // Near: leaves. Attached across several tiles, every one proven against the rig.
    const near = await call<Status>(page, "view", 5, 0, CALM, 150_000);
    expect(near.phase).toBe("ready");
    expect(near.reason).toBeUndefined();
    expect(near.motion).toBe(gpu ? "gpu" : "cpu");
    expect(near.tiles).toBeGreaterThan(1);
    expect(near.tiles).toBe(near.selectedTiles);
    expect(near.numSplats).toBe(near.numSplatsLoaded);
    expect(near.bakeResidualM).toBe(0);
    await call(page, "capture", "rest");
    await page.screenshot({ path: testInfo.outputPath(`${path}-rest.png`) });

    // Wind: it moves, and the frame shows it.
    for (let frame = 0; frame < 4; frame += 1) {
      await call<Status>(page, "step", 4 + frame * 0.05, WIND);
    }
    const blown = await call<Status>(page, "status");
    expect(blown.phase).toBe("ready");
    expect(blown.displaced).toBe(true);
    expect(blown.uploads).toBeGreaterThan(0);
    await call(page, "capture", "wind");
    await page.screenshot({ path: testInfo.outputPath(`${path}-wind.png`) });
    const moved = await call<{ meanAbs: number; changed: number }>(page, "diff", "rest", "wind");
    expect(moved.changed).toBeGreaterThan(0.001);

    // Far, still in wind: REPLACE hands the view to merged parents. Re-derived, not refused,
    // and the parents move too.
    const far = await call<Status>(page, "view", 40, 4.5, WIND, 150_000);
    expect(far.phase).toBe("ready");
    expect(far.rederivations).toBeGreaterThan(near.rederivations);
    // Parents already bound on the way in are reused, not re-proven.
    expect(far.tileBindings).toBeGreaterThanOrEqual(near.tileBindings);
    expect(far.tiles).toBe(far.selectedTiles);
    expect(far.displaced).toBe(true);
    expect(far.numSplats).toBeLessThan(near.numSplats);

    // Back near, still in wind, then calm.
    const again = await call<Status>(page, "view", 5, 5, WIND, 150_000);
    expect(again.phase).toBe("ready");
    expect(again.displaced).toBe(true);
    expect(again.rederivations).toBeGreaterThan(far.rederivations);
    // The leaves are still loaded: their bindings are reused, not recomputed.
    expect(again.tileBindings).toBe(far.tileBindings);
    expect(again.numSplats).toBe(near.numSplats);
    await call<Status>(page, "step", 6, CALM);
    const calm = await call<Status>(page, "step", 6.1, CALM);
    expect(calm.displaced).toBe(false);
    const quiet = await call<Status>(page, "step", 6.2, CALM);
    // One restoring write, then nothing: an idle scene stays idle.
    expect(quiet.uploads).toBe(calm.uploads);
    await call(page, "capture", "calm");
    await page.screenshot({ path: testInfo.outputPath(`${path}-calm.png`) });
    const restored = await call<{ meanAbs: number; changed: number }>(page, "diff", "rest", "calm");
    const record = { near, blown, far, again, calm, moved, restored };
    await testInfo.attach("statuses", {
      body: JSON.stringify(record, null, 1),
      contentType: "application/json",
    });
    // The measured pose again. Not asserted pixel-exact: the snapshot was rebuilt and its tiles
    // may aggregate in another order, which can flip the draw order of equal-depth splats. The
    // restore itself is byte-exact and asserted over the uploaded words in unit tests.
    expect(restored.meanAbs).toBeLessThan(moved.meanAbs / 10);
  });
}
