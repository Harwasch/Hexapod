/**
 * The synthetic tree, in a real browser, actually moving.
 *
 * Everything the deformer does arithmetically is unit-tested against a fake primitive. What no
 * unit test can answer is whether the engine's real packed buffer, the addressing parameters the
 * device reports and a real `Texture.copyFrom` compose into a tree that bends — so this loads
 * `data/tiles/synthetic-tree` through a disk route into a bare CesiumJS page and drives the
 * deformer frame by frame.
 *
 * **The screenshots are not representative of anything visual.** Headless GL here is
 * SwiftShader, frames run 740–1730 ms, and splat blending and refinement do not look like they
 * do on a GPU. They are evidence that the pipeline runs end to end, and a diffable artefact if
 * it stops; they are not evidence about how the motion reads. That question needs a human eye on
 * real hardware, and the sort-staleness artifact in particular cannot be judged here at all.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

/** `data/tiles/synthetic-tree`, from `apps/web/e2e/`. */
const TILE_ROOT = resolve(process.cwd(), "../../data/tiles/synthetic-tree");

const CONTENT_TYPES: Record<string, string> = {
  ".json": "application/json",
  ".glb": "model/gltf-binary",
};

/** The harness page, built here rather than committed: it is scaffolding, not an app route. */
const HARNESS_HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Living Survey harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const container = document.getElementById("viewer");
      const harness = await import("/src/dev/livingSurveyHarness.ts");
      window.__livingSurvey = await harness.startLivingSurveyHarness({
        container,
        tilesetUrl: "/fixture-tiles/splat/tileset.json",
        rigUrl: "/fixture-tiles/source/rig.json",
      });
      document.title = "Living Survey harness ready";
    </script>
  </body>
</html>`;

async function openHarness(page: Page): Promise<void> {
  // No API and no Ion: the fixture is served straight off disk, which is all a splat tileset
  // needs. S0 established that this is enough to exercise the whole path.
  await page.route("**/fixture-tiles/**", (route) => {
    const url = new URL(route.request().url());
    const relative = url.pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    const file = resolve(TILE_ROOT, relative);
    const extension = file.slice(file.lastIndexOf("."));
    return route.fulfill({
      status: 200,
      contentType: CONTENT_TYPES[extension] ?? "application/octet-stream",
      body: readFileSync(file),
    });
  });
  await page.route("**/__living-survey", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: HARNESS_HTML }),
  );
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.goto("/__living-survey");
}

interface DeformerStatusJson {
  phase: string;
  reason?: string;
  numSplats: number;
  uploads: number;
  lastUploadRows: number;
  lastUploadWords: number;
  displaced: boolean;
  bakeResidualM: number;
  geodeticAlignment: number;
  observedChecksum?: string;
  captures: number;
  numSplatsLoaded: number;
}

test.describe("Living Survey: the synthetic tree deforms", () => {
  test("attaches to the real primitive and moves the splats", async ({ page }, testInfo) => {
    test.setTimeout(180_000);
    await openHarness(page);
    await page.waitForFunction(() => "__livingSurvey" in window, undefined, { timeout: 60_000 });

    const ready = (await page.evaluate(async () => {
      const harness = (window as unknown as { __livingSurvey: Record<string, unknown> })
        .__livingSurvey;
      const wait = harness.waitUntilReady as (ms: number) => Promise<unknown>;
      await wait(90_000);
      return (harness.status as () => unknown)();
    })) as DeformerStatusJson;

    // Attached: single tile, right frame, right splats. The checksum here is the bit-exact one
    // over the un-baked positions — the gap S2 left open, closed against a real tiled capture.
    expect(ready.phase).toBe("ready");
    expect(ready.reason).toBeUndefined();
    expect(ready.numSplats).toBe(2000);
    expect(ready.captures).toBeGreaterThan(0);
    expect(ready.observedChecksum).toBe("fnv1a32:2000:e272f8b5");
    expect(ready.geodeticAlignment).toBeGreaterThan(1 - 1e-6);
    // Zero means our re-bake reproduces the engine's baked positions exactly, so wind → 0
    // restores the measured pose byte for byte rather than approximately.
    expect(ready.bakeResidualM).toBe(0);

    await page.screenshot({
      path: testInfo.outputPath("synthetic-tree-rest.png"),
      fullPage: false,
    });

    // Enough wind that the motion is unmistakable in a still frame. Not a calibrated speed:
    // `strength` is a simulated, arbitrary scale (see the rig's own note).
    const blown = (await page.evaluate(async () => {
      const harness = (window as unknown as { __livingSurvey: Record<string, unknown> })
        .__livingSurvey;
      const step = harness.step as (t: number, wind: unknown) => Promise<unknown>;
      const wind = { strength: 1, bearingDeg: 250 };
      let status: unknown;
      for (let frame = 0; frame < 6; frame += 1) status = await step(4 + frame * 0.05, wind);
      const displacement = (harness.displacementM as (t: number, w: unknown) => number)(4.25, wind);
      return { status, displacement };
    })) as { status: DeformerStatusJson; displacement: number };

    expect(blown.status.phase).toBe("ready");
    expect(blown.status.displaced).toBe(true);
    expect(blown.status.uploads).toBeGreaterThanOrEqual(6);
    // One row of 8192 texels for a 2000-splat tree: 32,768 words, 128 KB per frame.
    expect(blown.status.lastUploadRows).toBe(1);
    expect(blown.displacement).toBeGreaterThan(0.1);

    await page.screenshot({
      path: testInfo.outputPath("synthetic-tree-deformed.png"),
      fullPage: false,
    });

    // And back: the texture returns to the measured pose and the deformer stops uploading.
    const rest = (await page.evaluate(async () => {
      const harness = (window as unknown as { __livingSurvey: Record<string, unknown> })
        .__livingSurvey;
      const step = harness.step as (t: number, wind: unknown) => Promise<unknown>;
      const still = { strength: 0, bearingDeg: 250 };
      await step(5, still);
      const afterRestore = (harness.status as () => unknown)() as DeformerStatusJson;
      await step(6, still);
      await step(7, still);
      return { afterRestore, final: (harness.status as () => unknown)() };
    })) as { afterRestore: DeformerStatusJson; final: DeformerStatusJson };

    expect(rest.afterRestore.displaced).toBe(false);
    // One write to restore, then nothing: an idle scene stays idle.
    expect(rest.final.uploads).toBe(rest.afterRestore.uploads);
  });
});
