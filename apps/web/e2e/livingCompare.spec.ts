/**
 * Blind-comparison clips: the legacy nine-sine wind against Living Mode, same tree, same camera,
 * same scene clock — for a *human* two-choice test (ADR 0008, phase 1 pass criterion).
 *
 * Nothing here asserts that the motion is believable; no test can. It renders two numbered PNG
 * sequences of the synthetic tree through the real `CesiumSceneManager`, stepping the scene
 * clock by hand so every frame is exactly `t0 + k / fps`, and writes them under neutral labels
 * `A` and `B` with the unblinding key in a separate file. A person then views them side by side
 * (the ffmpeg lines are printed and written to `README.txt`) and says which looks more like a
 * tree in wind, without knowing which is which.
 *
 * **Skipped unless `LIVING_COMPARE=1`.** It is minutes of SwiftShader rendering, and headless
 * frames are not representative of a GPU (see `livingSurveyScene.spec.ts`): the clips are for
 * judging *motion*, and should be re-rendered on real hardware before anyone judges *looks*.
 *
 *     LIVING_COMPARE=1 LIVING_COMPARE_FRAMES=72 LIVING_COMPARE_OUT=/tmp/compare \
 *       PLAYWRIGHT_CHROMIUM_EXECUTABLE=/opt/pw-browsers/chromium \
 *       npx playwright test e2e/livingCompare.spec.ts
 *
 * Environment: `LIVING_COMPARE_FRAMES` (default 48), `LIVING_COMPARE_FPS` (24),
 * `LIVING_COMPARE_STRENGTH` (0.1, the default wind), `LIVING_COMPARE_SEED` (flips the A/B
 * assignment), `LIVING_COMPARE_OUT` (default: the test's output folder).
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const TILE_ROOT = resolve(process.cwd(), "../../data/tiles/synthetic-tree");
const LONGITUDE = -82.6966;
const LATITUDE = 28.0389;

const CONTENT_TYPES: Record<string, string> = {
  ".json": "application/json",
  ".glb": "model/gltf-binary",
};

const HARNESS_HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Living Mode comparison</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/livingSceneHarness.ts");
      window.__livingScene = await harness.startLivingSceneHarness({
        container: document.getElementById("viewer"),
        tilesetUrl: new URL("/fixture-tiles/splat/tileset.json", location.href).toString(),
        slug: "synthetic-tree",
        longitude: ${LONGITUDE},
        latitude: ${LATITUDE},
      });
      document.title = "Living Mode comparison ready";
    </script>
  </body>
</html>`;

const enabled = process.env.LIVING_COMPARE === "1";

function numberFromEnv(name: string, fallback: number): number {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

async function openHarness(page: Page): Promise<void> {
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
  await page.route("**/__living-compare", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: HARNESS_HTML }),
  );
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.goto("/__living-compare");
}

async function call<T>(page: Page, method: string, ...args: unknown[]): Promise<T> {
  const result = await page.evaluate(
    async ({ method, args }) => {
      const harness = (window as unknown as { __livingScene: Record<string, unknown> })
        .__livingScene;
      const fn = harness[method] as (...a: unknown[]) => unknown;
      return await fn.apply(harness, args);
    },
    { method, args },
  );
  return result as T;
}

test.describe("Living Mode: blind comparison clips", () => {
  test.skip(!enabled, "set LIVING_COMPARE=1 to render comparison clips");
  test.use({ viewport: { width: 720, height: 900 } });

  test("renders the legacy and Living Mode winds as anonymous A/B sequences", async ({
    page,
  }, testInfo) => {
    const frames = Math.floor(numberFromEnv("LIVING_COMPARE_FRAMES", 48));
    const fps = numberFromEnv("LIVING_COMPARE_FPS", 24);
    const strength = numberFromEnv("LIVING_COMPARE_STRENGTH", 0.1);
    const flip = Math.floor(numberFromEnv("LIVING_COMPARE_SEED", 1)) % 2 === 1;
    const out = process.env.LIVING_COMPARE_OUT ?? testInfo.outputPath("compare");
    test.setTimeout(120_000 + frames * 2 * 8_000);

    await openHarness(page);
    await page.waitForFunction(() => "__livingScene" in window, undefined, { timeout: 90_000 });
    const ready = await call<{ sites: { phase: string; motionEvidence: string | null }[] }>(
      page,
      "waitUntilReady",
      150_000,
    );
    expect(ready.sites[0]?.phase).toBe("ready");
    expect(ready.sites[0]?.motionEvidence).toBe("allometric");

    const models = flip ? (["auto", "legacy"] as const) : (["legacy", "auto"] as const);
    const key: Record<string, string> = {};
    await call(page, "setWind", { strength, bearingDeg: 250 });
    for (const [index, model] of models.entries()) {
      const label = index === 0 ? "A" : "B";
      key[label] = model === "auto" ? "living-mode (allometric)" : "legacy nine-sine";
      await call(page, "setMotionModel", model);
      const folder = join(out, label);
      mkdirSync(folder, { recursive: true });
      for (let k = 0; k < frames; k += 1) {
        // The same instants for both, well away from t = 0 so neither starts from rest.
        await call(page, "setTime", 600 + k / fps);
        const dataUrl = await call<string>(page, "grabFrame");
        const png = Buffer.from(dataUrl.replace(/^data:image\/png;base64,/, ""), "base64");
        writeFileSync(join(folder, `frame_${String(k).padStart(4, "0")}.png`), png);
      }
    }
    await call(page, "setWind", { strength: 0, bearingDeg: 250 });

    writeFileSync(join(out, "key.json"), `${JSON.stringify({ ...key, strength, fps }, null, 1)}\n`);
    const assemble = [
      `ffmpeg -framerate ${fps} -i A/frame_%04d.png -framerate ${fps} -i B/frame_%04d.png \\`,
      `  -filter_complex "[0:v][1:v]hstack=inputs=2" -c:v libx264 -pix_fmt yuv420p side_by_side.mp4`,
    ].join("\n");
    writeFileSync(
      join(out, "README.txt"),
      [
        "Living Mode blind comparison (docs/DECISIONS/0008-living-mode.md).",
        "",
        "Two PNG sequences of the same synthetic tree, same camera, same scene clock and wind",
        `(strength ${strength}, ${frames} frames at ${fps} fps). Show a viewer A and B side by`,
        "side, looped, and ask which looks more like a tree moving in wind. Do not open",
        "key.json until every viewer has answered. Rendered with SwiftShader: judge the motion,",
        "not the image quality, and re-render on a GPU before judging looks.",
        "",
        "Assemble a side-by-side clip:",
        assemble,
        "",
      ].join("\n"),
    );
    console.info(`comparison frames written to ${out}\n${assemble}`);
  });
});
