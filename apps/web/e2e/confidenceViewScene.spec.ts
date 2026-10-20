/**
 * The Confidence / Data age colour part in a real CesiumJS 1.145 with the engine patch's
 * `vertexColor` hook, on the committed synthetic yard (`src/dev/cellViewHarness.ts`).
 *
 * Under a grid that calls every cell weak, most of what the scan covers must turn red; under
 * one that calls every cell strong, green and not red; under one weak west of the yard's middle
 * and strong east of it, seen looking north, red on the left of the screen and green on the
 * right -- so the shader compiles, reads each splat's own cell through the root frame, and
 * tints by `weak-v1`. Off, the yard is not red. Data age four months old is the 3-6 months
 * colour (by hue: a wrong bucket fails), and with a later date's evidence everywhere it is
 * striped with light bands besides (a missing stripe fails).
 *
 * Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");
const EVIDENCE = process.env.W4A_EVIDENCE_DIR;

const HARNESS_HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Cell view harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/cellViewHarness.ts");
      window.__cellView = await harness.startCellViewHarness({
        container: document.getElementById("viewer"),
        url: "/fixture-tiles/synthetic-yard/splat/tileset.json",
      });
    </script>
  </body>
</html>`;

interface Counts {
  coverage: number;
  red: number;
  green: number;
  leftRed: number;
  rightRed: number;
  leftGreen: number;
  rightGreen: number;
  ageHue: number;
  light: number;
}

interface Harness {
  view(grid: string, mode?: string): Promise<Counts>;
  installed(): boolean;
}

test("weak cells draw red, strong ones green, each splat by its own cell", async ({ page }) => {
  // Seven views on software GL, the first waiting for the yard to refine: minutes, not seconds.
  test.setTimeout(420_000);
  await page.setViewportSize({ width: 960, height: 640 });
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/fixture-tiles/**", (route) => {
    const relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    const type = relative.endsWith(".json")
      ? "application/json"
      : relative.endsWith(".bin")
        ? "application/octet-stream"
        : "model/gltf-binary";
    return route.fulfill({
      status: 200,
      contentType: type,
      body: readFileSync(resolve(TILES, relative)),
    });
  });
  await page.route("**/cell-view-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: HARNESS_HTML }),
  );
  await page.goto("/cell-view-harness.html");
  await page.waitForFunction(() => "__cellView" in window, undefined, { timeout: 90_000 });

  const view = async (grid: string, mode = "confidence") => {
    const counts = await page.evaluate(
      ([g, m]) => (window as unknown as { __cellView: Harness }).__cellView.view(g, m),
      [grid, mode] as [string, string],
    );
    const name = mode === "confidence" ? grid : `${grid}-${mode}`;
    const path = EVIDENCE ? `${EVIDENCE}/scene-${name}.png` : test.info().outputPath(`${name}.png`);
    await page.screenshot({ path });
    await test.info().attach(name, { path, contentType: "image/png" });
    return counts;
  };

  // Twice: the first view also waits for the level-of-detail tiles to finish refining.
  await view("off");
  const off = await view("off");
  const weak = await view("weak");
  const installed = await page.evaluate(() =>
    (window as unknown as { __cellView: Harness }).__cellView.installed(),
  );
  const strong = await view("strong");
  const split = await view("split");
  const age = await view("weak", "data-age");
  const newer = await view("split", "data-age");
  test.info().annotations.push({
    type: "pixels",
    description: JSON.stringify({ off, weak, strong, split, age, newer }),
  });

  expect(errors.filter((e) => /shader|compile|link|uniform/i.test(e))).toEqual([]);
  expect(installed).toBe(true);
  expect(off.coverage).toBeGreaterThan(0.05);
  // The photo is not red; under an all-weak grid, most of what is covered is.
  expect(off.red).toBeLessThan(0.1);
  expect(weak.red).toBeGreaterThan(0.6);
  expect(weak.green).toBeLessThan(0.05);
  // All strong: green, not red.
  expect(strong.green).toBeGreaterThan(0.6);
  expect(strong.red).toBeLessThan(0.05);
  // Weak west, strong east, looking north: red left, green right.
  expect(split.leftRed).toBeGreaterThan(0.5);
  expect(split.rightRed).toBeLessThan(0.1);
  expect(split.rightGreen).toBeGreaterThan(0.5);
  // Data age, four months old: the 3-6 months colour (blue-violet), which neither the photo nor
  // a wrong bucket's colour is; with a later date's evidence everywhere, light stripes besides.
  expect(off.ageHue).toBeLessThan(0.1);
  expect(age.ageHue).toBeGreaterThan(0.6);
  expect(age.red).toBeLessThan(0.05);
  expect(age.green).toBeLessThan(0.05);
  expect(newer.coverage).toBeGreaterThan(off.coverage * 0.8);
  expect(newer.light).toBeGreaterThan(age.light + 0.15);
  expect(newer.ageHue + newer.light).toBeGreaterThan(0.7);
});
