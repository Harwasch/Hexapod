/**
 * The view-cone fade in a real CesiumJS 1.145 with the engine patch's `vertexVisibility` hook.
 *
 * `data/tiles/synthetic-tree-lod` carries the packer's own `viewcones.bin`; the synthetic tree
 * is seen from everywhere, so under it the tree must look exactly as it does with the hook
 * off. Under a grid that says every cell was seen from below only (`src/dev/viewConesHarness.ts`)
 * the tree must vanish seen from above, and be drawn as before seen from below: the hook
 * compiles, reads the right cell through the bake matrix, and fades by the camera's direction.
 *
 * Headless GL is SwiftShader: coverage is counted, not eyeballed.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");

const HARNESS_HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>View cones harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/viewConesHarness.ts");
      window.__viewCones = await harness.startViewConesHarness({
        container: document.getElementById("viewer"),
        url: "/fixture-tiles/synthetic-tree-lod/tileset.json",
      });
    </script>
  </body>
</html>`;

interface Harness {
  view(pitchDeg: number, rangeM: number, mode: string): Promise<{ coverage: number }>;
  installed(): boolean;
}

test("the view-cone hook fades by where the camera is, and nothing the capture saw", async ({
  page,
}) => {
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
  await page.route("**/view-cones-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: HARNESS_HTML }),
  );
  await page.goto("/view-cones-harness.html");
  await page.waitForFunction(() => "__viewCones" in window, undefined, { timeout: 60_000 });

  const view = (pitch: number, mode: string) =>
    page.evaluate(
      ([p, m]) => (window as unknown as { __viewCones: Harness }).__viewCones.view(p, 14, m),
      [pitch, mode] as [number, string],
    );

  // Twice: the first view also waits for the level-of-detail tiles to finish refining.
  await view(-60, "off");
  const aboveOff = await view(-60, "off");
  const aboveFile = await view(-60, "file");
  const aboveBelow = await view(-60, "below");
  const belowOff = await view(80, "off");
  const belowBelow = await view(80, "below");
  const installed = await page.evaluate(() =>
    (window as unknown as { __viewCones: Harness }).__viewCones.installed(),
  );
  await page.screenshot({ path: test.info().outputPath("below-only-from-below.png") });

  test.info().annotations.push({
    type: "coverage",
    description: JSON.stringify({ aboveOff, aboveFile, aboveBelow, belowOff, belowBelow }),
  });
  expect(errors.filter((e) => /shader|compile|link/i.test(e))).toEqual([]);
  expect(installed).toBe(true);
  expect(aboveOff.coverage).toBeGreaterThan(0.02);
  // The packer's own grid fades nothing on a scan seen from everywhere.
  expect(Math.abs(aboveFile.coverage - aboveOff.coverage)).toBeLessThan(0.002);
  // Seen from below only: gone from above...
  expect(aboveBelow.coverage).toBeLessThan(aboveOff.coverage * 0.02);
  // ...and as it was from below: nothing faded (re-sorting between frames moves coverage by a
  // few per cent either way, which is why this is a floor and not an equality).
  expect(belowOff.coverage).toBeGreaterThan(0.02);
  expect(belowBelow.coverage).toBeGreaterThan(belowOff.coverage * 0.9);
});
