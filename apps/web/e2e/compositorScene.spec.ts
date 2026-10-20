/**
 * The per-cell view rules in a real CesiumJS 1.145 with the engine patch's companions and
 * visibility hook, on the committed synthetic yard (`src/dev/compositorHarness.ts`): two dates
 * of the yard (the same tiles twice, so they overlap splat for splat), each splat tinted by its
 * date with a debug colour -- March red, June blue, an inferred layer green.
 *
 * - Both drawn with the rule's parts off: red and blue splats interleave in one sort, so the
 *   view is mixed (purple) -- the check below means nothing without it.
 * - Latest, March everywhere and June the east half, looking north: the left of the screen is
 *   red and the right blue, each cell's pixels from one date, almost no mix.
 * - March only north-west, June east, an inferred layer over all by its tile bounds: Latest
 *   fills the south-west green; Ground truth draws nothing there.
 *
 * And the plumbing: June's own primitive is drawn by March's, and the slots know their sources.
 * Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");
const EVIDENCE = process.env.W5B_EVIDENCE_DIR;

const HARNESS_HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Compositor harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/compositorHarness.ts");
      window.__compositor = await harness.startCompositorHarness({
        container: document.getElementById("viewer"),
        url: "/fixture-tiles/synthetic-yard/splat/tileset.json",
      });
    </script>
  </body>
</html>`;

interface Region {
  covered: number;
  red: number;
  blue: number;
  green: number;
  mixed: number;
}

interface Counts {
  coverage: number;
  left: Region;
  right: Region;
  southWest: Region;
  all: Region;
  companion: boolean;
  sources: { march: number; june: number; fill: number };
}

test("under Latest each cell's pixels come from one date; Ground truth hides the inferred", async ({
  page,
}) => {
  // Four views on software GL, each waiting for three tilesets to refine: about six minutes
  // here (2026-10-20), twice that allowed for a slower runner.
  test.setTimeout(720_000);
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
  await page.route("**/compositor-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: HARNESS_HTML }),
  );
  await page.goto("/compositor-harness.html");
  await page.waitForFunction(() => "__compositor" in window, undefined, { timeout: 90_000 });

  const view = async (scenario: string): Promise<Counts> => {
    const counts = await page.evaluate(
      (s) =>
        (
          window as unknown as { __compositor: { view(s: string): Promise<Counts> } }
        ).__compositor.view(s),
      scenario,
    );
    const path = EVIDENCE
      ? `${EVIDENCE}/scene-${scenario}.png`
      : test.info().outputPath(`${scenario}.png`);
    await page.screenshot({ path });
    await test.info().attach(scenario, { path, contentType: "image/png" });
    return counts;
  };

  const both = await view("both");
  const latest = await view("latest");
  const gapLatest = await view("gap-latest");
  const gapTruth = await view("gap-truth");
  test.info().annotations.push({
    type: "pixels",
    description: JSON.stringify({ both, latest, gapLatest, gapTruth }),
  });

  expect(errors.filter((e) => /shader|compile|link|uniform/i.test(e))).toEqual([]);
  // The plumbing: one primitive draws both dates, and its slots know whose they are.
  expect(latest.companion).toBe(true);
  expect(latest.sources.march).toBeGreaterThan(0);
  expect(latest.sources.june).toBeGreaterThan(0);
  expect(gapLatest.sources.fill).toBeGreaterThan(0);
  expect(both.coverage).toBeGreaterThan(0.05);
  // Without the rule, two dates of one place blend: the check below would bite.
  expect(both.all.mixed).toBeGreaterThan(0.3);
  // Latest: March west, June east, and next to no pixel of both.
  expect(latest.left.red).toBeGreaterThan(0.8);
  expect(latest.left.blue).toBeLessThan(0.05);
  expect(latest.right.blue).toBeGreaterThan(0.8);
  expect(latest.right.red).toBeLessThan(0.05);
  expect(latest.left.mixed + latest.right.mixed).toBeLessThan(0.05);
  expect(latest.all.mixed).toBeLessThan(both.all.mixed / 5);
  // Where no date saw, Latest draws the inferred layer; Ground truth draws nothing.
  expect(gapLatest.southWest.green).toBeGreaterThan(0.6);
  expect(gapTruth.southWest.green).toBeLessThan(0.02);
  expect(gapTruth.southWest.covered).toBeLessThan(gapLatest.southWest.covered / 4);
  expect(gapTruth.right.blue).toBeGreaterThan(0.8);
});
