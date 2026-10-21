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
 * - The seam between them, measured (ADR 0017): of the covered pixels within one cell (1 m) of
 *   it on screen, the share that blend both dates (`seamMixed`: after the background's own red
 *   and blue are taken out, the lesser of red and blue is at least a tenth of their sum). With
 *   splats kept by their centres alone (reach 0) a footprint's tail lies over the next date's
 *   splats and some 6% blend; with `SEAM_REACH` under 1% must, at the cost of a darker line.
 * - March only north-west, June east, an inferred layer over all by its tile bounds: Latest
 *   fills the south-west green; Ground truth draws nothing there.
 *
 * And the plumbing: June's own primitive is drawn by March's, and the slots know their sources.
 *
 * The same, measured the same way, under PlayCanvas (the app's default renderer): March drawn by
 * the app's PlayCanvas renderer, June and June's inferred layer streamed into its one app and
 * sort, each tile weighed by the plane in the copy pass (scanView/scanRule.ts). The seam's
 * figure is asserted under 1% there too.
 *
 * Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");
const EVIDENCE = process.env.W5B_EVIDENCE_DIR;

const harnessHtml = (renderer: "cesium" | "playcanvas") => `<!doctype html>
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
        renderer: "${renderer}",
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
  seam: { pixels: number; covered: number; mixed: number; profile: number[] };
  companion: boolean;
  sources: { march: number; june: number; fill: number };
}

/** Five views of the yard drawn by `renderer`, counted; asserted by each test. */
async function measure(page: Page, renderer: "cesium" | "playcanvas") {
  await page.setViewportSize({ width: 960, height: 640 });
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/fixture-tiles/**", (route) => {
    const relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    // PlayCanvas asks whether the yard has a native package beside its tileset: it has none.
    if (!existsSync(resolve(TILES, relative))) return route.fulfill({ status: 404, body: "" });
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
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml(renderer) }),
  );
  await page.goto("/compositor-harness.html");
  await page.waitForFunction(() => "__compositor" in window, undefined, { timeout: 90_000 });

  const view = async (scenario: string, seamReach?: number): Promise<Counts> => {
    const counts = await page.evaluate(
      ([s, reach]) =>
        (
          window as unknown as {
            __compositor: { view(s: string, o: { seamReach?: number }): Promise<Counts> };
          }
        ).__compositor.view(s, { seamReach: reach }),
      [scenario, seamReach] as const,
    );
    const base = seamReach === undefined ? scenario : `${scenario}-reach-${String(seamReach)}`;
    const name = renderer === "cesium" ? base : `${renderer}-${base}`;
    const path = EVIDENCE ? `${EVIDENCE}/scene-${name}.png` : test.info().outputPath(`${name}.png`);
    await page.screenshot({ path });
    await test.info().attach(name, { path, contentType: "image/png" });
    return counts;
  };

  const both = await view("both");
  const latest = await view("latest");
  const centres = await view("latest", 0);
  const gapLatest = await view("gap-latest");
  const gapTruth = await view("gap-truth");
  const figures = JSON.stringify({ renderer, both, latest, centres, gapLatest, gapTruth });
  test.info().annotations.push({ type: "pixels", description: figures });
  if (EVIDENCE) writeFileSync(`${EVIDENCE}/scene-${renderer}.json`, figures);
  return { errors, both, latest, centres, gapLatest, gapTruth };
}

/** What both renderers must show (ADR 0017's method and figures). */
function expectOneOwnerPerCell(m: Awaited<ReturnType<typeof measure>>): void {
  const { errors, both, latest, centres, gapLatest, gapTruth } = m;
  expect(errors.filter((e) => /shader|compile|link|uniform/i.test(e))).toEqual([]);
  // The plumbing: one sort draws both dates, and each date's splats know whose they are.
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
  // The seam: kept by centres alone, footprints blend there -- the measure bites; with the
  // seam's reach, under 1% of the covered pixels within a metre of it blend both dates, and
  // the seam is a line, not a hole (most of that band is still drawn).
  expect(centres.seam.covered).toBeGreaterThan(5_000);
  expect(centres.seam.mixed).toBeGreaterThan(0.02);
  expect(latest.seam.mixed).toBeLessThan(0.01);
  expect(latest.seam.covered).toBeGreaterThan(0.7 * centres.seam.covered);
  // Where no date saw, Latest draws the inferred layer; Ground truth draws nothing.
  expect(gapLatest.southWest.green).toBeGreaterThan(0.6);
  expect(gapTruth.southWest.green).toBeLessThan(0.02);
  expect(gapTruth.southWest.covered).toBeLessThan(gapLatest.southWest.covered / 4);
  expect(gapTruth.right.blue).toBeGreaterThan(0.8);
}

test("under Latest each cell's pixels come from one date; Ground truth hides the inferred", async ({
  page,
}) => {
  // Five views on software GL, each waiting for three tilesets to refine: about seven minutes
  // here (2026-10-20), twice that allowed for a slower runner.
  test.setTimeout(840_000);
  expectOneOwnerPerCell(await measure(page, "cesium"));
});

test("under PlayCanvas too, one date a cell in one sort, the seam as clean", async ({ page }) => {
  // The same five views drawn by the app's PlayCanvas renderer, every date streamed into its
  // one app; software GL again.
  test.setTimeout(840_000);
  expectOneOwnerPerCell(await measure(page, "playcanvas"));
});

/** The tree with SH bands, and its streamed package beside its tileset (where PlayCanvas looks). */
const TREE = "synthetic-tree-sh/";
const TREE_NATIVE = `${TREE}sog/`;

const sogHarnessHtml = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Rule on a streamed SOG</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/ruleSogHarness.ts");
      window.__ruleSog = await harness.startRuleSogHarness({
        container: document.getElementById("viewer"),
        url: "/fixture-tiles/${TREE}tileset.json",
      });
    </script>
  </body>
</html>`;

test("under PlayCanvas, a colour-only refresh of a natively streamed SH scan keeps the seam", async ({
  page,
}) => {
  // PlayCanvas re-copies an SH splat's colour alone as the camera moves, and for a SOG reads its
  // geometry back from the work buffer: the rule's weight must see the same spread there.
  test.setTimeout(300_000);
  await page.setViewportSize({ width: 960, height: 640 });
  await page.route("**/fixture-tiles/**", (route) => {
    let relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    if (relative.startsWith(TREE_NATIVE))
      relative = `synthetic-tree-sh-sog/${relative.slice(TREE_NATIVE.length)}`;
    const file = resolve(TILES, relative);
    if (!existsSync(file)) return route.fulfill({ status: 404, body: "" });
    return route.fulfill({ status: 200, body: readFileSync(file) });
  });
  await page.route("**/rule-sog-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: sogHarnessHtml }),
  );
  await page.goto("/rule-sog-harness.html");
  await page.waitForFunction(() => "__ruleSog" in window, undefined, { timeout: 120_000 });
  interface Harness {
    status(): { native: boolean; gaussians: number };
    cover(): number;
    move(m: number): Promise<void>;
  }
  const status = await page.evaluate(() =>
    (window as unknown as { __ruleSog: Harness }).__ruleSog.status(),
  );
  const before = await page.evaluate(() =>
    (window as unknown as { __ruleSog: Harness }).__ruleSog.cover(),
  );
  await page.evaluate(() => (window as unknown as { __ruleSog: Harness }).__ruleSog.move(4));
  const after = await page.evaluate(() =>
    (window as unknown as { __ruleSog: Harness }).__ruleSog.cover(),
  );
  const path = EVIDENCE
    ? `${EVIDENCE}/scene-playcanvas-sog-refresh.png`
    : test.info().outputPath("sog.png");
  await page.screenshot({ path });
  const figures = JSON.stringify({ status, before, after });
  test.info().annotations.push({ type: "pixels", description: figures });
  if (EVIDENCE) writeFileSync(`${EVIDENCE}/scene-playcanvas-sog-refresh.json`, figures);
  expect(status.native).toBe(true);
  expect(before).toBeGreaterThan(2_000);
  // The same view before and after: the refresh changes colours, never which splats are drawn.
  expect(after).toBeGreaterThan(0.9 * before);
});
