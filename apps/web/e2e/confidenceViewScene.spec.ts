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
 * The same grids drawn by PlayCanvas (the app's default renderer) and Spark, through the
 * shared cell look at one level per tile (scanView/cellModifier.ts): weak cells red-majority,
 * strong ones green, the split red left and green right -- and how many sampled splats read
 * the level CesiumJS would have given them. And the shared look itself run on the GPU against
 * its TypeScript twin, case for case (src/dev/cellLookGpuCheck.ts).
 *
 * Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

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
        renderer: new URLSearchParams(location.search).get("renderer") ?? "cesium",
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
  reading(): { tiles: number; steps: number; sampled: number; agree: number } | null;
}

/** Opens the harness drawn by `renderer`, collecting the page's errors into `errors`. */
async function openHarness(page: Page, renderer: string, errors: string[]): Promise<void> {
  await page.setViewportSize({ width: 960, height: 640 });
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/fixture-tiles/**", (route) => {
    const relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    // PlayCanvas asks whether the scan has a native package (sog/lod-meta.json): it has none.
    const file = resolve(TILES, relative);
    if (!existsSync(file)) return route.fulfill({ status: 404, body: "" });
    const type = relative.endsWith(".json")
      ? "application/json"
      : relative.endsWith(".bin")
        ? "application/octet-stream"
        : "model/gltf-binary";
    return route.fulfill({
      status: 200,
      contentType: type,
      body: readFileSync(file),
    });
  });
  await page.route(/\/cell-view-harness\.html/, (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: HARNESS_HTML }),
  );
  await page.goto(`/cell-view-harness.html?renderer=${renderer}`);
  await page.waitForFunction(() => "__cellView" in window, undefined, { timeout: 90_000 });
}

/** Draws `grid` in `mode` and counts the pixels; the screenshot goes with the evidence. */
async function viewOf(
  page: Page,
  prefix: string,
  grid: string,
  mode = "confidence",
): Promise<Counts> {
  const counts = await page.evaluate(
    ([g, m]) => (window as unknown as { __cellView: Harness }).__cellView.view(g, m),
    [grid, mode] as [string, string],
  );
  const name = `${prefix}${mode === "confidence" ? grid : `${grid}-${mode}`}`;
  const path = EVIDENCE ? `${EVIDENCE}/scene-${name}.png` : test.info().outputPath(`${name}.png`);
  await page.screenshot({ path });
  await test.info().attach(name, { path, contentType: "image/png" });
  return counts;
}

test("weak cells draw red, strong ones green, each splat by its own cell", async ({ page }) => {
  // Seven views on software GL, the first waiting for the yard to refine: minutes, not seconds.
  test.setTimeout(420_000);
  const errors: string[] = [];
  await openHarness(page, "cesium", errors);
  const view = (grid: string, mode?: string) => viewOf(page, "", grid, mode);

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

for (const renderer of ["playcanvas", "spark"] as const) {
  test(`under ${renderer}, weak cells draw red, strong ones green, a cell level a tile`, async ({
    page,
  }) => {
    test.setTimeout(480_000);
    const errors: string[] = [];
    await openHarness(page, renderer, errors);
    const view = (grid: string, mode?: string) => viewOf(page, `${renderer}-`, grid, mode);
    await view("off");
    const off = await view("off");
    const weak = await view("weak");
    const reading = await page.evaluate(() =>
      (window as unknown as { __cellView: Harness }).__cellView.reading(),
    );
    const strong = await view("strong");
    const split = await view("split");
    const age = await view("weak", "data-age");
    const newer = await view("split", "data-age");
    test.info().annotations.push({
      type: "pixels",
      description: JSON.stringify({ off, weak, strong, split, age, newer, reading }),
    });
    expect(errors.filter((e) => /shader|compile|link|uniform/i.test(e))).toEqual([]);
    // The look is on the renderer's tiles, and the level each reads agrees with CesiumJS's.
    expect(reading?.tiles ?? 0).toBeGreaterThan(0);
    expect(reading?.sampled ?? 0).toBeGreaterThan(0);
    expect((reading?.agree ?? 0) / Math.max(1, reading?.sampled ?? 0)).toBeGreaterThan(0.9);
    expect(off.coverage).toBeGreaterThan(0.05);
    expect(off.red).toBeLessThan(0.1);
    // Weak everywhere: red-majority of what the scan covers.
    expect(weak.red).toBeGreaterThan(0.6);
    expect(weak.green).toBeLessThan(0.05);
    expect(strong.green).toBeGreaterThan(0.6);
    expect(strong.red).toBeLessThan(0.05);
    expect(split.leftRed).toBeGreaterThan(0.5);
    expect(split.rightRed).toBeLessThan(0.1);
    expect(split.rightGreen).toBeGreaterThan(0.5);
    expect(age.ageHue).toBeGreaterThan(0.6);
    expect(newer.light).toBeGreaterThan(age.light + 0.15);
  });
}

test("the shared colour code draws on a GPU what its reference says, case for case", async ({
  page,
}) => {
  await page.route(/\/cell-look-check\.html/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/html",
      body: `<!doctype html><html><body><script type="module">
        const check = await import("/src/dev/cellLookGpuCheck.ts");
        window.__look = check.cellLookAgreement();
      </script></body></html>`,
    }),
  );
  await page.goto("/cell-look-check.html");
  await page.waitForFunction(() => "__look" in window, undefined, { timeout: 90_000 });
  const look = await page.evaluate(
    () =>
      (window as unknown as { __look: { cases: number; maxError: number; worst: unknown } }).__look,
  );
  test.info().annotations.push({ type: "look", description: JSON.stringify(look) });
  expect(look.cases).toBeGreaterThan(5000);
  // Bytes on the GPU, doubles here: within two steps of a byte, every colour and every cue.
  expect(look.maxError).toBeLessThanOrEqual(2 / 255);
});
