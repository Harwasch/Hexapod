/**
 * Two dates side by side under PlayCanvas (the app's default renderer) in the real PlayCanvas
 * the app runs, on the committed synthetic yard (`src/dev/compareHarness.ts`): the same yard
 * twice, June the scan PlayCanvas draws and March streamed beside it into its one app, each tile
 * coloured by the side of the divider its copy writes -- March red, June blue.
 *
 * - The cut switched off for one view: red and blue interleave on both sides (the dates overlap
 *   splat for splat) -- the check below means nothing without it.
 * - The divider at a half: left of it March's red, right of it June's blue, each at least 99% of
 *   the covered pixels outside a 2 px band either side of the cut, and no pixel of either date on
 *   the other's side at all, the columns next to the cut included (CesiumJS's split is exact
 *   there too).
 * - June alone before, and again once the comparison ends: drawn whole.
 * - Swiped to 0.3 and to 0.7 a step a frame: PlayCanvas copies nothing into its work buffer on
 *   the way (the divider is a uniform of the frame), and the picture is unchanged outside the
 *   columns the divider crossed.
 *
 * Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");
const EVIDENCE = process.env.PC4_EVIDENCE_DIR;

const harnessHtml = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Compare harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/compareHarness.ts");
      window.__compare = await harness.startCompareHarness({
        container: document.getElementById("viewer"),
        url: "/fixture-tiles/synthetic-yard/splat/tileset.json",
      });
    </script>
  </body>
</html>`;

interface Side {
  covered: number;
  red: number;
  blue: number;
}

interface Counts {
  dividerPx: number;
  width: number;
  left: Side;
  right: Side;
  tolerancePx: number;
  cut: { offset: number; covered: number; wrong: number }[];
  wrong: number;
  pixels?: string;
}

interface Harness {
  alone(): Promise<Counts>;
  uncut(): Promise<Counts>;
  compare(split: number): Promise<Counts>;
  swipe(split: number, steps: number): Promise<{ copies: number[]; counts: Counts }>;
  end(): Promise<Counts>;
}

test("under playcanvas, each date is drawn only on its side of the divider, cut exactly there", async ({
  page,
}) => {
  test.setTimeout(600_000);
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
  await page.route("**/compare-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml }),
  );
  await page.goto("/compare-harness.html");
  await page.waitForFunction(() => "__compare" in window, undefined, { timeout: 90_000 });

  const shot = async (name: string) => {
    const path = EVIDENCE ? `${EVIDENCE}/scene-${name}.png` : test.info().outputPath(`${name}.png`);
    await page.screenshot({ path });
    await test.info().attach(name, { path, contentType: "image/png" });
  };
  const harness = <K extends keyof Harness>(name: K, ...args: Parameters<Harness[K]>) =>
    page.evaluate(
      ([n, a]) => {
        const h = (window as unknown as { __compare: Record<string, (...x: unknown[]) => unknown> })
          .__compare;
        return h[n as string]?.(...(a as unknown[]));
      },
      [name, args] as const,
    ) as ReturnType<Harness[K]>;

  const alone = await harness("alone");
  await shot("playcanvas-june-alone");
  const half = await harness("compare", 0.5);
  await shot("playcanvas-compare-at-half");
  const uncut = await harness("uncut");
  const toLeft = await harness("swipe", 0.3, 10);
  await shot("playcanvas-compare-at-0.3");
  const toRight = await harness("swipe", 0.7, 20);
  await shot("playcanvas-compare-at-0.7");
  // Unchanged outside the columns the divider crossed (each image against the one at a half).
  const unchanged = await page.evaluate(
    ([a, b, c]) => {
      const h = (
        window as unknown as {
          __compare: { diff(a: unknown, b: unknown, x0: number, x1: number): number };
        }
      ).__compare;
      const band = (from: number, to: number) => [Math.floor(from) - 2, Math.ceil(to) + 2] as const;
      const [l0, l1] = band(b.dividerPx, a.dividerPx);
      const [r0, r1] = band(a.dividerPx, c.dividerPx);
      return {
        toLeft: h.diff(a, b, l0, l1),
        toRight: h.diff(a, c, r0, r1),
        pixels: (a.width * (a.pixels ? atob(a.pixels).length / 4 / a.width : 0)) | 0,
      };
    },
    [half, toLeft.counts, toRight.counts] as const,
  );
  const ended = await harness("end");
  await shot("playcanvas-compare-ended");
  const strip = (c: Counts) => ({ ...c, pixels: undefined });
  const figures = {
    alone,
    uncut,
    half: strip(half),
    toLeft: { copies: toLeft.copies, counts: strip(toLeft.counts) },
    toRight: { copies: toRight.copies, counts: strip(toRight.counts) },
    unchanged,
    ended,
    errors,
  };
  const json = JSON.stringify(figures);
  test.info().annotations.push({ type: "pixels", description: json });
  if (EVIDENCE) writeFileSync(`${EVIDENCE}/scene-compare.json`, JSON.stringify(figures, null, 2));
  console.info(json);

  expect(errors.filter((e) => /shader|compile|link|webgl/i.test(e))).toEqual([]);
  // June alone, before the comparison: drawn whole, across the yard.
  expect(alone.left.covered + alone.right.covered).toBeGreaterThan(150_000);
  // Without the cut both dates are drawn on both sides: the measure bites.
  expect(uncut.left.red).toBeLessThan(0.9);
  expect(uncut.right.blue).toBeLessThan(0.9);
  expect(uncut.wrong).toBeGreaterThan(1_000);
  for (const counts of [half, toLeft.counts, toRight.counts]) {
    expect(counts.left.covered).toBeGreaterThan(5_000);
    expect(counts.right.covered).toBeGreaterThan(5_000);
    expect(counts.left.red).toBeGreaterThanOrEqual(0.99);
    expect(counts.right.blue).toBeGreaterThanOrEqual(0.99);
    // Exact at the cut: no pixel of either date on the other's side, next to it or anywhere.
    expect(counts.wrong).toBe(0);
    expect(counts.cut.find((c) => c.offset === -1)?.covered ?? 0).toBeGreaterThan(0);
    expect(counts.cut.find((c) => c.offset === 0)?.covered ?? 0).toBeGreaterThan(0);
  }
  expect(half.dividerPx).toBeCloseTo(0.5 * half.width, 3);
  expect(toLeft.counts.dividerPx).toBeCloseTo(0.3 * half.width, 3);
  expect(toRight.counts.dividerPx).toBeCloseTo(0.7 * half.width, 3);
  // Swiping copies nothing into the work buffer, and changes nothing but the crossed columns.
  expect(toLeft.copies.every((c) => c === 0)).toBe(true);
  expect(toRight.copies.every((c) => c === 0)).toBe(true);
  expect(unchanged.toLeft).toBe(0);
  expect(unchanged.toRight).toBe(0);
  // Ended: June alone again, whole -- every tile's copy now writes no side, which cuts nothing.
  expect(ended.left.covered + ended.right.covered).toBeGreaterThan(150_000);
});
