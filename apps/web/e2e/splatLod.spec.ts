/**
 * A merged-parent REPLACE splat tileset in a real CesiumJS 1.145, from far, middle and near.
 *
 * `data/tiles/synthetic-tree-lod` is the committed synthetic tree packed by
 * tools/captures/splat_tiles.py at 1,500 gaussians a tile: a 114-gaussian merged root, merged
 * octants below it, and leaves holding the tree's 12,000 gaussians once. The same tree as one
 * tile (`data/tiles/synthetic-tree/splat`) is what "no holes" is measured against: at each
 * distance, the level-of-detail view must cover the canvas about as much as every gaussian
 * does. Thinned parents failed exactly this: one unenlarged gaussian per cell left the gaps
 * between them see-through from afar.
 *
 * Headless GL here is SwiftShader, so the screenshots (attached to the report) are a record
 * that the path runs, not a judgement of how it looks; coverage is counted, not eyeballed.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");

const HARNESS_HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Splat LoD harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/splatLodHarness.ts");
      window.__splatLod = await harness.startSplatLodHarness({
        container: document.getElementById("viewer"),
        lodUrl: "/fixture-tiles/synthetic-tree-lod/tileset.json",
        fullUrl: "/fixture-tiles/synthetic-tree/splat/tileset.json",
      });
    </script>
  </body>
</html>`;

interface View {
  tiles: string[];
  gaussians: number;
  coverage: number;
}

async function openHarness(page: Page): Promise<void> {
  await page.route("**/fixture-tiles/**", (route) => {
    const relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    return route.fulfill({
      status: 200,
      contentType: relative.endsWith(".json") ? "application/json" : "model/gltf-binary",
      body: readFileSync(resolve(TILES, relative)),
    });
  });
  await page.route("**/__splat-lod", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: HARNESS_HTML }),
  );
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.goto("/__splat-lod");
  await page.waitForFunction(() => "__splatLod" in window, undefined, { timeout: 60_000 });
}

async function view(page: Page, range: number, full: boolean): Promise<View> {
  return (await page.evaluate(
    async ([r, f]) => {
      const harness = (window as unknown as { __splatLod: Record<string, unknown> }).__splatLod;
      return await (harness.view as (range: number, full: boolean) => Promise<unknown>)(r, f);
    },
    [range, full] as const,
  )) as View;
}

test("a merged-parent REPLACE tileset has no holes at 50, 20 and 5 m", async ({
  page,
}, testInfo) => {
  test.setTimeout(300_000);
  await openHarness(page);
  const leaves = JSON.parse(
    readFileSync(resolve(TILES, "synthetic-tree-lod/tileset.json"), "utf8"),
  ) as { root: unknown };
  const errors = new Map<string, number>();
  const walk = (tile: {
    content: { uri: string };
    geometricError: number;
    children?: unknown[];
  }): void => {
    errors.set(tile.content.uri, tile.geometricError);
    (tile.children as (typeof tile)[] | undefined)?.forEach(walk);
  };
  walk(leaves.root as Parameters<typeof walk>[0]);

  const results: Record<string, { lod: View; full: View }> = {};
  for (const range of [50, 20, 5]) {
    const lod = await view(page, range, false);
    await page.screenshot({ path: testInfo.outputPath(`replace-${String(range)}m.png`) });
    const full = await view(page, range, true);
    await page.screenshot({ path: testInfo.outputPath(`full-${String(range)}m.png`) });
    results[String(range)] = { lod, full };
    // Covered where the full splat is: within 10% of its coverage, never a fraction of it.
    expect(full.coverage).toBeGreaterThan(0.001);
    expect(lod.coverage / full.coverage, `at ${String(range)} m`).toBeGreaterThan(0.9);
  }
  writeFileSync(testInfo.outputPath("views.json"), JSON.stringify(results, null, 1));

  // REPLACE, and refined by distance: far away mostly merged parents, near the leaves; no
  // parent ever drawn with one of its own descendants. (At 50 m in a 1280x720 view the root
  // has already given way to its children -- merged octants, and one packed leaf.)
  const far = results["50"]?.lod.tiles ?? [];
  expect(far.filter((uri) => (errors.get(uri) ?? 0) > 0).length).toBeGreaterThan(far.length / 2);
  const near = results["5"]?.lod.tiles ?? [];
  expect(near.length).toBeGreaterThan(1);
  expect(near.some((uri) => errors.get(uri) === 0)).toBe(true);
  for (const [range, { lod }] of Object.entries(results)) {
    for (const uri of lod.tiles) {
      const stem = uri.replace(/\.glb$/, "");
      const ancestorDrawn = lod.tiles.some(
        (other) => other !== uri && stem.startsWith(other.replace(/\.glb$/, "") + "-"),
      );
      expect(ancestorDrawn, `${uri} drawn with an ancestor at ${range} m`).toBe(false);
      if (uri !== "splat.glb") expect(lod.tiles, `root with ${uri}`).not.toContain("splat.glb");
    }
  }
  // Fewer gaussians far away than near: the merged levels are what distance buys.
  expect(results["50"]?.lod.gaussians).toBeLessThan(results["5"]?.lod.gaussians ?? 0);
});
