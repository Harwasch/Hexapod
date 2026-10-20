/**
 * The three splat renderers on the globe (cesium/scanView) agree on a scan wider than SPZ can
 * hold in one frame: the committed 10.4 km synthetic site (data/tiles/synthetic-far), whose
 * far tiles are written in frames of their own (translation-only tile transforms, nested) and
 * whose widest tiles keep 11 fractional bits. CesiumJS composes tile transforms natively; Spark
 * and PlayCanvas are handed each tile's offset by the shared tile tree (view/tiles.ts). Looking
 * at the north-east plot, 2.6 km east and 1.8 km north of the site's origin, all three must
 * draw it in the same place: a renderer that dropped a tile's frame would draw the plot
 * kilometres away and leave the screen empty.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");
const BACKGROUND = [0x10, 0x14, 0x1a];

const expected = JSON.parse(
  readFileSync(resolve(TILES, "synthetic-far/source/expected.json"), "utf8"),
) as { plots: Record<string, [number, number, number]> };

async function open(page: Page, target: [number, number, number]): Promise<void> {
  await page.route("**/fixture-tiles/**", (route) => {
    const relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    // No streamed PlayCanvas package for this scan: PlayCanvas streams the tiles.
    if (relative.includes("..") || relative.includes("/sog/")) {
      return route.fulfill({ status: 404, body: "" });
    }
    return route.fulfill({
      status: 200,
      contentType: relative.endsWith(".json") ? "application/json" : "model/gltf-binary",
      body: readFileSync(resolve(TILES, relative)),
    });
  });
  const html = `<!doctype html><html><head><style>
#v .cesium-widget, #v .cesium-widget > canvas:first-child { width: 100vw; height: 100vh; display: block; }
</style></head><body style="margin:0;background:#10141a">
<div id="v" style="position:relative;width:100vw;height:100vh"></div>
<script type="module">
const h = await import("/src/dev/scanRendererHarness.ts");
window.__scan = await h.startScanRendererHarness({ container: document.getElementById("v"),
  tilesetUrl: "/fixture-tiles/synthetic-far/splat/tileset.json", rangeM: 25,
  targetLocalM: ${JSON.stringify(target)} });
</script></body></html>`;
  await page.route("**/__scan-far", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: html }),
  );
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.goto("/__scan-far");
  await page.waitForFunction(() => "__scan" in window, undefined, { timeout: 180_000 });
}

/** Which of a coarse grid of screen cells hold scan (not background), from a screenshot. */
async function coverage(page: Page): Promise<boolean[]> {
  const png = (await page.screenshot()).toString("base64");
  return page.evaluate(
    async ([data, background]) => {
      const image = new Image();
      image.src = `data:image/png;base64,${data}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = 64;
      canvas.height = 40;
      const context = canvas.getContext("2d");
      if (!context) return [];
      context.drawImage(image, 0, 0, 64, 40);
      const pixels = context.getImageData(0, 0, 64, 40).data;
      const cells: boolean[] = [];
      for (let i = 0; i < 64 * 40; i++) {
        const d =
          Math.abs((pixels[i * 4] ?? 0) - (background[0] ?? 0)) +
          Math.abs((pixels[i * 4 + 1] ?? 0) - (background[1] ?? 0)) +
          Math.abs((pixels[i * 4 + 2] ?? 0) - (background[2] ?? 0));
        cells.push(d > 24);
      }
      return cells;
    },
    [png, BACKGROUND] as const,
  );
}

const share = (cells: boolean[]): number => cells.filter(Boolean).length / cells.length;
function overlap(a: boolean[], b: boolean[]): number {
  let both = 0;
  let either = 0;
  a.forEach((on, i) => {
    if (on && b[i]) both += 1;
    if (on || b[i]) either += 1;
  });
  return either > 0 ? both / either : 0;
}

test("CesiumJS, Spark and PlayCanvas draw a far tile's plot in the same place", async ({
  page,
}, testInfo) => {
  test.setTimeout(600_000);
  await page.setViewportSize({ width: 960, height: 600 });
  await open(page, expected.plots["north-east"] ?? [0, 0, 0]);
  const results: Record<string, unknown> = {};
  const masks: Record<string, boolean[]> = {};
  for (const kind of ["cesium", "spark", "playcanvas"] as const) {
    const status = await page.evaluate(`window.__scan.use(${JSON.stringify(kind)}, 120)`);
    await page.screenshot({ path: testInfo.outputPath(`far-${kind}.png`) });
    masks[kind] = await coverage(page);
    results[kind] = { status, coverage: share(masks[kind] ?? []) };
  }
  results.overlap = {
    spark: overlap(masks.cesium ?? [], masks.spark ?? []),
    playcanvas: overlap(masks.cesium ?? [], masks.playcanvas ?? []),
  };
  writeFileSync(testInfo.outputPath("far-renderers.json"), JSON.stringify(results, null, 1));
  console.info(JSON.stringify(results, null, 1));
  expect(share(masks.cesium ?? [])).toBeGreaterThan(0.02);
  for (const kind of ["spark", "playcanvas"] as const) {
    const status = results[kind] as {
      status: { tiles: number; native: boolean; error: string | null };
    };
    expect(status.status.error).toBeNull();
    expect(status.status.native).toBe(false);
    expect(status.status.tiles).toBeGreaterThan(0);
    expect((results.overlap as Record<string, number>)[kind]).toBeGreaterThan(0.5);
  }
});
