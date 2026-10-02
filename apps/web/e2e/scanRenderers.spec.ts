/**
 * The three splat renderers side by side (cesium/scanView): CesiumJS, Spark and PlayCanvas
 * draw the yard fixture from the same camera, and the dedicated renderers must cover the
 * screen where CesiumJS does -- their camera is Cesium's, converted into the scan's own
 * frame each frame, so a wrong frame or axis shows here as a scan drawn somewhere else.
 *
 * The `@webgpu` tests do the same for the PlayCanvas WebGPU trial (docs/WEBGPU_TRIAL.md), in
 * the Playwright project with software WebGPU: it must come up on WebGPU, not its WebGL2
 * fallback, and cover the screen where CesiumJS and PlayCanvas on WebGL2 do.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

import { webgpuAdapter } from "./webgpu";

const TILES = resolve(process.cwd(), "../../data/tiles");
const BACKGROUND = [0x10, 0x14, 0x1a];

/** The yard's PlayCanvas streamed level of detail (splat-transform's lod-meta.json), served
 *  where a scan's package keeps it: beside the tileset, in `sog/`. */
const NATIVE = "synthetic-yard/splat/sog/";

async function open(page: Page, options: { native: boolean } = { native: true }): Promise<void> {
  await page.route("**/fixture-tiles/**", (route) => {
    let relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    if (relative.startsWith(NATIVE)) {
      if (!options.native) return route.fulfill({ status: 404, body: "" });
      relative = `synthetic-yard-sog/${relative.slice(NATIVE.length)}`;
    }
    const contentType = relative.endsWith(".json")
      ? "application/json"
      : relative.endsWith(".webp")
        ? "image/webp"
        : "model/gltf-binary";
    return route.fulfill({
      status: 200,
      contentType,
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
  tilesetUrl: "/fixture-tiles/synthetic-yard/splat/tileset.json", rangeM: 45 });
</script></body></html>`;
  await page.route("**/__scan-renderers", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: html }),
  );
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.goto("/__scan-renderers");
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

test("Spark and PlayCanvas draw the scan where CesiumJS does", async ({ page }, testInfo) => {
  test.setTimeout(600_000);
  await page.setViewportSize({ width: 960, height: 600 });
  await open(page);
  const results: Record<string, unknown> = {};
  const masks: Record<string, boolean[]> = {};
  for (const kind of ["cesium", "spark", "playcanvas"] as const) {
    const status = await page.evaluate(`window.__scan.use(${JSON.stringify(kind)}, 90)`);
    await page.screenshot({ path: testInfo.outputPath(`${kind}.png`) });
    masks[kind] = await coverage(page);
    results[kind] = { status, coverage: share(masks[kind] ?? []) };
  }
  results.overlap = {
    spark: overlap(masks.cesium ?? [], masks.spark ?? []),
    playcanvas: overlap(masks.cesium ?? [], masks.playcanvas ?? []),
  };
  writeFileSync(testInfo.outputPath("renderers.json"), JSON.stringify(results, null, 1));
  console.info(JSON.stringify(results, null, 1));
  const cesium = share(masks.cesium ?? []);
  expect(cesium).toBeGreaterThan(0.02);
  for (const kind of ["spark", "playcanvas"] as const) {
    const status = results[kind] as {
      status: { tiles: number; native: boolean; error: string | null };
    };
    expect(status.status.error).toBeNull();
    // PlayCanvas streams the yard's own streamed package; Spark streams the tiles.
    expect(status.status.native).toBe(kind === "playcanvas");
    if (kind === "spark") expect(status.status.tiles).toBeGreaterThan(0);
    expect(share(masks[kind] ?? [])).toBeGreaterThan(cesium * 0.5);
    expect((results.overlap as Record<string, number>)[kind]).toBeGreaterThan(0.6);
  }
});

test("PlayCanvas streams the tiles for a scan with no streamed package", async ({ page }) => {
  test.setTimeout(300_000);
  await page.setViewportSize({ width: 960, height: 600 });
  await open(page, { native: false });
  const status: { native: boolean; tiles: number; error: string | null } = await page.evaluate(
    `window.__scan.use("playcanvas", 90)`,
  );
  expect(status.error).toBeNull();
  expect(status.native).toBe(false);
  expect(status.tiles).toBeGreaterThan(0);
  expect(share(await coverage(page))).toBeGreaterThan(0.1);
});

interface WebgpuStatus {
  kind: string;
  api: string | null;
  notice: string | null;
  native: boolean;
  tiles: number;
  frames: number;
  error: string | null;
}

for (const native of [true, false]) {
  test(
    `PlayCanvas on WebGPU draws the scan where CesiumJS and WebGL2 do (${native ? "streamed package" : "tiles"})`,
    { tag: "@webgpu" },
    async ({ page }, testInfo) => {
      test.setTimeout(600_000);
      await page.setViewportSize({ width: 960, height: 600 });
      const errors: string[] = [];
      page.on("console", (message) => {
        if (message.type() === "error") errors.push(message.text());
      });
      page.on("pageerror", (error) => errors.push(error.message));
      await open(page, { native });
      const adapter = await webgpuAdapter(page);
      test.skip(adapter === null, "this browser has no WebGPU adapter");
      const masks: Record<string, boolean[]> = {};
      const statuses: Record<string, WebgpuStatus> = {};
      for (const kind of ["cesium", "playcanvas", "playcanvas-webgpu"] as const) {
        statuses[kind] = await page.evaluate(
          `window.__scan.use(${JSON.stringify(kind)}, ${native ? 40 : 90})`,
        );
        await page.screenshot({ path: testInfo.outputPath(`${kind}.png`) });
        masks[kind] = await coverage(page);
      }
      const webgpu = statuses["playcanvas-webgpu"];
      const results = {
        adapter,
        statuses,
        coverage: Object.fromEntries(Object.entries(masks).map(([k, m]) => [k, share(m)])),
        overlap: {
          cesium: overlap(masks.cesium ?? [], masks["playcanvas-webgpu"] ?? []),
          webgl2: overlap(masks.playcanvas ?? [], masks["playcanvas-webgpu"] ?? []),
        },
        errors,
      };
      writeFileSync(testInfo.outputPath("webgpu.json"), JSON.stringify(results, null, 1));
      console.info(JSON.stringify(results, null, 1));
      expect(webgpu?.error).toBeNull();
      expect(webgpu?.api).toBe("webgpu");
      expect(webgpu?.notice).toBeNull();
      expect(webgpu?.native).toBe(native);
      if (!native) expect(webgpu?.tiles).toBeGreaterThan(0);
      expect(errors.filter((e) => /wgsl|webgpu|shader|pipeline/i.test(e))).toEqual([]);
      const cesium = share(masks.cesium ?? []);
      expect(cesium).toBeGreaterThan(0.02);
      expect(share(masks["playcanvas-webgpu"] ?? [])).toBeGreaterThan(cesium * 0.5);
      expect(results.overlap.cesium).toBeGreaterThan(0.6);
      // The same engine on the other API, from the same camera: nearly the same cells.
      expect(results.overlap.webgl2).toBeGreaterThan(0.8);
    },
  );
}
