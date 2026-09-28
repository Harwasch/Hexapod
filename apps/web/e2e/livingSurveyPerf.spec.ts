/**
 * Frame cost of the Living Survey's two motion paths, CPU and GPU, in a real CesiumJS.
 *
 * Opt-in (`LIVING_PERF=1`): it is a measurement, not a check, and under SwiftShader a frame of a
 * two-million-splat view takes seconds. `LIVING_PERF_TILES` may name a directory holding a large
 * level-of-detail `tileset.json` and its stamped `rig.json` (tools/captures: `synthetic_tree.py
 * --splats 2000000`, then `splat_tiles.py`, then `rig_tiles.py`); the committed synthetic tree
 * is always measured.
 *
 * For each tileset, path and motion model (the legacy model, and Living Mode where the rig
 * points at a `motion.json`): frames at calm (the deformer writes nothing — the scene's own
 * cost), then frames of wind. The first frame of wind is reported on its own: under Living Mode
 * the GPU path uploads the leaf-flutter texture then, once. `applyMs` is the motion path's CPU cost per frame, upload submit
 * included; `frameMs` is apply plus a rendered frame. **SwiftShader rasterises on the CPU**, so
 * `frameMs` here is dominated by software rendering and says little about a GPU; `applyMs` is
 * the number that transfers, and the GPU path's GPU-side cost is exactly what this cannot see.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");
const PERF_TILES = process.env.LIVING_PERF_TILES;

interface Case {
  readonly name: string;
  readonly tileset: string;
  readonly rig: string;
  readonly mse: number;
  readonly rangeM: number;
  /** Small for the large case: under SwiftShader, fill rate is what a frame of 2M splats costs. */
  readonly viewport: { width: number; height: number };
  readonly frames: number;
  /** The rig has a motion sidecar beside it, so Living Mode can be measured too. */
  readonly sidecar: boolean;
}

const CASES: Case[] = [
  {
    name: "synthetic-tree (1 tile, 12k)",
    tileset: "/perf/data/synthetic-tree/splat/tileset.json",
    rig: "/perf/data/synthetic-tree/source/rig.json",
    mse: 1,
    rangeM: 20,
    viewport: { width: 960, height: 600 },
    frames: 20,
    sidecar: true,
  },
  {
    name: "synthetic-tree-lod (REPLACE, 12k leaves)",
    tileset: "/perf/data/synthetic-tree-lod/tileset.json",
    rig: "/perf/data/synthetic-tree-lod/rig.json",
    mse: 16,
    rangeM: 6,
    viewport: { width: 960, height: 600 },
    frames: 20,
    sidecar: true,
  },
  ...(PERF_TILES !== undefined && existsSync(resolve(PERF_TILES, "tileset.json"))
    ? [
        {
          name: `large LOD (${PERF_TILES})`,
          tileset: "/perf/large/tileset.json",
          rig: "/perf/large/rig.json",
          // Fine enough that a near view selects most of the leaves: a million splats or more.
          mse: 2,
          rangeM: 6,
          viewport: { width: 320, height: 200 },
          frames: 2,
          sidecar: existsSync(resolve(PERF_TILES, "motion.json")),
        },
      ]
    : []),
];

function html(c: Case, gpu: boolean, living: boolean): string {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8" /><title>perf</title>
<style>html,body{margin:0;height:100%;background:#10141a}#viewer,#viewer .cesium-widget,#viewer canvas{width:100vw;height:100vh;display:block}</style>
</head><body><div id="viewer"></div><script type="module">
const harness = await import("/src/dev/livingSurveyHarness.ts");
window.__living = await harness.startLivingSurveyHarness({
  container: document.getElementById("viewer"),
  tilesetUrl: ${JSON.stringify(c.tileset)}, rigUrl: ${JSON.stringify(c.rig)},
  maximumScreenSpaceError: ${String(c.mse)}, gpu: ${String(gpu)}, living: ${String(living)},
});
</script></body></html>`;
}

async function open(page: Page, c: Case, gpu: boolean, living: boolean): Promise<void> {
  await page.route("**/perf/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    const [root, relative] = path.startsWith("/perf/data/")
      ? [TILES, path.slice("/perf/data/".length)]
      : [PERF_TILES ?? "", path.slice("/perf/large/".length)];
    if (relative.includes("..")) return route.abort();
    return route.fulfill({
      status: 200,
      contentType: relative.endsWith(".json") ? "application/json" : "model/gltf-binary",
      body: readFileSync(resolve(root, relative)),
    });
  });
  await page.route("**/__living-perf", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: html(c, gpu, living) }),
  );
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.goto("/__living-perf");
  await page.waitForFunction(() => "__living" in window, undefined, { timeout: 120_000 });
}

async function call<T>(page: Page, method: string, ...args: unknown[]): Promise<T> {
  return (await page.evaluate(
    async ([name, rest]) => {
      const harness = (
        window as unknown as { __living: Record<string, (...a: unknown[]) => unknown> }
      ).__living;
      const fn = harness[name];
      if (fn === undefined) throw new Error(`no harness method ${name}`);
      return await fn(...rest);
    },
    [method, args] as const,
  )) as T;
}

test.skip(process.env.LIVING_PERF !== "1", "a measurement: run with LIVING_PERF=1");

for (const c of CASES) {
  for (const [gpu, living] of [
    [false, false],
    [true, false],
    ...(c.sidecar
      ? [
          [false, true],
          [true, true],
        ]
      : []),
  ] as const) {
    const model = living ? "Living Mode" : "legacy model";
    test(`${c.name}, ${gpu ? "GPU" : "CPU"} path, ${model}`, async ({ page }, testInfo) => {
      test.setTimeout(3_600_000);
      await page.setViewportSize(c.viewport);
      await open(page, c, gpu, living);
      const settled = await call<Record<string, unknown>>(
        page,
        "view",
        c.rangeM,
        0,
        { strength: 0, bearingDeg: 250 },
        3_000_000,
      );
      expect(settled.phase).toBe("ready");
      const calm = await call<Record<string, unknown>>(page, "measure", c.frames, 1, {
        strength: 0,
        bearingDeg: 250,
      });
      // The first frame of wind: on the GPU path under Living Mode, the leaf-flutter texture
      // goes up here, once.
      const firstWind = await call<{ mean: number; max: number }>(page, "measureApply", 1, 1.9, {
        strength: 0.1,
        bearingDeg: 250,
      });
      const firstWindWords = (await call<Record<string, unknown>>(page, "status")).lastUploadWords;
      const wind = await call<Record<string, unknown>>(page, "measure", c.frames, 2, {
        strength: 0.1,
        bearingDeg: 250,
      });
      const applyOnly = await call<{ mean: number; max: number }>(page, "measureApply", 10, 3, {
        strength: 0.1,
        bearingDeg: 250,
      });
      const sort = await call<{ numSplats: number; ms: number[] }>(page, "sortMs", 3);
      const result = {
        case: c.name,
        path: gpu ? "gpu" : "cpu",
        model: living ? "living" : "legacy",
        tiles: settled.tiles,
        numSplats: settled.numSplats,
        lastDeriveMs: settled.lastDeriveMs,
        calm,
        firstWind: { ms: firstWind.mean, uploadWords: firstWindWords },
        wind,
        applyOnly,
        sort,
      };
      writeFileSync(testInfo.outputPath("timing.json"), JSON.stringify(result, null, 1));
      console.info(`LIVING_PERF ${JSON.stringify(result)}`);
      expect(wind.motion).toBe(gpu ? "gpu" : "cpu");
    });
  }
}
