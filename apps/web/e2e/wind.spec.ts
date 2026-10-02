/**
 * Wind on scene-object skins (step C1: `@twin/world` `skinWind.ts`, `cesium/skinWind.ts`) in a
 * real CesiumJS with the engine patch, on the committed synthetic yard.
 *
 * `data/tiles/synthetic-yard/skin/` holds the yard's tree (instance 1), a snag (9) and two
 * shrubs (10, 12) skinned, with their `dynamics`, and `materials.json`: the yard was segmented
 * with a stand-in embedder, so every instance reads `movable` and the priors drive none; the
 * records turn the wind on for 1, 9 and 10 and leave 12 on its prior. The route links
 * `instances.json`, `skin.json` and `materials.json` from the root's extras.
 *
 * What must hold: with wind on, the tree's pixels keep changing while its base, an unskinned
 * tree and the shrub the wind does not drive stay still; the motion is a function of the
 * harness clock (the same steps give the same frame); calm is the measured frame, pixel for
 * pixel. Frames go to `test.info().outputPath()` and, with `WIND_FRAMES_DIR`, there too.
 *
 * Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import { mkdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}
/** `src/dev/skinHarness.ts`, as the page exposes it (what this spec uses). */
interface SkinHarness {
  view(headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  frame(): number;
  difference(a: number, b: number, rect?: Rect, tol?: number): number;
  rectOf(id: number, grow?: number): Rect | null;
  windOn(strength: number, bearingDeg: number, t0?: number, seed?: number): Promise<void>;
  advance(seconds: number, fps?: number): Promise<void>;
  windOff(): Promise<void>;
  windSkins(): { instance: number; wind: boolean; stiffness: number; evidence: string }[];
  pointRect(instance: number, local: [number, number, number], radiusM: number): Rect | null;
}

const TILES = resolve(process.cwd(), "../../data/tiles");
const TILESET = "synthetic-yard/splat/tileset.json";
const INSTANCES = "../instances/instances.json";
const SKIN = "../skin/skin.json";
const MATERIALS = "../skin/materials.json";
/** The default strength of the wind control (`DEFAULT_WIND_STRENGTH`): 6.3 m/s. */
const STRENGTH = 0.1;

function harnessHtml(): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Wind harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/skinHarness.ts");
      window.__skin = await harness.startSkinHarness({
        container: document.getElementById("viewer"),
        url: "/fixture-tiles/${TILESET}",
        incremental: true,
        maximumScreenSpaceError: 1,
      });
    </script>
  </body>
</html>`;
}

async function open(page: Page, errors: string[]): Promise<void> {
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  const read = (relative: string): { instances?: unknown[]; skins?: unknown[] } =>
    JSON.parse(readFileSync(resolve(TILES, "synthetic-yard/splat", relative), "utf-8")) as {
      instances?: unknown[];
      skins?: unknown[];
    };
  const instances = read(INSTANCES).instances ?? [];
  const skins = read(SKIN).skins ?? [];
  await page.route("**/fixture-tiles/**", (route) => {
    const path = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    const relative = resolve("/", path).slice(1);
    if (relative !== path) return route.abort();
    if (relative === TILESET) {
      const tileset = JSON.parse(readFileSync(resolve(TILES, relative), "utf-8")) as {
        root: { extras?: Record<string, unknown> };
      };
      tileset.root.extras = {
        ...tileset.root.extras,
        instances: { uri: INSTANCES, count: instances.length },
        skin: { uri: SKIN, count: skins.length },
        materials: { uri: MATERIALS, count: 3 },
      };
      return route.fulfill({ status: 200, json: tileset });
    }
    const type = relative.endsWith(".json")
      ? "application/json"
      : relative.endsWith(".glb")
        ? "model/gltf-binary"
        : "application/octet-stream";
    return route.fulfill({
      status: 200,
      contentType: type,
      body: readFileSync(resolve(TILES, relative)),
    });
  });
  await page.route("**/wind-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml() }),
  );
  await page.goto("/wind-harness.html");
  await page.waitForFunction(() => "__skin" in window, undefined, { timeout: 120_000 });
}

function caller(page: Page) {
  return <K extends keyof SkinHarness>(
    method: K,
    ...args: Parameters<SkinHarness[K]>
  ): Promise<Awaited<ReturnType<SkinHarness[K]>>> =>
    page.evaluate(
      ([m, a]) => {
        const harness = (window as unknown as { __skin: Record<string, unknown> }).__skin;
        return (harness[m] as (...x: unknown[]) => unknown)(...a);
      },
      [method, args] as [string, unknown[]],
    ) as Promise<Awaited<ReturnType<SkinHarness[K]>>>;
}

// A smaller canvas: every frame here is drawn in software.
test.use({ viewport: { width: 960, height: 600 } });

const shaderErrors = (errors: string[]): string[] =>
  errors.filter((e) => /shader|compile|link|webgl/i.test(e));

test("the wind sways the skinned tree, its base and the rest stay still, calm is rest", async ({
  page,
}) => {
  test.setTimeout(1_500_000);
  const errors: string[] = [];
  await open(page, errors);
  const call = caller(page);
  const framesDir = process.env.WIND_FRAMES_DIR;
  if (framesDir) mkdirSync(framesDir, { recursive: true });
  const shot = async (name: string): Promise<void> => {
    // One screenshot each: a SwiftShader frame of the yard takes tens of seconds.
    await page.screenshot({
      path: framesDir ? resolve(framesDir, `${name}.png`) : test.info().outputPath(`${name}.png`),
    });
  };

  await call("view", 30, -35, 32);
  await call("view", 30, -35, 32);
  const rest = await call("frame");
  const restAgain = await call("frame");
  await shot("wind-rest");
  const tree = (await call("rectOf", 1, 1.2)) ?? undefined;
  // The trunk's foot: half a metre around the base, the lowest tenth of the tree.
  const base = (await call("pointRect", 1, [0, 0, 0.3], 0.45)) ?? undefined;
  const otherTree = (await call("rectOf", 5)) ?? undefined;
  const stillShrub = (await call("rectOf", 12, 0.8)) ?? undefined;

  await call("windOn", STRENGTH, 60, 1000);
  const skins = await call("windSkins");
  // Swing in, then three moments a little apart.
  await call("advance", 3);
  const a = await call("frame");
  await shot("wind-a");
  await call("advance", 0.8);
  const b = await call("frame");
  await shot("wind-b");
  await call("advance", 0.8);
  const c = await call("frame");
  await shot("wind-c");
  for (let k = 0; k < 3; k += 1) {
    await call("advance", 0.4);
    await shot(`wind-strip-${String(k)}`);
  }

  // The same steps from the same start: the same frame.
  await call("windOn", STRENGTH, 60, 1000);
  await call("advance", 3);
  const aAgain = await call("frame");

  await call("windOff");
  const calm = await call("frame");
  await shot("wind-calm");

  const measures = {
    skins,
    tree,
    base,
    restNoise: await call("difference", rest, restAgain),
    treeSways: await call("difference", rest, a, tree),
    treeAB: await call("difference", a, b, tree),
    treeBC: await call("difference", b, c, tree),
    baseAB: await call("difference", a, b, base),
    baseRest: await call("difference", rest, a, base),
    otherTree: await call("difference", rest, a, otherTree),
    stillShrub: await call("difference", rest, a, stillShrub),
    replay: await call("difference", a, aAgain, undefined, 0),
    calm: await call("difference", rest, calm, undefined, 0),
  };
  test.info().annotations.push({ type: "measures", description: JSON.stringify(measures) });
  console.info(JSON.stringify(measures));

  expect(shaderErrors(errors)).toEqual([]);
  expect(skins.filter((s) => s.wind).map((s) => s.instance)).toEqual([1, 9, 10]);
  expect(measures.restNoise).toBe(0);
  // The tree leans and keeps moving.
  expect(measures.treeSways).toBeGreaterThan(0.005);
  expect(measures.treeAB).toBeGreaterThan(0.003);
  expect(measures.treeBC).toBeGreaterThan(0.003);
  // Its base does not (a few edge pixels at most), and neither do the unskinned tree and the
  // shrub the wind does not drive.
  expect(measures.baseAB).toBeLessThan(0.15 * measures.treeAB);
  expect(measures.baseRest).toBeLessThan(0.15 * measures.treeSways);
  expect(measures.otherTree).toBeLessThan(0.002);
  expect(measures.stillShrub).toBeLessThan(0.002);
  // A function of the clock, and calm is the measured frame exactly.
  expect(measures.replay).toBe(0);
  expect(measures.calm).toBe(0);
});
