/**
 * Scene-object skins in a real CesiumJS 1.145 with the engine patch's `vertexMotion` hook (as
 * a motion chain) and its optional Jacobian, on the committed synthetic yard.
 *
 * `data/tiles/synthetic-yard/skin/` is the yard's tree (instance 1), a snag (9) and two shrubs
 * (10, 12) skinned by `tools/captures/skin_scene.py ... --only 1,9,10,12`, beside
 * `instances/`. The committed tiles stay byte-identical; the route below links both files from
 * the root's extras, as `segment_scene.link_instances` and `skin_scene.link_skin` would.
 *
 * What must hold: the skin compiles into the chain with its Jacobian; driving the tree's
 * handles moves its pixels while an unskinned tree and a skinned-but-undriven shrub stay put;
 * at rest again the frame is the measured one, pixel for pixel; the constant handle lifts a
 * shrub whole; covariances follow the skin (a shrub scaled up stays filled); and a hidden
 * object stays hidden while it moves (the visibility chain sees the displaced splats).
 *
 * Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}
interface HandleMotion {
  handle: number;
  z: number[];
}
/** `src/dev/skinHarness.ts`, as the page exposes it. */
interface SkinHarness {
  view(headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  skins(): { id: number; instance: number; handles: number; scale: number }[];
  drive(instance: number, motions: HandleMotion[]): Promise<void>;
  rigid(instance: number, yawDeg: number, shift: [number, number, number]): Promise<void>;
  wobble(instance: number, amplitude: number, handles?: number): void;
  stop(): void;
  rest(): Promise<void>;
  covariance(on: boolean): Promise<void>;
  hide(ids: number[]): Promise<void>;
  /** Hides every object but `id` (and what is below it). */
  isolate(id: number): Promise<void>;
  frame(): number;
  difference(a: number, b: number, rect?: Rect, tol?: number): number;
  coverage(a: number, rect?: Rect): number;
  rectOf(id: number, grow?: number): Rect | null;
  hooks(): { motion: string[]; jacobian: string[]; skin: boolean; active: boolean };
}

const TILES = resolve(process.cwd(), "../../data/tiles");
const TILESET = "synthetic-yard/splat/tileset.json";
const INSTANCES = "../instances/instances.json";
const SKIN = "../skin/skin.json";

function harnessHtml(): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Skin harness</title>
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
  await page.route("**/skin-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml() }),
  );
  await page.goto("/skin-harness.html");
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

const shaderErrors = (errors: string[]): string[] =>
  errors.filter((e) => /shader|compile|link|webgl/i.test(e));

/** A pure translation `t` as a handle's `Z`. */
const shift = (x: number, y: number, z: number): number[] => [0, 0, 0, x, 0, 0, 0, y, 0, 0, 0, z];

test("a driven object moves, the others stay put, and rest is the measured frame", async ({
  page,
}) => {
  test.setTimeout(600_000);
  const errors: string[] = [];
  await open(page, errors);
  const call = caller(page);

  await call("view", 30, -50, 45);
  await call("view", 30, -50, 45);
  const skins = await call("skins");
  const rest = await call("frame");
  const restAgain = await call("frame");
  await page.screenshot({ path: test.info().outputPath("rest.png") });
  const tree = (await call("rectOf", 1, 1.3)) ?? undefined;
  const otherTree = (await call("rectOf", 3)) ?? undefined;
  const shrub = (await call("rectOf", 10)) ?? undefined;

  // The tree's first two modes: a metre and a half east, a metre north, where each acts fully.
  await call("drive", 1, [
    { handle: 1, z: shift(1.5, 0, 0) },
    { handle: 2, z: shift(0, 1.0, 0) },
  ]);
  const driven = await call("frame");
  const hooks = await call("hooks");
  await page.screenshot({ path: test.info().outputPath("driven.png") });

  // Hidden while it moves: the visibility chain sees the displaced splats.
  // Hidden and moving, then hidden at rest: the same frame.
  await call("hide", [1]);
  const hidden = await call("frame");
  await call("rest");
  const hiddenAtRest = await call("frame");
  await call("hide", []);
  const calm = await call("frame");

  // The constant handle: the shrub lifted two metres, whole.
  await call("rigid", 10, 0, [0, 0, 2]);
  const lifted = await call("frame");
  await page.screenshot({ path: test.info().outputPath("lifted.png") });
  await call("rest");

  // Covariances follow the skin: the shrub scaled 2.5x about its base (Z_0 = [1.5·I | 0]),
  // with and without.
  // Everything else hidden, so what is counted is the shrub's own silhouette.
  const scale = 1.5;
  const grown = [scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0];
  await call("isolate", 10);
  const alone = await call("frame");
  await call("drive", 10, [{ handle: 0, z: grown }]);
  const big = await call("frame");
  await page.screenshot({ path: test.info().outputPath("scaled-covariance.png") });
  await call("covariance", false);
  const bigThin = await call("frame");
  await page.screenshot({ path: test.info().outputPath("scaled-no-covariance.png") });
  await call("covariance", true);
  await call("rest");
  await call("hide", []);
  const shrubBig = (await call("rectOf", 10, 2.6)) ?? undefined;

  const measures = {
    skins,
    hooks,
    tree,
    otherTree,
    shrub,
    restNoise: await call("difference", rest, restAgain),
    treeMoved: await call("difference", rest, driven, tree),
    otherTreeMoved: await call("difference", rest, driven, otherTree),
    shrubMoved: await call("difference", rest, driven, shrub),
    calmVsRest: await call("difference", rest, calm),
    shrubLifted: await call("difference", rest, lifted, shrub),
    treeUnderLift: await call("difference", rest, lifted, tree),
    hiddenVsDriven: await call("difference", driven, hidden, tree),
    hiddenMovingVsAtRest: await call("difference", hidden, hiddenAtRest),
    bigCoverage: await call("coverage", big, shrubBig),
    bigThinCoverage: await call("coverage", bigThin, shrubBig),
    restShrubCoverage: await call("coverage", alone, shrubBig),
  };
  test.info().annotations.push({ type: "measures", description: JSON.stringify(measures) });
  console.info(JSON.stringify(measures));

  expect(shaderErrors(errors)).toEqual([]);
  expect(hooks.motion).toEqual(["splatSkinMotion"]);
  expect(hooks.jacobian).toEqual(["splatSkinJacobian"]);
  expect(hooks.skin).toBe(true);
  expect(hooks.active).toBe(true);
  expect(skins.map((s) => s.instance)).toEqual([1, 9, 10, 12]);
  // Rendering is deterministic at rest.
  expect(measures.restNoise).toBeLessThan(0.001);
  // The driven tree moves...
  expect(measures.treeMoved).toBeGreaterThan(0.03);
  // ...an unskinned tree and a skinned shrub that is not driven do not.
  expect(measures.otherTreeMoved).toBeLessThan(0.01);
  expect(measures.shrubMoved).toBeLessThan(0.01);
  // Hidden, it is gone wherever it moved to.
  expect(measures.hiddenVsDriven).toBeGreaterThan(0.05);
  expect(measures.hiddenMovingVsAtRest).toBeLessThan(0.001);
  // At rest again, the measured frame.
  expect(measures.calmVsRest).toBeLessThan(0.001);
  // The constant handle moves the shrub and nothing else.
  expect(measures.shrubLifted).toBeGreaterThan(0.05);
  expect(measures.treeUnderLift).toBeLessThan(0.01);
  // Scaled up with covariances following, the shrub stays filled; without, it thins out.
  expect(measures.bigCoverage).toBeGreaterThan(measures.restShrubCoverage);
  expect(measures.bigCoverage).toBeGreaterThan(measures.bigThinCoverage * 1.1);
});

test("a wobbling tree, for the record", async ({ page }) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  await open(page, errors);
  const call = caller(page);
  await call("view", 30, -35, 30);
  await call("view", 30, -35, 30);
  const rest = await call("frame");
  await page.screenshot({ path: test.info().outputPath("wobble-0.png") });
  await call("wobble", 1, 0.8, 3);
  const changes: number[] = [];
  for (let k = 1; k <= 3; k += 1) {
    await page.waitForTimeout(700);
    await page.screenshot({ path: test.info().outputPath(`wobble-${String(k)}.png`) });
    changes.push(await call("difference", rest, await call("frame")));
  }
  await call("stop");
  test.info().annotations.push({ type: "measures", description: JSON.stringify({ changes }) });
  expect(shaderErrors(errors)).toEqual([]);
  expect(Math.max(...changes)).toBeGreaterThan(0.01);
});
