/**
 * Scene objects moving under the dedicated splat renderers (cesium/scanView: scanMotion.ts,
 * scanObjects.ts) -- the renderer variants of e2e/skin.spec.ts, wind.spec.ts and
 * telemetry.spec.ts, on the same committed synthetic yard and harness (`src/dev/skinHarness.ts`
 * with `renderer`), plus split objects drawn at their poses.
 *
 * Under PlayCanvas and Spark, as under CesiumJS: a driven skin moves its object's pixels while
 * an unskinned tree and an undriven shrub stay put, rest is the measured frame, the constant
 * handle lifts a shrub whole, a hidden object stays hidden while it moves, and a shrub scaled
 * up stays filled only with the covariance following; the wind sways the tree while its base
 * and the rest stay still, the same clock steps give the same frame and calm is the measured
 * frame exactly; telemetry moves the bound building along its path while unbound objects stay
 * still, and a silent source fades it back to the measured frame exactly; a split object is
 * drawn where its pose puts it and back where it was at rest. A scan PlayCanvas streams from
 * its own package carries no tile checksums: the store says the renderer cannot move it.
 *
 * Frames go to `test.info().outputPath()` and, with `MOTION_FRAMES_DIR`, there too.
 * Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import { existsSync, mkdirSync, readFileSync } from "node:fs";
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
interface Status {
  kind: string;
  active: boolean;
  tiles: number;
  native: boolean;
  error: string | null;
  instances: { tiles: number; matched: number } | null;
  motion: { updates: number; skinned: number; redrawn: number } | null;
  objects: number;
}
interface TelemetryStatus {
  instance: number;
  state: string;
  via: string;
  position: number[] | null;
}
/** `src/dev/skinHarness.ts`, as the page exposes it (what this spec uses). */
interface SkinHarness {
  view(headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  drive(instance: number, motions: HandleMotion[]): Promise<void>;
  rigid(instance: number, yawDeg: number, shift: [number, number, number]): Promise<void>;
  rest(): Promise<void>;
  covariance(on: boolean): Promise<void>;
  hide(ids: number[]): Promise<void>;
  isolate(id: number): Promise<void>;
  frame(): number;
  difference(a: number, b: number, rect?: Rect, tol?: number): number;
  maxDifference(a: number, b: number, rect?: Rect): number;
  coverage(a: number, rect?: Rect): number;
  rectOf(id: number, grow?: number): Rect | null;
  windOn(strength: number, bearingDeg: number, t0?: number, seed?: number): Promise<void>;
  advance(seconds: number, fps?: number): Promise<void>;
  windOff(): Promise<void>;
  windSkins(): { instance: number; wind: boolean }[];
  pointRect(instance: number, local: [number, number, number], radiusM: number): Rect | null;
  scanRect(point: [number, number, number], radiusM: number): Rect | null;
  telemetryReady(): Promise<void>;
  telemetryAt(ms: number): Promise<void>;
  telemetryStatus(): TelemetryStatus[];
  mute(sourceId: string, muted: boolean): void;
  rendererStatus(): Status | null;
  objectPose(
    instance: number,
    pose: { translation: number[]; rotation: number[] } | null,
  ): Promise<void>;
  motionGap(): { renderer: string; reason: string } | null;
}

const TILES = resolve(process.cwd(), "../../data/tiles");
const TILESET = "synthetic-yard/splat/tileset.json";
const INSTANCES = "../instances/instances.json";
const SKIN = "../skin/skin.json";
const MATERIALS = "../skin/materials.json";
const TELEMETRY = "../telemetry/telemetry.json";
/** The yard's PlayCanvas streamed package, served beside the tileset as a scan keeps it. */
const NATIVE = "synthetic-yard/splat/sog/";
/** A split object made at request time from one of the yard's leaf tiles (scan frame,
 *  origin at the scan's origin): a copy of what that tile draws, which a pose moves. */
const OBJECT_TILE = "splat_0-02.glb";
const OBJECT_CENTRE: [number, number, number] = [4.04, 8.09, 3.39];
const OBJECT_INSTANCE = 8;

type Renderer = "playcanvas" | "spark";

interface Features {
  telemetry?: boolean;
  materials?: boolean;
  objects?: boolean;
  native?: boolean;
}

function harnessHtml(renderer: Renderer): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Motion renderers harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer { position: relative; width: 100vw; height: 100vh; }
      #viewer .cesium-widget, #viewer .cesium-widget > canvas:first-child { width: 100vw; height: 100vh; display: block; }
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
        renderer: ${JSON.stringify(renderer)},
      });
    </script>
  </body>
</html>`;
}

async function open(
  page: Page,
  renderer: Renderer,
  features: Features,
  errors: string[],
): Promise<void> {
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  const read = (relative: string): Record<string, unknown[] | undefined> =>
    JSON.parse(readFileSync(resolve(TILES, "synthetic-yard/splat", relative), "utf-8")) as Record<
      string,
      unknown[] | undefined
    >;
  const instances = read(INSTANCES).instances ?? [];
  const skins = read(SKIN).skins ?? [];
  const bindings = read(TELEMETRY).bindings ?? [];
  const scan = JSON.parse(readFileSync(resolve(TILES, TILESET), "utf-8")) as {
    root: { transform: number[]; extras?: Record<string, unknown> };
  };
  await page.route("**/fixture-tiles/**", (route) => {
    const path = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    let relative = resolve("/", path).slice(1);
    if (relative !== path) return route.abort();
    if (relative === TILESET) {
      const tileset = JSON.parse(JSON.stringify(scan)) as typeof scan;
      tileset.root.extras = {
        ...tileset.root.extras,
        instances: { uri: INSTANCES, count: instances.length },
        skin: { uri: SKIN, count: skins.length },
        ...(features.materials ? { materials: { uri: MATERIALS, count: 3 } } : {}),
        ...(features.telemetry ? { telemetry: { uri: TELEMETRY, count: bindings.length } } : {}),
        ...(features.objects
          ? {
              objects: [
                {
                  uri: `objects/${String(OBJECT_INSTANCE)}/tileset.json`,
                  instance: OBJECT_INSTANCE,
                  origin: [0, 0, 0],
                  pose: { translation: [0, 0, 0], rotation: [0, 0, 0, 1] },
                  splats: 5088,
                },
              ],
            }
          : {}),
        ...(features.native ? {} : { nativeLod: false }),
      };
      return route.fulfill({ status: 200, json: tileset });
    }
    if (relative === `synthetic-yard/splat/objects/${String(OBJECT_INSTANCE)}/tileset.json`) {
      return route.fulfill({
        status: 200,
        json: {
          asset: { version: "1.1" },
          geometricError: 0,
          root: {
            transform: scan.root.transform,
            boundingVolume: { sphere: [...OBJECT_CENTRE, 8] },
            geometricError: 0,
            refine: "ADD",
            content: { uri: `../../${OBJECT_TILE}` },
            extras: { gaussians: 5088 },
          },
        },
      });
    }
    if (relative.startsWith(NATIVE))
      relative = `synthetic-yard-sog/${relative.slice(NATIVE.length)}`;
    const file = resolve(TILES, relative);
    if (!existsSync(file)) return route.fulfill({ status: 404, body: "" });
    const type = relative.endsWith(".json")
      ? "application/json"
      : relative.endsWith(".glb")
        ? "model/gltf-binary"
        : relative.endsWith(".webp")
          ? "image/webp"
          : "application/octet-stream";
    return route.fulfill({ status: 200, contentType: type, body: readFileSync(file) });
  });
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.route("**/motion-renderers.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml(renderer) }),
  );
  await page.goto("/motion-renderers.html");
  await page.waitForFunction(() => "__skin" in window, undefined, { timeout: 180_000 });
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

function shooter(page: Page, renderer: Renderer) {
  const framesDir = process.env.MOTION_FRAMES_DIR;
  if (framesDir) mkdirSync(framesDir, { recursive: true });
  return async (name: string): Promise<void> => {
    const file = `${renderer}-${name}.png`;
    await page.screenshot({
      path: framesDir ? resolve(framesDir, file) : test.info().outputPath(file),
    });
  };
}

// A smaller canvas: every frame here is drawn in software.
test.use({ viewport: { width: 960, height: 600 } });

const shaderErrors = (errors: string[]): string[] =>
  errors.filter((e) => /shader|compile|link|webgl|glsl/i.test(e));

const shift = (x: number, y: number, z: number): number[] => [0, 0, 0, x, 0, 0, 0, y, 0, 0, 0, z];

for (const renderer of ["playcanvas", "spark"] as const) {
  test.describe(`under ${renderer}`, () => {
    test(`skins move their objects under ${renderer}, covariances follow, rest is the measured frame`, async ({
      page,
    }) => {
      test.setTimeout(900_000);
      const errors: string[] = [];
      await open(page, renderer, {}, errors);
      const call = caller(page);
      const shot = shooter(page, renderer);

      await call("view", 30, -50, 45);
      await call("view", 30, -50, 45);
      const rest = await call("frame");
      const restAgain = await call("frame");
      await shot("skin-rest");
      const tree = (await call("rectOf", 1, 1.3)) ?? undefined;
      const otherTree = (await call("rectOf", 3)) ?? undefined;
      const shrub = (await call("rectOf", 10)) ?? undefined;

      await call("drive", 1, [
        { handle: 1, z: shift(1.5, 0, 0) },
        { handle: 2, z: shift(0, 1.0, 0) },
      ]);
      const driven = await call("frame");
      const status = await call("rendererStatus");
      await shot("skin-driven");

      await call("hide", [1]);
      const hidden = await call("frame");
      await call("rest");
      const hiddenAtRest = await call("frame");
      await call("hide", []);
      const calm = await call("frame");

      await call("rigid", 10, 0, [0, 0, 2]);
      const lifted = await call("frame");
      await shot("skin-lifted");
      await call("rest");

      const scale = 1.5;
      const grown = [scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0];
      await call("isolate", 10);
      const alone = await call("frame");
      await call("drive", 10, [{ handle: 0, z: grown }]);
      const big = await call("frame");
      await shot("skin-scaled-covariance");
      await call("covariance", false);
      const bigThin = await call("frame");
      await shot("skin-scaled-no-covariance");
      await call("covariance", true);
      await call("rest");
      await call("hide", []);
      const shrubBig = (await call("rectOf", 10, 2.6)) ?? undefined;

      const measures = {
        renderer,
        status,
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
      expect(status?.kind).toBe(renderer);
      expect(status?.error).toBeNull();
      expect(status?.native).toBe(false);
      expect(status?.motion?.skinned).toBeGreaterThan(0);
      expect(status?.instances?.matched).toBeGreaterThan(0);
      expect(measures.restNoise).toBeLessThan(0.001);
      expect(measures.treeMoved).toBeGreaterThan(0.03);
      expect(measures.otherTreeMoved).toBeLessThan(0.01);
      expect(measures.shrubMoved).toBeLessThan(0.01);
      expect(measures.hiddenVsDriven).toBeGreaterThan(0.05);
      expect(measures.hiddenMovingVsAtRest).toBeLessThan(0.001);
      expect(measures.calmVsRest).toBeLessThan(0.001);
      expect(measures.shrubLifted).toBeGreaterThan(0.05);
      expect(measures.treeUnderLift).toBeLessThan(0.01);
      expect(measures.bigCoverage).toBeGreaterThan(measures.restShrubCoverage);
      expect(measures.bigCoverage).toBeGreaterThan(measures.bigThinCoverage * 1.1);
    });

    test(`the wind sways the skinned tree under ${renderer}, calm is the measured frame`, async ({
      page,
    }) => {
      test.setTimeout(900_000);
      const errors: string[] = [];
      await open(page, renderer, { materials: true }, errors);
      const call = caller(page);
      const shot = shooter(page, renderer);

      await call("view", 30, -35, 32);
      await call("view", 30, -35, 32);
      const rest = await call("frame");
      const restAgain = await call("frame");
      await shot("wind-rest");
      const tree = (await call("rectOf", 1, 1.2)) ?? undefined;
      const base = (await call("pointRect", 1, [0, 0, 0.3], 0.45)) ?? undefined;
      const otherTree = (await call("rectOf", 5)) ?? undefined;
      const stillShrub = (await call("rectOf", 12, 0.8)) ?? undefined;

      await call("windOn", 0.1, 60, 1000);
      const skins = await call("windSkins");
      await call("advance", 3);
      const a = await call("frame");
      await shot("wind-a");
      await call("advance", 0.8);
      const b = await call("frame");
      await shot("wind-b");
      await call("advance", 0.8);
      const c = await call("frame");
      await shot("wind-c");

      await call("windOn", 0.1, 60, 1000);
      await call("advance", 3);
      const aAgain = await call("frame");

      await call("windOff");
      const calm = await call("frame");
      await shot("wind-calm");

      const measures = {
        renderer,
        status: await call("rendererStatus"),
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
        replayMax: await call("maxDifference", a, aAgain),
        calmMax: await call("maxDifference", rest, calm),
      };
      test.info().annotations.push({ type: "measures", description: JSON.stringify(measures) });
      console.info(JSON.stringify(measures));

      expect(shaderErrors(errors)).toEqual([]);
      expect(skins.filter((s) => s.wind).map((s) => s.instance)).toEqual([1, 9, 10]);
      expect(measures.restNoise).toBe(0);
      expect(measures.treeSways).toBeGreaterThan(0.005);
      expect(measures.treeAB).toBeGreaterThan(0.003);
      expect(measures.treeBC).toBeGreaterThan(0.003);
      expect(measures.baseAB).toBeLessThan(0.15 * measures.treeAB);
      expect(measures.baseRest).toBeLessThan(0.15 * measures.treeSways);
      expect(measures.otherTree).toBeLessThan(0.002);
      expect(measures.stillShrub).toBeLessThan(0.002);
      // The same steps give the same splats. PlayCanvas draws them in the same order, pixel for
      // pixel; Spark's sorter keeps splats at equal depth in the order it last had them, so a
      // replay reached from another frame may blend a few of those the other way round (a
      // handful of pixels, a few levels): bounded here, exact otherwise.
      if (renderer === "playcanvas") expect(measures.replay).toBe(0);
      else {
        expect(measures.replay).toBeLessThan(0.002);
        expect(measures.replayMax).toBeLessThan(24);
      }
      expect(measures.calm).toBe(0);
    });

    test(`telemetry moves the bound building under ${renderer}, stale fades to the measured frame`, async ({
      page,
    }) => {
      test.setTimeout(900_000);
      const errors: string[] = [];
      await open(page, renderer, { materials: true, telemetry: true }, errors);
      const call = caller(page);
      const shot = shooter(page, renderer);

      await call("view", 20, -40, 40);
      await call("view", 20, -40, 40);
      await call("telemetryReady");
      await call("telemetryAt", 0);
      const rest = await call("frame");
      await shot("telemetry-rest");
      const building = (await call("rectOf", 8, 1.1)) ?? undefined;
      const otherIds = [3, 7, 9];
      const others = await Promise.all(otherIds.map((id) => call("rectOf", id, 0.8)));

      const steps = [4000, 8000, 12000];
      const moved: { at: number; via: string; state: string; arrived: number; left: number }[] = [];
      const frames: number[] = [];
      for (const at of steps) {
        await call("telemetryAt", at);
        const frame = await call("frame");
        frames.push(frame);
        await shot(`telemetry-${String(at / 1000).padStart(2, "0")}s`);
        const status = (await call("telemetryStatus")).find((s) => s.instance === 8);
        const p = status?.position ?? [0, 0, 0];
        const there =
          (await call("scanRect", [p[0] ?? 0, p[1] ?? 0, (p[2] ?? 0) + 2], 1.2)) ?? undefined;
        moved.push({
          at,
          via: status?.via ?? "none",
          state: status?.state ?? "none",
          arrived: await call("difference", rest, frame, there),
          left: await call("difference", rest, frame, building),
        });
      }
      const status = await call("rendererStatus");
      await call("mute", "yard-loop", true);
      await call("mute", "shrub-loop", true);
      await call("telemetryAt", 21_000);
      const rested = await call("frame");
      await shot("telemetry-rested");
      await call("mute", "yard-loop", false);
      await call("telemetryAt", 26_000);
      const resumed = await call("frame");

      const measures = {
        renderer,
        status,
        moved,
        others: [] as number[],
        rested: await call("difference", rest, rested, building, 0),
        restedMax: await call("maxDifference", rest, rested, building),
        resumed: await call("difference", rest, resumed, building),
      };
      for (const f of frames)
        for (const r of others)
          measures.others.push(await call("difference", rest, f, r ?? undefined));
      test.info().annotations.push({ type: "measures", description: JSON.stringify(measures) });
      console.info(JSON.stringify(measures));

      expect(shaderErrors(errors)).toEqual([]);
      expect(status?.error).toBeNull();
      for (const m of moved) {
        expect(m.state).toBe("live");
        expect(m.via).toBe("rigid");
        expect(m.arrived).toBeGreaterThan(0.05);
        expect(m.left).toBeGreaterThan(0.05);
      }
      for (const value of measures.others) expect(value).toBeLessThan(0.002);
      expect(measures.rested).toBe(0);
      expect(measures.resumed).toBeGreaterThan(0.05);
    });

    test(`a split object is drawn at its pose under ${renderer}`, async ({ page }) => {
      test.setTimeout(600_000);
      const errors: string[] = [];
      await open(page, renderer, { objects: true }, errors);
      const call = caller(page);
      const shot = shooter(page, renderer);

      await call("view", 20, -40, 40);
      await call("view", 20, -40, 40);
      const rest = await call("frame");
      await shot("object-rest");
      const lift = 7;
      const above =
        (await call(
          "scanRect",
          [OBJECT_CENTRE[0], OBJECT_CENTRE[1], OBJECT_CENTRE[2] + lift],
          2,
        )) ?? undefined;
      await call("objectPose", OBJECT_INSTANCE, {
        translation: [0, 0, lift],
        rotation: [0, 0, 0, 1],
      });
      const lifted = await call("frame");
      await shot("object-lifted");
      await call("objectPose", OBJECT_INSTANCE, {
        translation: [0, 0, lift],
        rotation: [0, 0, Math.sin(Math.PI / 4), Math.cos(Math.PI / 4)],
      });
      const turned = await call("frame");
      await shot("object-turned");
      await call("objectPose", OBJECT_INSTANCE, null);
      const back = await call("frame");
      const status = await call("rendererStatus");

      const measures = {
        renderer,
        status,
        above,
        arrived: await call("difference", rest, lifted, above),
        turned: await call("difference", lifted, turned, above),
        back: await call("difference", rest, back, undefined, 0),
      };
      test.info().annotations.push({ type: "measures", description: JSON.stringify(measures) });
      console.info(JSON.stringify(measures));

      expect(shaderErrors(errors)).toEqual([]);
      expect(status?.objects).toBe(1);
      expect(measures.arrived).toBeGreaterThan(0.05);
      expect(measures.turned).toBeGreaterThan(0.02);
      expect(measures.back).toBe(0);
    });
  });
}

test("PlayCanvas's own streamed package cannot move objects: the store says so", async ({
  page,
}) => {
  test.setTimeout(600_000);
  const errors: string[] = [];
  await open(page, "playcanvas", { native: true, materials: true }, errors);
  const call = caller(page);
  await call("view", 30, -35, 32);
  const status = await call("rendererStatus");
  const gap = await call("motionGap");
  console.info(JSON.stringify({ status, gap }));
  expect(status?.native).toBe(true);
  expect(gap?.renderer).toBe("playcanvas");
  expect(gap?.reason).toMatch(/checksums/);
});
