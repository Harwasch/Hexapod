/**
 * Poke and drag on scene-object skins (`cesium/skinPoke.ts`, `@twin/world` `skinPoke.ts`) in a
 * real browser on the committed synthetic yard, with the real mouse, under CesiumJS, PlayCanvas
 * and Spark.
 *
 * The yard's skin here is `data/tiles/synthetic-yard/skin-wide/` (skin_variants.py
 * `freeform-stiff`): the tree (1) and the snag (9) at 32 handles -- two texels a splat, so the
 * wide rows are drawn by every renderer -- and the shrubs (10, 12) by size, with
 * `skin/materials.json` (the wind sways 1, 9 and 10: rooted; 12 reads movable).
 *
 * What must hold, with the poke tool on: a press on the tree's crown is taken (the camera's
 * inputs held, the camera does not move) and a drag bends the tree while its base and an
 * unskinned tree stay put; let go, it keeps moving (it rings) and, in time, comes to rest on
 * its own: the measured frame, pixel for pixel. A press on empty ground still turns the camera.
 * A drag on the movable shrub moves it whole. With the tool off, the same press on the tree
 * moves the camera and nothing else.
 *
 * Frames go to `test.info().outputPath()` and, with `POKE_FRAMES_DIR`, there too. Headless GL
 * is SwiftShader: pixels are counted, not eyeballed.
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
interface PokeStatus {
  holding: boolean;
  active: number;
  grabbed: { instance: number; label: string; hz: number | null; movable: boolean } | null;
  cameraInputs: boolean;
}
/** `src/dev/skinHarness.ts`, as the page exposes it (what this spec uses). */
interface SkinHarness {
  view(headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  frame(): number;
  difference(a: number, b: number, rect?: Rect, tol?: number): number;
  rectOf(id: number, grow?: number): Rect | null;
  pointRect(instance: number, local: [number, number, number], radiusM: number): Rect | null;
  skins(): { id: number; instance: number; handles: number; scale: number }[];
  pokeOn(t0?: number): Promise<void>;
  pokeAdvance(seconds: number, fps?: number): Promise<void>;
  pokeOff(): Promise<void>;
  pokeStatus(): PokeStatus;
  pokeOverlay(instance: number): number[] | null;
  screenPoint(instance: number, local: [number, number, number]): { x: number; y: number } | null;
  cameraPose(): { position: [number, number, number]; headingDeg: number };
}

type Renderer = "cesium" | "playcanvas" | "spark";

const TILES = resolve(process.cwd(), "../../data/tiles");
const TILESET = "synthetic-yard/splat/tileset.json";
const INSTANCES = "../instances/instances.json";
const SKIN = "../skin-wide/skin.json";
const MATERIALS = "../skin/materials.json";

function harnessHtml(renderer: Renderer): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Poke harness</title>
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
        renderer: ${JSON.stringify(renderer)},
      });
    </script>
  </body>
</html>`;
}

async function open(page: Page, renderer: Renderer, errors: string[]): Promise<void> {
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
  await page.route("**/poke-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml(renderer) }),
  );
  await page.goto("/poke-harness.html");
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

const moved = (a: [number, number, number], b: [number, number, number]): number =>
  Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

for (const renderer of ["cesium", "playcanvas", "spark"] as const) {
  test(`a poked tree bends, rings and comes to rest; the camera stays yours (${renderer})`, async ({
    page,
  }) => {
    test.setTimeout(1_800_000);
    const errors: string[] = [];
    await open(page, renderer, errors);
    const call = caller(page);
    const framesDir = process.env.POKE_FRAMES_DIR;
    if (framesDir) mkdirSync(framesDir, { recursive: true });
    const shot = async (name: string): Promise<void> => {
      const file = `${renderer}-${name}.png`;
      await page.screenshot({
        path: framesDir ? resolve(framesDir, file) : test.info().outputPath(file),
      });
    };

    await call("view", 30, -35, 32);
    await call("view", 30, -35, 32);
    const skins = await call("skins");
    const rest = await call("frame");
    await shot("poke-rest");
    const tree = (await call("rectOf", 1, 1.2)) ?? undefined;
    const base = (await call("pointRect", 1, [0, 0, 0.3], 0.45)) ?? undefined;
    const otherTree = (await call("rectOf", 5)) ?? undefined;

    // With the tool off, a drag on the crown is the camera's.
    const crown = await call("screenPoint", 1, [0, 0, 6.5]);
    if (!crown) throw new Error("the crown is off screen");
    const before = await call("cameraPose");

    await call("pokeOn", 0);
    // Press on the crown and drag sideways: the press is the poke's, not the camera's.
    await page.mouse.move(crown.x, crown.y);
    await page.mouse.down();
    const held = await call("pokeStatus");
    for (let k = 1; k <= 6; k += 1) await page.mouse.move(crown.x + 12 * k, crown.y - 2 * k);
    await call("pokeAdvance", 1.5);
    const pulled = await call("frame");
    await shot("poke-pulled");
    const during = await call("cameraPose");
    const overlay = await call("pokeOverlay", 1);
    await page.mouse.up();
    const released = await call("pokeStatus");
    // Ringing: two moments a little apart differ, and both differ from rest.
    await call("pokeAdvance", 0.35);
    const ringA = await call("frame");
    await shot("poke-ring-a");
    await call("pokeAdvance", 0.35);
    const ringB = await call("frame");
    await shot("poke-ring-b");
    for (let k = 0; k < 3; k += 1) {
      await call("pokeAdvance", 0.25);
      await shot(`poke-ring-strip-${String(k)}`);
    }
    // It comes to rest on its own: the overlay goes and the frame is the measured one.
    await call("pokeAdvance", 150, 30);
    const settled = await call("pokeStatus");
    const calm = await call("frame");
    await shot("poke-calm");

    // A press on empty space with the tool on still turns the camera.
    const corner = { x: 40, y: 40 };
    await page.mouse.move(corner.x, corner.y);
    await page.mouse.down();
    for (let k = 1; k <= 6; k += 1) await page.mouse.move(corner.x + 25 * k, corner.y + 4 * k);
    await page.mouse.up();
    await call("pokeAdvance", 0.1);
    const afterEmpty = await call("cameraPose");

    // The movable shrub (12) slides whole: the constant handle carries the pull.
    await call("view", 30, -35, 32);
    const shrub = await call("screenPoint", 12, [0, 0, 0.6]);
    let shrubHandle0 = 0;
    let shrubGrabbed: PokeStatus["grabbed"] = null;
    if (shrub) {
      await page.mouse.move(shrub.x, shrub.y);
      await page.mouse.down();
      shrubGrabbed = (await call("pokeStatus")).grabbed;
      for (let k = 1; k <= 5; k += 1) await page.mouse.move(shrub.x + 10 * k, shrub.y);
      await call("pokeAdvance", 1.5);
      const over = await call("pokeOverlay", 12);
      shrubHandle0 = over ? Math.hypot(over[3] ?? 0, over[7] ?? 0, over[11] ?? 0) : 0;
      await page.mouse.up();
      await call("pokeAdvance", 60, 30);
    }

    // The tool off: a drag on the crown turns the camera, and nothing is grabbed.
    await call("pokeOff");
    await call("view", 30, -35, 32);
    const offStart = await call("cameraPose");
    const crownAgain = await call("screenPoint", 1, [0, 0, 6.5]);
    if (crownAgain) {
      await page.mouse.move(crownAgain.x, crownAgain.y);
      await page.mouse.down();
      for (let k = 1; k <= 6; k += 1)
        await page.mouse.move(crownAgain.x + 12 * k, crownAgain.y - 2 * k);
      await page.mouse.up();
    }
    await call("pokeAdvance", 0.1);
    const offEnd = await call("cameraPose");
    const offStatus = await call("pokeStatus");

    const measures = {
      renderer,
      skins,
      held,
      released,
      settled,
      overlayHandles: overlay ? overlay.length / 12 : 0,
      treeBends: await call("difference", rest, pulled, tree),
      baseMoves: await call("difference", rest, pulled, base),
      otherTree: await call("difference", rest, pulled, otherTree),
      ringAB: await call("difference", ringA, ringB, tree),
      ringRest: await call("difference", rest, ringA, tree),
      calm: await call("difference", rest, calm, undefined, 0),
      cameraWhileHeld: moved(before.position, during.position),
      cameraOnEmpty: moved(during.position, afterEmpty.position),
      cameraToolOff: moved(offStart.position, offEnd.position),
      shrubGrabbed,
      shrubHandle0,
      offStatus,
    };
    test.info().annotations.push({ type: "measures", description: JSON.stringify(measures) });
    console.info(JSON.stringify(measures));

    expect(shaderErrors(errors)).toEqual([]);
    // The wide fixture: the tree at 32 handles.
    expect(skins.find((s) => s.instance === 1)?.handles).toBe(32);
    // The press was taken: held, the camera's inputs held, the tree grabbed, rooted.
    expect(held.holding).toBe(true);
    expect(held.cameraInputs).toBe(false);
    expect(held.grabbed?.instance).toBe(1);
    expect(held.grabbed?.movable).toBe(false);
    expect(measures.cameraWhileHeld).toBeLessThan(1e-6);
    expect(measures.overlayHandles).toBe(32);
    // The drag bends the tree, not its base nor its neighbour.
    expect(measures.treeBends).toBeGreaterThan(0.01);
    expect(measures.baseMoves).toBeLessThan(0.2 * measures.treeBends);
    // (The bent crown may pass in front of its neighbour's square on screen.)
    expect(measures.otherTree).toBeLessThan(0.1 * measures.treeBends);
    // Let go: the camera is given back, the tree rings, then rests exactly.
    expect(released.holding).toBe(false);
    expect(released.cameraInputs).toBe(true);
    expect(measures.ringAB).toBeGreaterThan(0.002);
    expect(measures.ringRest).toBeGreaterThan(0.002);
    expect(settled.active).toBe(0);
    expect(measures.calm).toBe(0);
    // A press on empty space is still the camera's.
    expect(measures.cameraOnEmpty).toBeGreaterThan(0.01);
    // The movable shrub moves whole.
    expect(shrubGrabbed?.instance).toBe(12);
    expect(shrubGrabbed?.movable).toBe(true);
    expect(shrubHandle0).toBeGreaterThan(0.01);
    // The tool off: the press is the camera's and nothing is grabbed.
    expect(measures.cameraToolOff).toBeGreaterThan(0.01);
    expect(offStatus.holding).toBe(false);
  });
}
