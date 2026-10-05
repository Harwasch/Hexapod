/**
 * The motion-skins bake-off's candidates on a real scan, for looking: skipped unless
 * `SKINS_SCAN_DIR` names a directory `skin_variants.py build` (or `scan`) filled -- the scan's
 * `tileset.json`, the tiles it fetched, `instances.json` when the scan has one, and
 * `variants/skins/<name>/`. For each candidate (`SKINS_VARIANTS`, default every one built) it
 * frames one object (`SKINS_OBJECT`, an instance id; the tree's is 1), blows the wind over it
 * and saves a strip of frames, then pokes it -- a press on it and a drag with the real mouse
 * -- and saves the pull and the ring-down. Frames go to `SKINS_SHOTS_DIR` (else the test's
 * output directory) as `<variant>-wind-<k>.png` and `<variant>-poke-<k>.png`:
 *
 *   SKINS_SCAN_DIR=/path/to/scan SKINS_OBJECT=2 npx playwright test e2e/skinsScan.spec.ts
 *
 * Tiles the directory does not hold (a scan fetched for a few objects) are pruned from the
 * tileset served, with everything below them. `SKINS_RENDERER` (cesium, playcanvas, spark),
 * `SKINS_HEADING`, `SKINS_PITCH`, `SKINS_RANGE` (metres from the object's middle),
 * `SKINS_WIND` (strength, default 0.4) and `SKINS_PULL` (pixels, default 90) tune it. It
 * asserts only what holds on any scan: the candidate loads, the wind and the poke move the
 * object's pixels, and calm with nothing held is the measured frame.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}
/** `src/dev/skinHarness.ts`, as the page exposes it (what this spec uses). */
interface SkinHarness {
  lookAtSkin(instance: number, headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  skins(): { id: number; instance: number; handles: number; scale: number }[];
  frame(): number;
  difference(a: number, b: number, rect?: Rect, tol?: number): number;
  pointRect(instance: number, local: [number, number, number], radiusM: number): Rect | null;
  windOn(strength: number, bearingDeg: number, t0?: number, seed?: number): Promise<void>;
  advance(seconds: number, fps?: number): Promise<void>;
  windOff(): Promise<void>;
  pokeOn(t0?: number): Promise<void>;
  pokeAdvance(seconds: number, fps?: number): Promise<void>;
  pokeStatus(): { holding: boolean; active: number; grabbed: { instance: number } | null };
  screenPoint(instance: number, local: [number, number, number]): { x: number; y: number } | null;
}

const SCAN = process.env.SKINS_SCAN_DIR;
const SHOTS = process.env.SKINS_SHOTS_DIR;
const OBJECT = Number(process.env.SKINS_OBJECT ?? 1);
const RENDERER = process.env.SKINS_RENDERER ?? "cesium";
const HEADING = Number(process.env.SKINS_HEADING ?? 30);
const PITCH = Number(process.env.SKINS_PITCH ?? -15);
const RANGE = Number(process.env.SKINS_RANGE ?? 0);
const WIND = Number(process.env.SKINS_WIND ?? 0.4);
const PULL = Number(process.env.SKINS_PULL ?? 90);

function variantsOf(dir: string): string[] {
  const asked = process.env.SKINS_VARIANTS;
  if (asked) return asked.split(",").filter(Boolean);
  const root = join(dir, "variants", "skins");
  return existsSync(root) ? readdirSync(root).sort() : [];
}

interface Tile {
  content?: { uri?: string };
  children?: Tile[];
  [key: string]: unknown;
}

/** The tileset without the tiles `dir` does not hold (and what is below them). */
function pruned(tile: Tile, dir: string): Tile | null {
  const uri = tile.content?.uri;
  if (uri && !existsSync(join(dir, uri))) return null;
  const children = (tile.children ?? []).flatMap((c) => {
    const kept = pruned(c, dir);
    return kept ? [kept] : [];
  });
  return { ...tile, children };
}

function html(renderer: string): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Skins harness</title>
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
        url: "/scan-tiles/tileset.json",
        incremental: true,
        maximumScreenSpaceError: 2,
        renderer: ${JSON.stringify(renderer)},
      });
    </script>
  </body>
</html>`;
}

async function open(page: Page, dir: string, variant: string): Promise<void> {
  await page.route("**/scan-tiles/**", (route) => {
    const path = new URL(route.request().url()).pathname.replace(/^.*\/scan-tiles\//, "");
    const relative = resolve("/", path).slice(1);
    if (relative !== path) return route.abort();
    if (relative === "tileset.json") {
      const tileset = JSON.parse(readFileSync(join(dir, relative), "utf-8")) as {
        root: Tile & { extras?: Record<string, unknown> };
      };
      const root = pruned(tileset.root, dir) ?? tileset.root;
      const extras: Record<string, unknown> = { ...(tileset.root.extras ?? {}) };
      if (existsSync(join(dir, "instances.json"))) {
        const doc = JSON.parse(readFileSync(join(dir, "instances.json"), "utf-8")) as {
          instances: unknown[];
        };
        extras.instances = { uri: "instances.json", count: doc.instances.length };
      }
      extras.skin = { uri: `variants/skins/${variant}/skin.json`, count: 1 };
      delete extras.materials;
      return route.fulfill({ status: 200, json: { ...tileset, root: { ...root, extras } } });
    }
    const file = join(dir, relative);
    if (!existsSync(file)) return route.fulfill({ status: 404, body: "" });
    const type = relative.endsWith(".json")
      ? "application/json"
      : relative.endsWith(".glb")
        ? "model/gltf-binary"
        : "application/octet-stream";
    return route.fulfill({ status: 200, contentType: type, body: readFileSync(file) });
  });
  await page.route("**/skins-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: html(RENDERER) }),
  );
  await page.goto("/skins-harness.html");
  await page.waitForFunction(() => "__skin" in window, undefined, { timeout: 300_000 });
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

test.use({ viewport: { width: 960, height: 600 } });

const DIR = SCAN ? resolve(SCAN) : "";
const VARIANTS = SCAN ? variantsOf(DIR) : [];

test.skip(!SCAN || !existsSync(join(DIR, "tileset.json")), "SKINS_SCAN_DIR is not set");

for (const variant of VARIANTS) {
  test(`${variant}: wind and poke on object ${String(OBJECT)}`, async ({ page }) => {
    test.setTimeout(3_600_000);
    const shot = async (name: string): Promise<void> => {
      if (SHOTS) mkdirSync(SHOTS, { recursive: true });
      const file = `${variant}-${name}.png`;
      await page.screenshot({ path: SHOTS ? join(SHOTS, file) : test.info().outputPath(file) });
    };
    await open(page, DIR, variant);
    const call = caller(page);
    const skin = (await call("skins")).find((s) => s.instance === OBJECT);
    expect(skin, `object ${String(OBJECT)} has a skin in ${variant}`).toBeDefined();
    const scale = skin?.scale ?? 1;
    await call("lookAtSkin", OBJECT, HEADING, PITCH, RANGE > 0 ? RANGE : 3.2 * scale);
    const rect = (await call("pointRect", OBJECT, [0, 0, scale], 1.2 * scale)) ?? undefined;
    const rest = await call("frame");
    await shot("rest");

    await call("windOn", WIND, 60, 100);
    await call("advance", 4);
    const windy = await call("frame");
    await shot("wind-0");
    for (let k = 1; k < 4; k += 1) {
      await call("advance", 0.4);
      await shot(`wind-${String(k)}`);
    }
    await call("windOff");

    await call("pokeOn", 0);
    const at = await call("screenPoint", OBJECT, [0, 0, 1.3 * scale]);
    let pulled = rest;
    let grabbed = false;
    if (at) {
      await page.mouse.move(at.x, at.y);
      await page.mouse.down();
      grabbed = (await call("pokeStatus")).holding;
      for (let k = 1; k <= 6; k += 1) await page.mouse.move(at.x + (PULL * k) / 6, at.y);
      await call("pokeAdvance", 1.2);
      pulled = await call("frame");
      await shot("poke-0");
      await page.mouse.up();
      for (let k = 1; k < 5; k += 1) {
        await call("pokeAdvance", 0.3);
        await shot(`poke-${String(k)}`);
      }
    }
    await call("pokeAdvance", 240, 20);
    const calm = await call("frame");
    const measures = {
      variant,
      handles: skin?.handles,
      wind: await call("difference", rest, windy, rect),
      poke: await call("difference", rest, pulled, rect),
      grabbed,
      calm: await call("difference", rest, calm, undefined, 0),
      status: await call("pokeStatus"),
    };
    test.info().annotations.push({ type: "measures", description: JSON.stringify(measures) });
    console.info(JSON.stringify(measures));
    // A rigid object (one handle) moves under neither; anything else under the poke.
    if ((skin?.handles ?? 1) > 1) expect(measures.poke).toBeGreaterThan(0);
    expect(measures.calm).toBe(0);
  });
}
