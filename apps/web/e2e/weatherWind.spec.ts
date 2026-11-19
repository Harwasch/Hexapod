/**
 * Weather wind (docs/LIVING_WEATHER_WIND.md) under the PlayCanvas renderer, on the committed
 * synthetic yard and the scene-object harness (`src/dev/skinHarness.ts`).
 *
 * **Dev machine only.** Weather wind is behind `VITE_LIVING_WEATHER` (off by default), and no
 * GitHub workflow sets it, so CI skips this file and spends no minutes on it. Run it with:
 *
 *   VITE_LIVING_WEATHER=1 E2E_PORT=5741 heavy pnpm --filter @twin/web exec playwright test e2e/weatherWind.spec.ts
 *
 * What must hold, driven exactly as the Settings switch drives the renderer (`setLabWeather`
 * with the scan's `instances.json?tier-a=1`):
 * - the yard as published (a stand-in embedder marked every instance static) selects Tier
 *   "off": its segmentation shows nothing that may move;
 * - with the tree (instance 1) marked as a plant, it selects Tier A and the tree's pixels move
 *   from one fixed clock to another while a far shrub stays still;
 * - As captured restores the measured frame exactly.
 *
 * Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

test.skip(
  !["1", "true", "yes"].includes((process.env.VITE_LIVING_WEATHER ?? "").toLowerCase()),
  "Weather wind is opt-in: run on the dev machine with VITE_LIVING_WEATHER=1.",
);

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}
interface Harness {
  view(headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  frame(): number;
  difference(a: number, b: number, rect?: Rect, tol?: number): number;
  maxDifference(a: number, b: number, rect?: Rect): number;
  rectOf(id: number, grow?: number): Rect | null;
  advance(seconds: number, fps?: number): Promise<void>;
}

const TILES = resolve(process.cwd(), "../../data/tiles");
const TILESET = "synthetic-yard/splat/tileset.json";
const INSTANCES_PATH = "synthetic-yard/instances/instances.json";
/** The fresh hour of the research fixtures (Nov 19 2024 04:00 UTC, Minnetonka). */
const FRESH = {
  meanMps: 8.51,
  gustMps: 15.4,
  directionDeg: 97,
  latitude: 44.944403,
  longitude: -93.425903,
};

function harnessHtml(): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Weather wind harness</title>
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
        renderer: "playcanvas",
      });
    </script>
  </body>
</html>`;
}

/** Opens the yard; `plants` lists instances served as vegetation rather than static. */
async function open(page: Page, plants: number[], errors: string[]): Promise<void> {
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  const instances = JSON.parse(readFileSync(resolve(TILES, INSTANCES_PATH), "utf-8")) as {
    instances: { id: number; properties?: Record<string, number> }[];
  };
  for (const instance of instances.instances)
    if (plants.includes(instance.id))
      instance.properties = { ...instance.properties, static: 0, rigid: 0, vegetation: 1 };
  const skins =
    (
      JSON.parse(readFileSync(resolve(TILES, "synthetic-yard/skin/skin.json"), "utf-8")) as {
        skins?: unknown[];
      }
    ).skins ?? [];
  const scan = JSON.parse(readFileSync(resolve(TILES, TILESET), "utf-8")) as {
    root: { extras?: Record<string, unknown> };
  };
  await page.route("**/fixture-tiles/**", (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname.replace(/^.*\/fixture-tiles\//, "");
    const relative = resolve("/", path).slice(1);
    if (relative !== path) return route.abort();
    if (relative === TILESET) {
      const tileset = JSON.parse(JSON.stringify(scan)) as typeof scan;
      tileset.root.extras = {
        ...tileset.root.extras,
        instances: { uri: "../instances/instances.json", count: instances.instances.length },
        // The harness waits for the yard's skins; at rest they do not move anything.
        skin: { uri: "../skin/skin.json", count: skins.length },
        nativeLod: false,
      };
      return route.fulfill({ status: 200, json: tileset });
    }
    if (relative === INSTANCES_PATH) return route.fulfill({ status: 200, json: instances });
    const file = resolve(TILES, relative);
    if (!existsSync(file)) return route.fulfill({ status: 404, body: "" });
    const type = relative.endsWith(".json")
      ? "application/json"
      : relative.endsWith(".glb")
        ? "model/gltf-binary"
        : "application/octet-stream";
    return route.fulfill({ status: 200, contentType: type, body: readFileSync(file) });
  });
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.route("**/weather-wind.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml() }),
  );
  await page.goto("/weather-wind.html");
  await page.waitForFunction(() => "__skin" in window, undefined, { timeout: 180_000 });
}

function caller(page: Page) {
  return <K extends keyof Harness>(
    method: K,
    ...args: Parameters<Harness[K]>
  ): Promise<Awaited<ReturnType<Harness[K]>>> =>
    page.evaluate(
      ([m, a]) => {
        const harness = (window as unknown as { __skin: Record<string, unknown> }).__skin;
        return (harness[m] as (...x: unknown[]) => unknown)(...a);
      },
      [method, args] as [string, unknown[]],
    ) as Promise<Awaited<ReturnType<Harness[K]>>>;
}

/** Puts the hour's wind on the yard as the Settings switch does, at a fixed clock. */
function blow(page: Page, clock: number): Promise<void> {
  return page.evaluate(
    `(async () => {
      const { setLabWeather } = await import("/src/state/livingWeather.ts");
      const { useLivingLab } = await import("/src/state/livingLab.ts");
      useLivingLab.getState().set({ clock: ${String(clock)} });
      const url = new URL("/fixture-tiles/${INSTANCES_PATH}?tier-a=1", location.href).toString();
      setLabWeather(${JSON.stringify(FRESH)}, url);
    })()`,
  );
}

/** What the renderer reported for the articulation it was asked for, and its tier. */
function report(page: Page): Promise<{ status: string; tier: string | null; current: boolean }> {
  return page.evaluate(`(async () => {
    const { useLivingLab } = await import("/src/state/livingLab.ts");
    const { weatherArticulationTier } = await import("/src/cesium/scanView/livingArticulation.ts");
    const lab = useLivingLab.getState();
    return {
      status: lab.articulationStatus,
      tier: weatherArticulationTier(lab.articulationUrl) ?? null,
      current: lab.articulationStatusUrl === lab.articulationUrl,
    };
  })()`);
}

function setClock(page: Page, clock: number): Promise<void> {
  return page.evaluate(`(async () => {
    const { useLivingLab } = await import("/src/state/livingLab.ts");
    useLivingLab.getState().set({ clock: ${String(clock)} });
  })()`);
}

function asCaptured(page: Page, on: boolean): Promise<void> {
  return page.evaluate(`(async () => {
    const { useAsCaptured } = await import("/src/state/asCaptured.ts");
    useAsCaptured.setState({ on: ${String(on)} });
  })()`);
}

const loaded = async (page: Page) => {
  await expect
    .poll(
      async () => {
        const r = await report(page);
        return r.current && r.status.startsWith("Simulated") ? r.tier : r.status;
      },
      { timeout: 120_000 },
    )
    .toMatch(/^(A|B|C|off)$/);
  return report(page);
};

// A smaller canvas: every frame here is drawn in software.
test.use({ viewport: { width: 960, height: 600 } });

const shaderErrors = (errors: string[]): string[] =>
  errors.filter((e) => /shader|compile|link|webgl|glsl/i.test(e));

test("a scan whose segmentation shows nothing that may move stays off", async ({ page }) => {
  test.setTimeout(600_000);
  const errors: string[] = [];
  await open(page, [], errors);
  await blow(page, 10);
  expect((await loaded(page)).tier).toBe("off");
  expect(shaderErrors(errors)).toEqual([]);
});

test("Tier A sways a plant from segmentation; As captured is the measured frame", async ({
  page,
}) => {
  test.setTimeout(900_000);
  const errors: string[] = [];
  await open(page, [1], errors);
  const call = caller(page);
  await call("view", 30, -35, 32);
  await call("view", 30, -35, 32);
  const rest = await call("frame");
  const tree = (await call("rectOf", 1, 1.2)) ?? undefined;
  const shrub = (await call("rectOf", 12, 0.8)) ?? undefined;

  await blow(page, 10);
  expect((await loaded(page)).tier).toBe("A");
  await call("advance", 0);
  const a = await call("frame");
  await setClock(page, 11.5);
  await call("advance", 0);
  const b = await call("frame");

  await asCaptured(page, true);
  await call("advance", 0);
  const captured = await call("frame");

  const measures = {
    treeSways: await call("difference", rest, a, tree),
    treeAB: await call("difference", a, b, tree),
    shrub: await call("difference", rest, a, shrub),
    capturedMax: await call("maxDifference", rest, captured),
  };
  test.info().annotations.push({ type: "measures", description: JSON.stringify(measures) });
  console.info(JSON.stringify(measures));

  expect(shaderErrors(errors)).toEqual([]);
  expect(measures.treeSways).toBeGreaterThan(0.003);
  expect(measures.treeAB).toBeGreaterThan(0.002);
  expect(measures.shrub).toBeLessThan(0.002);
  expect(measures.capturedMax).toBe(0);
});
