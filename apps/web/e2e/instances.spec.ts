/**
 * Scene instances (hide, highlight) in a real CesiumJS 1.145 with the engine patch's
 * `vertexVisibility` and `vertexColor` hooks, on the committed synthetic yard.
 *
 * `data/tiles/synthetic-yard/instances/` is the yard segmented against its own ground truth
 * (`tools/captures/segment_scene.py ... --truth labels.json --tile-gaussians 6000`). The
 * committed tiles stay byte-identical to what the packer writes (`test_scene_plants`), so the
 * route below links the file from the root's extras as `segment_scene.link_instances` would.
 *
 * What must hold: the hooks compile (no shader errors) and both are installed; hiding every
 * instance empties the scan (the ids line up with the splats drawn, in the app's incremental
 * mode and in aggregated mode); hiding the largest objects removes coverage; a highlight turns
 * them amber and dims the rest; and with the view-cone part in the same chain ("seen from
 * below only", seen from above) the scan still vanishes and comes back.
 *
 * Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

interface Measure {
  coverage: number;
  amber: number;
  luma: number;
  warmth: number;
}
interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}
/** `src/dev/instancesHarness.ts`, as the page exposes it. */
interface InstancesHarness {
  view(headingDeg: number, pitchDeg: number, rangeM: number): Promise<Measure>;
  instances(): { id: number; parent: number | null; level: number; splats: number }[];
  set(
    state: { hidden?: number[]; highlighted?: number[]; dimOthers?: boolean },
    rect?: Rect,
  ): Promise<Measure>;
  cones(mode: "off" | "below"): Promise<Measure>;
  rectOf(id: number): Rect | null;
  measure(rect?: Rect): Measure;
  hooks(): { visibility: string[]; color: boolean; table: boolean };
  tiles(): { uri: string; splats: number }[];
}

const TILES = resolve(process.cwd(), "../../data/tiles");
const TILESET = "synthetic-yard/splat/tileset.json";
const INSTANCES = "../instances/instances.json";

interface Options {
  incremental: boolean;
  maximumScreenSpaceError: number;
}

function harnessHtml(options: Options): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Instances harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/instancesHarness.ts");
      window.__instances = await harness.startInstancesHarness({
        container: document.getElementById("viewer"),
        url: "/fixture-tiles/${TILESET}",
        incremental: ${String(options.incremental)},
        maximumScreenSpaceError: ${String(options.maximumScreenSpaceError)},
      });
    </script>
  </body>
</html>`;
}

async function open(page: Page, options: Options, errors: string[]): Promise<void> {
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  const instances = JSON.parse(
    readFileSync(resolve(TILES, "synthetic-yard/instances/instances.json"), "utf-8"),
  ) as { instances: unknown[] };
  await page.route("**/fixture-tiles/**", (route) => {
    const relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    if (relative === TILESET) {
      const tileset = JSON.parse(readFileSync(resolve(TILES, relative), "utf-8")) as {
        root: { extras?: Record<string, unknown> };
      };
      tileset.root.extras = {
        ...tileset.root.extras,
        instances: { uri: INSTANCES, count: instances.instances.length },
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
  await page.route("**/instances-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml(options) }),
  );
  await page.goto("/instances-harness.html");
  await page.waitForFunction(() => "__instances" in window, undefined, { timeout: 120_000 });
}

/** Calls a harness method in the page. */
function caller(page: Page) {
  return <K extends keyof InstancesHarness>(
    method: K,
    ...args: Parameters<InstancesHarness[K]>
  ): Promise<Awaited<ReturnType<InstancesHarness[K]>>> =>
    page.evaluate(
      ([m, a]) => {
        const harness = (window as unknown as { __instances: Record<string, unknown> }).__instances;
        return (harness[m] as (...x: unknown[]) => unknown)(...a);
      },
      [method, args] as [string, unknown[]],
    ) as Promise<Awaited<ReturnType<InstancesHarness[K]>>>;
}

const shaderErrors = (errors: string[]): string[] =>
  errors.filter((e) => /shader|compile|link|webgl/i.test(e));

test("hidden objects vanish and a highlight turns amber, composed with the view cones", async ({
  page,
}) => {
  test.setTimeout(600_000);
  const errors: string[] = [];
  // The app's incremental primitive; a low screen-space error draws the leaves everywhere,
  // where all but ~2% of the yard's splats carry an instance.
  await open(page, { incremental: true, maximumScreenSpaceError: 1 }, errors);
  const call = caller(page);

  // Twice: the first view also waits for the level-of-detail tiles to finish refining.
  await call("view", 30, -50, 45);
  const baseline = await call("view", 30, -50, 45);
  await page.screenshot({ path: test.info().outputPath("baseline.png") });
  const hooks = await call("hooks");
  const tiles = await call("tiles");
  const list = await call("instances");
  const largest = list.filter((i) => i.parent === null).slice(0, 3);
  const hidden = await call("set", { hidden: largest.map((i) => i.id) });
  await page.screenshot({ path: test.info().outputPath("hidden-largest.png") });
  const all = await call("set", { hidden: list.map((i) => i.id) });
  await page.screenshot({ path: test.info().outputPath("hidden-all.png") });
  const shown = await call("set", {});
  // A smaller object, so what is dimmed is most of the frame.
  const target = largest[2]?.id ?? 1;
  const rect = (await call("rectOf", target)) ?? undefined;
  const before = await call("measure", rect);
  const tinted = await call("set", { highlighted: [target], dimOthers: false }, rect);
  const undimmed = await call("measure");
  const dimmed = await call("set", { highlighted: [target] });
  await page.screenshot({ path: test.info().outputPath("highlight.png") });
  await call("set", {});
  // The view-cone part shares the chain with the instance part: seen from above, a grid
  // "seen from below only" removes everything, hidden or not, and swapping the scan's own
  // grid back brings back all but what is hidden.
  const conesBelow = await call("cones", "below");
  const conesAndHidden = await call("set", { hidden: [largest[0]?.id ?? 1] });
  const hooksWithCones = await call("hooks");
  await call("cones", "off");
  const back = await call("set", { hidden: [largest[0]?.id ?? 1] });
  const restored = await call("set", {});

  test.info().annotations.push({
    type: "measures",
    description: JSON.stringify({
      hooks,
      tiles,
      largest,
      baseline,
      hidden,
      all,
      shown,
      rect,
      before,
      tinted,
      undimmed,
      dimmed,
      conesBelow,
      conesAndHidden,
      back,
      restored,
    }),
  });
  expect(shaderErrors(errors)).toEqual([]);
  expect(hooks.table).toBe(true);
  expect(hooks.color).toBe(true);
  expect(hooks.visibility).toEqual(["splatInstanceVisibility", "splatViewConeVisibility"]);
  expect(hooksWithCones.visibility).toEqual(hooks.visibility);
  expect(baseline.coverage).toBeGreaterThan(0.05);
  // Hiding the three largest objects takes a visible share of the scan away...
  expect(hidden.coverage).toBeLessThan(baseline.coverage * 0.9);
  // ...and hiding every object leaves only what no instance claims (8% of the coverage,
  // measured): the ids line up with the splats drawn.
  expect(all.coverage).toBeLessThan(baseline.coverage * 0.15);
  // Showing them again draws the scan as before.
  expect(shown.coverage).toBeGreaterThan(baseline.coverage * 0.95);
  expect(restored.coverage).toBeGreaterThan(baseline.coverage * 0.95);
  // A highlight pulls the object toward amber...
  expect(tinted.warmth).toBeGreaterThan(before.warmth * 1.3);
  // ...and, with "dim the rest" on, everything else darkens and thins.
  expect(dimmed.luma).toBeLessThan(undimmed.luma * 0.6);
  expect(dimmed.coverage).toBeLessThan(undimmed.coverage);
  // Composed with the view cones: gone from above under "below only"; back without it.
  expect(conesBelow.coverage).toBeLessThan(baseline.coverage * 0.02);
  expect(conesAndHidden.coverage).toBeLessThan(baseline.coverage * 0.02);
  expect(back.coverage).toBeLessThan(baseline.coverage * 0.98);
  expect(back.coverage).toBeGreaterThan(hidden.coverage);
});

test("the ids follow the splats in aggregated mode, at the app's screen-space error", async ({
  page,
}) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  await open(page, { incremental: false, maximumScreenSpaceError: 16 }, errors);
  const call = caller(page);
  await call("view", 30, -50, 45);
  const baseline = await call("view", 30, -50, 45);
  const tiles = await call("tiles");
  const list = await call("instances");
  const all = await call("set", { hidden: list.map((i) => i.id) });
  const lit = await call("set", { highlighted: list.map((i) => i.id), dimOthers: false });
  const restored = await call("set", {});
  test.info().annotations.push({
    type: "measures",
    description: JSON.stringify({ tiles, baseline, all, lit, restored }),
  });
  expect(shaderErrors(errors)).toEqual([]);
  // Coarse tiles carry an id only where every splat merged into one shares it, so more is
  // left than at the leaves.
  expect(all.coverage).toBeLessThan(baseline.coverage * 0.25);
  expect(lit.warmth).toBeGreaterThan(baseline.warmth + 20);
  expect(restored.coverage).toBeGreaterThan(baseline.coverage * 0.95);
});
