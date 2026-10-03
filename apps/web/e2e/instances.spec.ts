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
 * The same holds under the app's default splat renderer, PlayCanvas over the globe
 * (cesium/scanView): there the ids come from each tile's checksum as PlayCanvas decodes it, and
 * a hidden splat loses its opacity in PlayCanvas's own work buffer (scanInstances.ts).
 *
 * Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import { existsSync, readFileSync } from "node:fs";
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
  scan(): {
    kind: string;
    active: boolean;
    tiles: number;
    native: boolean;
    instances: { tiles: number; matched: number } | null;
  } | null;
}

const TILES = resolve(process.cwd(), "../../data/tiles");
const TILESET = "synthetic-yard/splat/tileset.json";
const INSTANCES = "../instances/instances.json";

interface Options {
  incremental: boolean;
  maximumScreenSpaceError: number;
  renderer?: "cesium" | "playcanvas" | "spark";
  /**
   * Mount the app's objects panel, and tag the yard's two largest objects (the yard is
   * segmented against its own truth, so it has no tags): the largest "conifer", the next
   * "bush", so the panel lists Trees and Shrubs & bushes over Other.
   */
  panel?: boolean;
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
      #panel { position: fixed; top: 12px; right: 12px; z-index: 10; width: 21rem; padding: 0.6rem; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    ${options.panel ? '<div id="panel" class="glass glass--strong"></div>' : ""}
    <script type="module">
      const harness = await import("/src/dev/instancesHarness.ts");
      window.__instances = await harness.startInstancesHarness({
        container: document.getElementById("viewer"),
        url: "/fixture-tiles/${TILESET}",
        incremental: ${String(options.incremental)},
        maximumScreenSpaceError: ${String(options.maximumScreenSpaceError)},
        renderer: "${options.renderer ?? "cesium"}",
        ${options.panel ? 'panel: document.getElementById("panel"),' : ""}
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
  ) as {
    instances: { id: number; parent: number | null; splats: number; tags?: unknown[] }[];
  };
  if (options.panel) {
    const roots = instances.instances
      .filter((i) => i.parent === null)
      .sort((a, b) => b.splats - a.splats);
    if (roots[0]) roots[0].tags = [{ label: "conifer", score: 0.6 }];
    if (roots[1]) roots[1].tags = [{ label: "bush", score: 0.5 }];
  }
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
        // The yard has no PlayCanvas streamed package here, and says so: nothing is probed.
        nativeLod: false,
      };
      return route.fulfill({ status: 200, json: tileset });
    }
    if (relative === "synthetic-yard/instances/instances.json") {
      return route.fulfill({ status: 200, json: instances });
    }
    const file = resolve(TILES, relative);
    if (!existsSync(file)) return route.fulfill({ status: 404, body: "" });
    const type = relative.endsWith(".json")
      ? "application/json"
      : relative.endsWith(".glb")
        ? "model/gltf-binary"
        : "application/octet-stream";
    return route.fulfill({
      status: 200,
      contentType: type,
      body: readFileSync(file),
    });
  });
  await page.route("**/instances-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml(options) }),
  );
  await page.goto("/instances-harness.html");
  try {
    await page.waitForFunction(() => "__instances" in window, undefined, { timeout: 180_000 });
  } catch (error) {
    throw new Error(`the harness never started; page errors: ${errors.join(" | ")}`, {
      cause: error,
    });
  }
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

for (const renderer of ["playcanvas", "spark"] as const) {
  test(`under ${renderer === "playcanvas" ? "PlayCanvas, the app's default renderer" : "Spark"}, objects hide, highlight and hide all`, async ({
    page,
  }) => {
    test.setTimeout(600_000);
    const errors: string[] = [];
    const requests: string[] = [];
    page.on("request", (request) => requests.push(request.url()));
    await open(page, { incremental: true, maximumScreenSpaceError: 16, renderer }, errors);
    const call = caller(page);
    await call("view", 30, -50, 45);
    const baseline = await call("view", 30, -50, 45);
    const scan = await call("scan");
    const hooks = await call("hooks");
    await page.screenshot({ path: test.info().outputPath(`${renderer}-baseline.png`) });
    const list = await call("instances");
    const largest = list.filter((i) => i.parent === null).slice(0, 3);
    const hidden = await call("set", { hidden: largest.map((i) => i.id) });
    await page.screenshot({ path: test.info().outputPath(`${renderer}-hidden-largest.png`) });
    // Hide all: every instance at once, as "Hide all N matches" does for a broad query.
    const all = await call("set", { hidden: list.map((i) => i.id) });
    await page.screenshot({ path: test.info().outputPath(`${renderer}-hidden-all.png`) });
    const shown = await call("set", {});
    const target = largest[2]?.id ?? 1;
    const rect = (await call("rectOf", target)) ?? undefined;
    const before = await call("measure", rect);
    const tinted = await call("set", { highlighted: [target], dimOthers: false }, rect);
    const undimmed = await call("measure");
    const dimmed = await call("set", { highlighted: [target] });
    await page.screenshot({ path: test.info().outputPath(`${renderer}-highlight.png`) });
    const both = await call("set", { hidden: [largest[0]?.id ?? 1], highlighted: [target] });
    await page.screenshot({ path: test.info().outputPath(`${renderer}-hide-and-highlight.png`) });
    const restored = await call("set", {});

    test.info().annotations.push({
      type: "measures",
      description: JSON.stringify({
        scan,
        hooks,
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
        both,
        restored,
      }),
    });
    expect(shaderErrors(errors)).toEqual([]);
    expect(errors.filter((e) => /404|Failed to load/i.test(e))).toEqual([]);
    // The tileset declares no native package: it is not probed for one.
    expect(requests.filter((url) => url.includes("lod-meta.json"))).toEqual([]);
    expect(scan?.kind).toBe(renderer);
    expect(scan?.native).toBe(false);
    expect(hooks.table).toBe(true);
    // Every tile PlayCanvas loaded is found in instances.json by its checksum.
    expect(scan?.instances?.tiles).toBeGreaterThan(0);
    expect(scan?.instances?.matched).toBe(scan?.instances?.tiles);
    // CesiumJS draws no splats here: what is counted is the dedicated renderer's.
    expect(baseline.coverage).toBeGreaterThan(0.05);
    expect(hidden.coverage).toBeLessThan(baseline.coverage * 0.9);
    // The app's screen-space error draws coarse tiles too, whose merged splats carry an id only
    // where all their children share it (as in aggregated mode above).
    expect(all.coverage).toBeLessThan(baseline.coverage * 0.25);
    expect(shown.coverage).toBeGreaterThan(baseline.coverage * 0.95);
    expect(restored.coverage).toBeGreaterThan(baseline.coverage * 0.95);
    expect(tinted.warmth).toBeGreaterThan(before.warmth * 1.3);
    expect(dimmed.luma).toBeLessThan(undimmed.luma * 0.6);
    expect(dimmed.coverage).toBeLessThan(undimmed.coverage);
    expect(both.coverage).toBeLessThan(dimmed.coverage);
  });
}

test("the objects panel hides a category, highlights it on a click, and resets, under PlayCanvas", async ({
  page,
}) => {
  test.setTimeout(600_000);
  const errors: string[] = [];
  await open(
    page,
    { incremental: true, maximumScreenSpaceError: 16, renderer: "playcanvas", panel: true },
    errors,
  );
  const call = caller(page);
  await call("view", 30, -50, 45);
  const baseline = await call("view", 30, -50, 45);
  const panel = page.getByTestId("instance-panel");
  const rows = panel.locator("ul[aria-label='Object categories'] > li");
  await expect(rows).toHaveCount(3);
  expect(
    await rows.evaluateAll((li) => li.map((e) => (e as HTMLElement).dataset.category)),
  ).toEqual(["trees", "shrubs", "other"]);
  await expect(panel).not.toContainText(/Object \d/);
  await page.locator("#panel").screenshot({ path: test.info().outputPath("panel.png") });

  await panel.getByRole("button", { name: "Hide Trees", exact: true }).click();
  await page.waitForTimeout(500);
  const hidden = await call("measure");
  await page.screenshot({ path: test.info().outputPath("panel-hidden-trees.png") });
  await expect(panel.getByRole("status")).toHaveText("Trees hidden");
  await panel.getByRole("button", { name: "Reset" }).click();
  await page.waitForTimeout(500);
  const reset = await call("measure");

  await panel.locator("li[data-category='trees'] > div > [data-row]").click();
  await page.waitForTimeout(500);
  const lit = await call("measure");
  await page.screenshot({ path: test.info().outputPath("panel-highlight-trees.png") });
  await panel.locator("li[data-category='trees'] > div > [data-row]").click();
  await page.waitForTimeout(500);
  const cleared = await call("measure");

  test.info().annotations.push({
    type: "measures",
    description: JSON.stringify({ baseline, hidden, reset, lit, cleared }),
  });
  expect(shaderErrors(errors)).toEqual([]);
  expect(hidden.coverage).toBeLessThan(baseline.coverage * 0.9);
  expect(reset.coverage).toBeGreaterThan(baseline.coverage * 0.95);
  expect(lit.warmth).toBeGreaterThan(baseline.warmth + 5);
  expect(lit.luma).toBeLessThan(baseline.luma);
  expect(cleared.coverage).toBeGreaterThan(baseline.coverage * 0.95);
});
