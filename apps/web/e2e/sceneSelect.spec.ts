/**
 * Selecting the synthetic yard's objects in the scene (cesium/sceneSelect, docs/SCENE_OBJECTS.md
 * "Selecting in the scene"), with the real mouse and keyboard, under PlayCanvas (the app's
 * default renderer) and Spark and CesiumJS: a click on a tree's crown selects the tree (or a
 * part of it) and gives the map the keyboard, `]` and Tab cycle to its parent or child (Tab
 * only from the map or the card: from the page's body it moves focus), the brush painted over
 * a shrub selects that shrub, and Hide in the selection card hides it (its pixels change to the
 * lawn behind it). The card is the HUD's (features/sites/ObjectCard.tsx), mounted by the
 * harness where the app's right dock puts it.
 *
 * `data/tiles/synthetic-yard/instances/` is the yard segmented against its own ground truth;
 * the route links it from the root's extras, as e2e/instances.spec.ts does. Screenshots go to
 * `SELECT_SHOTS_DIR` when set, else the test's output directory.
 */

import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}
interface State {
  candidates: number[];
  chain: number;
  index: number;
  selected: number | null;
  mode: string;
  paint: { best: number | null; iou: number; painted: number } | null;
  hidden: number[];
  highlighted: number[];
  custom: number;
}
/** `src/dev/sceneSelectHarness.ts`, as the page exposes it. */
interface Harness {
  view(id: number, headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  screenOf(id: number): { x: number; y: number; rect: Rect; splats: number } | null;
  instances(): { id: number; parent: number | null; splats: number }[];
  state(): State;
  coverage(rect?: Rect): number;
  hold(rect: Rect): void;
  changed(rect: Rect): number;
  frames(count: number): Promise<void>;
}

const TILES = resolve(process.cwd(), "../../data/tiles");
const TILESET = "synthetic-yard/splat/tileset.json";
const INSTANCES = "../instances/instances.json";
/** The yard's trees and shrubs, as segmented: a tree with its crown, a shrub. */
const TREE = 1;
const CROWN = 25;
const SHRUB = 10;

function harnessHtml(renderer: string): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Scene select harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; overflow: hidden; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      // The React plugin's preamble, which Vite injects into its own pages (the chip is React).
      import RefreshRuntime from "/@react-refresh";
      RefreshRuntime.injectIntoGlobalHook(window);
      window.$RefreshReg$ = () => {};
      window.$RefreshSig$ = () => (type) => type;
      window.__vite_plugin_react_preamble_installed__ = true;
    </script>
    <script type="module">
      const harness = await import("/src/dev/sceneSelectHarness.ts");
      window.__select = await harness.startSceneSelectHarness({
        container: document.getElementById("viewer"),
        url: "/fixture-tiles/${TILESET}",
        renderer: "${renderer}",
      });
    </script>
  </body>
</html>`;
}

async function open(page: Page, renderer: string, errors: string[]): Promise<void> {
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
        nativeLod: false,
      };
      return route.fulfill({ status: 200, json: tileset });
    }
    const file = resolve(TILES, relative);
    if (!existsSync(file)) return route.fulfill({ status: 404, body: "" });
    return route.fulfill({
      status: 200,
      contentType: relative.endsWith(".json") ? "application/json" : "model/gltf-binary",
      body: readFileSync(file),
    });
  });
  await page.route("**/scene-select-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml(renderer) }),
  );
  await page.goto("/scene-select-harness.html");
  await page.waitForFunction(() => "__select" in window, undefined, { timeout: 120_000 });
}

function caller(page: Page) {
  return <K extends keyof Harness>(
    method: K,
    ...args: Parameters<Harness[K]>
  ): Promise<Awaited<ReturnType<Harness[K]>>> =>
    page.evaluate(
      ([m, a]) => {
        const harness = (window as unknown as { __select: Record<string, unknown> }).__select;
        return (harness[m] as (...x: unknown[]) => unknown)(...a);
      },
      [method, args] as [string, unknown[]],
    ) as Promise<Awaited<ReturnType<Harness[K]>>>;
}

function shot(name: string): string {
  const dir = process.env.SELECT_SHOTS_DIR;
  if (!dir) return test.info().outputPath(name);
  mkdirSync(dir, { recursive: true });
  return join(dir, name);
}

for (const renderer of ["playcanvas", "spark", "cesium"] as const) {
  test(`under ${renderer}, a click selects a tree, ] cycles, the brush selects a shrub and Hide hides it`, async ({
    page,
  }) => {
    test.setTimeout(900_000);
    const errors: string[] = [];
    await open(page, renderer, errors);
    const call = caller(page);
    // Waits for the instances, too.
    await call("view", TREE, 30, -25, 28);
    const list = await call("instances");
    expect(list.length).toBeGreaterThan(0);
    const parentOf = new Map(list.map((i) => [i.id, i.parent]));
    const topOf = (id: number | null): number | null => {
      let at = id;
      for (let up = at === null ? null : (parentOf.get(at) ?? null); up !== null;) {
        at = up;
        up = parentOf.get(at) ?? null;
      }
      return at;
    };

    // ---- Click a tree --------------------------------------------------------------------
    const crown = await call("screenOf", CROWN);
    expect(crown, "the crown is on screen").not.toBeNull();
    await page.mouse.click(crown?.x ?? 0, crown?.y ?? 0);
    const card = page.getByTestId("selection-card");
    await expect(card).toBeVisible();
    await expect(card).toHaveAttribute("data-kind", "object");
    const picked = await call("state");
    await page.screenshot({ path: shot(`${renderer}-1-click-tree.png`) });
    expect(topOf(picked.selected)).toBe(TREE);
    expect(picked.highlighted).toContain(picked.selected);
    await expect(card.getByTestId("object-label")).not.toBeEmpty();
    await expect(card.getByTestId("object-candidates")).toHaveText(/ of /);
    // The hit gave the map the keyboard.
    expect(await page.evaluate(() => document.activeElement?.tagName)).toBe("CANVAS");

    // ---- Cycle: to the parent (or round to a child) ---------------------------------------
    await page.keyboard.press("]");
    const cycled = await call("state");
    await page.screenshot({ path: shot(`${renderer}-2-cycled.png`) });
    const before = picked.selected ?? -1;
    const after = cycled.selected ?? -1;
    expect(after).not.toBe(before);
    if (picked.index < picked.chain - 1) expect(after).toBe(parentOf.get(before));
    await page.keyboard.press("[");
    expect((await call("state")).selected).toBe(before);
    // Tab cycles from the map, as `]` does, and Shift+Tab comes back.
    await page.keyboard.press("Tab");
    expect((await call("state")).selected).toBe(after);
    await page.keyboard.press("Shift+Tab");
    expect((await call("state")).selected).toBe(before);
    // From the page's body Tab moves focus, and leaves the selection alone.
    await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
    await page.keyboard.press("Tab");
    expect((await call("state")).selected).toBe(before);
    expect(await page.evaluate(() => document.activeElement !== document.body)).toBe(true);
    await page.keyboard.press("Escape");
    expect((await call("state")).selected).toBeNull();
    await expect(card).toBeHidden();

    // ---- Paint over a shrub --------------------------------------------------------------
    await call("view", SHRUB, 20, -40, 7);
    const shrub = await call("screenOf", SHRUB);
    expect(shrub, "the shrub is on screen").not.toBeNull();
    const rect = shrub?.rect ?? { x: 0, y: 0, width: 0, height: 0 };
    await call("hold", rect);
    await page.keyboard.press("b");
    expect((await call("state")).mode).toBe("paint");
    // Back and forth over the shrub's middle, inside its outline.
    const inset = 0.15;
    const x0 = rect.x + rect.width * inset;
    const x1 = rect.x + rect.width * (1 - inset);
    const rows = Math.max(3, Math.ceil((rect.height * (1 - 2 * inset)) / 18));
    await page.mouse.move(x0, rect.y + rect.height * inset);
    await page.mouse.down();
    for (let r = 0; r <= rows; r += 1) {
      const y = rect.y + rect.height * inset + (rect.height * (1 - 2 * inset) * r) / rows;
      await page.mouse.move(r % 2 ? x1 : x0, y, { steps: 4 });
      await page.mouse.move(r % 2 ? x0 : x1, y, { steps: 8 });
    }
    await page.mouse.up();
    const painted = await call("state");
    await page.screenshot({ path: shot(`${renderer}-3-painted-shrub.png`) });
    expect(painted.paint?.painted ?? 0).toBeGreaterThan(0);
    expect(topOf(painted.selected)).toBe(SHRUB);
    expect(painted.paint?.iou ?? 0).toBeGreaterThan(0.3);

    // ---- Hide it from the card ------------------------------------------------------------
    await card.getByRole("button", { name: "Hide" }).click();
    await call("frames", 30);
    const hidden = await call("state");
    const change = await call("changed", rect);
    await page.screenshot({ path: shot(`${renderer}-4-hidden-shrub.png`) });
    expect(hidden.hidden).toContain(painted.selected);
    // The shrub is gone from where it was drawn: the lawn behind it shows instead.
    expect(change).toBeGreaterThan(0.05);

    test.info().annotations.push({
      type: "measures",
      description: JSON.stringify({ picked, cycled, painted, change, rect }),
    });
    expect(errors.filter((e) => /shader|compile|link|webgl/i.test(e))).toEqual([]);
  });
}
