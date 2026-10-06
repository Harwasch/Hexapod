/**
 * Undo and redo on a scan with objects, with the real keyboard (state/history.ts,
 * features/shell/undoHotkeys.ts, docs/SCENE_OBJECTS.md "Undo"): on the synthetic yard, the
 * selection card hides the tree, shows only a shrub, and Ctrl+Z twice brings back exactly
 * what was shown before; Ctrl+Shift+Z twice does both again. The page is the scene selection
 * harness (src/dev/sceneSelectHarness.ts, as e2e/sceneSelect.spec.ts drives it) with the app's
 * undo keys and toasts mounted beside it (src/dev/undoHarness.ts). Screenshots go to the
 * test's output directory.
 */

import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");
const TILESET = "synthetic-yard/splat/tileset.json";
const INSTANCES = "../instances/instances.json";
const ASSET = "harness";
/** The yard's tree and a shrub, as segmented (e2e/sceneSelect.spec.ts). */
const TREE = 1;
const SHRUB = 10;
const RENDERER = "playcanvas";

const HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Undo harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; overflow: hidden; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      import RefreshRuntime from "/@react-refresh";
      RefreshRuntime.injectIntoGlobalHook(window);
      window.$RefreshReg$ = () => {};
      window.$RefreshSig$ = () => (type) => type;
      window.__vite_plugin_react_preamble_installed__ = true;
    </script>
    <script type="module">
      (await import("/src/dev/undoHarness.ts")).mountUndo();
      const harness = await import("/src/dev/sceneSelectHarness.ts");
      window.__select = await harness.startSceneSelectHarness({
        container: document.getElementById("viewer"),
        url: "/fixture-tiles/${TILESET}",
        renderer: "${RENDERER}",
      });
    </script>
  </body>
</html>`;

async function open(page: Page, errors: string[]): Promise<void> {
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
  await page.route("**/undo-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: HTML }),
  );
  await page.goto("/undo-harness.html");
  await page.waitForFunction(() => "__select" in window, undefined, { timeout: 120_000 });
}

/** The objects store's hidden ids, sorted. */
function hidden(page: Page): Promise<number[]> {
  return page.evaluate(() => {
    const harness = (window as unknown as { __select: { state(): { hidden: number[] } } }).__select;
    return [...harness.state().hidden].sort((a, b) => a - b);
  });
}

/** Selects instance `id` of the yard as a click on it would (the stores the app uses). */
function select(page: Page, id: number): Promise<void> {
  return page.evaluate(`(async () => {
    const { useSceneSelect } = await import("/src/state/sceneSelect.ts");
    useSceneSelect.getState().select("${ASSET}", [${String(id)}], 1, 0, null);
  })()`);
}

test(`under ${RENDERER}, Ctrl+Z takes back the card's Hide and Show only, and Ctrl+Shift+Z does them again`, async ({
  page,
}) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  await open(page, errors);
  // Waits for the instances, too.
  await page.evaluate(
    ([id]) =>
      (window as unknown as { __select: { view(...a: number[]): Promise<void> } }).__select.view(
        id ?? 0,
        30,
        -30,
        30,
      ),
    [TREE],
  );
  const card = page.getByTestId("selection-card");
  const toast = page.getByTestId("undo-harness").getByRole("status");
  expect(await hidden(page)).toEqual([]);

  // ---- Hide the tree, then show only a shrub, from the card -----------------------------
  await select(page, TREE);
  await card.getByRole("button", { name: "Hide" }).click();
  await expect.poll(() => hidden(page)).toContain(TREE);
  const afterHide = await hidden(page);
  expect(afterHide).not.toContain(SHRUB);

  await select(page, SHRUB);
  await card.getByRole("button", { name: "Show only" }).click();
  await expect.poll(() => hidden(page)).not.toEqual(afterHide);
  const afterShowOnly = await hidden(page);
  expect(afterShowOnly).not.toContain(SHRUB);
  expect(afterShowOnly).toContain(TREE);
  expect(afterShowOnly.length).toBeGreaterThan(afterHide.length);
  await page.screenshot({ path: test.info().outputPath("1-show-only-shrub.png") });

  // ---- Ctrl+Z twice: what was shown before, exactly --------------------------------------
  await page.keyboard.press("Control+z");
  await expect.poll(() => hidden(page)).toEqual(afterHide);
  await expect(toast).toContainText(/^Undid: Show only /);
  await expect(toast.getByRole("button", { name: "Redo" })).toBeVisible();
  await page.keyboard.press("Control+z");
  await expect.poll(() => hidden(page)).toEqual([]);
  await expect(toast).toContainText(/^Undid: Hide /);
  await page.screenshot({ path: test.info().outputPath("2-undone.png") });
  await page.keyboard.press("Control+z");
  await expect(toast).toContainText("Nothing to undo");

  // ---- Ctrl+Shift+Z twice: both again ----------------------------------------------------
  await page.keyboard.press("Control+Shift+Z");
  await expect.poll(() => hidden(page)).toEqual(afterHide);
  await expect(toast).toContainText(/^Redid: Hide /);
  await page.keyboard.press("Control+Shift+Z");
  await expect.poll(() => hidden(page)).toEqual(afterShowOnly);
  await expect(toast).toContainText(/^Redid: Show only /);
  // The toast's Undo is one more way back.
  await toast.getByRole("button", { name: "Undo" }).click();
  await expect.poll(() => hidden(page)).toEqual(afterHide);
  await page.screenshot({ path: test.info().outputPath("3-redone-and-undone.png") });

  expect(errors.filter((e) => /shader|compile|link|webgl/i.test(e))).toEqual([]);
});
