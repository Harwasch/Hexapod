/**
 * Selecting the synthetic yard's objects in the scene (cesium/sceneSelect, docs/SCENE_OBJECTS.md
 * "Selecting in the scene"), with the real mouse and keyboard, under PlayCanvas (the app's
 * default renderer) and Spark and CesiumJS: a click on a tree's crown selects the whole tree
 * and gives the map the keyboard, `[` and Shift+Tab cycle to the part hit and `]` and Tab back
 * (Tab only from the map or the card: from the page's body it moves focus), a second click
 * there selects that part and a double-click only the tree, the brush painted over a shrub
 * lights it up before the stroke ends and selects it when it does, and Hide in the selection
 * card hides it (its pixels change to the lawn behind it). Then one stroke across two walls of
 * the shed selects both together (lib/sceneSelect.ts `bestSet`) -- not the shed, whose roof
 * was not painted, nor one wall -- and the card names the combination and acts on it as one:
 * Show only leaves both walls (the roof goes), Hide hides both. The card is the HUD's
 * (features/sites/ObjectCard.tsx), mounted by the harness where the app's right dock puts it.
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
  selectedIds: number[];
  combination: { ids: number[]; iou: number } | null;
  mode: string;
  paint: {
    ids: number[];
    best: number | null;
    iou: number;
    painted: number;
    live?: boolean;
  } | null;
  hidden: number[];
  highlighted: number[];
  custom: number;
}
/** `src/dev/sceneSelectHarness.ts`, as the page exposes it. */
interface Harness {
  view(id: number, headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  screenOf(id: number): { x: number; y: number; rect: Rect; splats: number } | null;
  visibleOf(id: number): { x: number; y: number; rect: Rect; splats: number } | null;
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
/** The shed: its roof, and two of its walls (one each side of a corner from where it is seen). */
const SHED = 8;
const ROOF = 90;
const WALLS = [91, 92] as const;

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
  test(`under ${renderer}, a click selects a tree and again its part, [ cycles, the brush selects a shrub and Hide hides it, and two walls painted in one stroke are selected together`, async ({
    page,
  }) => {
    // Four views of the yard on software GL, each waiting for its tiles to settle.
    test.setTimeout(1_500_000);
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
    // The whole tree first: the top of the chain the click hit, which holds the crown.
    expect(picked.selected).toBe(TREE);
    expect(picked.index).toBe(picked.chain - 1);
    expect(picked.chain, "the click hit a part of the tree").toBeGreaterThan(1);
    expect(picked.highlighted).toContain(picked.selected);
    await expect(card.getByTestId("object-label")).not.toBeEmpty();
    await expect(card.getByTestId("object-candidates")).toHaveText(/^1 of /);
    // The hit gave the map the keyboard.
    expect(await page.evaluate(() => document.activeElement?.tagName)).toBe("CANVAS");

    // ---- Cycle: `[` one level finer, `]` back up ------------------------------------------
    await page.keyboard.press("[");
    const cycled = await call("state");
    await page.screenshot({ path: shot(`${renderer}-2-cycled.png`) });
    const before = picked.selected ?? -1;
    const after = cycled.selected ?? -1;
    expect(parentOf.get(after)).toBe(before);
    await page.keyboard.press("]");
    expect((await call("state")).selected).toBe(before);
    // Shift+Tab cycles from the map, as `[` does, and Tab comes back.
    await page.keyboard.press("Shift+Tab");
    expect((await call("state")).selected).toBe(after);
    await page.keyboard.press("Tab");
    expect((await call("state")).selected).toBe(before);
    // From the page's body Tab moves focus, and leaves the selection alone.
    await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
    await page.keyboard.press("Tab");
    expect((await call("state")).selected).toBe(before);
    expect(await page.evaluate(() => document.activeElement !== document.body)).toBe(true);

    // ---- Click again: one level finer --------------------------------------------------------
    // Well after the first click: a click soon after it is the same click (a double-click).
    await page.waitForTimeout(500);
    await page.mouse.click(crown?.x ?? 0, crown?.y ?? 0);
    const drilled = await call("state");
    await page.screenshot({ path: shot(`${renderer}-3-click-again.png`) });
    expect(drilled.selected).not.toBe(before);
    expect(parentOf.get(drilled.selected ?? -1)).toBe(before);
    expect(drilled.highlighted).toContain(drilled.selected);
    await page.keyboard.press("Escape");
    expect((await call("state")).selected).toBeNull();
    await expect(card).toBeHidden();

    // ---- A double-click is one click: the whole tree, not a part of it ---------------------
    await page.waitForTimeout(500);
    await page.mouse.dblclick(crown?.x ?? 0, crown?.y ?? 0);
    expect((await call("state")).selected).toBe(TREE);
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
    // Before the stroke ends its best match so far is lit, and said on the card, but nothing
    // is selected yet. (The match is brought up to date at most every 100 ms.)
    await page.waitForTimeout(300);
    await call("frames", 2);
    const live = await call("state");
    await page.screenshot({ path: shot(`${renderer}-4-painting-shrub.png`) });
    expect(live.paint?.live).toBe(true);
    expect(topOf(live.paint?.best ?? null)).toBe(SHRUB);
    expect(live.highlighted).toContain(live.paint?.best);
    expect(live.selected).toBeNull();
    await expect(card.getByTestId("object-paint-hint")).toContainText("Best match");
    await page.mouse.up();
    const painted = await call("state");
    await page.screenshot({ path: shot(`${renderer}-5-painted-shrub.png`) });
    expect(painted.paint?.live).toBeFalsy();
    expect(painted.paint?.painted ?? 0).toBeGreaterThan(0);
    expect(topOf(painted.selected)).toBe(SHRUB);
    expect(painted.paint?.iou ?? 0).toBeGreaterThan(0.3);
    expect(painted.highlighted).toContain(painted.selected);

    // ---- Hide it from the card ------------------------------------------------------------
    await card.getByRole("button", { name: "Hide" }).click();
    await call("frames", 30);
    const hidden = await call("state");
    const change = await call("changed", rect);
    await page.screenshot({ path: shot(`${renderer}-6-hidden-shrub.png`) });
    expect(hidden.hidden).toContain(painted.selected);
    // The shrub is gone from where it was drawn: the lawn behind it shows instead.
    expect(change).toBeGreaterThan(0.05);

    // ---- One stroke across two walls of the shed: both, together -------------------------
    await page.keyboard.press("Escape");
    expect((await call("state")).mode).toBe("pick");
    await call("view", SHED, 20, -35, 10);
    // Where the brush meets each: its splats in front, as the brush sees them.
    const walls = await Promise.all(WALLS.map((wall) => call("visibleOf", wall)));
    const roof = await call("visibleOf", ROOF);
    for (const wall of walls) expect(wall?.splats ?? 0, "a wall in sight").toBeGreaterThan(100);
    expect(roof?.splats ?? 0, "the roof in sight").toBeGreaterThan(100);
    await page.keyboard.press("b");
    // Back and forth over the lower part of the first wall (below the roof), then on across to
    // the second.
    const none: Rect = { x: 0, y: 0, width: 0, height: 0 };
    const zigzag = (box: Rect): { x: number; y: number }[] => {
      const points: { x: number; y: number }[] = [];
      for (let r = 0; r <= 4; r += 1) {
        const y = box.y + box.height * (0.45 + 0.1 * r);
        const [from, to] = r % 2 ? [0.8, 0.2] : [0.2, 0.8];
        points.push({ x: box.x + box.width * from, y }, { x: box.x + box.width * to, y });
      }
      return points;
    };
    const path = [...zigzag(walls[0]?.rect ?? none), ...zigzag(walls[1]?.rect ?? none)];
    await page.mouse.move(path[0]?.x ?? 0, path[0]?.y ?? 0);
    await page.mouse.down();
    for (const point of path.slice(1)) await page.mouse.move(point.x, point.y, { steps: 6 });
    await page.waitForTimeout(300);
    await call("frames", 2);
    const both = await call("state");
    await page.screenshot({ path: shot(`${renderer}-7-painting-walls.png`) });
    expect(both.paint?.live).toBe(true);
    expect(both.paint?.ids.length ?? 0).toBeGreaterThan(0);
    for (const id of both.paint?.ids ?? []) expect(both.highlighted).toContain(id);
    await page.mouse.up();
    const together = await call("state");
    await page.screenshot({ path: shot(`${renderer}-8-painted-walls.png`) });
    // Both walls, as a combination -- or the shed, were it as good -- never one wall.
    const ids = together.selectedIds;
    expect(ids.every((id) => topOf(id) === SHED)).toBe(true);
    const label = card.getByTestId("object-label");
    if (together.combination) {
      for (const wall of WALLS) expect(ids).toContain(wall);
      expect(ids).not.toContain(SHED);
      await expect(label).toHaveAttribute("data-ids", ids.join(" "));
      await expect(label).toHaveText(/ \+ |parts of/);
      await expect(card).toContainText("overlap with the painted area");
      await expect(card.getByTestId("object-save-combination")).toBeVisible();
    } else {
      expect(ids).toEqual([SHED]);
    }
    for (const id of ids) expect(together.highlighted).toContain(id);
    // Show only leaves the selection, both walls: the roof and the lawn go.
    const roofTop = roof?.rect ?? none;
    const roofRect = { ...roofTop, height: roofTop.height * 0.4 };
    await call("hold", roofRect);
    await card.getByRole("button", { name: "Show only" }).click();
    await call("frames", 30);
    const only = await call("state");
    const roofChange = await call("changed", roofRect);
    await page.screenshot({ path: shot(`${renderer}-9-show-only-walls.png`) });
    for (const id of ids) expect(only.hidden).not.toContain(id);
    expect(only.hidden).toContain(TREE);
    if (together.combination) {
      expect(only.hidden).toContain(ROOF);
      expect(roofChange).toBeGreaterThan(0.05);
    }
    // Hide hides the selection, both walls.
    const wallBox = walls[0]?.rect ?? none;
    const wallRect = {
      x: wallBox.x + wallBox.width * 0.2,
      y: wallBox.y + wallBox.height * 0.45,
      width: wallBox.width * 0.6,
      height: wallBox.height * 0.4,
    };
    await call("hold", wallRect);
    await card.getByRole("button", { name: "Hide" }).click();
    await call("frames", 30);
    const gone = await call("state");
    const wallChange = await call("changed", wallRect);
    await page.screenshot({ path: shot(`${renderer}-10-hidden-walls.png`) });
    for (const id of ids) expect(gone.hidden).toContain(id);
    expect(gone.selectedIds).toEqual([]);
    expect(wallChange).toBeGreaterThan(0.05);

    test.info().annotations.push({
      type: "measures",
      description: JSON.stringify({
        picked,
        cycled,
        drilled,
        live,
        painted,
        change,
        rect,
        both,
        together,
        roofChange,
        wallChange,
      }),
    });
    expect(errors.filter((e) => /shader|compile|link|webgl/i.test(e))).toEqual([]);
  });
}
