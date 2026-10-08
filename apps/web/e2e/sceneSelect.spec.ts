/**
 * Selecting the synthetic yard's objects in the scene (cesium/sceneSelect, docs/SCENE_OBJECTS.md
 * "Selecting in the scene"), with the real mouse and keyboard, under PlayCanvas (the app's
 * default renderer) and Spark and CesiumJS: a click on a tree's crown selects the whole tree
 * and gives the map the keyboard, `[` and Shift+Tab cycle to the part hit and `]` and Tab back
 * (Tab only from the map or the card: from the page's body it moves focus), a second click
 * there selects that part and a double-click only the tree. The brush selects the whole
 * objects a stroke falls on (lib/sceneSelect.ts `paintPick`): a short stroke on the top of the
 * tree's crown selects the whole tree, and `[` the part of it painted; painted over a shrub it
 * lights the shrub up before the stroke ends and selects it when it does, and Hide in the
 * selection card hides it (its pixels change to the lawn behind it); one stroke from one shrub
 * to another selects both, not the lawn between them. Then one stroke across two walls of the
 * shed selects the shed, `[` the two walls painted together (`paintLevels`), and the card names
 * the combination and acts on it as one: Show only leaves both walls (the roof goes), Hide
 * hides both. The card is the HUD's (features/sites/ObjectCard.tsx), mounted by the harness
 * where the app's right dock puts it.
 *
 * `data/tiles/synthetic-yard/variants/objects/parts/` is the yard segmented against its own
 * ground truth, tagged and categorised as the real scans' files are (the same ids, hierarchy
 * and tiles as `instances/`, whose instances carry neither): the lawn is Grass & ground cover
 * and the path Paths & roads, the ground the brush leaves out beside objects. The route links
 * it from the root's extras, as e2e/instances.spec.ts does. Screenshots go to
 * `SELECT_SHOTS_DIR` when set, else the test's output directory.
 *
 * With `PUBLISHED_SCANS` set, the published spool and pumpkin too, under every objects variant
 * (the end of this file).
 */

import { execFile } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { promisify } from "node:util";

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
    rule?: string;
  } | null;
  hidden: number[];
  highlighted: number[];
  custom: number;
}
/** `src/dev/sceneSelectHarness.ts`, as the page exposes it. */
interface Harness {
  view(id: number | number[], headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  screenOf(id: number): { x: number; y: number; rect: Rect; splats: number } | null;
  visibleOf(id: number, top?: number): { x: number; y: number; rect: Rect; splats: number } | null;
  breakdown(points: { x: number; y: number }[]): {
    cells: number;
    dab: number;
    objects: { id: number; cover: number; share: number; ground: boolean }[];
  } | null;
  instances(): { id: number; parent: number | null; splats: number }[];
  state(): State;
  coverage(rect?: Rect): number;
  hold(rect: Rect): void;
  changed(rect: Rect): number;
  frames(count: number): Promise<void>;
}

const TILES = resolve(process.cwd(), "../../data/tiles");
const TILESET = "synthetic-yard/splat/tileset.json";
const INSTANCES = "../variants/objects/parts/instances.json";
/** The yard's trees and shrubs, as segmented: a tree with its crown, a shrub. */
const TREE = 1;
const CROWN = 25;
const SHRUB = 10;
/** Two more shrubs of the cluster, a little apart, and the lawn they stand on. */
const SHRUB_PAIR = [11, 13] as const;
const LAWN = [2, 4, 6] as const;
/** The shed: its roof, and two of its walls (one each side of a corner from where it is seen). */
const SHED = 8;
const ROOF = 90;
const WALLS = [91, 92] as const;

function harnessHtml(renderer: string, url = `/fixture-tiles/${TILESET}`): string {
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
        url: "${url}",
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
    readFileSync(resolve(TILES, "synthetic-yard/variants/objects/parts/instances.json"), "utf-8"),
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
  test(`under ${renderer}, a click selects a tree and again its part, [ cycles, a short stroke on the crown's top selects the whole tree, the brush selects a shrub and Hide hides it, one stroke selects two shrubs, and one across two walls the shed, then [ both walls`, async ({
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

    // ---- A short stroke on the top of the crown: the whole tree, `[` the part painted -----
    const crownTop = await call("visibleOf", TREE, 0.15);
    expect(crownTop?.splats ?? 0, "the crown's top in sight").toBeGreaterThan(30);
    await page.keyboard.press("b");
    expect((await call("state")).mode).toBe("paint");
    const topX = crownTop?.x ?? 0;
    const topY = crownTop?.y ?? 0;
    await page.mouse.move(topX - 25, topY);
    await page.mouse.down();
    await page.mouse.move(topX + 25, topY, { steps: 8 });
    await page.waitForTimeout(300);
    await call("frames", 2);
    const topLive = await call("state");
    expect(topLive.paint?.live).toBe(true);
    expect(topLive.paint?.ids).toEqual([TREE]);
    expect(topLive.highlighted).toContain(TREE);
    await page.mouse.up();
    const wholeTree = await call("state");
    await page.screenshot({ path: shot(`${renderer}-3b-short-stroke-tree.png`) });
    expect(wholeTree.selectedIds).toEqual([TREE]);
    expect(wholeTree.paint?.rule).toBe("objects");
    await expect(card.getByTestId("object-label")).toHaveText("Tree");
    await expect(card.getByTestId("object-candidates")).toHaveText(/^1 of /);
    // `[`: the parts of it the stroke covers; `]` back to the whole tree.
    await page.keyboard.press("[");
    const treeParts = await call("state");
    expect(treeParts.selectedIds.length).toBeGreaterThan(0);
    for (const id of treeParts.selectedIds) expect(parentOf.get(id)).toBe(TREE);
    await page.keyboard.press("]");
    expect((await call("state")).selectedIds).toEqual([TREE]);
    // The brush away, then the selection.
    await page.keyboard.press("Escape");
    expect((await call("state")).mode).toBe("pick");
    await page.keyboard.press("Escape");
    expect((await call("state")).selected).toBeNull();

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
    // The whole shrub, as the brush selects.
    expect(live.paint?.ids).toEqual([SHRUB]);
    expect(live.highlighted).toContain(SHRUB);
    expect(live.selected).toBeNull();
    await expect(card.getByTestId("object-paint-hint")).toContainText("Under the stroke: Shrub");
    await page.mouse.up();
    const painted = await call("state");
    await page.screenshot({ path: shot(`${renderer}-5-painted-shrub.png`) });
    expect(painted.paint?.live).toBeFalsy();
    expect(painted.paint?.painted ?? 0).toBeGreaterThan(0);
    expect(painted.selectedIds).toEqual([SHRUB]);
    expect(painted.paint?.iou ?? 0).toBeGreaterThan(0.3);
    expect(painted.highlighted).toContain(SHRUB);

    // ---- Hide it from the card ------------------------------------------------------------
    await card.getByRole("button", { name: "Hide" }).click();
    await call("frames", 30);
    const hidden = await call("state");
    const change = await call("changed", rect);
    await page.screenshot({ path: shot(`${renderer}-6-hidden-shrub.png`) });
    expect(hidden.hidden).toContain(painted.selected);
    // The shrub is gone from where it was drawn: the lawn behind it shows instead.
    expect(change).toBeGreaterThan(0.05);

    // ---- One stroke from one shrub to another: both, not the lawn between them -----------
    await call("view", [...SHRUB_PAIR], 20, -40, 7);
    const pair = await Promise.all(SHRUB_PAIR.map((id) => call("visibleOf", id)));
    for (const each of pair) expect(each?.splats ?? 0, "a shrub in sight").toBeGreaterThan(30);
    await page.mouse.move(pair[0]?.x ?? 0, pair[0]?.y ?? 0);
    await page.mouse.down();
    await page.mouse.move(pair[1]?.x ?? 0, pair[1]?.y ?? 0, { steps: 12 });
    await page.mouse.up();
    const twoShrubs = await call("state");
    await page.screenshot({ path: shot(`${renderer}-6b-painted-two-shrubs.png`) });
    for (const id of SHRUB_PAIR) expect(twoShrubs.selectedIds).toContain(id);
    for (const id of LAWN) expect(twoShrubs.selectedIds).not.toContain(id);
    for (const id of twoShrubs.selectedIds) expect(parentOf.get(id) ?? null).toBeNull();
    expect(twoShrubs.combination?.ids).toEqual(twoShrubs.selectedIds);
    // Two shrubs: "2 objects" (a name said twice), or each by its name.
    await expect(card.getByTestId("object-label")).toHaveText(/objects| \+ /);
    await expect(card).not.toContainText("overlap with the painted area");

    // ---- One stroke across two walls of the shed: the shed, then `[` both walls ----------
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
    // The whole shed lights up as the stroke goes on, though its roof was not painted.
    expect(both.paint?.ids).toEqual([SHED]);
    expect(both.highlighted).toContain(SHED);
    await page.mouse.up();
    const shed = await call("state");
    await page.screenshot({ path: shot(`${renderer}-8-painted-walls.png`) });
    expect(shed.selectedIds).toEqual([SHED]);
    const label = card.getByTestId("object-label");
    await expect(label).toHaveText("Shed");
    // `[`: the walls the stroke covers, together -- not the roof, nor one wall.
    await page.keyboard.press("[");
    const together = await call("state");
    await page.screenshot({ path: shot(`${renderer}-8b-walls.png`) });
    const ids = together.selectedIds;
    expect(ids.every((id) => topOf(id) === SHED)).toBe(true);
    for (const wall of WALLS) expect(ids).toContain(wall);
    expect(ids).not.toContain(SHED);
    expect(ids).not.toContain(ROOF);
    expect(together.combination?.ids).toEqual(ids);
    await expect(label).toHaveAttribute("data-ids", ids.join(" "));
    await expect(label).toHaveText(/ \+ |parts of/);
    await expect(card).not.toContainText("overlap with the painted area");
    await expect(card.getByTestId("object-save-combination")).toBeVisible();
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
    expect(only.hidden).toContain(ROOF);
    expect(roofChange).toBeGreaterThan(0.05);
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
        topLive,
        wholeTree,
        treeParts,
        live,
        painted,
        change,
        rect,
        twoShrubs,
        both,
        shed,
        together,
        roofChange,
        wallChange,
      }),
    });
    expect(errors.filter((e) => /shader|compile|link|webgl/i.test(e))).toEqual([]);
  });
}

// ---- The published spool and pumpkin, under every objects variant -----------------------

/*
 * A manual check on the live scans, skipped unless `PUBLISHED_SCANS` is set: the spool and the
 * pumpkin as published (each asset's tileset, tiles and instances.json, fetched from the public
 * bucket by `curl` -- it refuses some clients -- and cached in `PUBLISHED_CACHE`, as
 * e2e/instancesPublished.spec.ts does), with today's instances or one of the segmentation
 * variants' (`extras.variants.objects`) in their place. Per scan, variant
 * (`PUBLISHED_VARIANTS`, default all five) and renderer (`PUBLISHED_RENDERERS`, default all
 * three): a short stroke on the spool's top selects the whole spool, and so does painting
 * across it; a stroke from one pumpkin to the other selects both. What each selected (ids,
 * the card's name, the levels `[` steps through) goes to `PUBLISHED_SHOTS` with the
 * screenshots, else the test's output directory.
 *
 *   PUBLISHED_SCANS=1 PUBLISHED_SHOTS=/some/dir npx playwright test e2e/sceneSelect.spec.ts
 */

const run = promisify(execFile);
const BUCKET = "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/runs";
const CACHE = process.env.PUBLISHED_CACHE ?? join(tmpdir(), "hexapod-published");
const SHOTS = process.env.PUBLISHED_SHOTS;
const RENDERERS = (process.env.PUBLISHED_RENDERERS ?? "playcanvas,spark,cesium").split(",");
const VARIANTS = (
  process.env.PUBLISHED_VARIANTS ??
  "today,ground-first,feature-fields,concept-first-standin,concept-first"
).split(",");

interface PublishedScan {
  name: "spool" | "pumpkin";
  /** The splat package in the bucket: the asset's `source.url` (twin-api /api/v1/assets/{id}). */
  root: string;
  /** Per objects variant, what is painted over: the spool, or the two pumpkins. */
  objects: Record<string, number[]>;
  /** Heading, pitch (degrees) and range (the scan's units) the objects are seen from. */
  view: [number, number, number];
}

const PUBLISHED: PublishedScan[] = [
  {
    // Asset 29ad7e37-9ad1-42a8-a61d-a39e08ac6710. The spool is instance 1 in every file.
    name: "spool",
    root: "8e1cc115-cb80-4af2-81fc-dccaf6b65891/pf151f49bd03a5380/package/splat",
    objects: {
      today: [1],
      "ground-first": [1],
      "feature-fields": [1],
      "concept-first-standin": [1],
      "concept-first": [1],
    },
    view: [30, -35, 9],
  },
  {
    // Asset 695e9556-b376-48e5-ada5-1e9e7009cb21: the orange pumpkin, then the other.
    name: "pumpkin",
    root: "430c1932-5b6a-47b1-bb71-bb7fa2fec86b/pc936e13048d5e58f/package/splat",
    objects: {
      today: [2, 3],
      "ground-first": [1, 2],
      "feature-fields": [1, 2],
      "concept-first-standin": [2, 3],
      "concept-first": [2, 3],
    },
    view: [30, -40, 9],
  },
];

/** A file of the bucket, from the cache or fetched into it; null when the bucket has none. */
async function fromBucket(path: string): Promise<Buffer | null> {
  const file = join(CACHE, path);
  const missing = `${file}.404`;
  if (existsSync(file)) return readFileSync(file);
  if (existsSync(missing)) return null;
  mkdirSync(dirname(file), { recursive: true });
  const { stdout } = await run(
    "curl",
    ["-sS", "-A", "curl/8.5.0", "-o", file, "-w", "%{http_code}", `${BUCKET}/${path}`],
    { maxBuffer: 1 << 20 },
  );
  if (stdout.trim() !== "200") {
    writeFileSync(missing, "");
    return null;
  }
  return readFileSync(file);
}

test.describe("the published spool and pumpkin", () => {
  test.skip(!process.env.PUBLISHED_SCANS, "PUBLISHED_SCANS is not set");

  for (const scan of PUBLISHED) {
    for (const variant of VARIANTS) {
      for (const renderer of RENDERERS) {
        const what =
          scan.name === "spool"
            ? "a short stroke on the spool's top selects the whole spool, as painting across it does"
            : "a stroke across both pumpkins selects both";
        test(`${scan.name} with the ${variant} objects under ${renderer}: ${what}`, async ({
          page,
        }) => {
          test.setTimeout(1_800_000);
          const errors: string[] = [];
          page.on("pageerror", (error) => errors.push(error.message));
          const instances = `${scan.root}/instances.json`;
          await page.route("**/published/**", async (route) => {
            let path = new URL(route.request().url()).pathname.replace(/^.*\/published\//, "");
            if (path.includes("..")) return route.abort();
            // The variant's objects in place of today's.
            if (variant !== "today" && path === instances)
              path = `${scan.root}/variants/objects/${variant}/instances.json`;
            const body = await fromBucket(path);
            if (!body) return route.fulfill({ status: 404, body: "" });
            const contentType = path.endsWith(".json")
              ? "application/json"
              : path.endsWith(".glb")
                ? "model/gltf-binary"
                : "application/octet-stream";
            return route.fulfill({ status: 200, contentType, body });
          });
          await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) =>
            route.abort(),
          );
          await page.route("**/scene-select-harness.html", (route) =>
            route.fulfill({
              status: 200,
              contentType: "text/html",
              body: harnessHtml(renderer, `/published/${scan.root}/tileset.json`),
            }),
          );
          await page.goto("/scene-select-harness.html");
          await page.waitForFunction(() => "__select" in window, undefined, { timeout: 600_000 });
          const call = caller(page);
          const objects = scan.objects[variant] ?? [];
          await call("view", objects, ...scan.view);
          const label = page.getByTestId("selection-card").getByTestId("object-label");
          const file = (name: string): string => {
            const base = `${scan.name}-${variant}-${renderer}-${name}`;
            if (!SHOTS) return test.info().outputPath(base);
            mkdirSync(SHOTS, { recursive: true });
            return join(SHOTS, base);
          };
          /** What is selected now, as the card names it, and the levels `[` steps through. */
          const selected = async () => {
            const state = await call("state");
            return {
              ids: state.selectedIds,
              label: (await label.textContent()) ?? "",
              levels: state.candidates.length,
              rule: state.paint?.rule ?? null,
              painted: state.paint?.painted ?? 0,
            };
          };
          /**
           * Paints a stroke through `points` and says what it selected, what `[` steps down to,
           * and how the stroke fell on each object (`breakdown`, worked out before it).
           */
          const paint = async (name: string, points: { x: number; y: number }[]) => {
            const shares = await call("breakdown", points);
            const first = points[0] ?? { x: 0, y: 0 };
            await page.mouse.move(first.x, first.y);
            await page.mouse.down();
            for (const point of points.slice(1)) {
              await page.mouse.move(point.x, point.y, { steps: 8 });
            }
            await page.mouse.up();
            const done = await selected();
            await page.screenshot({ path: file(`${name}.png`) });
            let parts = null;
            if (done.levels > 1) {
              await page.keyboard.press("[");
              parts = await selected();
              await page.keyboard.press("]");
            }
            const objectsUnder = (shares?.objects ?? []).slice(0, 6).map((o) => ({
              id: o.id,
              share: Number(o.share.toFixed(3)),
              cells: Math.round(o.cover),
              ground: o.ground,
            }));
            return { ...done, parts, dab: Math.round(shares?.dab ?? 0), objectsUnder };
          };
          const results: Record<string, unknown> = { scan: scan.name, variant, renderer };
          const checks: (() => void)[] = [];
          await page.keyboard.press("b");
          if (scan.name === "spool") {
            const spool = objects[0] ?? 0;
            // ---- A short stroke on the top of the spool: the whole spool ----------------
            const top = await call("visibleOf", spool, 0.15);
            expect(top?.splats ?? 0, "the spool's top in sight").toBeGreaterThan(30);
            const x = top?.x ?? 0;
            const y = top?.y ?? 0;
            const short = await paint("1-short-stroke", [
              { x: x - 30, y },
              { x: x + 30, y },
            ]);
            results.short = short;
            checks.push(() => expect(short.ids, JSON.stringify(short)).toEqual([spool]));
            // ---- Painting across the spool: the spool ------------------------------------
            const whole = await call("visibleOf", spool);
            const box = whole?.rect ?? { x: 0, y: 0, width: 0, height: 0 };
            const at = (fx: number, fy: number) => ({
              x: box.x + box.width * fx,
              y: box.y + box.height * fy,
            });
            const zigzag = [at(0.2, 0.2)];
            for (let r = 0; r <= 4; r += 1) {
              zigzag.push(at(r % 2 ? 0.2 : 0.8, 0.2 + 0.15 * r));
              if (r < 4) zigzag.push(at(r % 2 ? 0.2 : 0.8, 0.35 + 0.15 * r));
            }
            const across = await paint("2-across", zigzag);
            results.across = across;
            // The spool; under concept first its bottom flange is a second Cable spool (4).
            const also = variant === "concept-first" ? [4] : [];
            checks.push(() => {
              expect(across.ids, JSON.stringify(across)).toContain(spool);
              for (const id of across.ids)
                expect([spool, ...also], JSON.stringify(across)).toContain(id);
            });
          } else {
            // ---- A stroke from one pumpkin to the other: both ----------------------------
            const ends = await Promise.all(objects.map((id) => call("visibleOf", id)));
            for (const end of ends)
              expect(end?.splats ?? 0, "a pumpkin in sight").toBeGreaterThan(30);
            const across = await paint(
              "1-across",
              ends.map((end) => ({ x: end?.x ?? 0, y: end?.y ?? 0 })),
            );
            results.across = across;
            checks.push(() => {
              for (const id of objects) expect(across.ids, JSON.stringify(across)).toContain(id);
            });
          }
          test.info().annotations.push({ type: "selected", description: JSON.stringify(results) });
          writeFileSync(file("selected.json"), JSON.stringify(results, null, 1));
          console.info(JSON.stringify(results));
          for (const check of checks) check();
          expect(errors.filter((e) => /shader|compile|link|webgl/i.test(e))).toEqual([]);
        });
      }
    }
  }
});
