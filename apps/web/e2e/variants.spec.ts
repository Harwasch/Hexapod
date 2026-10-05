/**
 * Comparing methods on one scan (lib/variants.ts): the synthetic yard with the bake-off
 * fixture's variants (`data/tiles/synthetic-yard/variants/`, tools/captures/yard_variants.py)
 * declared on its root as a publish declares them, drawn as the app draws a scan
 * (src/dev/variantsHarness.ts) under PlayCanvas (the app's default), Spark and CesiumJS.
 *
 * What must hold, with no reload and the camera where it is:
 *
 * - picking an objects variant in the app's "Compare methods" panel changes what the objects
 *   panel lists (the yard's two variants have other categories, and other counts);
 * - picking a fill variant draws its inferred layer and not the other's;
 * - Highlight changes the inferred layer's pixels -- purple -- and not one measured pixel;
 *   Hide removes the layer;
 * - picking a skins variant replaces the skin the scan's objects move by.
 *
 * Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");
const TILESET = "synthetic-yard/splat/tileset.json";

/** The fills' boxes in the yard's frame (tools/captures/yard_variants.py `hedge`, `mound`). */
const HEDGE = { min: [-3.2, 5.8, -0.2], max: [-1.6, 18.2, 1.8] };
const MOUND = { min: [32.8, 9.3, -0.4], max: [38.2, 14.7, 1.8] };

type Renderer = "playcanvas" | "spark" | "cesium";

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** `src/dev/variantsHarness.ts`, as the page exposes it. */
interface VariantsHarness {
  view(headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  settle(): Promise<void>;
  pick(system: string, name: string | null): Promise<void>;
  style(style: "show" | "highlight" | "hide"): Promise<void>;
  rectOfLocal(min: number[], max: number[]): Rect | null;
  measure(rect?: Rect): { coverage: number; purple: number };
  remember(slot?: string): void;
  changed(rect?: Rect, outside?: boolean, slot?: string): number;
  categories(): { name: string; objects: number }[];
  objects(): number;
  picks(): { picks: Record<string, string>; status: Record<string, { state: string }> };
  inferred(): { fillers: string[]; shown: number };
  skins(): number[] | null;
  scan(): {
    kind: string;
    active: boolean;
    tiles: number;
    motion: { skinned: number } | null;
  } | null;
}

function harnessHtml(renderer: Renderer): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Variants harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer { position: relative; width: max(200px, calc(100vw - 24rem)); height: 100vh; }
      #viewer .cesium-widget, #viewer .cesium-widget > canvas:first-child { width: 100%; height: 100vh; display: block; }
      #panel { position: fixed; top: 12px; right: 12px; z-index: 10; width: min(22rem, calc(100vw - 48px)); max-height: calc(100vh - 24px); overflow: auto; padding: 0.7rem; display: flex; flex-direction: column; gap: 0.8rem; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <div id="panel" class="glass glass--strong"></div>
    <script type="module">
      const harness = await import("/src/dev/variantsHarness.ts");
      window.__variants = await harness.startVariantsHarness({
        container: document.getElementById("viewer"),
        url: "/fixture-tiles/${TILESET}",
        renderer: "${renderer}",
        panel: document.getElementById("panel"),
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
  const read = (relative: string): Record<string, unknown> =>
    JSON.parse(readFileSync(resolve(TILES, relative), "utf-8")) as Record<string, unknown>;
  const { variants } = read("synthetic-yard/variants/variants.json");
  const instances = (read("synthetic-yard/instances/instances.json").instances ?? []) as unknown[];
  const skins = (read("synthetic-yard/skin/skin.json").skins ?? []) as unknown[];
  await page.route("**/fixture-tiles/**", (route) => {
    const relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    if (relative === TILESET) {
      const tileset = read(TILESET) as { root: { extras?: Record<string, unknown> } };
      // Today's objects and skin as the yard's other specs link them, and the variants as a
      // publish registers them (docs/SCENE_OBJECTS.md, "Variants").
      tileset.root.extras = {
        ...tileset.root.extras,
        instances: { uri: "../instances/instances.json", count: instances.length },
        skin: { uri: "../skin/skin.json", count: skins.length },
        variants,
        nativeLod: false,
      };
      return route.fulfill({ status: 200, json: tileset });
    }
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
  await page.route("**/variants-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml(renderer) }),
  );
  await page.goto("/variants-harness.html");
  try {
    await page.waitForFunction(() => "__variants" in window, undefined, { timeout: 180_000 });
  } catch (error) {
    throw new Error(`the harness never started; page errors: ${errors.join(" | ")}`, {
      cause: error,
    });
  }
}

function caller(page: Page) {
  return <K extends keyof VariantsHarness>(
    method: K,
    ...args: Parameters<VariantsHarness[K]>
  ): Promise<Awaited<ReturnType<VariantsHarness[K]>>> =>
    page.evaluate(
      ([m, a]) => {
        const harness = (window as unknown as { __variants: Record<string, unknown> }).__variants;
        return (harness[m] as (...x: unknown[]) => unknown)(...a);
      },
      [method, args] as [string, unknown[]],
    ) as Promise<Awaited<ReturnType<VariantsHarness[K]>>>;
}

const shaderErrors = (errors: string[]): string[] =>
  errors.filter((e) => /shader|compile|link|webgl/i.test(e));

/** The objects panel's category rows, as a person reads them. */
async function listed(page: Page): Promise<string[]> {
  const rows = page.getByRole("list", { name: "Object categories" }).locator(":scope > li");
  return (await rows.allInnerTexts()).map((t) => t.replace(/\s+/g, " ").trim());
}

for (const renderer of ["playcanvas", "spark", "cesium"] as const) {
  test(`methods switch in place under ${renderer}: objects, fill, inferred style, motion`, async ({
    page,
  }) => {
    test.setTimeout(1_500_000);
    const errors: string[] = [];
    await open(page, renderer, errors);
    const call = caller(page);
    // From the south, the yard in the middle, the hedge to the left, the mound to the right.
    await call("view", 0, -32, 52);
    if (renderer !== "cesium") expect((await call("scan"))?.active).toBe(true);

    // ---- Objects: the panel lists what the picked file says ----------------------------------
    const compare = page.getByTestId("compare-methods");
    await expect(compare.getByRole("group", { name: "Objects" })).toBeVisible();
    const today = await listed(page);
    expect(today.length).toBeGreaterThan(0);
    expect(await call("objects")).toBe(103);
    await compare
      .getByRole("group", { name: "Objects" })
      .getByRole("radio", { name: "A · Whole objects" })
      .click();
    await expect(page.getByText(/Each thing in the yard is one object/)).toBeVisible();
    await expect.poll(() => call("objects"), { timeout: 60_000 }).toBe(24);
    await call("settle");
    const whole = await listed(page);
    expect(whole).not.toEqual(today);
    expect(whole.join(" | ")).toMatch(/Trees/);
    expect(whole.join(" | ")).toMatch(/Buildings/);
    expect(whole.join(" | ")).not.toMatch(/Walls/);
    await compare
      .getByRole("group", { name: "Objects" })
      .getByRole("radio", { name: "B · Parts" })
      .click();
    await expect.poll(() => call("objects"), { timeout: 60_000 }).toBe(103);
    await call("settle");
    const parts = await listed(page);
    expect(parts.join(" | ")).toMatch(/Walls/);
    expect(parts).not.toEqual(whole);
    expect((await call("picks")).picks.objects).toBe("parts");

    // ---- Fill: the picked layer is drawn, the other is not -----------------------------------
    const hedge = await call("rectOfLocal", HEDGE.min, HEDGE.max);
    const mound = await call("rectOfLocal", MOUND.min, MOUND.max);
    if (!hedge || !mound) throw new Error("a fill is off screen");
    // Today: no inferred fill.
    await call("remember");
    await call("remember", "no fill");
    const fill = compare.getByRole("group", { name: "Fill" });
    await fill.getByRole("radio", { name: "Hedge" }).click();
    // Picking a fill while inferred layers are hidden shows them.
    await expect(page.getByRole("radio", { name: "Show inferred fill" })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    await expect.poll(async () => (await call("inferred")).fillers).toEqual(["fixture-hedge"]);
    await call("settle");
    expect((await call("inferred")).shown).toBe(1);
    expect(await call("changed", hedge)).toBeGreaterThan(0.03);
    expect(await call("changed", mound)).toBeLessThan(0.002);
    await page.screenshot({ path: test.info().outputPath("1-fill-hedge.png") });
    await fill.getByRole("radio", { name: "Mound" }).click();
    await expect.poll(async () => (await call("inferred")).fillers).toEqual(["fixture-mound"]);
    await call("settle");
    expect(await call("changed", mound)).toBeGreaterThan(0.03);
    expect(await call("changed", hedge)).toBeLessThan(0.002);
    await expect(page.getByTestId("inferred-legend")).toHaveText(
      "Inferred: generated where no camera saw. Not measured.",
    );

    // ---- Highlight: the inferred pixels change, purple; not one measured pixel does ----------
    const shown = await call("measure", mound);
    await page.screenshot({ path: test.info().outputPath("2-fill-mound-show.png") });
    await call("remember");
    await page.getByRole("radio", { name: "Highlight inferred fill" }).click();
    await call("settle");
    const lit = await call("measure", mound);
    await page.screenshot({ path: test.info().outputPath("3-fill-mound-highlight.png") });
    expect(lit.purple).toBeGreaterThan(shown.purple + 0.2);
    expect(await call("changed", mound)).toBeGreaterThan(0.03);
    expect(await call("changed", mound, true)).toBeLessThan(0.0005);

    // ---- Hide: the layer is gone, the frame the scan's own with no fill ----------------------
    await page.getByRole("radio", { name: "Hide inferred fill" }).click();
    await call("settle");
    expect((await call("inferred")).shown).toBe(0);
    expect(await call("measure", mound)).toMatchObject({ purple: 0 });
    expect(await call("changed", undefined, false, "no fill")).toBeLessThan(0.0005);
    await expect(page.getByTestId("inferred-legend")).toHaveCount(0);

    // ---- Motion: the skin the objects move by is the pick's ----------------------------------
    expect(await call("skins")).toEqual([1, 9, 10, 12]);
    const motion = compare.getByRole("group", { name: "Motion" });
    await motion.getByRole("radio", { name: "Tree only" }).click();
    await expect.poll(() => call("skins")).toEqual([1]);
    await motion.getByRole("radio", { name: "Snag and shrubs" }).click();
    await expect.poll(() => call("skins")).toEqual([9, 10, 12]);
    if (renderer !== "cesium") {
      // The overlay rebinds its tiles' skin weights to the new skin.
      await call("settle");
      await expect.poll(async () => (await call("scan"))?.motion?.skinned ?? 0).toBeGreaterThan(0);
    }
    await motion.getByRole("radio", { name: "Today" }).click();
    await expect.poll(() => call("skins")).toEqual([1, 9, 10, 12]);

    const status = (await call("picks")).status;
    for (const system of ["objects", "fill", "skins"]) expect(status[system]?.state).toBe("ready");
    expect(shaderErrors(errors)).toEqual([]);
  });
}

test("the Compare panel fits a phone", async ({ page }) => {
  test.setTimeout(300_000);
  await page.setViewportSize({ width: 400, height: 800 });
  const errors: string[] = [];
  await open(page, "cesium", errors);
  const compare = page.getByTestId("compare-methods");
  await expect(compare.getByRole("group", { name: "Motion" })).toBeVisible();
  // Every row and every choice in it on screen, none cut off.
  for (const system of ["Objects", "Fill", "Motion"]) {
    const row = compare.getByRole("group", { name: system });
    const box = await row.boundingBox();
    if (!box) throw new Error(`no ${system} row`);
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(400);
    for (const radio of await row.getByRole("radio").all()) {
      const r = await radio.boundingBox();
      expect((r?.x ?? 0) + (r?.width ?? 0)).toBeLessThanOrEqual(box.x + box.width + 0.5);
    }
  }
  // Keyboard: into the Objects row, a variant chosen with the arrows and Space.
  const objects = compare.getByRole("group", { name: "Objects" });
  await objects.getByRole("radio", { name: "Today" }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(objects.getByRole("radio", { name: "A · Whole objects" })).toBeFocused();
  await page.keyboard.press("Space");
  await expect(objects.getByRole("radio", { name: "A · Whole objects" })).toHaveAttribute(
    "aria-checked",
    "true",
  );
  await expect.poll(() => caller(page)("picks").then((p) => p.picks.objects)).toBe("whole");
  await expect(objects.getByText(/Each thing in the yard is one object/)).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
