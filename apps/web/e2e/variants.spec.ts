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
 * - picking a skins variant replaces the skin the scan's objects move by;
 * - a fill that supersedes measured splats (lib/supersedes.ts) hides them while it is shown,
 *   Show or Highlight, and Hide or another method gives back the untouched scan.
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
/** The yard's big tree (instance 1 of its instances.json): what the swap test supersedes. */
const TREE = { id: 1, min: [17.3, 14.6, 0.2], max: [24.0, 23.3, 9.9] };
const SUPERSEDES = "synthetic-yard/variants/fill/hedge/supersedes.json";

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

interface Entry {
  name: string;
  label: string;
  about: string;
  [file: string]: unknown;
}

/**
 * The yard's variants under the names the real bake-offs publish on the spool (labels and
 * descriptions as `skin_variants.py`, `publish_variants.py` and the segmentation bake-offs
 * write them): four methods a system, long names. Their files are the yard fixture's, in turn.
 */
function spoolLike(variants: Record<string, Entry[]>): Record<string, Entry[]> {
  const like = (system: string, names: [string, string, string][]): Entry[] =>
    names.map(([name, label, about], i) => {
      const files = variants[system] ?? [];
      const file = files[i % files.length] ?? files[0];
      return { ...file, name, label, about };
    });
  return {
    objects: like("objects", [
      [
        "ground-first",
        "A · Ground first",
        "Takes the ground out first, then splits what stands on it.",
      ],
      [
        "concept-first",
        "B · Concept first",
        "Names what is there first, then finds each thing's splats.",
      ],
      [
        "feature-fields",
        "C · Feature fields",
        "Groups splats by learned features, coarse to fine.",
      ],
    ]),
    fill: like("fill", [
      [
        "vace-1-3b",
        "Wan2.1-VACE 1.3B",
        "A video model walks a camera from a real photo towards what the scan missed.",
      ],
      [
        "wan22-5b",
        "Wan 2.2 TI2V-5B",
        "The same camera walk with Wan 2.2: the known pixels held fixed.",
      ],
      [
        "cosmos-p2-2b",
        "Cosmos-Predict2 2B",
        "The same camera walk with NVIDIA's Cosmos world model.",
      ],
      [
        "lama-baseline",
        "Per-view LaMa (baseline)",
        "Each view painted on its own by an image inpainter.",
      ],
    ]),
    skins: like("skins", [
      [
        "freeform",
        "FreeForm · size rule",
        "Today's method: skinning eigenmodes, 8 to 16 handles by size alone.",
      ],
      [
        "freeform-stiff",
        "FreeForm · stiffness rule",
        "The same modes, with handles by stiffness class.",
      ],
      [
        "pinned-stiff",
        "Pinned FreeForm · stiffness rule",
        "Modes solved with the base held still; handles by stiffness class.",
      ],
      [
        "tetfem-stiff",
        "Volume FEM · stiffness rule",
        "Linear finite elements on tetrahedra filling the object; base held.",
      ],
    ]),
  };
}

/**
 * A `supersedes.json` (lib/supersedes.ts) flagging every splat of instance `id` and what is
 * below it, in instances.json's own tiles: a stand-in for what a rebuilt surface replaces.
 */
function supersedesOf(
  doc: { instances: { id: number; parent?: number | null }[]; tiles: Record<string, number[]> },
  id: number,
) {
  const wanted = new Set([id]);
  for (let grew = true; grew;) {
    grew = false;
    for (const i of doc.instances)
      if (i.parent != null && wanted.has(i.parent) && !wanted.has(i.id)) {
        wanted.add(i.id);
        grew = true;
      }
  }
  let superseded = 0;
  const tiles: Record<string, number[]> = {};
  for (const [checksum, runs] of Object.entries(doc.tiles)) {
    const out: number[] = [];
    for (let k = 0; k + 1 < runs.length; k += 2) {
      const flag = wanted.has(runs[k] ?? 0) ? 1 : 0;
      const count = runs[k + 1] ?? 0;
      if (flag) superseded += count;
      const last = out.length - 2;
      if (last >= 0 && out[last] === flag) out[last + 1] = (out[last + 1] ?? 0) + count;
      else out.push(flag, count);
    }
    tiles[checksum] = out;
  }
  return { superseded, tiles };
}

async function open(
  page: Page,
  renderer: Renderer,
  errors: string[],
  {
    names = "fixture",
    supersede = false,
  }: { names?: "fixture" | "spool"; supersede?: boolean } = {},
): Promise<void> {
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  const read = (relative: string): Record<string, unknown> =>
    JSON.parse(readFileSync(resolve(TILES, relative), "utf-8")) as Record<string, unknown>;
  const fixture = read("synthetic-yard/variants/variants.json").variants as Record<string, Entry[]>;
  const variants = names === "spool" ? spoolLike(fixture) : fixture;
  if (supersede) {
    // The hedge method replaces the big tree's measured splats, as a rebuilt surface would.
    variants.fill = (variants.fill ?? []).map((entry) =>
      entry.name === "hedge"
        ? { ...entry, supersedes: "../variants/fill/hedge/supersedes.json" }
        : entry,
    );
  }
  const instancesDoc = read("synthetic-yard/instances/instances.json") as {
    instances: { id: number; parent?: number | null }[];
    tiles: Record<string, number[]>;
  };
  const instances = instancesDoc.instances as unknown[];
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
    if (supersede && relative === SUPERSEDES) {
      return route.fulfill({ status: 200, json: supersedesOf(instancesDoc, TREE.id) });
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

for (const renderer of ["playcanvas", "spark", "cesium"] as const) {
  test(`a fill that supersedes measured splats swaps them under ${renderer}; Hide restores them`, async ({
    page,
  }) => {
    test.setTimeout(900_000);
    const errors: string[] = [];
    await open(page, renderer, errors, { supersede: true });
    const call = caller(page);
    await call("view", 0, -32, 52);
    if (renderer !== "cesium") expect((await call("scan"))?.active).toBe(true);
    const tree = await call("rectOfLocal", TREE.min, TREE.max);
    const mound = await call("rectOfLocal", MOUND.min, MOUND.max);
    if (!tree || !mound) throw new Error("the tree or the mound is off screen");
    await call("remember", "untouched");
    await page.screenshot({ path: test.info().outputPath("1-untouched.png") });
    const fill = page.getByTestId("compare-methods").getByRole("group", { name: "Fill" });

    // Shown: the tree's measured splats are hidden while the method's layer is drawn.
    await fill.getByRole("radio", { name: "Hedge" }).click();
    await expect.poll(async () => (await call("inferred")).fillers).toEqual(["fixture-hedge"]);
    await call("settle");
    await page.screenshot({ path: test.info().outputPath("2-hedge-show.png") });
    const swapped = await call("changed", tree, false, "untouched");
    expect(swapped).toBeGreaterThan(0.05);
    expect(await call("changed", mound, false, "untouched")).toBeLessThan(0.002);

    // Highlight keeps the swap.
    await page.getByRole("radio", { name: "Highlight inferred fill" }).click();
    await call("settle");
    expect(await call("changed", tree, false, "untouched")).toBeGreaterThan(0.05);

    // Hide: the untouched scan, every pixel of it.
    await page.getByRole("radio", { name: "Hide inferred fill" }).click();
    await call("settle");
    await page.screenshot({ path: test.info().outputPath("3-hide.png") });
    expect(await call("changed", undefined, false, "untouched")).toBeLessThan(0.0005);

    // Shown again, swapped again; another method (no supersedes) gives the tree back.
    await page.getByRole("radio", { name: "Show inferred fill" }).click();
    await call("settle");
    expect(await call("changed", tree, false, "untouched")).toBeGreaterThan(0.05);
    await fill.getByRole("radio", { name: "Mound" }).click();
    await expect.poll(async () => (await call("inferred")).fillers).toEqual(["fixture-mound"]);
    await call("settle");
    await page.screenshot({ path: test.info().outputPath("4-mound.png") });
    expect(await call("changed", tree, false, "untouched")).toBeLessThan(0.002);
    expect((await call("picks")).status.fill?.state).toBe("ready");
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

test("long method names list one a row, in the app's own control, on a desktop and a phone", async ({
  page,
}) => {
  test.setTimeout(300_000);
  // Tall enough that the whole panel is on screen for its picture.
  await page.setViewportSize({ width: 1280, height: 1500 });
  const errors: string[] = [];
  await open(page, "cesium", errors, { names: "spool" });
  const compare = page.getByTestId("compare-methods");
  const motion = compare.getByRole("group", { name: "Motion" });
  await expect(motion).toBeVisible();
  // No native select anywhere: one radio group a system, a row a method when names are long.
  await expect(page.locator("#panel select")).toHaveCount(0);
  await expect(compare.getByRole("combobox")).toHaveCount(0);
  for (const system of ["Objects", "Fill", "Motion"]) {
    await expect(compare.getByRole("group", { name: system })).toHaveAttribute(
      "data-layout",
      "list",
    );
  }
  const methods = motion.getByRole("radiogroup", { name: "Motion method" });
  await expect(methods).toHaveAttribute("aria-orientation", "vertical");
  await expect(methods.getByRole("radio")).toHaveText([
    "Today",
    "FreeForm · size rule",
    "FreeForm · stiffness rule",
    "Pinned FreeForm · stiffness rule",
    "Volume FEM · stiffness rule",
  ]);
  // Each a row of its own, one under the other, as wide as the list.
  const boxes = await Promise.all(
    (await methods.getByRole("radio").all()).map((r) => r.boundingBox()),
  );
  for (let i = 1; i < boxes.length; i += 1) {
    expect(boxes[i]?.y ?? 0).toBeGreaterThanOrEqual((boxes[i - 1]?.y ?? 0) + 1);
    expect(Math.abs((boxes[i]?.width ?? 0) - (boxes[0]?.width ?? 0))).toBeLessThan(1);
  }
  // What to look for, under what the method does.
  await expect(motion.getByText(/Turn wind up in Settings › Simulated wind/)).toBeVisible();
  await expect(
    compare.getByRole("group", { name: "Fill" }).getByText(/Set Inferred to Highlight/),
  ).toBeVisible();
  await methods.getByRole("radio", { name: "Pinned FreeForm · stiffness rule" }).click();
  await expect.poll(() => caller(page)("picks").then((p) => p.picks.skins)).toBe("pinned-stiff");
  await expect(motion).toHaveAttribute("data-picked", "pinned-stiff");
  await compare
    .getByRole("group", { name: "Objects" })
    .getByRole("radio", { name: "A · Ground first" })
    .click();
  // The panel as it looks, glass and all, down to the end of the methods.
  const panel = await page.locator("#panel").boundingBox();
  const rows = await compare.boundingBox();
  if (panel && rows) {
    await page.screenshot({
      path: test.info().outputPath("restyle.png"),
      clip: {
        x: panel.x,
        y: panel.y,
        width: panel.width,
        height: rows.y + rows.height + 12 - panel.y,
      },
    });
  }

  // On a phone: every row and every method in it on screen, none cut off; and the keyboard.
  await page.setViewportSize({ width: 400, height: 800 });
  for (const system of ["Objects", "Fill", "Motion"]) {
    const row = compare.getByRole("group", { name: system });
    await row.scrollIntoViewIfNeeded();
    const box = await row.boundingBox();
    if (!box) throw new Error(`no ${system} row`);
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(400);
    for (const radio of await row.getByRole("radio").all()) {
      const r = await radio.boundingBox();
      expect((r?.x ?? 0) + (r?.width ?? 0)).toBeLessThanOrEqual(box.x + box.width + 0.5);
    }
  }
  await methods.getByRole("radio", { name: "Pinned FreeForm · stiffness rule" }).focus();
  await page.keyboard.press("ArrowDown");
  const fem = methods.getByRole("radio", { name: "Volume FEM · stiffness rule" });
  await expect(fem).toBeFocused();
  await page.keyboard.press("Space");
  await expect(fem).toHaveAttribute("aria-checked", "true");
  await expect.poll(() => caller(page)("picks").then((p) => p.picks.skins)).toBe("tetfem-stiff");
  await motion.scrollIntoViewIfNeeded();
  await motion.screenshot({ path: test.info().outputPath("restyle-phone-motion.png") });
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
