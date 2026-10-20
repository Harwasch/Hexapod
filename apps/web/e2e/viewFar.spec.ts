/**
 * The scan viewer (view.html) on a site wider than SPZ can hold in one frame: the committed
 * 10.4 km synthetic site (data/tiles/synthetic-far, tools/captures/synthetic_far.py), whose far
 * tiles are written in frames of their own (translation-only tile transforms, nested) and
 * whose widest tiles keep 11 fractional bits instead of 12.
 *
 * A reader that ignored a tile's frame would draw its splats at their local coordinates, the
 * offset short; one that ignored the header's bits would draw them at half (or twice) their
 * distance from the tile's origin. Both show here: the tiles' boxes, and where the plot is.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const SITE = "5a7c3e1f-0b2d-4e6a-9c8f-0000000000fa";
const FAR = resolve(import.meta.dirname, "../../../data/tiles/synthetic-far");
const TILES = resolve(FAR, "splat");
const TILESET = "https://tiles.example/sites/synthetic-far/splat/tileset.json";
/** view.html's clear colour (main.ts), which a cell with no scan in it shows. */
const BACKGROUND = [0x0b, 0x0c, 0x0a];

interface Expected {
  plots: Record<string, [number, number, number]>;
  plotLeaves: Record<string, string[]>;
  tiles: Record<string, { offset: [number, number, number]; fractionalBits: number }>;
}
const expected = JSON.parse(readFileSync(resolve(FAR, "source/expected.json"), "utf8")) as Expected;

/** The POSITION accessor's min and max of a tile's GLB: its centres in its own frame. */
function accessorBox(uri: string): { min: number[]; max: number[] } {
  const glb = readFileSync(resolve(TILES, uri));
  const length = glb.readUInt32LE(12);
  const json = JSON.parse(glb.subarray(20, 20 + length).toString("utf8")) as {
    accessors: { min?: number[]; max?: number[] }[];
    meshes: { primitives: { attributes: { POSITION: number } }[] }[];
  };
  const index = json.meshes[0]?.primitives[0]?.attributes.POSITION ?? 0;
  const accessor = json.accessors[index];
  return { min: accessor?.min ?? [], max: accessor?.max ?? [] };
}

interface ViewerDebug {
  __viewer: {
    lookAt(target: [number, number, number], distance: number): void;
    drawn(): { uri: string; min: number[]; max: number[] }[];
    hitAhead(): number | null;
  };
}

test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });

test.beforeEach(async ({ page }) => {
  await page.route(
    (url) => url.pathname === `/api/v1/sites/${SITE}`,
    (route) =>
      route.fulfill({
        json: {
          id: SITE,
          name: "Far site",
          createdAt: "2026-10-19T00:00:00Z",
          assets: [
            { representation: "gaussian-splat", source: { type: "3d-tiles-url", url: TILESET } },
          ],
        },
      }),
  );
  await page.route("https://tiles.example/**", async (route) => {
    const name = new URL(route.request().url()).pathname.split("/").pop() ?? "";
    await route.fulfill({
      body: readFileSync(resolve(TILES, name)),
      headers: { "access-control-allow-origin": "*" },
    });
  });
});

/** The share of the middle third of the screen (a 64 x 40 grid of cells) that holds scan. */
async function centreCoverage(page: Page, path: string): Promise<number> {
  const png = (await page.screenshot({ path })).toString("base64");
  return page.evaluate(
    async ([data, background]) => {
      const image = new Image();
      image.src = `data:image/png;base64,${data}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = 64;
      canvas.height = 40;
      const context = canvas.getContext("2d");
      if (!context) return 0;
      context.drawImage(image, 0, 0, 64, 40);
      const pixels = context.getImageData(0, 0, 64, 40).data;
      let cells = 0;
      let scan = 0;
      for (let y = 13; y < 27; y++) {
        for (let x = 21; x < 43; x++) {
          const i = (y * 64 + x) * 4;
          const d =
            Math.abs((pixels[i] ?? 0) - (background[0] ?? 0)) +
            Math.abs((pixels[i + 1] ?? 0) - (background[1] ?? 0)) +
            Math.abs((pixels[i + 2] ?? 0) - (background[2] ?? 0));
          cells += 1;
          if (d > 24) scan += 1;
        }
      }
      return scan / cells;
    },
    [png, BACKGROUND] as const,
  );
}

test("a far tile is drawn where it is: its frame composed and its header's bits read", async ({
  page,
}, testInfo) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`/view.html#${SITE}`);
  await expect(page.locator("#scan-name")).toHaveText("Far site");
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn", { timeout: 60_000 });

  // The root (11 bits: its content spans 7 km) is drawn the size its GLB says, in the scan's
  // frame. Read at 12 bits it would be half that.
  const debug = (): Promise<{ uri: string; min: number[]; max: number[] }[]> =>
    page.evaluate(() => (window as unknown as ViewerDebug).__viewer.drawn());
  const root = (await debug()).find((tile) => tile.uri === "splat.glb");
  expect(root).toBeDefined();
  const rootBox = accessorBox("splat.glb");
  expect(expected.tiles["splat.glb"]?.fractionalBits).toBe(11);
  for (let k = 0; k < 3; k++) {
    expect(Math.abs((root?.min[k] ?? Number.NaN) - (rootBox.min[k] ?? 0))).toBeLessThan(1);
    expect(Math.abs((root?.max[k] ?? Number.NaN) - (rootBox.max[k] ?? 0))).toBeLessThan(1);
  }

  // Up close to the north-east plot, 2.6 km east and 1.8 km north: its leaves stream in and
  // the merged root gives way.
  const plot = expected.plots["north-east"] ?? [0, 0, 0];
  const leaves = expected.plotLeaves["north-east"] ?? [];
  await page.evaluate(
    (target) => (window as unknown as ViewerDebug).__viewer.lookAt(target, 20),
    plot,
  );
  const viewer = page.locator("#viewer");
  for (const leaf of leaves) {
    await expect(viewer).toHaveAttribute(
      "data-tiles",
      new RegExp(`(^| )${leaf.replace(/\./g, "\\.")}( |$)`),
      {
        timeout: 60_000,
      },
    );
  }
  await expect(viewer).not.toHaveAttribute("data-tiles", /(^| )splat\.glb( |$)/);

  // Every tile drawn so far sits at its GLB's box moved by its frame's offset.
  const drawn = await debug();
  expect(drawn.length).toBeGreaterThan(3);
  for (const tile of drawn) {
    const own = accessorBox(tile.uri);
    const offset = expected.tiles[tile.uri]?.offset ?? [Number.NaN, 0, 0];
    for (let k = 0; k < 3; k++) {
      expect(
        Math.abs((tile.min[k] ?? 0) - ((own.min[k] ?? 0) + (offset[k] ?? 0))),
        tile.uri,
      ).toBeLessThan(1);
      expect(
        Math.abs((tile.max[k] ?? 0) - ((own.max[k] ?? 0) + (offset[k] ?? 0))),
        tile.uri,
      ).toBeLessThan(1);
    }
  }
  expect(drawn.some((tile) => tile.uri === leaves[0])).toBe(true);

  // The plot fills the middle of the screen. Give Spark a moment to sort what arrived.
  await page.waitForTimeout(1500);
  const here = await centreCoverage(page, testInfo.outputPath("far-plot.png"));
  // And where it would be drawn had each leaf's offset been ignored, there is nothing.
  const misplaced: number[] = [];
  for (const [index, leaf] of leaves.entries()) {
    const offset = expected.tiles[leaf]?.offset ?? [0, 0, 0];
    const wrong: [number, number, number] = [
      plot[0] - offset[0],
      plot[1] - offset[1],
      plot[2] - offset[2],
    ];
    await page.evaluate(
      (target) => (window as unknown as ViewerDebug).__viewer.lookAt(target, 20),
      wrong,
    );
    await page.waitForTimeout(1500);
    misplaced.push(
      await centreCoverage(page, testInfo.outputPath(`far-plot-unoffset-${String(index)}.png`)),
    );
  }
  console.info(JSON.stringify({ plot: here, misplaced }));
  expect(here).toBeGreaterThan(0.05);
  for (const share of misplaced) expect(share).toBeLessThan(0.01);
  expect(errors).toEqual([]);
});

test("an ADD tileset's far tile is a surface where it is drawn", async ({ page }) => {
  test.setTimeout(180_000);
  // Without a packaged grid, view.html builds collision from each tile's own splats. It reads
  // them as it loads each tile of an ADD tileset (main.ts `tileMesh`), the path that must have
  // them in the scan's frame; a REPLACE tileset streams its tiles without them. So the far
  // tileset is served as ADD, without its hexapod.collision declaration: every tile loads,
  // and the leaves' splats are the surfaces near the plot.
  const tileset = JSON.parse(readFileSync(resolve(TILES, "tileset.json"), "utf8")) as {
    root: { refine?: string; extras?: { collision?: unknown } };
  };
  tileset.root.refine = "ADD";
  delete tileset.root.extras?.collision;
  await page.route(TILESET, (route) =>
    route.fulfill({ json: tileset, headers: { "access-control-allow-origin": "*" } }),
  );
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`/view.html#${SITE}`);
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn", { timeout: 60_000 });
  const plot = expected.plots["north-east"] ?? [0, 0, 0];
  const leaves = expected.plotLeaves["north-east"] ?? [];
  const viewer = page.locator("#viewer");
  for (const leaf of leaves) {
    await expect(viewer).toHaveAttribute(
      "data-tiles",
      new RegExp(`(^| )${leaf.replace(/\./g, "\\.")}( |$)`),
      { timeout: 60_000 },
    );
  }
  const lookAt = (target: [number, number, number]): Promise<void> =>
    page.evaluate((at) => (window as unknown as ViewerDebug).__viewer.lookAt(at, 20), target);
  const hitAhead = (): Promise<number | null> =>
    page.evaluate(() => (window as unknown as ViewerDebug).__viewer.hitAhead());
  // Looking at the plot from 20 m: the first surface is the plot's, a little nearer than that
  // (a cell's clearance, and the plot's splats have size).
  await lookAt(plot);
  const hit = await hitAhead();
  // Where each leaf's splats would be had its offset been ignored: nothing to stop at.
  const misplaced: (number | null)[] = [];
  for (const leaf of leaves) {
    const offset = expected.tiles[leaf]?.offset ?? [0, 0, 0];
    await lookAt([plot[0] - offset[0], plot[1] - offset[1], plot[2] - offset[2]]);
    misplaced.push(await hitAhead());
  }
  console.info(JSON.stringify({ hit, misplaced }));
  expect(hit).not.toBeNull();
  expect(hit ?? Number.NaN).toBeGreaterThan(5);
  expect(hit ?? Number.NaN).toBeLessThan(20);
  for (const miss of misplaced) expect(miss).toBeNull();
  expect(errors).toEqual([]);
});
