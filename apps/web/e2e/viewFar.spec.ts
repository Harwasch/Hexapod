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
import { gzipSync } from "node:zlib";

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

/** What Spark holds centres in on the GPU (main.ts `__viewer.gpu`). */
interface GpuCentres {
  accumulator: { extSplats: boolean; textures: number };
  tiles: { uri: string; centres: "float32" | "float16" | "none" }[];
}

interface ViewerDebug {
  __viewer: {
    lookAt(target: [number, number, number], distance: number): void;
    drawn(): { uri: string; min: number[]; max: number[] }[];
    hitAhead(): number | null;
    gpu(): GpuCentres;
  };
}

/**
 * A splat tile as the pipeline packs one (splat_tiles.py `pack_spz`, then a glTF binary with
 * KHR_gaussian_splatting_compression_spz_2): SPZ v2 at 12 fractional bits, degree 0, white,
 * fully opaque, round with standard deviation `sigma` (m), unrotated.
 */
function splatGlb(centres: [number, number, number][], sigma: number): Buffer {
  const n = centres.length;
  const header = Buffer.alloc(16);
  header.writeUInt32LE(0x5053474e, 0);
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(n, 8);
  header.writeUInt8(0, 12);
  header.writeUInt8(12, 13);
  const positions = Buffer.alloc(n * 9);
  centres.flat().forEach((v, i) => positions.writeIntLE(Math.round(v * 4096), i * 3, 3));
  const scale = Math.round((Math.log(sigma) + 10) * 16);
  const spz = gzipSync(
    Buffer.concat([
      header,
      positions,
      Buffer.alloc(n, 255), // alpha
      Buffer.alloc(n * 3, 255), // colour
      Buffer.alloc(n * 3, scale),
      Buffer.alloc(n * 3, 128), // rotation: the identity's xyz
    ]),
  );
  const gltf = {
    asset: { version: "2.0" },
    extensionsUsed: ["KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz_2"],
    buffers: [{ byteLength: spz.length }],
    bufferViews: [{ buffer: 0, byteLength: spz.length }],
    meshes: [
      {
        primitives: [
          {
            attributes: {},
            extensions: {
              KHR_gaussian_splatting: {
                extensions: { KHR_gaussian_splatting_compression_spz_2: { bufferView: 0 } },
              },
            },
          },
        ],
      },
    ],
  };
  const pad = (b: Buffer, fill: number): Buffer =>
    Buffer.concat([b, Buffer.alloc((4 - (b.length % 4)) % 4, fill)]);
  const json = pad(Buffer.from(JSON.stringify(gltf)), 0x20);
  const bin = pad(spz, 0);
  const chunk = (type: number, body: Buffer): Buffer => {
    const head = Buffer.alloc(8);
    head.writeUInt32LE(body.length, 0);
    head.writeUInt32LE(type, 4);
    return Buffer.concat([head, body]);
  };
  const body = Buffer.concat([chunk(0x4e4f534a, json), chunk(0x004e4942, bin)]);
  const glb = Buffer.alloc(12);
  glb.writeUInt32LE(0x46546c67, 0);
  glb.writeUInt32LE(2, 4);
  glb.writeUInt32LE(12 + body.length, 8);
  return Buffer.concat([glb, body]);
}

/**
 * Separate bright marks across the middle band of the screen (a third of its height): runs
 * of columns whose brightest pixel there stands well clear of the background.
 */
async function marksAcross(page: Page, path: string): Promise<number[]> {
  const png = (await page.screenshot({ path })).toString("base64");
  return page.evaluate(
    async ([data, background]) => {
      const image = new Image();
      image.src = `data:image/png;base64,${data}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = image.width;
      canvas.height = image.height;
      const context = canvas.getContext("2d");
      if (!context) return [];
      context.drawImage(image, 0, 0);
      const { width, height } = canvas;
      const top = Math.round(height / 3);
      const pixels = context.getImageData(0, top, width, Math.round(height / 3)).data;
      const rows = pixels.length / 4 / width;
      const marks: number[] = [];
      let start = -1;
      for (let x = 0; x <= width; x++) {
        let bright = 0;
        for (let y = 0; x < width && y < rows; y++) {
          const i = (y * width + x) * 4;
          bright = Math.max(
            bright,
            Math.abs((pixels[i] ?? 0) - (background[0] ?? 0)) +
              Math.abs((pixels[i + 1] ?? 0) - (background[1] ?? 0)) +
              Math.abs((pixels[i + 2] ?? 0) - (background[2] ?? 0)),
          );
        }
        if (bright > 120 && start < 0) start = x;
        if (bright <= 120 && start >= 0) {
          marks.push((start + x - 1) / 2 / width);
          start = -1;
        }
      }
      return marks;
    },
    [png, BACKGROUND] as const,
  );
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
  // And Spark holds every one of them in 32-bit centres on the GPU: each tile's source
  // (ExtSplats, not half-float PackedSplats) and the accumulator each frame is generated into.
  const gpu = await page.evaluate(() => (window as unknown as ViewerDebug).__viewer.gpu());
  expect(gpu.accumulator).toEqual({ extSplats: true, textures: 3 });
  const drawnUris = new Set(drawn.map((tile) => tile.uri));
  const held = gpu.tiles.filter((tile) => drawnUris.has(tile.uri));
  expect(held.length).toBe(drawn.length);
  for (const tile of held) expect(tile.centres, tile.uri).toBe("float32");

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

test("a tile's centres are 32-bit: two splats 30 cm apart, 1 km from its origin, are drawn apart", async ({
  page,
}, testInfo) => {
  test.setTimeout(120_000);
  // One tile whose pair sits a kilometre along x from the tile's origin, 30 cm apart. As half
  // floats (Spark's packed splats, 0.5 m apart between 512 and 1024) both are 1000.0, one
  // mark; as 32-bit floats they are two. Four splats 6 m out frame the first view and stay
  // off screen. The accumulator cannot be told apart this way: Spark keeps a half-float one
  // relative to the eye, centimetres apart here -- what `gpu()` reads covers that.
  const pair: [number, number, number][] = [
    [999.85, 0, 0],
    [1000.15, 0, 0],
  ];
  const frame: [number, number, number][] = [
    [994, -6, 0],
    [1006, -6, 0],
    [994, 6, 0],
    [1006, 6, 0],
  ];
  const glb = splatGlb([...pair, ...frame], 0.015);
  const tileset = {
    asset: { version: "1.1" },
    geometricError: 20,
    root: {
      refine: "ADD",
      boundingVolume: { box: [1000, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 1] },
      geometricError: 0,
      content: { uri: "pair.glb" },
      extras: { gaussians: 6 },
    },
  };
  await page.route("https://tiles.example/**", async (route) => {
    const name = new URL(route.request().url()).pathname.split("/").pop() ?? "";
    await route.fulfill({
      body: name === "tileset.json" ? JSON.stringify(tileset) : glb,
      headers: { "access-control-allow-origin": "*" },
    });
  });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`/view.html#${SITE}`);
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn", { timeout: 60_000 });
  await page.evaluate((target) => (window as unknown as ViewerDebug).__viewer.lookAt(target, 2), [
    1000, 0, 0,
  ] as [number, number, number]);
  await page.waitForTimeout(1500);
  const gpu = await page.evaluate(() => (window as unknown as ViewerDebug).__viewer.gpu());
  const marks = await marksAcross(page, testInfo.outputPath("far-pair.png"));
  console.info(JSON.stringify({ gpu, marks }));
  // Two marks, either side of the middle of the screen (where 1000.0 is) by the same: 15 cm
  // at 2 m is 0.17 of the phone's width (its horizontal half-angle is 12 degrees).
  expect(marks.length).toBe(2);
  expect(Math.abs((marks[0] ?? 0) + (marks[1] ?? 0) - 1)).toBeLessThan(0.02);
  expect((marks[1] ?? 0) - (marks[0] ?? 0)).toBeGreaterThan(0.25);
  expect(gpu.tiles).toEqual([{ uri: "pair.glb", centres: "float32" }]);
  expect(errors).toEqual([]);
});
