/**
 * The scan viewer (view.html): a finished capture on its own, rendered with Spark.
 *
 * The render test serves the repository's committed synthetic-tree tile, so the path it
 * exercises is the real one: tileset.json -> splat.glb -> the SPZ inside it -> Spark.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

import { spzFromGlb } from "../src/view/glb";

const SITE = "d446c3d1-40c6-4659-991f-c80aedf6d706";
const TILES = resolve(import.meta.dirname, "../../../data/tiles/synthetic-tree/splat");
const TILESET = "https://tiles.example/sites/synthetic-tree/splat/tileset.json";

const capture = {
  id: "c1",
  name: "Garden tree",
  siteId: SITE,
  createdAt: "2026-09-24T15:00:00Z",
  metadata: { origin: "phone-key" },
  files: [],
  status: "complete",
};

// A phone: the viewer budgets splats by device (lib/detail.ts), and these tests count them.
test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });

test.beforeEach(async ({ page }) => {
  await page.route(
    (url) => url.pathname === "/api/v1/captures",
    (route) => route.fulfill({ json: [capture] }),
  );
  await page.route(
    (url) => url.pathname === "/api/v1/sites",
    (route) =>
      route.fulfill({
        json: [
          {
            id: SITE,
            name: "Synthetic tree",
            representations: ["gaussian-splat"],
            thumbnailUrl: null,
            createdAt: "2026-09-24T15:00:00Z",
          },
          // A seeded demo site with no capture behind it: not one of your scans.
          {
            id: "demo",
            name: "Demo",
            representations: ["gaussian-splat"],
            thumbnailUrl: null,
            createdAt: "2026-09-20T00:00:00Z",
          },
        ],
      }),
  );
  await page.route(
    (url) => url.pathname === `/api/v1/sites/${SITE}`,
    (route) =>
      route.fulfill({
        json: {
          id: SITE,
          name: "Garden tree",
          createdAt: "2026-09-24T15:00:00Z",
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

test("the gallery lists your scans and not the demo sites", async ({ page }) => {
  await page.goto("/view.html");
  await expect(page.locator("#gallery-status")).toContainText("1 scan");
  const card = page.locator(".card");
  await expect(card).toHaveCount(1);
  await expect(card).toContainText("Garden tree");
  await expect(card).toHaveAttribute("href", `#${SITE}`);
});

test("a scan opens on its own and renders", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`/view.html#${SITE}`);
  await expect(page.locator("#scan-name")).toHaveText("Garden tree");
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn", { timeout: 30_000 });
  await expect(page.locator("#viewer canvas")).toBeVisible();
  expect(errors).toEqual([]);
});

interface Tile {
  boundingVolume: { box: number[] };
  geometricError: number;
  content: { uri: string };
  extras: { gaussians: number };
  children: Tile[];
}

const tile = (uri: string, gaussians: number, error: number, children: Tile[] = []): Tile => ({
  boundingVolume: { box: [0, 0, 3, 4, 0, 0, 0, 4, 0, 0, 0, 4] },
  geometricError: error,
  content: { uri },
  extras: { gaussians },
  children,
});

/** Serves `tileset` for tileset.json and the committed tree's tile for every GLB it names. */
async function serveTileset(page: Page, tileset: unknown): Promise<string[]> {
  const fetched: string[] = [];
  await page.route("https://tiles.example/**", async (route) => {
    const name = new URL(route.request().url()).pathname.split("/").pop() ?? "";
    fetched.push(name);
    await route.fulfill({
      body:
        name === "tileset.json"
          ? JSON.stringify(tileset)
          : readFileSync(resolve(TILES, "splat.glb")),
      headers: { "access-control-allow-origin": "*" },
    });
  });
  return fetched;
}

test("a merged-parent scan swaps each parent for its children, as far as Detail allows", async ({
  page,
}) => {
  // The shape splat_tiles.py writes now: REPLACE, merged parents, leaves (error 0) holding
  // every gaussian once. Every tile's bytes are the committed tree's, so they render; the
  // counts say what a real 1.1M scan would.
  const tileset = {
    asset: { version: "1.1" },
    geometricError: 8,
    root: {
      refine: "REPLACE",
      ...tile("splat.glb", 40_000, 0.4, [
        tile("splat_1.glb", 30_000, 0.1, [
          tile("splat_1-7.glb", 300_000, 0),
          tile("splat_1-0.glb", 300_000, 0),
        ]),
        tile("splat_3.glb", 300_000, 0),
        tile("splat_02.glb", 200_000, 0),
      ]),
    },
  };
  const fetched = await serveTileset(page, tileset);
  // "Light", chosen on the phone page: Spark draws 200k and the viewer holds up to 4x.
  await page.addInitScript(() => {
    window.localStorage.setItem("twin.phoneOptions.v3", JSON.stringify({ detail: "200000" }));
  });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto(`/view.html#${SITE}`);

  await expect(page.locator("#scan-date")).toContainText("530,000 of 1,100,000 splats", {
    timeout: 30_000,
  });
  // The root alone, then all its children -- and the root is gone once they are up.
  // splat_1's leaves (600k for its 30k) would pass 800k, so it stays merged.
  expect(fetched.slice(0, 2)).toEqual(["tileset.json", "splat.glb"]);
  expect(fetched.slice(2).sort()).toEqual(["splat_02.glb", "splat_1.glb", "splat_3.glb"]);
  await expect(page.locator("#viewer")).toHaveAttribute(
    "data-tiles",
    "splat_1.glb splat_3.glb splat_02.glb",
  );
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn");
  expect(errors).toEqual([]);
});

test("a large scan streams with the camera: finer where it looks, coarse elsewhere", async ({
  page,
}) => {
  // Two regions 60 m apart, each a merged parent over two 300k leaves: 1.2M in all, more
  // than Light's 800k, so only one region can be fine at a time.
  const box = (x: number, half: number): number[] => [x, 0, 0, half, 0, 0, 0, half, 0, 0, 0, 4];
  const at = (x: number, half: number, entry: Tile): Tile => ({
    ...entry,
    boundingVolume: { box: box(x, half) },
  });
  const tileset = {
    asset: { version: "1.1" },
    geometricError: 40,
    root: {
      refine: "REPLACE",
      ...at(0, 45, {
        ...tile("splat.glb", 20_000, 4, [
          at(
            -30,
            12,
            tile("splat_0.glb", 40_000, 0.5, [
              at(-36, 6, tile("splat_0-0.glb", 300_000, 0)),
              at(-24, 6, tile("splat_0-1.glb", 300_000, 0)),
            ]),
          ),
          at(
            30,
            12,
            tile("splat_1.glb", 40_000, 0.5, [
              at(24, 6, tile("splat_1-0.glb", 300_000, 0)),
              at(36, 6, tile("splat_1-1.glb", 300_000, 0)),
            ]),
          ),
        ]),
      }),
    },
  };
  await serveTileset(page, tileset);
  await page.addInitScript(() => {
    window.localStorage.setItem("twin.phoneOptions.v3", JSON.stringify({ detail: "200000" }));
  });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`/view.html#${SITE}`);
  const viewer = page.locator("#viewer");
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn", { timeout: 30_000 });
  // From the overview, straight above and far back (the slow spin stopped, so the view holds
  // still), both regions are merged: 80k of 1.2M.
  await page.evaluate(() => (window as unknown as ViewerDebug).__viewer.lookAt([0, 0, 0], 400));
  await expect(viewer).toHaveAttribute("data-tiles", "splat_0.glb splat_1.glb", {
    timeout: 30_000,
  });
  // Walk up to the west region: it refines, the east one stays merged -- in the cut, so
  // turning round shows it at once.
  await page.evaluate(() => (window as unknown as ViewerDebug).__viewer.lookAt([-30, 0, 2], 8));
  await expect(viewer).toHaveAttribute(
    "data-tiles",
    /^(?=.*splat_0-0\.glb)(?=.*splat_0-1\.glb)(?=.*splat_1\.glb)(?!.*splat_1-)/,
    { timeout: 30_000 },
  );
  // And across to the east one: the west region goes back to its merged parent (still
  // loaded, so at once), the east one refines.
  await page.evaluate(() => (window as unknown as ViewerDebug).__viewer.lookAt([30, 0, 2], 8));
  await expect(viewer).toHaveAttribute(
    "data-tiles",
    /^(?=.*splat_1-0\.glb)(?=.*splat_1-1\.glb)(?=.*splat_0\.glb)(?!.*splat_0-)/,
    { timeout: 30_000 },
  );
  expect(errors).toEqual([]);
});

/** What view.html exposes for tests: the camera, moved to look at a scan point. */
interface ViewerDebug {
  __viewer: {
    lookAt(target: [number, number, number], distance: number): void;
    hitAhead(): number | null;
    distanceTo(point: [number, number, number]): number;
  };
}

test("a wheel over the scan zooms to its surface and stops there; Space goes through", async ({
  page,
}) => {
  // The committed tree: a crown about 8 m across, centred some 3.3 m up.
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`/view.html#${SITE}`);
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn", { timeout: 30_000 });
  const debug = (): Promise<number | null> =>
    page.evaluate(() => (window as unknown as ViewerDebug).__viewer.hitAhead());
  await page.evaluate(() => (window as unknown as ViewerDebug).__viewer.lookAt([0, 0, 3.3], 12));
  const surface = await debug();
  expect(surface).not.toBeNull();
  const size = page.viewportSize() ?? { width: 390, height: 844 };
  await page.mouse.move(size.width / 2, size.height / 2);
  for (let i = 0; i < 25; i++) await page.mouse.wheel(0, -100);
  await page.waitForTimeout(300);
  const left = (await debug()) ?? Number.NaN;
  // Up against the crown -- centimetres, the clearance -- and not through it.
  expect(left).toBeGreaterThan(0);
  expect(left).toBeLessThan(0.3);
  // Held Space: the next notches carry the camera through the surface, towards the middle.
  const centre = (): Promise<number> =>
    page.evaluate(() => (window as unknown as ViewerDebug).__viewer.distanceTo([0, 0, 3.3]));
  const outside = await centre();
  await page.keyboard.down("Space");
  for (let i = 0; i < 3; i++) await page.mouse.wheel(0, -100);
  await page.keyboard.up("Space");
  await page.waitForTimeout(300);
  expect(await centre()).toBeLessThan(outside - left);
  expect(errors).toEqual([]);
});

test("a scan packed before merged parents still loads additively", async ({ page }) => {
  const tileset = {
    asset: { version: "1.1" },
    geometricError: 8,
    root: {
      refine: "ADD",
      ...tile("splat.glb", 100_000, 0.4, [
        tile("splat_1.glb", 100_000, 0.1, [tile("splat_1-7.glb", 90_000, 0)]),
        tile("splat_3.glb", 100_000, 0.2),
        tile("splat_02.glb", 80_000, 0),
      ]),
    },
  };
  const fetched = await serveTileset(page, tileset);
  // "Light" again: 800k to download holds the whole 470k scan, every tile kept.
  await page.addInitScript(() => {
    window.localStorage.setItem("twin.phoneOptions.v3", JSON.stringify({ detail: "200000" }));
  });
  await page.goto(`/view.html#${SITE}`);
  await expect(page.locator("#viewer")).toHaveAttribute(
    "data-tiles",
    "splat.glb splat_1.glb splat_3.glb splat_02.glb splat_1-7.glb",
    { timeout: 30_000 },
  );
  expect(fetched.slice(0, 2)).toEqual(["tileset.json", "splat.glb"]);
  await expect(page.locator("#scan-date")).not.toContainText("splats loaded");
});

/** A coverage_enu.ply as tools/pipeline/quality.py writes it: points, tier, colour. */
function coveragePly(points: [number, number, number, number][]): Buffer {
  const header = Buffer.from(
    [
      "ply",
      "format binary_little_endian 1.0",
      `element vertex ${String(points.length)}`,
      "property float x",
      "property float y",
      "property float z",
      "property uchar red",
      "property uchar green",
      "property uchar blue",
      "property uchar tier",
      "end_header",
      "",
    ].join("\n"),
  );
  const body = Buffer.alloc(points.length * 16);
  points.forEach(([x, y, z, tier], i) => {
    body.writeFloatLE(x, i * 16);
    body.writeFloatLE(y, i * 16 + 4);
    body.writeFloatLE(z, i * 16 + 8);
    body.writeUInt8(tier === 2 ? 64 : 200, i * 16 + 12);
    body.writeUInt8(150, i * 16 + 13);
    body.writeUInt8(100, i * 16 + 14);
    body.writeUInt8(tier, i * 16 + 15);
  });
  return Buffer.concat([header, body]);
}

test("a scan with no coverage offers no Coverage toggle", async ({ page }) => {
  await page.goto(`/view.html#${SITE}`);
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn", { timeout: 30_000 });
  await expect(page.getByRole("button", { name: "Coverage" })).toBeHidden();
});

test("Coverage lays the quality bar's cloud and the camera path over the scan", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const coverageUrl = "https://tiles.example/runs/job/place/coverage_enu.ply";
  await page.route(
    (url) => url.pathname === `/api/v1/sites/${SITE}`,
    (route) =>
      route.fulfill({
        json: {
          id: SITE,
          name: "Garden tree",
          createdAt: "2026-09-24T15:00:00Z",
          metadata: { captureId: "c1", coverageUrl, keepVerifiedPct: 47.6 },
          assets: [
            { representation: "gaussian-splat", source: { type: "3d-tiles-url", url: TILESET } },
          ],
        },
      }),
  );
  let fetched = 0;
  await page.route(coverageUrl, async (route) => {
    fetched += 1;
    await route.fulfill({
      body: coveragePly([
        [0, 0, 1, 2],
        [0.5, 0, 1, 2],
        [1, 1, 1, 1],
        [3, 3, 0, 0],
        [4, 0, 1.5, 3],
        [0, 4, 1.5, 3],
      ]),
      headers: { "access-control-allow-origin": "*" },
    });
  });

  await page.goto(`/view.html#${SITE}`);
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn", { timeout: 30_000 });
  const toggle = page.getByRole("button", { name: "Coverage" });
  await expect(toggle).toBeVisible();
  // Not fetched until asked for: it is megabytes a phone need not spend unasked.
  expect(fetched).toBe(0);

  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-pressed", "true");
  const legend = page.locator("#coverage-legend");
  await expect(legend).toBeVisible();
  await expect(legend).toContainText("Camera path");
  await expect(legend).toContainText("2 kept · 1 context · 1 dropped");
  await expect(legend).toContainText("48% of the scene verified by held-out frames");

  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-pressed", "false");
  await expect(legend).toBeHidden();
  await toggle.click();
  await expect(legend).toBeVisible();
  expect(fetched).toBe(1);
  expect(errors).toEqual([]);
});

test("the viewer does not load CesiumJS", async ({ page }) => {
  const scripts: string[] = [];
  page.on("request", (request) => {
    if (request.resourceType() === "script") scripts.push(request.url());
  });
  await page.goto(`/view.html#${SITE}`);
  await expect(page.locator("#viewer-status")).toContainText("Drag to turn", { timeout: 30_000 });
  expect(scripts.filter((url) => /cesium/i.test(url))).toEqual([]);
});

// --- live mode ------------------------------------------------------------------------

function toArrayBuffer(buffer: Buffer): ArrayBuffer {
  const copy = new ArrayBuffer(buffer.byteLength);
  new Uint8Array(copy).set(buffer);
  return copy;
}

/** A `live-cameras` payload packed as tools/pipeline/live.py packs it: an orbit of `count`. */
function liveCameras(count: number): Record<string, unknown> {
  const scale = 10;
  const q = (value: number): number => Math.round((value / scale) * 32767);
  const cameras = Buffer.alloc(count * 12);
  for (let i = 0; i < count; i += 1) {
    const angle = (2 * Math.PI * i) / count;
    const at = i * 12;
    cameras.writeInt16LE(q(8 * Math.cos(angle)), at);
    cameras.writeInt16LE(q(8 * Math.sin(angle)), at + 2);
    cameras.writeInt16LE(q(3), at + 4);
    // Looking in at the tree, held upright: forward toward the axis, up +z.
    cameras.writeInt8(Math.round(-Math.cos(angle) * 127), at + 6);
    cameras.writeInt8(Math.round(-Math.sin(angle) * 127), at + 7);
    cameras.writeInt8(127, at + 11);
  }
  const points = Buffer.alloc(3 * 9);
  const sample: [number, number, number][] = [
    [0, 0, 1],
    [0.5, 0.5, 3],
    [-0.5, 0.2, 5],
  ];
  sample.forEach(([x, y, z], i) => {
    points.writeInt16LE(q(x), i * 9);
    points.writeInt16LE(q(y), i * 9 + 2);
    points.writeInt16LE(q(z), i * 9 + 4);
    points.writeUInt8(90, i * 9 + 6);
    points.writeUInt8(160, i * 9 + 7);
    points.writeUInt8(60, i * 9 + 8);
  });
  return {
    stageId: "pose",
    seq: count,
    final: count >= 12,
    registered: count,
    frames: 12,
    origin: [0, 0, 0],
    scale,
    up: [0, 0, 1],
    aspect: 1.333,
    cameraCount: count,
    cameras: cameras.toString("base64"),
    pointCount: 3,
    pointsTotal: 3,
    points: points.toString("base64"),
  };
}

const liveBase = {
  jobId: "j1",
  captureId: "c1",
  captureName: "Garden tree",
  recipe: "photo-reconstruct",
  error: null,
  siteId: null,
  stepsDone: 1,
  stepsStarted: 2,
  splat: null,
};

const posing = {
  stageId: "pose",
  impl: "colmap",
  status: "in-progress",
  ordinal: 1,
  startedAt: "2026-09-27T10:00:00Z",
  progress: null,
};

function training(done: number): Record<string, unknown> {
  return {
    stageId: "train",
    impl: "gsplat",
    status: "in-progress",
    ordinal: 3,
    startedAt: "2026-09-27T10:05:00Z",
    progress: { done, total: 30000, elapsedS: 600, remainingS: 1800 },
  };
}

const BUCKET = "https://bucket.example/twin-assets/runs/r/train/checkpoint/live";

function liveSplat(step: number, bytes: number): Record<string, unknown> {
  const name = `splat_${String(step).padStart(6, "0")}_1.spz`;
  return {
    stageId: "train",
    step,
    total: 30000,
    count: 1000,
    of: 4000,
    bytes,
    up: [0, 0, 1],
    name,
    url: `${BUCKET}/${name}?X-Amz-Signature=abc`,
  };
}

test("live mode draws cameras as they are solved, then swaps in each intermediate splat", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  // The committed test splat's SPZ, standing in for a training snapshot.
  const spz = Buffer.from(spzFromGlb(toArrayBuffer(readFileSync(resolve(TILES, "splat.glb")))));
  const states: Record<string, unknown> = {
    "pose-early": { ...liveBase, status: "in-progress", stage: posing, cameras: liveCameras(5) },
    "pose-late": { ...liveBase, status: "in-progress", stage: posing, cameras: liveCameras(12) },
    "train-1": {
      ...liveBase,
      status: "in-progress",
      stage: training(3600),
      cameras: liveCameras(12),
      splat: liveSplat(3000, spz.length),
    },
    "train-2": {
      ...liveBase,
      status: "in-progress",
      stage: training(8000),
      cameras: liveCameras(12),
      splat: liveSplat(7500, spz.length),
    },
    done: {
      ...liveBase,
      status: "complete",
      siteId: SITE,
      stage: { ...training(30000), status: "complete", progress: null },
      cameras: liveCameras(12),
      splat: liveSplat(7500, spz.length),
    },
  };
  let phase = "pose-early";
  await page.route(
    (url) => url.pathname === "/api/v1/captures/c1/live",
    (route) => route.fulfill({ json: states[phase] }),
  );
  const fetched: string[] = [];
  await page.route(`${BUCKET}/**`, async (route) => {
    fetched.push(new URL(route.request().url()).pathname.split("/").pop() ?? "");
    await route.fulfill({ body: spz, headers: { "access-control-allow-origin": "*" } });
  });

  await page.goto("/view.html#live/c1");
  const live = page.locator("#live");
  await expect(live).toBeVisible();
  await expect(page.locator("#live-name")).toHaveText("Garden tree");
  await expect(page.locator("#live-status")).toHaveText("Solving camera positions…");
  await expect(page.locator("#live-detail")).toHaveText("5 of 12 frames placed");
  await expect(live).toHaveAttribute("data-cameras", "5");
  await expect(live).toHaveAttribute("data-points", "3");

  phase = "pose-late";
  await expect(live).toHaveAttribute("data-cameras", "12");

  phase = "train-1";
  await expect(page.locator("#live-status")).toHaveText("Training the splat…");
  await expect(page.locator("#live-detail")).toContainText("Step 3,600 of 30,000 · 12%");
  await expect(page.locator("#live-detail")).toContainText("about 30 min left");
  await expect(page.locator("#live-track")).toHaveAttribute("aria-valuenow", "12");
  await expect(live).toHaveAttribute("data-splat-step", "3000", { timeout: 30_000 });

  phase = "train-2";
  await expect(live).toHaveAttribute("data-splat-step", "7500", { timeout: 30_000 });
  // Each snapshot is fetched once, however many polls name it.
  expect(fetched).toEqual(["splat_003000_1.spz", "splat_007500_1.spz"]);
  // The cameras stay while the splats change under them.
  await expect(live).toHaveAttribute("data-cameras", "12");

  phase = "done";
  const final = page.getByRole("link", { name: "Finished — open the final scan" });
  await expect(final).toBeVisible();
  await expect(final).toHaveAttribute("href", `#${SITE}`);
  await final.click();
  await expect(page.locator("#scan-name")).toHaveText("Garden tree");
  await expect(live).toBeHidden();
  expect(errors).toEqual([]);
});

test("live mode on a capture with no run says so", async ({ page }) => {
  await page.route(
    (url) => url.pathname === "/api/v1/captures/nothing/live",
    (route) => route.fulfill({ status: 404, json: { title: "Not found" } }),
  );
  await page.goto("/view.html#live/nothing");
  await expect(page.locator("#live-status")).toHaveText("Nothing to watch yet");
});

test("live mode does not scroll sideways on a narrow phone", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 640 });
  await page.route(
    (url) => url.pathname === "/api/v1/captures/c1/live",
    (route) =>
      route.fulfill({
        json: {
          ...liveBase,
          captureName: "A capture with a rather long name that has to wrap somewhere",
          status: "in-progress",
          stage: posing,
          cameras: liveCameras(7),
        },
      }),
  );
  await page.goto("/view.html#live/c1");
  await expect(page.locator("#live")).toHaveAttribute("data-cameras", "7");
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
