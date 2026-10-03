/**
 * The objects panel on the published scans, for a manual check on real data: skipped unless
 * `PUBLISHED_SCANS` is set. Each scan's package is fetched from the public bucket by `curl`
 * (the bucket refuses some clients; a curl User-Agent is answered) and cached on disk
 * (`PUBLISHED_CACHE`, default the OS temp dir), then served to the page under `/published/`,
 * so the page loads exactly the published tileset, tiles and instances.json.
 *
 *   PUBLISHED_SCANS=1 PUBLISHED_SHOTS=/some/dir npx playwright test e2e/instancesPublished.spec.ts
 *
 * `PUBLISHED_INSTANCES` names a directory of `<scan>.instances.json` served instead of the
 * published ones (`tools/captures/rebind_instances.py` output, before it is uploaded).
 *
 * `PUBLISHED_RENDERERS` picks the renderers (default `playcanvas`, the app's; also `cesium`,
 * `spark`). Per scan and renderer: the categories the panel lists, a category hidden from the
 * panel's eye (what changed on screen, and the coverage), and highlighted by a click on its row.
 * Screenshots (scan and panel) go to `PUBLISHED_SHOTS`, else the test's output directory.
 */

import { execFile } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { promisify } from "node:util";

import { expect, test, type Page } from "@playwright/test";

const run = promisify(execFile);

const BUCKET = "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/runs";
const CACHE = process.env.PUBLISHED_CACHE ?? join(tmpdir(), "hexapod-published");
const SHOTS = process.env.PUBLISHED_SHOTS;
const RENDERERS = (process.env.PUBLISHED_RENDERERS ?? "playcanvas").split(",") as (
  "cesium" | "playcanvas" | "spark"
)[];

interface Scan {
  name: string;
  run: string;
  /** The category hidden and highlighted, and the camera that shows it. */
  category: string;
  categoryId: string;
  view: [heading: number, pitch: number, range: number];
  /** An object clicked in the scene, and the views (heading, pitch, range) it is clicked from. */
  select: {
    id: number;
    views: [heading: number, pitch: number, range: number][];
    /**
     * Up to this range the object itself is what the click must select; beyond it what is in
     * front of it may cover it (the camp's canopy over its roofs from 300 m), and the click
     * must select some object of the scan.
     */
    exactWithinM: number;
  };
}

const SCANS: Scan[] = [
  {
    name: "pumpkin",
    run: "430c1932-5b6a-47b1-bb71-bb7fa2fec86b",
    category: "Fruit, vegetables & crops",
    categoryId: "produce",
    view: [30, -40, 7],
    // One of the big pumpkins, from close by to far out.
    select: {
      id: 51,
      views: [
        [30, -40, 6],
        [30, -35, 14],
        [30, -30, 32],
        [30, -30, 120],
      ],
      exactWithinM: 120,
    },
  },
  {
    name: "camp",
    run: "50c25673-0940-4574-9b96-0b21362f83ca",
    category: "Trees",
    categoryId: "trees",
    view: [200, -25, 90],
    // A roof among the cabins; from 300 m (coarse tiles, most splats unlabelled) the canopy
    // in front of it is what the click meets.
    select: {
      id: 4722,
      views: [
        [200, -45, 8],
        [200, -40, 14],
        [200, -35, 32],
        [200, -35, 300],
      ],
      exactWithinM: 32,
    },
  },
];

interface Measure {
  coverage: number;
  amber: number;
  luma: number;
  warmth: number;
}
interface Harness {
  view(headingDeg: number, pitchDeg: number, rangeM: number): Promise<Measure>;
  measure(): Measure;
  wait(frames: number): Promise<void>;
  remember(): void;
  changed(): number;
  categories(): { id: string; name: string; objects: number; splats: number }[];
  drawn(category: string): {
    tiles: number;
    tilesWithIds: number;
    splats: number;
    unlabelled: number;
    inCategory: number;
  };
  scan(): {
    kind: string;
    native: boolean;
    tiles: number;
    instances: { tiles: number; matched: number } | null;
  } | null;
}

/** A published object, from the cache or fetched into it; null when the bucket has none. */
async function published(path: string): Promise<Buffer | null> {
  const override = process.env.PUBLISHED_INSTANCES;
  const scan = SCANS.find((s) => path === `${s.run}/package/splat/instances.json`);
  if (override && scan && existsSync(join(override, `${scan.name}.instances.json`))) {
    return readFileSync(join(override, `${scan.name}.instances.json`));
  }
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

function harnessHtml(scan: Scan, renderer: string): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Published objects</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
      #panel { position: fixed; top: 12px; right: 12px; z-index: 10; width: 21rem; padding: 0.6rem; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <div id="panel" class="glass glass--strong"></div>
    <script type="module">
      const harness = await import("/src/dev/instancesHarness.ts");
      window.__instances = await harness.startInstancesHarness({
        container: document.getElementById("viewer"),
        url: "/published/${scan.run}/package/splat/tileset.json",
        renderer: "${renderer}",
        panel: document.getElementById("panel"),
      });
    </script>
  </body>
</html>`;
}

async function open(
  page: Page,
  scan: Scan,
  renderer: string,
  path = "/published-harness.html",
  global = "__instances",
): Promise<void> {
  await page.route("**/published/**", async (route) => {
    const path = new URL(route.request().url()).pathname.replace(/^.*\/published\//, "");
    if (path.includes("..")) return route.abort();
    const body = await published(path);
    if (!body) return route.fulfill({ status: 404, body: "" });
    const contentType = path.endsWith(".json")
      ? "application/json"
      : path.endsWith(".glb")
        ? "model/gltf-binary"
        : path.endsWith(".webp")
          ? "image/webp"
          : "application/octet-stream";
    return route.fulfill({ status: 200, contentType, body });
  });
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.route("**/published-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml(scan, renderer) }),
  );
  await page.goto(path);
  await page.waitForFunction((name) => name in window, global, { timeout: 600_000 });
}

function caller(page: Page) {
  return <K extends keyof Harness>(
    method: K,
    ...args: Parameters<Harness[K]>
  ): Promise<Awaited<ReturnType<Harness[K]>>> =>
    page.evaluate(
      ([m, a]) => {
        const harness = (window as unknown as { __instances: Record<string, unknown> }).__instances;
        return (harness[m] as (...x: unknown[]) => unknown)(...a);
      },
      [method, args] as [string, unknown[]],
    ) as Promise<Awaited<ReturnType<Harness[K]>>>;
}

/** Frames for the renderer to take a change in (PlayCanvas re-copies each tile it touches). */
const settle = (page: Page): Promise<void> => caller(page)("wait", 60);

test.skip(!process.env.PUBLISHED_SCANS, "PUBLISHED_SCANS is not set");

for (const scan of SCANS) {
  for (const renderer of RENDERERS) {
    test(`${scan.name} under ${renderer}: hide and highlight a category from the panel`, async ({
      page,
    }) => {
      test.setTimeout(3_600_000);
      const shot = (name: string): string => {
        const file = `${scan.name}-${renderer}-${name}.png`;
        if (!SHOTS) return test.info().outputPath(file);
        mkdirSync(SHOTS, { recursive: true });
        return join(SHOTS, file);
      };
      const errors: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      page.on("console", (message) => {
        if (message.type() === "error") errors.push(message.text());
      });
      await open(page, scan, renderer);
      const call = caller(page);
      await call("view", ...scan.view);
      const baseline = await call("view", ...scan.view);
      const status = await call("scan");
      const categories = await call("categories");
      // What is drawn now: tiles carrying ids, splats with none, splats of the category.
      const drawn = await call("drawn", scan.categoryId);
      await page.screenshot({ path: shot("before") });
      await page.locator("#panel").screenshot({ path: shot("panel") });

      await call("remember");
      await page.getByRole("button", { name: `Hide ${scan.category}`, exact: true }).click();
      await settle(page);
      const hidden = await call("measure");
      const hiddenChanged = await call("changed");
      await page.screenshot({ path: shot("hidden") });

      await page.getByRole("button", { name: "Reset" }).click();
      await settle(page);
      const reset = await call("measure");
      const resetChanged = await call("changed");

      await page.locator(`li[data-category="${scan.categoryId}"] > div > [data-row]`).click();
      await settle(page);
      const lit = await call("measure");
      await page.screenshot({ path: shot("highlight") });
      await page.locator("#panel").screenshot({ path: shot("panel-highlight") });
      // `PUBLISHED_EACH`: every category highlighted in turn, to review what went where.
      if (process.env.PUBLISHED_EACH) {
        for (const category of categories) {
          await page.locator(`li[data-category="${category.id}"] > div > [data-row]`).click();
          await settle(page);
          await page.screenshot({ path: shot(`each-${category.id}`) });
        }
      }

      const measures = {
        status,
        categories,
        drawn,
        baseline,
        hidden,
        hiddenChanged,
        reset,
        resetChanged,
        lit,
      };
      test.info().annotations.push({ type: "measures", description: JSON.stringify(measures) });
      if (SHOTS) {
        writeFileSync(
          join(SHOTS, `${scan.name}-${renderer}.json`),
          JSON.stringify(measures, null, 1),
        );
      }
      expect(errors.filter((e) => /shader|compile|link/i.test(e))).toEqual([]);
      expect(categories.map((c) => c.name)).toContain(scan.category);
      if (renderer !== "cesium") {
        expect(status?.native).toBe(false);
        expect(status?.instances?.matched).toBeGreaterThan(0);
      }
      // Hiding the category changes a visible share of the frame, and Reset brings it back.
      expect(hiddenChanged).toBeGreaterThan(0.01);
      expect(resetChanged).toBeLessThan(hiddenChanged / 4);
      // The highlight warms the category and dims the rest.
      expect(lit.warmth).toBeGreaterThan(baseline.warmth + 5);
      expect(lit.luma).toBeLessThan(baseline.luma);
    });
  }
}

// ---- Selecting in the scene -------------------------------------------------------------

/** `src/dev/sceneSelectHarness.ts`, as the page exposes it. */
interface SelectHarness {
  view(id: number, headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  screenOf(id: number): { x: number; y: number; splats: number } | null;
  instances(): { id: number; parent: number | null; splats: number }[];
  state(): { candidates: number[]; chain: number; index: number; selected: number | null };
  frames(count: number): Promise<void>;
  probe(x: number, y: number): Record<string, unknown>;
}

function selectHarnessHtml(scan: Scan, renderer: string): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Published scene select</title>
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
      const harness = await import("/src/dev/sceneSelectHarness.ts");
      window.__select = await harness.startSceneSelectHarness({
        container: document.getElementById("viewer"),
        url: "/published/${scan.run}/package/splat/tileset.json",
        renderer: "${renderer}",
      });
    </script>
  </body>
</html>`;
}

for (const scan of SCANS) {
  for (const renderer of RENDERERS) {
    test(`${scan.name} under ${renderer}: a click on an object selects it, near and far`, async ({
      page,
    }) => {
      test.setTimeout(3_600_000);
      await page.route("**/published-select.html", (route) =>
        route.fulfill({
          status: 200,
          contentType: "text/html",
          body: selectHarnessHtml(scan, renderer),
        }),
      );
      await open(page, scan, renderer, "/published-select.html", "__select");
      const call = <K extends keyof SelectHarness>(
        method: K,
        ...args: Parameters<SelectHarness[K]>
      ): Promise<Awaited<ReturnType<SelectHarness[K]>>> =>
        page.evaluate(
          ([m, a]) => {
            const harness = (window as unknown as { __select: Record<string, unknown> }).__select;
            return (harness[m] as (...x: unknown[]) => unknown)(...a);
          },
          [method, args] as [string, unknown[]],
        ) as Promise<Awaited<ReturnType<SelectHarness[K]>>>;
      const shot = (name: string): string => {
        const file = `${scan.name}-${renderer}-select-${name}.png`;
        if (!SHOTS) return test.info().outputPath(file);
        mkdirSync(SHOTS, { recursive: true });
        return join(SHOTS, file);
      };
      const [h0, p0, r0] = scan.select.views[0] ?? [0, -30, 10];
      await call("view", scan.select.id, h0, p0, r0);
      const parentOf = new Map((await call("instances")).map((i) => [i.id, i.parent]));
      const topOf = (id: number | null): number | null => {
        let at = id;
        for (let up = at === null ? null : (parentOf.get(at) ?? null); up !== null;) {
          at = up;
          up = parentOf.get(at) ?? null;
        }
        return at;
      };
      const results: Record<string, unknown>[] = [];
      for (const [heading, pitch, range] of scan.select.views) {
        await page.keyboard.press("Escape");
        await call("view", scan.select.id, heading, pitch, range);
        const target = await call("screenOf", scan.select.id);
        expect(target, `the object is on screen from ${String(range)} m`).not.toBeNull();
        const x = target?.x ?? 0;
        const y = target?.y ?? 0;
        const probe = await call("probe", x, y);
        await page.mouse.click(x, y);
        await call("frames", 5);
        const picked = await call("state");
        await page.screenshot({ path: shot(`${String(range)}m`) });
        const result = { range, x, y, probe, picked };
        results.push(result);
        console.info(JSON.stringify(result));
        if (range <= scan.select.exactWithinM) {
          expect(topOf(picked.selected), JSON.stringify(result)).toBe(topOf(scan.select.id));
        } else {
          expect(picked.selected, JSON.stringify(result)).not.toBeNull();
        }
        await expect(page.getByTestId("scene-select-label")).toBeVisible();
        if (picked.candidates.length > 1) {
          await page.keyboard.press("]");
          expect((await call("state")).index).toBe((picked.index + 1) % picked.candidates.length);
          await page.keyboard.press("[");
          expect((await call("state")).index).toBe(picked.index);
        }
      }
      test.info().annotations.push({ type: "measures", description: JSON.stringify(results) });
    });
  }
}
