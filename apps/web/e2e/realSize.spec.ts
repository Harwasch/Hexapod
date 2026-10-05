/**
 * Real-world scale for a scan (docs/DATA_MODEL.md "Runtime scale"), in a real CesiumJS.
 *
 * The committed synthetic tree, registered as a pipeline run would register it, through
 * `CesiumSceneManager` and `SiteManager` (src/dev/realSizeHarness.ts). First: a catalog scale
 * of 0.5 draws it at half its registered size about its placed origin -- its pixels on screen,
 * its bounding sphere and its solids (what the camera bumps into and a measurement lands on)
 * all half. Then the Set real size tool on it with the API mocked: two points measured on the
 * scan, the true length typed, the scan previewed at the scale that says, Escape putting it
 * back, and Save sending exactly what `PUT /assets/{id}/scale` validates -- after a 401 that
 * asks for the write token, which the retry then carries.
 *
 * Headless GL is SwiftShader: sizes are measured, not looked at.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");
const SITE_ID = "77777777-7777-4777-8777-777777777777";
const ASSET_ID = "88888888-8888-4888-8888-888888888888";

const harnessHtml = (scale: number) => `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Real size harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #000; overflow: hidden; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
      #panel {
        position: fixed; top: 12px; right: 12px; width: 22rem; z-index: 10;
        padding: 12px; box-sizing: border-box; background: #10141a; color: #fff;
      }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <div id="panel"></div>
    <script type="module">
      // The React plugin's preamble, which Vite injects into its own pages (the tool is React).
      import RefreshRuntime from "/@react-refresh";
      RefreshRuntime.injectIntoGlobalHook(window);
      window.$RefreshReg$ = () => {};
      window.$RefreshSig$ = () => (type) => type;
      window.__vite_plugin_react_preamble_installed__ = true;
    </script>
    <script type="module">
      const harness = await import("/src/dev/realSizeHarness.ts");
      window.__realSize = await harness.startRealSizeHarness({
        container: document.getElementById("viewer"),
        panel: document.getElementById("panel"),
        tilesetUrl: new URL("/fixture-tiles/synthetic-tree-lod/tileset.json", location.href).toString(),
        scale: ${String(scale)},
      });
    </script>
  </body>
</html>`;

interface Harness {
  radius(): number;
  shownScale(): number | null;
  previewScale(scale: number | null): boolean;
  edgeEast(heightM: number): number | null;
  screenOf(local: [number, number, number]): { x: number; y: number } | null;
  extent(): Promise<{ width: number; height: number; pixels: number }>;
  settle(): Promise<void>;
  measuredM(): number | null;
  record(): unknown;
}

/** Routes the fixture tiles and the harness page, once per page; `scale` is the page's query. */
async function route(page: Page, errors: string[]): Promise<void> {
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/fixture-tiles/**", (request) => {
    const relative = new URL(request.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return request.abort();
    return request.fulfill({
      status: 200,
      contentType: relative.endsWith(".json") ? "application/json" : "application/octet-stream",
      body: readFileSync(resolve(TILES, relative)),
    });
  });
  await page.route(/\/__real-size\?scale=/, (request) => {
    const scale = Number(new URL(request.request().url()).searchParams.get("scale") ?? "1");
    return request.fulfill({ status: 200, contentType: "text/html", body: harnessHtml(scale) });
  });
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (request) => request.abort());
}

/** The harness with the tree registered at `scale`, its first view settled. */
async function open(page: Page, scale: number): Promise<void> {
  await page.goto(`/__real-size?scale=${String(scale)}`);
  await page.waitForFunction(() => "__realSize" in window, undefined, { timeout: 240_000 });
}

function call<K extends keyof Harness>(
  page: Page,
  method: K,
  ...args: Parameters<Harness[K]>
): Promise<Awaited<ReturnType<Harness[K]>>> {
  return page.evaluate(
    ([m, a]) => {
      const harness = (window as unknown as { __realSize: Record<string, unknown> }).__realSize;
      return (harness[m] as (...x: unknown[]) => unknown)(...a);
    },
    [method, args] as [string, unknown[]],
  ) as Promise<Awaited<ReturnType<Harness[K]>>>;
}

test("a placed splat at renderConfig.scale 0.5 is drawn and solid at half its size", async ({
  page,
}) => {
  test.setTimeout(400_000);
  const errors: string[] = [];
  await route(page, errors);
  await open(page, 0.5);
  expect(await call(page, "shownScale")).toBe(0.5);
  const half = {
    radius: await call(page, "radius"),
    extent: await call(page, "extent"),
    // The crown's west side, level with the middle of the tree as drawn.
    edge: await call(page, "edgeEast", 1.5),
  };

  // The same scan registered at 1, from the same camera, in a fresh page.
  await open(page, 1);
  expect(await call(page, "shownScale")).toBe(1);
  const full = {
    radius: await call(page, "radius"),
    extent: await call(page, "extent"),
    edge: await call(page, "edgeEast", 3),
  };
  test.info().annotations.push({ type: "sizes", description: JSON.stringify({ half, full }) });

  expect(half.radius / full.radius).toBeCloseTo(0.5, 3);
  expect(half.edge).not.toBeNull();
  expect(full.edge).not.toBeNull();
  // Solid where it is drawn: the collider meets it half as far from the origin.
  expect((half.edge ?? 0) / (full.edge ?? 1)).toBeCloseTo(0.5, 2);
  // Drawn: its box on screen is half as tall and half as wide.
  expect(full.extent.pixels).toBeGreaterThan(500);
  expect(half.extent.height / full.extent.height).toBeGreaterThan(0.42);
  expect(half.extent.height / full.extent.height).toBeLessThan(0.58);
  expect(half.extent.width / full.extent.width).toBeGreaterThan(0.42);
  expect(half.extent.width / full.extent.width).toBeLessThan(0.58);

  // A preview of half on the scan registered at 1: its sphere and its solids follow at once.
  expect(await call(page, "previewScale", 0.5)).toBe(true);
  expect(await call(page, "radius")).toBeCloseTo(half.radius, 6);
  await expect.poll(() => call(page, "edgeEast", 1.5)).toBeCloseTo(half.edge ?? 0, 2);
  expect(errors).toEqual([]);
});

test("Set real size: measured on the scan, previewed, cancelled, then saved with the token", async ({
  page,
}) => {
  test.setTimeout(400_000);
  const errors: string[] = [];
  const sent: { body: unknown; authorization: string | null }[] = [];
  let saved: Record<string, unknown> | null = null;
  await page.route(`**/api/v1/assets/${ASSET_ID}/scale`, async (route) => {
    const request = route.request();
    const body = request.postDataJSON() as {
      scale: number;
      evidence: Record<string, unknown>;
    };
    sent.push({ body, authorization: await request.headerValue("authorization") });
    // The first try has no token, and the server wants one.
    if (sent.length === 1) {
      return route.fulfill({
        status: 401,
        json: { title: "Unauthorized", status: 401, detail: "A write token is required." },
      });
    }
    saved = { scale: body.scale, scaleEvidence: body.evidence };
    return route.fulfill({
      status: 200,
      json: {
        id: ASSET_ID,
        siteId: SITE_ID,
        provider: "3d-tiles-url",
        name: "Gaussian splat",
        representation: "gaussian-splat",
        source: {
          type: "3d-tiles-url",
          url: new URL("/fixture-tiles/synthetic-tree-lod/tileset.json", request.url()).toString(),
        },
        footprint: null,
        observedAt: null,
        validFrom: null,
        validTo: null,
        resolution: null,
        crs: null,
        license: null,
        attribution: [],
        provenance: { georefMethod: "exif-gps", scaleSource: "manual", uncertaintyM: 10 },
        renderConfig: {
          maximumScreenSpaceError: 1,
          pointCloudShading: null,
          clipsWorld: false,
          clipFootprint: "catalog",
          heightOffsetM: 0,
          scale: body.scale,
          scaleEvidence: body.evidence,
        },
        defaultVisible: true,
        createdAt: "2026-01-01T00:00:00Z",
        updatedAt: "2026-01-01T00:00:00Z",
      },
    });
  });
  // The record fetched afresh after the save, as the API answers it: the asset at its saved
  // scale (the page's own copy, which the save already put it in).
  await page.route(`**/api/v1/sites/${SITE_ID}`, async (route) => {
    const record = await call(page, "record");
    return record && saved
      ? route.fulfill({ status: 200, json: record })
      : route.fulfill({ status: 404, json: { title: "Not found", status: 404 } });
  });
  await route(page, errors);
  await open(page, 1);

  const measure = async (): Promise<number> => {
    await page.getByTestId("real-size-measure").click();
    await expect(page.getByTestId("real-size-picking")).toContainText("point 1 of 2");
    // Low on the trunk, then up in the crown: wherever the rays meet the tree's solids.
    for (const local of [
      [0, 0, 0.8],
      [0, 0, 6],
    ] as [number, number, number][]) {
      const at = await call(page, "screenOf", local);
      expect(at).not.toBeNull();
      await page.mouse.click(at?.x ?? 0, at?.y ?? 0);
    }
    await expect(page.getByTestId("real-size-measured")).toBeVisible();
    const measured = (await call(page, "measuredM")) ?? 0;
    expect(measured).toBeGreaterThan(1);
    return measured;
  };

  await page.getByTestId("real-size-open").click();
  let measured = await measure();
  // Really half as long: the preview draws the tree at half its size.
  await page.getByTestId("real-size-true").fill((measured / 2).toFixed(4));
  await expect.poll(() => call(page, "shownScale")).toBeCloseTo(0.5, 3);
  await expect(page.getByTestId("real-size-preview")).toContainText("×0.5");

  // Escape: the tree as it was, nothing sent.
  await page.keyboard.press("Escape");
  await expect.poll(() => call(page, "shownScale")).toBe(1);
  await expect(page.getByTestId("real-size-open")).toBeVisible();
  expect(sent).toEqual([]);

  // Again, and saved this time.
  await page.getByTestId("real-size-open").click();
  measured = await measure();
  const trueLength = Number((measured / 2.5).toFixed(4));
  await page.getByTestId("real-size-true").fill(String(trueLength));
  await expect.poll(() => call(page, "shownScale")).toBeCloseTo(0.4, 3);
  await page.getByTestId("real-size-save").click();

  // The server wants a token: asked for, then the same request again, carrying it.
  await expect(page.getByTestId("write-token-form")).toContainText("save a scan's size");
  await page.getByTestId("write-token-input").fill("e2e-token");
  await page.getByTestId("write-token-save").click();
  await expect.poll(() => sent.length).toBe(2);
  expect(sent[0]?.authorization).toBeNull();
  expect(sent[1]?.authorization).toBe("Bearer e2e-token");
  const body = sent[1]?.body as {
    scale: number;
    evidence: {
      method: string;
      measuredLengthM: number;
      trueLengthM: number;
      measuredAtScale: number;
    };
  };
  expect(body.evidence.method).toBe("measured-length");
  expect(body.evidence.measuredAtScale).toBe(1);
  expect(body.evidence.trueLengthM).toBeCloseTo(trueLength, 9);
  expect(body.evidence.measuredLengthM).toBeCloseTo(measured, 6);
  // What the API checks: the length implies the scale sent beside it, within a thousandth.
  const implied =
    (body.evidence.measuredAtScale * body.evidence.trueLengthM) / body.evidence.measuredLengthM;
  expect(Math.abs(implied - body.scale)).toBeLessThan(1e-3 * body.scale);

  // Saved: the tool closes and the record keeps the scan at the scale it saved.
  await expect(page.getByTestId("real-size-open")).toBeVisible();
  await expect(page.getByTestId("real-size-current")).toContainText("×0.4");
  await expect.poll(() => call(page, "shownScale")).toBeCloseTo(body.scale, 9);
  expect(errors).toEqual([]);
});
