/**
 * The camera against a splat, in a real CesiumJS: it stops at the surface it is driven into,
 * slides along it rather than sticking, passes through with Space held, and a wheel over the
 * splat zooms to it and stops short -- none of which CesiumJS does itself, since a splat
 * writes no depth and has no triangles to collide with (cesium/SplatCollider.ts).
 *
 * The committed synthetic tree, packed as a merged-parent LOD tileset: a crown some 8 m
 * across on a trunk. Headless GL is SwiftShader; positions are measured, not looks.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const TILES = resolve(process.cwd(), "../../data/tiles");

const HARNESS_HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Splat navigation harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-viewer, #viewer .cesium-widget, #viewer canvas {
        width: 100vw; height: 100vh; display: block;
      }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/collisionHarness.ts");
      window.__nav = await harness.startCollisionHarness(
        document.getElementById("viewer"),
        "/fixture-tiles/synthetic-tree-lod/tileset.json",
      );
      await window.__nav.ready();
      window.__navReady = true;
    </script>
  </body>
</html>`;

interface Nav {
  place(range: number, headingDeg?: number): void;
  distanceToCentre(): number;
  hitAhead(): number | null;
  eye(): [number, number, number];
  drive(stepM: number, frames: number, sideways?: number): Promise<number | null>;
  toasts: string[];
}

async function open(page: Page): Promise<void> {
  await page.route("**/fixture-tiles/**", (route) => {
    const relative = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    return route.fulfill({
      status: 200,
      contentType: relative.endsWith(".json") ? "application/json" : "model/gltf-binary",
      body: readFileSync(resolve(TILES, relative)),
    });
  });
  await page.route("**/__splat-nav", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: HARNESS_HTML }),
  );
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.goto("/__splat-nav");
  await page.waitForFunction(() => "__navReady" in window, undefined, { timeout: 180_000 });
}

const nav = <T>(page: Page, run: (nav: Nav) => T | Promise<T>): Promise<T> =>
  page.evaluate(
    // The function is serialised; it finds the harness itself.
    `(${run.toString()})(window.__nav)`,
  );

test("the camera stops at a splat, slides along it, and passes through with Space", async ({
  page,
}) => {
  test.setTimeout(400_000);
  await open(page);

  // Ten metres out, level, looking at the middle of the tree: crown, and the trunk at its axis.
  await nav(page, (n) => n.place(10));
  expect(await nav(page, (n) => n.hitAhead())).not.toBeNull();

  // Driven straight at it, 0.5 m a frame for 40 frames (20 m, which would end 10 m beyond
  // the middle): it meets the tree and never gets inside its clearance of any splat surface
  // -- it stops, or slides round what it met (a trunk is narrow), as a game camera does.
  const closest = await nav(page, (n) => n.drive(0.5, 40));
  expect(closest).not.toBeNull();
  expect(closest ?? 0).toBeGreaterThan(0.8);
  expect(await nav(page, (n) => n.toasts)).toContain("Hold Space to pass through");

  // Pressed into it at an angle, it keeps moving along it rather than sticking (4 m asked
  // sideways), still never inside.
  await nav(page, (n) => n.place(6));
  await nav(page, (n) => n.drive(0.3, 15));
  const [x0, y0, z0] = await nav(page, (n) => n.eye());
  const sliding = await nav(page, (n) => n.drive(0.2, 20, 0.2));
  const [x1, y1, z1] = await nav(page, (n) => n.eye());
  expect(Math.hypot(x1 - x0, y1 - y0, z1 - z0)).toBeGreaterThan(1);
  if (sliding !== null) expect(sliding).toBeGreaterThan(0.8);

  // Space held: straight through the middle and out the other side.
  await nav(page, (n) => n.place(10));
  await page.locator("#viewer canvas").first().focus();
  await page.keyboard.down("Space");
  const through = await nav(page, (n) => n.drive(0.5, 30));
  await page.keyboard.up("Space");
  // Through the crown: right up against (and into) splats on the way.
  expect(through ?? 1).toBeLessThan(0.8);
  // 15 m driven from 10 m out: 5 m past the centre.
  expect(await nav(page, (n) => n.distanceToCentre())).toBeGreaterThan(3);
  expect(await nav(page, (n) => n.hitAhead())).toBeNull();
});

test("a wheel over a splat zooms to its surface and stops short of it", async ({ page }) => {
  test.setTimeout(400_000);
  await open(page);
  await nav(page, (n) => n.place(12));
  const surface = (await nav(page, (n) => n.hitAhead())) ?? 0;
  expect(surface).toBeGreaterThan(0);
  const size = page.viewportSize() ?? { width: 1280, height: 720 };
  await page.mouse.move(size.width / 2, size.height / 2);
  for (let i = 0; i < 30; i++) await page.mouse.wheel(0, -100);
  await page.waitForTimeout(500);
  const left = (await nav(page, (n) => n.hitAhead())) ?? Number.NaN;
  // Right up against it -- centimetres to decimetres, the clearance -- and never through.
  expect(left).toBeGreaterThan(0);
  expect(left).toBeLessThan(0.5);
  expect(await nav(page, (n) => n.distanceToCentre())).toBeGreaterThan(12 - surface - 0.5);
});

interface Explore {
  explore: {
    enter(): void;
    exit(): void;
    mode(): string;
    heading(): number;
    height(): number;
  };
  distanceToCentre(): number;
  hitAhead(): number | null;
  clearanceRatio(): number | null;
  place(range: number, headingDeg?: number): void;
}

test("explore: WASD flies toward the tree and stops at it, Space rises, F walks, drag looks", async ({
  page,
}) => {
  test.setTimeout(400_000);
  await open(page);
  const run = <T>(fn: (n: Explore) => T): Promise<T> =>
    page.evaluate(`(${fn.toString()})(window.__nav)`);
  await run((n) => n.place(12));
  // Nothing to stand on here (the globe is hidden, the tree has no ground): it starts flying.
  await run((n) => n.explore.enter());
  expect(await run((n) => n.explore.mode())).toBe("fly");
  // Space rises when flying (out in the open, before meeting the tree).
  const low = await run((n) => n.explore.height());
  // Held two seconds: 4 m/s, less under SwiftShader's slow frames (each step is at most 0.1 s,
  // so a slow machine moves slower rather than jumping).
  await page.keyboard.down("Space");
  await page.waitForTimeout(2000);
  await page.keyboard.up("Space");
  expect(await run((n) => n.explore.height())).toBeGreaterThan(low + 0.5);
  const start = await run((n) => n.distanceToCentre());
  // Held until it has come 3 m in (real time, so SwiftShader's slow frames only make it
  // longer), then it meets the crown: stopped at it or sliding round it, never inside its
  // clearance.
  await page.keyboard.down("KeyW");
  await expect
    .poll(() => run((n) => n.distanceToCentre()), { timeout: 60_000 })
    .toBeLessThan(start - 3);
  await page.waitForTimeout(3000);
  await page.keyboard.up("KeyW");
  const ratio = await run((n) => n.clearanceRatio());
  if (ratio !== null) expect(ratio).toBeGreaterThan(0.8);
  // F switches to walking (and back).
  await page.keyboard.press("KeyF");
  expect(await run((n) => n.explore.mode())).toBe("walk");
  await page.keyboard.press("KeyF");
  expect(await run((n) => n.explore.mode())).toBe("fly");
  // Dragging turns the view.
  const before = await run((n) => n.explore.heading());
  const size = page.viewportSize() ?? { width: 1280, height: 720 };
  await page.mouse.move(size.width / 2, size.height / 2);
  await page.mouse.down();
  await page.mouse.move(size.width / 2 + 200, size.height / 2, { steps: 5 });
  await page.mouse.up();
  const after = await run((n) => n.explore.heading());
  expect(Math.abs(after - before)).toBeGreaterThan(0.05);
  await run((n) => n.explore.exit());
});
