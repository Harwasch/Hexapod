/**
 * The Living Survey wired into a real scene.
 *
 * `livingSurvey.spec.ts` proves the deformer against the engine. This one proves the manager
 * against the whole `CesiumSceneManager`: the interception installed at scene construction, a
 * site loaded through `SiteManager`, the tick on `scene.preUpdate` in `requestRenderMode`, and
 * the two properties that decide whether the feature is honest — **an idle scene stays idle**
 * and **zero wind is the measured pose, not an approximation of it**.
 *
 * How idleness is measured without disturbing it: `scene.preUpdate` fires every animation frame
 * whether or not anything renders, so the test waits on that and watches the manager's own calls
 * to `scene.requestRender()` — counted inside its tick alone — stop. A test that waited by
 * calling `requestRender()` itself would be waiting for the thing it is trying to observe, and a
 * test that counted drawn frames would be counting the rest of the scene too.
 *
 * **The frames here are not representative of anything visual.** Headless GL is SwiftShader,
 * frames run 740–1730 ms, and splat blending does not look like it does on a GPU. They are
 * only ever compared for *inequality* — that a frame changed — which is a statement about the
 * pipeline running end to end, not about how the motion reads. That still needs a human eye on
 * real hardware.
 *
 * Frames are deliberately **not** compared for equality, and the reason is the artifact this
 * whole sprint is about: the splat sorter runs asynchronously off canonical positions the
 * deformer never touches, so two frames at the same scene time can differ in draw order alone.
 * Byte-exactness of the restoring write is asserted where it is actually decidable — over the
 * uploaded words, in `src/__tests__/livingSurvey.test.ts`.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

/** `data/tiles/synthetic-tree`, from `apps/web/e2e/`. */
const TILE_ROOT = resolve(process.cwd(), "../../data/tiles/synthetic-tree");

/** Where `synthetic_tree.py` places the fixture, and where `captures.json` registers it. */
const LONGITUDE = -82.6966;
const LATITUDE = 28.0389;

const CONTENT_TYPES: Record<string, string> = {
  ".json": "application/json",
  ".glb": "model/gltf-binary",
};

const HARNESS_HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Living Survey scene harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/livingSceneHarness.ts");
      window.__livingScene = await harness.startLivingSceneHarness({
        container: document.getElementById("viewer"),
        tilesetUrl: new URL("/fixture-tiles/splat/tileset.json", location.href).toString(),
        slug: "synthetic-tree",
        longitude: ${LONGITUDE},
        latitude: ${LATITUDE},
      });
      document.title = "Living Survey scene harness ready";
    </script>
  </body>
</html>`;

async function openHarness(page: Page): Promise<void> {
  await page.route("**/fixture-tiles/**", (route) => {
    const url = new URL(route.request().url());
    const relative = url.pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    const file = resolve(TILE_ROOT, relative);
    const extension = file.slice(file.lastIndexOf("."));
    return route.fulfill({
      status: 200,
      contentType: CONTENT_TYPES[extension] ?? "application/octet-stream",
      body: readFileSync(file),
    });
  });
  // The rig the manager goes looking for: `livingRigs.ts` resolves `../source/rig.json` against
  // the tileset URL, which is exactly where the capture pipeline puts it on disk.
  await page.route("**/__living-scene", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: HARNESS_HTML }),
  );
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.goto("/__living-scene");
}

interface Counts {
  ticks: number;
  livingRequests: number;
  renderRequests: number;
  renders: number;
}

interface Status {
  wind: { strength: number; bearingDeg: number };
  animating: boolean;
  sites: {
    siteSlug: string;
    phase: string;
    reason?: string;
    numSplats: number;
    displaced: boolean;
    rigSourceNote: string;
    maxDisplacementM: number;
    sortStaleness: number;
  }[];
  captures: number;
}

/** Calls one harness method in the page and returns its result. */
async function call<T>(page: Page, method: string, ...args: unknown[]): Promise<T> {
  const result = await page.evaluate(
    async ({ method, args }) => {
      const harness = (window as unknown as { __livingScene: Record<string, unknown> })
        .__livingScene;
      const fn = harness[method] as (...a: unknown[]) => unknown;
      return await fn.apply(harness, args);
    },
    { method, args },
  );
  return result as T;
}

test.describe("Living Survey: the scene", () => {
  test("wind moves the tree, calm restores it, and idle stays idle", async ({ page }) => {
    test.setTimeout(300_000);
    await openHarness(page);
    await page.waitForFunction(() => "__livingScene" in window, undefined, { timeout: 90_000 });

    await call(page, "waitUntilReady", 150_000);
    const ready = await call<Status>(page, "status");
    // Attached through the real SiteManager, from a rig found beside the tileset, using a
    // packed buffer the interception saw because it was installed at scene construction.
    expect(ready.sites).toHaveLength(1);
    expect(ready.sites[0]?.phase).toBe("ready");
    expect(ready.sites[0]?.reason).toBeUndefined();
    expect(ready.sites[0]?.numSplats).toBe(2000);
    expect(ready.sites[0]?.siteSlug).toBe("synthetic-tree");
    expect(ready.captures).toBeGreaterThan(0);
    // Wind starts at 0 and the first thing anyone sees is the measurement.
    expect(ready.wind.strength).toBe(0);
    expect(ready.animating).toBe(false);

    await call(page, "setTime", 0);
    const measured = await call<string>(page, "grabFrameHash");

    // Wind up. Two different scene times must give two different frames, and both must differ
    // from the measured one — the motion is real, and it is driven by the scene clock.
    const wind = { strength: 0.6, bearingDeg: 250 };
    await call(page, "setWind", wind);
    await call(page, "setTime", 4);
    const blownA = await call<string>(page, "grabFrameHash");
    await call(page, "setTime", 9);
    const blownB = await call<string>(page, "grabFrameHash");

    expect(blownA).not.toBe(measured);
    expect(blownB).not.toBe(measured);
    expect(blownB).not.toBe(blownA);

    const blowing = await call<Status>(page, "status");
    expect(blowing.animating).toBe(true);
    expect(blowing.sites[0]?.displaced).toBe(true);
    expect(blowing.sites[0]?.maxDisplacementM).toBeGreaterThan(0.1);
    expect(blowing.sites[0]?.sortStaleness).toBeGreaterThan(1);

    // A frame keeps being asked for, every tick, while the wind blows.
    const gusting = await call<Counts>(page, "counts");
    const stillGusting = await call<Counts>(page, "waitTicks", 6, 120_000);
    expect(stillGusting.livingRequests - gusting.livingRequests).toBeGreaterThanOrEqual(5);

    // A snapshot is of the survey, not of the simulation. The live frames at these two scene
    // times differ (asserted above); their snapshots must not, because `snapshot()` pins both at
    // the same measured pose — same wind, same camera, same render settings, so the only thing
    // that could differ is where the splats are.
    await call(page, "setTime", 4);
    const heldA = await call<string>(page, "snapshotHash");
    await call(page, "setTime", 9);
    expect(await call<string>(page, "snapshotHash")).toBe(heldA);
    // And the hold is a photograph, not a change in the weather.
    expect((await call<Status>(page, "status")).wind.strength).toBe(wind.strength);

    // Calm. The deformer owes one restoring write, and after that the scene must go quiet:
    // ticks keep coming, renders stop. Three ticks of slack for the restore, then twelve ticks
    // in which nothing at all may be asked for.
    await call(page, "setWind", { strength: 0, bearingDeg: 250 });
    const restoring = await call<Counts>(page, "waitTicks", 3, 120_000);
    const idle = await call<Counts>(page, "waitTicks", 12, 120_000);
    expect(idle.ticks - restoring.ticks).toBeGreaterThanOrEqual(12);
    expect(idle.livingRequests - restoring.livingRequests).toBe(0);

    const rested = await call<Status>(page, "status");
    expect(rested.animating).toBe(false);
    expect(rested.sites[0]?.displaced).toBe(false);
    expect(rested.sites[0]?.phase).toBe("ready");

    // Destroy releases the tick and stops reporting; nothing is left listening.
    const probeBefore = await call<{ preUpdate: number }>(page, "probe");
    await call(page, "destroyLiving");
    const probeAfter = await call<{ preUpdate: number }>(page, "probe");
    expect(probeAfter.preUpdate).toBe(probeBefore.preUpdate - 1);
    expect((await call<Status>(page, "status")).sites).toHaveLength(0);
  });
});
