/**
 * Sunlight in a real CesiumJS 1.145 with the engine patch's `vertexColor` hook, and in
 * PlayCanvas 2.22.6 drawing the scan as the app has it (ScanRendererHost, its per-frame hook:
 * cesium/scanView/sunHook.ts), on a slab built here (`src/dev/sunlightSlab.ts`): a wall facing
 * east at the synthetic yard, flat discs on its south half, round splats of the same grey on its
 * north half, seen from the east (`src/dev/sunlightHarness.ts`).
 *
 * Under each renderer the relight compiles and lights the discs by the sun of the scene clock:
 * brighter at 07:15 local solar time, the sun in the east, than at 16:45, the sun in the west,
 * behind the wall, each by the mirror's arithmetic (lib/splatNormals.ts). The round splats
 * have no normal and are as photographed by day. At night everything is at the ambient floor.
 * With the sun off, the clock changes nothing; with As captured on, the sun asked for changes
 * nothing either. With Confidence drawn over the wall, the look is tinted by the relit
 * luminance under both -- the sun before the look, CesiumJS's order, which PlayCanvas keeps by
 * the word its copy pass leaves beside each splat -- not the look relit afterwards. Headless GL
 * is SwiftShader: pixels are measured, not eyeballed.
 */

import { gzipSync } from "node:zlib";

import { expect, test } from "@playwright/test";

import { slabGlb, slabSplats, slabSpz, slabTileset } from "../src/dev/sunlightSlab";

const EVIDENCE = process.env.W6B_EVIDENCE_DIR;

const HARNESS_HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Sunlight harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/sunlightHarness.ts");
      window.__sunlight = await harness.startSunlightHarness({
        container: document.getElementById("viewer"),
        url: "/sunlight-slab/tileset.json",
        renderer: new URLSearchParams(location.search).get("renderer") ?? "cesium",
      });
    </script>
  </body>
</html>`;

interface SlabView {
  coverage: number;
  flat: number;
  round: number;
  predictedFlat: number;
  predictedCellFlat: number;
  sine: number;
  parts: string[];
  reported: { at: string; sine: number } | null;
}

/** 07:15 and 16:45 local solar time at 123.88° W on the June solstice, and 00:00. */
const MORNING = "2026-06-21T15:30:00Z";
const EVENING = "2026-06-22T01:00:00Z";
const NIGHT = "2026-06-21T08:21:00Z";

/** What each renderer's relight is called where the harness looks for it. */
const RELIT = { cesium: "splatSunlightColor", playcanvas: "sunHook" } as const;

for (const renderer of ["cesium", "playcanvas"] as const) {
  test(`under ${renderer}, an east-facing slab is lit by the morning sun, not the evening one, and only when asked`, async ({
    page,
  }) => {
    test.setTimeout(300_000);
    await page.setViewportSize({ width: 960, height: 640 });
    const errors: string[] = [];
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    page.on("pageerror", (error) => errors.push(error.message));
    const splats = slabSplats();
    const glb = slabGlb(splats, gzipSync(slabSpz(splats), { level: 6 }));
    await page.route("**/sunlight-slab/**", (route) => {
      const path = new URL(route.request().url()).pathname;
      if (path.endsWith("tileset.json"))
        return route.fulfill({ status: 200, json: slabTileset(splats) });
      if (path.endsWith("slab.glb"))
        return route.fulfill({
          status: 200,
          contentType: "model/gltf-binary",
          body: Buffer.from(glb),
        });
      return route.fulfill({ status: 404, body: "" });
    });
    await page.route(/\/sunlight-harness\.html(\?.*)?$/, (route) =>
      route.fulfill({ status: 200, contentType: "text/html", body: HARNESS_HTML }),
    );
    await page.goto(`/sunlight-harness.html?renderer=${renderer}`);
    await page.waitForFunction(() => "__sunlight" in window, undefined, { timeout: 90_000 });

    const view = async (
      name: string,
      options: { sun: string; at: string; asCaptured?: boolean; cell?: boolean },
    ) => {
      const result = await page.evaluate(
        (o) =>
          (
            window as unknown as {
              __sunlight: { view(o: unknown): Promise<SlabView> };
            }
          ).__sunlight.view(o),
        options,
      );
      const file = `scene-${renderer}-${name}.png`;
      const path = EVIDENCE ? `${EVIDENCE}/${file}` : test.info().outputPath(file);
      await page.screenshot({ path });
      await test.info().attach(name, { path, contentType: "image/png" });
      return result;
    };

    // Twice: the first view also waits for the tile.
    await view("off-morning", { sun: "off", at: MORNING });
    const offMorning = await view("off-morning", { sun: "off", at: MORNING });
    const offEvening = await view("off-evening", { sun: "off", at: EVENING });
    const morning = await view("now-morning", { sun: "now", at: MORNING });
    const evening = await view("now-evening", { sun: "now", at: EVENING });
    const night = await view("now-night", { sun: "now", at: NIGHT });
    const asCaptured = await view("as-captured-morning", {
      sun: "now",
      at: MORNING,
      asCaptured: true,
    });
    const backOn = await view("now-evening-again", { sun: "now", at: EVENING });
    // Confidence over the wall, every cell strong: the look first unlit, then by each sun.
    const cellOff = await view("confidence-off", { sun: "off", at: EVENING, cell: true });
    const cellMorning = await view("confidence-morning", { sun: "now", at: MORNING, cell: true });
    const cellEvening = await view("confidence-evening", { sun: "now", at: EVENING, cell: true });
    const off = await view("off-again", { sun: "off", at: EVENING });
    const numbers = {
      renderer,
      offMorning,
      offEvening,
      morning,
      evening,
      night,
      asCaptured,
      backOn,
      cellOff,
      cellMorning,
      cellEvening,
      off,
    };
    test.info().annotations.push({ type: "brightness", description: JSON.stringify(numbers) });
    console.info(JSON.stringify(numbers));

    expect(errors.filter((e) => /shader|compile|link|uniform/i.test(e))).toEqual([]);
    expect(offMorning.coverage).toBeGreaterThan(0.05);
    expect(offMorning.parts).toEqual([]);
    // The sun off: the clock changes nothing.
    expect(Math.abs(offEvening.flat - offMorning.flat)).toBeLessThan(2);
    expect(Math.abs(offEvening.round - offMorning.round)).toBeLessThan(2);

    // On: the part is in the chain, and it reported the frame it drew.
    expect(morning.parts).toEqual([RELIT[renderer]]);
    expect(morning.reported?.at).toBe("2026-06-21T15:30:00.000Z");
    expect(morning.sine).toBeGreaterThan(0.3);
    expect(evening.sine).toBeGreaterThan(0.3);
    // The morning sun is in the east, in front of the wall's face; the evening one behind it.
    expect(morning.predictedFlat).toBeGreaterThan(0.85);
    expect(evening.predictedFlat).toBeCloseTo(0.35, 2);
    expect(morning.flat).toBeGreaterThan(1.6 * evening.flat);
    // Lit, the discs are as photographed less what the lambert term takes; not brighter.
    expect(morning.flat).toBeLessThanOrEqual(offMorning.flat + 2);
    expect(morning.flat).toBeGreaterThan(0.75 * offMorning.flat);
    // And by the mirror's arithmetic (lib/splatNormals.ts): the wall is opaque, so a pixel is the
    // splat's colour times the factor, give or take the dark background at its edges and through
    // its 2 % transparency -- a tenth either way is that, from first principles, not a fit.
    expect(Math.abs(morning.flat / offMorning.flat - morning.predictedFlat)).toBeLessThan(0.1);
    expect(Math.abs(evening.flat / offMorning.flat - evening.predictedFlat)).toBeLessThan(0.1);
    // The round splats have no normal: as photographed by day, whichever side the sun is.
    expect(Math.abs(morning.round - offMorning.round)).toBeLessThan(3);
    expect(Math.abs(evening.round - offMorning.round)).toBeLessThan(3);

    // Night: the sun below the horizon, everything at the ambient floor, and reported so.
    expect(night.sine).toBeLessThan(0);
    expect(night.reported?.sine).toBeLessThan(0);
    expect(night.round).toBeLessThan(0.6 * offMorning.round);
    expect(night.flat).toBeLessThan(0.6 * offMorning.flat);

    // As captured: the sun asked for, nothing relit, nothing reported.
    expect(asCaptured.parts).toEqual([]);
    expect(asCaptured.reported).toBeNull();
    expect(Math.abs(asCaptured.flat - offMorning.flat)).toBeLessThan(2);
    expect(Math.abs(asCaptured.round - offMorning.round)).toBeLessThan(2);
    // And back.
    expect(Math.abs(backOn.flat - evening.flat)).toBeLessThan(2);

    // Confidence: the look of the relit splat, as CesiumJS orders them (sun 3, look 5) -- by the
    // mirror's arithmetic, give or take the same tenth -- and not the look relit afterwards,
    // which the evening sun (behind the wall: the discs at the ambient floor) tells apart.
    expect(cellOff.coverage).toBeGreaterThan(0.05);
    expect(cellEvening.predictedCellFlat - cellEvening.predictedFlat).toBeGreaterThan(0.3);
    for (const lit of [cellMorning, cellEvening]) {
      expect(Math.abs(lit.flat / cellOff.flat - lit.predictedCellFlat)).toBeLessThan(0.1);
    }
    expect(cellEvening.flat / cellOff.flat - cellEvening.predictedFlat).toBeGreaterThan(0.2);
    // The round splats keep their look by day.
    expect(Math.abs(cellEvening.round - cellOff.round)).toBeLessThan(3);

    expect(off.parts).toEqual([]);
    expect(Math.abs(off.flat - offMorning.flat)).toBeLessThan(2);
  });
}
