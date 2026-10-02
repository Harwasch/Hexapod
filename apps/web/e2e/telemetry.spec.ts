/**
 * Telemetry drives rigid instances (step C3: `cesium/telemetry.ts`, `cesium/splatRigid.ts`) in
 * a real CesiumJS with the engine patch, on the committed synthetic yard.
 *
 * `data/tiles/synthetic-yard/telemetry/telemetry.json` binds two synthetic sources: the box
 * building (instance 8, unskinned: the rigid part of the motion chain, keyed by instance ids)
 * drives a 2.5 m circle north of where it was scanned, its readings in ECEF at 2 Hz; the shrub
 * 10 (skinned, and swayed by the wind's materials) a 0.6 m circle, its readings geodetic at
 * 5 Hz (its skin's constant handle). Both paths start at the rest pose. The route links
 * `instances.json`, `skin.json`, `materials.json` and `telemetry.json` from the root's extras.
 *
 * What must hold: the building follows its path (the pose shown is the path at the playout
 * time, its pixels leave its rest place and arrive where the path is) while unbound objects
 * stay still; once its source falls silent it holds, then fades back to the measured frame
 * exactly; the shrub, set to freeze, holds its last pose; a bound skin is not swayed by the
 * wind (its elastic handles stay at rest) while the tree still is; the same clock gives the
 * same frame. Frames go to `test.info().outputPath()` and, with `TELEMETRY_FRAMES_DIR`, there.
 *
 * Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import { mkdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface TelemetryStatus {
  instance: number;
  source: string;
  state: string;
  ageMs: number | null;
  via: string;
  position: number[] | null;
  orientation: number[] | null;
  playoutMs: number;
}

/** `src/dev/skinHarness.ts`, as the page exposes it (what this spec uses). */
interface SkinHarness {
  view(headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  frame(): number;
  difference(a: number, b: number, rect?: Rect, tol?: number): number;
  rectOf(id: number, grow?: number): Rect | null;
  scanRect(point: [number, number, number], radiusM: number): Rect | null;
  telemetryReady(): Promise<void>;
  telemetryAt(ms: number): Promise<void>;
  telemetryStatus(): TelemetryStatus[];
  mute(sourceId: string, muted: boolean): void;
  telemetryReset(ms: number): Promise<void>;
  rigidInfo(ids: number[]): { driven: number[]; active: boolean; slots: number[] };
  skinHandles(instance: number): number[] | null;
  windOn(strength: number, bearingDeg: number, t0?: number, seed?: number): Promise<void>;
  advance(seconds: number, fps?: number): Promise<void>;
  windSkins(): { instance: number; wind: boolean; claimed: boolean }[];
}

const TILES = resolve(process.cwd(), "../../data/tiles");
const TILESET = "synthetic-yard/splat/tileset.json";
const INSTANCES = "../instances/instances.json";
const SKIN = "../skin/skin.json";
const MATERIALS = "../skin/materials.json";
const TELEMETRY = "../telemetry/telemetry.json";

/** The building's path (telemetry.json "yard-loop"), scan frame. */
const LOOP = { centre: [20.0, 8.72045, -0.113], radius: 2.5, periodS: 24, phaseDeg: -90 };
/** The building's rest place (its base centre), scan frame. */
const BUILDING_REST: [number, number, number] = [20.0, 6.22045, -0.113];

function loopAt(ms: number): [number, number, number] {
  const theta = ((LOOP.phaseDeg + (360 * ms) / 1000 / LOOP.periodS) * Math.PI) / 180;
  return [
    (LOOP.centre[0] ?? 0) + LOOP.radius * Math.cos(theta),
    (LOOP.centre[1] ?? 0) + LOOP.radius * Math.sin(theta),
    LOOP.centre[2] ?? 0,
  ];
}

function harnessHtml(): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Telemetry harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/skinHarness.ts");
      window.__skin = await harness.startSkinHarness({
        container: document.getElementById("viewer"),
        url: "/fixture-tiles/${TILESET}",
        incremental: true,
        maximumScreenSpaceError: 1,
        sorter: true,
      });
    </script>
  </body>
</html>`;
}

async function open(page: Page, errors: string[]): Promise<void> {
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  const read = (relative: string): Record<string, unknown[] | undefined> =>
    JSON.parse(readFileSync(resolve(TILES, "synthetic-yard/splat", relative), "utf-8")) as Record<
      string,
      unknown[] | undefined
    >;
  const instances = read(INSTANCES).instances ?? [];
  const skins = read(SKIN).skins ?? [];
  const bindings = read(TELEMETRY).bindings ?? [];
  await page.route("**/fixture-tiles/**", (route) => {
    const path = new URL(route.request().url()).pathname.replace(/^.*\/fixture-tiles\//, "");
    const relative = resolve("/", path).slice(1);
    if (relative !== path) return route.abort();
    if (relative === TILESET) {
      const tileset = JSON.parse(readFileSync(resolve(TILES, relative), "utf-8")) as {
        root: { extras?: Record<string, unknown> };
      };
      tileset.root.extras = {
        ...tileset.root.extras,
        instances: { uri: INSTANCES, count: instances.length },
        skin: { uri: SKIN, count: skins.length },
        materials: { uri: MATERIALS, count: 3 },
        telemetry: { uri: TELEMETRY, count: bindings.length },
      };
      return route.fulfill({ status: 200, json: tileset });
    }
    const type = relative.endsWith(".json")
      ? "application/json"
      : relative.endsWith(".glb")
        ? "model/gltf-binary"
        : "application/octet-stream";
    return route.fulfill({
      status: 200,
      contentType: type,
      body: readFileSync(resolve(TILES, relative)),
    });
  });
  await page.route("**/telemetry-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: harnessHtml() }),
  );
  await page.goto("/telemetry-harness.html");
  await page.waitForFunction(() => "__skin" in window, undefined, { timeout: 120_000 });
}

function caller(page: Page) {
  return <K extends keyof SkinHarness>(
    method: K,
    ...args: Parameters<SkinHarness[K]>
  ): Promise<Awaited<ReturnType<SkinHarness[K]>>> =>
    page.evaluate(
      ([m, a]) => {
        const harness = (window as unknown as { __skin: Record<string, unknown> }).__skin;
        return (harness[m] as (...x: unknown[]) => unknown)(...a);
      },
      [method, args] as [string, unknown[]],
    ) as Promise<Awaited<ReturnType<SkinHarness[K]>>>;
}

// A smaller canvas: every frame here is drawn in software.
test.use({ viewport: { width: 960, height: 600 } });

const shaderErrors = (errors: string[]): string[] =>
  errors.filter((e) => /shader|compile|link|webgl/i.test(e));

function shooter(page: Page) {
  const framesDir = process.env.TELEMETRY_FRAMES_DIR;
  if (framesDir) mkdirSync(framesDir, { recursive: true });
  return async (name: string): Promise<void> => {
    await page.screenshot({
      path: framesDir ? resolve(framesDir, `${name}.png`) : test.info().outputPath(`${name}.png`),
    });
  };
}

const statusOf = (all: TelemetryStatus[], instance: number): TelemetryStatus => {
  const found = all.find((s) => s.instance === instance);
  if (!found) throw new Error(`no binding for ${String(instance)}`);
  return found;
};

test("telemetry moves the bound building along its path, the rest stay still, stale fades to rest", async ({
  page,
}) => {
  test.setTimeout(1_500_000);
  const errors: string[] = [];
  await open(page, errors);
  const call = caller(page);
  const shot = shooter(page);

  await call("view", 20, -40, 40);
  await call("view", 20, -40, 40);
  await call("telemetryReady");
  await call("telemetryAt", 0);
  const rest = await call("frame");
  await shot("telemetry-rest");
  const startStatus = await call("telemetryStatus");

  const building = (await call("rectOf", 8, 1.1)) ?? undefined;
  // Unbound objects clear of the building's loop on screen (the tree 1's sphere spans it).
  const otherIds = [3, 7, 9];
  const others = await Promise.all(otherIds.map((id) => call("rectOf", id, 0.8)));
  const shrub = (await call("rectOf", 10, 1.4)) ?? undefined;

  const steps = [4000, 8000, 12000, 16000];
  const frames: number[] = [];
  const poses: { at: number; status: TelemetryStatus; error: number; arrived: number }[] = [];
  for (const at of steps) {
    await call("telemetryAt", at);
    const frame = await call("frame");
    frames.push(frame);
    await shot(`telemetry-${String(at / 1000).padStart(2, "0")}s`);
    const status = statusOf(await call("telemetryStatus"), 8);
    const expected = loopAt(status.playoutMs);
    const p = status.position ?? [NaN, NaN, NaN];
    const error = Math.hypot(
      (p[0] ?? 0) - expected[0],
      (p[1] ?? 0) - expected[1],
      (p[2] ?? 0) - expected[2],
    );
    // Something is now drawn where the path says the building's middle is (2 m up).
    const there = (await call("scanRect", [expected[0], expected[1], 2], 1.2)) ?? undefined;
    poses.push({ at, status, error, arrived: await call("difference", rest, frame, there) });
  }
  const rigid = await call("rigidInfo", [8, 90, 91, 92, 93, 94, 95, 96, 1, 10]);
  const shrubHandles = await call("skinHandles", 10);
  const shrubStatus = statusOf(await call("telemetryStatus"), 10);

  // The building's source falls silent (and the shrub's): held, then the building fades back.
  await call("mute", "yard-loop", true);
  await call("mute", "shrub-loop", true);
  await call("telemetryAt", 19_400);
  const fading = await call("frame");
  await shot("telemetry-fading");
  const fadingStatus = await call("telemetryStatus");
  await call("telemetryAt", 21_000);
  const rested = await call("frame");
  await shot("telemetry-rested");
  const restedStatus = await call("telemetryStatus");
  await call("telemetryAt", 22_000);
  const restedLater = await call("frame");
  // Readings again: live again.
  await call("mute", "yard-loop", false);
  await call("telemetryAt", 26_000);
  const resumed = await call("frame");
  await shot("telemetry-resumed");
  const resumedStatus = statusOf(await call("telemetryStatus"), 8);

  const measures = {
    building,
    startStatus,
    rigid,
    shrubStatus,
    poses: poses.map((p) => ({
      at: p.at,
      state: p.status.state,
      via: p.status.via,
      playoutMs: p.status.playoutMs,
      errorM: p.error,
      arrived: p.arrived,
    })),
    leaves: await Promise.all(frames.map((f) => call("difference", rest, f, building))),
    steps: await Promise.all(
      frames.slice(1).map((f, i) => call("difference", frames[i] ?? 0, f, building)),
    ),
    others: [] as Record<string, number>[],
    shrubMoves: await call("difference", rest, frames[1] ?? 0, shrub),
    fading: {
      states: fadingStatus.map((s) => s.state),
      building: await call("difference", rest, fading, building),
    },
    rested: {
      states: restedStatus.map((s) => s.state),
      building: await call("difference", rest, rested, building, 0),
      shrubFrozen: await call("difference", rested, restedLater, shrub, 0),
      shrubNotRest: await call("difference", rest, rested, shrub),
    },
    resumed: {
      state: resumedStatus.state,
      building: await call("difference", rest, resumed, building),
    },
  };
  for (const f of frames) {
    const row: Record<string, number> = {};
    for (const [k, r] of others.entries()) {
      const diff: number = await call("difference", rest, f, r ?? undefined);
      row[String(otherIds[k])] = diff;
    }
    measures.others.push(row);
  }
  test.info().annotations.push({ type: "measures", description: JSON.stringify(measures) });
  console.info(JSON.stringify(measures));

  expect(shaderErrors(errors)).toEqual([]);
  // Before the first reading arrives: rest.
  expect(startStatus.map((s) => s.state)).toEqual(["none", "none"]);
  // The building goes through the rigid part (all its leaves), the shrub through its skin.
  expect(rigid.driven).toEqual([8]);
  expect(rigid.active).toBe(true);
  expect(rigid.slots).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 0, 0]);
  expect(shrubStatus.via).toBe("skin");
  expect(shrubStatus.state).toBe("live");
  expect(shrubHandles).not.toBeNull();
  // Handle 0 carries the motion; the elastic handles stay at rest.
  expect((shrubHandles ?? []).slice(0, 12).some((v) => v !== 0)).toBe(true);
  expect((shrubHandles ?? []).slice(12).every((v) => v === 0)).toBe(true);
  for (const p of poses) {
    expect(p.status.state).toBe("live");
    expect(p.status.via).toBe("rigid");
    // The pose shown is the path at the playout time (ECEF in, scan frame out).
    expect(p.error).toBeLessThan(0.02);
    expect(p.arrived).toBeGreaterThan(0.05);
  }
  // The building leaves its place and keeps moving; the unbound objects do not.
  for (const leave of measures.leaves) expect(leave).toBeGreaterThan(0.05);
  for (const step of measures.steps) expect(step).toBeGreaterThan(0.03);
  for (const other of measures.others)
    for (const value of Object.values(other)) expect(value).toBeLessThan(0.002);
  expect(measures.shrubMoves).toBeGreaterThan(0.01);
  // Silent: held, then fading (the shrub frozen), then the measured frame exactly.
  expect(measures.fading.states).toEqual(["stale", "stale"]);
  expect(measures.fading.building).toBeGreaterThan(0.01);
  expect(measures.rested.states).toEqual(["rest", "stale"]);
  expect(measures.rested.building).toBe(0);
  expect(measures.rested.shrubFrozen).toBe(0);
  expect(measures.rested.shrubNotRest).toBeGreaterThan(0.005);
  expect(measures.resumed.state).toBe("live");
  expect(measures.resumed.building).toBeGreaterThan(0.05);
  // The building left from BUILDING_REST: the first frame's pose is a few degrees round.
  expect(Math.hypot(...BUILDING_REST.map((v, i) => v - (loopAt(0)[i] ?? 0)))).toBeLessThan(1e-9);
});

test("a bound skin is left by the wind, the tree still sways, the same clock gives the same frame", async ({
  page,
}) => {
  test.setTimeout(1_500_000);
  const errors: string[] = [];
  await open(page, errors);
  const call = caller(page);
  const shot = shooter(page);

  await call("view", 20, -40, 40);
  await call("view", 20, -40, 40);
  await call("telemetryReady");
  await call("telemetryAt", 0);
  const rest = await call("frame");
  const tree = (await call("rectOf", 1, 1.2)) ?? undefined;

  await call("windOn", 0.1, 60, 1000);
  const skins = await call("windSkins");
  await call("telemetryAt", 5000);
  await call("advance", 3);
  const a = await call("frame");
  await shot("telemetry-wind");
  const handles = await call("skinHandles", 10);

  // Replay: wind from the same start, telemetry from an empty track at the same clock.
  await call("telemetryReset", 0);
  await call("windOn", 0.1, 60, 1000);
  await call("telemetryAt", 5000);
  await call("advance", 3);
  const again = await call("frame");

  const measures = {
    skins,
    treeSways: await call("difference", rest, a, tree),
    replay: await call("difference", a, again, undefined, 0),
    elasticAtRest: (handles ?? []).slice(12).every((v) => v === 0),
  };
  test.info().annotations.push({ type: "measures", description: JSON.stringify(measures) });
  console.info(JSON.stringify(measures));

  expect(shaderErrors(errors)).toEqual([]);
  // The shrub's material sways it, but telemetry holds it: the wind leaves it alone.
  const shrub = skins.find((s) => s.instance === 10);
  expect(shrub?.wind).toBe(true);
  expect(shrub?.claimed).toBe(true);
  expect(skins.filter((s) => s.claimed).map((s) => s.instance)).toEqual([10]);
  expect(handles).not.toBeNull();
  expect(measures.elasticAtRest).toBe(true);
  expect(measures.treeSways).toBeGreaterThan(0.005);
  expect(measures.replay).toBe(0);
});
